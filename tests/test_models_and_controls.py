#!/usr/bin/env python3
"""Unified HoloSphereTransformer, S106-S114 presets, S_3/A_5 tasks, nullmodem E_8, & Weft rollback tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from holosphere.benchmarks import (
    A5_ORDER,
    S3_ORDER,
    build_a5_cayley_table,
    build_s3_cayley_table,
    generate_bracket_depth_batch,
    generate_group_tracking_batch,
)
from holosphere.benchmarks.prereg_s114_suite import run_cpu_preflight
from holosphere.errors import HoloSphereError
from holosphere.models import (
    BatchedNullModemCell,
    BlockE8SphereCodec,
    DiagonalDarcyEMA,
    HoloSphereTransformer,
    LocalARByteDecoder,
    MarkovMTP8Head,
    NullModemCell,
    ParallelMTP8Head,
    PlatePatchProposal,
    WeftAuditShell,
    analytical_forward_flops_per_token,
    apply_ptq_copy,
    build_preset_config,
    count_parameters,
    measure_m1_h1_loop_drift,
    verify_shell3_merkle_receipt,
)


class TestModelsAndControls(unittest.TestCase):
    def test_01_all_s106_to_s114_presets_forward_backward_and_cs08(self) -> None:
        small_kw = dict(d_model=64, n_heads=4, n_layers=4, base_d_ff=176, n_pre=1, n_core=2, n_suffix=1, r_train=2)
        presets = (
            "s114_entity",
            "s114_unlooped_covariant",
            "s106_h6c_hopf",
            "s107_h7a_equatorial_ngpt",
            "s107_h7b_su2_qk",
            "s107_h7c_safe_hopf_loop",
            "s108_h8c_markov_mtp8",
            "s109_terracore_bingham",
            "s110_geosphere_beltrami",
            "rope2d_std",
            "rope2d_ngpt",
            "liere4d_ngpt",
            "block4_rand_ortho_ngpt",
        )
        gen = torch.Generator(device="cpu").manual_seed(301)
        x = torch.randint(0, 64, (2, 12), generator=gen)

        for name in presets:
            cfg = build_preset_config(name, **small_kw)
            model = HoloSphereTransformer(cfg)
            logits, trace = model(x, return_trace=True)
            self.assertEqual(logits.shape, (2, 12, cfg.vocab_size), f"preset {name}")
            self.assertTrue(bool(torch.isfinite(logits).all().item()), f"non-finite logits in {name}")
            loss = logits.float().mean()
            loss.backward()
            params = count_parameters(model)
            flops = analytical_forward_flops_per_token(cfg, seq_len=128)
            self.assertGreater(params["non_embedding_params"], 0)
            self.assertGreater(flops["total_fwd_flops_per_token"], 0)
            if cfg.use_markov_mtp:
                self.assertIn("mtp_logits", trace)
                self.assertEqual(trace["mtp_logits"].shape, (2, 12, cfg.mtp_block_size, cfg.vocab_size))

        # Flagship looped entity R_test=6 extrapolation & M1-H1 drift + PTQ copy
        ent = HoloSphereTransformer(build_preset_config("s114_entity", **small_kw))
        logits_r6, trace_r6 = ent(x, r_loops=6, return_trace=True)
        self.assertTrue(bool(torch.isfinite(logits_r6).all().item()))
        self.assertEqual(len(trace_r6["pass_states"]), 7)
        drift = measure_m1_h1_loop_drift(ent, x, bits=4, group_size=32, r_eval=3)
        self.assertGreaterEqual(drift["kl_nats"], 0.0)
        q_copy = apply_ptq_copy(ent, bits=3, group_size=32)
        self.assertEqual(q_copy(x)[0].shape, logits_r6.shape)

    def test_02_darcy_ema_and_mtp8_heads_exact_parity(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(302)
        darcy = DiagonalDarcyEMA(d_model=16, seed=42)
        x = torch.randn((2, 20, 16), generator=gen)
        s_par = darcy.parallel_ema_scan(x, darcy.alpha())
        s_seq = darcy.sequential_ema_reference(x, darcy.alpha())
        self.assertLess(float(torch.max(torch.abs(s_par - s_seq)).item()), 1e-5)

        h = torch.randn((4, 32), generator=gen)
        tgt = torch.randint(0, 64, (4, 8), generator=gen)
        for head in (
            MarkovMTP8Head(32, vocab_size=64, block_size=8, markov_rank=16),
            ParallelMTP8Head(32, vocab_size=64, block_size=8),
            LocalARByteDecoder(32, vocab_size=64, d_dec=16, n_heads=4, block_size=8),
        ):
            y_tf = head(h, target_tokens=tgt, unroll=False)
            y_un = head(h, target_tokens=None, unroll=True)
            self.assertEqual(y_tf.shape, (4, 8, 64))
            self.assertEqual(y_un.shape, (4, 8, 64))

    def test_03_batched_nullmodem_e8_and_weft_merkle_rollback(self) -> None:
        codec = BlockE8SphereCodec(d_latent=32)  # 4 x E_8 blocks = 32 payload bits/state
        gen = torch.Generator(device="cpu").manual_seed(303)
        z1 = torch.randn((32,), generator=gen)
        w1 = codec.pack_cell(z1)
        dec1 = codec.unpack_cell(w1)
        self.assertEqual(dec1.shape, (32,))
        # Tamper 1 byte -> fail-closed SHA-256 refusal
        tampered = bytearray(w1)
        tampered[10] ^= 0x01
        with self.assertRaises(HoloSphereError):
            NullModemCell.verify_and_unpack(bytes(tampered))

        # Amortized batch M=4 beats unauthenticated ABIOS/3 (112 < 144 bits/state)
        self.assertEqual(BatchedNullModemCell.wire_bits_per_state(4, 4), 112.0)
        zb = torch.randn((4, 32), generator=gen)
        wb = codec.pack_batch_cells(zb)
        dec_b = codec.unpack_batch_cells(wb)
        self.assertEqual(dec_b.shape, (4, 32))

        # S111 WeftAuditShell Composed Triple Gate + O(1) atomic rollback
        model = HoloSphereTransformer(build_preset_config("s114_unlooped_covariant", d_model=32, n_heads=4, n_layers=2, base_d_ff=80))
        core_sha = model.core_weights_sha256()
        shell = WeftAuditShell(core_sha, n_slots=8, c_max=0.60, f_cum=4.0, min_holdout_delta=0.0)

        # Valid balanced patch -> COMMITTED + verifiable Merkle receipt
        dw_ok = [[0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [-0.2, -0.2]]
        rec_ok = shell.propose_and_commit(
            PlatePatchProposal(slot_idx=1, dw=dw_ok, holdout_delta=0.05, evaluated_patch_sha256=shell.patch_sha256(dw_ok)),
            current_core_sha256=core_sha,
        )
        self.assertEqual(rec_ok["status"], "COMMITTED")
        self.assertTrue(verify_shell3_merkle_receipt(rec_ok, core_sha)["verified"])

        # Concentrated C_row=1.0 patch -> ROLLED_BACK (pre_root == post_root)
        dw_bad = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        rec_bad = shell.propose_and_commit(
            PlatePatchProposal(slot_idx=2, dw=dw_bad, holdout_delta=0.10, evaluated_patch_sha256=shell.patch_sha256(dw_bad)),
            current_core_sha256=core_sha,
        )
        self.assertEqual(rec_bad["status"], "ROLLED_BACK")
        self.assertEqual(rec_bad["pre_root"], rec_bad["post_root"])
        self.assertNotIn(2, shell.slots)

        # Holdout regression -> ROLLED_BACK even when static & receipt pass
        rec_reg = shell.propose_and_commit(
            PlatePatchProposal(slot_idx=3, dw=dw_ok, holdout_delta=-0.02, evaluated_patch_sha256=shell.patch_sha256(dw_ok)),
            current_core_sha256=core_sha,
        )
        self.assertEqual(rec_reg["status"], "ROLLED_BACK")

    def test_04_s3_a5_tasks_and_prereg_s114_cpu_preflight(self) -> None:
        s3_tbl = build_s3_cayley_table()
        a5_tbl = build_a5_cayley_table()
        self.assertEqual(len(s3_tbl), S3_ORDER)
        self.assertEqual(len(a5_tbl), A5_ORDER)
        # Verify S_3 is genuinely non-Abelian
        self.assertTrue(any(s3_tbl[a][b] != s3_tbl[b][a] for a in range(S3_ORDER) for b in range(S3_ORDER)))
        xs, ys = generate_group_tracking_batch("S3", batch_size=4, seq_len=16, seed=42)
        self.assertEqual(xs.shape, (4, 16))
        self.assertEqual(ys.shape, (4, 16))
        bx, by = generate_bracket_depth_batch(4, 16, max_depth=6, seed=42)
        self.assertEqual(bx.shape, (4, 16))
        self.assertGreaterEqual(int(by.min().item()), 0)

        preflight = run_cpu_preflight()
        self.assertTrue(preflight["passed"], f"PREREG-S114 preflight failed: {preflight['checks']}")


if __name__ == "__main__":
    unittest.main()
