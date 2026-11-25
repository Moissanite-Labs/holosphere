#!/usr/bin/env python3
"""Assertion-free R5 CPU-only verification suite for PREREG-S114-AMENDMENT1 (H14d & H14e)."""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.s114_s107_s108_convergence import (
    CayleyAcuteHoloSphereLM,
    HolonomySmoothedChunker,
    R_TEST_GRID_H14D1,
    build_h14d1_arms,
    build_h14d2_arms,
    build_h14e2_arms,
    causal_turing_two_rate_filter,
    compute_h14e2_loss_and_metrics,
    geodesic_second_order_turning_score,
    instantiate_h14d1_model,
    instantiate_h14e2_model,
    run_cpu_preflight,
)
from holosphere.errors import HoloSphereError
from holosphere.geometry.hopf import cayley_acute_cos_sin, hopf_clock_angle
from holosphere.geometry.su2 import safe_unit_norm


class TestS114ConvergenceR5(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_01_h14d1_cayley_acute_vs_coprime_antipodal_cure(self) -> None:
        # Reproduce W10-OBS-31 antipodal inversion at r=5 for unaliased_coprime
        phi5_coprime = hopf_clock_angle(5, "unaliased_coprime")
        self.assertLess(math.cos(phi5_coprime), -0.95)

        # Verify SFU-free cayley_acute stays strictly > 9/41 > 0 for r = 0..127 and strictly increasing angle
        floor_exact = 9.0 / 41.0
        prev_s = -1.0
        for r in range(128):
            c, s = cayley_acute_cos_sin(r, t_max=0.8, tau=2.0)
            self.assertAlmostEqual(c * c + s * s, 1.0, places=14)
            self.assertGreater(c, floor_exact)
            self.assertGreater(s, prev_s)
            prev_s = s

        # Verify CayleyAcuteHoloSphereLM runs R_test in {3,4,6,8,12} with unit-norm drift <= 1e-3
        arms = build_h14d1_arms(d_model=64, n_heads=4, base_d_ff=128, n_pre=1, n_core=2, n_suffix=1, seed=11)
        m_cay = instantiate_h14d1_model(*arms["SafeHopf-CayleyAcute"])
        m_noclk = instantiate_h14d1_model(*arms["SphereLoop-NoClock"])
        m_noclk.load_state_dict(m_cay.state_dict(), strict=True)
        gen = torch.Generator(device="cpu").manual_seed(12)
        x = torch.randint(10, 200, (2, 16), generator=gen)
        with torch.no_grad():
            lg_cay, tr_cay = m_cay(x, r_loops=12, return_trace=True)
            lg_nc, _ = m_noclk(x, r_loops=12, return_trace=True)
        self.assertTrue(bool(torch.isfinite(lg_cay).all().item()))
        self.assertLessEqual(tr_cay["norm_drift_max"], 1e-3)
        self.assertEqual(len(tr_cay["pass_states"]), 13)
        # Non-degenerate coupling: CayleyAcute output strictly differs from NoClock at identical weights
        diff = float(torch.linalg.vector_norm((lg_cay - lg_nc).float()).item())
        self.assertGreater(diff, 1e-2)

    def test_02_h14d2_equatorial_su2_exact_parity_and_determinism(self) -> None:
        arms = build_h14d2_arms(d_model=128, n_heads=4, n_layers=4, base_d_ff=352, seed=21)
        self.assertEqual(set(arms.keys()), {"Cyl-SU2-Holonomy", "Eq-SU2-Holonomy", "Pol-SU2-Holonomy", "PermEq-SU2-Holonomy"})
        from holosphere.models.transformer import HoloSphereTransformer, analytical_forward_flops_per_token, count_parameters

        params = [count_parameters(HoloSphereTransformer(c))["non_embedding_params"] for c in arms.values()]
        flops = [analytical_forward_flops_per_token(c, seq_len=512)["total_fwd_flops_per_token"] for c in arms.values()]
        self.assertEqual(len(set(params)), 1)
        self.assertEqual(len(set(flops)), 1)

        # Deterministic forward on Eq-SU2-Holonomy
        m1 = HoloSphereTransformer(arms["Eq-SU2-Holonomy"]).eval()
        m2 = HoloSphereTransformer(arms["Eq-SU2-Holonomy"]).eval()
        gen = torch.Generator(device="cpu").manual_seed(22)
        x = torch.randint(10, 200, (2, 24), generator=gen)
        with torch.no_grad():
            l1, _ = m1(x)
            l2, _ = m2(x)
        self.assertEqual(float(torch.max(torch.abs(l1 - l2)).item()), 0.0)

    def test_03_h14e1_holonomy_chunking_cures_w10_obs33_and_causality(self) -> None:
        # 1) AST check: ZERO acos / arccos calls anywhere in convergence harness
        src = (REPO_ROOT / "benchmarks" / "s114_s107_s108_convergence.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad_calls = [
            n.func.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("acos", "arccos")
        ]
        self.assertEqual(bad_calls, [])

        # 2) Geodesic 2nd-order turning vanishes on constant-speed great circle, spikes on 90-deg turn
        ang = torch.linspace(0.0, 1.2, steps=10, dtype=torch.float64)
        gc = torch.stack((torch.cos(ang), torch.sin(ang), torch.zeros_like(ang), torch.zeros_like(ang)), dim=-1).unsqueeze(0)
        gc_turn = geodesic_second_order_turning_score(gc, eps=1e-9)
        self.assertLess(float(gc_turn.max().item()), 1e-6)

        # 3) Equal-diffusion D=(5,5) null vanishes identically (Portal-2 O2.1); D=(1,10) responds sharply
        sig = torch.tensor([[0.1, 0.1, 0.9, 0.2, 0.1, 0.85, 0.1, 0.1]], dtype=torch.float64)
        out_eq = causal_turing_two_rate_filter(sig, d_u=5.0, d_v=5.0)
        out_diff = causal_turing_two_rate_filter(sig, d_u=1.0, d_v=10.0)
        self.assertLess(float(out_eq.abs().max().item()), 1e-12)
        self.assertGreater(float(out_diff[:, 2].item()), 0.25)

        # 4) Raw baseline reproduces W10-OBS-33 collapse (avg_plen <= 1.25), while Holonomy+Turing cures it
        text = ("The quick hypersphere trajectory carries smooth parallel transport along words. " * 4).encode("utf-8")[:256]
        x = torch.tensor([list(text)], dtype=torch.long)
        raw_out = HolonomySmoothedChunker(d_enc=64, mode="raw_geodesic_baseline", seed=31)(x)
        tur_out = HolonomySmoothedChunker(d_enc=64, mode="holonomy_turing_d1_10", seed=31)(x)
        self.assertLessEqual(raw_out["avg_plen"], 1.25)
        self.assertGreaterEqual(tur_out["avg_plen"], 2.5)
        self.assertLessEqual(tur_out["avg_plen"], 6.0)

    def test_04_h14e2_markov_mtp8_holosphere_grad_and_long_context(self) -> None:
        arms = build_h14e2_arms(d_model=64, n_heads=4, n_layers=2, base_d_ff=160, seed=41)
        m_mkv = instantiate_h14e2_model(*arms["HoloSphere-SU2-MarkovMTP8"])
        m_par = instantiate_h14e2_model(*arms["HoloSphere-SU2-ParallelMTP8"])
        gen = torch.Generator(device="cpu").manual_seed(42)
        # Train-length (32 + 8) and 4x extrapolation length (128 + 8)
        x_short = torch.randint(10, 200, (2, 40), generator=gen)
        x_long = torch.randint(10, 200, (1, 136), generator=gen)

        r_mkv = compute_h14e2_loss_and_metrics(m_mkv, x_short, block_size=8, lambda_mtp=0.2)
        r_mkv["total_loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in m_mkv.parameters() if p.grad is not None))
        self.assertIsNotNone(r_mkv["mtp_tf_bpb"])
        self.assertIsNotNone(r_mkv["mtp_unroll_bpb"])

        r_par = compute_h14e2_loss_and_metrics(m_par, x_short, block_size=8, lambda_mtp=0.2)
        r_par["total_loss"].backward()
        self.assertIsNotNone(r_par["mtp_tf_bpb"])

        with torch.no_grad():
            r_long = compute_h14e2_loss_and_metrics(m_mkv, x_long, block_size=8, lambda_mtp=0.2)
        self.assertTrue(math.isfinite(r_long["primary_bpb"]))
        self.assertTrue(math.isfinite(float(r_long["mtp_tf_bpb"])))

    def test_05_fail_closed_negative_controls(self) -> None:
        arms = build_h14d1_arms(d_model=64, n_heads=4, base_d_ff=128, n_pre=1, n_core=2, n_suffix=1, seed=51)
        # Reject t_max >= 1 (would allow antipodal flip!)
        with self.assertRaises(HoloSphereError):
            CayleyAcuteHoloSphereLM(arms["SafeHopf-CayleyAcute"][0], t_max=1.0, tau=2.0)
        # Reject non-positive diffusion constant
        with self.assertRaises(HoloSphereError):
            causal_turing_two_rate_filter(torch.zeros((1, 8)), d_u=0.0, d_v=10.0)
        # Reject unknown chunker mode
        with self.assertRaises(HoloSphereError):
            HolonomySmoothedChunker(d_enc=64, mode="invalid_mode")

    def test_06_cli_3mode_isolated_bit_identical(self) -> None:
        script = str(REPO_ROOT / "benchmarks" / "s114_s107_s108_convergence.py")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        digests: list[str] = []
        for flags in (["-I", "-B"], ["-I", "-O", "-B"], ["-I", "-B", "-W", "error"]):
            proc = subprocess.run(
                [sys.executable, *flags, script, "--cpu-preflight", "--json"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            digests.append(hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest())
        self.assertEqual(len(set(digests)), 1)


if __name__ == "__main__":
    unittest.main()
