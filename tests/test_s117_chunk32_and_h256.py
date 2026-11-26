#!/usr/bin/env python3
"""R5 tests for Track S117 Chunk32 Woodbury ops, H_256 x bent4, 23-bit Gumbel, & Razor ABI (zero bare asserts)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holosphere import chunk32_ops as c32
from holosphere.errors import HoloSphereError


class TestS117Chunk32AndH256(unittest.TestCase):
    def test_01_h256_bent4_256x256_orthogonal_zero_const_warps_and_gumbel23(self) -> None:
        hres = c32.verify_h256_bent4_exhaustive()
        self.assertEqual(hres["pairs_checked"], 32896)
        self.assertEqual(hres["off_diag_errors"], 0)
        self.assertEqual(hres["const_warp_pairs"], 0)
        self.assertEqual(hres["max_abs_warp_sum"], 8)

        # Negative injectors killing every require_ok inside verify_h256_bent4_exhaustive
        with self.assertRaises(HoloSphereError) as ctx_diag:
            c32.verify_h256_bent4_exhaustive(inject_fn=lambda b, w, l: 0)
        self.assertEqual(ctx_diag.exception.code, "H256_NOT_ORTHOGONAL")
        with self.assertRaises(HoloSphereError) as ctx_off:
            c32.verify_h256_bent4_exhaustive(inject_fn=lambda b, w, l: 127)
        self.assertEqual(ctx_off.exception.code, "H256_NOT_ORTHOGONAL")
        # Raw H_256 (without bent4) is 256x256 orthogonal but has 64 constant warps!
        with self.assertRaises(HoloSphereError) as ctx_cw:
            c32.verify_h256_bent4_exhaustive(
                inject_fn=lambda b, w, l: -127 if ((b & ((((w & 7) << 5) | (l & 31)) ^ ((l & 7) << 5))).bit_count() & 1) else 127
            )
        self.assertEqual(ctx_cw.exception.code, "H256_CONST_WARPS")
        # Orthogonal column-flipped H_256 with 8 flipped lanes in each warp -> 0 const warps, max_abs_warp_sum = 16 != 8!
        with self.assertRaises(HoloSphereError) as ctx_pb:
            c32.verify_h256_bent4_exhaustive(
                inject_fn=lambda b, w, l: (
                    (-1 if l < 8 else 1)
                    * (-127 if ((b & ((((w & 7) << 5) | (l & 31)) ^ ((l & 7) << 5))).bit_count() & 1) else 127)
                )
            )
        self.assertEqual(ctx_pb.exception.code, "H256_PARSEVAL_BOUND")

        gres = c32.verify_gumbel_23bit_finite()
        self.assertEqual(gres["naive_32bit_0xFFFFFFFF_u"], 1.0)
        self.assertGreater(gres["u23_min"], 0.0)
        self.assertLess(gres["u23_max"], 1.0)
        self.assertAlmostEqual(gres["gumbel_min"], -2.811541, places=5)
        self.assertAlmostEqual(gres["gumbel_max"], 16.635532, places=4)

        # Negative u_fn / g_fn controls killing every require_ok inside verify_gumbel_23bit_finite
        with self.assertRaises(HoloSphereError) as ctx_u:
            c32.verify_gumbel_23bit_finite(u_fn=lambda u32: 0.5)
        self.assertEqual(ctx_u.exception.code, "GUMBEL_U_BOUNDS")
        with self.assertRaises(HoloSphereError) as ctx_u2:
            c32.verify_gumbel_23bit_finite(u_fn=lambda u32: c32.gumbel_23bit_u_f32(u32) if u32 == 0 else 0.5)
        self.assertEqual(ctx_u2.exception.code, "GUMBEL_U_BOUNDS")
        for bad_g in (
            lambda u32, t: float("nan"),
            lambda u32, t: c32.gumbel_23bit_finite_f32(u32, t) if u32 == 0 else float("inf"),
            lambda u32, t: -2.811541 + 2.0**-17 if (u32 == 0 and t == 1.0) else c32.gumbel_23bit_finite_f32(u32, t),
            lambda u32, t: 16.635532 + 2.0**-14 if (u32 == 0xFFFFFFFF and t == 1.0) else c32.gumbel_23bit_finite_f32(u32, t),
        ):
            with self.assertRaises(HoloSphereError) as ctx_gr:
                c32.verify_gumbel_23bit_finite(g_fn=bad_g)
            self.assertEqual(ctx_gr.exception.code, "GUMBEL_RANGE")
        for bad_u, bad_g in (
            (lambda u32: 0.0 if u32 == 1 else c32.gumbel_23bit_u_f32(u32), c32.gumbel_23bit_finite_f32),
            (lambda u32: 1.0 if u32 == 1 else c32.gumbel_23bit_u_f32(u32), c32.gumbel_23bit_finite_f32),
            (c32.gumbel_23bit_u_f32, lambda u32, t: float("nan") if u32 == 1 else c32.gumbel_23bit_finite_f32(u32, t)),
            (c32.gumbel_23bit_u_f32, lambda u32, t: -99.0 if u32 == 1 else c32.gumbel_23bit_finite_f32(u32, t)),
            (c32.gumbel_23bit_u_f32, lambda u32, t: 99.0 if u32 == 1 else c32.gumbel_23bit_finite_f32(u32, t)),
        ):
            with self.assertRaises(HoloSphereError) as ctx_gp:
                c32.verify_gumbel_23bit_finite(u_fn=bad_u, g_fn=bad_g)
            self.assertEqual(ctx_gp.exception.code, "GUMBEL_PROBE_FAIL")
        with self.assertRaises(HoloSphereError) as ctx_t0:
            c32.verify_gumbel_23bit_finite(g_fn=lambda u32, t: 1.0 if t == 0.0 else c32.gumbel_23bit_finite_f32(u32, t))
        self.assertEqual(ctx_t0.exception.code, "GUMBEL_TEMP0")

        for t in range(32):
            self.assertAlmostEqual(c32.pow_gamma_u5_py(0.9, t), 0.9**t, places=12)
        self.assertEqual(c32.pow_gamma_u5_py(0.0, 0), 1.0)
        for bad_t in (-1, 32, True):
            with self.assertRaises(HoloSphereError) as ctx_pt:
                c32.pow_gamma_u5_py(0.9, bad_t)  # type: ignore[arg-type]
            self.assertEqual(ctx_pt.exception.code, "BAD_POW_T")

    def test_02_t8192_vram_budget_ptxas_receipts_and_razor_abi_header(self) -> None:
        bud = c32.estimate_t8192_vram_budget()
        self.assertTrue(bud["fits_8gib_ga104m"])
        self.assertGreater(bud["vram_compression_ratio"], 20.0)
        self.assertEqual(bud["w10_obs_60_smem"]["choice_a_smem_per_warp_bytes"], 4224)
        self.assertEqual(bud["w10_obs_60_smem"]["choice_b_smem_per_warp_bytes"], 6336)
        self.assertEqual(bud["w10_obs_60_smem"]["choice_a_smem_reduction_x"], 6.0)
        self.assertLessEqual(bud["w10_obs_60_smem"]["choice_b_16warps_smem_bytes"], 102400)

        # Exact lower valid boundary (batch=1, seq_len=32, d_model=32, n_layers=1, chunk_size=32)
        bud_min = c32.estimate_t8192_vram_budget(batch=1, seq_len=32, d_model=32, n_layers=1, chunk_size=32)
        self.assertEqual(bud_min["shape"]["B"], 1)
        for bad_shape_kw in (
            {"batch": 0},
            {"batch": True},
            {"batch": 10**400},
            {"seq_len": 0},
            {"seq_len": True},
            {"d_model": 0},
            {"d_model": 30},
            {"d_model": True},
            {"n_layers": 0},
            {"n_layers": True},
        ):
            with self.assertRaises(HoloSphereError) as ctx_s:
                c32.estimate_t8192_vram_budget(**bad_shape_kw)  # type: ignore[arg-type]
            self.assertEqual(ctx_s.exception.code, "BAD_SHAPE")
        for bad_chunk_kw in (
            {"chunk_size": 0},
            {"chunk_size": -4},
            {"chunk_size": True},
            {"seq_len": 100, "chunk_size": 32},
        ):
            with self.assertRaises(HoloSphereError) as ctx_c:
                c32.estimate_t8192_vram_budget(**bad_chunk_kw)  # type: ignore[arg-type]
            self.assertEqual(ctx_c.exception.code, "BAD_CHUNK")

        rec = c32.verify_s117_ptxas_receipt()
        self.assertEqual(rec["total_kernels"], 11)
        self.assertEqual(rec["zero_spill_kernels"], 11)
        self.assertEqual(rec["sass_LDL"], 0)
        self.assertEqual(rec["sass_STL"], 0)
        self.assertEqual(rec["sass_MEMBAR_GL"], 0)
        self.assertEqual(rec["sass_RED_SYS"], 2)
        self.assertEqual(rec["sass_IDP4A"], 16)

        v3rec = c32.verify_v3_frontier_ptxas_receipt()
        self.assertEqual(v3rec["total_kernels"], 2)
        self.assertEqual(v3rec["zero_spill_kernels"], 2)
        self.assertEqual(v3rec["sass_HMMA_16816"], 28)
        self.assertEqual(v3rec["sass_IDP4A"], 24)
        self.assertEqual(v3rec["sass_LDL"], 0)
        self.assertEqual(v3rec["sass_STL"], 0)

        # Complete B2 fail-closed tamper controls on BOTH S117 & V3 ptxas receipts
        td = Path(tempfile.mkdtemp(prefix="holo_s117_r5_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        # Missing .cu / missing .json
        with self.assertRaises(HoloSphereError) as ctx_nocu:
            c32.verify_s117_ptxas_receipt(td)
        self.assertEqual(ctx_nocu.exception.code, "PTXAS_NO_CU")
        shutil.copy2(c32.DEFAULT_CSRC_DIR / c32.CU_FILENAME, td / c32.CU_FILENAME)
        with self.assertRaises(HoloSphereError) as ctx_norec:
            c32.verify_s117_ptxas_receipt(td)
        self.assertEqual(ctx_norec.exception.code, "PTXAS_NO_RECEIPT")
        # Malformed JSON & non-dict JSON root
        for bad_raw in ("{bad json", "[]"):
            (td / c32.PTXAS_FILENAME).write_text(bad_raw, encoding="utf-8")
            with self.assertRaises(HoloSphereError) as ctx_json:
                c32.verify_s117_ptxas_receipt(td)
            self.assertEqual(ctx_json.exception.code, "PTXAS_BAD_JSON")

        rec_src = json.loads((c32.DEFAULT_CSRC_DIR / c32.PTXAS_FILENAME).read_text(encoding="utf-8"))
        for code, mut in (
            ("PTXAS_CU_SHA", lambda r: r.__setitem__("cu_sha256", "0" * 64)),
            ("PTXAS_NONZERO", lambda r: r.__setitem__("nonzero_spill_kernels", 1)),
            ("PTXAS_COUNT", lambda r: r["kernels"].pop()),
            ("PTXAS_COUNT", lambda r: r.__setitem__("total_kernels", 10)),
            ("PTXAS_ZERO_SPILL_KEY", lambda r: r.pop("zero_spill_kernels")),
            ("PTXAS_SPILL", lambda r: r["kernels"][0].__setitem__("stack_frame_bytes", 16)),
            ("PTXAS_SPILL", lambda r: r["kernels"][0].__setitem__("spill_store_bytes", 16)),
            ("PTXAS_SPILL", lambda r: r["kernels"][0].__setitem__("spill_load_bytes", 16)),
            ("PTXAS_SPILL", lambda r: r["kernels"][0].__setitem__("zero_spill", False)),
            ("PTXAS_SPILL", lambda r: r["kernels"].__setitem__(0, "bad_entry")),
            ("PTXAS_SYMBOLS", lambda r: r["kernels"][0].__setitem__("symbol", "")),
            ("PTXAS_SYMBOLS", lambda r: r["kernels"][0].__setitem__("symbol", ["unhashable"])),
            ("PTXAS_SYMBOLS", lambda r: r.__setitem__("kernels", [copy.deepcopy(r["kernels"][0])] * 11)),
            ("PTXAS_SYMBOLS", lambda r: r["kernels"][0].__setitem__("symbol", "bogus_symbol")),
            ("SASS_BAD_SCHEMA", lambda r: r.__setitem__("sass_instruction_counts", [])),
            ("SASS_LOCAL_MEM", lambda r: r["sass_instruction_counts"].__setitem__("LDL", 1)),
            ("SASS_LOCAL_MEM", lambda r: r["sass_instruction_counts"].__setitem__("STL", 1)),
            ("SASS_CROSS_CTA_BAR", lambda r: r["sass_instruction_counts"].__setitem__("MEMBAR_GL", 1)),
            ("SASS_CROSS_CTA_BAR", lambda r: r["sass_instruction_counts"].__setitem__("ATOMG_GPU", 1)),
            ("SASS_MISSING_OPS", lambda r: r["sass_instruction_counts"].__setitem__("RED_SYS", 1)),
            ("SASS_MISSING_OPS", lambda r: r["sass_instruction_counts"].__setitem__("IDP4A", 15)),
            ("SASS_MISSING_OPS", lambda r: r["sass_instruction_counts"].__setitem__("RED_SYS", True)),
        ):
            bad = copy.deepcopy(rec_src)
            mut(bad)
            (td / c32.PTXAS_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(HoloSphereError) as ctx_t:
                c32.verify_s117_ptxas_receipt(td)
            self.assertEqual(ctx_t.exception.code, code)

        # V3 Frontier tamper controls incl. B2 duplicate-row substitution & SASS op boundaries
        shutil.copy2(c32.DEFAULT_CSRC_DIR / c32.V3_CU_FILENAME, td / c32.V3_CU_FILENAME)
        v3_src = json.loads((c32.DEFAULT_CSRC_DIR / c32.V3_PTXAS_FILENAME).read_text(encoding="utf-8"))
        for code, mut in (
            ("PTXAS_CU_SHA", lambda r: r.__setitem__("cu_sha256", "0" * 64)),
            ("PTXAS_NONZERO", lambda r: r.__setitem__("nonzero_spill_kernels", 1)),
            ("PTXAS_ZERO_SPILL_KEY", lambda r: r.pop("zero_spill_kernels")),
            ("PTXAS_SYMBOLS", lambda r: r.__setitem__("kernels", [copy.deepcopy(r["kernels"][0])] * 2)),
            ("SASS_V3_OPS", lambda r: r["sass_instruction_counts"].__setitem__("HMMA_16816", 27)),
            ("SASS_V3_OPS", lambda r: r["sass_instruction_counts"].__setitem__("IDP4A", 23)),
            ("SASS_V3_OPS", lambda r: r["sass_instruction_counts"].__setitem__("HMMA_16816", True)),
        ):
            bad_v3 = copy.deepcopy(v3_src)
            mut(bad_v3)
            (td / c32.V3_PTXAS_FILENAME).write_text(json.dumps(bad_v3), encoding="utf-8")
            with self.assertRaises(HoloSphereError) as ctx_v3:
                c32.verify_v3_frontier_ptxas_receipt(td)
            self.assertEqual(ctx_v3.exception.code, code)

        # Razor 64 B .sys ABI header presence & pure-stdlib torch-blocked import check
        abi_hdr = REPO_ROOT / "include" / "razor_abi_4byte_sys.h"
        self.assertTrue(abi_hdr.is_file())
        hdr_txt = abi_hdr.read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: Apache-2.0", hdr_txt)
        self.assertIn("__ORDER_LITTLE_ENDIAN__", hdr_txt)
        self.assertIn("__atomic_load_n(ptr, __ATOMIC_ACQUIRE)", hdr_txt)
        self.assertIn("#include <intrin.h>", hdr_txt)
        self.assertIn("_ReadWriteBarrier();", hdr_txt)
        self.assertNotIn("_InterlockedCompareExchange", hdr_txt)
        self.assertNotIn("std::atomic<uint32_t>*", hdr_txt)
        self.assertNotIn("_Atomic uint32_t*", hdr_txt)
        self.assertIn("HoloSphereRazorCanary64", hdr_txt)
        self.assertIn("HoloSpherePiRollback16", hdr_txt)

        probe = (
            f"import sys; sys.modules['torch'] = None; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from holosphere import chunk32_ops as c; "
            "print(c.estimate_t8192_vram_budget()['fits_8gib_ga104m'])"
        )
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "True")

        # _require_torch negative test & load_s117_cuda_lib nonexistent-path fail-closed return None
        self.assertIsNone(c32.load_s117_cuda_lib(td / "nonexistent_s117.so"))
        old_has = c32._HAS_TORCH
        try:
            c32._HAS_TORCH = False
            with self.assertRaises(HoloSphereError) as ctx_nt:
                c32._require_torch()
            self.assertEqual(ctx_nt.exception.code, "NO_TORCH")
        finally:
            c32._HAS_TORCH = old_has

    @unittest.skipUnless(c32._HAS_TORCH, "requires torch")
    def test_03_torch_fp64_choice_a_b_gradcheck_fullgraph_and_b3_guards(self) -> None:
        import torch

        res = c32.run_cpu_gradcheck_self_test()
        self.assertTrue(res["gradcheck_choice_a_pass"])
        self.assertTrue(res["gradcheck_choice_b_pass"])
        self.assertTrue(res["torch_compile_fullgraph_aot_eager_pass"])
        self.assertLess(res["choice_a_4224B"]["worst_err_vs_autograd"], 1e-13)
        self.assertLess(res["choice_b_6336B"]["worst_err_vs_autograd"], 1e-13)
        self.assertLess(res["torch_compile_fullgraph_worst_err"], 1e-13)
        self.assertLess(res["h256_embed_ortho_max_err"], 1e-5)

        # validate_gradcheck_self_test_report exact-boundary negative controls
        for code, mut in (
            ("W10_OBS_60_DRIFT", lambda r: r["choice_a_4224B"].__setitem__("worst_err_vs_autograd", 1e-13)),
            ("W10_OBS_60_DRIFT", lambda r: r["choice_b_6336B"].__setitem__("worst_err_vs_autograd", 1e-13)),
            ("TRACK_A_GRADCHECK_FAIL", lambda r: r.__setitem__("gradcheck_choice_a_pass", False)),
            ("TRACK_A_GRADCHECK_FAIL", lambda r: r.__setitem__("gradcheck_choice_b_pass", False)),
            ("COMPILE_FULLGRAPH_DRIFT", lambda r: r.__setitem__("torch_compile_fullgraph_worst_err", 1e-13)),
            ("EMBED_ORTHO", lambda r: r.__setitem__("h256_embed_ortho_max_err", 1e-5)),
        ):
            bad_rep = copy.deepcopy(res)
            mut(bad_rep)
            with self.assertRaises(HoloSphereError) as ctx_r:
                c32.validate_gradcheck_self_test_report(bad_rep)
            self.assertEqual(ctx_r.exception.code, code)

        # h256_bent4_embed_torch d_model=512 valid + d_model=128 invalid
        e512 = c32.h256_bent4_embed_torch(torch.tensor([0, 1], dtype=torch.uint8), d_model=512)
        self.assertEqual(tuple(e512.shape), (2, 512))
        with self.assertRaises(HoloSphereError) as ctx_dm:
            c32.h256_bent4_embed_torch(torch.tensor([0], dtype=torch.uint8), d_model=128)
        self.assertEqual(ctx_dm.exception.code, "BAD_DMODEL")

        # B3 guards on chunk32_holosphere_layer (zero/neg/bool chunk_size, bad shape, non-finite params, eps<=0)
        x_ok = torch.ones((1, 4, 32), dtype=torch.float64)
        w_ok = torch.zeros((1, 4, 1, 8, 3), dtype=torch.float64)
        W_ok = torch.zeros((1, 32, 32), dtype=torch.float64)
        S_ok = torch.zeros((1, 1, 32, 32), dtype=torch.float64)
        U_ok = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]] * 8]], dtype=torch.float64)
        # Positive minimal valid boundary (chunk_size=1, T=4, D=32)
        y_ok, _, _ = c32.chunk32_holosphere_layer(x_ok, w_ok, W_ok, S_ok, U_ok, chunk_size=1)
        self.assertTrue(bool(torch.isfinite(y_ok).all().item()))

        for bad_c in (0, -2, True):
            with self.assertRaises(HoloSphereError) as ctx_bc:
                c32.chunk32_holosphere_layer(x_ok, w_ok, W_ok, S_ok, U_ok, chunk_size=bad_c)  # type: ignore[arg-type]
            self.assertEqual(ctx_bc.exception.code, "BAD_CHUNK")
        for bad_x, cs in (
            (torch.ones((4, 32), dtype=torch.float64), 2),
            (torch.ones((1, 0, 32), dtype=torch.float64), 2),
            (torch.ones((1, 4, 0), dtype=torch.float64), 2),
            (torch.ones((1, 4, 30), dtype=torch.float64), 2),
            (torch.ones((1, 5, 32), dtype=torch.float64), 2),
        ):
            with self.assertRaises(HoloSphereError) as ctx_bs:
                c32.chunk32_holosphere_layer(bad_x, w_ok, W_ok, S_ok, U_ok, chunk_size=cs)
            self.assertEqual(ctx_bs.exception.code, "BAD_SEQ_SHAPE")
        for bad_pkw in (
            {"cayley_t": float("nan")},
            {"gamma": float("inf")},
            {"eta": float("-inf")},
            {"eps": float("nan")},
            {"eps": 0.0},
            {"eps": -0.5},
            {"cayley_t": True},
        ):
            with self.assertRaises(HoloSphereError) as ctx_bp:
                c32.chunk32_holosphere_layer(x_ok, w_ok, W_ok, S_ok, U_ok, chunk_size=2, **bad_pkw)  # type: ignore[arg-type]
            self.assertEqual(ctx_bp.exception.code, "BAD_PARAM")

    def test_04_ptxas_check_40_kernel_parser_receipts_and_cli(self) -> None:
        import io
        import os
        from contextlib import redirect_stderr, redirect_stdout
        from holosphere.kernels import ptxas_check as pc

        # 1. parse_ptxas_verbose positive + negative controls
        ok_log = (
            "ptxas info    : Compiling entry function 'k1' for 'sm_86'\n"
            "ptxas info    : Function properties for k1\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 48 registers, 400 bytes cmem[0]\n"
            "ptxas info    : Compiling entry function 'k2' for 'sm_86'\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 96 registers, 400 bytes cmem[0]\n"
        )
        parsed = pc.parse_ptxas_verbose(ok_log, expected_kernels=2)
        self.assertEqual(parsed["total_kernels"], 2)
        self.assertEqual(parsed["zero_spill_kernels"], 2)
        self.assertEqual(parsed["max_registers"], 96)

        with self.assertRaises(HoloSphereError) as cm_t:
            pc.parse_ptxas_verbose(123)  # type: ignore[arg-type]
        self.assertEqual(cm_t.exception.code, "PTXAS_BAD_LOG")
        with self.assertRaises(HoloSphereError) as cm_nk:
            pc.parse_ptxas_verbose("no kernels here\n")
        self.assertEqual(cm_nk.exception.code, "PTXAS_NO_KERNELS")
        with self.assertRaises(HoloSphereError) as cm_cnt:
            pc.parse_ptxas_verbose(ok_log, expected_kernels=3)
        self.assertEqual(cm_cnt.exception.code, "PTXAS_COUNT_MISMATCH")
        with self.assertRaises(HoloSphereError) as cm_inc:
            pc.parse_ptxas_verbose("ptxas info : Compiling entry function 'k1' for 'sm_86'\n")
        self.assertEqual(cm_inc.exception.code, "PTXAS_INCOMPLETE_STATS")
        with self.assertRaises(HoloSphereError) as cm_inc2:
            pc.parse_ptxas_verbose(
                "ptxas info : Compiling entry function 'k1' for 'sm_86'\n"
                "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            )
        self.assertEqual(cm_inc2.exception.code, "PTXAS_INCOMPLETE_STATS")
        for bad_sp in (
            "    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n",
            "    0 bytes stack frame, 16 bytes spill stores, 0 bytes spill loads\n",
            "    0 bytes stack frame, 0 bytes spill stores, 16 bytes spill loads\n",
        ):
            bad_log = (
                "ptxas info : Compiling entry function 'k1' for 'sm_86'\n"
                + bad_sp
                + "ptxas info : Used 48 registers\n"
            )
            with self.assertRaises(HoloSphereError) as cm_sp:
                pc.parse_ptxas_verbose(bad_log, expected_kernels=1)
            self.assertEqual(cm_sp.exception.code, "PTXAS_SPILL_DETECTED")
            # A1: non-zero spill BEFORE first entry header must fail closed immediately
            with self.assertRaises(HoloSphereError) as cm_pre:
                pc.parse_ptxas_verbose(bad_sp + ok_log)
            self.assertEqual(cm_pre.exception.code, "PTXAS_SPILL_DETECTED")
            # A1: non-zero spill overwritten by later 0/0/0 line on same kernel must fail closed
            overwritten = (
                "ptxas info : Compiling entry function 'k1' for 'sm_86'\n"
                + bad_sp
                + "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                + "ptxas info : Used 48 registers\n"
            )
            with self.assertRaises(HoloSphereError) as cm_ov:
                pc.parse_ptxas_verbose(overwritten, expected_kernels=1)
            self.assertEqual(cm_ov.exception.code, "PTXAS_SPILL_DETECTED")

        # A1: orphan 0/0/0 spill or register line before first entry header -> PTXAS_ORPHAN_STATS
        with self.assertRaises(HoloSphereError) as cm_osp:
            pc.parse_ptxas_verbose("    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" + ok_log)
        self.assertEqual(cm_osp.exception.code, "PTXAS_ORPHAN_STATS")
        with self.assertRaises(HoloSphereError) as cm_org:
            pc.parse_ptxas_verbose("ptxas info : Used 32 registers\n" + ok_log)
        self.assertEqual(cm_org.exception.code, "PTXAS_ORPHAN_STATS")

        # A1: duplicate spill or register stats on same kernel -> PTXAS_DUPLICATE_STATS
        dup_sp = (
            "ptxas info : Compiling entry function 'k1' for 'sm_86'\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info : Used 48 registers\n"
        )
        with self.assertRaises(HoloSphereError) as cm_dsp:
            pc.parse_ptxas_verbose(dup_sp, expected_kernels=1)
        self.assertEqual(cm_dsp.exception.code, "PTXAS_DUPLICATE_STATS")
        dup_rg = (
            "ptxas info : Compiling entry function 'k1' for 'sm_86'\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info : Used 48 registers\n"
            "ptxas info : Used 48 registers\n"
        )
        with self.assertRaises(HoloSphereError) as cm_drg:
            pc.parse_ptxas_verbose(dup_rg, expected_kernels=1)
        self.assertEqual(cm_drg.exception.code, "PTXAS_DUPLICATE_STATS")

        # 2. verify_all_40_receipts positive & missing-core control
        rec40 = pc.verify_all_40_receipts()
        self.assertEqual(rec40["total_kernels_catalogued"], 40)
        self.assertEqual(rec40["receipt_verified_zero_spill_kernels"], 13)
        self.assertEqual(rec40["v3_frontier_receipt"]["zero_spill_kernels"], 2)

        td = Path(tempfile.mkdtemp(prefix="holo_ptxas40_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        with self.assertRaises(HoloSphereError) as cm_miss:
            pc.verify_all_40_receipts(td)
        self.assertEqual(cm_miss.exception.code, "PTXAS_NO_CU")

        # 3. compile_all_40_kernels negative & mocked-nvcc positive controls (CPU-safe, 0 GPU)
        with self.assertRaises(HoloSphereError) as cm_ba:
            pc.compile_all_40_kernels(nvcc="  ")
        self.assertEqual(cm_ba.exception.code, "NVCC_BAD_ARG")
        with self.assertRaises(HoloSphereError) as cm_ar:
            pc.compile_all_40_kernels(arch="compute_86")
        self.assertEqual(cm_ar.exception.code, "NVCC_BAD_ARCH")
        with self.assertRaises(HoloSphereError) as cm_un:
            pc.compile_all_40_kernels(nvcc="/nonexistent/nvcc_holo_test")
        self.assertEqual(cm_un.exception.code, "NVCC_UNAVAILABLE")

        def _fake_nvcc(name: str, body: str) -> str:
            p = td / name
            p.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
            os.chmod(p, 0o700)
            return str(p)

        nv_ver_fail = _fake_nvcc("nv_vf.sh", "exit 1")
        with self.assertRaises(HoloSphereError) as cm_vf:
            pc.compile_all_40_kernels(nvcc=nv_ver_fail)
        self.assertEqual(cm_vf.exception.code, "NVCC_VERSION_FAIL")

        nv_comp_fail = _fake_nvcc(
            "nv_cf.sh",
            'if [ "$1" = "--version" ]; then echo "Build mock"; exit 0; fi\necho "err" >&2\nexit 1',
        )
        with self.assertRaises(HoloSphereError) as cm_cf:
            pc.compile_all_40_kernels(nvcc=nv_comp_fail)
        self.assertEqual(cm_cf.exception.code, "NVCC_COMPILE_FAIL")

        # Missing .cu inside custom csrc_dir
        nv_ok_stub = _fake_nvcc(
            "nv_ok.sh",
            'if [ "$1" = "--version" ]; then echo "Build cuda_mock"; exit 0; fi\n'
            'case "$*" in\n'
            '  *su2_holonomy_scan_sm86.cu*) N=27 ;;\n'
            '  *holosphere_h256_chunk32_sm86.cu*) N=11 ;;\n'
            '  *holosphere_v3_frontier_sm86.cu*) N=2 ;;\n'
            '  *) N=0 ;;\n'
            'esac\n'
            'i=1\n'
            'while [ "$i" -le "$N" ]; do\n'
            '  echo "ptxas info : Compiling entry function \'k$i\' for \'sm_86\'" >&2\n'
            '  echo "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" >&2\n'
            '  echo "ptxas info : Used 48 registers" >&2\n'
            '  i=$((i + 1))\n'
            'done\n'
            'exit 0',
        )
        with self.assertRaises(HoloSphereError) as cm_mc:
            pc.compile_all_40_kernels(csrc_dir=td, nvcc=nv_ok_stub)
        self.assertEqual(cm_mc.exception.code, "PTXAS_NO_CU")

        # Mocked subprocess error during compile
        orig_run = pc.subprocess.run
        calls = {"n": 0}
        try:
            def _boom_on_second(cmd, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    return orig_run(cmd, **kw)
                raise OSError("compile_boom")
            pc.subprocess.run = _boom_on_second
            with self.assertRaises(HoloSphereError) as cm_ce:
                pc.compile_all_40_kernels(nvcc=nv_ok_stub)
            self.assertEqual(cm_ce.exception.code, "NVCC_COMPILE_ERROR")
        finally:
            pc.subprocess.run = orig_run

        # Mocked full 40/40 compile pass
        comp40 = pc.compile_all_40_kernels(nvcc=nv_ok_stub)
        self.assertEqual(comp40["total_kernels"], 40)
        self.assertEqual(comp40["zero_spill_kernels"], 40)

        # 4. CLI flags positive & fail-closed
        out_b, err_b = io.StringIO(), io.StringIO()
        with redirect_stdout(out_b), redirect_stderr(err_b):
            rc = pc.main(["--verify-receipts", "--compile", "--nvcc", nv_ok_stub, "--arch", "sm_86", "--json"])
        self.assertEqual(rc, 0, err_b.getvalue())
        cli_obj = json.loads(out_b.getvalue())
        self.assertEqual(cli_obj["compile"]["zero_spill_kernels"], 40)
        self.assertEqual(cli_obj["receipts"]["total_kernels_catalogued"], 40)

        out_def, err_def = io.StringIO(), io.StringIO()
        with redirect_stdout(out_def), redirect_stderr(err_def):
            self.assertEqual(pc.main([]), 0)
        self.assertIn("OK ptxas_check", out_def.getvalue())

        for bad_argv in (["--nvcc"], ["--arch"], ["--unknown-flag"]):
            err_bad = io.StringIO()
            with redirect_stderr(err_bad):
                self.assertEqual(pc.main(bad_argv), 2)
            self.assertIn("CLI_BAD_ARG", err_bad.getvalue())


if __name__ == "__main__":
    unittest.main()
