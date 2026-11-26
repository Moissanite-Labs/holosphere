#!/usr/bin/env python3
"""R5 tests for HoloSphere-V3 E2E Fused Training + Inference Engine (CPU-only, zero bare asserts)."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holosphere import e2e_engine as e2e
from holosphere import e2e_inference as e2e_inf
from holosphere.errors import HoloSphereError


class TestHoloSphereV3E2EEngineR5(unittest.TestCase):
    def test_01_ptxas_16_of_16_zero_spill_and_tamper_controls(self) -> None:
        res = e2e.verify_e2e_ptxas_receipt()
        self.assertEqual(res["total_kernels"], 16)
        self.assertEqual(res["zero_spill_kernels"], 16)
        self.assertEqual(res["units"]["training_ext"]["kernel_count"], 4)
        self.assertEqual(res["units"]["inference_runtime"]["kernel_count"], 12)
        self.assertEqual(res["units"]["training_ext"]["HMMA_16816"], 28)
        self.assertEqual(res["units"]["inference_runtime"]["RED_SYS"], 6)
        self.assertEqual(res["units"]["inference_runtime"]["MEMBAR_SYS"], 9)
        self.assertEqual(res["units"]["training_ext"]["LDL"], 0)
        self.assertEqual(res["units"]["inference_runtime"]["STL"], 0)

        # Verify canonical razor_abi_4byte_sys.h reuse & F1 launcher bounds (D<=512, H<=16)
        inf_cu = (e2e.DEFAULT_CSRC_DIR / e2e.INFER_CU_FILENAME).read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: Apache-2.0", inf_cu)
        self.assertIn('#include "../../../include/razor_abi_4byte_sys.h"', inf_cu)
        self.assertNotIn("struct alignas(64) HoloSphereRazorCanary64", inf_cu)
        self.assertNotIn("struct alignas(16) HoloSpherePiRollback16", inf_cu)
        self.assertIn("holosphere_razor_poll_acquire", inf_cu)
        self.assertIn("d_model > 512", inf_cu)
        self.assertIn("n_heads > 16", inf_cu)
        self.assertIn("d_model != 512 || vocab_size < 256", inf_cu)
        self.assertNotIn("50M: D=768", inf_cu)
        ext_cu = (e2e.DEFAULT_CSRC_DIR / e2e.EXT_CU_FILENAME).read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: Apache-2.0", ext_cu)
        self.assertNotIn("100% RF", ext_cu)
        self.assertNotIn("99% SMEM", ext_cu)

        # Complete fail-closed tamper controls killing every require_ok inside verify_e2e_ptxas_receipt
        td = Path(tempfile.mkdtemp(prefix="holo_e2e_r5_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        with self.assertRaises(HoloSphereError) as ctx_mr:
            e2e.verify_e2e_ptxas_receipt(td)
        self.assertEqual(ctx_mr.exception.code, "E2E_MISSING_RECEIPT")

        for bad_txt, exp_code in (
            ("{bad json", "E2E_BAD_JSON"),
            ("[]", "E2E_BAD_JSON"),
            ('{"x": NaN}', "E2E_NONFINITE_JSON"),
            ('{"x": Infinity}', "E2E_NONFINITE_JSON"),
            ('{"x": 1e500}', "E2E_NONFINITE_JSON"),
            ('{"nonzero_spill_kernels": 7, "nonzero_spill_kernels": 0}', "E2E_DUP_JSON_KEY"),
        ):
            (td / e2e.PTXAS_RECEIPT_FILENAME).write_text(bad_txt, encoding="utf-8")
            with self.assertRaises(HoloSphereError) as ctx_bj:
                e2e.verify_e2e_ptxas_receipt(td)
            self.assertEqual(ctx_bj.exception.code, exp_code)

        rec_src = json.loads((e2e.DEFAULT_CSRC_DIR / e2e.PTXAS_RECEIPT_FILENAME).read_text(encoding="utf-8"))
        # Missing .cu source
        (td / e2e.PTXAS_RECEIPT_FILENAME).write_text(json.dumps(rec_src), encoding="utf-8")
        with self.assertRaises(HoloSphereError) as ctx_mcu:
            e2e.verify_e2e_ptxas_receipt(td)
        self.assertEqual(ctx_mcu.exception.code, "E2E_MISSING_CU")

        shutil.copy2(e2e.DEFAULT_CSRC_DIR / e2e.EXT_CU_FILENAME, td / e2e.EXT_CU_FILENAME)
        shutil.copy2(e2e.DEFAULT_CSRC_DIR / e2e.INFER_CU_FILENAME, td / e2e.INFER_CU_FILENAME)

        for code, mut in (
            ("E2E_PTXAS_SCHEMA", lambda r: r.__setitem__("schema", "bad.v0")),
            ("E2E_PTXAS_ARCH", lambda r: r.__setitem__("arch", "sm_75")),
            ("E2E_NONZERO_SPILL", lambda r: r.__setitem__("nonzero_spill_kernels", 1)),
            ("E2E_NONZERO_SPILL", lambda r: r.__setitem__("nonzero_spill_kernels", False)),
            ("E2E_TOTAL_KERNELS", lambda r: r.__setitem__("total_kernels", 15)),
            ("E2E_TOTAL_KERNELS", lambda r: r.__setitem__("total_kernels", True)),
            ("E2E_UNITS_SCHEMA", lambda r: r.__setitem__("units", [])),
            ("E2E_UNITS_SCHEMA", lambda r: r["units"].__setitem__("training_ext", "bad")),
            ("E2E_CU_SHA_MISMATCH", lambda r: r["units"]["training_ext"].__setitem__("cu_filename", "wrong.cu")),
            ("E2E_CU_SHA_MISMATCH", lambda r: r["units"]["training_ext"].__setitem__("cu_sha256", "0" * 64)),
            ("E2E_UNIT_SPILL", lambda r: r["units"]["training_ext"].__setitem__("nonzero_spill_kernels", 1)),
            ("E2E_UNIT_SPILL", lambda r: r["units"]["training_ext"].__setitem__("nonzero_spill_kernels", False)),
            ("E2E_KERNEL_COUNT", lambda r: r["units"]["training_ext"]["kernels"].pop()),
            ("E2E_KERNEL_COUNT", lambda r: r["units"]["training_ext"].__setitem__("total_kernels", 3)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("stack_frame_bytes", 8)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("spill_store_bytes", 8)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("spill_load_bytes", 8)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("zero_spill", False)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("registers", 0)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("registers", True)),
            ("E2E_KERNEL_SPILL", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("smem_bytes", -1)),
            ("E2E_SYMBOLS", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("symbol", "")),
            ("E2E_SYMBOLS", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("symbol", ["unhashable"])),
            ("E2E_SYMBOLS", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("symbol", "bogus")),
            (
                "E2E_SYMBOLS",
                lambda r: r["units"]["training_ext"].__setitem__(
                    "kernels", [copy.deepcopy(r["units"]["training_ext"]["kernels"][0])] * 4
                ),
            ),
            ("E2E_SASS_SCHEMA", lambda r: r["units"]["training_ext"]["kernels"][0].__setitem__("sass", [])),
            ("E2E_SASS_SCHEMA", lambda r: r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("SHFL", True)),
            ("E2E_SASS_SCHEMA", lambda r: r["units"]["training_ext"].__setitem__("sass_instruction_counts", [])),
            ("E2E_SASS_LOCAL", lambda r: r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("LDL", 1)),
            ("E2E_SASS_LOCAL", lambda r: r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("STL", 1)),
            ("E2E_SASS_L2BAR", lambda r: r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("ATOMG_GPU", 1)),
            ("E2E_SASS_L2BAR", lambda r: r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("MEMBAR_GL", 1)),
            ("E2E_SASS_AGG", lambda r: r["units"]["training_ext"]["sass_instruction_counts"].__setitem__("SHFL", 0)),
            (
                "E2E_HMMA_COUNT",
                lambda r: (
                    r["units"]["training_ext"]["kernels"][0]["sass"].__setitem__("HMMA_16816", 27),
                    r["units"]["training_ext"]["sass_instruction_counts"].__setitem__("HMMA_16816", 27),
                ),
            ),
            (
                "E2E_RED_SYS",
                lambda r: (
                    r["units"]["inference_runtime"]["kernels"][8]["sass"].__setitem__("RED_SYS", 1),
                    r["units"]["inference_runtime"]["sass_instruction_counts"].__setitem__("RED_SYS", 5),
                ),
            ),
            (
                "E2E_RED_SYS",
                lambda r: (
                    r["units"]["inference_runtime"]["kernels"][8]["sass"].__setitem__("MEMBAR_SYS", 2),
                    r["units"]["inference_runtime"]["sass_instruction_counts"].__setitem__("MEMBAR_SYS", 8),
                ),
            ),
        ):
            bad = copy.deepcopy(rec_src)
            mut(bad)
            (td / e2e.PTXAS_RECEIPT_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(HoloSphereError) as ctx_t:
                e2e.verify_e2e_ptxas_receipt(td)
            self.assertEqual(ctx_t.exception.code, code)

    def test_02_zone_b_live_receipts_and_tamper_controls(self) -> None:
        live = e2e.verify_e2e_live_receipts()
        self.assertEqual(live["zone_b_receipt_sha256"], "c989c697dc5f7b0106971755239bcd1d56ceab9365b789891bc55c9e13b8c3a6")
        self.assertGreater(live["core_6L_fwd_bwd_tok_s"], 130000.0)
        self.assertGreater(live["row3_15m_prefill_tok_s"], 45000.0)
        self.assertGreater(live["row3_15m_train_gc_tok_s"], 23000.0)
        self.assertEqual(live["a16_u64_blocks"], 945280)
        self.assertLess(live["infer_max_logit_abs_err"], 1e-5)

        td = Path(tempfile.mkdtemp(prefix="holo_live_r5_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        with self.assertRaises(HoloSphereError) as ctx_mzb:
            e2e.verify_e2e_live_receipts(td)
        self.assertEqual(ctx_mzb.exception.code, "E2E_MISSING_ZONE_B")

        zb_src = json.loads((e2e.DEFAULT_CSRC_DIR / e2e.ZONE_B_RECEIPT_FILENAME).read_text(encoding="utf-8"))
        ptx_src = json.loads((e2e.DEFAULT_CSRC_DIR / e2e.PTXAS_RECEIPT_FILENAME).read_text(encoding="utf-8"))
        tr_src = json.loads((e2e.DEFAULT_CSRC_DIR / e2e.TRAIN_LIVE_FILENAME).read_text(encoding="utf-8"))
        inf_src = json.loads((e2e.DEFAULT_CSRC_DIR / e2e.INFER_LIVE_FILENAME).read_text(encoding="utf-8"))

        def _resign_zb(zb_obj: dict) -> str:
            body = {k: v for k, v in zb_obj.items() if k != "receipt_sha256"}
            sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            zb_obj["receipt_sha256"] = sha
            return sha

        def _write_all(zb: dict, ptx: dict, tr: dict, inf: dict) -> None:
            (td / e2e.ZONE_B_RECEIPT_FILENAME).write_text(json.dumps(zb), encoding="utf-8")
            (td / e2e.PTXAS_RECEIPT_FILENAME).write_text(json.dumps(ptx), encoding="utf-8")
            (td / e2e.TRAIN_LIVE_FILENAME).write_text(json.dumps(tr), encoding="utf-8")
            (td / e2e.INFER_LIVE_FILENAME).write_text(json.dumps(inf), encoding="utf-8")

        (td / e2e.ZONE_B_RECEIPT_FILENAME).write_text(json.dumps(zb_src), encoding="utf-8")
        with self.assertRaises(HoloSphereError) as ctx_mptx:
            e2e.verify_e2e_live_receipts(td)
        self.assertEqual(ctx_mptx.exception.code, "E2E_MISSING_RECEIPT")

        (td / e2e.PTXAS_RECEIPT_FILENAME).write_text(json.dumps(ptx_src), encoding="utf-8")
        with self.assertRaises(HoloSphereError) as ctx_mtr:
            e2e.verify_e2e_live_receipts(td)
        self.assertEqual(ctx_mtr.exception.code, "E2E_MISSING_TRAIN_LIVE")

        (td / e2e.TRAIN_LIVE_FILENAME).write_text(json.dumps(tr_src), encoding="utf-8")
        with self.assertRaises(HoloSphereError) as ctx_minf:
            e2e.verify_e2e_live_receipts(td)
        self.assertEqual(ctx_minf.exception.code, "E2E_MISSING_INFER_LIVE")

        for target, resign, code, mut in (
            ("zb", False, "E2E_ZONE_B_INVALID", lambda d: d.__setitem__("clean_hardware_run", False)),
            ("zb", False, "E2E_ZONE_B_INVALID", lambda d: d.__setitem__("anomalies", ["throttled"])),
            ("zb", False, "E2E_ZONE_B_INVALID", lambda d: d.__setitem__("receipt_sha256", "not-hex")),
            ("zb", False, "E2E_ZONE_B_SHA", lambda d: d.__setitem__("receipt_sha256", "a" * 64)),
            ("zb", False, "E2E_ZONE_B_SHA", lambda d: d.__setitem__("profile_id", "tampered")),
            ("zb", True, "E2E_ZONE_B_PINS", lambda d: d.__setitem__("determinism_pins", {})),
            ("ptx", False, "E2E_ZONE_B_BINDING", lambda d: d.__setitem__("zone_b_receipt_sha256", "b" * 64)),
            ("zb", True, "E2E_ZONE_B_EXEC", lambda d: d["execution"].__setitem__("exit_code", 1)),
            ("zb", True, "E2E_ZONE_B_EXEC", lambda d: d["execution"].__setitem__("exit_code", False)),
            ("zb", True, "E2E_ZONE_B_EXEC", lambda d: d["end_gpu_telemetry"]["clocks_event_reasons"].__setitem__("thermal_throttled", True)),
            ("zb", True, "E2E_ZONE_B_DEVICE", lambda d: d["end_gpu_telemetry"].__setitem__("gpu_name", "CPU")),
            ("tr", False, "E2E_ZONE_B_DEVICE", lambda d: d.__setitem__("device", "CPU")),
            ("inf", False, "E2E_ZONE_B_DEVICE", lambda d: d.__setitem__("device", "CPU")),
            ("tr", False, "E2E_TRAIN_SCHEMA", lambda d: d.__setitem__("schema", "bad")),
            ("tr", False, "E2E_TRAIN_PARITY", lambda d: d["fp64_autograd_parity_T96_3chunks"].__setitem__("worst_err", 5e-4)),
            ("tr", False, "E2E_TRAIN_PARITY", lambda d: d["fp64_autograd_parity_T96_3chunks"].__setitem__("worst_err", True)),
            ("tr", False, "E2E_TRAIN_PARITY", lambda d: d["torch_compile_fullgraph_inductor"].__setitem__("max_diff_vs_eager", 1e-6)),
            ("tr", False, "E2E_TRAIN_PARITY", lambda d: d.__setitem__("fused_a16_ste_cuda_max_diff_vs_pytorch", 1e-5)),
            ("tr", False, "E2E_TRAIN_BENCH", lambda d: d["benchmarks"]["core_chunk32_6L_B4_T8192_fwd_bwd"].__setitem__("tok_per_sec_6L_fwd_bwd", 0.0)),
            ("inf", False, "E2E_INFER_SCHEMA", lambda d: d.__setitem__("schema", "bad")),
            ("inf", False, "E2E_INFER_PARITY", lambda d: d["a16_packing"].__setitem__("total_syndrome_errors", 1)),
            ("inf", False, "E2E_INFER_PARITY", lambda d: d["real_weight_parity_32tok"].__setitem__("clean_canary_sec", 1)),
            ("inf", False, "E2E_INFER_PARITY", lambda d: d["real_weight_parity_32tok"].__setitem__("max_logit_abs_err_vs_pytorch_fp32", 1e-3)),
            ("inf", False, "E2E_INFER_SECDED", lambda d: d["live_sec16_heal_and_razor_tier1"].__setitem__("sec16_logit_max_abs_diff", 1e-6)),
            ("inf", False, "E2E_INFER_SECDED", lambda d: d["live_ded1_weft_pi16_rollback_recovery"].__setitem__("live_sys_canary_ded_count", 0)),
            ("inf", False, "E2E_INFER_SECDED", lambda d: d["live_ded1_weft_pi16_rollback_recovery"].__setitem__("weft_pi16_checkpoint_bytes", 32)),
            ("inf", False, "E2E_INFER_SECDED", lambda d: d["live_ded1_weft_pi16_rollback_recovery"].__setitem__("razor_ded_latch_epochs", 32)),
        ):
            zb_b, ptx_b, tr_b, inf_b = (
                copy.deepcopy(zb_src),
                copy.deepcopy(ptx_src),
                copy.deepcopy(tr_src),
                copy.deepcopy(inf_src),
            )
            mut({"zb": zb_b, "ptx": ptx_b, "tr": tr_b, "inf": inf_b}[target])
            if resign:
                new_sha = _resign_zb(zb_b)
                ptx_b["zone_b_receipt_sha256"] = new_sha
            _write_all(zb_b, ptx_b, tr_b, inf_b)
            with self.assertRaises(HoloSphereError) as ctx_l:
                e2e.verify_e2e_live_receipts(td)
            self.assertEqual(ctx_l.exception.code, code)

    def test_03_stdlib_a16_secded_dual_swar_weft16_razor64_and_pure_stdlib_probe(self) -> None:
        st = e2e.run_e2e_stdlib_self_test()
        self.assertEqual(st["a16_clean_syndrome_errors"], 0)
        self.assertLess(st["a16_dot_parity_err"], 1e-12)
        self.assertEqual(st["a16_sec_healed"], 16)
        self.assertEqual(st["a16_ded_caught"], 120)
        self.assertEqual(st["weft_pi16_bytes"], 16)
        self.assertEqual(st["razor_canary64_bytes"], 64)

        # Report validator negative controls killing every require_ok inside validate_e2e_stdlib_self_test_report
        with self.assertRaises(HoloSphereError) as ctx_st0:
            e2e.validate_e2e_stdlib_self_test_report("bad")  # type: ignore[arg-type]
        self.assertEqual(ctx_st0.exception.code, "E2E_ST_SCHEMA")
        for code, mut in (
            ("A16_CLEAN_SYN", lambda r: r.__setitem__("a16_clean_syndrome_errors", 1)),
            ("A16_DOT_PARITY", lambda r: r.__setitem__("a16_dot_parity_err", 1e-12)),
            ("A16_SWAR_CLEAN", lambda r: r.__setitem__("a16_swar_clean_zero", False)),
            ("A16_SEC_FAIL", lambda r: r.__setitem__("a16_sec_healed", 15)),
            ("A16_DED_FAIL", lambda r: r.__setitem__("a16_ded_caught", 119)),
            ("WEFT_PI16_DRIFT", lambda r: r.__setitem__("weft_pi16_err", 1e-6)),
            ("RAZOR64_DRIFT", lambda r: r.__setitem__("razor_canary64_roundtrip_ok", False)),
        ):
            bad_st = copy.deepcopy(st)
            mut(bad_st)
            with self.assertRaises(HoloSphereError) as ctx_st:
                e2e.validate_e2e_stdlib_self_test_report(bad_st)
            self.assertEqual(ctx_st.exception.code, code)

        # Negative controls on pure-stdlib primitives
        for bad_u64 in (-1, 1 << 64, True):
            with self.assertRaises(HoloSphereError) as ctx_u64:
                e2e.a16_extract_odd16_u64(bad_u64)  # type: ignore[arg-type]
            self.assertEqual(ctx_u64.exception.code, "A16_BAD_U64")
            with self.assertRaises(HoloSphereError) as ctx_d64:
                e2e.a16_blockh4_dot16_stdlib(bad_u64, [0.0] * 16)  # type: ignore[arg-type]
            self.assertEqual(ctx_d64.exception.code, "A16_BAD_U64")
        for bad_u16 in (-1, 0x10000, True):
            with self.assertRaises(HoloSphereError) as ctx_u16:
                e2e.a16_secded_syndrome_u16(bad_u16)  # type: ignore[arg-type]
            self.assertEqual(ctx_u16.exception.code, "A16_BAD_U16")
        for bad_u32 in (-1, 1 << 32, True):
            with self.assertRaises(HoloSphereError) as ctx_u32:
                e2e.dual_a16_syndrome_nonzero_swar_u32(bad_u32)  # type: ignore[arg-type]
            self.assertEqual(ctx_u32.exception.code, "A16_BAD_U32")
        for bad_x16 in (5, {i: 0.0 for i in range(16)}, [0.0] * 15, [float("nan")] + [0.0] * 15, [1e300] + [0.0] * 15, [True] + [0.0] * 15):
            with self.assertRaises(HoloSphereError) as ctx_x16:
                e2e.a16_blockh4_dot16_stdlib(0, bad_x16)  # type: ignore[arg-type]
            self.assertEqual(ctx_x16.exception.code, "A16_BAD_X16")
        for bad_row in (5, {i: 0.0 for i in range(32)}, [], [0.0] * 16, [float("inf")] + [0.0] * 31, [1e300] + [0.0] * 31, [10**400] + [0.0] * 31):
            with self.assertRaises(HoloSphereError) as ctx_r:
                e2e.a16_blockh4_pack_row_stdlib(bad_row)  # type: ignore[arg-type]
            self.assertEqual(ctx_r.exception.code, "A16_BAD_ROW")
        # F8: reject scale overflow beyond FP16 ceiling (1e7 / 3e38) while allowing exact boundary 245640.0
        p_edge, sc_edge, _ = e2e.a16_blockh4_pack_row_stdlib([245640.0] * 32)
        self.assertEqual(len(p_edge), 2)
        self.assertEqual(sc_edge[0], 65504.0)
        for ovf_row in ([1e7] + [0.0] * 31, [3e38] * 32):
            with self.assertRaises(HoloSphereError) as ctx_ov:
                e2e.a16_blockh4_pack_row_stdlib(ovf_row)
            self.assertEqual(ctx_ov.exception.code, "A16_OVERFLOW")

        # Weft 16 B Pi_{t_0} exact dyadic boundary (n2 = 1 + 2^-10 == 1 + WEFT_PI16_UNIT_TOL) + negative controls
        pi_edge = e2e.pack_weft_pi16(1.0, 2.0**-5, 0.0, 0.0)
        self.assertEqual(len(pi_edge), 16)
        self.assertEqual(e2e.unpack_weft_pi16(pi_edge), (1.0, 2.0**-5, 0.0, 0.0))

        for bad_w in (float("nan"), 1e300, True):
            with self.assertRaises(HoloSphereError) as ctx_p1:
                e2e.pack_weft_pi16(bad_w, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
            self.assertEqual(ctx_p1.exception.code, "WEFT_PI16_NONFINITE")
        with self.assertRaises(HoloSphereError) as ctx_p2:
            e2e.pack_weft_pi16(2.0, 0.0, 0.0, 0.0)
        self.assertEqual(ctx_p2.exception.code, "WEFT_PI16_NONUNIT")
        with self.assertRaises(HoloSphereError) as ctx_p3:
            e2e.unpack_weft_pi16(b"\x00" * 12)
        self.assertEqual(ctx_p3.exception.code, "WEFT_PI16_LEN")
        with self.assertRaises(HoloSphereError) as ctx_p4:
            e2e.unpack_weft_pi16(struct.pack("<4f", float("nan"), 0.0, 0.0, 0.0))
        self.assertEqual(ctx_p4.exception.code, "WEFT_PI16_NONFINITE")
        with self.assertRaises(HoloSphereError) as ctx_p5:
            e2e.unpack_weft_pi16(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(ctx_p5.exception.code, "WEFT_PI16_NONUNIT")

        # check_cuda_rc & check_a16_zero_syndrome boundary & negative controls
        e2e.check_cuda_rc(0, "OK_CODE")
        for bad_rc in (1, -1, True):
            with self.assertRaises(HoloSphereError) as ctx_rc:
                e2e.check_cuda_rc(bad_rc, "TEST_CUDA_RC")  # type: ignore[arg-type]
            self.assertEqual(ctx_rc.exception.code, "TEST_CUDA_RC")
        e2e_inf.check_a16_zero_syndrome(0)
        for bad_syn in (1, True):
            with self.assertRaises(HoloSphereError) as ctx_syn:
                e2e_inf.check_a16_zero_syndrome(bad_syn)  # type: ignore[arg-type]
            self.assertEqual(ctx_syn.exception.code, "A16_TORCH_PACK_SYN_ERR")

        # F1-bis tied FWHT_256 LM head dimension guard (D == 512, V >= 256)
        e2e_inf.check_tied_fwht256_dims(512, 256)
        e2e_inf.check_tied_fwht256_dims(512, 260)
        for bad_d, bad_v in ((256, 256), (768, 260), (512, 255), (512, 0), (True, 256), (512, True)):
            with self.assertRaises(HoloSphereError) as ctx_fwht:
                e2e_inf.check_tied_fwht256_dims(bad_d, bad_v)  # type: ignore[arg-type]
            self.assertEqual(ctx_fwht.exception.code, "FWHT_HEAD_DIMS")

        # compile_and_load_cuda_lib F2 timeout, OSError, non-zero exit, missing .so & corrupt .so controls (CPU-safe)
        ext_cu_path = e2e.DEFAULT_CSRC_DIR / e2e.EXT_CU_FILENAME
        for bad_to in (0.0, -1.0, float("nan"), True):
            with self.assertRaises(HoloSphereError) as ctx_to:
                e2e.compile_and_load_cuda_lib(ext_cu_path, nvcc_bin="/usr/bin/false", timeout=bad_to)  # type: ignore[arg-type]
            self.assertEqual(ctx_to.exception.code, "NVCC_BAD_TIMEOUT")

        before_tmp = set(Path(tempfile.gettempdir()).glob("holosphere_v3_e2e_*"))
        for bad_nvcc in ("/nonexistent/holosphere_nvcc_missing", "/usr/bin/false"):
            with self.assertRaises(HoloSphereError) as ctx_nv:
                e2e.compile_and_load_cuda_lib(ext_cu_path, nvcc_bin=bad_nvcc)
            self.assertEqual(ctx_nv.exception.code, "NVCC_COMPILE_FAIL")
        with self.assertRaises(HoloSphereError) as ctx_true:
            e2e.compile_and_load_cuda_lib(ext_cu_path, nvcc_bin="/usr/bin/true")
        self.assertEqual(ctx_true.exception.code, "NVCC_LOAD_FAIL")
        self.assertIn("without producing", str(ctx_true.exception))

        # Fake compiler that writes a corrupt .so -> CDLL OSError wrapped as NVCC_LOAD_FAIL & tmp_dir cleaned
        td_fake = Path(tempfile.mkdtemp(prefix="holo_fake_nvcc_"))
        self.addCleanup(lambda: shutil.rmtree(td_fake, ignore_errors=True))
        fake_nvcc = td_fake / "fake_nvcc.sh"
        fake_nvcc.write_text('#!/bin/sh\nwhile [ "$#" -gt 0 ]; do if [ "$1" = "-o" ]; then printf "BADELF" > "$2"; exit 0; fi; shift; done\nexit 0\n', encoding="utf-8")
        fake_nvcc.chmod(fake_nvcc.stat().st_mode | stat.S_IXUSR)
        with self.assertRaises(HoloSphereError) as ctx_badso:
            e2e.compile_and_load_cuda_lib(ext_cu_path, nvcc_bin=str(fake_nvcc))
        self.assertEqual(ctx_badso.exception.code, "NVCC_LOAD_FAIL")
        self.assertIn("Failed to load", str(ctx_badso.exception))
        after_tmp = set(Path(tempfile.gettempdir()).glob("holosphere_v3_e2e_*"))
        self.assertEqual(after_tmp - before_tmp, set())

        # Bad path types raise typed HoloSphereError (never raw TypeError)
        for bad_path in (5, "", b"/tmp/x"):
            with self.assertRaises(HoloSphereError) as ctx_bp1:
                e2e.verify_e2e_ptxas_receipt(bad_path)
            self.assertEqual(ctx_bp1.exception.code, "E2E_MISSING_RECEIPT")
            with self.assertRaises(HoloSphereError) as ctx_bp2:
                e2e.verify_e2e_live_receipts(bad_path)
            self.assertEqual(ctx_bp2.exception.code, "E2E_MISSING_ZONE_B")
            with self.assertRaises(HoloSphereError) as ctx_bp3:
                e2e.compile_and_load_cuda_lib(bad_path)
            self.assertEqual(ctx_bp3.exception.code, "CU_NOT_FOUND")

        # Razor 64 B ABI F9 strict key & negative controls
        with self.assertRaises(HoloSphereError) as ctx_r1:
            e2e.pack_razor_canary64("bad")  # type: ignore[arg-type]
        self.assertEqual(ctx_r1.exception.code, "RAZOR64_BAD_FIELDS")
        for bad_keys in ({}, {"sec_count_u3": 5}):
            with self.assertRaises(HoloSphereError) as ctx_rk:
                e2e.pack_razor_canary64(bad_keys)
            self.assertEqual(ctx_rk.exception.code, "RAZOR64_BAD_KEYS")
        full_bad_u32 = {k: 0 for k in e2e._RAZOR_FIELDS}
        full_bad_u32["sec_count_u32"] = True  # type: ignore[assignment]
        with self.assertRaises(HoloSphereError) as ctx_r2:
            e2e.pack_razor_canary64(full_bad_u32)
        self.assertEqual(ctx_r2.exception.code, "RAZOR64_BAD_U32")
        with self.assertRaises(HoloSphereError) as ctx_r3:
            e2e.unpack_razor_canary64(b"\x00" * 60)
        self.assertEqual(ctx_r3.exception.code, "RAZOR64_LEN")
        with self.assertRaises(HoloSphereError) as ctx_r4:
            e2e.unpack_razor_canary64(b"\x00" * 60 + struct.pack("<I", 1))
        self.assertEqual(ctx_r4.exception.code, "RAZOR64_RESERVED")

        # F6 CLI --check / --json / human-readable differentiation on both e2e_engine and e2e_inference
        buf_e_j = io.StringIO()
        with redirect_stdout(buf_e_j):
            self.assertEqual(e2e.main(["--check", "--json"]), 0)
        self.assertEqual(json.loads(buf_e_j.getvalue())["schema"], "holosphere.track_s117.e2e_engine_cli.v1")
        buf_e_t = io.StringIO()
        with redirect_stdout(buf_e_t):
            self.assertEqual(e2e.main(["--check"]), 0)
        self.assertIn("PASS holosphere.e2e_engine", buf_e_t.getvalue())

        buf_i_j = io.StringIO()
        with redirect_stdout(buf_i_j):
            self.assertEqual(e2e_inf.main(["--check", "--json"]), 0)
        inf_cli = json.loads(buf_i_j.getvalue())
        self.assertEqual(inf_cli["schema"], "holosphere.track_s117.e2e_inference_cli.v1")
        self.assertNotIn("stdlib_a16_secded", inf_cli)
        buf_i_t = io.StringIO()
        with redirect_stdout(buf_i_t):
            self.assertEqual(e2e_inf.main(["--self-test"]), 0)
        self.assertIn("PASS holosphere.e2e_inference", buf_i_t.getvalue())

        # Missing CU source guard in compile_and_load_cuda_lib + _require_torch guard
        with self.assertRaises(HoloSphereError) as ctx_ncu:
            e2e.compile_and_load_cuda_lib(REPO_ROOT / "nonexistent_v3_kernel.cu")
        self.assertEqual(ctx_ncu.exception.code, "CU_NOT_FOUND")
        old_has = e2e._HAS_TORCH
        try:
            e2e._HAS_TORCH = False
            with self.assertRaises(HoloSphereError) as ctx_nt:
                e2e._require_torch()
            self.assertEqual(ctx_nt.exception.code, "NO_TORCH")
        finally:
            e2e._HAS_TORCH = old_has

        # Pure-stdlib torch-blocked import & CLI probe
        probe = (
            f"import sys; sys.modules['torch'] = None; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from holosphere import e2e_engine, e2e_inference; "
            "r1 = e2e_engine.verify_e2e_ptxas_receipt(); "
            "r2 = e2e_engine.verify_e2e_live_receipts(); "
            "r3 = e2e_engine.run_e2e_stdlib_self_test(); "
            "print(r1['total_kernels'], r2['a16_u64_blocks'], r3['a16_ded_caught'])"
        )
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "16 945280 120")

    @unittest.skipUnless(e2e._HAS_TORCH, "requires torch")
    def test_04_torch_cpu_fp64_autograd_compile_a16_packer_and_guards(self) -> None:
        import torch

        res = e2e.run_cpu_e2e_autograd_self_test()
        self.assertLess(res["worst_fp64_autograd_err"], 1e-12)
        self.assertLess(res["torch_compile_fullgraph_diff"], 1e-12)
        self.assertEqual(res["a16_ste_grad_err"], 0.0)
        self.assertTrue(res["fast_vring_ste_finite"])

        # Report validator negative controls killing every require_ok inside validate_cpu_e2e_autograd_report
        with self.assertRaises(HoloSphereError) as ctx_cr0:
            e2e.validate_cpu_e2e_autograd_report("bad")  # type: ignore[arg-type]
        self.assertEqual(ctx_cr0.exception.code, "V3_CPU_REPORT_SCHEMA")
        for code, mut in (
            ("V3_CPU_AUTOGRAD_DRIFT", lambda r: r.__setitem__("worst_fp64_autograd_err", 1e-12)),
            ("V3_COMPILE_DRIFT", lambda r: r.__setitem__("torch_compile_fullgraph_diff", 1e-12)),
            ("V3_VRING_GRAD_NONFINITE", lambda r: r.__setitem__("fast_vring_ste_finite", False)),
            ("A16_STE_GRAD", lambda r: r.__setitem__("a16_ste_grad_err", 1e-6)),
        ):
            bad_cr = copy.deepcopy(res)
            mut(bad_cr)
            with self.assertRaises(HoloSphereError) as ctx_cr:
                e2e.validate_cpu_e2e_autograd_report(bad_cr)
            self.assertEqual(ctx_cr.exception.code, code)

        # HoloSphereV3BareMetalInferEngine & pack_linear_a16_blockh4_torch negative controls (CPU-safe)
        with self.assertRaises(HoloSphereError) as ctx_ck1:
            e2e_inf.HoloSphereV3BareMetalInferEngine(REPO_ROOT / "nonexistent_ckpt.pt", "cpu")
        self.assertEqual(ctx_ck1.exception.code, "CKPT_MISSING")
        with self.assertRaises(HoloSphereError) as ctx_ck2:
            e2e_inf.HoloSphereV3BareMetalInferEngine(REPO_ROOT / "README.md", "cpu")
        self.assertEqual(ctx_ck2.exception.code, "CUDA_REQ")
        old_inf_torch = e2e_inf._HAS_TORCH
        try:
            e2e_inf._HAS_TORCH = False
            with self.assertRaises(HoloSphereError) as ctx_it:
                e2e_inf.pack_linear_a16_blockh4_torch(torch.zeros((2, 32)))
            self.assertEqual(ctx_it.exception.code, "NO_TORCH")
        finally:
            e2e_inf._HAS_TORCH = old_inf_torch

        # Vectorized PyTorch CPU A_16 packer parity vs pure-stdlib packer
        row = [math.sin(0.37 * c + 0.19) * 0.45 for c in range(64)]
        p_std, sc_std, deq_std = e2e.a16_blockh4_pack_row_stdlib(row)
        w_t = torch.tensor([row], dtype=torch.float32)
        p_t, sc_t, deq_t, serr = e2e_inf.pack_linear_a16_blockh4_torch(w_t)
        self.assertEqual(serr, 0)
        self.assertEqual([int(v) & 0xFFFFFFFFFFFFFFFF for v in p_t[0].tolist()], p_std)
        self.assertLess(max(abs(a - b) for a, b in zip(deq_t[0].tolist(), deq_std)), 1e-6)
        self.assertEqual(sc_t.shape, (1, 2))
        self.assertEqual(len(sc_std), 2)

        # Negative controls on pack_linear_a16_blockh4_torch (F8 bool/int dtype & scale overflow)
        for bad_t in (5, torch.zeros((32,))):
            with self.assertRaises(HoloSphereError) as ctx_s1:
                e2e_inf.pack_linear_a16_blockh4_torch(bad_t)
            self.assertEqual(ctx_s1.exception.code, "A16_BAD_SHAPE")
        for bad_dt in (torch.zeros((2, 32), dtype=torch.bool), torch.zeros((2, 32), dtype=torch.int64)):
            with self.assertRaises(HoloSphereError) as ctx_dt:
                e2e_inf.pack_linear_a16_blockh4_torch(bad_dt)
            self.assertEqual(ctx_dt.exception.code, "A16_BAD_DTYPE")
        with self.assertRaises(HoloSphereError) as ctx_s2:
            e2e_inf.pack_linear_a16_blockh4_torch(torch.zeros((2, 30)))
        self.assertEqual(ctx_s2.exception.code, "A16_IN_DIV32")
        bad_w = torch.zeros((2, 32))
        bad_w[0, 0] = float("nan")
        with self.assertRaises(HoloSphereError) as ctx_s3:
            e2e_inf.pack_linear_a16_blockh4_torch(bad_w)
        self.assertEqual(ctx_s3.exception.code, "A16_NONFINITE")
        # Exact FP16 boundary passes; 1e7 / 3e38 fail closed with A16_OVERFLOW
        _, sc_b, _, serr_b = e2e_inf.pack_linear_a16_blockh4_torch(torch.full((1, 32), 245640.0, dtype=torch.float32))
        self.assertEqual(serr_b, 0)
        self.assertEqual(float(sc_b[0, 0].item()), 65504.0)
        for ov_t in (torch.full((1, 32), 1e7, dtype=torch.float32), torch.full((1, 32), 3e38, dtype=torch.float32)):
            with self.assertRaises(HoloSphereError) as ctx_ov_t:
                e2e_inf.pack_linear_a16_blockh4_torch(ov_t)
            self.assertEqual(ctx_ov_t.exception.code, "A16_OVERFLOW")

        # F3: CPU project_boundary_tangent=1 matches CUDA G4 (orthogonal to U_0 on T_{U_0} S^3) vs 0
        gen_t = torch.Generator(device="cpu").manual_seed(777)
        x_p = torch.randn((1, 64, 32), generator=gen_t, dtype=torch.float64)
        x_p = x_p / x_p.norm(dim=-1, keepdim=True)
        w_p = 0.1 * torch.randn((1, 64, 1, 8, 3), generator=gen_t, dtype=torch.float64)
        wq_p = 0.08 * torch.randn((1, 32, 32), generator=gen_t, dtype=torch.float64)
        u0_raw = torch.randn((1, 1, 8, 4), generator=gen_t, dtype=torch.float64)
        u0_unit = u0_raw / u0_raw.norm(dim=-1, keepdim=True)
        u0_p1 = u0_unit.clone().requires_grad_(True)
        y_p1, se_p1, ue_p1 = e2e.holosphere_v3_chunk32_fused(x_p, w_p, wq_p, None, u0_p1, project_boundary_tangent=1)
        (y_p1.square().mean() + 0.1 * se_p1.square().mean() + 0.2 * ue_p1.square().mean()).backward()
        radial_p1 = float((u0_p1.grad * u0_unit).sum(dim=-1).abs().max().item())
        self.assertLess(radial_p1, 1e-12)
        u0_p0 = u0_unit.clone().requires_grad_(True)
        y_p0, se_p0, ue_p0 = e2e.holosphere_v3_chunk32_fused(x_p, w_p, wq_p, None, u0_p0, project_boundary_tangent=0)
        (y_p0.square().mean() + 0.1 * se_p0.square().mean() + 0.2 * ue_p0.square().mean()).backward()
        self.assertGreater(float((u0_p1.grad - u0_p0.grad).abs().max().item()), 1e-4)

        # Negative controls on holosphere_v3_chunk32_fused, FusedA16BlockH4STECuda & FastHardFMNMXTangentCayleySTE
        x_ok = torch.ones((1, 32, 32), dtype=torch.float64)
        w_ok = torch.zeros((1, 32, 1, 8, 3), dtype=torch.float64)
        wq_ok = torch.zeros((1, 32, 32), dtype=torch.float64)
        u0_ok = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]] * 8]], dtype=torch.float64)

        for bad_x in (
            5,
            torch.ones((32, 32), dtype=torch.float64),
            torch.ones((1, 16, 32), dtype=torch.float64),
            torch.ones((1, 32, 30), dtype=torch.float64),
        ):
            with self.assertRaises(HoloSphereError) as ctx_bx:
                e2e.holosphere_v3_chunk32_fused(bad_x, w_ok, wq_ok)
            self.assertEqual(ctx_bx.exception.code, "V3_BAD_SHAPE")

        for bad_kw in (
            {"cayley_t": float("nan")},
            {"cayley_t": 1e300},
            {"gamma": float("inf")},
            {"eta": True},
            {"eps": 0.0},
            {"eps": -1e-6},
            {"project_boundary_tangent": 2},
            {"project_boundary_tangent": True},
        ):
            with self.assertRaises(HoloSphereError) as ctx_bp:
                e2e.holosphere_v3_chunk32_fused(x_ok, w_ok, wq_ok, **bad_kw)  # type: ignore[arg-type]
            self.assertEqual(ctx_bp.exception.code, "V3_BAD_PARAM")

        for bad_w_in in (5, torch.zeros((1, 32, 1, 8, 4), dtype=torch.float64)):
            with self.assertRaises(HoloSphereError) as ctx_bw:
                e2e.holosphere_v3_chunk32_fused(x_ok, bad_w_in, wq_ok)
            self.assertEqual(ctx_bw.exception.code, "V3_BAD_WEIGHT_SHAPE")

        for bad_s0 in (5, torch.zeros((1, 1, 16, 16), dtype=torch.float64)):
            with self.assertRaises(HoloSphereError) as ctx_bst:
                e2e.holosphere_v3_chunk32_fused(x_ok, w_ok, wq_ok, S_0=bad_s0, U_0=u0_ok)
            self.assertEqual(ctx_bst.exception.code, "V3_BAD_STATE_SHAPE")

        # F3 dtype, finiteness, and U_0 unit-norm guards
        with self.assertRaises(HoloSphereError) as ctx_dt1:
            e2e.holosphere_v3_chunk32_fused(x_ok.to(torch.int64), w_ok.to(torch.int64), wq_ok.to(torch.int64))
        self.assertEqual(ctx_dt1.exception.code, "V3_BAD_DTYPE")
        with self.assertRaises(HoloSphereError) as ctx_dt2:
            e2e.holosphere_v3_chunk32_fused(x_ok.float(), w_ok, wq_ok)
        self.assertEqual(ctx_dt2.exception.code, "V3_BAD_DTYPE")

        x_nan = x_ok.clone()
        x_nan[0, 0, 0] = float("nan")
        with self.assertRaises(HoloSphereError) as ctx_nf:
            e2e.holosphere_v3_chunk32_fused(x_nan, w_ok, wq_ok)
        self.assertEqual(ctx_nf.exception.code, "V3_NONFINITE")

        u0_zero = torch.zeros_like(u0_ok)
        with self.assertRaises(HoloSphereError) as ctx_u0:
            e2e.holosphere_v3_chunk32_fused(x_ok, w_ok, wq_ok, U_0=u0_zero)
        self.assertEqual(ctx_u0.exception.code, "V3_U0_NONUNIT")

        # F8 FusedA16BlockH4STECuda dtype/shape/nonfinite/overflow + exact boundary pass
        q_edge = e2e.FusedA16BlockH4STECuda.apply(torch.full((1, 32), 245640.0, dtype=torch.float32))
        self.assertTrue(bool(torch.isfinite(q_edge).all().item()))
        with self.assertRaises(HoloSphereError) as ctx_ste_dt:
            e2e.FusedA16BlockH4STECuda.apply(torch.zeros((1, 32), dtype=torch.bool))
        self.assertEqual(ctx_ste_dt.exception.code, "A16_STE_DTYPE")
        with self.assertRaises(HoloSphereError) as ctx_ste:
            e2e.FusedA16BlockH4STECuda.apply(torch.zeros((1, 16)))
        self.assertEqual(ctx_ste.exception.code, "A16_STE_SHAPE")
        w_ste_nan = torch.zeros((1, 32))
        w_ste_nan[0, 0] = float("inf")
        with self.assertRaises(HoloSphereError) as ctx_ste_nf:
            e2e.FusedA16BlockH4STECuda.apply(w_ste_nan)
        self.assertEqual(ctx_ste_nf.exception.code, "A16_STE_NONFINITE")
        with self.assertRaises(HoloSphereError) as ctx_ste_ov:
            e2e.FusedA16BlockH4STECuda.apply(torch.full((1, 32), 1e7, dtype=torch.float32))
        self.assertEqual(ctx_ste_ov.exception.code, "A16_STE_OVERFLOW")

        # F9 FastHardFMNMXTangentCayleySTE tau > 0, eps > 0 guards
        sim_ok = torch.zeros((1, 4, 1, 8), dtype=torch.float32)
        vr_ok = torch.ones((1, 1, 8, 32), dtype=torch.float32)
        for bad_tau, bad_eps in ((0.0, 1e-6), (-0.5, 1e-6), (float("nan"), 1e-6), (0.25, 0.0), (0.25, -1e-6)):
            with self.assertRaises(HoloSphereError) as ctx_vrp:
                e2e.FastHardFMNMXTangentCayleySTE.apply(sim_ok, vr_ok, bad_tau, bad_eps)
            self.assertEqual(ctx_vrp.exception.code, "VRING_BAD_PARAM")


if __name__ == "__main__":
    unittest.main()
