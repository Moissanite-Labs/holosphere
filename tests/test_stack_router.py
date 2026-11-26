#!/usr/bin/env python3
"""3-mode isolated (-I) unit tests for holosphere.stack_router (zero bare asserts)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holosphere.errors import HoloSphereError
from holosphere.stack_router import (
    GA104M_SAFE_L2_PIN_BYTES,
    SAFE_CSR_SPARSITY_CEILING,
    build_optimal_stack_plan,
    compute_2dsd_kv_sharing_ratio,
    evaluate_csr_crossover,
    reconcile_razor_canary_step,
)
from holosphere.telemetry import HoloSphereTuningProfile


class TestInferenceStackRouter(unittest.TestCase):
    def test_01_cold_to_hot_and_batch_tier_routing(self) -> None:
        p_b1 = build_optimal_stack_plan(
            model_family="qwen2",
            target_bpw=Fraction(9, 2),
            workload="code",
            batch_size=1,
            step_tokens=1,
            seq_len=8192,
        )
        self.assertEqual(p_b1.quant_recipe, "hA16_BlockH4_quad_lpt_p0375_g128")
        self.assertEqual(p_b1.lm_head_recipe, "hA16_BlockH4_lm_head_4p5bpw")
        self.assertEqual(p_b1.kernel_target, "silikernels::sm86_s69_sorted_blockh4_dp4a")
        self.assertEqual(p_b1.l2_pin_bytes, GA104M_SAFE_L2_PIN_BYTES)
        self.assertEqual(p_b1.kv_sharing_ratio_q, "512/131")
        self.assertEqual(len(p_b1.plan_sha256()), 64)

        p_b16 = build_optimal_stack_plan(
            model_family="llama",
            target_bpw=Fraction(9, 2),
            workload="prose",
            batch_size=16,
            step_tokens=1,
        )
        self.assertEqual(p_b16.kept_ffn_down_cols, 10)
        self.assertEqual(p_b16.quant_recipe, "hA16_BlockH4_quad_snake_p0000_g128")
        self.assertEqual(p_b16.kernel_target, "silikernels::sm86_edp_b16_tiled_gemm")

        p_b32 = build_optimal_stack_plan(
            model_family="smollm2",
            target_bpw=Fraction(3, 1),
            batch_size=32,
            step_tokens=1,
            csr_sparsity=Fraction(9, 2000),
        )
        self.assertEqual(p_b32.quant_recipe, "b30_hA16_BlockH4_quad_snake_sharednorm")
        self.assertTrue(p_b32.csr_enabled)
        self.assertEqual(p_b32.kernel_target, "silikernels::sm86_cp_async_imma_16832")

        p_cpu = build_optimal_stack_plan(
            model_family="qwen2",
            has_sm86_cuda=False,
            has_avx512_vnni=True,
        )
        self.assertEqual(p_cpu.execution_backend, "torch_cpu")
        self.assertEqual(p_cpu.hot_layout, "avx512_dual_force_64b_cacheline")
        self.assertEqual(p_cpu.kernel_target, "silikernels::avx512_dual_force_vpermb_vpdpbusd")

    def test_02_fail_closed_guards_gemma_unpack_receipt_and_csr(self) -> None:
        with self.assertRaises(HoloSphereError) as cm_unpack:
            build_optimal_stack_plan(model_family="qwen2", serving_mode="on_the_fly_unpack")
        self.assertEqual(cm_unpack.exception.code, "STACK_ON_THE_FLY_UNPACK_FORBIDDEN")

        with self.assertRaises(HoloSphereError) as cm_unver:
            build_optimal_stack_plan(model_family="qwen2", receipt_verified=False)
        self.assertEqual(cm_unver.exception.code, "STACK_UNVERIFIED_ARCHIVE")

        with self.assertRaises(HoloSphereError) as cm_gemma:
            build_optimal_stack_plan(model_family="gemma3", allow_plain_a16_on_gemma=True)
        self.assertEqual(cm_gemma.exception.code, "STACK_GEMMA_PLAIN_A16_FORBIDDEN")

        p_gemma = build_optimal_stack_plan(model_family="gemma3")
        self.assertEqual(p_gemma.quant_recipe, "hA16BG128_block_h2")
        self.assertEqual(p_gemma.butterfly_ops_per_8block, 8)

        c_portal = evaluate_csr_crossover(Fraction(57, 100_000), bits_per_outlier=48)
        self.assertEqual(c_portal["distortion_ratio_q"], "26/35")
        self.assertTrue(c_portal["csr_recommended"])

        c_sqz = evaluate_csr_crossover(Fraction(9, 2_000), bits_per_outlier=32)
        self.assertEqual(c_sqz["distortion_ratio_q"], "25/28")
        self.assertTrue(c_sqz["csr_recommended"])

        c_dense = evaluate_csr_crossover(Fraction(1, 32), bits_per_outlier=32)
        self.assertEqual(c_dense["distortion_ratio_q"], "4/3")
        self.assertFalse(c_dense["csr_recommended"])

        self.assertTrue(evaluate_csr_crossover(SAFE_CSR_SPARSITY_CEILING)["within_safe_ceiling"])
        with self.assertRaises(HoloSphereError) as cm_csr:
            build_optimal_stack_plan(model_family="qwen2", csr_sparsity=Fraction(1, 32))
        self.assertEqual(cm_csr.exception.code, "STACK_CSR_CEILING_EXCEEDED")

    def test_03_2dsd_kv_ratios_and_razor_canary_step_bridge(self) -> None:
        self.assertEqual(compute_2dsd_kv_sharing_ratio(2048, loops_r=4), Fraction(128, 35))
        self.assertEqual(compute_2dsd_kv_sharing_ratio(8192, loops_r=4), Fraction(512, 131))
        self.assertEqual(compute_2dsd_kv_sharing_ratio(32768, loops_r=4), Fraction(2048, 515))
        self.assertEqual(compute_2dsd_kv_sharing_ratio(32768, loops_r=2), Fraction(1024, 513))

        base = HoloSphereTuningProfile(backend="cuda_sm86", block_t=128, microbatch_size=16)

        r_ded = reconcile_razor_canary_step(
            base_profile=base,
            sec_delta=0,
            ded_delta=1,
            non_lsb_delta=0,
            rollback_token_u32=211,
            clean_streak_u32=20,
            hold_remaining_u32=0,
            ded_latch_remaining_u32=0,
        )
        self.assertEqual(r_ded["tier"], 2)
        self.assertTrue(r_ded["rollback_triggered"])
        self.assertEqual(r_ded["aligned_rollback_token_u32"], 192)
        self.assertEqual(r_ded["profile"]["microbatch_size"], 8)
        self.assertEqual(r_ded["profile"]["block_t"], 32)
        self.assertEqual(len(r_ded["receipt_sha256"]), 64)

        r_sec = reconcile_razor_canary_step(
            base_profile=base,
            sec_delta=2,
            ded_delta=0,
            non_lsb_delta=0,
            rollback_token_u32=128,
            clean_streak_u32=5,
            hold_remaining_u32=0,
            ded_latch_remaining_u32=0,
        )
        self.assertEqual(r_sec["tier"], 1)
        self.assertFalse(r_sec["rollback_triggered"])
        self.assertEqual(r_sec["profile"]["block_t"], 64)

        r_clean = reconcile_razor_canary_step(
            base_profile=HoloSphereTuningProfile(backend="cuda_sm86", block_t=128, microbatch_size=None),
            sec_delta=0,
            ded_delta=0,
            non_lsb_delta=0,
            rollback_token_u32=256,
            clean_streak_u32=16,
            hold_remaining_u32=0,
            ded_latch_remaining_u32=0,
        )
        self.assertEqual(r_clean["tier"], 0)
        self.assertEqual(r_clean["action"], "TIER0_CLEAN_EDP_OPTIMAL")
        self.assertEqual(r_clean["profile"]["microbatch_size"], 16)


if __name__ == "__main__":
    unittest.main()
