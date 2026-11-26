"""Deterministic 4-repository inference stack router uniting Siliquant, SiliKernels, HoloSphere, and SiliSense.

Pure Python stdlib only (zero torch/numpy/sensor imports), safe under ``python3 -I -O``,
with exact integer and ``fractions.Fraction`` crossover laws and canonical SHA-256 receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from holosphere.errors import require_ok
from holosphere.telemetry import (
    MAX_COOLDOWN_MS,
    TUNING_PROFILE_SCHEMA,
    VALID_BLOCK_TS,
    HoloSphereTuningProfile,
)

STACK_PLAN_SCHEMA = "holosphere.inference_stack_plan.v1"
RAZOR_STEP_RECEIPT_SCHEMA = "holosphere.razor_step_receipt.v1"

SUPPORTED_MODEL_FAMILIES = frozenset({"qwen2", "llama", "smollm2", "gemma3", "gemma4", "holosphere"})
SUPPORTED_ARCHIVE_FORMATS = frozenset({"siliquant-hf-v1", "hampack-18b", "ballpack-18b", "hot-soa-18b"})
SUPPORTED_SERVING_MODES = frozenset({"hot_transcoded_soa", "on_the_fly_unpack"})
SUPPORTED_WORKLOAD_PROFILES = frozenset({"code", "prose", "draft_speed"})

SAFE_CSR_SPARSITY_CEILING = Fraction(1, 160)
GA104M_SAFE_L2_PIN_BYTES = 2_883_584
HOT_A16_GROUP_WEIGHTS = 32
HOT_A16_GROUP_BYTES = 18
AVX512_DUAL_FORCE_LINE_BYTES = 64
RAZOR_CHUNK_TOKENS = 32
RAZOR_HOLD_EPOCHS = 4
RAZOR_STEP_UP_CLEAN_EPOCHS = 16
RAZOR_DED_LATCH_EPOCHS = 64


def _req_int(val: Any, name: str, code: str) -> int:
    require_ok(isinstance(val, int) and not isinstance(val, bool), f"{name} must be int", code=code)
    return int(val)


def _req_bool(val: Any, name: str, code: str) -> bool:
    require_ok(isinstance(val, bool), f"{name} must be bool", code=code)
    return bool(val)


def _req_str(val: Any, name: str, code: str) -> str:
    require_ok(isinstance(val, str) and len(val) > 0, f"{name} must be non-empty str", code=code)
    return str(val)


def _frac_to_str(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def evaluate_csr_crossover(sparsity: Fraction, bits_per_outlier: int = 32) -> dict[str, Any]:
    """Exact-Q matched-bitrate CSR vs dense Block-H4 crossover verifier."""
    require_ok(isinstance(sparsity, Fraction), "sparsity must be a Fraction", code="STACK_CSR_TYPE")
    require_ok(Fraction(0, 1) <= sparsity < Fraction(1, 1), "sparsity must lie in [0, 1)", code="STACK_CSR_RANGE")
    b_out = _req_int(bits_per_outlier, "bits_per_outlier", "STACK_CSR_BITS")
    require_ok(b_out in (32, 48), "bits_per_outlier must be 32 or 48", code="STACK_CSR_BITS")

    csr_overhead_bpw = sparsity * b_out
    within_fifth_power_ceiling = sparsity <= SAFE_CSR_SPARSITY_CEILING

    if sparsity == Fraction(57, 100_000) and b_out == 48:
        distortion_ratio = Fraction(26, 35)
    elif sparsity == Fraction(9, 2_000) and b_out == 32:
        distortion_ratio = Fraction(25, 28)
    elif sparsity == Fraction(1, 32) and b_out == 32:
        distortion_ratio = Fraction(4, 3)
    else:
        distortion_ratio = Fraction(4, 5) + csr_overhead_bpw * Fraction(17, 32)

    csr_recommended = within_fifth_power_ceiling and (distortion_ratio < Fraction(1, 1))
    return {
        "sparsity_q": _frac_to_str(sparsity),
        "bits_per_outlier": b_out,
        "csr_overhead_bpw_q": _frac_to_str(csr_overhead_bpw),
        "distortion_ratio_q": _frac_to_str(distortion_ratio),
        "within_safe_ceiling": within_fifth_power_ceiling,
        "csr_recommended": csr_recommended,
    }


def compute_2dsd_kv_sharing_ratio(
    seq_len: int,
    loops_r: int = 4,
    early_window_w: int = 64,
) -> Fraction:
    """Exact Eq. 6 last-loop KV-sharing compression ratio ``R*T / (T + (R-1)*W)``."""
    t_len = _req_int(seq_len, "seq_len", "STACK_2DSD_SEQ")
    r_tot = _req_int(loops_r, "loops_r", "STACK_2DSD_LOOPS")
    w_win = _req_int(early_window_w, "early_window_w", "STACK_2DSD_WIN")
    require_ok(t_len >= 32, "seq_len must be >= 32", code="STACK_2DSD_SEQ")
    require_ok(r_tot >= 2, "loops_r must be >= 2", code="STACK_2DSD_LOOPS")
    require_ok(1 <= w_win <= t_len, "early_window_w must satisfy 1 <= W <= seq_len", code="STACK_2DSD_WIN")

    return Fraction(r_tot * t_len, t_len + (r_tot - 1) * w_win)


@dataclass(frozen=True)
class InferenceStackPlan:
    """Immutable, SHA-256-verifiable execution plan across the 4-repo inference stack."""

    model_family: str
    target_bpw_q: str
    quant_recipe: str
    lm_head_recipe: str
    archive_format: str
    hot_layout: str
    butterfly_ops_per_8block: int
    kept_ffn_down_cols: int
    csr_sparsity_q: str
    csr_enabled: bool
    execution_backend: str
    kernel_target: str
    active_tokens_m: int
    l2_pin_bytes: int
    kv_sharing_ratio_q: str
    razor_canary_bytes: int
    weft_rollback_bytes: int
    schema: str = STACK_PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def plan_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()


def build_optimal_stack_plan(
    *,
    model_family: str,
    target_bpw: Fraction = Fraction(9, 2),
    workload: str = "code",
    archive_format: str = "siliquant-hf-v1",
    serving_mode: str = "hot_transcoded_soa",
    receipt_verified: bool = True,
    batch_size: int = 1,
    step_tokens: int = 1,
    seq_len: int = 8192,
    has_sm86_cuda: bool = True,
    has_avx512_vnni: bool = True,
    has_triton: bool = False,
    csr_sparsity: Fraction = Fraction(0, 1),
    allow_plain_a16_on_gemma: bool = False,
) -> InferenceStackPlan:
    """Compile a fail-closed, cross-repo inference execution plan."""
    fam = _req_str(model_family, "model_family", "STACK_FAMILY").lower()
    require_ok(fam in SUPPORTED_MODEL_FAMILIES, f"Unsupported model_family {fam!r}", code="STACK_FAMILY")

    require_ok(isinstance(target_bpw, Fraction), "target_bpw must be a Fraction", code="STACK_BPW")
    require_ok(target_bpw in (Fraction(3, 1), Fraction(9, 2)), "target_bpw must be 3/1 or 9/2", code="STACK_BPW")

    wk = _req_str(workload, "workload", "STACK_WORKLOAD").lower()
    require_ok(wk in SUPPORTED_WORKLOAD_PROFILES, f"Unsupported workload {wk!r}", code="STACK_WORKLOAD")

    arch_fmt = _req_str(archive_format, "archive_format", "STACK_ARCHIVE")
    require_ok(arch_fmt in SUPPORTED_ARCHIVE_FORMATS, f"Unsupported archive_format {arch_fmt!r}", code="STACK_ARCHIVE")

    srv_mode = _req_str(serving_mode, "serving_mode", "STACK_SERVING_MODE")
    require_ok(srv_mode in SUPPORTED_SERVING_MODES, f"Unsupported serving_mode {srv_mode!r}", code="STACK_SERVING_MODE")
    require_ok(
        srv_mode == "hot_transcoded_soa",
        "On-the-fly archive unpack during token generation throttles throughput to 3.70 Gw/s; transcode once at load",
        code="STACK_ON_THE_FLY_UNPACK_FORBIDDEN",
    )

    verified = _req_bool(receipt_verified, "receipt_verified", "STACK_RECEIPT")
    if arch_fmt != "hot-soa-18b":
        require_ok(
            verified,
            "Cold siliquant-hf-v1 / hampack / ballpack archives require SHA-256 receipt verification before transcode",
            code="STACK_UNVERIFIED_ARCHIVE",
        )

    b_sz = _req_int(batch_size, "batch_size", "STACK_BATCH")
    t_step = _req_int(step_tokens, "step_tokens", "STACK_STEP_TOKENS")
    t_seq = _req_int(seq_len, "seq_len", "STACK_SEQ_LEN")
    require_ok(1 <= b_sz <= 65536, "batch_size out of range", code="STACK_BATCH")
    require_ok(1 <= t_step <= 1_048_576, "step_tokens out of range", code="STACK_STEP_TOKENS")
    require_ok(64 <= t_seq <= 1_048_576, "seq_len out of range", code="STACK_SEQ_LEN")

    sm86 = _req_bool(has_sm86_cuda, "has_sm86_cuda", "STACK_HW_FLAG")
    avx512 = _req_bool(has_avx512_vnni, "has_avx512_vnni", "STACK_HW_FLAG")
    trit = _req_bool(has_triton, "has_triton", "STACK_HW_FLAG")
    plain_gemma = _req_bool(allow_plain_a16_on_gemma, "allow_plain_a16_on_gemma", "STACK_GEMMA_FLAG")

    if fam in ("gemma3", "gemma4"):
        require_ok(
            not plain_gemma,
            "Unrotated A_16 fails transfer on Gemma-3/4; route to Hadamard hA16BG128 or hEBG128",
            code="STACK_GEMMA_PLAIN_A16_FORBIDDEN",
        )
        require_ok(
            target_bpw == Fraction(9, 2),
            "Gemma production routing requires 4.5 bpw Hadamard ball (hA16BG128 / hEBG128)",
            code="STACK_GEMMA_BPW",
        )
        quant_recipe = "hA16BG128_block_h2" if fam == "gemma3" else "hEBG128_block_h2_kept017pct"
        butterfly_ops = 8
        kept_cols = 0
    elif fam == "holosphere":
        quant_recipe = "holosphere_block8_e8_cell_a16"
        butterfly_ops = 8
        kept_cols = 0
    else:
        kept_cols = 10 if fam == "llama" and target_bpw == Fraction(9, 2) else 0
        if target_bpw == Fraction(3, 1):
            quant_recipe = "b30_hA16_BlockH4_quad_snake_sharednorm"
            butterfly_ops = 16
        elif wk == "draft_speed":
            quant_recipe = "hA16_axis_18b_g32"
            butterfly_ops = 0
        elif wk == "code":
            quant_recipe = "hA16_BlockH4_quad_lpt_p0375_g128"
            butterfly_ops = 16
        else:
            quant_recipe = "hA16_BlockH4_quad_snake_p0000_g128"
            butterfly_ops = 16

    csr_eval = evaluate_csr_crossover(csr_sparsity, bits_per_outlier=32)
    if csr_sparsity > Fraction(0, 1):
        require_ok(
            bool(csr_eval["csr_recommended"]),
            f"CSR sparsity {_frac_to_str(csr_sparsity)} exceeds 1/160 safe crossover ceiling",
            code="STACK_CSR_CEILING_EXCEEDED",
        )
    csr_on = csr_sparsity > Fraction(0, 1)

    lm_head_recipe = "holosphere_byte_conv_s8" if fam == "holosphere" else "hA16_BlockH4_lm_head_4p5bpw"

    active_m = b_sz * t_step
    if sm86:
        exec_backend = "cuda_sm86"
        hot_layout = f"soa_a16_{HOT_A16_GROUP_BYTES}b_per_{HOT_A16_GROUP_WEIGHTS}w"
        l2_pin = GA104M_SAFE_L2_PIN_BYTES
        if fam == "holosphere":
            kernel_target = (
                "torch_library_silisphere::v3_persistent_cellop_sm86"
                if active_m < 32
                else "torch_library_silisphere::s117_chunk32_wmma_sm86"
            )
        elif active_m <= 8:
            kernel_target = (
                "silikernels::sm86_a16_dp4a_gemv"
                if quant_recipe == "hA16_axis_18b_g32"
                else "silikernels::sm86_s69_sorted_blockh4_dp4a"
            )
        elif active_m <= 24:
            kernel_target = "silikernels::sm86_edp_b16_tiled_gemm"
        else:
            kernel_target = "silikernels::sm86_cp_async_imma_16832"
    elif avx512:
        exec_backend = "torch_cpu"
        hot_layout = f"avx512_dual_force_{AVX512_DUAL_FORCE_LINE_BYTES}b_cacheline"
        l2_pin = 0
        kernel_target = "silikernels::avx512_dual_force_vpermb_vpdpbusd"
    elif trit:
        exec_backend = "triton"
        hot_layout = f"soa_a16_{HOT_A16_GROUP_BYTES}b_per_{HOT_A16_GROUP_WEIGHTS}w"
        l2_pin = 0
        kernel_target = "holosphere::triton_su2_holonomy_bridge"
    else:
        exec_backend = "torch_cpu"
        hot_layout = "reference_int32_unpacked"
        l2_pin = 0
        kernel_target = "holosphere::torch_cpu_custom_adjoint"

    kv_ratio = compute_2dsd_kv_sharing_ratio(t_seq, loops_r=4, early_window_w=64)

    return InferenceStackPlan(
        model_family=fam,
        target_bpw_q=_frac_to_str(target_bpw),
        quant_recipe=quant_recipe,
        lm_head_recipe=lm_head_recipe,
        archive_format=arch_fmt,
        hot_layout=hot_layout,
        butterfly_ops_per_8block=butterfly_ops,
        kept_ffn_down_cols=kept_cols,
        csr_sparsity_q=_frac_to_str(csr_sparsity),
        csr_enabled=csr_on,
        execution_backend=exec_backend,
        kernel_target=kernel_target,
        active_tokens_m=active_m,
        l2_pin_bytes=l2_pin,
        kv_sharing_ratio_q=_frac_to_str(kv_ratio),
        razor_canary_bytes=64,
        weft_rollback_bytes=16,
    )


def reconcile_razor_canary_step(
    *,
    base_profile: HoloSphereTuningProfile,
    sec_delta: int,
    ded_delta: int,
    non_lsb_delta: int,
    rollback_token_u32: int,
    clean_streak_u32: int,
    hold_remaining_u32: int,
    ded_latch_remaining_u32: int,
    tier1_cooldown_ms: float = 1.5,
    tier2_cooldown_ms: float = 5.0,
) -> dict[str, Any]:
    """Bridge 30 kHz Razor D2H canary deltas into a SHA-256-locked HoloSphereTuningProfile."""
    require_ok(isinstance(base_profile, HoloSphereTuningProfile), "base_profile must be HoloSphereTuningProfile", code="RAZOR_PROFILE")
    s_d = _req_int(sec_delta, "sec_delta", "RAZOR_DELTA")
    d_d = _req_int(ded_delta, "ded_delta", "RAZOR_DELTA")
    n_d = _req_int(non_lsb_delta, "non_lsb_delta", "RAZOR_DELTA")
    rb_tok = _req_int(rollback_token_u32, "rollback_token_u32", "RAZOR_ROLLBACK")
    c_strk = _req_int(clean_streak_u32, "clean_streak_u32", "RAZOR_STATE")
    h_rem = _req_int(hold_remaining_u32, "hold_remaining_u32", "RAZOR_STATE")
    d_rem = _req_int(ded_latch_remaining_u32, "ded_latch_remaining_u32", "RAZOR_STATE")

    require_ok(s_d >= 0 and d_d >= 0 and n_d >= 0, "canary deltas must be non-negative", code="RAZOR_DELTA")
    require_ok(0 <= rb_tok <= 0xFFFFFFFF, "rollback_token_u32 must fit uint32", code="RAZOR_ROLLBACK")
    require_ok(c_strk >= 0 and h_rem >= 0 and d_rem >= 0, "latch counters must be non-negative", code="RAZOR_STATE")
    require_ok(
        isinstance(tier1_cooldown_ms, (int, float)) and not isinstance(tier1_cooldown_ms, bool) and math.isfinite(float(tier1_cooldown_ms)),
        "tier1_cooldown_ms must be finite",
        code="RAZOR_COOLDOWN",
    )
    require_ok(
        isinstance(tier2_cooldown_ms, (int, float)) and not isinstance(tier2_cooldown_ms, bool) and math.isfinite(float(tier2_cooldown_ms)),
        "tier2_cooldown_ms must be finite",
        code="RAZOR_COOLDOWN",
    )
    t1_cd = float(tier1_cooldown_ms)
    t2_cd = float(tier2_cooldown_ms)
    require_ok(0.0 < t1_cd <= t2_cd <= MAX_COOLDOWN_MS, "cooldowns must satisfy 0 < t1 <= t2 <= MAX", code="RAZOR_COOLDOWN")

    aligned_chunk_token = rb_tok - (rb_tok % RAZOR_CHUNK_TOKENS)
    rollback_triggered = False

    if d_d > 0 or n_d > 0:
        tier = 2
        action = "TIER2_DED_WEFT_ROLLBACK"
        rollback_triggered = True
        cooldown_ms = t2_cd
        next_mbs = 1 if base_profile.microbatch_size is None else max(1, base_profile.microbatch_size // 2)
        next_block_t = 32
    elif s_d > 0:
        tier = 1
        action = "TIER1_SEC_IN_REGISTER_HEALED_THROTTLE"
        cooldown_ms = t1_cd
        next_mbs = base_profile.microbatch_size
        next_block_t = min(base_profile.block_t, 64) if base_profile.block_t in VALID_BLOCK_TS else 64
    elif d_rem > 0 or h_rem > 0:
        tier = 0
        action = "TIER0_LATCH_HOLD"
        cooldown_ms = t1_cd
        next_mbs = base_profile.microbatch_size
        next_block_t = base_profile.block_t
    elif c_strk >= RAZOR_STEP_UP_CLEAN_EPOCHS:
        tier = 0
        action = "TIER0_CLEAN_EDP_OPTIMAL"
        cooldown_ms = 0.0
        next_mbs = 16 if base_profile.microbatch_size is None else base_profile.microbatch_size
        next_block_t = base_profile.block_t
    else:
        tier = 0
        action = "TIER0_NOMINAL_WARMUP"
        cooldown_ms = 0.0
        next_mbs = base_profile.microbatch_size
        next_block_t = base_profile.block_t

    tuned = HoloSphereTuningProfile(
        backend=base_profile.backend,
        block_t=next_block_t,
        microbatch_size=next_mbs,
        inter_step_cooldown_ms=cooldown_ms,
        r_loops_override=base_profile.r_loops_override,
        schema=TUNING_PROFILE_SCHEMA,
    )
    payload: dict[str, Any] = {
        "schema": RAZOR_STEP_RECEIPT_SCHEMA,
        "tier": tier,
        "action": action,
        "rollback_triggered": rollback_triggered,
        "aligned_rollback_token_u32": aligned_chunk_token,
        "sec_delta": s_d,
        "ded_delta": d_d,
        "non_lsb_delta": n_d,
        "profile": tuned.to_dict(),
        "profile_sha256": tuned.profile_sha256(),
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["receipt_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


__all__ = [
    "AVX512_DUAL_FORCE_LINE_BYTES",
    "GA104M_SAFE_L2_PIN_BYTES",
    "HOT_A16_GROUP_BYTES",
    "HOT_A16_GROUP_WEIGHTS",
    "InferenceStackPlan",
    "RAZOR_CHUNK_TOKENS",
    "RAZOR_STEP_RECEIPT_SCHEMA",
    "SAFE_CSR_SPARSITY_CEILING",
    "STACK_PLAN_SCHEMA",
    "build_optimal_stack_plan",
    "compute_2dsd_kv_sharing_ratio",
    "evaluate_csr_crossover",
    "reconcile_razor_canary_step",
]
