#!/usr/bin/env python3
"""End-to-End Fused HoloSphere-V3 C++/CUDA Training Engine, Receipt Verifier, & ABI Bridge.

Provides:
  1. Fail-closed verification of ``ptxas_e2e_engine_zero_spill_receipt.json`` (16/16 zero-spill
     Ampere ``sm_86`` kernels across ``holosphere_v3_ext.cu`` and ``holosphere_v3_infer_sm86.cu``)
     and the companion Zone-B RTX 3080 live telemetry & benchmark receipts.
  2. Pure-stdlib ``A_16 Block-H4`` ``[16, 11, 4]`` Extended Hamming packer, 0-POPC dual-SWAR
     hot-path syndrome mirror, inline 1-bit SEC bit-exact repair / 2-bit DED detector, exact
     16-byte ``Weft`` ``HoloSpherePiRollback16`` serializer, and 64-byte ``HoloSphereRazorCanary64``
     wire ABI serializer matching ``include/razor_abi_4byte_sys.h``.
  3. On-demand ephemeral ``sm_86`` compiler loader (compiles to a private temp directory removed
     on exit so zero ``.so``/``.o`` binaries ever enter the repository).
  4. PyTorch ``TORCH_LIBRARY`` custom ops (``holosphere_v3::chunk32_fused_fwd`` and
     ``holosphere_v3::chunk32_fused_bwd``) with ``@torch.library.register_fake``,
     ``torch.library.register_autograd``, ``torch.compile(fullgraph=True)`` zero-graph-break
     compatibility, ``FusedA16BlockH4STECuda``, and zero-materialization
     ``FastHardFMNMXTangentCayleySTE``.

Pure-stdlib-importable (zero ``torch`` or CUDA required for receipt verification & ABI checks).
Zero bare ``assert`` statements. Licensed under Apache-2.0.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from holosphere.errors import HoloSphereError, require_ok

DEFAULT_CSRC_DIR = Path(__file__).resolve().parent / "kernels" / "csrc"
EXT_CU_FILENAME = "holosphere_v3_ext.cu"
INFER_CU_FILENAME = "holosphere_v3_infer_sm86.cu"
PTXAS_RECEIPT_FILENAME = "ptxas_e2e_engine_zero_spill_receipt.json"
ZONE_B_RECEIPT_FILENAME = "mlos_zone_b_s117_e2e_engine.json"
TRAIN_LIVE_FILENAME = "s117_e2e_training_live_results.json"
INFER_LIVE_FILENAME = "s117_e2e_inference_live_results.json"

LIVE_SOURCE_BINDING_SCHEMA = "holosphere.track_s117.live_source_binding.v1"
LIVE_CU_REL_BY_UNIT: Dict[str, str] = {
    "training_ext": "holosphere/kernels/csrc/" + EXT_CU_FILENAME,
    "inference_runtime": "holosphere/kernels/csrc/" + INFER_CU_FILENAME,
}
ENERGY_STATUS_UNCONTROLLED_PROVISIONAL = "uncontrolled_provisional"

EXPECTED_V3_EXT_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN17holosphere_v3_ext35holosphere_v3_chunk32_bwd_wmma_sm86EPKfS1_S1_S1_S1_S1_PfS2_S2_S2_ffffiiiii",
        "_ZN17holosphere_v3_ext40holosphere_v3_a16_blockh4_ste_quant_sm86EPKfPfi",
        "_ZN17holosphere_v3_ext41holosphere_v3_chunk32_fwd_persistent_sm86EPKfPK6float3S1_S1_PK6float4PfS8_S8_S8_PS5_S8_S9_ffffiiiii",
        "_ZN17holosphere_v3_ext46holosphere_v3_su2_3line_gu_bwd_persistent_sm86EPKfS1_S1_PK6float3PK6float4S7_PfPS2_PS5_iiiiii",
    }
)

EXPECTED_V3_INFER_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN19holosphere_v3_infer23v3_infer_vring_mix_sm86EPKfS1_S1_S1_Pfi",
        "_ZN19holosphere_v3_infer24v3_infer_swiglu_act_sm86EPKfS1_Pfi",
        "_ZN19holosphere_v3_infer24v3_infer_vring_step_sm86EPKfS1_S1_S1_PfS2_S2_S2_PKii",
        "_ZN19holosphere_v3_infer26v3_infer_sphere_resid_sm86EPKfS1_S1_Pfi",
        "_ZN19holosphere_v3_infer29v3_infer_sample_gumbel23_sm86EPKfPKjPiPffi",
        "_ZN19holosphere_v3_infer29v3_infer_weft_state_copy_sm86EPfPKfi",
        "_ZN19holosphere_v3_infer30v3_infer_inject_byteconv4_sm86EPKiPKfPfPiS4_i",
        "_ZN19holosphere_v3_infer31v3_infer_cayley_sfast_step_sm86EPKfPK6float3S1_S1_PfP6float4S5_ffffi",
        "_ZN19holosphere_v3_infer33v3_infer_a16_qkc_gemv_secded_sm86EPKfPKmPK6__halfS3_S6_S3_S6_PfS7_S7_P23HoloSphereRazorCanary64ii",
        "_ZN19holosphere_v3_infer34v3_infer_tied_fwht256_lm_head_sm86EPKfPff",
        "_ZN19holosphere_v3_infer36v3_infer_a16_swiglu_gemv_secded_sm86EPKfPKmPK6__halfS3_S6_PfP23HoloSphereRazorCanary64ii",
        "_ZN19holosphere_v3_infer37v3_infer_a16_blockh4_gemv_secded_sm86EPKfPKmPK6__halfPfP23HoloSphereRazorCanary64fii",
    }
)

_SASS_AGG_KEYS: Tuple[str, ...] = (
    "LDL",
    "STL",
    "ATOMG_GPU",
    "MEMBAR_GL",
    "MEMBAR_SYS",
    "RED_SYS",
    "HMMA_16816",
    "SHFL",
    "total_insns",
)

_RAZOR_FIELDS: Tuple[str, ...] = (
    "sec_count_u32",
    "ded_count_u32",
    "epoch_u32",
    "rollback_token_u32",
    "non_lsb_canary_u32",
    "last_syndrome_u32",
    "ewma_ber_q16_u32",
    "clean_streak_u32",
    "hold_remaining_u32",
    "ded_latch_remaining_u32",
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - pure stdlib mode
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ============================================================================
# 1. Strict Type, Numeric, & JSON Validation Helpers (No Bool-as-Int, No NaN)
# ============================================================================


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_nonneg_int(val: Any) -> bool:
    return _is_strict_int(val) and val >= 0


def _is_pos_int(val: Any) -> bool:
    return _is_strict_int(val) and val > 0


def _is_u32(val: Any) -> bool:
    return _is_strict_int(val) and 0 <= val <= 0xFFFFFFFF


def _is_u64(val: Any) -> bool:
    return _is_strict_int(val) and 0 <= val <= 0xFFFFFFFFFFFFFFFF


def _is_finite_real(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    try:
        fv = float(val)
    except OverflowError:
        return False
    return math.isfinite(fv)


def _is_finite_f32(val: Any) -> bool:
    if not _is_finite_real(val):
        return False
    try:
        packed = struct.pack("<f", float(val))
    except OverflowError:
        return False
    return math.isfinite(struct.unpack("<f", packed)[0])


def _coerce_path(val: Any, code: str) -> Path:
    require_ok(
        isinstance(val, (str, os.PathLike)) and not isinstance(val, bytes) and str(val) != "",
        "Path argument must be a non-empty str or PathLike",
        code=code,
    )
    return Path(os.fspath(val))


def _is_hex64(val: Any) -> bool:
    return isinstance(val, str) and len(val) == 64 and all(c in "0123456789abcdef" for c in val)


def _reject_non_finite_json_constant(token: str) -> Any:
    raise HoloSphereError("E2E_NONFINITE_JSON", f"Non-finite JSON constant rejected: {token!r}")


def _parse_finite_json_float(raw: str) -> float:
    val = float(raw)
    if not math.isfinite(val):
        raise HoloSphereError("E2E_NONFINITE_JSON", f"Overflowing JSON float rejected: {raw!r}")
    return val


def _reject_dup_json_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise HoloSphereError("E2E_DUP_JSON_KEY", f"Duplicate JSON key rejected: {k!r}")
        out[k] = v
    return out


def _load_json_object_strict(path: Path, *, missing_code: str, bad_code: str) -> Dict[str, Any]:
    require_ok(path.is_file(), f"Missing JSON artifact: {path.name}", code=missing_code)
    try:
        raw_txt = path.read_text(encoding="utf-8")
        parsed = json.loads(
            raw_txt,
            parse_constant=_reject_non_finite_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_dup_json_pairs,
        )
    except HoloSphereError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise HoloSphereError(bad_code, f"Malformed JSON in {path.name}: {exc}") from exc
    require_ok(isinstance(parsed, dict), f"JSON root must be an object in {path.name}", code=bad_code)
    return parsed


def _f16_round_py(x: float) -> float:
    """Exact IEEE-754 binary16 round-trip via stdlib ``struct``."""
    return float(struct.unpack("<e", struct.pack("<e", float(x)))[0])


def _f32_round_py(x: float) -> float:
    """Exact IEEE-754 binary32 round-trip via stdlib ``struct``."""
    return float(struct.unpack("<f", struct.pack("<f", float(x)))[0])


# ============================================================================
# 2. Fail-Closed ptxas/SASS & Live Zone-B Receipt Verification
# ============================================================================


def verify_e2e_ptxas_receipt(csrc_dir: Any = DEFAULT_CSRC_DIR) -> Dict[str, Any]:
    """Fail-closed SHA-256, 0 B spill, symbol-set, and SASS invariant verification (16/16 kernels)."""
    csrc_path = _coerce_path(csrc_dir, "E2E_MISSING_RECEIPT")
    rec = _load_json_object_strict(
        csrc_path / PTXAS_RECEIPT_FILENAME,
        missing_code="E2E_MISSING_RECEIPT",
        bad_code="E2E_BAD_JSON",
    )
    require_ok(
        rec.get("schema") == "holosphere.track_s117.e2e_engine_ptxas_zero_spill.v1",
        "Unexpected ptxas receipt schema",
        code="E2E_PTXAS_SCHEMA",
    )
    require_ok(rec.get("arch") == "sm_86", "Expected arch sm_86", code="E2E_PTXAS_ARCH")
    require_ok(
        _is_strict_int(rec.get("nonzero_spill_kernels")) and rec["nonzero_spill_kernels"] == 0,
        "nonzero_spill_kernels must be integer 0",
        code="E2E_NONZERO_SPILL",
    )
    n_expected_total = len(EXPECTED_V3_EXT_SYMBOLS) + len(EXPECTED_V3_INFER_SYMBOLS)
    require_ok(
        _is_strict_int(rec.get("total_kernels")) and rec["total_kernels"] == n_expected_total,
        f"total_kernels must equal {n_expected_total}",
        code="E2E_TOTAL_KERNELS",
    )
    units = rec.get("units")
    require_ok(
        isinstance(units, dict) and set(units.keys()) == {"training_ext", "inference_runtime"},
        "units must contain exact keys training_ext and inference_runtime",
        code="E2E_UNITS_SCHEMA",
    )

    summary_units: Dict[str, Any] = {}
    total_kernels = 0
    for uname, cu_name, expected_syms in (
        ("training_ext", EXT_CU_FILENAME, EXPECTED_V3_EXT_SYMBOLS),
        ("inference_runtime", INFER_CU_FILENAME, EXPECTED_V3_INFER_SYMBOLS),
    ):
        u = units[uname]
        require_ok(isinstance(u, dict), f"Unit {uname} must be a JSON object", code="E2E_UNITS_SCHEMA")
        cu_path = csrc_path / cu_name
        require_ok(cu_path.is_file(), f"Missing CUDA source {cu_name}", code="E2E_MISSING_CU")
        cu_sha = hashlib.sha256(cu_path.read_bytes()).hexdigest()
        require_ok(u.get("cu_filename") == cu_name, f"cu_filename mismatch in {uname}", code="E2E_CU_SHA_MISMATCH")
        require_ok(u.get("cu_sha256") == cu_sha, f"cu_sha256 mismatch vs {cu_name}", code="E2E_CU_SHA_MISMATCH")
        require_ok(
            _is_strict_int(u.get("nonzero_spill_kernels")) and u["nonzero_spill_kernels"] == 0,
            f"nonzero_spill_kernels must be 0 in {uname}",
            code="E2E_UNIT_SPILL",
        )
        kernels = u.get("kernels")
        n_exp = len(expected_syms)
        require_ok(
            isinstance(kernels, list)
            and len(kernels) == n_exp
            and _is_strict_int(u.get("total_kernels"))
            and u["total_kernels"] == n_exp,
            f"{uname} expected exact {n_exp} kernels",
            code="E2E_KERNEL_COUNT",
        )

        seen_syms: List[str] = []
        calc_agg: Dict[str, int] = {k: 0 for k in _SASS_AGG_KEYS}
        for k in kernels:
            require_ok(
                isinstance(k, dict)
                and _is_strict_int(k.get("stack_frame_bytes"))
                and k["stack_frame_bytes"] == 0
                and _is_strict_int(k.get("spill_store_bytes"))
                and k["spill_store_bytes"] == 0
                and _is_strict_int(k.get("spill_load_bytes"))
                and k["spill_load_bytes"] == 0
                and _is_pos_int(k.get("registers"))
                and _is_nonneg_int(k.get("smem_bytes"))
                and k.get("zero_spill") is True,
                f"Spill or invalid kernel entry in {uname}",
                code="E2E_KERNEL_SPILL",
            )
            sym = k.get("symbol")
            require_ok(isinstance(sym, str) and sym != "", f"Missing symbol name in {uname}", code="E2E_SYMBOLS")
            seen_syms.append(sym)
            ks = k.get("sass")
            require_ok(isinstance(ks, dict), f"Missing per-kernel sass map in {sym}", code="E2E_SASS_SCHEMA")
            for skey in _SASS_AGG_KEYS:
                sval = ks.get(skey)
                require_ok(_is_nonneg_int(sval), f"Invalid sass count {skey} in {sym}", code="E2E_SASS_SCHEMA")
                calc_agg[skey] += int(sval)
            require_ok(ks["LDL"] == 0 and ks["STL"] == 0, f"LDL/STL forbidden in {sym}", code="E2E_SASS_LOCAL")
            require_ok(
                ks["ATOMG_GPU"] == 0 and ks["MEMBAR_GL"] == 0,
                f"Cross-CTA ATOMG_GPU/MEMBAR_GL forbidden in {sym}",
                code="E2E_SASS_L2BAR",
            )

        require_ok(
            len(set(seen_syms)) == n_exp and frozenset(seen_syms) == expected_syms,
            f"Symbol set mismatch or duplicate row in {uname}",
            code="E2E_SYMBOLS",
        )
        sass = u.get("sass_instruction_counts")
        require_ok(isinstance(sass, dict), f"Missing sass_instruction_counts in {uname}", code="E2E_SASS_SCHEMA")
        for skey in _SASS_AGG_KEYS:
            uval = sass.get(skey)
            require_ok(
                _is_nonneg_int(uval) and uval == calc_agg[skey],
                f"Aggregate sass mismatch for {skey} in {uname}",
                code="E2E_SASS_AGG",
            )
        total_kernels += len(kernels)
        summary_units[uname] = {
            "cu_sha256": cu_sha,
            "kernel_count": len(kernels),
            "HMMA_16816": int(sass["HMMA_16816"]),
            "RED_SYS": int(sass["RED_SYS"]),
            "MEMBAR_SYS": int(sass["MEMBAR_SYS"]),
            "SHFL": int(sass["SHFL"]),
            "LDL": int(sass["LDL"]),
            "STL": int(sass["STL"]),
            "total_insns": int(sass["total_insns"]),
        }

    require_ok(summary_units["training_ext"]["HMMA_16816"] >= 28, "Expected >= 28 HMMA.16816", code="E2E_HMMA_COUNT")
    require_ok(
        summary_units["inference_runtime"]["RED_SYS"] >= 6 and summary_units["inference_runtime"]["MEMBAR_SYS"] >= 9,
        "Expected >= 6 RED_SYS and >= 9 MEMBAR_SYS for 3 SECDED GEMV kernels",
        code="E2E_RED_SYS",
    )
    return {
        "schema": "holosphere.track_s117.e2e_ptxas_verified.v1",
        "total_kernels": total_kernels,
        "zero_spill_kernels": total_kernels,
        "units": summary_units,
    }


def live_results_sha256(obj: dict) -> str:
    """Canonical SHA-256 hex of a live-results JSON object, excluding the ``source_binding`` key."""
    require_ok(isinstance(obj, dict), "live-results object must be a JSON object", code="E2E_LIVE_RESULTS_SHA")
    body = {k: v for k, v in obj.items() if k != "source_binding"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _verify_live_source_binding(
    csrc_path: Path,
    ptxas_units: Any,
    live_obj: Dict[str, Any],
    *,
    ptxas_unit: str,
    live_filename: str,
) -> Dict[str, Any]:
    """Fail-closed source-binding, committed-.cu hash, results self-hash, and energy-status gate."""
    sb = live_obj.get("source_binding")
    require_ok(
        isinstance(sb, dict),
        f"{live_filename}: source_binding missing or not a JSON object",
        code="E2E_LIVE_SRC_BINDING",
    )
    require_ok(
        sb.get("binding_schema") == LIVE_SOURCE_BINDING_SCHEMA
        and sb.get("ptxas_unit") == ptxas_unit
        and sb.get("cu_rel") == LIVE_CU_REL_BY_UNIT[ptxas_unit]
        and _is_hex64(sb.get("cu_sha256"))
        and _is_hex64(sb.get("bench_source_cu_sha256"))
        and _is_hex64(sb.get("results_sha256"))
        and isinstance(sb.get("bench_ran_on_bound_source"), bool)
        and (sb["bench_ran_on_bound_source"] is True) == (sb["bench_source_cu_sha256"] == sb["cu_sha256"]),
        f"{live_filename}: source_binding schema/ptxas_unit/cu_rel/sha-fields/bench-flag invalid",
        code="E2E_LIVE_SRC_BINDING",
    )
    unit = ptxas_units.get(ptxas_unit) if isinstance(ptxas_units, dict) else None
    cu_path = csrc_path / Path(sb["cu_rel"]).name
    require_ok(
        cu_path.is_file()
        and isinstance(unit, dict)
        and _is_hex64(unit.get("cu_sha256"))
        and _is_hex64(unit.get("zone_b_pre_refactor_cu_sha256")),
        f"{live_filename}: bound .cu source missing or ptxas receipt unit malformed",
        code="E2E_LIVE_SRC_SHA",
    )
    actual_cu_sha = hashlib.sha256(cu_path.read_bytes()).hexdigest()
    require_ok(
        actual_cu_sha == sb["cu_sha256"]
        and sb["cu_sha256"] == unit["cu_sha256"]
        and sb["bench_source_cu_sha256"] == unit["zone_b_pre_refactor_cu_sha256"],
        f"{live_filename}: .cu file SHA-256 mismatch vs source_binding / ptxas receipt",
        code="E2E_LIVE_SRC_SHA",
    )
    require_ok(
        live_results_sha256(live_obj) == sb["results_sha256"],
        f"{live_filename}: live results tampered (results_sha256 mismatch)",
        code="E2E_LIVE_RESULTS_SHA",
    )
    em = live_obj.get("energy_measurement")
    require_ok(
        isinstance(em, dict) and em.get("status") == ENERGY_STATUS_UNCONTROLLED_PROVISIONAL,
        f"{live_filename}: energy_measurement.status must be 'uncontrolled_provisional'",
        code="E2E_ENERGY_STATUS",
    )
    return sb


def verify_e2e_live_receipts(csrc_dir: Any = DEFAULT_CSRC_DIR) -> Dict[str, Any]:
    """Fail-closed verification of the stamped Zone-B RTX 3080 training & inference live receipts.

    The Zone-B envelope check verifies only the envelope's integrity (canonical self-hash,
    determinism pins, exit code, and GPU throttle flags); it is NOT a controlled-energy
    certificate. Each live JSON is additionally bound to its committed ``.cu`` source via
    ``source_binding``, self-hashed via ``results_sha256``, and required to declare its
    energy/power figures ``uncontrolled_provisional``.
    """
    csrc_path = _coerce_path(csrc_dir, "E2E_MISSING_ZONE_B")
    zb = _load_json_object_strict(
        csrc_path / ZONE_B_RECEIPT_FILENAME,
        missing_code="E2E_MISSING_ZONE_B",
        bad_code="E2E_BAD_ZONE_B_JSON",
    )
    require_ok(
        zb.get("schema") == "mlos-run-receipt/v0.1"
        and zb.get("determinism_zone") == "B"
        and zb.get("clean_hardware_run") is True
        and zb.get("anomalies") == []
        and _is_hex64(zb.get("receipt_sha256")),
        "Invalid or unclean Zone-B hardware receipt",
        code="E2E_ZONE_B_INVALID",
    )
    zb_body = {k: v for k, v in zb.items() if k != "receipt_sha256"}
    zb_recomputed = hashlib.sha256(
        json.dumps(zb_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    require_ok(
        zb.get("receipt_sha256") == zb_recomputed,
        "Zone-B receipt_sha256 canonical self-hash mismatch",
        code="E2E_ZONE_B_SHA",
    )
    pins = zb.get("determinism_pins")
    require_ok(
        isinstance(pins, dict)
        and all(
            isinstance(pins.get(pk), str) and pins[pk] != ""
            for pk in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "TORCH_USE_DETERMINISTIC_ALGORITHMS")
        ),
        "Zone-B determinism_pins missing required keys",
        code="E2E_ZONE_B_PINS",
    )
    ptxas_rec = _load_json_object_strict(
        csrc_path / PTXAS_RECEIPT_FILENAME,
        missing_code="E2E_MISSING_RECEIPT",
        bad_code="E2E_BAD_JSON",
    )
    require_ok(
        ptxas_rec.get("zone_b_receipt_sha256") == zb["receipt_sha256"],
        "ptxas zone_b_receipt_sha256 binding mismatch vs Zone-B receipt",
        code="E2E_ZONE_B_BINDING",
    )
    exe = zb.get("execution")
    end_gpu = zb.get("end_gpu_telemetry")
    end_clk = end_gpu.get("clocks_event_reasons") if isinstance(end_gpu, dict) else None
    require_ok(
        isinstance(exe, dict)
        and _is_strict_int(exe.get("exit_code"))
        and exe["exit_code"] == 0
        and isinstance(end_clk, dict)
        and end_clk.get("thermal_throttled") is False
        and end_clk.get("slowdown_active") is False,
        "Zone-B execution exit_code or GPU throttle check failed",
        code="E2E_ZONE_B_EXEC",
    )

    tr = _load_json_object_strict(
        csrc_path / TRAIN_LIVE_FILENAME,
        missing_code="E2E_MISSING_TRAIN_LIVE",
        bad_code="E2E_BAD_TRAIN_JSON",
    )
    tr_sb = _verify_live_source_binding(
        csrc_path, ptxas_rec.get("units"), tr, ptxas_unit="training_ext", live_filename=TRAIN_LIVE_FILENAME
    )
    require_ok(
        tr.get("schema") == "holosphere.track_s117.e2e_training_engine_live.v1",
        "Unexpected training live receipt schema",
        code="E2E_TRAIN_SCHEMA",
    )
    fp64 = tr.get("fp64_autograd_parity_T96_3chunks")
    comp = tr.get("torch_compile_fullgraph_inductor")
    benches = tr.get("benchmarks")
    require_ok(
        isinstance(fp64, dict)
        and fp64.get("pass_lt_5e4") is True
        and _is_finite_real(fp64.get("worst_err"))
        and 0.0 <= float(fp64["worst_err"]) < 5e-4
        and isinstance(comp, dict)
        and comp.get("zero_graph_breaks_pass") is True
        and _is_finite_real(comp.get("max_diff_vs_eager"))
        and float(comp["max_diff_vs_eager"]) == 0.0
        and _is_finite_real(tr.get("fused_a16_ste_cuda_max_diff_vs_pytorch"))
        and float(tr["fused_a16_ste_cuda_max_diff_vs_pytorch"]) == 0.0,
        "Training live parity/compile gate failed",
        code="E2E_TRAIN_PARITY",
    )
    core_b = benches.get("core_chunk32_6L_B4_T8192_fwd_bwd") if isinstance(benches, dict) else None
    pref_b = benches.get("row3_15m_B1_T8192_prefill_fused") if isinstance(benches, dict) else None
    trn_b = benches.get("row3_15m_B4_T8192_train_fused_gc") if isinstance(benches, dict) else None
    require_ok(
        isinstance(core_b, dict)
        and _is_finite_real(core_b.get("tok_per_sec_6L_fwd_bwd"))
        and float(core_b["tok_per_sec_6L_fwd_bwd"]) > 0.0
        and isinstance(pref_b, dict)
        and _is_finite_real(pref_b.get("tok_per_sec"))
        and float(pref_b["tok_per_sec"]) > 0.0
        and isinstance(trn_b, dict)
        and _is_finite_real(trn_b.get("tok_per_sec"))
        and float(trn_b["tok_per_sec"]) > 0.0,
        "Training live benchmark numbers missing or non-positive",
        code="E2E_TRAIN_BENCH",
    )

    inf = _load_json_object_strict(
        csrc_path / INFER_LIVE_FILENAME,
        missing_code="E2E_MISSING_INFER_LIVE",
        bad_code="E2E_BAD_INFER_JSON",
    )
    inf_sb = _verify_live_source_binding(
        csrc_path, ptxas_rec.get("units"), inf, ptxas_unit="inference_runtime", live_filename=INFER_LIVE_FILENAME
    )
    require_ok(
        inf.get("schema") == "holosphere.track_s117.e2e_inference_live.v1",
        "Unexpected inference live receipt schema",
        code="E2E_INFER_SCHEMA",
    )
    gpu_name = end_gpu.get("gpu_name") if isinstance(end_gpu, dict) else None
    require_ok(
        isinstance(gpu_name, str)
        and "NVIDIA" in gpu_name
        and tr.get("device") == gpu_name
        and inf.get("device") == gpu_name,
        "Live receipt GPU device identity mismatch or non-NVIDIA device",
        code="E2E_ZONE_B_DEVICE",
    )
    a16 = inf.get("a16_packing")
    par = inf.get("real_weight_parity_32tok")
    sec = inf.get("live_sec16_heal_and_razor_tier1")
    ded = inf.get("live_ded1_weft_pi16_rollback_recovery")
    require_ok(
        isinstance(a16, dict)
        and _is_strict_int(a16.get("total_syndrome_errors"))
        and a16["total_syndrome_errors"] == 0
        and _is_pos_int(a16.get("total_a16_u64_blocks"))
        and _is_pos_int(a16.get("total_linear_layers_packed"))
        and isinstance(par, dict)
        and par.get("exact_token_match_32_of_32") is True
        and _is_strict_int(par.get("clean_canary_sec"))
        and par["clean_canary_sec"] == 0
        and _is_strict_int(par.get("clean_canary_ded"))
        and par["clean_canary_ded"] == 0
        and _is_finite_real(par.get("max_logit_abs_err_vs_pytorch_fp32"))
        and 0.0 <= float(par["max_logit_abs_err_vs_pytorch_fp32"]) < 1e-3,
        "Inference live A_16 packing or real-weight parity check failed",
        code="E2E_INFER_PARITY",
    )
    require_ok(
        isinstance(sec, dict)
        and sec.get("sec16_tokens_bit_exact_32_of_32") is True
        and _is_finite_real(sec.get("sec16_logit_max_abs_diff"))
        and float(sec["sec16_logit_max_abs_diff"]) == 0.0
        and _is_pos_int(sec.get("live_sys_canary_sec_count"))
        and isinstance(ded, dict)
        and _is_pos_int(ded.get("live_sys_canary_ded_count"))
        and ded.get("recovered_tokens_bit_exact_64_of_64") is True
        and _is_finite_real(ded.get("rollback_state_max_abs_diff"))
        and float(ded["rollback_state_max_abs_diff"]) == 0.0
        and _is_strict_int(ded.get("weft_pi16_checkpoint_bytes"))
        and ded["weft_pi16_checkpoint_bytes"] == 16
        and _is_strict_int(ded.get("razor_ded_latch_epochs"))
        and ded["razor_ded_latch_epochs"] == 64,
        "Inference live SEC/DED or Weft 16 B rollback check failed",
        code="E2E_INFER_SECDED",
    )
    return {
        "schema": "holosphere.track_s117.e2e_live_receipts_verified.v1",
        "zone_b_receipt_sha256": zb["receipt_sha256"],
        "core_6L_fwd_bwd_tok_s": float(core_b["tok_per_sec_6L_fwd_bwd"]),
        "row3_15m_prefill_tok_s": float(pref_b["tok_per_sec"]),
        "row3_15m_train_gc_tok_s": float(trn_b["tok_per_sec"]),
        "a16_u64_blocks": int(a16["total_a16_u64_blocks"]),
        "infer_max_logit_abs_err": float(par["max_logit_abs_err_vs_pytorch_fp32"]),
        "sec16_bit_exact": True,
        "ded1_weft_rollback_bit_exact": True,
        "train_live_cu_sha256": tr_sb["cu_sha256"],
        "infer_live_cu_sha256": inf_sb["cu_sha256"],
        "live_bench_ran_on_bound_source": bool(
            tr_sb["bench_ran_on_bound_source"] and inf_sb["bench_ran_on_bound_source"]
        ),
        "live_results_sha256": {
            "training": tr_sb["results_sha256"],
            "inference": inf_sb["results_sha256"],
        },
        "energy_status": ENERGY_STATUS_UNCONTROLLED_PROVISIONAL,
    }


# ============================================================================
# 3. Pure-Stdlib A_16 Block-H4 [16, 11, 4], Dual-SWAR, Weft 16 B, & Razor 64 B
# ============================================================================


def a16_extract_odd16_u64(w64: int) -> int:
    """Extract the 16 nibble LSBs of a packed ``uint64_t`` A_16 Block-H4 codeword."""
    require_ok(_is_u64(w64), "w64 must be an integer in [0, 2^64 - 1]", code="A16_BAD_U64")
    lo = w64 & 0xFFFFFFFF
    hi = (w64 >> 32) & 0xFFFFFFFF

    def _lsbs8(w32: int) -> int:
        m = w32 & 0x11111111
        m = (m | (m >> 3)) & 0x03030303
        m = (m | (m >> 6)) & 0x000F000F
        return (m | (m >> 12)) & 0xFF

    return _lsbs8(lo) | (_lsbs8(hi) << 8)


def a16_secded_syndrome_u16(w16: int) -> int:
    """Exact 5-bit Extended Hamming [16, 11, 4] syndrome matching ``razor_a16_secded_syndrome``."""
    require_ok(_is_strict_int(w16) and 0 <= w16 <= 0xFFFF, "w16 must be uint16", code="A16_BAD_U16")
    p0 = (w16 & 0xFFFF).bit_count() & 1
    p1 = (w16 & 0xAAAA).bit_count() & 1
    p2 = (w16 & 0xCCCC).bit_count() & 1
    p3 = (w16 & 0xF0F0).bit_count() & 1
    p4 = (w16 & 0xFF00).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def a16_blockh4_syndrome_u64(w64: int) -> int:
    """Exact 5-bit Extended Hamming [16, 11, 4] syndrome of a packed ``uint64_t`` codeword."""
    return a16_secded_syndrome_u16(a16_extract_odd16_u64(w64))


def dual_a16_syndrome_nonzero_swar_u32(w32: int) -> int:
    """Exact CPU mirror of ``holosphere_v3_infer::dual_a16_syndrome_nonzero_swar`` (0-POPC hot path)."""
    require_ok(_is_u32(w32), "w32 must be uint32", code="A16_BAD_U32")
    a = (w32 ^ (w32 >> 1)) & 0xFFFFFFFF
    b = (a ^ (a >> 2)) & 0xFFFFFFFF
    c = (b ^ (b >> 4)) & 0xFFFFFFFF
    d = (c ^ (c >> 8)) & 0xFFFFFFFF
    x1 = w32 & 0xAAAAAAAA
    x1 ^= x1 >> 2
    x1 ^= x1 >> 4
    x1 ^= x1 >> 8
    x2 = a & 0x44444444
    x2 ^= x2 >> 4
    x2 ^= x2 >> 8
    p3 = (b ^ (b >> 8)) & 0x00100010
    p4 = c & 0x01000100
    return (d & 0x00010001) | (x1 & 0x00020002) | (x2 & 0x00040004) | p3 | p4


def a16_blockh4_secded_heal_u64(w64: int) -> Tuple[int, int, int]:
    """Inline 1-bit SEC repair & 2-bit DED flag (never miscorrects DED!). Returns ``(w64_out, sec, ded)``."""
    syn = a16_blockh4_syndrome_u64(w64)
    if syn == 0:
        return w64, 0, 0
    if (syn & 1) == 1:
        pos = (syn >> 1) & 15
        return w64 ^ (1 << (pos * 4)), 1, 0
    return w64, 0, 1


def a16_blockh4_dot16_stdlib(w64: int, x16: Sequence[float]) -> float:
    """Exact CPU mirror of ``holosphere_v3_infer::a16_blockh4_dot16`` (H_4 fast butterfly)."""
    require_ok(_is_u64(w64), "w64 must be uint64", code="A16_BAD_U64")
    require_ok(
        isinstance(x16, (list, tuple)) and len(x16) == 16 and all(_is_finite_f32(v) for v in x16),
        "x16 must be a 16-element list/tuple of finite FP32 reals",
        code="A16_BAD_X16",
    )
    acc = 0.0
    for g in range(4):
        nibs = (w64 >> (g * 16)) & 0xFFFF
        c = [((nibs >> (i * 4)) & 0xF) for i in range(4)]
        c = [v - 16 if v >= 8 else v for v in c]
        s01 = c[0] + c[1]
        d01 = c[0] - c[1]
        s23 = c[2] + c[3]
        d23 = c[2] - c[3]
        base = g * 4
        acc += float(s01 + s23) * float(x16[base + 0])
        acc += float(d01 + d23) * float(x16[base + 1])
        acc += float(s01 - s23) * float(x16[base + 2])
        acc += float(d01 - d23) * float(x16[base + 3])
    return acc


def a16_blockh4_pack_row_stdlib(row: Sequence[float]) -> Tuple[List[int], List[float], List[float]]:
    """Pure-stdlib exact ``A_16 Block-H4`` ``[16, 11, 4]`` packer (len(row) > 0 and divisible by 32)."""
    require_ok(
        isinstance(row, (list, tuple))
        and len(row) > 0
        and len(row) % 32 == 0
        and all(_is_finite_f32(v) for v in row),
        "row must be a non-empty list/tuple of finite FP32 reals with len divisible by 32",
        code="A16_BAD_ROW",
    )
    n = len(row)
    packed_u64: List[int] = []
    scales_f16: List[float] = []
    dequant: List[float] = []

    parity_masks = (
        (1, (3, 5, 7, 9, 11, 13, 15)),
        (2, (3, 6, 7, 10, 11, 14, 15)),
        (4, (5, 6, 7, 12, 13, 14, 15)),
        (8, (9, 10, 11, 12, 13, 14, 15)),
    )

    for g32 in range(0, n, 32):
        blk = [float(row[g32 + i]) for i in range(32)]
        # H_4 transform on 8 groups of 4: u_raw = 0.5 * blk @ H_4
        u_raw: List[float] = []
        for g4 in range(0, 32, 4):
            a0, a1, a2, a3 = blk[g4], blk[g4 + 1], blk[g4 + 2], blk[g4 + 3]
            u_raw.extend(
                (
                    0.5 * (a0 + a1 + a2 + a3),
                    0.5 * (a0 - a1 + a2 - a3),
                    0.5 * (a0 + a1 - a2 - a3),
                    0.5 * (a0 - a1 - a2 + a3),
                )
            )
        amax_raw = max(abs(v) for v in u_raw)
        require_ok(
            math.isfinite(amax_raw) and amax_raw <= A16_MAX_URAW_ABS,
            f"A_16 Block-H4 scale overflow: max|u_raw|={amax_raw} exceeds FP16 scale bound {A16_MAX_URAW_ABS}",
            code="A16_OVERFLOW",
        )
        amax = max(amax_raw, 1e-6)
        sc = _f16_round_py(amax / 7.5)
        scales_f16.append(sc)

        for half in range(2):
            u16 = [min(max(u_raw[half * 16 + j] / sc, -8.0), 7.0) for j in range(16)]
            c16 = [max(-8, min(7, int(round(v)))) for v in u16]

            def _snap(pos: int, req_lsb: int) -> None:
                cur = c16[pos]
                if (cur & 1) != req_lsb:
                    step = 1 if u16[pos] >= float(cur) else -1
                    if cur == 7 and step > 0:
                        step = -1
                    elif cur == -8 and step < 0:
                        step = 1
                    c16[pos] = cur + step

            for p_pos, idxs in parity_masks:
                req = 0
                for j in idxs:
                    req ^= c16[j] & 1
                _snap(p_pos, req)
            req0 = 0
            for j in range(1, 16):
                req0 ^= c16[j] & 1
            _snap(0, req0)

            w64 = 0
            for j in range(16):
                w64 |= (c16[j] & 0xF) << (j * 4)
            packed_u64.append(w64)

            for g4 in range(0, 16, 4):
                c0, c1, c2, c3 = float(c16[g4]), float(c16[g4 + 1]), float(c16[g4 + 2]), float(c16[g4 + 3])
                dequant.extend(
                    (
                        0.5 * sc * (c0 + c1 + c2 + c3),
                        0.5 * sc * (c0 - c1 + c2 - c3),
                        0.5 * sc * (c0 + c1 - c2 - c3),
                        0.5 * sc * (c0 - c1 - c2 + c3),
                    )
                )

    return packed_u64, scales_f16, dequant


WEFT_PI16_UNIT_TOL: float = 2.0**-10  # Exact IEEE-754 dyadic boundary (0.0009765625)
A16_MAX_URAW_ABS: float = 65504.0 * 7.5  # Exact FP16 scale representability ceiling (491280.0)


def check_cuda_rc(rc: int, code: str, detail: str = "CUDA kernel") -> None:
    """Fail-closed CUDA / kernel launch return-code gate."""
    require_ok(_is_strict_int(rc) and rc == 0, f"{detail} failed (rc={rc})", code=code)


def pack_weft_pi16(w: float, x: float, y: float, z: float) -> bytes:
    """Serialize unit quaternion ``Pi_{t_0} in SU(2)`` into exact 16-byte ``HoloSpherePiRollback16``."""
    require_ok(
        _is_finite_f32(w) and _is_finite_f32(x) and _is_finite_f32(y) and _is_finite_f32(z),
        "Pi_{t_0} quaternion components must be finite FP32 reals",
        code="WEFT_PI16_NONFINITE",
    )
    fw, fx, fy, fz = _f32_round_py(w), _f32_round_py(x), _f32_round_py(y), _f32_round_py(z)
    n2 = fw * fw + fx * fx + fy * fy + fz * fz
    require_ok(abs(n2 - 1.0) <= WEFT_PI16_UNIT_TOL, f"Pi_{{t_0}} must lie on S^3 (norm^2={n2})", code="WEFT_PI16_NONUNIT")
    return struct.pack("<4f", fw, fx, fy, fz)


def unpack_weft_pi16(raw: bytes) -> Tuple[float, float, float, float]:
    """Deserialize and validate exact 16-byte ``HoloSpherePiRollback16``."""
    require_ok(isinstance(raw, (bytes, bytearray)) and len(raw) == 16, "Pi_{t_0} wire record must be 16 bytes", code="WEFT_PI16_LEN")
    fw, fx, fy, fz = struct.unpack("<4f", bytes(raw))
    require_ok(
        math.isfinite(fw) and math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz),
        "Unpacked Pi_{t_0} contains non-finite float",
        code="WEFT_PI16_NONFINITE",
    )
    n2 = fw * fw + fx * fx + fy * fy + fz * fz
    require_ok(abs(n2 - 1.0) <= WEFT_PI16_UNIT_TOL, f"Unpacked Pi_{{t_0}} non-unit (norm^2={n2})", code="WEFT_PI16_NONUNIT")
    return fw, fx, fy, fz


def pack_razor_canary64(fields: Dict[str, int]) -> bytes:
    """Serialize 64-byte cacheline-aligned ``HoloSphereRazorCanary64`` matching ``include/razor_abi_4byte_sys.h``."""
    require_ok(isinstance(fields, dict), "Razor canary fields must be a dict", code="RAZOR64_BAD_FIELDS")
    require_ok(
        set(fields.keys()) == set(_RAZOR_FIELDS),
        "Razor canary fields must contain the exact 10 ABI field keys (no unknown or missing keys)",
        code="RAZOR64_BAD_KEYS",
    )
    words: List[int] = []
    for name in _RAZOR_FIELDS:
        v = fields[name]
        require_ok(_is_u32(v), f"Razor field {name} must be uint32", code="RAZOR64_BAD_U32")
        words.append(int(v))
    words.extend((0, 0, 0, 0, 0, 0))
    return struct.pack("<16I", *words)


def unpack_razor_canary64(raw: bytes) -> Dict[str, int]:
    """Deserialize 64-byte ``HoloSphereRazorCanary64`` wire record and enforce zero reserved padding."""
    require_ok(isinstance(raw, (bytes, bytearray)) and len(raw) == 64, "Razor canary record must be 64 bytes", code="RAZOR64_LEN")
    words = struct.unpack("<16I", bytes(raw))
    require_ok(all(w == 0 for w in words[10:]), "Razor reserved_u32[6] padding must be zero", code="RAZOR64_RESERVED")
    return {name: int(words[idx]) for idx, name in enumerate(_RAZOR_FIELDS)}


def validate_e2e_stdlib_self_test_report(rep: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-closed invariant gate for ``run_e2e_stdlib_self_test`` output."""
    require_ok(isinstance(rep, dict), "stdlib self-test report must be a dict", code="E2E_ST_SCHEMA")
    require_ok(
        _is_strict_int(rep.get("a16_clean_syndrome_errors")) and rep["a16_clean_syndrome_errors"] == 0,
        "Expected 0 clean A_16 syndrome errors",
        code="A16_CLEAN_SYN",
    )
    require_ok(
        _is_finite_real(rep.get("a16_dot_parity_err")) and 0.0 <= float(rep["a16_dot_parity_err"]) < 1e-12,
        "A_16 dot parity error exceeded 1e-12",
        code="A16_DOT_PARITY",
    )
    require_ok(
        rep.get("a16_swar_clean_zero") is True,
        "Dual-SWAR false positive on clean codewords",
        code="A16_SWAR_CLEAN",
    )
    require_ok(
        _is_strict_int(rep.get("a16_sec_healed")) and rep["a16_sec_healed"] == 16,
        "Expected 16/16 SEC bit-exact heals",
        code="A16_SEC_FAIL",
    )
    require_ok(
        _is_strict_int(rep.get("a16_ded_caught")) and rep["a16_ded_caught"] == 120,
        "Expected 120/120 DED detections without miscorrection",
        code="A16_DED_FAIL",
    )
    require_ok(
        _is_strict_int(rep.get("weft_pi16_bytes"))
        and rep["weft_pi16_bytes"] == 16
        and _is_finite_real(rep.get("weft_pi16_err"))
        and float(rep["weft_pi16_err"]) < 1e-6,
        "Weft Pi_{t_0} 16 B round-trip drift",
        code="WEFT_PI16_DRIFT",
    )
    require_ok(
        _is_strict_int(rep.get("razor_canary64_bytes"))
        and rep["razor_canary64_bytes"] == 64
        and rep.get("razor_canary64_roundtrip_ok") is True,
        "Razor 64 B ABI round-trip mismatch",
        code="RAZOR64_DRIFT",
    )
    return rep


def run_e2e_stdlib_self_test() -> Dict[str, Any]:
    """Pure-stdlib verification of A_16 packing, dual-SWAR exactness, 16/16 SEC, 120/120 DED, & 16 B/64 B ABI."""
    row = [math.sin(0.37 * c + 0.19) * 0.45 for c in range(64)]
    x_vec = [math.cos(0.23 * c - 0.11) for c in range(64)]
    packed_u64, scales, deq = a16_blockh4_pack_row_stdlib(row)

    clean_syn = sum(1 for blk in packed_u64 if a16_blockh4_syndrome_u64(blk) != 0)

    dot_kern = 0.0
    for g32 in range(2):
        d32 = a16_blockh4_dot16_stdlib(packed_u64[2 * g32], x_vec[g32 * 32 : g32 * 32 + 16]) + a16_blockh4_dot16_stdlib(
            packed_u64[2 * g32 + 1], x_vec[g32 * 32 + 16 : g32 * 32 + 32]
        )
        dot_kern += 0.5 * scales[g32] * d32
    dot_ref = sum(a * b for a, b in zip(deq, x_vec))
    dot_err = abs(dot_kern - dot_ref)

    blk0 = packed_u64[0]
    blk1 = packed_u64[1]
    odd1 = a16_extract_odd16_u64(blk1)
    swar_clean_zero = dual_a16_syndrome_nonzero_swar_u32(a16_extract_odd16_u64(blk0) | (odd1 << 16)) == 0

    sec_healed = 0
    for bit in range(16):
        corrupted = blk0 ^ (1 << (bit * 4))
        sw = dual_a16_syndrome_nonzero_swar_u32(a16_extract_odd16_u64(corrupted) | (odd1 << 16))
        healed, sec_f, ded_f = a16_blockh4_secded_heal_u64(corrupted)
        if sw != 0 and sec_f == 1 and ded_f == 0 and healed == blk0:
            sec_healed += 1

    ded_caught = 0
    for i in range(16):
        for j in range(i + 1, 16):
            corrupted = blk0 ^ (1 << (i * 4)) ^ (1 << (j * 4))
            sw = dual_a16_syndrome_nonzero_swar_u32(a16_extract_odd16_u64(corrupted) | (odd1 << 16))
            healed, sec_f, ded_f = a16_blockh4_secded_heal_u64(corrupted)
            if sw != 0 and sec_f == 0 and ded_f == 1 and healed == corrupted:
                ded_caught += 1

    pi_raw = pack_weft_pi16(0.6, 0.8, 0.0, 0.0)
    pi_out = unpack_weft_pi16(pi_raw)
    pi_err = max(abs(a - b) for a, b in zip((0.6, 0.8, 0.0, 0.0), pi_out))

    razor_in = {
        "sec_count_u32": 512,
        "ded_count_u32": 1,
        "epoch_u32": 4,
        "rollback_token_u32": 32,
        "non_lsb_canary_u32": 0,
        "last_syndrome_u32": 0x26,
        "ewma_ber_q16_u32": 128,
        "clean_streak_u32": 0,
        "hold_remaining_u32": 4,
        "ded_latch_remaining_u32": 64,
    }
    razor_raw = pack_razor_canary64(razor_in)
    razor_out = unpack_razor_canary64(razor_raw)

    report = {
        "schema": "holosphere.track_s117.e2e_stdlib_self_test.v1",
        "a16_clean_syndrome_errors": clean_syn,
        "a16_dot_parity_err": dot_err,
        "a16_swar_clean_zero": swar_clean_zero,
        "a16_sec_healed": sec_healed,
        "a16_ded_caught": ded_caught,
        "weft_pi16_bytes": len(pi_raw),
        "weft_pi16_err": pi_err,
        "weft_pi16_sha256": hashlib.sha256(pi_raw).hexdigest(),
        "razor_canary64_bytes": len(razor_raw),
        "razor_canary64_roundtrip_ok": (razor_out == razor_in),
        "razor_canary64_sha256": hashlib.sha256(razor_raw).hexdigest(),
    }
    return validate_e2e_stdlib_self_test_report(report)


# ============================================================================
# 4. Ephemeral On-Demand sm_86 Compiler Loader (ELF = 0 in Repository)
# ============================================================================

_EXT_CACHE: Dict[str, Tuple[ctypes.CDLL, str]] = {}


def _find_nvcc() -> str:
    for cand in ("/opt/cuda/bin/nvcc", shutil.which("nvcc") or ""):
        if cand and Path(cand).is_file():
            return cand
    raise HoloSphereError("NVCC_MISSING", "nvcc compiler not found on PATH or /opt/cuda/bin/nvcc")


def _configure_ext_signatures(lib: ctypes.CDLL, filename: str) -> None:
    if filename != EXT_CU_FILENAME:
        return
    lib.launch_holosphere_v3_chunk32_fwd_persistent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_holosphere_v3_chunk32_fwd_persistent.restype = ctypes.c_int

    lib.launch_holosphere_v3_chunk32_bwd_wmma.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_holosphere_v3_chunk32_bwd_wmma.restype = ctypes.c_int

    lib.launch_holosphere_v3_su2_3line_gu_bwd_persistent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_holosphere_v3_su2_3line_gu_bwd_persistent.restype = ctypes.c_int

    lib.launch_holosphere_v3_a16_blockh4_ste_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_holosphere_v3_a16_blockh4_ste_quant.restype = ctypes.c_int


def compile_and_load_cuda_lib(
    cu_path: Any,
    arch: str = "sm_86",
    *,
    nvcc_bin: Optional[str] = None,
    timeout: float = 120.0,
) -> ctypes.CDLL:
    """Compile a ``.cu`` file into a private temp directory (auto-removed on failure or atexit -> ELF=0)."""
    cu_p = _coerce_path(cu_path, "CU_NOT_FOUND")
    require_ok(cu_p.is_file(), f"Missing CUDA source {cu_p}", code="CU_NOT_FOUND")
    require_ok(_is_finite_real(timeout) and float(timeout) > 0.0, "timeout must be positive finite seconds", code="NVCC_BAD_TIMEOUT")
    nvcc = nvcc_bin if nvcc_bin is not None else _find_nvcc()
    key = f"{cu_p.resolve()}::{arch}::{nvcc}"
    if key in _EXT_CACHE:
        return _EXT_CACHE[key][0]
    tmp_dir = tempfile.mkdtemp(prefix="holosphere_v3_e2e_")
    so_path = Path(tmp_dir) / f"{cu_p.stem}.so"
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "-Xcompiler",
        "-fPIC,-Wall,-Wextra,-Werror",
        "-shared",
        "-o",
        str(so_path),
        str(cu_p),
    ]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=float(timeout))
        except (OSError, subprocess.SubprocessError) as exc:
            raise HoloSphereError("NVCC_COMPILE_FAIL", f"nvcc launch failed: {exc}") from exc
        if proc.returncode != 0:
            raise HoloSphereError("NVCC_COMPILE_FAIL", f"nvcc failed ({proc.returncode}): {proc.stderr[-800:]}")
        require_ok(so_path.is_file(), f"nvcc exited 0 without producing {so_path.name}", code="NVCC_LOAD_FAIL")
        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError as exc:
            raise HoloSphereError("NVCC_LOAD_FAIL", f"Failed to load compiled library {so_path.name}: {exc}") from exc
        _configure_ext_signatures(lib, cu_p.name)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    atexit.register(shutil.rmtree, tmp_dir, True)
    _EXT_CACHE[key] = (lib, tmp_dir)
    return lib


def load_v3_training_ext(csrc_dir: Path = DEFAULT_CSRC_DIR) -> ctypes.CDLL:
    return compile_and_load_cuda_lib(csrc_dir / EXT_CU_FILENAME)


# ============================================================================
# 5. PyTorch TORCH_LIBRARY Custom Ops (holosphere_v3::chunk32_fused_fwd/bwd)
#    + torch.compile(fullgraph=True) Zero-Graph-Break Integration
# ============================================================================


def _require_torch() -> None:
    require_ok(_HAS_TORCH, "PyTorch is required for holosphere_v3 custom ops", code="NO_TORCH")


if _HAS_TORCH:

    def _tq_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3],
                a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] + a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2],
                a[..., 0] * b[..., 2] - a[..., 1] * b[..., 3] + a[..., 2] * b[..., 0] + a[..., 3] * b[..., 1],
                a[..., 0] * b[..., 3] + a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1] + a[..., 3] * b[..., 0],
            ),
            dim=-1,
        )

    def _tq_conj(q: torch.Tensor) -> torch.Tensor:
        return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

    def _tq_cayley(w: torch.Tensor) -> torch.Tensor:
        w2 = (w * w).sum(dim=-1, keepdim=True)
        inv = 1.0 / (1.0 + w2)
        return torch.cat(((1.0 - w2) * inv, 2.0 * w * inv), dim=-1)

    def _tq_unit_norm(q: torch.Tensor) -> torch.Tensor:
        return q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-24)

    def _tq_cayley_pullback(w: torch.Tensor, gq: torch.Tensor) -> torch.Tensor:
        w2 = (w * w).sum(dim=-1, keepdim=True)
        inv = 1.0 / (1.0 + w2)
        dot_wv = (w * gq[..., 1:]).sum(dim=-1, keepdim=True)
        coeff = -4.0 * inv * inv * (gq[..., :1] + dot_wv)
        return 2.0 * inv * gq[..., 1:] + coeff * w

    def _cpu_ref_chunk32_bwd(
        x_seq: torch.Tensor,
        xt_out: torch.Tensor,
        zs_out: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        s_ckpts: torch.Tensor,
        u_ends: torch.Tensor,
        grad_y: torch.Tensor,
        grad_s_end: torch.Tensor,
        grad_u_end: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        project_boundary_tangent: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact closed-form reverse-mode Chunk32 + SU(2) adjoint (zero nested autograd)."""
        bsz, seq_len, d_model = x_seq.shape
        n_heads = d_model // 32
        n_chunks = seq_len // 32
        t2 = cayley_t * cayley_t
        c_st = (1.0 - t2) / (1.0 + t2)
        s_st = (2.0 * cayley_t) / (1.0 + t2)

        gx_all = torch.empty_like(x_seq)
        gw_all = torch.empty_like(w_su2)
        grad_wq = torch.zeros_like(W_q)
        gS = grad_s_end.clone()
        gU = grad_u_end.clone()

        for ch in range(n_chunks - 1, -1, -1):
            t0 = ch * 32
            # Recompute intra-chunk S_prev, sp, v, v_inv, err from O(T/32) checkpoint s_ckpts[ch]
            S_cur = s_ckpts[ch]
            s_prev_buf: List[torch.Tensor] = []
            sp_buf: List[torch.Tensor] = []
            z_buf: List[torch.Tensor] = []
            dot_zx_buf: List[torch.Tensor] = []
            v_inv_buf: List[torch.Tensor] = []
            uhat_buf: List[torch.Tensor] = []
            err_buf: List[torch.Tensor] = []
            for lt in range(32):
                t = t0 + lt
                x_t = x_seq[:, t].view(bsz, n_heads, 32)
                xt_t = xt_out[:, t].view(bsz, n_heads, 32)
                zs_t = zs_out[:, t].view(bsz, n_heads, 32)
                sp_t = torch.einsum("bhrc,bhc->bhr", S_cur, xt_t)
                z_t = zs_t + sp_t
                dot_zx = (z_t * x_t).sum(dim=-1, keepdim=True)
                v_t = z_t - dot_zx * x_t
                v_inv = torch.rsqrt((v_t * v_t).sum(dim=-1, keepdim=True) + eps)
                u_hat = v_t * v_inv
                y_t = c_st * x_t + s_st * u_hat
                err_t = y_t - sp_t
                s_prev_buf.append(S_cur)
                sp_buf.append(sp_t)
                z_buf.append(z_t)
                dot_zx_buf.append(dot_zx)
                v_inv_buf.append(v_inv)
                uhat_buf.append(u_hat)
                err_buf.append(err_t)
                S_cur = gamma * S_cur + eta * err_t.unsqueeze(-1) * xt_t.unsqueeze(-2)

            U_cur = u_ends[ch]
            for lt in range(31, -1, -1):
                t = t0 + lt
                x_t = x_seq[:, t].view(bsz, n_heads, 32)
                xt_t = xt_out[:, t].view(bsz, n_heads, 32)
                w_t = w_su2[:, t]
                gy_t = grad_y[:, t].view(bsz, n_heads, 32)

                g_err = eta * torch.einsum("bhrc,bhc->bhr", gS, xt_t)
                g_xt_S = eta * torch.einsum("bhrc,bhr->bhc", gS, err_buf[lt])
                gS_prev = gamma * gS

                g_y_tot = gy_t + g_err
                gx_geo = c_st * g_y_tot
                g_uhat = s_st * g_y_tot
                u_hat = uhat_buf[lt]
                v_inv = v_inv_buf[lt]
                g_v = (g_uhat - (g_uhat * u_hat).sum(dim=-1, keepdim=True) * u_hat) * v_inv

                gv_dot_x = (g_v * x_t).sum(dim=-1, keepdim=True)
                g_z = g_v - gv_dot_x * x_t
                gx_geo = gx_geo - dot_zx_buf[lt] * g_v - gv_dot_x * z_buf[lt]

                g_sp = g_z - g_err
                gS = gS_prev + g_sp.unsqueeze(-1) * xt_t.unsqueeze(-2)
                g_xt_sp = torch.einsum("bhrc,bhr->bhc", s_prev_buf[lt], g_sp)
                grad_wq = grad_wq + torch.einsum("bhe,bhd->hed", g_z, xt_t)
                g_xt_wq = torch.einsum("bhe,hed->bhd", g_z, W_q)
                g_xt = g_xt_S + g_xt_sp + g_xt_wq

                x_q = x_t.view(bsz, n_heads, 8, 4)
                gxt_q = g_xt.view(bsz, n_heads, 8, 4)
                gx_t = gx_geo + _tq_mul(U_cur, gxt_q).view(bsz, n_heads, 32)
                gx_all[:, t] = gx_t.reshape(bsz, d_model)

                gU = gU + _tq_mul(x_q, _tq_conj(gxt_q))
                q_t = _tq_cayley(w_t)
                U_prev = _tq_mul(_tq_conj(q_t), U_cur)
                g_q = _tq_mul(gU, _tq_conj(U_prev))
                gw_all[:, t] = _tq_cayley_pullback(w_t, g_q)
                gU = _tq_mul(_tq_conj(q_t), gU)
                U_cur = U_prev

            if project_boundary_tangent:
                radial = (gU * U_cur).sum(dim=-1, keepdim=True)
                gU = gU - radial * U_cur

        return gx_all, gw_all, grad_wq, gS, gU

    def _cpu_ref_chunk32_fwd(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x_seq.shape
        n_heads = d_model // 32
        n_chunks = seq_len // 32
        t2 = cayley_t * cayley_t
        c_st = (1.0 - t2) / (1.0 + t2)
        s_st = (2.0 * cayley_t) / (1.0 + t2)

        S = S_0.clone()
        U = U_0.clone()
        ys: List[torch.Tensor] = []
        xts: List[torch.Tensor] = []
        zss: List[torch.Tensor] = []
        s_ckpts: List[torch.Tensor] = []
        u_ends: List[torch.Tensor] = []

        for ch in range(n_chunks):
            s_ckpts.append(S.clone())
            for lt in range(32):
                t = ch * 32 + lt
                x_t = x_seq[:, t].view(bsz, n_heads, 32)
                w_t = w_su2[:, t]
                U = _tq_mul(_tq_cayley(w_t), U)
                xt_t = _tq_mul(_tq_conj(U), x_t.view(bsz, n_heads, 8, 4)).view(bsz, n_heads, 32)
                zs_t = torch.einsum("bhd,hed->bhe", xt_t, W_q)
                sp_t = torch.einsum("bhrc,bhc->bhr", S, xt_t)
                z_t = zs_t + sp_t
                v_t = z_t - (z_t * x_t).sum(dim=-1, keepdim=True) * x_t
                v_inv = torch.rsqrt((v_t * v_t).sum(dim=-1, keepdim=True) + eps)
                y_t = c_st * x_t + s_st * (v_t * v_inv)
                err_t = y_t - sp_t
                S = gamma * S + eta * err_t.unsqueeze(-1) * xt_t.unsqueeze(-2)
                ys.append(y_t.reshape(bsz, d_model))
                xts.append(xt_t.reshape(bsz, d_model))
                zss.append(zs_t.reshape(bsz, d_model))
            u_ends.append(U.clone())

        return (
            torch.stack(ys, dim=1),
            torch.stack(xts, dim=1),
            torch.stack(zss, dim=1),
            torch.stack(s_ckpts, dim=0),
            torch.stack(u_ends, dim=0),
            S,
            U,
        )

    @torch.library.custom_op("holosphere_v3::chunk32_fused_fwd", mutates_args=())
    def _op_v3_chunk32_fused_fwd(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        project_boundary_tangent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del project_boundary_tangent
        require_ok(
            bool(torch.isfinite(x_seq).all().item())
            and bool(torch.isfinite(w_su2).all().item())
            and bool(torch.isfinite(W_q).all().item())
            and bool(torch.isfinite(S_0).all().item())
            and bool(torch.isfinite(U_0).all().item()),
            "Chunk32 fused inputs must be finite (no NaN/Inf)",
            code="V3_NONFINITE",
        )
        u0_n2 = (U_0.float() * U_0.float()).sum(dim=-1)
        require_ok(
            bool(((u0_n2 - 1.0).abs() <= WEFT_PI16_UNIT_TOL).all().item()),
            "U_0 quaternions must lie on unit sphere S^3 within WEFT_PI16_UNIT_TOL",
            code="V3_U0_NONUNIT",
        )
        bsz, seq_len, d_model = x_seq.shape
        n_heads = d_model // 32
        n_chunks = seq_len // 32

        if not x_seq.is_cuda:
            return _cpu_ref_chunk32_fwd(x_seq, w_su2, W_q, S_0, U_0, cayley_t, gamma, eta, eps)

        with torch.amp.autocast("cuda", enabled=False):
            x_c = x_seq.float().contiguous()
            w_c = w_su2.float().contiguous()
            wq_c = W_q.float().contiguous()
            s0_c = S_0.float().contiguous()
            u0_c = U_0.float().contiguous()
            dev = x_c.device
            bh = bsz * n_heads

            y_out = torch.empty_like(x_c)
            xt_out = torch.empty_like(x_c)
            zs_out = torch.empty_like(x_c)
            s_ckpts = torch.empty((n_chunks, bsz, n_heads, 32, 32), dtype=torch.float32, device=dev)
            u_ends = torch.empty((n_chunks, bsz, n_heads, 8, 4), dtype=torch.float32, device=dev)
            s_end = torch.empty_like(s0_c)
            u_end = torch.empty_like(u0_c)

            lib = load_v3_training_ext()
            stream_ptr = ctypes.c_void_p(int(torch.cuda.current_stream(dev).cuda_stream))
            rc = lib.launch_holosphere_v3_chunk32_fwd_persistent(
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(w_c.data_ptr()),
                ctypes.c_void_p(wq_c.data_ptr()),
                ctypes.c_void_p(s0_c.data_ptr()),
                ctypes.c_void_p(u0_c.data_ptr()),
                ctypes.c_void_p(y_out.data_ptr()),
                ctypes.c_void_p(xt_out.data_ptr()),
                ctypes.c_void_p(zs_out.data_ptr()),
                ctypes.c_void_p(s_ckpts.data_ptr()),
                ctypes.c_void_p(u_ends.data_ptr()),
                ctypes.c_void_p(s_end.data_ptr()),
                ctypes.c_void_p(u_end.data_ptr()),
                float(cayley_t),
                float(gamma),
                float(eta),
                float(eps),
                int(n_heads),
                int(seq_len * d_model),
                int(d_model),
                int(n_chunks),
                int(bh),
                stream_ptr,
            )
            check_cuda_rc(rc, "V3_CUDA_FWD_ERR", "fwd_persistent")
            return y_out, xt_out, zs_out, s_ckpts, u_ends, s_end, u_end

    @torch.library.register_fake("holosphere_v3::chunk32_fused_fwd")
    def _op_v3_chunk32_fused_fwd_fake(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        project_boundary_tangent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del w_su2, W_q, cayley_t, gamma, eta, eps, project_boundary_tangent
        _bsz, seq_len, _d_model = x_seq.shape
        n_chunks = seq_len // 32
        out_dtype = torch.float32 if x_seq.is_cuda else x_seq.dtype
        return (
            x_seq.new_empty(x_seq.shape, dtype=out_dtype),
            x_seq.new_empty(x_seq.shape, dtype=out_dtype),
            x_seq.new_empty(x_seq.shape, dtype=out_dtype),
            x_seq.new_empty((n_chunks, *S_0.shape), dtype=out_dtype),
            x_seq.new_empty((n_chunks, *U_0.shape), dtype=out_dtype),
            S_0.new_empty(S_0.shape, dtype=out_dtype),
            U_0.new_empty(U_0.shape, dtype=out_dtype),
        )

    @torch.library.custom_op("holosphere_v3::chunk32_fused_bwd", mutates_args=())
    def _op_v3_chunk32_fused_bwd(
        x_seq: torch.Tensor,
        xt_out: torch.Tensor,
        zs_out: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        s_ckpts: torch.Tensor,
        u_ends: torch.Tensor,
        grad_y: torch.Tensor,
        grad_s_end: torch.Tensor,
        grad_u_end: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        project_boundary_tangent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x_seq.shape
        n_heads = d_model // 32
        n_chunks = seq_len // 32
        bh = bsz * n_heads

        if not x_seq.is_cuda:
            return _cpu_ref_chunk32_bwd(
                x_seq, xt_out, zs_out, w_su2, W_q, s_ckpts, u_ends,
                grad_y, grad_s_end, grad_u_end, cayley_t, gamma, eta, eps,
                project_boundary_tangent=project_boundary_tangent,
            )

        with torch.amp.autocast("cuda", enabled=False):
            x_c = x_seq.float().contiguous()
            xt_c = xt_out.float().contiguous()
            zs_c = zs_out.float().contiguous()
            w_c = w_su2.float().contiguous()
            wq_c = W_q.float().contiguous()
            sck_c = s_ckpts.float().contiguous()
            ue_c = u_ends.float().contiguous()
            gy_c = grad_y.float().contiguous()
            gS_in = grad_s_end.float().contiguous()
            gU_in = grad_u_end.float().contiguous()
            dev = x_c.device

            gz_out = torch.empty_like(x_c)
            gxt_sf = torch.empty_like(x_c)
            gx_geo = torch.empty_like(x_c)
            gS_out = torch.empty_like(gS_in)
            gx_out = torch.empty_like(x_c)
            gw_out = torch.empty_like(w_c)
            gU_out = torch.empty_like(gU_in)

            lib = load_v3_training_ext()
            stream_ptr = ctypes.c_void_p(int(torch.cuda.current_stream(dev).cuda_stream))
            rc1 = lib.launch_holosphere_v3_chunk32_bwd_wmma(
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(xt_c.data_ptr()),
                ctypes.c_void_p(zs_c.data_ptr()),
                ctypes.c_void_p(sck_c.data_ptr()),
                ctypes.c_void_p(gy_c.data_ptr()),
                ctypes.c_void_p(gS_in.data_ptr()),
                ctypes.c_void_p(gz_out.data_ptr()),
                ctypes.c_void_p(gxt_sf.data_ptr()),
                ctypes.c_void_p(gx_geo.data_ptr()),
                ctypes.c_void_p(gS_out.data_ptr()),
                float(cayley_t),
                float(gamma),
                float(eta),
                float(eps),
                int(n_heads),
                int(seq_len * d_model),
                int(d_model),
                int(n_chunks),
                int(bh),
                stream_ptr,
            )
            check_cuda_rc(rc1, "V3_CUDA_BWD_WMMA_ERR", "bwd_wmma")

            gz_v = gz_out.view(bsz, seq_len, n_heads, 32)
            xt_v = xt_c.view(bsz, seq_len, n_heads, 32)
            grad_wq = torch.einsum("bthe,bthd->hed", gz_v, xt_v).float().contiguous()
            gxt_w = torch.einsum("bthe,hed->bthd", gz_v, wq_c).reshape(bsz, seq_len, d_model).float()
            gxt_tot = (gxt_sf + gxt_w).float().contiguous()

            rc3 = lib.launch_holosphere_v3_su2_3line_gu_bwd_persistent(
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(gxt_tot.data_ptr()),
                ctypes.c_void_p(gx_geo.data_ptr()),
                ctypes.c_void_p(w_c.data_ptr()),
                ctypes.c_void_p(ue_c.data_ptr()),
                ctypes.c_void_p(gU_in.data_ptr()),
                ctypes.c_void_p(gx_out.data_ptr()),
                ctypes.c_void_p(gw_out.data_ptr()),
                ctypes.c_void_p(gU_out.data_ptr()),
                int(project_boundary_tangent),
                int(n_heads),
                int(seq_len * d_model),
                int(d_model),
                int(n_chunks),
                int(bh),
                stream_ptr,
            )
            check_cuda_rc(rc3, "V3_CUDA_BWD_SU2_ERR", "su2_bwd_persistent")
            return gx_out, gw_out, grad_wq, gS_out, gU_out

    @torch.library.register_fake("holosphere_v3::chunk32_fused_bwd")
    def _op_v3_chunk32_fused_bwd_fake(
        x_seq: torch.Tensor,
        xt_out: torch.Tensor,
        zs_out: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        s_ckpts: torch.Tensor,
        u_ends: torch.Tensor,
        grad_y: torch.Tensor,
        grad_s_end: torch.Tensor,
        grad_u_end: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        project_boundary_tangent: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del xt_out, zs_out, s_ckpts, u_ends, grad_y, cayley_t, gamma, eta, eps, project_boundary_tangent
        return (
            torch.empty_like(x_seq),
            torch.empty_like(w_su2),
            torch.empty_like(W_q),
            torch.empty_like(grad_s_end),
            torch.empty_like(grad_u_end),
        )

    def _setup_v3_ctx(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, ...]) -> None:
        x_seq, w_su2, W_q, _S0, _U0, cayley_t, gamma, eta, eps, proj_tan = inputs
        _y, xt_out, zs_out, s_ckpts, u_ends, _se, _ue = output
        ctx.save_for_backward(x_seq, xt_out, zs_out, w_su2, W_q, s_ckpts, u_ends)
        ctx.params = (cayley_t, gamma, eta, eps, proj_tan)

    def _v3_autograd_bwd(
        ctx: Any,
        grad_y: torch.Tensor,
        _grad_xt: Optional[torch.Tensor],
        _grad_zs: Optional[torch.Tensor],
        _grad_sck: Optional[torch.Tensor],
        _grad_ue: Optional[torch.Tensor],
        grad_s_end: Optional[torch.Tensor],
        grad_u_end: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        x_seq, xt_out, zs_out, w_su2, W_q, s_ckpts, u_ends = ctx.saved_tensors
        cayley_t, gamma, eta, eps, proj_tan = ctx.params
        gS_in = torch.zeros_like(s_ckpts[0]) if grad_s_end is None else grad_s_end.contiguous()
        gU_in = torch.zeros_like(u_ends[0]) if grad_u_end is None else grad_u_end.contiguous()
        gx, gw, gwq, gs0, gu0 = torch.ops.holosphere_v3.chunk32_fused_bwd(
            x_seq.contiguous(),
            xt_out.contiguous(),
            zs_out.contiguous(),
            w_su2.contiguous(),
            W_q.contiguous(),
            s_ckpts.contiguous(),
            u_ends.contiguous(),
            grad_y.contiguous(),
            gS_in,
            gU_in,
            cayley_t,
            gamma,
            eta,
            eps,
            proj_tan,
        )
        return gx, gw, gwq, gs0, gu0, None, None, None, None, None

    torch.library.register_autograd(
        "holosphere_v3::chunk32_fused_fwd",
        _v3_autograd_bwd,
        setup_context=_setup_v3_ctx,
    )

    def _a16_blockh4_ste_cpu(w: torch.Tensor) -> torch.Tensor:
        w_f = w.float().contiguous()
        flat = w_f.view(-1, 32)
        h4 = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
            dtype=torch.float32,
            device=w.device,
        )
        u_raw = 0.5 * torch.matmul(flat.view(-1, 8, 4), h4)
        amax = u_raw.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6, max=65504.0 * 7.5)
        sc = (amax / 7.5).to(torch.float16).float()
        u_sc = (u_raw / sc).clamp(-8.0, 7.0)
        c_int = torch.round(u_sc).clamp(-8.0, 7.0).to(torch.int32).view(-1, 2, 16)
        u_16 = u_sc.view(-1, 2, 16)
        for p_pos, idxs in (
            (1, (3, 5, 7, 9, 11, 13, 15)),
            (2, (3, 6, 7, 10, 11, 14, 15)),
            (4, (5, 6, 7, 12, 13, 14, 15)),
            (8, (9, 10, 11, 12, 13, 14, 15)),
        ):
            req = torch.zeros_like(c_int[..., 0])
            for j in idxs:
                req = req ^ (c_int[..., j] & 1)
            cur = c_int[..., p_pos]
            need = (cur & 1) != req
            step = torch.where(u_16[..., p_pos] >= cur.float(), torch.ones_like(cur), -torch.ones_like(cur))
            step = torch.where((cur == 7) & (step > 0), -torch.ones_like(cur), step)
            step = torch.where((cur == -8) & (step < 0), torch.ones_like(cur), step)
            c_int[..., p_pos] = torch.where(need, cur + step, cur)
        req0 = torch.zeros_like(c_int[..., 0])
        for j in range(1, 16):
            req0 = req0 ^ (c_int[..., j] & 1)
        cur0 = c_int[..., 0]
        need0 = (cur0 & 1) != req0
        step0 = torch.where(u_16[..., 0] >= cur0.float(), torch.ones_like(cur0), -torch.ones_like(cur0))
        step0 = torch.where((cur0 == 7) & (step0 > 0), -torch.ones_like(cur0), step0)
        step0 = torch.where((cur0 == -8) & (step0 < 0), torch.ones_like(cur0), step0)
        c_int[..., 0] = torch.where(need0, cur0 + step0, cur0)
        r_int8 = torch.matmul(c_int.float().view(-1, 8, 4), h4)
        return (0.5 * r_int8 * sc).reshape(w.shape).to(w.dtype)

    class FusedA16BlockH4STECuda(torch.autograd.Function):
        """Single-launch sm_86 ``A_16 Block-H4`` ``[16, 11, 4]`` STE with exact CPU fallback."""

        @staticmethod
        def forward(ctx: Any, w: torch.Tensor) -> torch.Tensor:
            del ctx
            require_ok(
                isinstance(w, torch.Tensor) and w.is_floating_point(),
                "Weight must be a floating-point Tensor (not bool/int)",
                code="A16_STE_DTYPE",
            )
            require_ok(w.numel() > 0 and w.numel() % 32 == 0, "Weight numel must be positive mult of 32", code="A16_STE_SHAPE")
            require_ok(bool(torch.isfinite(w).all().item()), "Weight contains non-finite NaN/Inf", code="A16_STE_NONFINITE")
            h4 = torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
                dtype=torch.float32,
                device=w.device,
            )
            u_raw_check = 0.5 * torch.matmul(w.float().contiguous().view(-1, 8, 4), h4)
            u_max_t = u_raw_check.abs().amax()
            require_ok(
                bool(torch.isfinite(u_max_t).item()) and float(u_max_t.item()) <= A16_MAX_URAW_ABS,
                f"A_16 STE weight exceeds FP16 scale ceiling {A16_MAX_URAW_ABS}",
                code="A16_STE_OVERFLOW",
            )
            if not w.is_cuda:
                return _a16_blockh4_ste_cpu(w)
            with torch.amp.autocast("cuda", enabled=False):
                w_c = w.float().contiguous()
                out = torch.empty_like(w_c)
                lib = load_v3_training_ext()
                s_ptr = ctypes.c_void_p(int(torch.cuda.current_stream(w_c.device).cuda_stream))
                rc = lib.launch_holosphere_v3_a16_blockh4_ste_quant(
                    ctypes.c_void_p(w_c.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    int(w_c.numel()),
                    s_ptr,
                )
                check_cuda_rc(rc, "A16_STE_CUDA_ERR", "a16_blockh4_ste_quant")
                return out.to(w.dtype)

        @staticmethod
        def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
            del ctx
            return grad_out

    class FastHardFMNMXTangentCayleySTE(torch.autograd.Function):
        """Zero-materialization ``O(B*T*H*32)`` Hard-FMNMX tangent-space Cayley STE."""

        @staticmethod
        def forward(
            ctx: Any,
            sim_scores: torch.Tensor,
            v_ring: torch.Tensor,
            tau: float,
            eps: float,
        ) -> torch.Tensor:
            require_ok(
                _is_finite_real(tau) and float(tau) > 0.0 and _is_finite_real(eps) and float(eps) > 0.0,
                "FastHardFMNMXTangentCayleySTE tau and eps must be positive finite reals",
                code="VRING_BAD_PARAM",
            )
            idx = sim_scores.argmax(dim=-1)
            hard_sel = torch.nn.functional.one_hot(idx, num_classes=sim_scores.shape[-1]).to(sim_scores.dtype)
            y_hard = torch.einsum("bthk,bhkd->bthd", hard_sel, v_ring)
            ctx.save_for_backward(sim_scores, v_ring, y_hard)
            ctx.tau = float(tau)
            ctx.eps = float(eps)
            return y_hard

        @staticmethod
        def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
            sim_scores, v_ring, y_hard = ctx.saved_tensors
            p_soft = torch.softmax(sim_scores / ctx.tau, dim=-1)
            y_soft_raw = torch.einsum("bthk,bhkd->bthd", p_soft, v_ring)
            dot_sy = (y_soft_raw * y_hard).sum(dim=-1, keepdim=True)
            v_tan = y_soft_raw - dot_sy * y_hard
            inv_n = torch.rsqrt((v_tan * v_tan).sum(dim=-1, keepdim=True) + ctx.eps)
            g_tan = grad_out - (grad_out * y_hard).sum(dim=-1, keepdim=True) * y_hard
            g_ysoft = (g_tan - (g_tan * (v_tan * inv_n)).sum(dim=-1, keepdim=True) * (v_tan * inv_n)) * inv_n
            g_p = torch.einsum("bthd,bhkd->bthk", g_ysoft, v_ring)
            g_sim = (p_soft * (g_p - (p_soft * g_p).sum(dim=-1, keepdim=True))) / ctx.tau
            g_vring = torch.einsum("bthk,bthd->bhkd", p_soft, g_ysoft)
            return g_sim, g_vring, None, None


def holosphere_v3_chunk32_fused(
    x_seq: Any,
    w_su2: Any,
    W_q: Any,
    S_0: Optional[Any] = None,
    U_0: Optional[Any] = None,
    *,
    cayley_t: float = 0.25,
    gamma: float = 0.90,
    eta: float = 0.15,
    eps: float = 1e-6,
    project_boundary_tangent: int = 1,
) -> Tuple[Any, Any, Any]:
    """Single-launch-per-stage fused persistent Chunk32 op (``torch.compile(fullgraph=True)``-safe)."""
    _require_torch()
    require_ok(
        isinstance(x_seq, torch.Tensor)
        and x_seq.ndim == 3
        and x_seq.shape[0] > 0
        and x_seq.shape[1] > 0
        and x_seq.shape[1] % 32 == 0
        and x_seq.shape[2] > 0
        and x_seq.shape[2] % 32 == 0,
        "x_seq must be a 3D Tensor [B>0, T mult 32, D mult 32]",
        code="V3_BAD_SHAPE",
    )
    require_ok(
        _is_finite_f32(cayley_t)
        and _is_finite_f32(gamma)
        and _is_finite_f32(eta)
        and _is_finite_f32(eps)
        and float(eps) > 0.0
        and _is_strict_int(project_boundary_tangent)
        and project_boundary_tangent in (0, 1),
        "Non-finite or invalid Chunk32 scalar parameters",
        code="V3_BAD_PARAM",
    )
    bsz, seq_len, d_model = x_seq.shape
    n_heads = d_model // 32
    require_ok(
        isinstance(w_su2, torch.Tensor)
        and isinstance(W_q, torch.Tensor)
        and tuple(w_su2.shape) == (bsz, seq_len, n_heads, 8, 3)
        and tuple(W_q.shape) == (n_heads, 32, 32),
        "Unexpected w_su2 or W_q tensor/shape",
        code="V3_BAD_WEIGHT_SHAPE",
    )
    if S_0 is None:
        S_0 = torch.zeros((bsz, n_heads, 32, 32), dtype=x_seq.dtype, device=x_seq.device)
    if U_0 is None:
        U_0 = torch.zeros((bsz, n_heads, 8, 4), dtype=x_seq.dtype, device=x_seq.device)
        U_0[..., 0] = 1.0
    require_ok(
        isinstance(S_0, torch.Tensor)
        and isinstance(U_0, torch.Tensor)
        and tuple(S_0.shape) == (bsz, n_heads, 32, 32)
        and tuple(U_0.shape) == (bsz, n_heads, 8, 4),
        "Unexpected S_0 or U_0 tensor/shape",
        code="V3_BAD_STATE_SHAPE",
    )
    require_ok(
        x_seq.is_floating_point()
        and w_su2.dtype == x_seq.dtype
        and W_q.dtype == x_seq.dtype
        and S_0.dtype == x_seq.dtype
        and U_0.dtype == x_seq.dtype
        and w_su2.device == x_seq.device
        and W_q.device == x_seq.device
        and S_0.device == x_seq.device
        and U_0.device == x_seq.device,
        "Chunk32 fused inputs must share a single floating-point dtype and device",
        code="V3_BAD_DTYPE",
    )
    y_out, _xt, _zs, _sck, _ue, s_end, u_end = torch.ops.holosphere_v3.chunk32_fused_fwd(
        x_seq.contiguous(),
        w_su2.contiguous(),
        W_q.contiguous(),
        S_0.contiguous(),
        U_0.contiguous(),
        float(cayley_t),
        float(gamma),
        float(eta),
        float(eps),
        int(project_boundary_tangent),
    )
    return y_out, s_end, u_end


def validate_cpu_e2e_autograd_report(rep: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-closed invariant gate for ``run_cpu_e2e_autograd_self_test`` output."""
    require_ok(isinstance(rep, dict), "CPU autograd report must be a dict", code="V3_CPU_REPORT_SCHEMA")
    require_ok(
        _is_finite_real(rep.get("worst_fp64_autograd_err")) and 0.0 <= float(rep["worst_fp64_autograd_err"]) < 1e-12,
        "CPU FP64 autograd parity drift exceeded 1e-12",
        code="V3_CPU_AUTOGRAD_DRIFT",
    )
    require_ok(
        _is_finite_real(rep.get("torch_compile_fullgraph_diff")) and 0.0 <= float(rep["torch_compile_fullgraph_diff"]) < 1e-12,
        "torch.compile fullgraph drift exceeded 1e-12",
        code="V3_COMPILE_DRIFT",
    )
    require_ok(
        rep.get("fast_vring_ste_finite") is True,
        "FastHardFMNMXTangentCayleySTE produced non-finite gradients",
        code="V3_VRING_GRAD_NONFINITE",
    )
    require_ok(
        _is_finite_real(rep.get("a16_ste_grad_err")) and float(rep["a16_ste_grad_err"]) == 0.0,
        "A_16 STE backward identity mismatch",
        code="A16_STE_GRAD",
    )
    return rep


def run_cpu_e2e_autograd_self_test() -> Dict[str, Any]:
    """Deterministic CPU FP64 autograd + ``torch.compile(fullgraph=True, aot_eager)`` self-test."""
    _require_torch()
    gen = torch.Generator(device="cpu").manual_seed(11704)
    bsz, seq_len, d_model, n_heads = 1, 64, 32, 1

    x_base = torch.randn((bsz, seq_len, d_model), generator=gen, dtype=torch.float64)
    x_h = x_base.view(bsz, seq_len, n_heads, 32)
    x_base = (x_h / x_h.norm(dim=-1, keepdim=True)).reshape(bsz, seq_len, d_model)
    w_base = 0.12 * torch.randn((bsz, seq_len, n_heads, 8, 3), generator=gen, dtype=torch.float64)
    wq_base = 0.08 * torch.randn((n_heads, 32, 32), generator=gen, dtype=torch.float64)
    s0_base = 0.04 * torch.randn((bsz, n_heads, 32, 32), generator=gen, dtype=torch.float64)
    u0_raw = torch.randn((bsz, n_heads, 8, 4), generator=gen, dtype=torch.float64)
    u0_base = u0_raw / u0_raw.norm(dim=-1, keepdim=True)

    x_a = x_base.clone().requires_grad_(True)
    w_a = w_base.clone().requires_grad_(True)
    wq_a = wq_base.clone().requires_grad_(True)
    s0_a = s0_base.clone().requires_grad_(True)
    u0_a = u0_base.clone().requires_grad_(True)
    y_a, se_a, ue_a = holosphere_v3_chunk32_fused(x_a, w_a, wq_a, s0_a, u0_a, project_boundary_tangent=0)
    (y_a.square().mean() + 0.25 * se_a.square().mean() + 0.1 * ue_a.square().mean()).backward()

    x_b = x_base.clone().requires_grad_(True)
    w_b = w_base.clone().requires_grad_(True)
    wq_b = wq_base.clone().requires_grad_(True)
    s0_b = s0_base.clone().requires_grad_(True)
    u0_b = u0_base.clone().requires_grad_(True)
    y_b, _, _, _, _, se_b, ue_b = _cpu_ref_chunk32_fwd(x_b, w_b, wq_b, s0_b, u0_b, 0.25, 0.90, 0.15, 1e-6)
    (y_b.square().mean() + 0.25 * se_b.square().mean() + 0.1 * ue_b.square().mean()).backward()

    errs = {
        "fwd_y": float((y_a.detach() - y_b.detach()).abs().max().item()),
        "grad_x": float((x_a.grad - x_b.grad).abs().max().item()),
        "grad_w": float((w_a.grad - w_b.grad).abs().max().item()),
        "grad_wq": float((wq_a.grad - wq_b.grad).abs().max().item()),
        "grad_s0": float((s0_a.grad - s0_b.grad).abs().max().item()),
        "grad_u0": float((u0_a.grad - u0_b.grad).abs().max().item()),
    }
    worst_err = max(errs.values())

    def _compiled_fn(x_i: torch.Tensor, w_i: torch.Tensor, wq_i: torch.Tensor) -> torch.Tensor:
        y_i, s_i, u_i = holosphere_v3_chunk32_fused(x_i, w_i, wq_i)
        return y_i.sum() + 0.1 * s_i.sum() + 0.05 * u_i.sum()

    compiled = torch.compile(_compiled_fn, fullgraph=True, backend="aot_eager")
    xc1, wc1, wqc1 = x_base.clone().requires_grad_(True), w_base.clone().requires_grad_(True), wq_base.clone().requires_grad_(True)
    xc2, wc2, wqc2 = x_base.clone().requires_grad_(True), w_base.clone().requires_grad_(True), wq_base.clone().requires_grad_(True)
    out_e = _compiled_fn(xc1, wc1, wqc1)
    out_e.backward()
    out_c = compiled(xc2, wc2, wqc2)
    out_c.backward()
    comp_diff = max(
        float((out_e.detach() - out_c.detach()).abs().item()),
        float((xc1.grad - xc2.grad).abs().max().item()),
        float((wc1.grad - wc2.grad).abs().max().item()),
        float((wqc1.grad - wqc2.grad).abs().max().item()),
    )

    sim = torch.randn((1, 8, 2, 8), generator=gen, dtype=torch.float64, requires_grad=True)
    vring = torch.randn((1, 2, 8, 32), generator=gen, dtype=torch.float64)
    vring = (vring / vring.norm(dim=-1, keepdim=True)).detach().requires_grad_(True)
    y_fast = FastHardFMNMXTangentCayleySTE.apply(sim, vring, 0.25, 1e-6)
    g_out = torch.randn_like(y_fast)
    y_fast.backward(g_out)
    vring_ok = bool(torch.isfinite(sim.grad).all().item() and torch.isfinite(vring.grad).all().item())

    w_ste = torch.randn((4, 64), generator=gen, dtype=torch.float32, requires_grad=True)
    q_ste = FusedA16BlockH4STECuda.apply(w_ste)
    q_ste.sum().backward()
    ste_grad_err = float((w_ste.grad - torch.ones_like(w_ste)).abs().max().item())

    report = {
        "schema": "holosphere.track_s117.e2e_cpu_autograd_self_test.v1",
        "worst_fp64_autograd_err": worst_err,
        "torch_compile_fullgraph_diff": comp_diff,
        "a16_ste_grad_err": ste_grad_err,
        "fast_vring_ste_finite": vring_ok,
    }
    return validate_cpu_e2e_autograd_report(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HoloSphere-V3 E2E Engine verifier & CPU self-test.")
    parser.add_argument("--check", action="store_true", help="Alias for --verify-receipt.")
    parser.add_argument("--verify-receipt", action="store_true", help="Verify ptxas & Zone-B live receipts.")
    parser.add_argument("--self-test", action="store_true", help="Run pure-stdlib A_16 / Weft / Razor self-test.")
    parser.add_argument("--cpu-autograd-test", action="store_true", help="Run CPU FP64 autograd & compile self-test.")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    args = parser.parse_args(argv)

    verify_flag = args.verify_receipt or args.check
    run_all = not (verify_flag or args.self_test or args.cpu_autograd_test)
    out: Dict[str, Any] = {"schema": "holosphere.track_s117.e2e_engine_cli.v1"}
    if verify_flag or run_all:
        out["ptxas_receipt"] = verify_e2e_ptxas_receipt()
        out["live_receipts"] = verify_e2e_live_receipts()
    if args.self_test or run_all:
        out["stdlib_self_test"] = run_e2e_stdlib_self_test()
    if args.cpu_autograd_test:
        out["cpu_autograd_self_test"] = run_cpu_e2e_autograd_self_test()

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        sections = ",".join(k for k in sorted(out) if k != "schema")
        print(f"PASS holosphere.e2e_engine sections={sections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
