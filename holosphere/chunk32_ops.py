"""Chunk32 O(T/32)-VRAM Woodbury custom ops, H_256 x bent4 byte injection, and 23-bit Gumbel.

Binds the Track S117 bare-metal Ampere ``sm_86`` kernels in
``holosphere/kernels/csrc/holosphere_h256_chunk32_sm86.cu``:
  - Zero-VRAM orthonormal ``H_256 x bent4`` byte injection (``0 / 2,048`` constant warps)
  - Strictly-inside-``(0, 1)`` 23-bit dyadic Gumbel noise + deterministic ``FMNMX`` argmax
  - 3-stage ``C = 32`` chunked ``O(T/32)``-VRAM forward & closed-form Woodbury backward
    (Choice A ``4,224 B/warp`` & Choice B ``6,336 B/warp``) + 3-line non-Abelian ``SU(2)`` ``gU``
  - ``TORCH_LIBRARY`` custom ops (``holosphere_s117::chunk32_fwd`` / ``chunk32_bwd``) with
    ``@torch.library.register_fake``, ``torch.library.register_autograd``, explicit
    ``x.contiguous()`` stride guards, and ``torch.compile(fullgraph=True)`` parity
  - Pure-stdlib fallback (zero ``torch`` / ``numpy`` required for exact algebraic, budget,
    and ``ptxas_s117_zero_spill_receipt.json`` verification)
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
from typing import Any, Dict, List, Optional, Tuple

from holosphere.errors import HoloSphereError, require_ok

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - stock stdlib mode
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

CU_FILENAME = "holosphere_h256_chunk32_sm86.cu"
PTXAS_FILENAME = "ptxas_s117_zero_spill_receipt.json"
V3_CU_FILENAME = "holosphere_v3_frontier_sm86.cu"
V3_PTXAS_FILENAME = "ptxas_v3_frontier_zero_spill_receipt.json"
DEFAULT_CSRC_DIR = Path(__file__).resolve().parent / "kernels" / "csrc"

EXPECTED_S117_KERNEL_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN9s117_sm8637s117_uniform_cell_16warp_d512_v2_sm86EPKhPKjPKfPK6float3S3_PK6__halfS3_P6float4PfSE_PhSE_Pjffiii",
        "_ZN9s117_sm8634s117_uniform_cell_16warp_d512_sm86EPKhPKaPKfPK6float3S5_PK6__halfS5_PKjP6float4PfPhSG_Pjffii",
        "_ZN9s117_sm8627s117_spec8_fused_4imma_sm86EPKaPKfS1_S3_S1_S3_Pffi",
        "_ZN9s117_sm8634s117_sprefix_int8_shadow_imma_sm86EPKfPKaS1_PaPfS5_S5_i",
        "_ZN9s117_sm8631s117_spec8_woodbury_commit_sm86EPfP6float4PKaPKfPK6__halfPK6float3S0_ffiii",
        "_ZN9s117_sm8626s117_su2_3line_gu_bwd_sm86EPKfS1_S1_PK6float3PK6float4S7_PfPS2_PS5_iiiii",
        "_ZN9s117_sm8640s117_sfast_geo_chunk32_bwd_choice_b_sm86EPKfS1_S1_PK6__halfS1_S1_PfS5_S5_S5_ffffiiii",
        "_ZN9s117_sm8631s117_sfast_geo_chunk32_bwd_sm86EPKfS1_S1_PK6__halfS1_S1_PfS5_S5_S5_ffffiiii",
        "_ZN9s117_sm8631s117_sfast_geo_chunk32_fwd_sm86EPKfS1_PK6float3PK6__halfPK6float4PfSB_PS5_PS8_ffffiiiii",
        "_ZN9s117_sm8635s117_gumbel_fmnmx_23bit_cellop_sm86EPKfPKjPhPffi",
        "_ZN9s117_sm8632s117_h256_bent4_byte_inject_sm86EPKhPai",
    }
)

EXPECTED_V3_KERNEL_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN7s117_v326s117_chunk32_bwd_wmma_sm86EPKfS1_S1_S1_S1_S1_PfS2_S2_S2_ffffiiiii",
        "_ZN7s117_v337s117_uniform_cell_16warp_d512_v3_sm86EPKhPK5uint4PKfPK6float3S4_S4_PKjP6float4PfSE_PhSE_Pjffiii",
    }
)


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


# ============================================================================
# 1. Pure-Stdlib H_256 x bent4 Byte Injection, 23-Bit Gumbel, & VRAM Model
# ============================================================================


def h256_bent4_byte_inject_s8(utf8_byte: int, warp_id: int, lane_id: int) -> int:
    """Bit-exact CPU mirror of ``s117_sm86::h256_bent4_byte_inject_s8``."""
    d = ((warp_id & 7) << 5) | (lane_id & 31)
    db = d ^ ((lane_id & 7) << 5)
    bent4 = ((lane_id & 3) & ((lane_id >> 2) & 3)).bit_count() & 1
    parity = (((utf8_byte & 0xFF) & db).bit_count() ^ bent4) & 1
    return -127 if parity else 127


def verify_h256_bent4_exhaustive(inject_fn: Any = h256_bent4_byte_inject_s8) -> Dict[str, Any]:
    """Exhaustively prove 256x256 orthogonality & 0/2048 constant warps (max |warp_sum| == 8)."""
    vecs: List[List[int]] = [
        [inject_fn(b, d >> 5, d & 31) for d in range(256)]
        for b in range(256)
    ]
    diag_expected = 256 * 127 * 127
    off_diag_errors = 0
    diag_errors = 0
    for b1 in range(256):
        v1 = vecs[b1]
        for b2 in range(b1, 256):
            v2 = vecs[b2]
            dot = sum(a * c for a, c in zip(v1, v2))
            if b1 == b2:
                if dot != diag_expected:
                    diag_errors += 1
            else:
                if dot != 0:
                    off_diag_errors += 1

    const_warps = 0
    max_abs_warp_sum = 0
    for b in range(256):
        vb = vecs[b]
        for w in range(8):
            sl = vb[w * 32 : (w + 1) * 32]
            s_unit = sum(1 if x > 0 else -1 for x in sl)
            abs_s = abs(s_unit)
            if abs_s > max_abs_warp_sum:
                max_abs_warp_sum = abs_s
            if abs_s == 32:
                const_warps += 1

    require_ok(diag_errors == 0 and off_diag_errors == 0, "H_256 x bent4 failed 256x256 orthogonality", code="H256_NOT_ORTHOGONAL")
    require_ok(const_warps == 0, f"Expected 0/2048 constant warps, got {const_warps}", code="H256_CONST_WARPS")
    require_ok(max_abs_warp_sum == 8, f"Expected max |warp_sum| == 8, got {max_abs_warp_sum}", code="H256_PARSEVAL_BOUND")
    return {
        "pairs_checked": 256 * 257 // 2,
        "diag_norm_sq": diag_expected,
        "off_diag_errors": off_diag_errors,
        "const_warp_pairs": const_warps,
        "total_warp_pairs": 2048,
        "max_abs_warp_sum": max_abs_warp_sum,
    }


def gumbel_23bit_u_f32(u32_val: int) -> float:
    """Exact IEEE-754 FP32 value of ``((u32 >> 9) + 0.5f) * 2^-23``."""
    top23 = (u32_val & 0xFFFFFFFF) >> 9
    return _f32(_f32(float(top23)) * _f32(2.0**-23) + _f32(2.0**-24))


def gumbel_23bit_finite_f32(u32_val: int, temperature: float = 1.0) -> float:
    """Exact FP32 mirror of ``s117_sm86::gumbel_23bit_finite``."""
    u = gumbel_23bit_u_f32(u32_val)
    neg_lg2_u = -_f32(math.log2(u))
    lg2_neg_lg2_u = _f32(math.log2(neg_lg2_u))
    g = _f32(_f32(-0.69314718) * lg2_neg_lg2_u + _f32(0.36651292))
    return _f32(_f32(temperature) * g)


def verify_gumbel_23bit_finite(
    u_fn: Any = gumbel_23bit_u_f32,
    g_fn: Any = gumbel_23bit_finite_f32,
) -> Dict[str, Any]:
    """Prove 23-bit Gumbel is strictly finite over extremes including ``0xFFFFFF80..0xFFFFFFFF``."""
    naive_u_max = _f32((_f32(float(0xFFFFFFFF)) + _f32(0.5)) * _f32(2.0**-32))

    u_min = u_fn(0x00000000)
    u_max = u_fn(0xFFFFFFFF)
    require_ok(
        u_min == _f32(2.0**-24) and u_max == _f32(1.0 - 2.0**-24),
        f"Bad u endpoints: u_min={u_min}, u_max={u_max}",
        code="GUMBEL_U_BOUNDS",
    )

    g_min = g_fn(0x00000000, 1.0)
    g_max = g_fn(0xFFFFFFFF, 1.0)
    require_ok(
        math.isfinite(g_min)
        and math.isfinite(g_max)
        and abs(g_min - (-2.811541)) < 2.0**-17
        and abs(g_max - 16.635532) < 2.0**-14,
        f"Unexpected Gumbel range: g_min={g_min}, g_max={g_max}",
        code="GUMBEL_RANGE",
    )

    rng = random.Random(11702)
    probes = [0, 1, 0x7FFFFFFF, 0x80000000] + list(range(0xFFFFFF00, 0x100000000)) + [rng.getrandbits(32) for _ in range(4096)]
    for u32 in probes:
        u = u_fn(u32)
        g = g_fn(u32, 1.0)
        require_ok(0.0 < u < 1.0 and math.isfinite(g) and g_min <= g <= g_max, f"u32={u32:#x} failed", code="GUMBEL_PROBE_FAIL")
    require_ok(g_fn(0xFFFFFFFF, 0.0) == 0.0, "temp=0 must be exact 0.0", code="GUMBEL_TEMP0")
    return {
        "naive_32bit_0xFFFFFFFF_u": naive_u_max,
        "u23_min": u_min,
        "u23_max": u_max,
        "gumbel_min": round(g_min, 6),
        "gumbel_max": round(g_max, 6),
        "probes_verified": len(probes),
    }


def pow_gamma_u5_py(gamma: float, t: int) -> float:
    """Exact 5-mul binary exponentiation for ``gamma^t`` (``0 <= t <= 31``, zero ``gamma^-1``)."""
    require_ok(isinstance(t, int) and not isinstance(t, bool) and 0 <= t <= 31, f"t={t} out of 0..31 range", code="BAD_POW_T")
    res = 1.0
    base = float(gamma)
    if t & 1:
        res *= base
    base *= base
    if t & 2:
        res *= base
    base *= base
    if t & 4:
        res *= base
    base *= base
    if t & 8:
        res *= base
    base *= base
    if t & 16:
        res *= base
    return res


def estimate_t8192_vram_budget(
    *,
    batch: int = 4,
    seq_len: int = 8192,
    d_model: int = 512,
    n_layers: int = 12,
    chunk_size: int = 32,
) -> Dict[str, Any]:
    """Compute exact activation checkpoint VRAM for naive per-token vs ``C=32`` chunked ops."""
    require_ok(
        isinstance(batch, int)
        and not isinstance(batch, bool)
        and isinstance(seq_len, int)
        and not isinstance(seq_len, bool)
        and isinstance(d_model, int)
        and not isinstance(d_model, bool)
        and isinstance(n_layers, int)
        and not isinstance(n_layers, bool)
        and batch > 0
        and seq_len > 0
        and d_model > 0
        and d_model % 32 == 0
        and n_layers > 0,
        "Invalid model dims",
        code="BAD_SHAPE",
    )
    require_ok(
        isinstance(chunk_size, int) and not isinstance(chunk_size, bool) and chunk_size > 0 and seq_len % chunk_size == 0,
        "seq_len must be divisible by positive int chunk_size",
        code="BAD_CHUNK",
    )
    n_heads = d_model // 32
    n_chunks = seq_len // chunk_size
    n_spinors = d_model // 4

    naive_sfast_bytes = batch * n_layers * seq_len * n_heads * 32 * 32 * 2
    naive_u_bytes = batch * n_layers * seq_len * n_spinors * 4 * 4
    naive_act_bytes = batch * n_layers * seq_len * d_model * 2 * 6
    naive_total_bytes = naive_sfast_bytes + naive_u_bytes + naive_act_bytes

    chunk_sfast_bytes = batch * n_layers * n_chunks * n_heads * 32 * 32 * 2
    chunk_u_bytes = batch * n_layers * n_chunks * n_spinors * 4 * 4
    chunk_act_bytes = batch * n_layers * seq_len * (d_model + n_heads * 2)
    chunk_total_bytes = chunk_sfast_bytes + chunk_u_bytes + chunk_act_bytes

    phase1_smem_per_warp = (4 + 8) * 32 * 33 * 2  # 25,344 B
    choice_a_smem_per_warp = 2 * 32 * 33 * 2      #  4,224 B
    choice_b_smem_per_warp = 3 * 32 * 33 * 2      #  6,336 B

    mib = float(1024 * 1024)
    try:
        naive_mib = round(naive_total_bytes / mib, 2)
        sfast_mib = round(chunk_sfast_bytes / mib, 2)
        u_mib = round(chunk_u_bytes / mib, 2)
        act_mib = round(chunk_act_bytes / mib, 2)
        chunk_mib = round(chunk_total_bytes / mib, 2)
        ratio = round(naive_total_bytes / float(chunk_total_bytes), 2)
    except OverflowError:
        require_ok(False, "VRAM byte count overflowed float64", code="BAD_SHAPE")
        return {}
    return {
        "shape": {"B": batch, "T": seq_len, "D": d_model, "H": n_heads, "L": n_layers, "C": chunk_size},
        "naive_total_mib": naive_mib,
        "chunk32_sfast_mib": sfast_mib,
        "chunk32_u_mib": u_mib,
        "chunk32_act_mib": act_mib,
        "chunk32_total_mib": chunk_mib,
        "vram_compression_ratio": ratio,
        "fits_8gib_ga104m": (chunk_total_bytes < 2 * 1024 * 1024 * 1024),
        "w10_obs_60_smem": {
            "phase1_smem_per_warp_bytes": phase1_smem_per_warp,
            "choice_a_smem_per_warp_bytes": choice_a_smem_per_warp,
            "choice_b_smem_per_warp_bytes": choice_b_smem_per_warp,
            "choice_a_smem_reduction_x": round(phase1_smem_per_warp / float(choice_a_smem_per_warp), 2),
            "choice_b_smem_reduction_x": round(phase1_smem_per_warp / float(choice_b_smem_per_warp), 2),
            "choice_a_16warps_smem_bytes": choice_a_smem_per_warp * 16,
            "choice_b_16warps_smem_bytes": choice_b_smem_per_warp * 16,
            "ga104m_smem_limit_bytes": 102400,
        },
    }


def _verify_ptxas_receipt_core(
    *,
    csrc_dir: Path,
    cu_filename: str,
    rec_filename: str,
    expected_symbols: frozenset[str],
    require_zero_spill_key: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    cu_path = csrc_dir / cu_filename
    rec_path = csrc_dir / rec_filename
    require_ok(cu_path.is_file(), f"Missing {cu_path}", code="PTXAS_NO_CU")
    require_ok(rec_path.is_file(), f"Missing {rec_path}", code="PTXAS_NO_RECEIPT")
    cu_sha = hashlib.sha256(cu_path.read_bytes()).hexdigest()
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        require_ok(False, f"Malformed receipt JSON ({rec_path}): {exc}", code="PTXAS_BAD_JSON")
        return cu_sha, {}, [], {}
    require_ok(isinstance(rec, dict), "Receipt root must be a JSON object", code="PTXAS_BAD_JSON")
    require_ok(rec.get("cu_sha256") == cu_sha, f"cu_sha256 mismatch vs {cu_filename}", code="PTXAS_CU_SHA")
    require_ok(rec.get("nonzero_spill_kernels") == 0, "nonzero_spill_kernels must be 0", code="PTXAS_NONZERO")
    kernels = rec.get("kernels")
    n_exp = len(expected_symbols)
    require_ok(
        isinstance(kernels, list) and len(kernels) == n_exp and rec.get("total_kernels") == n_exp,
        f"Expected {n_exp} kernels in {rec_filename}",
        code="PTXAS_COUNT",
    )
    if require_zero_spill_key:
        require_ok(rec.get("zero_spill_kernels") == n_exp, f"zero_spill_kernels must equal {n_exp}", code="PTXAS_ZERO_SPILL_KEY")
    seen_symbols: List[str] = []
    for k in kernels:
        require_ok(
            isinstance(k, dict)
            and k.get("stack_frame_bytes") == 0
            and k.get("spill_store_bytes") == 0
            and k.get("spill_load_bytes") == 0
            and k.get("zero_spill") is True,
            f"Kernel spill or invalid entry detected: {k}",
            code="PTXAS_SPILL",
        )
        sym = k.get("symbol")
        require_ok(isinstance(sym, str) and sym != "", "Kernel entry missing symbol name", code="PTXAS_SYMBOLS")
        seen_symbols.append(sym)
    require_ok(
        len(set(seen_symbols)) == n_exp and frozenset(seen_symbols) == expected_symbols,
        f"Kernel symbol set mismatch or duplicate row detected in {rec_filename}",
        code="PTXAS_SYMBOLS",
    )
    sass = rec.get("sass_instruction_counts")
    require_ok(isinstance(sass, dict), "sass_instruction_counts must be a JSON object", code="SASS_BAD_SCHEMA")
    require_ok(sass.get("LDL") == 0 and sass.get("STL") == 0, "SASS local memory LDL/STL forbidden", code="SASS_LOCAL_MEM")
    require_ok(sass.get("MEMBAR_GL") == 0 and sass.get("ATOMG_GPU") == 0, "Cross-CTA MEMBAR_GL/ATOMG_GPU forbidden", code="SASS_CROSS_CTA_BAR")
    return cu_sha, rec, kernels, sass


def verify_s117_ptxas_receipt(csrc_dir: Path = DEFAULT_CSRC_DIR) -> Dict[str, Any]:
    """Fail-closed verification of ``ptxas_s117_zero_spill_receipt.json`` against ``holosphere_h256_chunk32_sm86.cu``."""
    cu_sha, rec, kernels, sass = _verify_ptxas_receipt_core(
        csrc_dir=csrc_dir,
        cu_filename=CU_FILENAME,
        rec_filename=PTXAS_FILENAME,
        expected_symbols=EXPECTED_S117_KERNEL_SYMBOLS,
        require_zero_spill_key=True,
    )
    red_sys = sass.get("RED_SYS")
    idp4a = sass.get("IDP4A")
    require_ok(
        isinstance(red_sys, int) and not isinstance(red_sys, bool) and red_sys >= 2
        and isinstance(idp4a, int) and not isinstance(idp4a, bool) and idp4a >= 16,
        "Missing .sys canary or IDP4A ops",
        code="SASS_MISSING_OPS",
    )
    return {
        "total_kernels": len(kernels),
        "zero_spill_kernels": rec["zero_spill_kernels"],
        "cu_sha256": cu_sha,
        "sass_LDL": sass["LDL"],
        "sass_STL": sass["STL"],
        "sass_MEMBAR_GL": sass["MEMBAR_GL"],
        "sass_RED_SYS": red_sys,
        "sass_IDP4A": idp4a,
    }


def verify_v3_frontier_ptxas_receipt(csrc_dir: Path = DEFAULT_CSRC_DIR) -> Dict[str, Any]:
    """Fail-closed verification of ``ptxas_v3_frontier_zero_spill_receipt.json`` against ``holosphere_v3_frontier_sm86.cu``."""
    cu_sha, rec, kernels, sass = _verify_ptxas_receipt_core(
        csrc_dir=csrc_dir,
        cu_filename=V3_CU_FILENAME,
        rec_filename=V3_PTXAS_FILENAME,
        expected_symbols=EXPECTED_V3_KERNEL_SYMBOLS,
        require_zero_spill_key=True,
    )
    hmma = sass.get("HMMA_16816")
    idp4a = sass.get("IDP4A")
    require_ok(
        isinstance(hmma, int) and not isinstance(hmma, bool) and hmma >= 28
        and isinstance(idp4a, int) and not isinstance(idp4a, bool) and idp4a >= 24,
        "Missing V3 HMMA/IDP4A ops",
        code="SASS_V3_OPS",
    )
    return {
        "total_kernels": len(kernels),
        "zero_spill_kernels": rec["zero_spill_kernels"],
        "cu_sha256": cu_sha,
        "sass_HMMA_16816": hmma,
        "sass_IDP4A": idp4a,
        "sass_LDL": sass["LDL"],
        "sass_STL": sass["STL"],
    }


# ============================================================================
# 2. C++ TORCH_LIBRARY Extension Blueprint & Dynamic ctypes Loader
# ============================================================================

TORCH_LIBRARY_CPP_BLUEPRINT = r"""// HoloSphere S117 C++ PyTorch Extension Blueprint (holosphere_s117)
// Compile against libholosphere_h256_chunk32_sm86.so with -arch=sm_86 -O3.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

extern "C" {
int launch_s117_sfast_geo_chunk32_fwd(
    const float* x_chunk, const float* z_slow_chunk, const float* w_su2_chunk,
    const void* S_ckpt_0, const float* U_ckpt_0,
    float* y_out, float* x_tilde_out, void* S_ckpt_end, float* U_ckpt_end,
    float cayley_t, float gamma, float eta, float eps, int apply_nr_polish,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream);

int launch_s117_sfast_geo_chunk32_bwd(
    const float* x_chunk, const float* x_tilde_chunk, const float* z_slow_chunk,
    const void* S_ckpt_0, const float* grad_y_chunk, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream);

int launch_s117_sfast_geo_chunk32_bwd_choice_b(
    const float* x_chunk, const float* x_tilde_chunk, const float* z_slow_chunk,
    const void* S_ckpt_0, const float* grad_y_chunk, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream);
}

static inline void check_s117_cuda_f32(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous (stride_t = D)");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
}

TORCH_LIBRARY(holosphere_s117, m) {
    m.def("stage1_bwd(Tensor x, Tensor x_tilde, Tensor z_slow, Tensor S_ckpt_0, "
          "Tensor grad_y, Tensor grad_S_in, float cayley_t, float gamma, float eta, "
          "float eps, bool use_choice_b) -> (Tensor, Tensor, Tensor, Tensor)");
}
"""

_CUDA_LIB_CACHE: Optional[ctypes.CDLL] = None


def load_s117_cuda_lib(lib_path: Optional[Path] = None) -> Optional[ctypes.CDLL]:
    """Dynamically load compiled ``libholosphere_h256_chunk32_sm86.so`` if provided via ``S117_SM86_LIB``."""
    global _CUDA_LIB_CACHE
    if _CUDA_LIB_CACHE is not None and lib_path is None:
        return _CUDA_LIB_CACHE
    cand = lib_path
    if cand is None:
        env_p = os.environ.get("S117_SM86_LIB", "").strip()
        if env_p:
            cand = Path(env_p)
    if cand is None or not cand.is_file():
        return None
    lib = ctypes.CDLL(str(cand))
    if lib_path is None:
        _CUDA_LIB_CACHE = lib
    return lib


# ============================================================================
# 3. PyTorch Closed-Form Woodbury Custom Ops (Zero-Transpose [B, T, D])
# ============================================================================


def _require_torch() -> None:
    require_ok(_HAS_TORCH, "PyTorch is required for holosphere_s117 custom ops", code="NO_TORCH")


def h256_bent4_embed_torch(byte_ids: Any, d_model: int = 256) -> Any:
    """Zero-VRAM ``H_256 x bent4`` byte injection returning unit-norm ``S^31`` slices ``[..., d_model]``."""
    _require_torch()
    require_ok(d_model in (256, 512), "d_model must be 256 or 512", code="BAD_DMODEL")
    flat = byte_ids.reshape(-1).to(torch.int64).tolist()
    inv_sqrt32 = 1.0 / math.sqrt(32.0)
    rows: List[List[float]] = []
    for b in flat:
        row256 = [inv_sqrt32 if h256_bent4_byte_inject_s8(int(b), d >> 5, d & 31) > 0 else -inv_sqrt32 for d in range(256)]
        if d_model == 512:
            rows.append(row256 + [-v for v in row256])
        else:
            rows.append(row256)
    out = torch.tensor(rows, dtype=torch.float32, device=byte_ids.device)
    return out.reshape(*byte_ids.shape, d_model)


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

    def _tq_cayley_pullback(w: torch.Tensor, gq: torch.Tensor) -> torch.Tensor:
        w2 = (w * w).sum(dim=-1, keepdim=True)
        inv = 1.0 / (1.0 + w2)
        dot_wv = (w * gq[..., 1:]).sum(dim=-1, keepdim=True)
        coeff = -4.0 * inv * inv * (gq[..., :1] + dot_wv)
        return 2.0 * inv * gq[..., 1:] + coeff * w

    def _chunk_fwd_btd_torch(
        x_ch: torch.Tensor,
        w_ch: torch.Tensor,
        W_q: torch.Tensor,
        S_cur: torch.Tensor,
        U_cur: torch.Tensor,
        *,
        c_st: float,
        s_st: float,
        gamma: float,
        eta: float,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, clen, d_model = x_ch.shape
        n_heads = d_model // 32
        W_bh = W_q.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, 32, 32) if W_q.ndim == 2 else W_q.unsqueeze(0).expand(bsz, n_heads, 32, 32)
        W_flat = W_bh.reshape(bsz * n_heads, 32, 32)
        S_t = S_cur.reshape(bsz * n_heads, 32, 32)
        U_t = U_cur.reshape(bsz * n_heads, 8, 4)

        ys: List[torch.Tensor] = []
        xts: List[torch.Tensor] = []
        zss: List[torch.Tensor] = []
        for t in range(clen):
            x_t = x_ch[:, t].reshape(bsz * n_heads, 32)
            w_t = w_ch[:, t].reshape(bsz * n_heads, 8, 3)
            U_t = _tq_mul(_tq_cayley(w_t), U_t)
            xt = _tq_mul(_tq_conj(U_t), x_t.reshape(bsz * n_heads, 8, 4)).reshape(bsz * n_heads, 32)
            zs = torch.bmm(W_flat, xt.unsqueeze(-1)).squeeze(-1)
            sp = torch.bmm(S_t, xt.unsqueeze(-1)).squeeze(-1)
            z = zs + sp
            proj = (z * x_t).sum(dim=-1, keepdim=True)
            v = z - proj * x_t
            v_inv = torch.rsqrt((v * v).sum(dim=-1, keepdim=True) + eps)
            y = c_st * x_t + s_st * (v * v_inv)
            err = y - sp
            S_t = gamma * S_t + eta * torch.bmm(err.unsqueeze(-1), xt.unsqueeze(-2))
            ys.append(y.reshape(bsz, d_model))
            xts.append(xt.reshape(bsz, d_model))
            zss.append(zs.reshape(bsz, d_model))

        return (
            torch.stack(ys, dim=1),
            torch.stack(xts, dim=1),
            torch.stack(zss, dim=1),
            S_t.reshape(bsz, n_heads, 32, 32),
            U_t.reshape(bsz, n_heads, 8, 4),
        )

    def _chunk_3stage_bwd_woodbury_btd_torch(
        x_ch: torch.Tensor,
        xt_ch: torch.Tensor,
        zs_ch: torch.Tensor,
        w_ch: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_end: torch.Tensor,
        gy_ch: torch.Tensor,
        gS_in: torch.Tensor,
        gU_in: torch.Tensor,
        *,
        c_st: float,
        s_st: float,
        gamma: float,
        eta: float,
        eps: float,
        use_choice_b: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, clen, d_model = x_ch.shape
        n_heads = d_model // 32
        bh = bsz * n_heads
        W_bh = W_q.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, 32, 32) if W_q.ndim == 2 else W_q.unsqueeze(0).expand(bsz, n_heads, 32, 32)
        W_flat = W_bh.reshape(bh, 32, 32)
        S0_flat = S_0.reshape(bh, 32, 32)

        xt_buf: List[torch.Tensor] = []
        err_buf: List[torch.Tensor] = []
        spred_buf: List[torch.Tensor] = []
        S_t = S0_flat
        for t in range(clen):
            x_t = x_ch[:, t].reshape(bh, 32)
            xt = xt_ch[:, t].reshape(bh, 32)
            zs = zs_ch[:, t].reshape(bh, 32)
            sp = torch.bmm(S_t, xt.unsqueeze(-1)).squeeze(-1)
            z = zs + sp
            v = z - (z * x_t).sum(dim=-1, keepdim=True) * x_t
            v_inv = torch.rsqrt((v * v).sum(dim=-1, keepdim=True) + eps)
            err = (c_st * x_t + s_st * (v * v_inv)) - sp
            xt_buf.append(xt)
            err_buf.append(err)
            if use_choice_b:
                spred_buf.append(sp)
            S_t = gamma * S_t + eta * torch.bmm(err.unsqueeze(-1), xt.unsqueeze(-2))
        del S_t

        gS = gS_in.reshape(bh, 32, 32)
        gz_list: List[Optional[torch.Tensor]] = [None] * clen
        gxt_sf_list: List[Optional[torch.Tensor]] = [None] * clen
        gx_geo_list: List[Optional[torch.Tensor]] = [None] * clen

        for t in range(clen - 1, -1, -1):
            x_t = x_ch[:, t].reshape(bh, 32)
            xt = xt_buf[t]
            err_t = err_buf[t]
            zs = zs_ch[:, t].reshape(bh, 32)
            gy = gy_ch[:, t].reshape(bh, 32)
            g_t = pow_gamma_u5_py(gamma, t)

            if use_choice_b:
                sp = spred_buf[t]
            else:
                sp = g_t * torch.bmm(S0_flat, xt.unsqueeze(-1)).squeeze(-1)
                w_fwd = eta
                for k in range(t - 1, -1, -1):
                    dot_x = (xt_buf[k] * xt).sum(dim=-1, keepdim=True)
                    sp = sp + (w_fwd * dot_x) * err_buf[k]
                    w_fwd *= gamma

            ge = eta * torch.bmm(gS, xt.unsqueeze(-1)).squeeze(-1)
            z = zs + sp
            proj = (z * x_t).sum(dim=-1, keepdim=True)
            v = z - proj * x_t
            v_inv = torch.rsqrt((v * v).sum(dim=-1, keepdim=True) + eps)
            v_inv3 = v_inv * v_inv * v_inv

            gx_next = gy + ge
            gu = s_st * gx_next
            gv = v_inv * gu - (v_inv3 * (gu * v).sum(dim=-1, keepdim=True)) * v
            gv_dot_x = (gv * x_t).sum(dim=-1, keepdim=True)
            gz = gv - gv_dot_x * x_t
            gx_geo = c_st * gx_next - proj * gv - gv_dot_x * z

            v_t = gz - ge
            st_v = g_t * torch.bmm(S0_flat.transpose(-1, -2), v_t.unsqueeze(-1)).squeeze(-1)
            w_adj = eta
            for k in range(t - 1, -1, -1):
                dot_e = (err_buf[k] * v_t).sum(dim=-1, keepdim=True)
                st_v = st_v + (w_adj * dot_e) * xt_buf[k]
                w_adj *= gamma
            gst_e = eta * torch.bmm(gS.transpose(-1, -2), err_t.unsqueeze(-1)).squeeze(-1)
            gS = gamma * gS + torch.bmm(v_t.unsqueeze(-1), xt.unsqueeze(-2))

            gz_list[t] = gz
            gxt_sf_list[t] = st_v + gst_e
            gx_geo_list[t] = gx_geo

        gz_bh = torch.stack([item for item in gz_list if item is not None], dim=1)
        xt_bh = torch.stack(xt_buf, dim=1)
        gxt_sf_bh = torch.stack([item for item in gxt_sf_list if item is not None], dim=1)
        gx_geo_bh = torch.stack([item for item in gx_geo_list if item is not None], dim=1)

        gW_bh = torch.bmm(gz_bh.transpose(1, 2), xt_bh).reshape(bsz, n_heads, 32, 32)
        gW_out = gW_bh.sum(dim=(0, 1)) if W_q.ndim == 2 else gW_bh.sum(dim=0)
        gxt_tot = gxt_sf_bh + torch.bmm(gz_bh, W_flat)

        U_post = U_end.reshape(bh, 8, 4)
        gU = gU_in.reshape(bh, 8, 4)
        gx_list: List[Optional[torch.Tensor]] = [None] * clen
        gw_list: List[Optional[torch.Tensor]] = [None] * clen

        for t in range(clen - 1, -1, -1):
            w_t = w_ch[:, t].reshape(bh, 8, 3)
            q_inv = _tq_cayley(-w_t)
            U_prev = _tq_mul(q_inv, U_post)
            xs = x_ch[:, t].reshape(bh, 8, 4)
            gxt4 = gxt_tot[:, t].reshape(bh, 8, 4)
            gx_rot = _tq_mul(U_post, gxt4).reshape(bh, 32)
            gx_list[t] = (gx_geo_bh[:, t] + gx_rot).reshape(bsz, d_model)

            gU = gU + _tq_mul(xs, _tq_conj(gxt4))
            gq = _tq_mul(gU, _tq_conj(U_prev))
            gU = _tq_mul(q_inv, gU)
            gw_list[t] = _tq_cayley_pullback(w_t, gq).reshape(bsz, n_heads, 8, 3)
            U_post = U_prev

        gx_ch = torch.stack([item for item in gx_list if item is not None], dim=1)
        gw_ch = torch.stack([item for item in gw_list if item is not None], dim=1)
        return gx_ch, gw_ch, gW_out, gS.reshape(bsz, n_heads, 32, 32), gU.reshape(bsz, n_heads, 8, 4)

    def _validate_chunk32_params(
        x_seq: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        chunk_size: int,
    ) -> Tuple[int, int, int]:
        require_ok(
            isinstance(chunk_size, int) and not isinstance(chunk_size, bool) and chunk_size > 0,
            f"chunk_size must be a positive int, got {chunk_size!r}",
            code="BAD_CHUNK",
        )
        require_ok(
            x_seq.ndim == 3
            and x_seq.shape[1] > 0
            and x_seq.shape[2] > 0
            and x_seq.shape[2] % 32 == 0
            and x_seq.shape[1] % chunk_size == 0,
            "Expected [B, T, D] with D>0, D%32==0, T>0, T%C==0",
            code="BAD_SEQ_SHAPE",
        )
        require_ok(
            all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
                for v in (cayley_t, gamma, eta, eps)
            )
            and float(eps) > 0.0,
            "cayley_t, gamma, eta, eps must be finite scalars with eps > 0",
            code="BAD_PARAM",
        )
        return int(x_seq.shape[0]), int(x_seq.shape[1]), int(x_seq.shape[2])

    @torch.library.custom_op("holosphere_s117::chunk32_fwd", mutates_args=())
    def _op_chunk32_fwd(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        chunk_size: int,
        use_choice_b: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _bsz, seq_len, _d_model = _validate_chunk32_params(x_seq, cayley_t, gamma, eta, eps, chunk_size)
        n_chunks = seq_len // chunk_size
        t2 = cayley_t * cayley_t
        c_st = (1.0 - t2) / (1.0 + t2)
        s_st = (2.0 * cayley_t) / (1.0 + t2)

        S_ckpts: List[torch.Tensor] = []
        U_ckpts_end: List[torch.Tensor] = []
        y_chunks: List[torch.Tensor] = []
        S_cur = S_0
        U_cur = U_0
        for ch in range(n_chunks):
            t0 = ch * chunk_size
            t1 = t0 + chunk_size
            S_ckpts.append(S_cur)
            y_c, _, _, S_cur, U_cur = _chunk_fwd_btd_torch(
                x_seq[:, t0:t1],
                w_su2[:, t0:t1],
                W_q,
                S_cur,
                U_cur,
                c_st=c_st,
                s_st=s_st,
                gamma=gamma,
                eta=eta,
                eps=eps,
            )
            U_ckpts_end.append(U_cur)
            y_chunks.append(y_c)
        return (
            torch.cat(y_chunks, dim=1),
            S_cur,
            U_cur,
            torch.stack(S_ckpts, dim=0),
            torch.stack(U_ckpts_end, dim=0),
        )

    @torch.library.register_fake("holosphere_s117::chunk32_fwd")
    def _op_chunk32_fwd_fake(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        chunk_size: int,
        use_choice_b: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _bsz, seq_len, _d_model = _validate_chunk32_params(x_seq, cayley_t, gamma, eta, eps, chunk_size)
        n_chunks = seq_len // chunk_size
        return (
            torch.empty_like(x_seq),
            torch.empty_like(S_0),
            torch.empty_like(U_0),
            x_seq.new_empty((n_chunks, *S_0.shape)),
            x_seq.new_empty((n_chunks, *U_0.shape)),
        )

    @torch.library.custom_op("holosphere_s117::chunk32_bwd", mutates_args=())
    def _op_chunk32_bwd(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_ckpts: torch.Tensor,
        U_ckpts_end: torch.Tensor,
        grad_y: torch.Tensor,
        grad_S_end: torch.Tensor,
        grad_U_end: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        chunk_size: int,
        use_choice_b: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _seq_len, d_model = x_seq.shape
        n_heads = d_model // 32
        bh = bsz * n_heads
        n_chunks = S_ckpts.shape[0]
        t2 = cayley_t * cayley_t
        c_st = (1.0 - t2) / (1.0 + t2)
        s_st = (2.0 * cayley_t) / (1.0 + t2)

        W_bh = W_q.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, 32, 32) if W_q.ndim == 2 else W_q.unsqueeze(0).expand(bsz, n_heads, 32, 32)
        W_flat = W_bh.reshape(bh, 32, 32)

        grad_x_chunks: List[Optional[torch.Tensor]] = [None] * n_chunks
        grad_w_chunks: List[Optional[torch.Tensor]] = [None] * n_chunks
        grad_W_acc = torch.zeros_like(W_q)
        gS = grad_S_end
        gU = grad_U_end

        for ch in range(n_chunks - 1, -1, -1):
            t0 = ch * chunk_size
            t1 = t0 + chunk_size
            x_ch = x_seq[:, t0:t1]
            w_ch = w_su2[:, t0:t1]
            S_0_ch = S_ckpts[ch]
            U_end_ch = U_ckpts_end[ch]

            U_p = U_end_ch.reshape(bh, 8, 4)
            xts_rev: List[torch.Tensor] = []
            for t in range(chunk_size - 1, -1, -1):
                xt_t = _tq_mul(_tq_conj(U_p), x_ch[:, t].reshape(bh, 8, 4)).reshape(bh, 32)
                xts_rev.append(xt_t)
                U_p = _tq_mul(_tq_cayley(-w_ch[:, t].reshape(bh, 8, 3)), U_p)
            xt_bh = torch.stack(list(reversed(xts_rev)), dim=1)
            zs_bh = torch.bmm(xt_bh, W_flat.transpose(-1, -2))
            xt_ch = xt_bh.reshape(bsz, n_heads, chunk_size, 32).transpose(1, 2).reshape(bsz, chunk_size, d_model)
            zs_ch = zs_bh.reshape(bsz, n_heads, chunk_size, 32).transpose(1, 2).reshape(bsz, chunk_size, d_model)

            gx_c, gw_c, gW_c, gS, gU = _chunk_3stage_bwd_woodbury_btd_torch(
                x_ch,
                xt_ch,
                zs_ch,
                w_ch,
                W_q,
                S_0_ch,
                U_end_ch,
                grad_y[:, t0:t1],
                gS,
                gU,
                c_st=c_st,
                s_st=s_st,
                gamma=gamma,
                eta=eta,
                eps=eps,
                use_choice_b=use_choice_b,
            )
            grad_x_chunks[ch] = gx_c
            grad_w_chunks[ch] = gw_c
            grad_W_acc = grad_W_acc + gW_c

        return (
            torch.cat([c for c in grad_x_chunks if c is not None], dim=1),
            torch.cat([c for c in grad_w_chunks if c is not None], dim=1),
            grad_W_acc,
            gS,
            gU,
        )

    @torch.library.register_fake("holosphere_s117::chunk32_bwd")
    def _op_chunk32_bwd_fake(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_ckpts: torch.Tensor,
        U_ckpts_end: torch.Tensor,
        grad_y: torch.Tensor,
        grad_S_end: torch.Tensor,
        grad_U_end: torch.Tensor,
        cayley_t: float,
        gamma: float,
        eta: float,
        eps: float,
        chunk_size: int,
        use_choice_b: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty_like(x_seq),
            torch.empty_like(w_su2),
            torch.empty_like(W_q),
            torch.empty_like(grad_S_end),
            torch.empty_like(grad_U_end),
        )

    def _setup_chunk32_context(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, ...]) -> None:
        x_seq, w_su2, W_q, _S_0, _U_0, cayley_t, gamma, eta, eps, chunk_size, use_choice_b = inputs
        _y, _Se, _Ue, S_ckpts, U_ckpts_end = output
        ctx.save_for_backward(x_seq, w_su2, W_q, S_ckpts, U_ckpts_end)
        ctx.params = (cayley_t, gamma, eta, eps, chunk_size, use_choice_b)

    def _chunk32_autograd_backward(
        ctx: Any,
        grad_y: torch.Tensor,
        grad_S_end: torch.Tensor,
        grad_U_end: torch.Tensor,
        _grad_S_ckpts: Optional[torch.Tensor],
        _grad_U_ckpts_end: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        x_seq, w_su2, W_q, S_ckpts, U_ckpts_end = ctx.saved_tensors
        cayley_t, gamma, eta, eps, chunk_size, use_choice_b = ctx.params
        gx, gw, gW, gS0, gU0 = torch.ops.holosphere_s117.chunk32_bwd(
            x_seq.contiguous(),
            w_su2.contiguous(),
            W_q.contiguous(),
            S_ckpts.contiguous(),
            U_ckpts_end.contiguous(),
            grad_y.contiguous(),
            grad_S_end.contiguous(),
            grad_U_end.contiguous(),
            cayley_t,
            gamma,
            eta,
            eps,
            chunk_size,
            use_choice_b,
        )
        return gx, gw, gW, gS0, gU0, None, None, None, None, None, None

    torch.library.register_autograd(
        "holosphere_s117::chunk32_fwd",
        _chunk32_autograd_backward,
        setup_context=_setup_chunk32_context,
    )

    def chunk32_holosphere_layer(
        x_seq: torch.Tensor,
        w_su2: torch.Tensor,
        W_q: torch.Tensor,
        S_0: torch.Tensor,
        U_0: torch.Tensor,
        *,
        cayley_t: float = 0.25,
        gamma: float = 0.90,
        eta: float = 0.15,
        eps: float = 1e-12,
        chunk_size: int = 32,
        use_choice_b: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``torch.compile(fullgraph=True)``-safe wrapper with explicit ``x.contiguous()`` stride guard."""
        _validate_chunk32_params(x_seq, cayley_t, gamma, eta, eps, chunk_size)
        y, S_end, U_end, _, _ = torch.ops.holosphere_s117.chunk32_fwd(
            x_seq.contiguous(),
            w_su2.contiguous(),
            W_q.contiguous(),
            S_0.contiguous(),
            U_0.contiguous(),
            cayley_t,
            gamma,
            eta,
            eps,
            chunk_size,
            use_choice_b,
        )
        return y, S_end, U_end


def validate_gradcheck_self_test_report(report: Dict[str, Any]) -> None:
    """Fail-closed numerical parity and gradcheck gate for ``run_cpu_gradcheck_self_test``."""
    require_ok(
        report["choice_a_4224B"]["worst_err_vs_autograd"] < 1e-13
        and report["choice_b_6336B"]["worst_err_vs_autograd"] < 1e-13,
        "Choice A/B autograd worst drift >= 1e-13",
        code="W10_OBS_60_DRIFT",
    )
    require_ok(
        report["gradcheck_choice_a_pass"] is True and report["gradcheck_choice_b_pass"] is True,
        "torch.autograd.gradcheck failed",
        code="TRACK_A_GRADCHECK_FAIL",
    )
    require_ok(
        report["torch_compile_fullgraph_worst_err"] < 1e-13,
        "torch.compile(fullgraph=True) drift >= 1e-13",
        code="COMPILE_FULLGRAPH_DRIFT",
    )
    require_ok(
        report["h256_embed_ortho_max_err"] < 1e-5,
        "H_256 embed ortho err >= 1e-5",
        code="EMBED_ORTHO",
    )


def run_cpu_gradcheck_self_test() -> Dict[str, Any]:
    """Verify Choice A & Choice B closed-form Woodbury backward + ``torch.compile(fullgraph=True)`` in FP64."""
    _require_torch()
    gen = torch.Generator(device="cpu").manual_seed(11799)
    bsz, seq_len, n_heads = 1, 64, 2
    d_model = n_heads * 32

    x_raw = torch.randn((bsz, seq_len, n_heads, 32), generator=gen, dtype=torch.float64)
    x_seq = (x_raw / x_raw.norm(dim=-1, keepdim=True)).reshape(bsz, seq_len, d_model).detach().requires_grad_(True)
    w_su2 = (torch.rand((bsz, seq_len, n_heads, 8, 3), generator=gen, dtype=torch.float64) * 1.4 - 0.7).detach().requires_grad_(True)
    W_q = (torch.randn((n_heads, 32, 32), generator=gen, dtype=torch.float64) * 0.15).detach().requires_grad_(True)
    S_0 = (torch.randn((bsz, n_heads, 32, 32), generator=gen, dtype=torch.float64) * 0.12).detach().requires_grad_(True)
    u_raw = torch.randn((bsz, n_heads, 8, 4), generator=gen, dtype=torch.float64)
    U_0 = (u_raw / u_raw.norm(dim=-1, keepdim=True)).detach().requires_grad_(True)

    gy = torch.randn((bsz, seq_len, d_model), generator=gen, dtype=torch.float64)
    gSe = torch.randn((bsz, n_heads, 32, 32), generator=gen, dtype=torch.float64)
    gUe = torch.randn((bsz, n_heads, 8, 4), generator=gen, dtype=torch.float64)

    c_st = (1.0 - 0.25**2) / (1.0 + 0.25**2)
    s_st = (2.0 * 0.25) / (1.0 + 0.25**2)
    y_r, _, _, Se_r, Ue_r = _chunk_fwd_btd_torch(
        x_seq, w_su2, W_q, S_0, U_0, c_st=c_st, s_st=s_st, gamma=0.90, eta=0.15, eps=1e-12
    )
    loss_r = (y_r * gy).sum() + (Se_r * gSe).sum() + (Ue_r * gUe).sum()
    grads_r = torch.autograd.grad(loss_r, (x_seq, w_su2, W_q, S_0, U_0))

    names = ("grad_x", "grad_w_su2", "grad_W_q", "grad_S_0", "grad_U_0")
    choice_reports: Dict[str, Any] = {}

    for label, use_b in (("choice_a_4224B", False), ("choice_b_6336B", True)):
        y_c, Se_c, Ue_c = chunk32_holosphere_layer(
            x_seq, w_su2, W_q, S_0, U_0, chunk_size=32, use_choice_b=use_b
        )
        loss_c = (y_c * gy).sum() + (Se_c * gSe).sum() + (Ue_c * gUe).sum()
        grads_c = torch.autograd.grad(loss_c, (x_seq, w_su2, W_q, S_0, U_0))
        errs = {k: float((a - b).abs().max().item()) for k, a, b in zip(names, grads_c, grads_r)}
        worst = max(errs.values())
        choice_reports[label] = {
            "max_err_vs_autograd": {k: float(f"{v:.3e}") for k, v in errs.items()},
            "worst_err_vs_autograd": float(f"{worst:.3e}"),
        }

    gc_a = torch.autograd.gradcheck(
        lambda xs, ws, Wq, S0, U0: chunk32_holosphere_layer(xs, ws, Wq, S0, U0, chunk_size=32, use_choice_b=False),
        (x_seq, w_su2, W_q, S_0, U_0),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
        fast_mode=True,
    )
    gc_b = torch.autograd.gradcheck(
        lambda xs, ws, Wq, S0, U0: chunk32_holosphere_layer(xs, ws, Wq, S0, U0, chunk_size=32, use_choice_b=True),
        (x_seq, w_su2, W_q, S_0, U_0),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
        fast_mode=True,
    )

    def _compiled_target(xs: torch.Tensor, ws: torch.Tensor, Wq: torch.Tensor, S0: torch.Tensor, U0: torch.Tensor) -> torch.Tensor:
        y_o, Se_o, Ue_o = chunk32_holosphere_layer(xs, ws, Wq, S0, U0, chunk_size=32, use_choice_b=True)
        return (y_o * gy).sum() + (Se_o * gSe).sum() + (Ue_o * gUe).sum()

    compiled_fn = torch.compile(_compiled_target, fullgraph=True, backend="aot_eager")
    loss_comp = compiled_fn(x_seq, w_su2, W_q, S_0, U_0)
    grads_comp = torch.autograd.grad(loss_comp, (x_seq, w_su2, W_q, S_0, U_0))
    compile_worst = max(float((a - b).abs().max().item()) for a, b in zip(grads_comp, grads_r))

    emb = h256_bent4_embed_torch(torch.arange(256, dtype=torch.uint8), d_model=256)
    gram = emb @ emb.T
    eye = torch.eye(256, dtype=torch.float32) * 8.0
    embed_ortho_err = float((gram - eye).abs().max().item())

    report = {
        "layout": {"B": bsz, "T": seq_len, "D": d_model, "H": n_heads, "stride_t": d_model},
        "choice_a_4224B": choice_reports["choice_a_4224B"],
        "choice_b_6336B": choice_reports["choice_b_6336B"],
        "gradcheck_choice_a_pass": bool(gc_a),
        "gradcheck_choice_b_pass": bool(gc_b),
        "torch_compile_fullgraph_aot_eager_pass": True,
        "torch_compile_fullgraph_worst_err": float(f"{compile_worst:.3e}"),
        "h256_embed_ortho_max_err": float(f"{embed_ortho_err:.3e}"),
    }
    validate_gradcheck_self_test_report(report)
    return report


__all__ = [
    "CU_FILENAME",
    "DEFAULT_CSRC_DIR",
    "EXPECTED_S117_KERNEL_SYMBOLS",
    "EXPECTED_V3_KERNEL_SYMBOLS",
    "HoloSphereError",
    "PTXAS_FILENAME",
    "TORCH_LIBRARY_CPP_BLUEPRINT",
    "V3_CU_FILENAME",
    "V3_PTXAS_FILENAME",
    "_HAS_TORCH",
    "estimate_t8192_vram_budget",
    "gumbel_23bit_finite_f32",
    "gumbel_23bit_u_f32",
    "h256_bent4_byte_inject_s8",
    "h256_bent4_embed_torch",
    "load_s117_cuda_lib",
    "pow_gamma_u5_py",
    "run_cpu_gradcheck_self_test",
    "validate_gradcheck_self_test_report",
    "verify_gumbel_23bit_finite",
    "verify_h256_bent4_exhaustive",
    "verify_s117_ptxas_receipt",
    "verify_v3_frontier_ptxas_receipt",
]
if _HAS_TORCH:
    __all__.append("chunk32_holosphere_layer")
