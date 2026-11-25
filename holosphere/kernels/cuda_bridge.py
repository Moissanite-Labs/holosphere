"""Bare-metal Ampere sm_86 CUDA/PTX ctypes bridge for holosphere.kernels (Track S114 V2).

Zero GPU initialization on import or under ``CUDA_VISIBLE_DEVICES=""``.
Binds all 27 zero-spill ``extern "C"`` entrypoints from ``csrc/su2_holonomy_scan_sm86.cu``:
  - V1 fiber-major ``[F, T, C]`` holonomy scan, untransport, SLERP/Hopf, Hopf-S^3, & A16 dequant (9 kernels)
  - V2 zero-permute strided BHTD holonomy scan in FP32 & direct BF16 for both ``su2_exp`` & SFU-free
    ``cayley_su2`` (8 kernels)
  - V2 direct-BF16 spherical SLERP + Safe-Hopf with ``m2 <= eps^2`` antipodal guard (2 kernels)
  - V2 Beltrami ``[C, K] = 0`` ``Cayley2x2`` & zero-HBM-epilogue ``Hopf-S^3 + Cayley2x2`` in FP32/BF16 (8 kernels)
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Any

import torch

from holosphere.errors import require_ok

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_DEFAULT_LIB_NAMES = ("libsu2_holonomy_sm86.so", "libsu2_holonomy_scan_sm86.so", "su2_holonomy_scan_sm86.so")
_CACHED_LIB: ctypes.CDLL | None = None
_CACHED_TRIED = False


def find_cuda_library() -> Path | None:
    """Return path to compiled sm_86 shared library if staged locally or via HOLOSPHERE_SM86_LIB."""
    for env_key in ("HOLOSPHERE_SM86_LIB", "SILISPHERE_SM86_LIB"):
        env_val = os.environ.get(env_key)
        if env_val:
            cand = Path(env_val)
            if cand.is_file():
                return cand
    for name in _DEFAULT_LIB_NAMES:
        cand = _CSRC_DIR / name
        if cand.is_file():
            return cand
    return None


def _configure_signatures(lib: ctypes.CDLL) -> None:
    vp = ctypes.c_void_p
    ci = ctypes.c_int
    ci64 = ctypes.c_int64
    cf = ctypes.c_float
    # V1 core 9 kernels
    lib.silisphere_su2_holonomy_fwd_f32.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, ci, ci, cf, ci, cf, vp]
    lib.silisphere_su2_holonomy_fwd_f32.restype = ci
    lib.silisphere_su2_holonomy_bwd_f32.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, ci, ci, cf, ci, cf, vp]
    lib.silisphere_su2_holonomy_bwd_f32.restype = ci
    lib.silisphere_su2_untransport_fwd_f32.argtypes = [vp, vp, vp, ci64, vp]
    lib.silisphere_su2_untransport_fwd_f32.restype = ci
    lib.silisphere_su2_untransport_bwd_f32.argtypes = [vp, vp, vp, vp, vp, ci64, vp]
    lib.silisphere_su2_untransport_bwd_f32.restype = ci
    lib.silisphere_spherical_slerp_hopf_fwd_f32.argtypes = [vp, vp, vp, vp, ci, ci, cf, cf, cf, cf, vp]
    lib.silisphere_spherical_slerp_hopf_fwd_f32.restype = ci
    lib.silisphere_spherical_slerp_hopf_bwd_f32.argtypes = [vp, vp, vp, vp, vp, vp, vp, ci, ci, cf, cf, cf, cf, vp]
    lib.silisphere_spherical_slerp_hopf_bwd_f32.restype = ci
    lib.silisphere_hopf_s3_residual_fwd_f32.argtypes = [vp, vp, vp, ci64, vp]
    lib.silisphere_hopf_s3_residual_fwd_f32.restype = ci
    lib.silisphere_hopf_s3_residual_bwd_f32.argtypes = [vp, vp, vp, vp, vp, ci64, vp]
    lib.silisphere_hopf_s3_residual_bwd_f32.restype = ci
    if hasattr(lib, "holosphere_siliquant_a16_dequant_hopf_f32"):
        lib.holosphere_siliquant_a16_dequant_hopf_f32.argtypes = [vp, vp, vp, ci64, cf, cf, vp]
        lib.holosphere_siliquant_a16_dequant_hopf_f32.restype = ci
    # V2 Zero-Permute Strided BHTD Holonomy (FP32 & BF16, su2_exp & cayley_su2)
    bhtd_fwd_args = [
        vp, vp, vp, vp, vp, vp, vp, vp,
        ci, ci, ci, ci,
        ci64, ci64, ci64, ci64, ci64, ci64,
        cf, ci, cf, vp,
    ]
    bhtd_bwd_args = [
        vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
        ci, ci, ci, ci,
        ci64, ci64, ci64, ci64, ci64, ci64,
        cf, ci, cf, vp,
    ]
    for sfx in ("f32", "bf16", "cayley_f32", "cayley_bf16"):
        fwd_fn = getattr(lib, f"silisphere_su2_holonomy_bhtd_fwd_{sfx}", None)
        bwd_fn = getattr(lib, f"silisphere_su2_holonomy_bhtd_bwd_{sfx}", None)
        if fwd_fn is not None:
            fwd_fn.argtypes = bhtd_fwd_args
            fwd_fn.restype = ci
        if bwd_fn is not None:
            bwd_fn.argtypes = bhtd_bwd_args
            bwd_fn.restype = ci
    # V2 Direct-BF16 Spherical SLERP + Safe-Hopf
    if hasattr(lib, "silisphere_spherical_slerp_hopf_fwd_bf16"):
        lib.silisphere_spherical_slerp_hopf_fwd_bf16.argtypes = [vp, vp, vp, vp, ci, ci, cf, cf, cf, cf, vp]
        lib.silisphere_spherical_slerp_hopf_fwd_bf16.restype = ci
        lib.silisphere_spherical_slerp_hopf_bwd_bf16.argtypes = [vp, vp, vp, vp, vp, vp, vp, ci, ci, cf, cf, cf, cf, vp]
        lib.silisphere_spherical_slerp_hopf_bwd_bf16.restype = ci
    # V2 Beltrami Cayley2x2 & Fused Hopf-S^3 + Cayley2x2 (FP32 & BF16)
    for sfx in ("f32", "bf16"):
        for nm, args in (
            (f"holosphere_cayley2x2_fwd_{sfx}", [vp, vp, vp, ci64, vp]),
            (f"holosphere_cayley2x2_bwd_{sfx}", [vp, vp, vp, vp, vp, ci64, vp]),
            (f"holosphere_hopf_s3_cayley2x2_fwd_{sfx}", [vp, vp, vp, vp, ci64, vp]),
            (f"holosphere_hopf_s3_cayley2x2_bwd_{sfx}", [vp, vp, vp, vp, vp, vp, vp, ci64, vp]),
        ):
            fn = getattr(lib, nm, None)
            if fn is not None:
                fn.argtypes = args
                fn.restype = ci


def load_cuda_library() -> ctypes.CDLL | None:
    """Lazily load and configure the staged sm_86 shared library via ctypes."""
    global _CACHED_LIB, _CACHED_TRIED
    if _CACHED_TRIED:
        return _CACHED_LIB
    lib_path = find_cuda_library()
    if lib_path is not None:
        lib = ctypes.CDLL(str(lib_path))
        _configure_signatures(lib)
        _CACHED_LIB = lib
    _CACHED_TRIED = True
    return _CACHED_LIB


def is_cuda_kernel_available() -> bool:
    """Return True iff the compiled sm_86 library is staged AND a CUDA device is active."""
    if find_cuda_library() is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ptr(t: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if t is None else int(t.data_ptr()))


def _stream_ptr(device: torch.device) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(torch.cuda.current_stream(device).cuda_stream))


def _reduce_broadcast_grad(grad: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    out = grad
    while out.ndim > len(target_shape):
        out = out.sum(dim=0)
    for dim, size in enumerate(target_shape):
        if size == 1 and out.shape[dim] != 1:
            out = out.sum(dim=dim, keepdim=True)
    return out


def _to_fiber_major_4(x: torch.Tensor, n_blocks: int) -> torch.Tensor:
    bsz, n_heads, seq_len, _ = x.shape
    return x.reshape(bsz, n_heads, seq_len, n_blocks, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 4)


def _from_fiber_major_4(x_f: torch.Tensor, bsz: int, n_heads: int, seq_len: int, n_blocks: int) -> torch.Tensor:
    return x_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz, n_heads, seq_len, 4 * n_blocks)


def _match_q4_layout(t: torch.Tensor, bsz: int, n_heads: int, seq_len: int, d_head: int) -> tuple[torch.Tensor, bool]:
    st = t.stride()
    if t.is_contiguous():
        return t, False
    if st == (seq_len * n_heads * d_head, d_head, n_heads * d_head, 1):
        return t, True
    return t.contiguous(), False


def _match_raw_layout(r: torch.Tensor, bsz: int, n_heads: int, seq_len: int, n_blocks: int) -> tuple[torch.Tensor, bool]:
    st = r.stride()
    if r.is_contiguous():
        return r, False
    if st == (seq_len * n_heads * n_blocks * 3, n_blocks * 3, n_heads * n_blocks * 3, 3, 1):
        return r, True
    return r.contiguous(), False


class _CudaSU2HolonomyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, raw_f: torch.Tensor, q_f: torch.Tensor, k_f: torch.Tensor, v_f: torch.Tensor | None, angle_scale: float, use_tanh: bool, eps: float):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        in_dtype = q_f.dtype
        raw_w = raw_f.float().contiguous()
        q_w = q_f.float().contiguous()
        k_w = k_f.float().contiguous()
        v_w = v_f.float().contiguous() if v_f is not None else None
        n_fibers, seq_len, _ = raw_w.shape
        u_w = torch.empty((n_fibers, seq_len, 4), device=raw_w.device, dtype=torch.float32)
        qo_w = torch.empty_like(u_w)
        ko_w = torch.empty_like(u_w)
        vo_w = torch.empty_like(u_w) if v_w is not None else None
        rc = lib.silisphere_su2_holonomy_fwd_f32(
            _ptr(raw_w), _ptr(q_w), _ptr(k_w), _ptr(v_w),
            _ptr(u_w), _ptr(qo_w), _ptr(ko_w), _ptr(vo_w),
            int(n_fibers), int(seq_len), float(angle_scale), int(1 if use_tanh else 0),
            float(eps), _stream_ptr(raw_w.device),
        )
        require_ok(rc == 0, f"silisphere_su2_holonomy_fwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        ctx.angle_scale = angle_scale
        ctx.use_tanh = use_tanh
        ctx.eps = eps
        ctx.in_dtype = in_dtype
        ctx.has_v = v_w is not None
        ctx.save_for_backward(raw_w, u_w, q_w, k_w, v_w if v_w is not None else torch.empty(0, device=raw_w.device))
        v_ret = vo_w.to(in_dtype) if vo_w is not None else torch.empty(0, device=raw_w.device, dtype=in_dtype)
        # Preserve SU(2) holonomy path U in FP32 to prevent BF16 S^3 manifold drift (W10-OBS-36).
        return qo_w.to(in_dtype), ko_w.to(in_dtype), v_ret, u_w

    @staticmethod
    def backward(ctx: Any, dq_out: torch.Tensor | None, dk_out: torch.Tensor | None, dv_out: torch.Tensor | None, du_out: torch.Tensor | None):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        raw_w, u_w, q_w, k_w, v_saved = ctx.saved_tensors
        v_w = v_saved if ctx.has_v else None
        dq_w = dq_out.float().contiguous() if dq_out is not None else torch.zeros_like(q_w)
        dk_w = dk_out.float().contiguous() if dk_out is not None else torch.zeros_like(k_w)
        dv_w = dv_out.float().contiguous() if (ctx.has_v and dv_out is not None and dv_out.numel() > 0) else None
        du_w = du_out.float().contiguous() if du_out is not None else None
        n_fibers, seq_len, _ = raw_w.shape
        d_raw = torch.empty_like(raw_w)
        dq_in = torch.empty_like(q_w)
        dk_in = torch.empty_like(k_w)
        dv_in = torch.empty_like(v_w) if (v_w is not None and dv_w is not None) else None
        rc = lib.silisphere_su2_holonomy_bwd_f32(
            _ptr(raw_w), _ptr(u_w), _ptr(q_w), _ptr(k_w), _ptr(v_w),
            _ptr(du_w), _ptr(dq_w), _ptr(dk_w), _ptr(dv_w),
            _ptr(d_raw), _ptr(dq_in), _ptr(dk_in), _ptr(dv_in),
            int(n_fibers), int(seq_len), float(ctx.angle_scale), int(1 if ctx.use_tanh else 0),
            float(ctx.eps), _stream_ptr(raw_w.device),
        )
        require_ok(rc == 0, f"silisphere_su2_holonomy_bwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        dt = ctx.in_dtype
        return d_raw.to(dt), dq_in.to(dt), dk_in.to(dt), (dv_in.to(dt) if dv_in is not None else None), None, None, None


class _CudaSU2HolonomyBHTDFn(torch.autograd.Function):
    """V2 Zero-Permute Direct FP32/BF16 BHTD Holonomy Scan (supports su2_exp & cayley_su2)."""

    @staticmethod
    def forward(
        ctx: Any,
        omega: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        angle_scale: float,
        use_tanh: bool,
        eps: float,
        map_mode: str,
    ):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        bsz, n_heads, seq_len, d_head = q.shape
        n_blocks = d_head // 4
        in_dtype = q.dtype
        use_bf16 = in_dtype == torch.bfloat16
        io_dtype = torch.bfloat16 if use_bf16 else torch.float32

        q_io = q if q.dtype == io_dtype else q.to(io_dtype)
        k_io = k if k.dtype == io_dtype else k.to(io_dtype)
        v_io = (v if v.dtype == io_dtype else v.to(io_dtype)) if v is not None else None
        om_io = omega if omega.dtype == io_dtype else omega.to(io_dtype)

        q_io, q_tr = _match_q4_layout(q_io, bsz, n_heads, seq_len, d_head)
        k_io, k_tr = _match_q4_layout(k_io, bsz, n_heads, seq_len, d_head)
        v_tr = q_tr
        if v_io is not None:
            v_io, v_tr = _match_q4_layout(v_io, bsz, n_heads, seq_len, d_head)
        if k_tr != q_tr or v_tr != q_tr:
            q_io, k_io = q_io.contiguous(), k_io.contiguous()
            if v_io is not None:
                v_io = v_io.contiguous()
            q_tr = False
        om_io, om_tr = _match_raw_layout(om_io, bsz, n_heads, seq_len, n_blocks)

        require_ok(
            q_io.stride(0) % 4 == 0 and q_io.stride(1) % 4 == 0 and q_io.stride(2) % 4 == 0,
            f"q/k/v strides must be divisible by 4, got {q_io.stride()}",
            code="CUDA_STRIDE",
        )
        require_ok(
            om_io.stride(0) % 3 == 0 and om_io.stride(1) % 3 == 0 and om_io.stride(2) % 3 == 0,
            f"omega strides must be divisible by 3, got {om_io.stride()}",
            code="CUDA_STRIDE",
        )
        in_sb, in_sh, in_st = q_io.stride(0) // 4, q_io.stride(1) // 4, q_io.stride(2) // 4
        raw_sb, raw_sh, raw_st = om_io.stride(0) // 3, om_io.stride(1) // 3, om_io.stride(2) // 3

        u_out = torch.empty((bsz, n_heads, seq_len, n_blocks, 4), device=q_io.device, dtype=torch.float32)
        q_out = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
        k_out = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
        v_out = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype) if v_io is not None else None

        sfx = ("cayley_" if map_mode == "cayley" else "") + ("bf16" if use_bf16 else "f32")
        fwd_fn = getattr(lib, f"silisphere_su2_holonomy_bhtd_fwd_{sfx}")
        rc = fwd_fn(
            _ptr(om_io), _ptr(q_io), _ptr(k_io), _ptr(v_io),
            _ptr(u_out), _ptr(q_out), _ptr(k_out), _ptr(v_out),
            int(bsz), int(n_heads), int(seq_len), int(n_blocks),
            int(in_sb), int(in_sh), int(in_st),
            int(raw_sb), int(raw_sh), int(raw_st),
            float(angle_scale), int(1 if use_tanh else 0), float(eps),
            _stream_ptr(q_io.device),
        )
        require_ok(rc == 0, f"silisphere_su2_holonomy_bhtd_fwd_{sfx} failed rc={rc}", code="CUDA_LAUNCH")

        ctx.angle_scale = angle_scale
        ctx.use_tanh = use_tanh
        ctx.eps = eps
        ctx.map_mode = map_mode
        ctx.in_dtype = in_dtype
        ctx.io_dtype = io_dtype
        ctx.use_bf16 = use_bf16
        ctx.has_v = v_io is not None
        ctx.q_tr = q_tr
        ctx.om_tr = om_tr
        ctx.strides = (in_sb, in_sh, in_st, raw_sb, raw_sh, raw_st)
        ctx.save_for_backward(om_io, u_out, q_io, k_io, v_io if v_io is not None else torch.empty(0, device=q_io.device))
        v_ret = v_out if v_out is not None else torch.empty(0, device=q_io.device, dtype=io_dtype)
        return q_out.to(in_dtype), k_out.to(in_dtype), v_ret.to(in_dtype), u_out

    @staticmethod
    def backward(ctx: Any, dq_out: torch.Tensor | None, dk_out: torch.Tensor | None, dv_out: torch.Tensor | None, du_out: torch.Tensor | None):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        om_io, u_out, q_io, k_io, v_saved = ctx.saved_tensors
        v_io = v_saved if ctx.has_v else None
        bsz, n_heads, seq_len, d_head = q_io.shape
        n_blocks = d_head // 4
        io_dtype = ctx.io_dtype
        in_sb, in_sh, in_st, raw_sb, raw_sh, raw_st = ctx.strides

        dq_o = dq_out.to(io_dtype).contiguous() if dq_out is not None else torch.zeros((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
        dk_o = dk_out.to(io_dtype).contiguous() if dk_out is not None else torch.zeros((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
        dv_o = dv_out.to(io_dtype).contiguous() if (ctx.has_v and dv_out is not None and dv_out.numel() > 0) else None
        du_e = du_out.to(torch.float32).contiguous() if du_out is not None else None

        if ctx.q_tr:
            dq_in = torch.empty((bsz, seq_len, n_heads, d_head), device=q_io.device, dtype=io_dtype).transpose(1, 2)
            dk_in = torch.empty((bsz, seq_len, n_heads, d_head), device=q_io.device, dtype=io_dtype).transpose(1, 2)
            dv_in = torch.empty((bsz, seq_len, n_heads, d_head), device=q_io.device, dtype=io_dtype).transpose(1, 2) if (v_io is not None and dv_o is not None) else None
        else:
            dq_in = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
            dk_in = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype)
            dv_in = torch.empty((bsz, n_heads, seq_len, d_head), device=q_io.device, dtype=io_dtype) if (v_io is not None and dv_o is not None) else None
        if ctx.om_tr:
            d_raw = torch.empty((bsz, seq_len, n_heads, n_blocks, 3), device=q_io.device, dtype=io_dtype).transpose(1, 2)
        else:
            d_raw = torch.empty((bsz, n_heads, seq_len, n_blocks, 3), device=q_io.device, dtype=io_dtype)

        sfx = ("cayley_" if ctx.map_mode == "cayley" else "") + ("bf16" if ctx.use_bf16 else "f32")
        bwd_fn = getattr(lib, f"silisphere_su2_holonomy_bhtd_bwd_{sfx}")
        rc = bwd_fn(
            _ptr(om_io), _ptr(u_out), _ptr(q_io), _ptr(k_io), _ptr(v_io),
            _ptr(du_e), _ptr(dq_o), _ptr(dk_o), _ptr(dv_o),
            _ptr(d_raw), _ptr(dq_in), _ptr(dk_in), _ptr(dv_in),
            int(bsz), int(n_heads), int(seq_len), int(n_blocks),
            int(in_sb), int(in_sh), int(in_st),
            int(raw_sb), int(raw_sh), int(raw_st),
            float(ctx.angle_scale), int(1 if ctx.use_tanh else 0), float(ctx.eps),
            _stream_ptr(q_io.device),
        )
        require_ok(rc == 0, f"silisphere_su2_holonomy_bhtd_bwd_{sfx} failed rc={rc}", code="CUDA_LAUNCH")
        dt = ctx.in_dtype
        return (
            d_raw if d_raw.dtype == dt else d_raw.to(dt),
            dq_in if dq_in.dtype == dt else dq_in.to(dt),
            dk_in if dk_in.dtype == dt else dk_in.to(dt),
            (dv_in if dv_in.dtype == dt else dv_in.to(dt)) if dv_in is not None else None,
            None,
            None,
            None,
            None,
        )


class _CudaSU2UntransportFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, o_q: torch.Tensor, pi_q: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        in_dtype = o_q.dtype
        o_w = o_q.float().contiguous()
        u_w = pi_q.float().contiguous()
        out_w = torch.empty_like(o_w)
        rc = lib.silisphere_su2_untransport_fwd_f32(_ptr(u_w), _ptr(o_w), _ptr(out_w), int(o_w.numel() // 4), _stream_ptr(o_w.device))
        require_ok(rc == 0, f"silisphere_su2_untransport_fwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        ctx.in_dtype = in_dtype
        ctx.u_dtype = pi_q.dtype
        ctx.save_for_backward(o_w, u_w)
        return out_w.to(in_dtype)

    @staticmethod
    def backward(ctx: Any, do_out: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        o_w, u_w = ctx.saved_tensors
        do_w = do_out.float().contiguous()
        do_in = torch.empty_like(o_w)
        du_out = torch.empty_like(u_w)
        rc = lib.silisphere_su2_untransport_bwd_f32(
            _ptr(u_w), _ptr(o_w), _ptr(do_w), _ptr(do_in), _ptr(du_out),
            int(o_w.numel() // 4), _stream_ptr(o_w.device),
        )
        require_ok(rc == 0, f"silisphere_su2_untransport_bwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        return do_in.to(ctx.in_dtype), du_out.to(ctx.u_dtype)


class _CudaHopfS3ResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, delta: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        in_dtype = h.dtype
        h_w = h.float().contiguous()
        d_w = delta.float().contiguous()
        out_w = torch.empty_like(h_w)
        rc = lib.silisphere_hopf_s3_residual_fwd_f32(_ptr(h_w), _ptr(d_w), _ptr(out_w), int(h_w.numel() // 4), _stream_ptr(h_w.device))
        require_ok(rc == 0, f"silisphere_hopf_s3_residual_fwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        ctx.in_dtype = in_dtype
        ctx.save_for_backward(h_w, d_w)
        return out_w.to(in_dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        h_w, d_w = ctx.saved_tensors
        go_w = grad_out.float().contiguous()
        gh = torch.empty_like(h_w)
        gd = torch.empty_like(d_w)
        rc = lib.silisphere_hopf_s3_residual_bwd_f32(
            _ptr(h_w), _ptr(d_w), _ptr(go_w), _ptr(gh), _ptr(gd),
            int(h_w.numel() // 4), _stream_ptr(h_w.device),
        )
        require_ok(rc == 0, f"silisphere_hopf_s3_residual_bwd_f32 failed rc={rc}", code="CUDA_LAUNCH")
        return gh.to(ctx.in_dtype), gd.to(ctx.in_dtype)


class _CudaCayley2x2Fn(torch.autograd.Function):
    """V2 Beltrami [C, K] = 0 2x2 Block-Diagonal Cayley SO(d) Rotation (FP32 & direct BF16)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, omega: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        in_dtype = x.dtype
        om_dtype = omega.dtype
        om_orig_shape = tuple(omega.shape)
        n_pairs = x.shape[-1] // 2
        om_full = omega.expand(*x.shape[:-1], n_pairs)
        use_bf16 = in_dtype == torch.bfloat16 and hasattr(lib, "holosphere_cayley2x2_fwd_bf16")
        io_dtype = torch.bfloat16 if use_bf16 else torch.float32
        x_w = x.to(io_dtype).contiguous()
        om_w = om_full.to(io_dtype).contiguous()
        y_w = torch.empty_like(x_w)
        fn = lib.holosphere_cayley2x2_fwd_bf16 if use_bf16 else lib.holosphere_cayley2x2_fwd_f32
        rc = fn(_ptr(x_w), _ptr(om_w), _ptr(y_w), int(x_w.numel() // 4), _stream_ptr(x_w.device))
        require_ok(rc == 0, f"holosphere_cayley2x2_fwd failed rc={rc}", code="CUDA_LAUNCH")
        ctx.use_bf16 = use_bf16
        ctx.in_dtype = in_dtype
        ctx.om_dtype = om_dtype
        ctx.om_orig_shape = om_orig_shape
        ctx.save_for_backward(x_w, om_w)
        return y_w if y_w.dtype == in_dtype else y_w.to(in_dtype)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        x_w, om_w = ctx.saved_tensors
        gy_w = grad_y.to(x_w.dtype).contiguous()
        gx_w = torch.empty_like(x_w)
        gw_full = torch.empty_like(om_w)
        fn = lib.holosphere_cayley2x2_bwd_bf16 if ctx.use_bf16 else lib.holosphere_cayley2x2_bwd_f32
        rc = fn(_ptr(x_w), _ptr(om_w), _ptr(gy_w), _ptr(gx_w), _ptr(gw_full), int(x_w.numel() // 4), _stream_ptr(x_w.device))
        require_ok(rc == 0, f"holosphere_cayley2x2_bwd failed rc={rc}", code="CUDA_LAUNCH")
        gw = _reduce_broadcast_grad(gw_full.float() if ctx.use_bf16 else gw_full, ctx.om_orig_shape)
        return (
            gx_w if gx_w.dtype == ctx.in_dtype else gx_w.to(ctx.in_dtype),
            gw if gw.dtype == ctx.om_dtype else gw.to(ctx.om_dtype),
        )


class _CudaHopfS3Cayley2x2Fn(torch.autograd.Function):
    """V2 Zero-HBM-Epilogue Fused Hopf-S^3 Residual + Beltrami Cayley2x2 Rotation (FP32 & BF16)."""

    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, delta: torch.Tensor, omega: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        in_dtype = h.dtype
        om_dtype = omega.dtype
        om_orig_shape = tuple(omega.shape)
        n_pairs = h.shape[-1] // 2
        om_full = omega.expand(*h.shape[:-1], n_pairs)
        use_bf16 = (
            in_dtype == torch.bfloat16
            and delta.dtype == torch.bfloat16
            and hasattr(lib, "holosphere_hopf_s3_cayley2x2_fwd_bf16")
        )
        io_dtype = torch.bfloat16 if use_bf16 else torch.float32
        h_w = h.to(io_dtype).contiguous()
        d_w = delta.to(io_dtype).contiguous()
        om_w = om_full.to(io_dtype).contiguous()
        out_w = torch.empty_like(h_w)
        fn = lib.holosphere_hopf_s3_cayley2x2_fwd_bf16 if use_bf16 else lib.holosphere_hopf_s3_cayley2x2_fwd_f32
        rc = fn(_ptr(h_w), _ptr(d_w), _ptr(om_w), _ptr(out_w), int(h_w.numel() // 4), _stream_ptr(h_w.device))
        require_ok(rc == 0, f"holosphere_hopf_s3_cayley2x2_fwd failed rc={rc}", code="CUDA_LAUNCH")
        ctx.use_bf16 = use_bf16
        ctx.in_dtype = in_dtype
        ctx.om_dtype = om_dtype
        ctx.om_orig_shape = om_orig_shape
        ctx.save_for_backward(h_w, d_w, om_w)
        return out_w if out_w.dtype == in_dtype else out_w.to(in_dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        lib = load_cuda_library()
        require_ok(lib is not None, "libsu2_holonomy_sm86.so not loaded", code="CUDA_LIB")
        h_w, d_w, om_w = ctx.saved_tensors
        go_w = grad_out.to(h_w.dtype).contiguous()
        gh_w = torch.empty_like(h_w)
        gd_w = torch.empty_like(d_w)
        gw_full = torch.empty_like(om_w)
        fn = lib.holosphere_hopf_s3_cayley2x2_bwd_bf16 if ctx.use_bf16 else lib.holosphere_hopf_s3_cayley2x2_bwd_f32
        rc = fn(
            _ptr(h_w), _ptr(d_w), _ptr(om_w), _ptr(go_w),
            _ptr(gh_w), _ptr(gd_w), _ptr(gw_full),
            int(h_w.numel() // 4), _stream_ptr(h_w.device),
        )
        require_ok(rc == 0, f"holosphere_hopf_s3_cayley2x2_bwd failed rc={rc}", code="CUDA_LAUNCH")
        gw = _reduce_broadcast_grad(gw_full.float() if ctx.use_bf16 else gw_full, ctx.om_orig_shape)
        return (
            gh_w if gh_w.dtype == ctx.in_dtype else gh_w.to(ctx.in_dtype),
            gd_w if gd_w.dtype == ctx.in_dtype else gd_w.to(ctx.in_dtype),
            gw if gw.dtype == ctx.om_dtype else gw.to(ctx.om_dtype),
        )


def cuda_su2_holonomy_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    *,
    angle_scale: float | None = 0.75,
    transport_v: bool = False,
    eps: float = 1e-6,
    map_mode: str = "exp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused sm_86 CUDA SU(2) holonomy scan + covariant (Q, K[, V]) transport (V2 zero-permute fast path)."""
    require_ok(map_mode in ("exp", "cayley"), f"unknown map_mode {map_mode!r}", code="BAD_MAP_MODE")
    require_ok(math.isfinite(eps) and eps > 0.0, f"eps must be finite and > 0, got {eps}", code="BAD_EPS")
    use_tanh = angle_scale is not None
    scale_val = 1.0 if angle_scale is None else float(angle_scale)
    lib = load_cuda_library()
    if lib is not None and hasattr(lib, "silisphere_su2_holonomy_bhtd_fwd_f32"):
        q_rot, k_rot, v_raw, pi_path = _CudaSU2HolonomyBHTDFn.apply(
            omega, q, k, v if transport_v else None, scale_val, use_tanh, float(eps), map_mode
        )
        return q_rot, k_rot, (v_raw if transport_v else v), pi_path

    require_ok(map_mode == "exp", "V1 fiber-major library only supports map_mode='exp'", code="BAD_MAP_MODE")
    bsz, n_heads, seq_len, d_head = q.shape
    n_blocks = d_head // 4
    raw_f = omega.permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 3)
    q_f = _to_fiber_major_4(q, n_blocks)
    k_f = _to_fiber_major_4(k, n_blocks)
    v_f = _to_fiber_major_4(v, n_blocks) if transport_v else None
    qo_f, ko_f, vo_f, u_f = _CudaSU2HolonomyFn.apply(raw_f, q_f, k_f, v_f, scale_val, use_tanh, float(eps))
    q_rot = _from_fiber_major_4(qo_f, bsz, n_heads, seq_len, n_blocks)
    k_rot = _from_fiber_major_4(ko_f, bsz, n_heads, seq_len, n_blocks)
    v_rot = _from_fiber_major_4(vo_f, bsz, n_heads, seq_len, n_blocks) if transport_v else v
    pi_path = u_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous()
    return q_rot, k_rot, v_rot, pi_path


def cuda_su2_inverse_untransport(o: torch.Tensor, pi_path: torch.Tensor) -> torch.Tensor:
    """Execute bare-metal sm_86 inverse covariant un-transport L(Pi^{-1}) O."""
    bsz, n_heads, seq_len, d_head = o.shape
    o_q = o.reshape(bsz, n_heads, seq_len, d_head // 4, 4)
    return _CudaSU2UntransportFn.apply(o_q, pi_path).reshape_as(o)


def cuda_hopf_s3_residual(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Execute bare-metal sm_86 H6c Native Hopf-S^3 residual step."""
    return _CudaHopfS3ResidualFn.apply(h, delta)


def cuda_cayley2x2_rotate(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Execute V2 bare-metal sm_86 Beltrami Cayley2x2 rotation (FP32/BF16, 0 B spills)."""
    return _CudaCayley2x2Fn.apply(x, omega)


def cuda_hopf_s3_cayley2x2_step(h: torch.Tensor, delta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Execute V2 zero-HBM-epilogue fused Hopf-S^3 + Cayley2x2 step (FP32/BF16, 0 B spills)."""
    return _CudaHopfS3Cayley2x2Fn.apply(h, delta, omega)


def cuda_bridge_status() -> dict[str, Any]:
    """Non-intrusive CPU-safe status probe (never triggers CUDA context initialization)."""
    lib_path = find_cuda_library()
    cu_src = _CSRC_DIR / "su2_holonomy_scan_sm86.cu"
    return {
        "backend": "cuda_sm86",
        "source_staged": cu_src.is_file(),
        "library_staged": lib_path is not None,
        "source_relpath": "holosphere/kernels/csrc/su2_holonomy_scan_sm86.cu",
        "kernel_count_v2": 27,
    }
