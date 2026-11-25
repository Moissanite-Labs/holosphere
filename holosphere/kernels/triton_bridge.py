"""Triton 3.9 bridge wiring holosphere.kernels.su2_holonomy_triton into unified autograd dispatch.

Zero GPU initialization on import or under ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch


def is_triton_installed() -> bool:
    """Return True iff the ``triton`` package is importable (CPU-safe check)."""
    try:
        importlib.import_module("triton")
        return True
    except Exception:
        return False


def load_triton_kernel_module() -> Any | None:
    """Load ``holosphere.kernels.su2_holonomy_triton`` if staged locally, else None."""
    try:
        return importlib.import_module("holosphere.kernels.su2_holonomy_triton")
    except ModuleNotFoundError:
        return None


def is_triton_kernel_available() -> bool:
    """Return True iff Triton + local fused kernel module are staged AND CUDA is active."""
    mod = load_triton_kernel_module()
    if mod is None or not bool(getattr(mod, "TRITON_AVAILABLE", False)):
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class _TritonSU2HolonomyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, raw_f: torch.Tensor, q_f: torch.Tensor, k_f: torch.Tensor, v_f: torch.Tensor | None, angle_scale: float, use_tanh: bool, eps: float):  # type: ignore[override]
        mod = load_triton_kernel_module()
        in_dtype = q_f.dtype
        raw_w = raw_f.float().contiguous()
        q_w = q_f.float().contiguous()
        k_w = k_f.float().contiguous()
        v_w = v_f.float().contiguous() if v_f is not None else None
        u_w, qo_w, ko_w, vo_w = mod.triton_su2_holonomy_fwd(
            raw_w, q_w, k_w, v_w, angle_scale=angle_scale, use_tanh=use_tanh, eps=eps
        )
        ctx.angle_scale = angle_scale
        ctx.use_tanh = use_tanh
        ctx.eps = eps
        ctx.in_dtype = in_dtype
        ctx.has_v = v_w is not None
        ctx.save_for_backward(raw_w, u_w, q_w, k_w, v_w if v_w is not None else torch.empty(0, device=raw_w.device))
        v_ret = vo_w.to(in_dtype) if vo_w is not None else torch.empty(0, device=raw_w.device, dtype=in_dtype)
        return qo_w.to(in_dtype), ko_w.to(in_dtype), v_ret, u_w.to(in_dtype)

    @staticmethod
    def backward(ctx: Any, dq_out: torch.Tensor | None, dk_out: torch.Tensor | None, dv_out: torch.Tensor | None, du_out: torch.Tensor | None):  # type: ignore[override]
        mod = load_triton_kernel_module()
        raw_w, u_w, q_w, k_w, v_saved = ctx.saved_tensors
        v_w = v_saved if ctx.has_v else None
        dq_w = dq_out.float().contiguous() if dq_out is not None else torch.zeros_like(q_w)
        dk_w = dk_out.float().contiguous() if dk_out is not None else torch.zeros_like(k_w)
        dv_w = dv_out.float().contiguous() if (ctx.has_v and dv_out is not None and dv_out.numel() > 0) else None
        du_w = du_out.float().contiguous() if du_out is not None else None
        d_raw, dq_in, dk_in, dv_in = mod.triton_su2_holonomy_bwd(
            raw_w, u_w, q_w, k_w, v_w, du_w, dq_w, dk_w, dv_w,
            angle_scale=ctx.angle_scale, use_tanh=ctx.use_tanh, eps=ctx.eps,
        )
        dt = ctx.in_dtype
        return d_raw.to(dt), dq_in.to(dt), dk_in.to(dt), (dv_in.to(dt) if dv_in is not None else None), None, None, None


def triton_su2_holonomy_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    *,
    angle_scale: float | None = 0.75,
    transport_v: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute fused Triton SU(2) holonomy scan + covariant (Q, K[, V]) transport."""
    bsz, n_heads, seq_len, d_head = q.shape
    n_blocks = d_head // 4
    raw_f = omega.permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 3)
    q_f = q.reshape(bsz, n_heads, seq_len, n_blocks, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 4)
    k_f = k.reshape(bsz, n_heads, seq_len, n_blocks, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 4)
    v_f = (
        v.reshape(bsz, n_heads, seq_len, n_blocks, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz * n_heads * n_blocks, seq_len, 4)
        if transport_v
        else None
    )
    use_tanh = angle_scale is not None
    scale_val = 1.0 if angle_scale is None else float(angle_scale)
    qo_f, ko_f, vo_f, u_f = _TritonSU2HolonomyFn.apply(raw_f, q_f, k_f, v_f, scale_val, use_tanh, float(eps))
    q_rot = qo_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz, n_heads, seq_len, d_head)
    k_rot = ko_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz, n_heads, seq_len, d_head)
    v_rot = (
        vo_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous().reshape(bsz, n_heads, seq_len, d_head)
        if transport_v
        else v
    )
    pi_path = u_f.reshape(bsz, n_heads, n_blocks, seq_len, 4).permute(0, 1, 3, 2, 4).contiguous()
    return q_rot, k_rot, v_rot, pi_path


def triton_bridge_status() -> dict[str, Any]:
    """Non-intrusive CPU-safe status probe."""
    return {
        "backend": "triton",
        "triton_installed": is_triton_installed(),
        "module_staged": load_triton_kernel_module() is not None,
    }
