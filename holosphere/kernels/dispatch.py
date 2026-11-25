"""Typed multi-backend dispatch layer unifying bare-metal sm_86 CUDA, Triton, & PyTorch CPU ops."""

from __future__ import annotations

import math
from typing import Any

import torch

from holosphere.errors import HoloSphereError, require_ok
from holosphere.geometry.hopf import (
    cayley_acute_cos_sin,
    hopf_clock_angle,
    hopf_fiber_rotate,
    hopf_fiber_rotate_cs,
)
from holosphere.kernels.cuda_bridge import (
    cuda_bridge_status,
    cuda_hopf_s3_residual,
    cuda_su2_holonomy_transport,
    cuda_su2_inverse_untransport,
    is_cuda_kernel_available,
)
from holosphere.kernels.torch_ops import (
    torch_fused_spherical_step,
    torch_su2_holonomy_transport,
    torch_su2_inverse_untransport,
)
from holosphere.kernels.triton_bridge import (
    is_triton_kernel_available,
    triton_bridge_status,
    triton_su2_holonomy_transport,
)

VALID_BACKENDS = frozenset({"auto", "cuda_sm86", "triton", "torch_cpu"})


def available_backends() -> dict[str, Any]:
    """Report CPU-safe availability of all kernel backends (zero GPU initialization)."""
    return {
        "torch_cpu": {"backend": "torch_cpu", "available": True, "custom_adjoint": True},
        "triton": triton_bridge_status(),
        "cuda_sm86": cuda_bridge_status(),
    }


def _resolve_backend(backend: str, tensor: torch.Tensor) -> str:
    require_ok(backend in VALID_BACKENDS, f"unknown backend {backend!r}", code="BAD_BACKEND")
    if backend == "torch_cpu":
        return "torch_cpu"
    if backend == "cuda_sm86":
        if not tensor.is_cuda or tensor.dtype == torch.float64 or not is_cuda_kernel_available():
            raise HoloSphereError("BACKEND_UNAVAILABLE", "cuda_sm86 backend requires compiled sm_86 lib + CUDA non-FP64 tensor")
        return "cuda_sm86"
    if backend == "triton":
        if not tensor.is_cuda or tensor.dtype == torch.float64 or not is_triton_kernel_available():
            raise HoloSphereError("BACKEND_UNAVAILABLE", "triton backend requires staged kernel + CUDA non-FP64 tensor")
        return "triton"
    # Auto dispatch: prefer bare-metal sm_86 CUDA -> Triton -> exact PyTorch custom-adjoint fallback
    if tensor.is_cuda and tensor.dtype != torch.float64:
        if is_cuda_kernel_available():
            return "cuda_sm86"
        if is_triton_kernel_available():
            return "triton"
    return "torch_cpu"


def su2_holonomy_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    *,
    angle_scale: float | None = 0.75,
    transport_v: bool = False,
    eps: float = 1e-6,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SU(2) exponential map, associative holonomy scan, and covariant (Q, K[, V]) transport."""
    resolved = _resolve_backend(backend, q)
    if resolved == "cuda_sm86":
        return cuda_su2_holonomy_transport(
            q, k, v, omega, angle_scale=angle_scale, transport_v=transport_v, eps=eps
        )
    if resolved == "triton":
        return triton_su2_holonomy_transport(
            q, k, v, omega, angle_scale=angle_scale, transport_v=transport_v, eps=eps
        )
    return torch_su2_holonomy_transport(
        q, k, v, omega, angle_scale=angle_scale, transport_v=transport_v, eps=eps, use_custom_adjoint=True
    )


def su2_inverse_untransport(
    o: torch.Tensor,
    pi_path: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Inverse covariant un-transport L(Pi^{-1}) O = L(conj(Pi)) O (H14c)."""
    resolved = _resolve_backend(backend, o)
    if resolved == "cuda_sm86":
        return cuda_su2_inverse_untransport(o, pi_path)
    return torch_su2_inverse_untransport(o, pi_path)


def fused_spherical_step(
    h: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor | float | None = None,
    *,
    mode: str = "hopf_s3",
    phi: torch.Tensor | float = 0.0,
    eps: float = 1e-4,
    backend: str = "auto",
) -> torch.Tensor:
    """Unified register-fused spherical residual / Safe-Hopf loop step."""
    resolved = _resolve_backend(backend, h)
    if resolved == "cuda_sm86" and mode == "hopf_s3":
        return cuda_hopf_s3_residual(h, delta)
    return torch_fused_spherical_step(h, delta, alpha, mode=mode, phi=phi, eps=eps)


def safe_hopf_clock(
    x: torch.Tensor,
    pass_idx: int,
    *,
    law: str = "acute_cone",
    r_train: int = 3,
    phi_max: float = 0.42 * math.pi,
    tau: float = 2.0,
    t_max: float = 0.8,
    backend: str = "auto",
) -> torch.Tensor:
    """Apply O(d) isoclinic Hopf S^1 fiber clock (supports SFU-free ``cayley_acute`` & ``acute_cone``)."""
    _ = _resolve_backend(backend, x)
    if law == "cayley_acute":
        c, s = cayley_acute_cos_sin(pass_idx, t_max=t_max, tau=tau)
        return hopf_fiber_rotate_cs(x, c, s)
    phi = hopf_clock_angle(pass_idx, clock_law=law, r_train=r_train, phi_max=phi_max, tau=tau, t_max=t_max)
    return hopf_fiber_rotate(x, phi)
