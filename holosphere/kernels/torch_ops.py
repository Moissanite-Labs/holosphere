"""Exact PyTorch differentiable reference & custom-adjoint operators for HoloSphere kernels.

Implements the exact mathematical oracle for ``H14a``, ``H14b``, and ``H14c``, including:
  - ``SU2UnitPrefixScanFunction``: custom ``torch.autograd.Function`` using the exact
    O(1)-multiply additive suffix-sum reverse-mode adjoint on SU(2):
      S_t = sum_{j=t}^{T-1} (g_j * conj(Pi_j))
      dL / du_t = conj(Pi_{t-1}) * S_t * Pi_t
    matching the exact warp-shuffle backward algorithm of the bare-metal ``sm_86`` kernel!
  - ``torch_su2_holonomy_transport``: fused exp -> scan -> covariant (Q, K[, V]) transport.
  - ``torch_su2_inverse_untransport``: covariant L(Pi^{-1}) O un-transport.
  - ``torch_fused_spherical_step``: unified Hopf-S^3 (H6c), SLERP (nGPT), and non-antipodal
    acute-cone Safe-Hopf loop step (H7c + W10-OBS-31).
"""

from __future__ import annotations

import math

import torch

from holosphere.errors import require_ok
from holosphere.geometry.beltrami import cayley_2x2_rotate
from holosphere.geometry.hopf import hopf_fiber_rotate, hopf_fiber_rotate_cs, hopf_s3_residual_step
from holosphere.geometry.su2 import (
    cayley_su2,
    quat_conj,
    quat_inverse_untransport,
    quat_left_transport,
    quat_mul,
    safe_quat_unit_norm,
    safe_slerp_step,
    su2_exp,
    su2_parallel_prefix_scan,
)


class SU2UnitPrefixScanFunction(torch.autograd.Function):
    """Exact custom-adjoint associative prefix product on unit quaternions in SU(2).

    Forward:
      u_hat[t] = delta_u[t] / sqrt(||delta_u[t]||^2 + eps^2)
      Pi[t]    = u_hat[0] * u_hat[1] * ... * u_hat[t]   (Hillis-Steele O(log T) scan)

    Backward (exact for unit quaternions ||u_hat[t]|| == 1):
      S[t]         = sum_{j=t}^{T-1} grad_Pi[j] * conj(Pi[j])    (pure additive suffix sum!)
      grad_u_hat[t] = conj(Pi[t-1]) * S[t] * Pi[t]               (with Pi[-1] = (1,0,0,0))
      grad_delta_u[t] = (grad_u_hat[t] - <grad_u_hat[t], u_hat[t]> u_hat[t]) / norm[t]
    """

    @staticmethod
    def forward(ctx, delta_u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:  # type: ignore[override]
        # delta_u: [..., T, n_blocks, 4] with seq dim == -3
        require_ok(math.isfinite(eps) and eps > 0.0, f"eps must be finite and > 0, got {eps}", code="BAD_EPS")
        require_ok(delta_u.ndim >= 3 and delta_u.shape[-1] == 4, "delta_u must be [..., T, n_blocks, 4]", code="SCAN_SHAPE")
        n2 = torch.sum(delta_u * delta_u, dim=-1, keepdim=True)
        active = n2 > (eps * eps)
        norm = torch.sqrt(n2 + (eps * eps))
        u_hat = delta_u / norm
        id_q = torch.zeros_like(u_hat)
        id_q[..., 0] = 1.0
        u_hat = torch.where(active, u_hat, id_q)
        pi = su2_parallel_prefix_scan(u_hat, seq_dim=-3, eps=eps)
        ctx.save_for_backward(u_hat, norm, pi, active)
        return pi

    @staticmethod
    def backward(ctx, grad_pi: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        u_hat, norm, pi, active = ctx.saved_tensors
        # 1. Per-step pullback g_j * conj(Pi_j) -> pure additive suffix sum S_t = sum_{j>=t} term_j
        term = quat_mul(grad_pi, quat_conj(pi))
        s_suffix = torch.flip(torch.cumsum(torch.flip(term, dims=(-3,)), dim=-3), dims=(-3,))

        # 2. Shift Pi right by 1 along seq dim (-3), inserting identity quaternion (1, 0, 0, 0) at t=0
        id_q = torch.zeros_like(pi[..., :1, :, :])
        id_q[..., 0] = 1.0
        pi_prev = torch.cat((id_q, pi[..., :-1, :, :]), dim=-3)

        # 3. Exact unit-quaternion adjoint: grad_u_hat[t] = conj(Pi[t-1]) * S[t] * Pi[t]
        grad_u_hat = quat_mul(quat_conj(pi_prev), quat_mul(s_suffix, pi))

        # 4. Chain rule through smooth unit normalization (masked to 0 on constant identity fallback)
        dot_ug = torch.sum(grad_u_hat * u_hat, dim=-1, keepdim=True)
        grad_delta_u = torch.where(active, (grad_u_hat - dot_ug * u_hat) / norm, torch.zeros_like(grad_u_hat))
        return grad_delta_u, None


def su2_unit_prefix_scan_fast(delta_u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable SU(2) prefix product along dim -3 using exact additive suffix-sum backward."""
    return SU2UnitPrefixScanFunction.apply(delta_u, float(eps))


def torch_su2_holonomy_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    *,
    angle_scale: float | None = 0.75,
    transport_v: bool = False,
    eps: float = 1e-6,
    use_custom_adjoint: bool = True,
    map_mode: str = "exp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable PyTorch reference for fused SU(2) holonomy scan + covariant transport.

    Args:
      q, k, v: [B, H, T, D] with D % 4 == 0 (n_blocks = D // 4).
      omega:   [B, H, T, n_blocks, 3] Lie-algebra su(2) generator.
      angle_scale: If not None, maps omega -> angle_scale * tanh(omega); else uses raw omega.
      transport_v: If True (H14c), also left-transports V by Pi(t) for full covariant attention.
      map_mode: ``"exp"`` (su2_exp) or ``"cayley"`` (SFU-free rational cayley_su2).
    Returns:
      (q_rot, k_rot, v_rot, pi_path) where pi_path has shape [B, H, T, n_blocks, 4] and is
      preserved in FP32/FP64 even under BF16 inputs to eliminate S^3 manifold drift.
    """
    require_ok(map_mode in ("exp", "cayley"), f"unknown map_mode {map_mode!r}", code="BAD_MAP_MODE")
    require_ok(math.isfinite(eps) and eps > 0.0, f"eps must be finite and > 0, got {eps}", code="BAD_EPS")
    if angle_scale is not None:
        require_ok(math.isfinite(angle_scale), f"angle_scale must be finite, got {angle_scale}", code="BAD_ANGLE_SCALE")
    require_ok(q.ndim == 4 and k.shape == q.shape and v.shape == q.shape, "q, k, v must share [B, H, T, D]", code="QKV_SHAPE")
    bsz, n_heads, seq_len, d_head = q.shape
    require_ok(d_head % 4 == 0, f"d_head must be divisible by 4, got {d_head}", code="QKV_DIM")
    n_blocks = d_head // 4
    expected_omega = (bsz, n_heads, seq_len, n_blocks, 3)
    require_ok(
        tuple(omega.shape) == expected_omega,
        f"omega shape {tuple(omega.shape)} != {expected_omega}",
        code="OMEGA_SHAPE",
    )

    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    w = omega.to(compute_dtype)
    if angle_scale is not None:
        w = float(angle_scale) * torch.tanh(w)
    delta_u = cayley_su2(w) if map_mode == "cayley" else su2_exp(w, eps=eps)
    if use_custom_adjoint:
        pi_path = su2_unit_prefix_scan_fast(delta_u, eps=eps)
    else:
        pi_path = su2_parallel_prefix_scan(delta_u, seq_dim=-3, eps=eps)

    pi_cast = pi_path.to(q.dtype)
    q_rot = quat_left_transport(q, pi_cast)
    k_rot = quat_left_transport(k, pi_cast)
    v_rot = quat_left_transport(v, pi_cast) if transport_v else v
    # Preserve pi_path in FP32/FP64 so S^3 manifold norm stays <= 1e-6 under BF16 autocast;
    # note Q/K/V transport multiplies execute in q.dtype so BF16 L(Pi^-1)L(Pi) is approximate.
    return q_rot, k_rot, v_rot, pi_path


def torch_su2_inverse_untransport(o: torch.Tensor, pi_path: torch.Tensor) -> torch.Tensor:
    """Exact differentiable inverse un-transport L(Pi^{-1}) O for H14c covariant attention."""
    return quat_inverse_untransport(o, pi_path.to(o.dtype))


def torch_hopf_s3_cayley2x2_step(h: torch.Tensor, delta: torch.Tensor, omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable reference for V2 fused Hopf-S^3 residual + Beltrami Cayley2x2 epilogue."""
    require_ok(math.isfinite(eps) and eps > 0.0, f"eps must be finite and > 0, got {eps}", code="BAD_EPS")
    z = hopf_s3_residual_step(h, delta, eps=eps)
    return cayley_2x2_rotate(z, omega)


def torch_fused_spherical_step(
    h: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor | float | None = None,
    *,
    mode: str = "hopf_s3",
    phi: torch.Tensor | float = 0.0,
    c_s: tuple[float, float] | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Differentiable PyTorch reference for the unified spherical/Hopf residual & loop update."""
    require_ok(math.isfinite(eps) and eps > 0.0, f"eps must be finite and > 0, got {eps}", code="BAD_EPS")
    if mode == "hopf_s3":
        return hopf_s3_residual_step(h, delta, eps=min(eps, 1e-6))
    if mode == "slerp":
        require_ok(alpha is not None, "mode='slerp' requires alpha", code="MISSING_ALPHA")
        return safe_slerp_step(h, delta, alpha, eps=eps)
    if mode == "safe_hopf_loop":
        require_ok(alpha is not None, "mode='safe_hopf_loop' requires alpha", code="MISSING_ALPHA")
        rot_delta = hopf_fiber_rotate(delta, phi)
        return safe_slerp_step(h, rot_delta, alpha, eps=eps)
    if mode == "safe_hopf_cayley_loop":
        require_ok(alpha is not None, "mode='safe_hopf_cayley_loop' requires alpha", code="MISSING_ALPHA")
        require_ok(c_s is not None and len(c_s) == 2, "mode='safe_hopf_cayley_loop' requires c_s=(c, s)", code="MISSING_CS")
        rot_delta = hopf_fiber_rotate_cs(delta, c_s[0], c_s[1])
        return safe_slerp_step(h, rot_delta, alpha, eps=eps)
    raise ValueError(f"Unknown fused_spherical_step mode: {mode!r}")
