"""Exact SU(2) ~= S^3 quaternion algebra and path-ordered holonomy primitives.

One unit quaternion q = (w, x, y, z) in SU(2) maps 1-to-1 to a single Ampere ``float4``
128-bit register quad (LDG.E.128), enabling warp-shuffle associative holonomy scans and
covariant (Q, K, V) parallel transport with zero HBM intermediate round-trips.
"""

from __future__ import annotations

from typing import Any

import torch

from holosphere.errors import require_ok


def safe_unit_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-4) -> torch.Tensor:
    """Smooth unit-sphere projection x / sqrt(||x||_2^2 + eps^2) (zero 0/0 NaN at x=0)."""
    require_ok(eps > 0.0, f"eps must be > 0, got {eps}", code="BAD_EPS")
    denom = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + (eps * eps))
    return x / denom


def safe_slerp_step(
    h: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor | float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Antipodal-safe spherical update Norm(cos(alpha) * Norm(h) + sin(alpha) * Norm(delta)).

    Smooth floor sqrt(||x||^2 + eps^2) keeps forward and backward finite even at exact
    antipodal cancellation delta == -h with alpha == pi/4.
    """
    require_ok(eps > 0.0, f"eps must be > 0, got {eps}", code="BAD_EPS")
    h_u = safe_unit_norm(h, dim=-1, eps=eps)
    d_u = safe_unit_norm(delta, dim=-1, eps=eps)
    alpha_t = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(float(alpha), device=h.device, dtype=h.dtype)
    mixed = torch.cos(alpha_t) * h_u + torch.sin(alpha_t) * d_u
    return safe_unit_norm(mixed, dim=-1, eps=eps)


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact Hamilton product on last dimension (w, x, y, z)."""
    require_ok(p.shape[-1] == 4 and q.shape[-1] == 4, "quat_mul requires last dim == 4", code="QUAT_SHAPE")
    pw, px, py, pz = p.unbind(dim=-1)
    qw, qx, qy, qz = q.unbind(dim=-1)
    return torch.stack(
        (
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate q_bar = (w, -x, -y, -z) == q^{-1} for unit q in SU(2)."""
    require_ok(q.shape[-1] == 4, "quat_conj requires last dim == 4", code="QUAT_SHAPE")
    qw, qx, qy, qz = q.unbind(dim=-1)
    return torch.stack((qw, -qx, -qy, -qz), dim=-1)


def su2_exp(omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact exponential map su(2) -> SU(2): omega in R^3 -> unit quaternion (w, x, y, z).

    Uses smooth norm theta = sqrt(||omega||^2 + eps^2) so gradients remain smooth at omega = 0.
    """
    require_ok(omega.shape[-1] == 3, f"su2_exp requires last dim == 3, got {omega.shape[-1]}", code="SU2_SHAPE")
    require_ok(eps > 0.0, f"eps must be > 0, got {eps}", code="BAD_EPS")
    theta_sq = torch.sum(omega * omega, dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + (eps * eps))
    w = torch.cos(theta)
    sinc = torch.sin(theta) / theta
    q = torch.cat((w, sinc * omega), dim=-1)
    return safe_unit_norm(q, dim=-1, eps=eps)


def cayley_su2(omega: torch.Tensor) -> torch.Tensor:
    """SFU-free pure-rational Cayley map su(2) -> SU(2) ~= S^3 (0 trig, 0 rsqrt).

    Maps omega in R^3 to:
      q_cayley(omega) = ((1 - ||omega||^2) / (1 + ||omega||^2),  2 * omega / (1 + ||omega||^2))
    Satisfies the exact algebraic unit-sphere identity (1 - r^2)^2 + 4 r^2 == (1 + r^2)^2,
    so ||q_cayley(omega)||_2 == 1 identically for all omega in R^3. The antipodal pole -I
    is bounded strictly away whenever ||omega||_2 <= t_max < 1 (giving w >= (1-t_max^2)/(1+t_max^2) > 0),
    unifying S^1 (``cayley_acute_cos_sin``), SO(2) (``Cayley2x2AutogradFn``), and SU(2) (``cayley_su2``).
    """
    require_ok(omega.shape[-1] == 3, f"cayley_su2 requires last dim == 3, got {omega.shape[-1]}", code="SU2_SHAPE")
    compute_dtype = torch.float64 if omega.dtype == torch.float64 else torch.float32
    wf = omega.to(compute_dtype)
    r2 = torch.sum(wf * wf, dim=-1, keepdim=True)
    inv_d = 1.0 / (1.0 + r2)
    q0 = (1.0 - r2) * inv_d
    q_vec = (2.0 * inv_d) * wf
    return torch.cat((q0, q_vec), dim=-1).to(omega.dtype)


def su2_parallel_prefix_scan(delta_u: torch.Tensor, seq_dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """Associative Hillis-Steele right-accumulating prefix product along ``seq_dim``.

    Computes Pi(t) = du(0) * du(1) * ... * du(t) in ceil(log2 T) parallel rounds, so that
    for any s < t the relative interval holonomy is strictly path-ordered:
      Pi(s)^{-1} * Pi(t) = du(s+1) * ... * du(t)
    and covariant value un-transport computes:
      Pi(t)^{-1} * Pi(s) = (du(s+1) * ... * du(t))^{-1}.
    """
    require_ok(delta_u.shape[-1] == 4, "su2_parallel_prefix_scan requires last dim == 4", code="QUAT_SHAPE")
    ndim = delta_u.ndim
    dim = seq_dim if seq_dim >= 0 else ndim + seq_dim
    require_ok(0 <= dim < ndim - 1, f"invalid seq_dim={seq_dim} for ndim={ndim}", code="BAD_SEQ_DIM")
    moved = delta_u.movedim(dim, 1) if dim != 1 else delta_u
    seq_len = moved.shape[1]
    require_ok(seq_len >= 1, "sequence length must be >= 1", code="EMPTY_SEQ")

    cur = safe_unit_norm(moved, dim=-1, eps=eps)
    offset = 1
    while offset < seq_len:
        left = cur[:, :-offset]
        right = cur[:, offset:]
        prod = safe_unit_norm(quat_mul(left, right), dim=-1, eps=eps)
        cur = torch.cat((cur[:, :offset], prod), dim=1)
        offset *= 2
    return cur.movedim(1, dim) if dim != 1 else cur


def su2_sequential_prefix_scan(delta_u: torch.Tensor, seq_dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """Exact sequential reference Pi(t) = Pi(t-1) * du(t) for numerical parity verification."""
    require_ok(delta_u.shape[-1] == 4, "su2_sequential_prefix_scan requires last dim == 4", code="QUAT_SHAPE")
    ndim = delta_u.ndim
    dim = seq_dim if seq_dim >= 0 else ndim + seq_dim
    require_ok(0 <= dim < ndim - 1, f"invalid seq_dim={seq_dim} for ndim={ndim}", code="BAD_SEQ_DIM")
    moved = delta_u.movedim(dim, 1) if dim != 1 else delta_u
    seq_len = moved.shape[1]
    require_ok(seq_len >= 1, "sequence length must be >= 1", code="EMPTY_SEQ")

    outs: list[torch.Tensor] = [safe_unit_norm(moved[:, 0], dim=-1, eps=eps)]
    for t in range(1, seq_len):
        outs.append(safe_unit_norm(quat_mul(outs[-1], moved[:, t]), dim=-1, eps=eps))
    res = torch.stack(outs, dim=1)
    return res.movedim(1, dim) if dim != 1 else res


def quat_left_transport(vec: torch.Tensor, pi_path: torch.Tensor) -> torch.Tensor:
    """Left-multiply 4D blocks of vec [..., D] by unit quaternions pi_path [..., D//4, 4]."""
    d = vec.shape[-1]
    require_ok(d % 4 == 0, f"quat_left_transport requires D % 4 == 0, got {d}", code="QUAT_DIM")
    n_blocks = d // 4
    require_ok(
        pi_path.shape[-2:] == (n_blocks, 4),
        f"pi_path trailing shape {tuple(pi_path.shape[-2:])} != ({n_blocks}, 4)",
        code="QUAT_SHAPE",
    )
    vb = vec.reshape(*vec.shape[:-1], n_blocks, 4)
    return quat_mul(pi_path, vb).reshape_as(vec)


def quat_inverse_untransport(vec: torch.Tensor, pi_path: torch.Tensor) -> torch.Tensor:
    """Apply inverse unit-quaternion left-transport L(Pi^{-1}) vec = L(conj(Pi)) vec."""
    return quat_left_transport(vec, quat_conj(pi_path))


@torch.no_grad()
def verify_su2_interval_holonomy(
    delta_u: torch.Tensor,
    s: int,
    t: int,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Verify exact path-ordered interval identity Pi(s)^{-1} Pi(t) == du(s+1) * ... * du(t)."""
    seq_len = delta_u.shape[1]
    require_ok(0 <= s < t < seq_len, f"require 0 <= s < t < {seq_len}, got s={s}, t={t}", code="BAD_INTERVAL")
    pi = su2_parallel_prefix_scan(delta_u, seq_dim=1, eps=eps)
    lhs = quat_mul(quat_conj(pi[:, s]), pi[:, t])
    rhs = safe_unit_norm(delta_u[:, s + 1], dim=-1, eps=eps)
    for idx in range(s + 2, t + 1):
        rhs = safe_unit_norm(quat_mul(rhs, delta_u[:, idx]), dim=-1, eps=eps)
    max_err = float(torch.max(torch.abs(lhs - rhs)).item())
    norm_err = float(torch.max(torch.abs(torch.linalg.vector_norm(pi, dim=-1) - 1.0)).item())
    return {
        "s": s,
        "t": t,
        "max_interval_error": max_err,
        "max_unit_norm_error": norm_err,
    }
