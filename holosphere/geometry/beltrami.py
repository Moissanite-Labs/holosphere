"""Beltrami [C, K] = 0 helicity-preserving 2x2 block-diagonal Cayley SO(d) operators (S110/S112)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from holosphere.errors import require_ok


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


def _reduce_broadcast_grad(grad: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Sum gradient across any leading or size-1 dimensions expanded during broadcasting."""
    out = grad
    while out.ndim > len(target_shape):
        out = out.sum(dim=0)
    for dim, size in enumerate(target_shape):
        if size == 1 and out.shape[dim] != 1:
            out = out.sum(dim=dim, keepdim=True)
    return out


class Cayley2x2AutogradFn(torch.autograd.Function):
    """Exact closed-form 2x2 block-diagonal Cayley SO(d) forward & reverse-mode adjoint.

    Matches the bare-metal Ampere ``sm_86`` register-recompute kernel contract (1 reciprocal +
    7 FMAs per 2D pair, 0 trig, 0 matrix inverses, 0 extra HBM reads of y):
      Forward:
        d = 1 / (1 + w^2),  c = (1 - w^2) * d,  s = (2 w) * d
        y_0 =  c * x_0 + s * x_1,   y_1 = -s * x_0 + c * x_1
      Backward (from dQ/dw = (2 / (1 + w^2)) * J * Q(w)):
        dL/dx_0 =  c * g_0 - s * g_1,   dL/dx_1 =  s * g_0 + c * g_1
        dL/dw   = (2 / (1 + w^2)) * (g_0 * y_1 - g_1 * y_0)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        require_ok(x.ndim >= 1 and x.shape[-1] > 0 and x.shape[-1] % 2 == 0, "x last dim must be positive even", code="CAYLEY_DIM")
        n_pairs = x.shape[-1] // 2
        require_ok(
            omega.ndim >= 1 and omega.shape[-1] == n_pairs,
            f"omega last dim must equal d_model // 2 ({n_pairs}), got {tuple(omega.shape)}",
            code="CAYLEY_OMEGA_SHAPE",
        )
        compute_dtype = torch.float64 if (x.dtype == torch.float64 or omega.dtype == torch.float64) else torch.float32
        xf = x.to(compute_dtype).reshape(*x.shape[:-1], n_pairs, 2)
        wf = omega.to(compute_dtype)
        w2 = wf * wf
        inv_d = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * inv_d
        s = (2.0 * wf) * inv_d
        x0, x1 = xf.unbind(dim=-1)
        y0 = c * x0 + s * x1
        y1 = -s * x0 + c * x1
        ctx.save_for_backward(x, omega)
        return torch.stack((y0, y1), dim=-1).reshape_as(x).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # type: ignore[override]
        x, omega = ctx.saved_tensors
        n_pairs = x.shape[-1] // 2
        compute_dtype = (
            torch.float64
            if (x.dtype == torch.float64 or omega.dtype == torch.float64 or grad_y.dtype == torch.float64)
            else torch.float32
        )
        xf = x.to(compute_dtype).reshape(*x.shape[:-1], n_pairs, 2)
        gf = grad_y.to(compute_dtype).reshape(*grad_y.shape[:-1], n_pairs, 2)
        wf = omega.to(compute_dtype)

        w2 = wf * wf
        inv_d = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * inv_d
        s = (2.0 * wf) * inv_d
        x0, x1 = xf.unbind(dim=-1)
        g0, g1 = gf.unbind(dim=-1)

        grad_x: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            gx0 = c * g0 - s * g1
            gx1 = s * g0 + c * g1
            grad_x = torch.stack((gx0, gx1), dim=-1).reshape_as(x).to(x.dtype)

        grad_omega: torch.Tensor | None = None
        if ctx.needs_input_grad[1]:
            y0 = c * x0 + s * x1
            y1 = -s * x0 + c * x1
            gw_full = (2.0 * inv_d) * (g0 * y1 - g1 * y0)
            grad_omega = _reduce_broadcast_grad(gw_full, tuple(omega.shape)).to(omega.dtype)

        return grad_x, grad_omega


def cayley_2x2_rotate(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Apply exact 2x2 Cayley SO(d) rotation using closed-form reverse-mode adjoint."""
    return Cayley2x2AutogradFn.apply(x, omega)


class CayleyBlock2x2SO(nn.Module):
    """Closed-form 2x2 block-diagonal Cayley SO(d) feature rotation (4 FMAs/pair, zero LU).

    For each pair (x_{2j}, x_{2j+1}) and skew parameter omega_j:
      c_j = (1 - w_j^2) / (1 + w_j^2),  s_j = 2 w_j / (1 + w_j^2)
      y_{2j}   =  c_j * x_{2j} + s_j * x_{2j+1}
      y_{2j+1} = -s_j * x_{2j} + c_j * x_{2j+1}
    Exact orthogonality Q^T Q = I_d, det(Q) = +1, <h, K h> = 0, and [C, K] = 0 with paired C.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_rank: int = 16,
        data_dependent: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        require_ok(d_model > 0 and d_model % 2 == 0, "d_model must be positive even", code="CAYLEY_DIM")
        self.d_model = d_model
        self.n_pairs = d_model // 2
        self.data_dependent = bool(data_dependent)
        self.bottleneck_rank = int(bottleneck_rank) if data_dependent else 0

        g = _seeded_gen(seed)
        with torch.no_grad():
            init_omega = torch.empty(self.n_pairs).uniform_(-0.35, 0.35, generator=g)
        self.base_omega = nn.Parameter(init_omega)
        if self.data_dependent:
            require_ok(self.bottleneck_rank > 0, "bottleneck_rank must be > 0", code="CAYLEY_RANK")
            self.w_down = nn.Linear(d_model, self.bottleneck_rank, bias=False)
            self.w_up = nn.Linear(self.bottleneck_rank, self.n_pairs, bias=False)
            with torch.no_grad():
                self.w_down.weight.normal_(0.0, 1.0 / math.sqrt(float(d_model)), generator=_seeded_gen(seed + 1))
                self.w_up.weight.normal_(0.0, 0.05 / math.sqrt(float(self.bottleneck_rank)), generator=_seeded_gen(seed + 2))
        else:
            self.w_down = None
            self.w_up = None

    def compute_omega(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if not self.data_dependent or x is None or self.w_down is None or self.w_up is None:
            return self.base_omega
        delta = 0.75 * torch.tanh(self.w_up(F.silu(self.w_down(x))))
        return self.base_omega.view(*([1] * (x.ndim - 1)), self.n_pairs) + delta

    @staticmethod
    def apply_cayley_2x2(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return cayley_2x2_rotate(x, omega)

    def explicit_matrix(self, omega: torch.Tensor | None = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = (self.base_omega if omega is None else omega).detach().to(dtype).reshape(self.n_pairs)
        w2 = w * w
        inv_d = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * inv_d
        s = (2.0 * w) * inv_d
        q = torch.zeros((self.d_model, self.d_model), dtype=dtype, device=w.device)
        idx = torch.arange(self.n_pairs, device=w.device)
        q[2 * idx, 2 * idx] = c
        q[2 * idx, 2 * idx + 1] = s
        q[2 * idx + 1, 2 * idx] = -s
        q[2 * idx + 1, 2 * idx + 1] = c
        return q

    def explicit_skew_matrix(self, omega: torch.Tensor | None = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = (self.base_omega if omega is None else omega).detach().to(dtype).reshape(self.n_pairs)
        k = torch.zeros((self.d_model, self.d_model), dtype=dtype, device=w.device)
        idx = torch.arange(self.n_pairs, device=w.device)
        k[2 * idx, 2 * idx + 1] = -w
        k[2 * idx + 1, 2 * idx] = w
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_cayley_2x2(x, self.compute_omega(x))


def build_paired_beltrami_operator(d_model: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Parity-balanced symmetric shell operator C = diag(lam_j I_2) commuting with 2x2 Cayley."""
    require_ok(d_model >= 4 and d_model % 4 == 0, "d_model must be positive multiple of 4", code="BELTRAMI_DIM")
    n_half = (d_model // 2) // 2
    diag_vals: list[float] = []
    for q in range(n_half):
        lam = float(q + 1) / float(n_half)
        diag_vals.extend([lam, lam, -lam, -lam])
    return torch.diag(torch.tensor(diag_vals, dtype=dtype))


@torch.no_grad()
def run_beltrami_helicity_diagnostic(
    cayley_module: CayleyBlock2x2SO,
    n_steps: int = 8,
    seed: int = 42,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Verify Portal-3 Beltrami [C, K] = 0 helicity conservation & O3.5 symmetric-collapse control."""
    d = cayley_module.d_model
    c_op = build_paired_beltrami_operator(d, dtype=dtype)
    k_comm = cayley_module.explicit_skew_matrix(dtype=dtype).cpu()
    q_comm = cayley_module.explicit_matrix(dtype=dtype).cpu()

    v0 = torch.randn(d, generator=_seeded_gen(seed), dtype=dtype)
    v0 = v0 / torch.linalg.vector_norm(v0)

    def _moments(v: torch.Tensor) -> list[float]:
        out: list[float] = []
        cv = v.clone()
        for _ in range(5):
            out.append(float(torch.dot(v, cv).item()))
            cv = c_op @ cv
        return out

    comm_frob = float(torch.linalg.matrix_norm(c_op @ k_comm - k_comm @ c_op, ord="fro").item())
    m0 = _moments(v0)
    v = v0.clone()
    max_moment_drift = 0.0
    for _ in range(n_steps):
        v = q_comm @ v
        for a, b in zip(m0, _moments(v)):
            max_moment_drift = max(max_moment_drift, abs(a - b))

    # O3.5 symmetric residual collapse control
    v_sym = v0.clone()
    rayleigh_seq: list[float] = []
    for _ in range(n_steps + 1):
        e_cur = float(torch.dot(v_sym, v_sym).item())
        h_cur = float(torch.dot(v_sym, c_op @ v_sym).item())
        rayleigh_seq.append(h_cur / e_cur)
        v_sym = v_sym + 0.25 * (c_op @ v_sym)

    return {
        "d_model": d,
        "n_steps": n_steps,
        "commuting_commutator_frob": comm_frob,
        "commuting_max_moment_drift_m0_m4": max_moment_drift,
        "o35_symmetric_strictly_monotone_collapse": all(
            rayleigh_seq[i + 1] > rayleigh_seq[i] for i in range(n_steps)
        ),
    }


@torch.no_grad()
def measure_trace_beltrami_rayleigh(pass_states: Sequence[torch.Tensor]) -> list[float]:
    """Compute mean Beltrami Rayleigh quotient H/E = <h, C h> / ||h||^2 across loop passes."""
    if not pass_states:
        return []
    d = pass_states[0].shape[-1]
    if d % 4 != 0:
        return []
    c_diag = torch.diagonal(build_paired_beltrami_operator(d)).to(pass_states[0].device)
    out: list[float] = []
    for h in pass_states:
        hf = h.float()
        e_val = (hf * hf).sum(dim=-1).clamp_min(1e-12)
        h_val = (hf * hf * c_diag.view(1, 1, d)).sum(dim=-1)
        out.append(float((h_val / e_val).mean().item()))
    return out
