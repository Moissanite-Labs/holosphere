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
        orig_dtype = x.dtype
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        wf = omega.float()
        w2 = wf * wf
        inv_denom = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * inv_denom
        s = (2.0 * wf) * inv_denom
        x_even, x_odd = xf.unbind(-1)
        y_even = c * x_even + s * x_odd
        y_odd = -s * x_even + c * x_odd
        return torch.stack((y_even, y_odd), dim=-1).reshape_as(x).to(orig_dtype)

    def explicit_matrix(self, omega: torch.Tensor | None = None) -> torch.Tensor:
        w = (self.base_omega if omega is None else omega).detach().float().reshape(self.n_pairs)
        w2 = w * w
        inv_d = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * inv_d
        s = (2.0 * w) * inv_d
        q = torch.zeros((self.d_model, self.d_model), dtype=torch.float32, device=w.device)
        idx = torch.arange(self.n_pairs, device=w.device)
        q[2 * idx, 2 * idx] = c
        q[2 * idx, 2 * idx + 1] = s
        q[2 * idx + 1, 2 * idx] = -s
        q[2 * idx + 1, 2 * idx + 1] = c
        return q

    def explicit_skew_matrix(self, omega: torch.Tensor | None = None) -> torch.Tensor:
        w = (self.base_omega if omega is None else omega).detach().float().reshape(self.n_pairs)
        k = torch.zeros((self.d_model, self.d_model), dtype=torch.float32, device=w.device)
        idx = torch.arange(self.n_pairs, device=w.device)
        k[2 * idx, 2 * idx + 1] = -w
        k[2 * idx + 1, 2 * idx] = w
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_cayley_2x2(x, self.compute_omega(x))


def build_paired_beltrami_operator(d_model: int) -> torch.Tensor:
    """Parity-balanced symmetric shell operator C = diag(lam_j I_2) commuting with 2x2 Cayley."""
    require_ok(d_model >= 4 and d_model % 4 == 0, "d_model must be positive multiple of 4", code="BELTRAMI_DIM")
    n_half = (d_model // 2) // 2
    diag_vals: list[float] = []
    for q in range(n_half):
        lam = float(q + 1) / float(n_half)
        diag_vals.extend([lam, lam, -lam, -lam])
    return torch.diag(torch.tensor(diag_vals, dtype=torch.float32))


@torch.no_grad()
def run_beltrami_helicity_diagnostic(
    cayley_module: CayleyBlock2x2SO,
    n_steps: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify Portal-3 Beltrami [C, K] = 0 helicity conservation & O3.5 symmetric-collapse control."""
    d = cayley_module.d_model
    c_op = build_paired_beltrami_operator(d)
    k_comm = cayley_module.explicit_skew_matrix().cpu()
    q_comm = cayley_module.explicit_matrix().cpu()

    v0 = torch.randn(d, generator=_seeded_gen(seed), dtype=torch.float32)
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
