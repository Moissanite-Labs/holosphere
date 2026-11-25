"""Exact Darboux isospectral intertwining & dark-mode kernel diagnostics (S109 / S112 Portal 5).

Verifies the universal algebraic laws proved in Lean 4 (Track S112 Thms 28-31):
  1. Intertwining: A (A^T A + c I_n) = (A A^T + c I_m) A
  2. Dark-mode annihilation: k in ker(A) => A(x + k) = A x and <A^T y, k> = 0
  3. Kernel energy fraction: ||P_{ker(A)} x||^2 / ||x||^2 lower-bounds reconstruction error.
"""

from __future__ import annotations

from typing import Any

import torch

from holosphere.errors import require_ok


@torch.no_grad()
def darboux_intertwining_diagnostic(
    a_mat: torch.Tensor,
    shift_c: float = 0.25,
    svd_tol: float = 1e-6,
) -> dict[str, Any]:
    """Verify Darboux intertwining and measure dark-mode kernel dimension of linear map A [m, n]."""
    require_ok(a_mat.ndim == 2, f"a_mat must be 2D, got shape {tuple(a_mat.shape)}", code="DARBOUX_SHAPE")
    af = a_mat.detach().to(dtype=torch.float64, device="cpu")
    m, n = af.shape
    l1 = af.T @ af + shift_c * torch.eye(n, dtype=torch.float64)
    l2 = af @ af.T + shift_c * torch.eye(m, dtype=torch.float64)
    intertwining_err = float(torch.linalg.matrix_norm(af @ l1 - l2 @ af, ord="fro").item())

    _, svals, vh = torch.linalg.svd(af, full_matrices=True)
    s_max = float(svals[0].item()) if svals.numel() > 0 else 0.0
    rank = int((svals > max(svd_tol * s_max, 1e-12)).sum().item()) if s_max > 0.0 else 0
    dark_dim = n - rank

    dark_orth_err = 0.0
    dark_annihilation_err = 0.0
    if dark_dim > 0:
        v_dark = vh[rank:, :].T  # [n, dark_dim]
        dark_annihilation_err = float(torch.linalg.matrix_norm(af @ v_dark, ord="fro").item())
        dark_orth_err = float(torch.linalg.matrix_norm(af.T @ (af @ v_dark), ord="fro").item())

    return {
        "shape": [int(m), int(n)],
        "shift_c": float(shift_c),
        "rank": rank,
        "dark_dim": dark_dim,
        "intertwining_frob_err": intertwining_err,
        "dark_annihilation_err": dark_annihilation_err,
        "dark_orthogonality_err": dark_orth_err,
    }


@torch.no_grad()
def measure_dark_mode_energy_fraction(
    a_mat: torch.Tensor,
    x_samples: torch.Tensor,
    svd_tol: float = 1e-6,
) -> dict[str, Any]:
    """Compute exact fraction of activation energy residing in ker(A) (the Darboux dark subspace)."""
    require_ok(a_mat.ndim == 2, "a_mat must be 2D [m, n]", code="DARBOUX_SHAPE")
    af = a_mat.detach().to(dtype=torch.float64, device="cpu")
    m, n = af.shape
    require_ok(x_samples.shape[-1] == n, f"x_samples last dim {x_samples.shape[-1]} != {n}", code="DARBOUX_DIM")
    xf = x_samples.detach().to(dtype=torch.float64, device="cpu").reshape(-1, n)
    _, svals, vh = torch.linalg.svd(af, full_matrices=True)
    s_max = float(svals[0].item()) if svals.numel() > 0 else 0.0
    rank = int((svals > max(svd_tol * s_max, 1e-12)).sum().item()) if s_max > 0.0 else 0
    dark_dim = n - rank

    total_e = (xf * xf).sum(dim=-1).clamp_min(1e-18)
    if dark_dim == 0:
        dark_frac = torch.zeros_like(total_e)
    else:
        v_dark = vh[rank:, :].T  # [n, dark_dim]
        proj_dark = (xf @ v_dark) @ v_dark.T
        dark_e = (proj_dark * proj_dark).sum(dim=-1)
        dark_frac = dark_e / total_e

    return {
        "m": int(m),
        "n": int(n),
        "rank": rank,
        "dark_dim": dark_dim,
        "mean_dark_energy_fraction": float(dark_frac.mean().item()),
        "max_dark_energy_fraction": float(dark_frac.max().item()),
    }
