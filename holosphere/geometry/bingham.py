"""SO(k)-equivariant vector-proximal Poloidal Bingham operator S_Y^H(u) (S109/S110)."""

from __future__ import annotations

import torch
import torch.nn as nn

from holosphere.errors import require_ok


class VectorProximalPoloidalBingham(nn.Module):
    """SO(k)-equivariant vector-proximal Poloidal Bingham shrinkage without post-renorm.

    Splits u = (u_pol, u_tor) with exact Hodge orthogonality <u_pol, u_tor> = 0:
      S_Y^H(u) = max(0, 1 - tau_Y / ||u_pol||_2) * u_pol  (+)  u_tor
    Guarantees SO(k) equivariance on the k-dimensional Poloidal subspace, exact Stagnant-Lid
    zeroing when ||u_pol||_2 <= tau_Y, and finite forward/backward gradients at u_pol == 0.
    """

    def __init__(
        self,
        d_model: int,
        k_poloidal: int | None = None,
        tau_y: float = 0.0,
        yield_number: float = 0.5,
        post_renorm: bool = False,
    ) -> None:
        super().__init__()
        require_ok(
            not post_renorm,
            "post_renorm=True is forbidden: post-renormalization breaks SO(k) equivariance",
            code="BINGHAM_POST_RENORM_FORBIDDEN",
        )
        k_pol = (d_model // 2) if k_poloidal is None else int(k_poloidal)
        require_ok(0 < k_pol < d_model, f"k_poloidal must satisfy 0 < k < d_model ({k_pol}, {d_model})", code="BINGHAM_DIM")
        require_ok(tau_y >= 0.0, f"tau_y must be >= 0, got {tau_y}", code="BINGHAM_TAU")
        require_ok(0.0 <= yield_number <= 1.0, f"yield_number must be in [0, 1], got {yield_number}", code="BINGHAM_Y")
        self.d_model = d_model
        self.k_poloidal = k_pol
        self.yield_number = float(yield_number)
        self.register_buffer("tau_y", torch.tensor(float(tau_y), dtype=torch.float32))

    @torch.no_grad()
    def calibrate_from_activations(
        self,
        u_samples: torch.Tensor,
        yield_number: float | None = None,
    ) -> dict[str, float]:
        """Freeze tau_Y from empirical quantile of ||u_pol||_2."""
        y_val = self.yield_number if yield_number is None else float(yield_number)
        require_ok(0.0 <= y_val <= 1.0, f"yield_number must be in [0, 1], got {y_val}", code="BINGHAM_Y")
        u_pol = u_samples[..., : self.k_poloidal].float()
        norms = torch.linalg.vector_norm(u_pol, ord=2, dim=-1).reshape(-1)
        require_ok(norms.numel() > 0, "calibration activation tensor must be non-empty", code="BINGHAM_EMPTY")
        if y_val <= 0.0:
            tau = 0.0
        elif y_val >= 1.0:
            tau = float(norms.max().item()) * (1.0 + 1e-5) + 1e-6
        else:
            tau = float(torch.quantile(norms, y_val).item())
        self.yield_number = y_val
        self.tau_y.fill_(tau)
        return {
            "yield_number": y_val,
            "tau_y": tau,
            "plug_fraction": float((norms <= tau).float().mean().item()),
        }

    def forward(
        self,
        u: torch.Tensor,
        tau_override: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        orig_dtype = u.dtype
        uf = u.float()
        u_pol = uf[..., : self.k_poloidal]
        u_tor = uf[..., self.k_poloidal :]
        tau_val = float(self.tau_y.item()) if tau_override is None else float(tau_override)
        require_ok(tau_val >= 0.0, f"tau_override must be >= 0, got {tau_val}", code="BINGHAM_TAU")

        if tau_val <= 0.0:
            out_pol = u_pol
            plug_frac = 0.0
        else:
            norm_sq = (u_pol * u_pol).sum(dim=-1, keepdim=True)
            tau_sq = tau_val * tau_val
            active = norm_sq > tau_sq
            safe_norm = torch.sqrt(torch.where(active, norm_sq, torch.ones_like(norm_sq)))
            scale = torch.where(active, (1.0 - tau_val / safe_norm).clamp_min(0.0), torch.zeros_like(norm_sq))
            out_pol = scale * u_pol
            with torch.no_grad():
                plug_frac = float((~active).float().mean().item())

        return torch.cat((out_pol, u_tor), dim=-1).to(orig_dtype), {
            "plug_fraction": plug_frac,
            "tau_y": tau_val,
        }
