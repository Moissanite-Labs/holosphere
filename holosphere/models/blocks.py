"""Kernel-isomorphic HoloSphere attention & transformer blocks.

Every block lowers directly to the fused ``holosphere.kernels`` primitives:
  - ``su2_holonomy_transport`` (fused exp -> associative quaternion scan -> Q, K[, V] transport)
  - ``su2_inverse_untransport`` (H14c covariant value un-transport L(Pi^{-1}) O)
  - ``fused_spherical_step`` (H6c Hopf-S^3 or nGPT SLERP residual step)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from holosphere.errors import require_ok
from holosphere.geometry.beltrami import CayleyBlock2x2SO
from holosphere.geometry.bingham import VectorProximalPoloidalBingham
from holosphere.geometry.su2 import safe_unit_norm
from holosphere.kernels.dispatch import (
    cayley_2x2_step,
    fused_spherical_step,
    hopf_s3_cayley_2x2_step,
    su2_holonomy_transport,
    su2_inverse_untransport,
)
from holosphere.models.controls import (
    Block4RandOrthoPositional,
    DiagonalDarcyEMA,
    LieRE4DAbelianPositional,
    RMSNorm,
    apply_rope_2d,
)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP supporting both Euclidean Pre-Norm and nGPT hypersphere scaling."""

    def __init__(self, d_model: int, d_ff: int, ngpt_scale: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ngpt_scale = bool(ngpt_scale)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        if self.ngpt_scale:
            self.su = nn.Parameter(torch.ones(d_ff))
            self.sv = nn.Parameter(torch.ones(d_ff))
        else:
            self.su = None
            self.sv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.w_up(x)
        v = self.w_gate(x)
        if self.ngpt_scale and self.su is not None and self.sv is not None:
            u = u * self.su
            v = v * (self.sv * math.sqrt(float(self.d_model)))
        return self.w_down(F.silu(v) * u)


class CovariantSU2Attention(nn.Module):
    """Causal self-attention lowering directly to fused SU(2) holonomy transport (or matched controls).

    When ``pos_mode == "su2_holonomy"`` and ``transport_v == True`` (Track S114 H14c):
      1. Computes path-ordered prefix holonomy Pi_t = du_0 * ... * du_t in SU(2).
      2. Left-transports Q_t, K_t, AND V_t in registers using the SAME Pi_t.
      3. Runs causal Flash/SDPA attention on (Q', K', V').
      4. Un-transports attention output O_t <- L(Pi_t^{-1}) O_t = sum_s w_{ts} L(Pi_t^{-1} Pi_s) V_s
         at ZERO extra prefix-scan cost!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        pos_mode: str = "su2_holonomy",
        su2_map_mode: str = "exp",
        transport_v: bool = True,
        angle_scale: float = 0.75,
        ngpt_qk_norm: bool = True,
        backend: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__()
        require_ok(d_model % n_heads == 0, "d_model must be divisible by n_heads", code="ATTN_DIM")
        require_ok(su2_map_mode in ("exp", "cayley"), f"unknown su2_map_mode {su2_map_mode!r}", code="ATTN_MAP")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.pos_mode = pos_mode
        self.su2_map_mode = su2_map_mode
        self.transport_v = bool(transport_v)
        self.angle_scale = float(angle_scale)
        self.ngpt_qk_norm = bool(ngpt_qk_norm)
        self.backend = backend

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.w_omega: nn.Linear | None = None
        self.liere: LieRE4DAbelianPositional | None = None
        self.rand_ortho: Block4RandOrthoPositional | None = None
        self._sqk_block_tied = pos_mode in ("su2_holonomy", "liere4d", "block4_rand_ortho")

        if pos_mode == "su2_holonomy":
            require_ok(self.d_head % 4 == 0, "su2_holonomy requires d_head % 4 == 0", code="ATTN_QUAT_DIM")
            self.n_blocks = self.d_head // 4
            self.w_omega = nn.Linear(d_model, 3 * n_heads * self.n_blocks, bias=False)
        elif pos_mode == "liere4d":
            self.n_blocks = self.d_head // 4
            self.liere = LieRE4DAbelianPositional(n_heads, self.d_head, seed=seed)
        elif pos_mode == "block4_rand_ortho":
            self.n_blocks = self.d_head // 4
            self.rand_ortho = Block4RandOrthoPositional(n_heads, self.d_head, seed=seed)
        elif pos_mode == "rope2d":
            self.n_blocks = self.d_head // 2
        else:
            raise ValueError(f"Unknown pos_mode: {pos_mode!r}")

        if self.ngpt_qk_norm:
            if self._sqk_block_tied:
                # Block-4 tied sqk preserves exact SU(2) 4-block equivariance: L(U)(c_b * q_b) == c_b * L(U)(q_b).
                self.sqk: nn.Parameter | None = nn.Parameter(torch.ones(1, n_heads, 1, self.n_blocks, 1))
            else:
                self.sqk = nn.Parameter(torch.ones(1, n_heads, 1, self.d_head))
        else:
            self.sqk = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        pi_path: torch.Tensor | None = None
        if self.pos_mode == "su2_holonomy":
            require_ok(self.w_omega is not None, "w_omega missing", code="ATTN_STATE")
            omega = (
                self.w_omega(x)
                .view(bsz, seq_len, self.n_heads, self.n_blocks, 3)
                .transpose(1, 2)
            )
            q, k, v, pi_path = su2_holonomy_transport(
                q,
                k,
                v,
                omega,
                angle_scale=self.angle_scale,
                transport_v=self.transport_v,
                backend=self.backend,
                map_mode=self.su2_map_mode,
            )
        elif self.pos_mode == "rope2d":
            q = apply_rope_2d(q)
            k = apply_rope_2d(k)
        elif self.pos_mode == "liere4d":
            require_ok(self.liere is not None, "liere missing", code="ATTN_STATE")
            q = self.liere(q)
            k = self.liere(k)
        elif self.pos_mode == "block4_rand_ortho":
            require_ok(self.rand_ortho is not None, "rand_ortho missing", code="ATTN_STATE")
            q = self.rand_ortho(q)
            k = self.rand_ortho(k)

        if self.ngpt_qk_norm and self.sqk is not None:
            if self._sqk_block_tied:
                sqk_exp = self.sqk.expand(1, self.n_heads, 1, self.n_blocks, 4).reshape(1, self.n_heads, 1, self.d_head)
            else:
                sqk_exp = self.sqk
            q = safe_unit_norm(q, dim=-1, eps=1e-4) * sqk_exp
            k = safe_unit_norm(k, dim=-1, eps=1e-4) * sqk_exp
            scale = math.sqrt(float(self.d_head))
        else:
            scale = 1.0 / math.sqrt(float(self.d_head))

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        if self.pos_mode == "su2_holonomy" and self.transport_v and pi_path is not None:
            attn_out = su2_inverse_untransport(attn_out, pi_path, backend=self.backend)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.w_o(attn_out)


class HoloSphereBlock(nn.Module):
    """Unified kernel-isomorphic layer combining covariant SU(2) attention, Hopf/SLERP step, & Geosphere ops."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        pos_mode: str = "su2_holonomy",
        su2_map_mode: str = "exp",
        transport_v: bool = True,
        angle_scale: float = 0.75,
        residual_mode: str = "hopf_s3",
        ngpt_qk_norm: bool = True,
        use_cayley: bool = False,
        cayley_rank: int = 16,
        use_bingham: bool = False,
        bingham_yield: float = 0.5,
        bingham_tau: float = 0.0,
        use_darcy: bool = False,
        alpha_init: float = 0.05,
        backend: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_mode = residual_mode
        self.backend = backend

        self.attn = CovariantSU2Attention(
            d_model,
            n_heads,
            pos_mode=pos_mode,
            su2_map_mode=su2_map_mode,
            transport_v=transport_v,
            angle_scale=angle_scale,
            ngpt_qk_norm=ngpt_qk_norm,
            backend=backend,
            seed=seed,
        )
        self.mlp = SwiGLUMLP(d_model, d_ff, ngpt_scale=(residual_mode == "slerp"))

        if residual_mode == "slerp":
            init_logit = math.log(alpha_init / max(1e-4, 1.0 - alpha_init))
            self.alpha_a_logit: nn.Parameter | None = nn.Parameter(torch.full((d_model,), init_logit))
            self.alpha_m_logit: nn.Parameter | None = nn.Parameter(torch.full((d_model,), init_logit))
            self.norm1: RMSNorm | None = None
            self.norm2: RMSNorm | None = None
        else:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.alpha_a_logit = None
            self.alpha_m_logit = None

        self.cayley = CayleyBlock2x2SO(d_model, bottleneck_rank=cayley_rank, seed=seed + 10) if use_cayley else None
        self.darcy = DiagonalDarcyEMA(d_model, seed=seed + 20) if use_darcy else None
        self.bingham = (
            VectorProximalPoloidalBingham(d_model, tau_y=bingham_tau, yield_number=bingham_yield, post_renorm=False)
            if use_bingham
            else None
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        cayley_done = False
        if self.residual_mode == "slerp":
            require_ok(self.alpha_a_logit is not None and self.alpha_m_logit is not None, "slerp alphas missing", code="BLK_STATE")
            h = safe_unit_norm(h, dim=-1, eps=1e-4)
            alpha_a = torch.sigmoid(self.alpha_a_logit) * (0.45 * math.pi)
            h = fused_spherical_step(h, self.attn(h), alpha_a, mode="slerp", eps=1e-4, backend=self.backend)
            if self.darcy is not None:
                h = safe_unit_norm(self.darcy(h), dim=-1, eps=1e-4)
            alpha_m = torch.sigmoid(self.alpha_m_logit) * (0.45 * math.pi)
            h = fused_spherical_step(h, self.mlp(h), alpha_m, mode="slerp", eps=1e-4, backend=self.backend)
        elif self.residual_mode == "hopf_s3":
            require_ok(self.norm1 is not None and self.norm2 is not None, "RMSNorm missing", code="BLK_STATE")
            h = fused_spherical_step(h, self.attn(self.norm1(h)), mode="hopf_s3", backend=self.backend)
            if self.darcy is not None:
                h = self.darcy(h)
            mlp_out = self.mlp(self.norm2(h))
            if self.cayley is not None and not self.cayley.data_dependent:
                h = hopf_s3_cayley_2x2_step(h, mlp_out, self.cayley.base_omega, backend=self.backend)
                cayley_done = True
            else:
                h = fused_spherical_step(h, mlp_out, mode="hopf_s3", backend=self.backend)
        else:
            require_ok(self.norm1 is not None and self.norm2 is not None, "RMSNorm missing", code="BLK_STATE")
            h = h + self.attn(self.norm1(h))
            if self.darcy is not None:
                h = self.darcy(h)
            h = h + self.mlp(self.norm2(h))

        if self.cayley is not None and not cayley_done:
            h = cayley_2x2_step(h, self.cayley.compute_omega(h), backend=self.backend)
        if self.bingham is not None:
            h, _ = self.bingham(h)
        return h
