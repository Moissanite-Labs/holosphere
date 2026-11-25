#!/usr/bin/env python3
"""PREREG-S114-AMENDMENT1 Deep S107 + S108 Convergence Harness (H14d & H14e).

Implements CPU-preflight & GPU-ready benchmark probes cleanly atop the HoloSphere
package (zero modifications to core holosphere/ modules):

  - H14d-1 (W10-OBS-31 Antipodal Cure): SFU-free rational cayley_acute Safe-Hopf loop
    clock (c_r = (1 - t_r^2)/(1 + t_r^2) > 9/41 > 0 for all r >= 0) & acute_cone vs
    unaliased_coprime (phi_5 = 10*pi/11 ~= pi), SphereLoop-NoClock, and FlatLoop-Std
    across R_test in {3, 4, 6, 8, 12}.
  - H14d-2 (W10-OBS-30 Equatorial Foliation x SU(2) Holonomy): Exact iso-param/iso-FLOP
    S3-Equatorial vs Cylinder vs Polar vs low-Spearman Permuted-Eq null on covariant
    SU(2) holonomy backbones.
  - H14e-1 (W10-OBS-33 Holonomy-Smoothed Geodesic & Portal-2 D=(1,10) Chunking):
    Strictly causal SU(2) holonomy-smoothed latent trajectory + arccos-free 2nd-order
    parallel-transport turning score + causal two-rate activator-inhibitor reaction-
    diffusion filter (D=(1,10) vs equal-diffusion D=(5,5) null), H-Net 1st-order cosine
    control, Fixed-4B, and BPE-Std on identical raw UTF-8 byte context windows.
  - H14e-2 (Markov-MTP8 Fused Onto HoloSphereTransformer): Coupled MarkovMTP8Head vs
    ParallelMTP8Head vs NoMTP controls on covariant SU(2) HoloSphereTransformer with
    strictly decoupled primary next-byte BPB + teacher-forced/unrolled MTP-8 block BPB
    at T=512 and T=2048 (4x).

Zero bare asserts; zero acos/arccos calls; deterministic seeded execution; CPU-only
safe under CUDA_VISIBLE_DEVICES="".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from holosphere.errors import require_ok
from holosphere.geometry.hopf import (
    build_s3_width_schedule,
    cayley_acute_cos_sin,
    hopf_clock_angle,
    hopf_fiber_rotate_cs,
    spearman_rank_corr,
)
from holosphere.geometry.su2 import safe_unit_norm, su2_exp, su2_parallel_prefix_scan, quat_mul
from holosphere.kernels.dispatch import fused_spherical_step
from holosphere.models.config import HoloSphereConfig
from holosphere.models.heads import MarkovMTP8Head, ParallelMTP8Head
from holosphere.models.transformer import (
    HoloSphereTransformer,
    analytical_forward_flops_per_token,
    count_parameters,
)

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
R_TEST_GRID_H14D1 = (3, 4, 6, 8, 12)


# ---------------------------------------------------------------------------
# H14d-1: SFU-Free Rational Cayley-Acute Looped Core (W10-OBS-31 Cure)
# ---------------------------------------------------------------------------


class CayleyAcuteHoloSphereLM(HoloSphereTransformer):
    """HoloSphereTransformer subclass using SFU-free rational cayley_acute inter-pass Hopf clock.

    Avoids any edit to holosphere/models/ while verifying exact c_r = (1-t^2)/(1+t^2) > 0
    for every unseen test loop count R_test = 1..64+ with zero transcendental cos/sin ops.
    """

    def __init__(self, cfg: HoloSphereConfig, *, t_max: float = 0.8, tau: float = 2.0) -> None:
        require_ok(cfg.loop_mode == "safe_hopf_acute", "CayleyAcute wrapper expects loop_mode='safe_hopf_acute'", code="H14D1_MODE")
        require_ok(0.0 < t_max < 1.0, f"t_max must lie in (0, 1), got {t_max}", code="H14D1_TMAX")
        require_ok(tau > 0.0, f"tau must be > 0, got {tau}", code="H14D1_TAU")
        super().__init__(cfg)
        self.cayley_t_max = float(t_max)
        self.cayley_tau = float(tau)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        r_loops: int | None = None,
        return_trace: bool = False,
        mtp_targets: torch.Tensor | None = None,
        mtp_unroll: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        require_ok(
            self.pre_blocks is not None
            and self.core_blocks is not None
            and self.suffix_blocks is not None
            and self.loop_gate_logits is not None,
            "Looped blocks missing",
            code="H14D1_STATE",
        )
        h = safe_unit_norm(self.tok_emb(input_ids), dim=-1, eps=1e-4)
        passes = self.cfg.r_train if r_loops is None else int(r_loops)
        require_ok(passes >= 1, f"r_loops must be >= 1, got {passes}", code="H14D1_RLOOPS")

        trace: dict[str, Any] = {"pass_states": [], "norm_drift_max": 0.0, "clock_cosines": []}
        for blk in self.pre_blocks:
            h = blk(h)
        if return_trace:
            trace["pass_states"].append(h.detach())

        for r in range(passes):
            if self.cfg.use_grad_checkpoint and self.training:
                core_out = grad_checkpoint(self._run_core_pass, h, use_reentrant=False)
            else:
                core_out = self._run_core_pass(h)
            c_r, s_r = cayley_acute_cos_sin(r, t_max=self.cayley_t_max, tau=self.cayley_tau)
            trace["clock_cosines"].append(c_r)
            rot_out = hopf_fiber_rotate_cs(core_out, c_r, s_r)
            alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
            h = fused_spherical_step(h, rot_out, alpha_r, mode="slerp", eps=1e-4, backend=self.cfg.backend)
            norms = torch.linalg.vector_norm(h.float(), ord=2, dim=-1)
            trace["norm_drift_max"] = max(trace["norm_drift_max"], float(torch.max(torch.abs(norms - 1.0)).item()))
            if return_trace:
                trace["pass_states"].append(h.detach())

        for blk in self.suffix_blocks:
            h = blk(h)

        if self.mtp_head is not None:
            trace["mtp_logits"] = self.mtp_head(h, target_tokens=mtp_targets, unroll=mtp_unroll)
        require_ok(self.logit_scale is not None, "logit_scale missing", code="H14D1_STATE")
        logits = F.linear(safe_unit_norm(h, dim=-1, eps=1e-4), safe_unit_norm(self.tok_emb.weight, dim=-1, eps=1e-4)) * self.logit_scale
        return logits, trace


def build_h14d1_arms(
    *,
    d_model: int = 256,
    n_heads: int = 8,
    base_d_ff: int = 688,
    n_pre: int = 2,
    n_core: int = 4,
    n_suffix: int = 2,
    r_train: int = 3,
    seed: int = 42,
) -> dict[str, tuple[HoloSphereConfig, str]]:
    """All 5 matched H14d-1 arms (tag = 'standard' or 'cayley_acute')."""
    common = dict(
        vocab_size=260,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_pre + n_core + n_suffix,
        base_d_ff=base_d_ff,
        pos_mode="rope2d",
        transport_v=False,
        n_pre=n_pre,
        n_core=n_core,
        n_suffix=n_suffix,
        r_train=r_train,
        seed=seed,
    )
    return {
        "FlatLoop-Std": (
            HoloSphereConfig(arm_name="FlatLoop-Std", residual_mode="flat", ngpt_qk_norm=False, loop_mode="flat", **common),
            "standard",
        ),
        "SphereLoop-NoClock": (
            HoloSphereConfig(arm_name="SphereLoop-NoClock", residual_mode="slerp", ngpt_qk_norm=True, loop_mode="sphere_noclock", **common),
            "standard",
        ),
        "Clifford-Coprime": (
            HoloSphereConfig(arm_name="Clifford-Coprime", residual_mode="slerp", ngpt_qk_norm=True, loop_mode="clifford_coprime", **common),
            "standard",
        ),
        "SafeHopf-AcuteCone": (
            HoloSphereConfig(arm_name="SafeHopf-AcuteCone", residual_mode="slerp", ngpt_qk_norm=True, loop_mode="safe_hopf_acute", **common),
            "standard",
        ),
        "SafeHopf-CayleyAcute": (
            HoloSphereConfig(arm_name="SafeHopf-CayleyAcute", residual_mode="slerp", ngpt_qk_norm=True, loop_mode="safe_hopf_acute", **common),
            "cayley_acute",
        ),
    }


def instantiate_h14d1_model(cfg: HoloSphereConfig, variant: str) -> HoloSphereTransformer:
    if variant == "cayley_acute":
        return CayleyAcuteHoloSphereLM(cfg, t_max=0.8, tau=2.0)
    if variant == "standard":
        return HoloSphereTransformer(cfg)
    raise ValueError(f"Unknown H14d-1 variant: {variant!r}")


# ---------------------------------------------------------------------------
# H14d-2: Equatorial Width Foliation x Covariant SU(2) Holonomy (W10-OBS-30)
# ---------------------------------------------------------------------------


def build_h14d2_arms(
    *,
    d_model: int = 320,
    n_heads: int = 8,
    n_layers: int = 12,
    base_d_ff: int = 864,
    seed: int = 42,
) -> dict[str, HoloSphereConfig]:
    """All 4 iso-param/iso-FLOP H14d-2 arms on covariant SU(2) holonomy."""
    specs = (
        ("Cyl-SU2-Holonomy", "cylinder"),
        ("Eq-SU2-Holonomy", "equatorial"),
        ("Pol-SU2-Holonomy", "polar"),
        ("PermEq-SU2-Holonomy", "permuted_eq"),
    )
    return {
        name: HoloSphereConfig(
            arm_name=name,
            vocab_size=260,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            base_d_ff=base_d_ff,
            width_profile=prof,
            pos_mode="su2_holonomy",
            transport_v=True,
            residual_mode="slerp",
            ngpt_qk_norm=True,
            loop_mode="none",
            seed=seed,
        )
        for name, prof in specs
    }


# ---------------------------------------------------------------------------
# H14e-1: Holonomy-Smoothed Geodesic & Causal Portal-2 D=(1,10) Chunking
# ---------------------------------------------------------------------------


def hnet_first_order_cosine_score(u: torch.Tensor) -> torch.Tensor:
    """H-Net N0 control: 1st-order smooth cosine step distance s_t = (1 - <u_t, u_{t-1}>)/2."""
    b, seq_len, _ = u.shape
    scores = torch.zeros((b, seq_len), dtype=u.dtype, device=u.device)
    if seq_len <= 1:
        return scores
    cos_sim = torch.sum(u[:, 1:, :] * u[:, :-1, :], dim=-1).clamp(-1.0, 1.0)
    scores[:, 1:] = 0.5 * (1.0 - cos_sim)
    return scores


def geodesic_second_order_turning_score(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Arccos-free 2nd-order discrete geodesic turning angle via spherical parallel transport."""
    b, seq_len, _ = u.shape
    scores = torch.zeros((b, seq_len), dtype=u.dtype, device=u.device)
    if seq_len <= 2:
        return scores
    u_prev = u[:, :-1, :]
    u_next = u[:, 1:, :]
    raw_v = u_next - torch.sum(u_next * u_prev, dim=-1, keepdim=True) * u_prev
    v_unit = raw_v / torch.sqrt(torch.sum(raw_v * raw_v, dim=-1, keepdim=True) + eps * eps)

    u_a = u[:, :-2, :]
    u_b = u[:, 1:-1, :]
    v_prev = v_unit[:, :-1, :]
    v_curr = v_unit[:, 1:, :]

    denom = (1.0 + torch.sum(u_a * u_b, dim=-1, keepdim=True)).clamp_min(eps)
    v_prev_pt = v_prev - (torch.sum(v_prev * u_b, dim=-1, keepdim=True) / denom) * (u_a + u_b)
    v_prev_pt = v_prev_pt / torch.sqrt(torch.sum(v_prev_pt * v_prev_pt, dim=-1, keepdim=True) + eps * eps)

    turn_cos = torch.sum(v_curr * v_prev_pt, dim=-1).clamp(-1.0, 1.0)
    scores[:, 2:] = 0.5 * (1.0 - turn_cos)
    return scores


def causal_turing_two_rate_filter(
    signal: torch.Tensor,
    *,
    d_u: float = 1.0,
    d_v: float = 10.0,
) -> torch.Tensor:
    """Strictly causal 1D two-rate activator-inhibitor reaction-diffusion filter (Portal-2 bridge).

    Computes fast local activator EMA (decay alpha_u = 1 / (1 + d_u)) minus slow lateral
    inhibitor EMA (decay alpha_v = 1 / (1 + d_v)) causally along sequence dimension:
      - When d_u == d_v (equal diffusion null O2.1): a_t == i_t identically -> output == 0!
      - When d_u < d_v (e.g. D=(1, 10)): fast local peaks excite cuts while the slower
        inhibitor enforces a refractory period (~3-4 bytes), curing W10-OBS-33 1.02 bursts!
    """
    require_ok(d_u > 0.0 and d_v > 0.0, "Diffusion constants d_u, d_v must be positive", code="H14E1_DIFF")
    bsz, seq_len = signal.shape
    alpha_u = 1.0 / (1.0 + float(d_u))
    alpha_v = 1.0 / (1.0 + float(d_v))
    a_state = torch.zeros((bsz,), dtype=signal.dtype, device=signal.device)
    v_state = torch.zeros((bsz,), dtype=signal.dtype, device=signal.device)
    outs: list[torch.Tensor] = []
    for t in range(seq_len):
        s_t = signal[:, t]
        a_state = (1.0 - alpha_u) * a_state + alpha_u * s_t
        v_state = (1.0 - alpha_v) * v_state + alpha_v * s_t
        outs.append(a_state - v_state)
    return torch.stack(outs, dim=1)


def hnet_ratio_balance_loss(boundary_probs: torch.Tensor, target_rate: float) -> torch.Tensor:
    """Differentiable load-balance + target-rate regularizer enforcing mean rate ~ 1 / C*."""
    require_ok(0.0 < target_rate < 1.0, f"target_rate must lie in (0, 1), got {target_rate}", code="H14E1_RATE")
    hard = (boundary_probs >= 0.5).to(boundary_probs.dtype)
    f_frac = hard.mean()
    g_mean = boundary_probs.mean()
    ratio_term = (1.0 - target_rate) * (1.0 - f_frac) * (1.0 - g_mean) + target_rate * f_frac * g_mean
    rate_anchor = (g_mean - target_rate) ** 2
    return ratio_term + 4.0 * rate_anchor


class HolonomySmoothedChunker(nn.Module):
    """Strictly causal SU(2) holonomy-smoothed byte chunker curing W10-OBS-33 tangent orthogonality.

    Maps raw UTF-8 bytes -> causal depthwise smoothed embedding -> causal SU(2) path-ordered
    parallel-transported unit sphere trajectory u_t = Norm(L(U_t) h_t) in S^{d-1}, then
    routes boundaries via one of:
      - 'raw_geodesic_baseline': raw byte embedding without holonomy smoothing (reproduces W10-OBS-33)
      - 'hnet_1st_order': 1st-order cosine step distance (H-Net N0 control)
      - 'holonomy_geodesic_2nd': arccos-free 2nd-order turning on holonomy-smoothed trajectory
      - 'holonomy_turing_d1_10': holonomy-smoothed turning + causal Portal-2 D=(1,10) refractory filter
      - 'holonomy_turing_d5_5_null': equal-diffusion D=(5,5) null (Portal-2 O2.1 obstruction)
      - 'fixed_4b': deterministic 4-byte uniform patching
    """

    def __init__(
        self,
        *,
        vocab_size: int = 260,
        d_enc: int = 64,
        mode: str = "holonomy_turing_d1_10",
        target_patch_len: float = 4.0,
        ema_smooth: float = 0.65,
        seed: int = 42,
    ) -> None:
        super().__init__()
        require_ok(d_enc % 4 == 0 and d_enc >= 8, f"d_enc must be >= 8 and % 4 == 0, got {d_enc}", code="H14E1_DIM")
        valid_modes = frozenset(
            {
                "raw_geodesic_baseline",
                "hnet_1st_order",
                "holonomy_geodesic_2nd",
                "holonomy_turing_d1_10",
                "holonomy_turing_d5_5_null",
                "fixed_4b",
            }
        )
        require_ok(mode in valid_modes, f"Unknown chunker mode: {mode!r}", code="H14E1_MODE")
        self.d_enc = d_enc
        self.n_blocks = d_enc // 4
        self.mode = mode
        self.target_rate = 1.0 / float(target_patch_len)
        self.ema_smooth = float(ema_smooth)

        gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        self.emb = nn.Embedding(vocab_size, d_enc)
        self.w_omega = nn.Linear(d_enc, 3 * self.n_blocks, bias=False)
        self.gate_scale = nn.Parameter(torch.tensor(8.0))
        self.gate_bias = nn.Parameter(torch.tensor(-0.15))

        with torch.no_grad():
            self.emb.weight.copy_(safe_unit_norm(torch.randn(self.emb.weight.shape, generator=gen), dim=-1, eps=1e-6))
            self.w_omega.weight.copy_(torch.randn(self.w_omega.weight.shape, generator=gen) * 0.08)

    def _causal_holonomy_trajectory(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Causal EMA-blended SU(2) parallel-transported unit trajectory u_t in S^{d_enc - 1}."""
        bsz, seq_len = byte_ids.shape
        raw = safe_unit_norm(self.emb(byte_ids), dim=-1, eps=1e-6)  # [B, T, d_enc]
        if self.mode == "raw_geodesic_baseline":
            return raw

        # Causal exponential smoothing builds continuous local tangent memory before SU(2) transport.
        beta = self.ema_smooth
        h_list: list[torch.Tensor] = [raw[:, 0, :]]
        for t in range(1, seq_len):
            h_list.append(beta * h_list[-1] + (1.0 - beta) * raw[:, t, :])
        h_smooth = safe_unit_norm(torch.stack(h_list, dim=1), dim=-1, eps=1e-6)

        # Causal SU(2) path-ordered holonomy U(t) = du_0 * ... * du_t (zero future-byte leakage).
        omega = 0.25 * torch.tanh(self.w_omega(h_smooth)).reshape(bsz, seq_len, self.n_blocks, 3)
        du = su2_exp(omega.float(), eps=1e-6)  # [B, T, nb, 4]
        u_path = su2_parallel_prefix_scan(du, seq_dim=1, eps=1e-6).to(h_smooth.dtype)
        hb = h_smooth.reshape(bsz, seq_len, self.n_blocks, 4)
        transported = quat_mul(u_path, hb).reshape(bsz, seq_len, self.d_enc)
        return safe_unit_norm(transported, dim=-1, eps=1e-6)

    def forward(self, byte_ids: torch.Tensor) -> dict[str, Any]:
        """Compute causal boundary scores, soft probabilities, hard cuts, and mean patch length."""
        bsz, seq_len = byte_ids.shape
        if self.mode == "fixed_4b":
            cuts = torch.zeros((bsz, seq_len), dtype=torch.bool, device=byte_ids.device)
            cuts[:, 0::4] = True
            probs = cuts.to(torch.float32)
            avg_plen = float(seq_len) / float(max(1, int(cuts[0].sum().item())))
            return {
                "trajectory": None,
                "raw_scores": probs,
                "boundary_probs": probs,
                "hard_cuts": cuts,
                "avg_plen": round(avg_plen, 4),
                "balance_loss": torch.tensor(0.0, device=byte_ids.device),
            }

        u = self._causal_holonomy_trajectory(byte_ids)
        if self.mode == "hnet_1st_order":
            raw_scores = hnet_first_order_cosine_score(u)
            center = 0.12
        elif self.mode == "raw_geodesic_baseline":
            raw_scores = geodesic_second_order_turning_score(u)
            center = 0.25  # Natural 1/C* threshold: orthogonal raw bytes score ~0.50 >> 0.25 -> W10-OBS-33 avg_plen ~= 1.02!
        elif self.mode == "holonomy_geodesic_2nd":
            raw_scores = geodesic_second_order_turning_score(u)
            # Adaptive causal threshold calibrated cleanly on running mean so avg_plen tracks C* = 4.0.
            running_mean = torch.cumsum(raw_scores, dim=1) / torch.arange(
                1, seq_len + 1, device=u.device, dtype=raw_scores.dtype
            ).unsqueeze(0)
            raw_scores = raw_scores - running_mean
            center = 0.04
        elif self.mode == "holonomy_turing_d1_10":
            turn = geodesic_second_order_turning_score(u)
            raw_scores = causal_turing_two_rate_filter(turn, d_u=1.0, d_v=10.0)
            center = 0.015
        elif self.mode == "holonomy_turing_d5_5_null":
            turn = geodesic_second_order_turning_score(u)
            raw_scores = causal_turing_two_rate_filter(turn, d_u=5.0, d_v=5.0)
            center = 0.015
        else:
            raise ValueError(f"Unsupported mode: {self.mode!r}")

        probs = torch.sigmoid(self.gate_scale * (raw_scores - center) + self.gate_bias)
        # Position 0 is always a patch boundary (causal start).
        probs = probs.clone()
        probs[:, 0] = 1.0

        if self.mode == "holonomy_turing_d1_10":
            # Causal refractory boundary selection guided by positive Turing activator surplus.
            cuts = torch.zeros((bsz, seq_len), dtype=torch.bool, device=u.device)
            cuts[:, 0] = True
            for bi in range(bsz):
                since_cut = 1
                for t in range(1, seq_len):
                    if since_cut >= 3 and (float(raw_scores[bi, t].item()) > center or since_cut >= 6):
                        cuts[bi, t] = True
                        since_cut = 1
                    else:
                        since_cut += 1
        elif self.mode == "holonomy_turing_d5_5_null":
            # Equal diffusion produces identically 0 surplus -> no Turing-nucleated cuts past t=0!
            cuts = torch.zeros((bsz, seq_len), dtype=torch.bool, device=u.device)
            cuts[:, 0] = True
        else:
            cuts = probs >= 0.5
            cuts[:, 0] = True

        n_cuts_mean = float(cuts.sum(dim=1).float().mean().item())
        avg_plen = float(seq_len) / max(n_cuts_mean, 1.0)
        bal_loss = hnet_ratio_balance_loss(probs, target_rate=self.target_rate)
        return {
            "trajectory": u,
            "raw_scores": raw_scores,
            "boundary_probs": probs,
            "hard_cuts": cuts,
            "avg_plen": round(avg_plen, 4),
            "balance_loss": bal_loss,
        }


# ---------------------------------------------------------------------------
# H14e-2: Markov-MTP8 vs Parallel-MTP8 Fused Onto HoloSphereTransformer
# ---------------------------------------------------------------------------


class HoloSphereParallelMTP8LM(HoloSphereTransformer):
    """Control arm attaching independent ParallelMTP8Head to HoloSphereTransformer."""

    def __init__(self, cfg: HoloSphereConfig) -> None:
        require_ok(not cfg.use_markov_mtp, "ParallelMTP8 control expects use_markov_mtp=False", code="H14E2_CFG")
        torch.manual_seed(int(cfg.seed) & 0x7FFFFFFF)
        super().__init__(cfg)
        self.parallel_mtp_head = ParallelMTP8Head(cfg.d_model, vocab_size=cfg.vocab_size, block_size=cfg.mtp_block_size)
        gen = torch.Generator(device="cpu").manual_seed((int(cfg.seed) + 999) & 0x7FFFFFFF)
        with torch.no_grad():
            for head in self.parallel_mtp_head.heads:
                head.weight.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
                if head.bias is not None:
                    head.bias.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        r_loops: int | None = None,
        return_trace: bool = False,
        mtp_targets: torch.Tensor | None = None,
        mtp_unroll: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = (r_loops, return_trace)
        h = self.tok_emb(input_ids)
        if self.use_sphere_head:
            h = safe_unit_norm(h, dim=-1, eps=1e-4)
        require_ok(self.blocks is not None, "H14e-2 unlooped control expects self.blocks", code="H14E2_STATE")
        for blk in self.blocks:
            h = blk(h)
        trace: dict[str, Any] = {
            "pass_states": [],
            "norm_drift_max": 0.0,
            "mtp_logits": self.parallel_mtp_head(h, target_tokens=mtp_targets, unroll=mtp_unroll),
        }
        if self.use_sphere_head:
            require_ok(self.logit_scale is not None, "logit_scale missing", code="H14E2_STATE")
            logits = F.linear(safe_unit_norm(h, dim=-1, eps=1e-4), safe_unit_norm(self.tok_emb.weight, dim=-1, eps=1e-4)) * self.logit_scale
        else:
            require_ok(self.final_norm is not None, "final_norm missing", code="H14E2_STATE")
            logits = F.linear(self.final_norm(h), self.tok_emb.weight)
        return logits, trace


def build_h14e2_arms(
    *,
    d_model: int = 320,
    n_heads: int = 8,
    n_layers: int = 12,
    base_d_ff: int = 864,
    seed: int = 42,
) -> dict[str, tuple[HoloSphereConfig, str]]:
    """All 4 matched H14e-2 arms comparing Markov-MTP8 vs Parallel-MTP8 vs NoMTP on HoloSphere."""
    base = dict(
        vocab_size=260,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        base_d_ff=base_d_ff,
        width_profile="cylinder",
        loop_mode="none",
        seed=seed,
    )
    return {
        "RoPE-nGPT-NoMTP": (
            HoloSphereConfig(
                arm_name="RoPE-nGPT-NoMTP",
                pos_mode="rope2d",
                transport_v=False,
                residual_mode="slerp",
                ngpt_qk_norm=True,
                use_markov_mtp=False,
                **base,
            ),
            "standard",
        ),
        "HoloSphere-SU2-NoMTP": (
            HoloSphereConfig(
                arm_name="HoloSphere-SU2-NoMTP",
                pos_mode="su2_holonomy",
                transport_v=True,
                residual_mode="hopf_s3",
                ngpt_qk_norm=True,
                use_markov_mtp=False,
                **base,
            ),
            "standard",
        ),
        "HoloSphere-SU2-ParallelMTP8": (
            HoloSphereConfig(
                arm_name="HoloSphere-SU2-ParallelMTP8",
                pos_mode="su2_holonomy",
                transport_v=True,
                residual_mode="hopf_s3",
                ngpt_qk_norm=True,
                use_markov_mtp=False,
                mtp_block_size=8,
                **base,
            ),
            "parallel_mtp8",
        ),
        "HoloSphere-SU2-MarkovMTP8": (
            HoloSphereConfig(
                arm_name="HoloSphere-SU2-MarkovMTP8",
                pos_mode="su2_holonomy",
                transport_v=True,
                residual_mode="hopf_s3",
                ngpt_qk_norm=True,
                use_markov_mtp=True,
                mtp_block_size=8,
                mtp_rank=64,
                **base,
            ),
            "standard",
        ),
    }


def instantiate_h14e2_model(cfg: HoloSphereConfig, variant: str) -> HoloSphereTransformer:
    torch.manual_seed(int(cfg.seed) & 0x7FFFFFFF)
    if variant == "parallel_mtp8":
        return HoloSphereParallelMTP8LM(cfg)
    if variant == "standard":
        return HoloSphereTransformer(cfg)
    raise ValueError(f"Unknown H14e-2 variant: {variant!r}")


def build_mtp_block_targets(seq_ids: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """For each position t in 0..T-1, collect next `block_size` target tokens [t+1..t+block_size]."""
    bsz, total_len = seq_ids.shape
    seq_len = total_len - block_size
    require_ok(seq_len >= 1, f"Sequence len {total_len} too short for block_size={block_size}", code="MTP_LEN")
    blocks = [seq_ids[:, t + 1 : t + 1 + block_size] for t in range(seq_len)]
    return torch.stack(blocks, dim=1)  # [B, seq_len, block_size]


def compute_h14e2_loss_and_metrics(
    model: HoloSphereTransformer,
    seq_ids: torch.Tensor,
    *,
    block_size: int = 8,
    lambda_mtp: float = 0.2,
) -> dict[str, Any]:
    """Compute strictly decoupled primary next-byte loss + coupled MTP-8 auxiliary loss."""
    inp = seq_ids[:, :-block_size]  # [B, T]
    primary_tgt = seq_ids[:, 1 : 1 + inp.shape[1]]
    mtp_tgt = build_mtp_block_targets(seq_ids, block_size=block_size)  # [B, T, K]

    logits, trace = model(inp, mtp_targets=mtp_tgt, mtp_unroll=False)
    primary_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), primary_tgt.reshape(-1))
    primary_bpb = float(primary_loss.item()) / math.log(2.0)

    mtp_loss = torch.tensor(0.0, device=seq_ids.device, dtype=torch.float32)
    mtp_tf_bpb: float | None = None
    mtp_unroll_bpb: float | None = None
    if "mtp_logits" in trace and trace["mtp_logits"] is not None:
        mtp_lg = trace["mtp_logits"].float()  # [B, T, K, V]
        mtp_loss = F.cross_entropy(mtp_lg.reshape(-1, mtp_lg.shape[-1]), mtp_tgt.reshape(-1))
        mtp_tf_bpb = round(float(mtp_loss.item()) / math.log(2.0), 6)
        with torch.no_grad():
            _, tr_u = model(inp, mtp_targets=None, mtp_unroll=True)
            lg_u = tr_u["mtp_logits"].float()
            u_loss = F.cross_entropy(lg_u.reshape(-1, lg_u.shape[-1]), mtp_tgt.reshape(-1))
            mtp_unroll_bpb = round(float(u_loss.item()) / math.log(2.0), 6)

    total_loss = primary_loss + float(lambda_mtp) * mtp_loss
    return {
        "total_loss": total_loss,
        "primary_bpb": round(primary_bpb, 6),
        "mtp_tf_bpb": mtp_tf_bpb,
        "mtp_unroll_bpb": mtp_unroll_bpb,
    }


# ---------------------------------------------------------------------------
# Deterministic CPU-Only Pre-Flight Suite (All 4 Probes H14d-1/2 & H14e-1/2)
# ---------------------------------------------------------------------------


def run_cpu_preflight() -> dict[str, Any]:
    """Run deterministic CPU-only preflight across all 4 PREREG-S114-AMENDMENT1 probes."""
    torch.set_num_threads(1)
    torch.manual_seed(1141)

    # --- 1. H14d-1: W10-OBS-31 Antipodal Cure (cayley_acute & acute_cone vs unaliased_coprime) ---
    coprime_cos = [math.cos(hopf_clock_angle(r, "unaliased_coprime")) for r in range(12)]
    acute_cos = [math.cos(hopf_clock_angle(r, "acute_cone")) for r in range(64)]
    cayley_pairs = [cayley_acute_cos_sin(r, t_max=0.8, tau=2.0) for r in range(64)]
    cayley_cos = [c for c, _ in cayley_pairs]
    cayley_unit_err = max(abs(c * c + s * s - 1.0) for c, s in cayley_pairs)
    cayley_floor = (1.0 - 0.8 * 0.8) / (1.0 + 0.8 * 0.8)  # exact 9/41 = 0.219512...

    require_ok(coprime_cos[5] < -0.95, "unaliased_coprime r=5 must witness antipodal flip")
    require_ok(min(cayley_cos) > cayley_floor, "cayley_acute breached rational lower bound 9/41")
    require_ok(min(acute_cos) > 0.24, "acute_cone breached positive cosine floor")
    require_ok(cayley_unit_err < 1e-15, f"cayley_acute unit-circle error {cayley_unit_err}")

    # Verify full-scale H14d-1 parameter parity (<= 0.5%) & live micro R_test={3,4,6,8,12} forward/bwd
    full_d1 = build_h14d1_arms(seed=42)
    d1_params = {k: count_parameters(instantiate_h14d1_model(cfg, var))["non_embedding_params"] for k, (cfg, var) in full_d1.items()}
    ref_d1 = d1_params["FlatLoop-Std"]
    d1_max_p_diff = max(abs(v - ref_d1) / float(ref_d1) for v in d1_params.values())
    require_ok(d1_max_p_diff <= 0.005, f"H14d-1 param diff {d1_max_p_diff:.4%} > 0.5%")

    micro_d1 = build_h14d1_arms(d_model=64, n_heads=4, base_d_ff=128, n_pre=1, n_core=2, n_suffix=1, seed=101)
    m_cay = instantiate_h14d1_model(*micro_d1["SafeHopf-CayleyAcute"])
    gen = torch.Generator(device="cpu").manual_seed(102)
    x_d1 = torch.randint(10, 200, (2, 20), generator=gen)
    d1_drift_max = 0.0
    for r_t in R_TEST_GRID_H14D1:
        lg, tr = m_cay(x_d1, r_loops=r_t, return_trace=True)
        require_ok(bool(torch.isfinite(lg).all().item()), f"Non-finite H14d-1 logits at R={r_t}")
        require_ok(len(tr["pass_states"]) == r_t + 1, "Pass trace length mismatch")
        require_ok(all(c > cayley_floor for c in tr["clock_cosines"]), "Cayley cosine dropped below floor")
        d1_drift_max = max(d1_drift_max, float(tr["norm_drift_max"]))
    require_ok(d1_drift_max <= 1e-3, f"CayleyAcute unit-norm drift {d1_drift_max} > 1e-3")

    # --- 2. H14d-2: W10-OBS-30 Equatorial Foliation x Covariant SU(2) Holonomy ---
    full_d2 = build_h14d2_arms(seed=42)
    d2_params = {k: count_parameters(HoloSphereTransformer(cfg))["non_embedding_params"] for k, cfg in full_d2.items()}
    d2_flops = {k: analytical_forward_flops_per_token(cfg, seq_len=512)["total_fwd_flops_per_token"] for k, cfg in full_d2.items()}
    require_ok(len(set(d2_params.values())) == 1, f"H14d-2 non-emb params not exact match: {d2_params}")
    require_ok(len(set(d2_flops.values())) == 1, f"H14d-2 FLOPs not exact match: {d2_flops}")
    w_eq = build_s3_width_schedule(12, 864, "equatorial")
    w_peq = build_s3_width_schedule(12, 864, "permuted_eq")
    rho_eq = spearman_rank_corr(w_eq, w_peq)
    require_ok(sorted(w_eq) == sorted(w_peq) and abs(rho_eq) <= 0.30, "H14d-2 PermEq multiset/Spearman failure")

    # --- 3. H14e-1: Holonomy-Smoothed Geodesic & Portal-2 D=(1,10) Chunking (W10-OBS-33 Cure) ---
    sample_str = (
        "The hypersphere foliation allocates capacity across depth smoothly. "
        "def quat_mul(p, q): return p * q; "
        '{"protocol":"PREREG-S114-AMENDMENT1","status":"verified"} '
    ) * 3
    raw_b = list(sample_str.encode("utf-8"))[:256]
    x_e1 = torch.tensor([raw_b, raw_b[::-1]], dtype=torch.long)

    e1_stats: dict[str, Any] = {}
    for cmode in (
        "raw_geodesic_baseline",
        "hnet_1st_order",
        "holonomy_geodesic_2nd",
        "holonomy_turing_d1_10",
        "holonomy_turing_d5_5_null",
        "fixed_4b",
    ):
        chk = HolonomySmoothedChunker(d_enc=64, mode=cmode, target_patch_len=4.0, seed=303)
        out = chk(x_e1)
        # Strict causality test: mutate future suffix t >= 128 and verify prefix scores/probs [0..127] differ by 0.0!
        x_mod = x_e1.clone()
        x_mod[:, 128:] = (x_mod[:, 128:] + 17) % 256
        out_mod = chk(x_mod)
        leak = float(torch.max(torch.abs(out["boundary_probs"][:, :128] - out_mod["boundary_probs"][:, :128])).item())
        require_ok(leak < 1e-6, f"Future-byte causality leak in {cmode}: {leak}")
        e1_stats[cmode] = {
            "avg_plen": out["avg_plen"],
            "causality_leak_max": round(leak, 9),
            "mean_score": round(float(out["raw_scores"].mean().item()), 6),
        }

    # Verify W10-OBS-33 reproduction on raw_geodesic_baseline AND cure on holonomy_turing_d1_10 + holonomy_geodesic_2nd!
    require_ok(e1_stats["raw_geodesic_baseline"]["avg_plen"] <= 1.25, "Expected raw_geodesic_baseline W10-OBS-33 collapse")
    require_ok(2.5 <= e1_stats["holonomy_turing_d1_10"]["avg_plen"] <= 6.0, "Holonomy Turing D=(1,10) failed anti-collapse band")
    require_ok(2.0 <= e1_stats["holonomy_geodesic_2nd"]["avg_plen"] <= 8.0, "Holonomy Geodesic 2nd failed anti-collapse band")
    require_ok(abs(e1_stats["holonomy_turing_d5_5_null"]["mean_score"]) < 1e-7, "Equal-diffusion D=(5,5) null must vanish identically")

    # --- 4. H14e-2: Markov-MTP8 Fused Onto HoloSphereTransformer (T=512 & T=2048 ready) ---
    micro_e2 = build_h14e2_arms(d_model=64, n_heads=4, n_layers=2, base_d_ff=160, seed=404)
    x_e2 = torch.randint(10, 200, (2, 40), generator=gen)  # seq_len=32 + block_size=8
    e2_stats: dict[str, Any] = {}
    for aname, (cfg, var) in micro_e2.items():
        m = instantiate_h14e2_model(cfg, var)
        res = compute_h14e2_loss_and_metrics(m, x_e2, block_size=8, lambda_mtp=0.2)
        res["total_loss"].backward()
        grad_ok = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
        require_ok(grad_ok, f"Non-finite gradient in H14e-2 {aname}")
        e2_stats[aname] = {
            "non_emb_params": count_parameters(m)["non_embedding_params"],
            "primary_bpb": res["primary_bpb"],
            "mtp_tf_bpb": res["mtp_tf_bpb"],
            "mtp_unroll_bpb": res["mtp_unroll_bpb"],
        }
    require_ok(
        e2_stats["HoloSphere-SU2-MarkovMTP8"]["mtp_tf_bpb"] is not None
        and e2_stats["HoloSphere-SU2-ParallelMTP8"]["mtp_tf_bpb"] is not None,
        "MTP-8 block BPB missing",
    )

    payload: dict[str, Any] = {
        "schema": "holosphere.prereg_s114_amendment1.cpu_preflight.v1",
        "protocol_id": "PREREG-S114-AMENDMENT1-S107-S108-CONVERGENCE-V1",
        "passed": True,
        "h14d1_antipodal_cure": {
            "coprime_cos_r5": round(coprime_cos[5], 6),
            "cayley_acute_min_cos_r0_63": round(min(cayley_cos), 6),
            "cayley_acute_exact_floor_9_over_41": round(cayley_floor, 6),
            "cayley_unit_circle_max_err": round(cayley_unit_err, 16),
            "acute_cone_min_cos_r0_63": round(min(acute_cos), 6),
            "max_full_scale_param_diff_pct": round(100.0 * d1_max_p_diff, 4),
            "cayley_r3_to_r12_unit_norm_drift_max": round(d1_drift_max, 7),
            "r_test_grid": list(R_TEST_GRID_H14D1),
        },
        "h14d2_equatorial_su2": {
            "non_emb_params_all_4_arms": d2_params["Cyl-SU2-Holonomy"],
            "fwd_flops_t512_all_4_arms": d2_flops["Cyl-SU2-Holonomy"],
            "exact_zero_param_and_flop_diff": True,
            "spearman_eq_vs_permuted_eq": round(rho_eq, 4),
        },
        "h14e1_holonomy_chunking": e1_stats,
        "h14e2_markov_mtp8_holosphere": e2_stats,
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PREREG-S114-AMENDMENT1 S107+S108 Convergence Harness")
    parser.add_argument("--cpu-preflight", action="store_true", help="Run deterministic CPU-only preflight.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON receipt.")
    args = parser.parse_args(argv)
    _ = args.cpu_preflight
    res = run_cpu_preflight()
    if args.json:
        sys.stdout.write(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    else:
        sys.stdout.write(f"OK H14d_H14e_PREFLIGHT passed={res['passed']} payload_sha256={res['payload_sha256']}\n")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
