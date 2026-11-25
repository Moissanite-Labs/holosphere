"""Flagship HoloSphereTransformer: unified kernel-isomorphic hypersphere LM & exact CS-08 / M1-H1 accounting."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from holosphere.errors import require_ok
from holosphere.geometry.hopf import build_s3_width_schedule, cayley_acute_cos_sin, hopf_clock_angle
from holosphere.geometry.su2 import safe_unit_norm
from holosphere.kernels.dispatch import _resolve_backend, fused_spherical_step
from holosphere.models.blocks import HoloSphereBlock
from holosphere.models.config import HoloSphereConfig
from holosphere.models.controls import RMSNorm
from holosphere.models.heads import MarkovMTP8Head
from holosphere.telemetry import HoloSphereStepTelemetry, HoloSphereTuningProfile, build_step_telemetry


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


class HoloSphereTransformer(nn.Module):
    """Unified kernel-isomorphic transformer consolidating S106-S114 primitives.

    Supports:
      - Covariant (Q, K[, V]) SU(2) path-ordered holonomy scan (H7b / H14a / H14c)
      - Native Hopf-S^3 (H6c) or antipodal-safe SLERP (nGPT) or flat residual updates
      - Non-antipodal acute-cone Safe-Hopf looped core (H7c + W10-OBS-31 fix)
      - S^3-latitude width foliation schedules (H7a)
      - Optional Beltrami 2x2 Cayley, Poloidal Bingham, & Darcy EMA operators (S109/S110)
      - Optional Markov-MTP8 coupled multi-token draft head (S108 H8c2)
      - Cryptographic Core weight SHA-256 pinning for S111 Weft audited live-update shells.
    """

    def __init__(self, cfg: HoloSphereConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        gen = _seeded_gen(cfg.seed)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        with torch.no_grad():
            self.tok_emb.weight.copy_(torch.randn(self.tok_emb.weight.shape, generator=gen) * 0.02)
            if cfg.residual_mode == "slerp":
                self.tok_emb.weight.copy_(safe_unit_norm(self.tok_emb.weight, dim=-1, eps=1e-4))

        self.is_looped = cfg.loop_mode != "none"
        self.layer_widths: list[int] = []

        def _make_block(d_ff: int, b_seed: int) -> HoloSphereBlock:
            return HoloSphereBlock(
                cfg.d_model,
                cfg.n_heads,
                d_ff,
                pos_mode=cfg.pos_mode,
                transport_v=cfg.transport_v,
                angle_scale=cfg.angle_scale,
                residual_mode=cfg.residual_mode,
                ngpt_qk_norm=cfg.ngpt_qk_norm,
                use_cayley=cfg.use_cayley,
                cayley_rank=cfg.cayley_rank,
                use_bingham=cfg.use_bingham,
                bingham_yield=cfg.bingham_yield,
                bingham_tau=cfg.bingham_tau,
                use_darcy=cfg.use_darcy,
                backend=cfg.backend,
                seed=b_seed,
            )

        dff_offset = (cfg.d_model // 4) if (cfg.pos_mode == "su2_holonomy" and cfg.compensate_su2_dff) else 0

        if not self.is_looped:
            widths = build_s3_width_schedule(cfg.n_layers, cfg.base_d_ff, cfg.width_profile)
            if dff_offset > 0:
                widths = [w - dff_offset for w in widths]
                require_ok(all(w > 0 for w in widths), "SU(2) d_ff offset drove d_ff <= 0", code="DFF_UNDERFLOW")
            self.layer_widths = widths
            self.blocks: nn.ModuleList | None = nn.ModuleList(
                [_make_block(widths[i], cfg.seed + 100 + i) for i in range(cfg.n_layers)]
            )
            self.pre_blocks: nn.ModuleList | None = None
            self.core_blocks: nn.ModuleList | None = None
            self.suffix_blocks: nn.ModuleList | None = None
            self.loop_gate_logits: nn.Parameter | None = None
            self.register_buffer("rand_pass_rots", torch.empty(0))
        else:
            self.blocks = None
            eff_dff = cfg.base_d_ff - dff_offset
            require_ok(eff_dff > 0, "SU(2) d_ff offset drove looped d_ff <= 0", code="DFF_UNDERFLOW")
            n_phys = cfg.n_pre + cfg.n_core + cfg.n_suffix
            self.layer_widths = [eff_dff] * n_phys
            self.pre_blocks = nn.ModuleList([_make_block(eff_dff, cfg.seed + 200 + i) for i in range(cfg.n_pre)])
            self.core_blocks = nn.ModuleList([_make_block(eff_dff, cfg.seed + 300 + i) for i in range(cfg.n_core)])
            self.suffix_blocks = nn.ModuleList([_make_block(eff_dff, cfg.seed + 400 + i) for i in range(cfg.n_suffix)])
            self.loop_gate_logits = nn.Parameter(torch.full((1,), -0.85))
            if cfg.loop_mode == "sphere_randrot":
                rot_gen = _seeded_gen(cfg.seed + 777)
                rots: list[torch.Tensor] = []
                for _ in range(12):
                    q_mat, r_mat = torch.linalg.qr(torch.randn((cfg.d_model, cfg.d_model), generator=rot_gen))
                    signs = torch.sign(torch.diag(r_mat))
                    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                    rots.append(q_mat * signs.unsqueeze(0))
                self.register_buffer("rand_pass_rots", torch.stack(rots, dim=0))
            else:
                self.register_buffer("rand_pass_rots", torch.empty(0))

        self.use_sphere_head = (cfg.residual_mode == "slerp") or (
            cfg.loop_mode
            in ("safe_hopf_acute", "safe_hopf_cayley_acute", "clifford_coprime", "sphere_noclock", "sphere_randrot")
        )
        if self.use_sphere_head:
            self.final_norm: RMSNorm | None = None
            self.logit_scale: nn.Parameter | None = nn.Parameter(torch.full((1,), math.sqrt(float(cfg.d_model))))
        else:
            self.final_norm = RMSNorm(cfg.d_model)
            self.logit_scale = None

        self.mtp_head: MarkovMTP8Head | None = (
            MarkovMTP8Head(cfg.d_model, vocab_size=cfg.vocab_size, block_size=cfg.mtp_block_size, markov_rank=cfg.mtp_rank)
            if cfg.use_markov_mtp
            else None
        )

        # One-way SiliSense telemetry & step-boundary tuning state (zero weight mutation, zero sensor I/O)
        self._tuning_profile: HoloSphereTuningProfile = HoloSphereTuningProfile(backend=cfg.backend)
        self._tuning_profile.validate()
        self._last_step_telemetry: HoloSphereStepTelemetry | None = None
        self._step_counter: int = 0
        self._var_depth_gen: torch.Generator = _seeded_gen(cfg.seed + 9110)

        # Deterministic weight initialization from cfg.seed (zero global RNG leakage)
        with torch.no_grad():
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.copy_(torch.randn(mod.weight.shape, generator=gen) * 0.02)
            if self.use_sphere_head:
                self.normalize_sphere_weights_()

    @property
    def tuning_profile(self) -> HoloSphereTuningProfile:
        """Current validated step-boundary execution profile."""
        return self._tuning_profile

    @property
    def last_step_telemetry(self) -> HoloSphereStepTelemetry | None:
        """Most recent opt-in step telemetry record (None if telemetry has not been emitted)."""
        return self._last_step_telemetry

    def apply_tuning_profile(self, profile: HoloSphereTuningProfile) -> HoloSphereTuningProfile:
        """Apply a validated semantics-preserving tuning profile between steps (zero weight mutation)."""
        require_ok(
            isinstance(profile, HoloSphereTuningProfile),
            f"expected HoloSphereTuningProfile, got {type(profile).__name__}",
            code="TUNE_TYPE",
        )
        profile.validate()
        self._tuning_profile = profile
        for mod in self.modules():
            if isinstance(mod, HoloSphereBlock):
                mod.backend = profile.backend
                mod.attn.backend = profile.backend
        return profile

    def sample_var_depth_passes(self) -> int:
        """Sample a deterministic batch-uniform loop count r in [r_train_min, r_train_max]."""
        lo = int(self.cfg.r_train_min)
        hi = int(self.cfg.r_train_max)
        if lo == hi:
            return lo
        offset = int(torch.randint(0, hi - lo + 1, (1,), generator=self._var_depth_gen).item())
        return lo + offset

    def record_step_telemetry(
        self,
        *,
        batch_size: int,
        seq_len: int,
        r_loops: int,
        step_ms: float,
        norm_drift_max: float = 0.0,
        resolved_backend: str = "torch_cpu",
        step_idx: int | None = None,
    ) -> HoloSphereStepTelemetry:
        """Record a validated step-boundary telemetry snapshot for external governors (e.g. silisense)."""
        idx = self._step_counter if step_idx is None else int(step_idx)
        rec = build_step_telemetry(
            step_idx=idx,
            seq_len=seq_len,
            batch_size=batch_size,
            r_loops=r_loops,
            backend=resolved_backend,
            block_t=self._tuning_profile.block_t,
            step_ms=step_ms,
            norm_drift_max=norm_drift_max,
        )
        self._last_step_telemetry = rec
        self._step_counter = idx + 1
        return rec

    def _run_core_pass(self, h: torch.Tensor) -> torch.Tensor:
        require_ok(self.core_blocks is not None, "core_blocks missing", code="MODEL_STATE")
        out = h
        for blk in self.core_blocks:
            out = blk(out)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        r_loops: int | None = None,
        return_trace: bool = False,
        mtp_targets: torch.Tensor | None = None,
        mtp_unroll: bool = False,
        emit_telemetry: bool = False,
        step_ms: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass returning ``(logits, trace)`` (plus optional ``mtp_logits`` / ``step_telemetry``)."""
        t0 = time.perf_counter() if (emit_telemetry and step_ms is None) else 0.0
        active_backend = self._tuning_profile.backend

        h = self.tok_emb(input_ids)
        if self.use_sphere_head:
            h = safe_unit_norm(h, dim=-1, eps=1e-4)

        trace: dict[str, Any] = {"pass_states": [], "norm_drift_max": 0.0}
        passes_used = 1

        if not self.is_looped:
            require_ok(self.blocks is not None, "blocks missing", code="MODEL_STATE")
            for blk in self.blocks:
                h = blk(h)
                if self.cfg.residual_mode == "slerp":
                    norms = torch.linalg.vector_norm(h.float(), ord=2, dim=-1)
                    trace["norm_drift_max"] = max(trace["norm_drift_max"], float(torch.max(torch.abs(norms - 1.0)).item()))
        else:
            require_ok(
                self.pre_blocks is not None
                and self.core_blocks is not None
                and self.suffix_blocks is not None
                and self.loop_gate_logits is not None,
                "looped blocks missing",
                code="MODEL_STATE",
            )
            if r_loops is not None:
                passes = int(r_loops)
            elif self._tuning_profile.r_loops_override is not None:
                passes = int(self._tuning_profile.r_loops_override)
            elif self.cfg.var_depth_train and self.training:
                passes = self.sample_var_depth_passes()
            else:
                passes = self.cfg.r_train
            require_ok(passes >= 1, f"r_loops must be >= 1, got {passes}", code="BAD_RLOOPS")
            passes_used = passes
            trace["r_loops_used"] = passes

            for blk in self.pre_blocks:
                h = blk(h)
            if return_trace:
                trace["pass_states"].append(h.detach())

            for r in range(passes):
                if self.cfg.use_grad_checkpoint and self.training:
                    core_out = grad_checkpoint(self._run_core_pass, h, use_reentrant=False)
                else:
                    core_out = self._run_core_pass(h)

                if self.cfg.loop_mode == "flat":
                    gate = torch.sigmoid(self.loop_gate_logits)
                    h = h + gate * core_out
                elif self.cfg.loop_mode == "sphere_noclock":
                    alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
                    h = fused_spherical_step(h, core_out, alpha_r, mode="slerp", eps=1e-4, backend=active_backend)
                elif self.cfg.loop_mode == "sphere_randrot":
                    alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
                    q_rot = self.rand_pass_rots[r % self.rand_pass_rots.shape[0]].to(core_out.dtype)
                    rot_out = torch.matmul(core_out, q_rot.transpose(0, 1))
                    h = fused_spherical_step(h, rot_out, alpha_r, mode="slerp", eps=1e-4, backend=active_backend)
                elif self.cfg.loop_mode == "safe_hopf_acute":
                    alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
                    phi_val = hopf_clock_angle(
                        r,
                        clock_law="acute_cone",
                        phi_max=self.cfg.acute_phi_max,
                        tau=self.cfg.acute_tau,
                    )
                    h = fused_spherical_step(
                        h, core_out, alpha_r, mode="safe_hopf_loop", phi=phi_val, eps=1e-4, backend=active_backend
                    )
                elif self.cfg.loop_mode == "safe_hopf_cayley_acute":
                    alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
                    c_val, s_val = cayley_acute_cos_sin(
                        r,
                        t_max=self.cfg.cayley_acute_t_max,
                        tau=self.cfg.acute_tau,
                    )
                    h = fused_spherical_step(
                        h,
                        core_out,
                        alpha_r,
                        mode="safe_hopf_cayley_loop",
                        c_s=(c_val, s_val),
                        eps=1e-4,
                        backend=active_backend,
                    )
                elif self.cfg.loop_mode == "clifford_coprime":
                    alpha_r = torch.sigmoid(self.loop_gate_logits) * (0.45 * math.pi)
                    phi_val = hopf_clock_angle(r, clock_law="unaliased_coprime", r_train=self.cfg.r_train)
                    h = fused_spherical_step(
                        h, core_out, alpha_r, mode="safe_hopf_loop", phi=phi_val, eps=1e-4, backend=active_backend
                    )
                else:
                    raise ValueError(f"Unknown loop_mode: {self.cfg.loop_mode!r}")

                if self.use_sphere_head:
                    norms = torch.linalg.vector_norm(h.float(), ord=2, dim=-1)
                    trace["norm_drift_max"] = max(trace["norm_drift_max"], float(torch.max(torch.abs(norms - 1.0)).item()))
                if return_trace:
                    trace["pass_states"].append(h.detach())

            for blk in self.suffix_blocks:
                h = blk(h)

        if self.mtp_head is not None:
            trace["mtp_logits"] = self.mtp_head(h, target_tokens=mtp_targets, unroll=mtp_unroll)

        if self.use_sphere_head:
            require_ok(self.logit_scale is not None, "logit_scale missing", code="MODEL_STATE")
            h_norm = safe_unit_norm(h, dim=-1, eps=1e-4)
            w_emb = safe_unit_norm(self.tok_emb.weight, dim=-1, eps=1e-4)
            logits = F.linear(h_norm, w_emb) * self.logit_scale
        else:
            require_ok(self.final_norm is not None, "final_norm missing", code="MODEL_STATE")
            logits = F.linear(self.final_norm(h), self.tok_emb.weight)

        if emit_telemetry:
            elapsed_ms = float(step_ms) if step_ms is not None else (time.perf_counter() - t0) * 1000.0
            resolved_backend = _resolve_backend(active_backend, h)
            tel = self.record_step_telemetry(
                batch_size=int(input_ids.shape[0]),
                seq_len=int(input_ids.shape[1]),
                r_loops=passes_used,
                step_ms=elapsed_ms,
                norm_drift_max=float(trace["norm_drift_max"]),
                resolved_backend=resolved_backend,
            )
            trace["step_telemetry"] = tel.to_dict()
            trace["tuning_profile_sha256"] = self._tuning_profile.profile_sha256()

        return logits, trace

    @torch.no_grad()
    def normalize_sphere_weights_(self) -> None:
        """Retract nGPT/hypersphere linear rows onto the unit sphere after each optimizer step."""
        if not self.use_sphere_head:
            return
        self.tok_emb.weight.copy_(safe_unit_norm(self.tok_emb.weight, dim=-1, eps=1e-4))
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(safe_unit_norm(mod.weight, dim=-1, eps=1e-4))

    @torch.no_grad()
    def core_weights_sha256(self) -> str:
        """Deterministic SHA-256 digest of all model parameter bytes (for S111 frozen-Core audits)."""
        h = hashlib.sha256()
        for name, param in sorted(self.named_parameters(), key=lambda kv: kv[0]):
            h.update(name.encode("utf-8"))
            h.update(param.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
        return h.hexdigest()


def count_parameters(model: HoloSphereTransformer) -> dict[str, int]:
    """Exact total, embedding, and non-embedding parameter accounting."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel()
    return {
        "total_params": int(total),
        "embedding_params": int(emb),
        "non_embedding_params": int(total - emb),
    }


def analytical_forward_flops_per_token(
    cfg: HoloSphereConfig,
    seq_len: int,
    r_loops: int | None = None,
) -> dict[str, int]:
    """Exact forward MAC -> FLOPs/token accounting (2 FLOPs per MAC, incl. SU(2) scan & covariant transport)."""
    d = cfg.d_model
    v = cfg.vocab_size
    dff_offset = (d // 4) if (cfg.pos_mode == "su2_holonomy" and cfg.compensate_su2_dff) else 0

    def _layer_flops(d_ff: int) -> int:
        proj = 2 * (4 * d * d + 3 * d * d_ff)
        attn_qk_av = 2 * seq_len * d
        pos_extra = 0
        if cfg.pos_mode == "su2_holonomy":
            n_q = d // 4
            w_omega_flops = 2 * d * (3 * n_q)
            scan_steps = max(1, int(math.ceil(math.log2(max(2, seq_len)))))
            scan_flops = 28 * n_q * scan_steps
            # Q & K transport (2 * 28 * n_q) + optional covariant V transport & O un-transport (2 * 28 * n_q)
            n_transports = 4 if cfg.transport_v else 2
            pos_extra = w_omega_flops + scan_flops + n_transports * 28 * n_q
        elif cfg.pos_mode in ("liere4d", "block4_rand_ortho"):
            pos_extra = 2 * 16 * (d // 4)
        cayley_extra = 0
        if cfg.use_cayley:
            n_pairs = d // 2
            cayley_extra = 2 * d * cfg.cayley_rank + 2 * cfg.cayley_rank * n_pairs + 8 * n_pairs
        return proj + attn_qk_av + pos_extra + cayley_extra

    if cfg.loop_mode == "none":
        widths = [w - dff_offset for w in build_s3_width_schedule(cfg.n_layers, cfg.base_d_ff, cfg.width_profile)]
        core_flops = sum(_layer_flops(w) for w in widths)
        eff_layers = cfg.n_layers
    else:
        passes = cfg.r_train if r_loops is None else int(r_loops)
        eff_dff = cfg.base_d_ff - dff_offset
        core_flops = (
            cfg.n_pre * _layer_flops(eff_dff)
            + passes * cfg.n_core * _layer_flops(eff_dff)
            + cfg.n_suffix * _layer_flops(eff_dff)
        )
        if cfg.loop_mode in ("safe_hopf_acute", "safe_hopf_cayley_acute", "clifford_coprime"):
            core_flops += passes * (6 * d)
        elif cfg.loop_mode == "sphere_randrot":
            core_flops += passes * (2 * d * d)
        eff_layers = cfg.n_pre + passes * cfg.n_core + cfg.n_suffix

    lm_head_flops = 2 * d * v
    total_fwd = core_flops + lm_head_flops
    return {
        "effective_layers": int(eff_layers),
        "core_fwd_flops_per_token": int(core_flops),
        "lm_head_fwd_flops_per_token": int(lm_head_flops),
        "total_fwd_flops_per_token": int(total_fwd),
        "est_train_flops_per_token": int(3 * total_fwd),
    }


@torch.no_grad()
def quantize_tensor_rtn_group(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Symmetric RTN per-output-channel group quantization (3-bit or 4-bit)."""
    require_ok(bits in (3, 4), f"bits must be 3 or 4, got {bits}", code="PTQ_BITS")
    require_ok(w.ndim == 2, f"expected 2D linear weight, got {tuple(w.shape)}", code="PTQ_SHAPE")
    out_dim, in_dim = w.shape
    qmax = float((1 << (bits - 1)) - 1)
    wf = w.float()
    pad = (group_size - (in_dim % group_size)) % group_size
    if pad > 0:
        wf = F.pad(wf, (0, pad))
    groups = wf.reshape(out_dim, -1, group_size)
    scale = torch.clamp(torch.amax(torch.abs(groups), dim=-1, keepdim=True) / qmax, min=1e-8)
    deq = (torch.clamp(torch.round(groups / scale), min=-qmax, max=qmax) * scale).reshape(out_dim, in_dim + pad)[:, :in_dim]
    return deq.to(w.dtype)


@torch.no_grad()
def apply_ptq_copy(
    model: HoloSphereTransformer,
    bits: int,
    group_size: int = 128,
    core_only: bool = False,
) -> HoloSphereTransformer:
    """Return a deep copy with RTN group quantization applied to linear layers."""
    q_model = copy.deepcopy(model)
    target_root: nn.Module = q_model.core_blocks if (core_only and q_model.core_blocks is not None) else q_model
    for mod in target_root.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.copy_(quantize_tensor_rtn_group(mod.weight, bits=bits, group_size=group_size))
    return q_model


@torch.no_grad()
def measure_m1_h1_loop_drift(
    model_fp: HoloSphereTransformer,
    input_ids: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    r_eval: int = 3,
) -> dict[str, Any]:
    """Measure Track S96 M1-H1 loop quantization drift (Delta_r / Delta_1 and KL)."""
    require_ok(model_fp.is_looped, "measure_m1_h1_loop_drift requires a looped HoloSphereTransformer", code="M1H1_LOOP")
    model_q = apply_ptq_copy(model_fp, bits=bits, group_size=group_size, core_only=True)
    logits_fp_r, trace_fp_r = model_fp(input_ids, r_loops=r_eval, return_trace=True)
    logits_q_r, trace_q_r = model_q(input_ids, r_loops=r_eval, return_trace=True)
    _, trace_fp_1 = model_fp(input_ids, r_loops=1, return_trace=True)
    _, trace_q_1 = model_q(input_ids, r_loops=1, return_trace=True)

    delta_r = float(torch.linalg.vector_norm((trace_q_r["pass_states"][-1] - trace_fp_r["pass_states"][-1]).float()).item())
    delta_1 = float(torch.linalg.vector_norm((trace_q_1["pass_states"][-1] - trace_fp_1["pass_states"][-1]).float()).item())
    drift_ratio = delta_r / max(delta_1, 1e-8)

    p_fp = F.softmax(logits_fp_r.float(), dim=-1)
    kl_r = float(torch.mean(torch.sum(p_fp * (F.log_softmax(logits_fp_r.float(), dim=-1) - F.log_softmax(logits_q_r.float(), dim=-1)), dim=-1)).item())
    return {
        "arm": model_fp.cfg.arm_name,
        "bits": bits,
        "r_eval": r_eval,
        "delta_r": round(delta_r, 6),
        "delta_1": round(delta_1, 6),
        "delta_r_over_delta_1": round(drift_ratio, 6),
        "kl_nats": round(kl_r, 6),
    }
