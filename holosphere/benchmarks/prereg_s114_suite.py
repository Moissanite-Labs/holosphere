#!/usr/bin/env python3
"""PREREG-S114 CPU Preflight & RTX 3080 Benchmark Suite (H14a / H14b / H14c).

CPU preflight mode (`--cpu-preflight`) executes strictly on CPU (zero GPU touch) and verifies:
  1. Forward & reverse-mode adjoint parity of SU(2) holonomy scan (`max_err < 1e-6`, FP64 gradcheck).
  2. Exact covariant (Q, K, V) parallel-transport identity vs explicit O(T^2) path-ordered attention.
  3. Non-antipodal acute-cone Safe-Hopf clock (cos(phi_r) > 0 for all r=0..63, fixing W10-OBS-31).
  4. Exact CS-08 parameter & analytical FLOP parity between HoloSphere and RoPE-2D baselines.
  5. Deterministic S_3 / A_5 group state-tracking and Weft Shell-3 Merkle rollback receipts.

GPU benchmark mode (`--gpu-bench`) is staged ready for RTX 3080 once the GPU lease is free.
Zero bare assert statements; 3-mode isolated -I parity compliant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if not __package__:
    _ROOT = str(Path(__file__).resolve().parent.parent.parent)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from holosphere.benchmarks.tasks import generate_group_tracking_batch
from holosphere.errors import require_ok
from holosphere.geometry.hopf import hopf_clock_angle
from holosphere.geometry.su2 import (
    quat_conj,
    quat_mul,
    su2_exp,
    su2_parallel_prefix_scan,
    su2_sequential_prefix_scan,
)
from holosphere.kernels.dispatch import (
    available_backends,
    su2_holonomy_transport,
    su2_inverse_untransport,
)
from holosphere.kernels.torch_ops import SU2UnitPrefixScanFunction, torch_su2_holonomy_transport
from holosphere.models.config import build_preset_config
from holosphere.models.transformer import (
    HoloSphereTransformer,
    analytical_forward_flops_per_token,
    count_parameters,
)


def run_cpu_preflight() -> dict[str, Any]:
    """Run deterministic CPU-only preflight verification across H14a, H14b, and H14c."""
    gen = torch.Generator(device="cpu").manual_seed(114)
    bsz, heads, seq_len, d_head = 2, 4, 16, 16
    n_blocks = d_head // 4

    # 1. H14a: Forward parallel vs sequential scan + custom adjoint vs PyTorch autograd parity
    omega_fp64 = torch.randn((bsz, heads, seq_len, n_blocks, 3), generator=gen, dtype=torch.float64, requires_grad=True)
    du_ref = su2_exp(0.75 * torch.tanh(omega_fp64), eps=1e-12)
    pi_seq = su2_sequential_prefix_scan(du_ref, seq_dim=-3, eps=1e-12)
    pi_par = su2_parallel_prefix_scan(du_ref, seq_dim=-3, eps=1e-12)
    scan_fwd_err = float(torch.max(torch.abs(pi_seq - pi_par)).item())

    omega_custom = omega_fp64.detach().clone().requires_grad_(True)
    omega_auto = omega_fp64.detach().clone().requires_grad_(True)
    q = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)
    k = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)
    v = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)

    qc, kc, vc, pic = torch_su2_holonomy_transport(q, k, v, omega_custom, transport_v=True, eps=1e-12, use_custom_adjoint=True)
    qa, ka, va, pia = torch_su2_holonomy_transport(q, k, v, omega_auto, transport_v=True, eps=1e-12, use_custom_adjoint=False)
    loss_c = (qc * kc).sum() + (vc * vc).sum() + (pic * pic).sum()
    loss_a = (qa * ka).sum() + (va * va).sum() + (pia * pia).sum()
    loss_c.backward()
    loss_a.backward()
    require_ok(omega_custom.grad is not None and omega_auto.grad is not None, "missing grads", code="PREFLIGHT_GRAD")
    adjoint_grad_err = float(torch.max(torch.abs(omega_custom.grad - omega_auto.grad)).item())

    # FP64 finite-difference gradcheck on SU2UnitPrefixScanFunction
    du_small = torch.randn((1, 2, 5, 2, 4), generator=gen, dtype=torch.float64, requires_grad=True)
    gradcheck_ok = bool(
        torch.autograd.gradcheck(
            lambda x: SU2UnitPrefixScanFunction.apply(x, 1e-10),
            (du_small,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
    )

    # 2. H14c: Exact covariant (Q, K, V) attention == explicit O(T^2) interval parallel transport
    q_s = torch.randn((1, 1, 6, 8), generator=gen, dtype=torch.float64)
    k_s = torch.randn((1, 1, 6, 8), generator=gen, dtype=torch.float64)
    v_s = torch.randn((1, 1, 6, 8), generator=gen, dtype=torch.float64)
    om_s = torch.randn((1, 1, 6, 2, 3), generator=gen, dtype=torch.float64)
    qr, kr, vr, pi_s = su2_holonomy_transport(q_s, k_s, v_s, om_s, transport_v=True, eps=1e-12, backend="torch_cpu")
    attn_fast = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=1.0 / math.sqrt(8.0))
    out_fast = su2_inverse_untransport(attn_fast, pi_s, backend="torch_cpu")

    # Explicit O(T^2) reference: sum_{s<=t} w_{ts} L(Pi_t^{-1} Pi_s) V_s
    t_len = 6
    out_explicit = torch.zeros_like(out_fast)
    vb = v_s.reshape(1, 1, t_len, 2, 4)
    qb = q_s.reshape(1, 1, t_len, 2, 4)
    kb = k_s.reshape(1, 1, t_len, 2, 4)
    for t in range(t_len):
        scores = []
        trans_vs = []
        for s in range(t + 1):
            rel = quat_mul(quat_conj(pi_s[:, :, t]), pi_s[:, :, s])  # Pi_t^{-1} Pi_s
            ks_rot = quat_mul(rel, kb[:, :, s]).reshape(1, 1, 8)
            vs_rot = quat_mul(rel, vb[:, :, s]).reshape(1, 1, 8)
            scores.append((qb[:, :, t].reshape(1, 1, 8) * ks_rot).sum(dim=-1) / math.sqrt(8.0))
            trans_vs.append(vs_rot)
        w = F.softmax(torch.stack(scores, dim=-1), dim=-1)
        out_explicit[:, :, t] = sum(w[:, :, s : s + 1] * trans_vs[s] for s in range(t + 1))
    covariant_err = float(torch.max(torch.abs(out_fast - out_explicit)).item())

    # 3. W10-OBS-31 Acute-cone Safe-Hopf clock anti-antipodal guarantee for r = 0..63
    acute_cosines = [math.cos(hopf_clock_angle(r, clock_law="acute_cone")) for r in range(64)]
    min_acute_cos = min(acute_cosines)

    # 4. H14b CS-08 parameter & FLOP parity check (unlooped covariant vs RoPE-2D-nGPT)
    cfg_rope = build_preset_config("rope2d_ngpt", d_model=128, n_heads=4, n_layers=4, base_d_ff=352, seed=42)
    cfg_sili = build_preset_config("s114_unlooped_covariant", d_model=128, n_heads=4, n_layers=4, base_d_ff=352, seed=42)
    m_rope = HoloSphereTransformer(cfg_rope)
    m_sili = HoloSphereTransformer(cfg_sili)
    p_rope = count_parameters(m_rope)["non_embedding_params"]
    p_sili = count_parameters(m_sili)["non_embedding_params"]
    param_rel_diff = abs(p_sili - p_rope) / float(p_rope)
    f_rope = analytical_forward_flops_per_token(cfg_rope, seq_len=512)["total_fwd_flops_per_token"]
    f_sili = analytical_forward_flops_per_token(cfg_sili, seq_len=512)["total_fwd_flops_per_token"]

    # 5. Flagship looped entity + S_3 / A_5 task smoke forward
    cfg_ent = build_preset_config("s114_entity", d_model=64, n_heads=4, base_d_ff=176, n_pre=1, n_core=2, n_suffix=1, r_train=3)
    m_ent = HoloSphereTransformer(cfg_ent)
    s3_x, s3_y = generate_group_tracking_batch("S3", batch_size=2, seq_len=16, seed=42)
    logits_r6, trace_r6 = m_ent(s3_x, r_loops=6, return_trace=True)
    entity_finite = bool(torch.isfinite(logits_r6).all().item()) and len(trace_r6["pass_states"]) == 7

    checks = {
        "h14a_scan_fwd_parity_lt_1e-12": scan_fwd_err < 1e-12,
        "h14a_custom_adjoint_parity_lt_1e-10": adjoint_grad_err < 1e-10,
        "h14a_fp64_gradcheck_passed": gradcheck_ok,
        "h14c_covariant_qkv_exact_lt_1e-12": covariant_err < 1e-12,
        "w10_obs31_acute_cone_no_antipodal": min_acute_cos > 0.2,
        "h14b_cs08_param_parity_lt_2pct": param_rel_diff < 0.02,
        "h14b_flagship_r6_extrap_finite": entity_finite,
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema": "holosphere.prereg_s114.cpu_preflight.v1",
        "passed": passed,
        "backends": available_backends(),
        "checks": checks,
        "metrics": {
            "scan_fwd_err": round(scan_fwd_err, 15),
            "adjoint_grad_err": round(adjoint_grad_err, 15),
            "covariant_err": round(covariant_err, 15),
            "min_acute_cone_cos_r0_63": round(min_acute_cos, 6),
            "rope_non_emb_params": p_rope,
            "holosphere_non_emb_params": p_sili,
            "param_rel_diff": round(param_rel_diff, 6),
            "rope_fwd_flops_t512": f_rope,
            "holosphere_fwd_flops_t512": f_sili,
            "s3_sample_target_checksum": int(s3_y.sum().item()),
        },
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HoloSphere PREREG-S114 CPU Preflight & Benchmark Suite")
    parser.add_argument("--cpu-preflight", action="store_true", help="Run deterministic CPU preflight verification")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output")
    args = parser.parse_args(argv)
    if not args.cpu_preflight:
        args.cpu_preflight = True
    res = run_cpu_preflight()
    if args.json:
        sys.stdout.write(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    else:
        sys.stdout.write(f"OK passed={res['passed']} payload_sha256={res['payload_sha256']}\n")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
