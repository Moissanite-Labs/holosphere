#!/usr/bin/env python3
"""Tests for SiliSense telemetry/tuning contract, Cayley2x2 closed-form adjoint, & var-depth presets (zero bare asserts)."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from holosphere.errors import HoloSphereError
from holosphere.geometry.beltrami import (
    Cayley2x2AutogradFn,
    CayleyBlock2x2SO,
    build_paired_beltrami_operator,
    cayley_2x2_rotate,
    run_beltrami_helicity_diagnostic,
)
from holosphere.geometry.hopf import cayley_decay_cos_sin, hopf_clock_angle
from holosphere.geometry.su2 import cayley_su2, safe_quat_unit_norm, safe_slerp_step, safe_unit_norm, su2_exp, su2_parallel_prefix_scan
from holosphere.kernels.dispatch import (
    cayley_2x2_step,
    fused_spherical_step,
    hopf_s3_cayley_2x2_step,
    safe_hopf_clock,
    su2_holonomy_transport,
)
from holosphere.kernels.torch_ops import SU2UnitPrefixScanFunction
from holosphere.models import (
    CovariantSU2Attention,
    HoloSphereTransformer,
    analytical_forward_flops_per_token,
    build_preset_config,
)
from holosphere.telemetry import (
    HoloSphereStepTelemetry,
    HoloSphereTuningProfile,
    build_step_telemetry,
)


def _naive_cayley_2x2_ref(x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    xf = x.reshape(*x.shape[:-1], -1, 2)
    w2 = omega * omega
    inv_d = 1.0 / (1.0 + w2)
    c = (1.0 - w2) * inv_d
    s = (2.0 * omega) * inv_d
    x0, x1 = xf.unbind(dim=-1)
    y0 = c * x0 + s * x1
    y1 = -s * x0 + c * x1
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


class TestTelemetryCayleyAndVarDepth(unittest.TestCase):
    def test_01_cayley2x2_closed_form_adjoint_gradcheck_and_beltrami_fp64(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(401)

        # 1. Exact forward & backward parity vs PyTorch autograd on BOTH per-token & broadcast omega
        for omega_shape in ((2, 5, 6), (6,)):
            x0 = torch.randn((2, 5, 12), generator=gen, dtype=torch.float64)
            w0 = torch.randn(omega_shape, generator=gen, dtype=torch.float64) * 0.6

            xc = x0.clone().requires_grad_(True)
            wc = w0.clone().requires_grad_(True)
            xr = x0.clone().requires_grad_(True)
            wr = w0.clone().requires_grad_(True)

            yc = cayley_2x2_rotate(xc, wc)
            yr = _naive_cayley_2x2_ref(xr, wr)
            self.assertLess(float(torch.max(torch.abs(yc - yr)).item()), 1e-14)

            # Exact norm preservation ||y||_2 == ||x||_2
            nx = torch.linalg.vector_norm(xc, dim=-1)
            ny = torch.linalg.vector_norm(yc, dim=-1)
            self.assertLess(float(torch.max(torch.abs(ny - nx)).item()), 1e-13)

            weights = torch.randn(x0.shape, generator=gen, dtype=x0.dtype)
            (yc * weights).sum().backward()
            (yr * weights).sum().backward()
            self.assertIsNotNone(xc.grad)
            self.assertIsNotNone(wc.grad)
            self.assertIsNotNone(xr.grad)
            self.assertIsNotNone(wr.grad)
            self.assertLess(float(torch.max(torch.abs(xc.grad - xr.grad)).item()), 1e-13)
            self.assertLess(float(torch.max(torch.abs(wc.grad - wr.grad)).item()), 1e-13)

        # 2. Strict FP64 finite-difference torch.autograd.gradcheck (< 1e-10)
        x_gc = torch.randn((2, 3, 8), generator=gen, dtype=torch.float64, requires_grad=True)
        w_gc = (torch.randn((2, 3, 4), generator=gen, dtype=torch.float64) * 0.5).detach().requires_grad_(True)
        ok_full = torch.autograd.gradcheck(
            lambda x, w: Cayley2x2AutogradFn.apply(x, w),
            (x_gc, w_gc),
            eps=1e-6,
            atol=1e-10,
            rtol=1e-8,
        )
        self.assertTrue(ok_full)

        w_bcast = (torch.randn((4,), generator=gen, dtype=torch.float64) * 0.5).detach().requires_grad_(True)
        ok_bcast = torch.autograd.gradcheck(
            lambda x, w: Cayley2x2AutogradFn.apply(x, w),
            (x_gc, w_bcast),
            eps=1e-6,
            atol=1e-10,
            rtol=1e-8,
        )
        self.assertTrue(ok_bcast)

        # 3. Exact FP64 Beltrami [C, K] = 0, Q^T Q = I, and m0..m4 conservation (< 1e-12)
        cayley = CayleyBlock2x2SO(d_model=16, bottleneck_rank=4, seed=42)
        diag = run_beltrami_helicity_diagnostic(cayley, n_steps=10, seed=42, dtype=torch.float64)
        self.assertLess(diag["commuting_commutator_frob"], 1e-12)
        self.assertLess(diag["commuting_max_moment_drift_m0_m4"], 1e-12)
        self.assertTrue(diag["o35_symmetric_strictly_monotone_collapse"])

        q_mat = cayley.explicit_matrix(dtype=torch.float64)
        k_mat = cayley.explicit_skew_matrix(dtype=torch.float64)
        c_mat = build_paired_beltrami_operator(16, dtype=torch.float64)
        eye = torch.eye(16, dtype=torch.float64)
        self.assertLess(float(torch.linalg.matrix_norm(q_mat.T @ q_mat - eye, ord="fro").item()), 1e-12)
        self.assertLess(float(torch.linalg.matrix_norm(c_mat @ k_mat - k_mat @ c_mat, ord="fro").item()), 1e-12)

    def test_02_silisense_telemetry_boundary_pure_stdlib_and_fail_closed(self) -> None:
        # AST audit: holosphere/telemetry.py must stay pure stdlib with zero sensor/hardware/silisense imports
        tel_path = REPO_ROOT / "holosphere" / "telemetry.py"
        src = tel_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(tel_path))
        forbidden_mods = {"silisense", "torch", "ctypes", "subprocess", "fcntl", "pynvml"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], forbidden_mods)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                self.assertNotIn(node.module.split(".")[0], forbidden_mods)
        for token in ("/dev/video", "/dev/snd", "intel-rapl", "nvidia-smi"):
            self.assertNotIn(token, src.lower())

        # Pure-stdlib import check with torch explicitly blocked (mirrors stock Python / silisense)
        probe = (
            "import sys; sys.modules['torch'] = None; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from holosphere import HoloSphereStepTelemetry, HoloSphereTuningProfile; "
            "p = HoloSphereTuningProfile(backend='torch_cpu', block_t=64); "
            "print(p.profile_sha256())"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(proc.stdout.strip()), 64)

        # Valid roundtrip & deterministic SHA-256
        prof = HoloSphereTuningProfile(
            backend="torch_cpu",
            block_t=64,
            microbatch_size=4,
            inter_step_cooldown_ms=2.5,
            r_loops_override=2,
        )
        prof2 = HoloSphereTuningProfile.from_dict(prof.to_dict())
        self.assertEqual(prof, prof2)
        self.assertEqual(prof.profile_sha256(), prof2.profile_sha256())
        self.assertEqual(len(prof.profile_sha256()), 64)

        # D2 regression: int 0 vs float 0.0 vs -0.0 MUST produce bit-identical canonical profile_sha256()
        p_int = HoloSphereTuningProfile(backend="torch_cpu", block_t=64, inter_step_cooldown_ms=0)
        p_flt = HoloSphereTuningProfile(backend="torch_cpu", block_t=64, inter_step_cooldown_ms=0.0)
        p_neg = HoloSphereTuningProfile(backend="torch_cpu", block_t=64, inter_step_cooldown_ms=-0.0)
        p_dct = HoloSphereTuningProfile.from_dict({"backend": "torch_cpu", "block_t": 64, "inter_step_cooldown_ms": 0})
        self.assertEqual(p_int.profile_sha256(), p_flt.profile_sha256())
        self.assertEqual(p_int.profile_sha256(), p_neg.profile_sha256())
        self.assertEqual(p_int.profile_sha256(), p_dct.profile_sha256())
        self.assertIsInstance(p_int.inter_step_cooldown_ms, float)

        # N12 regression: unhashable backend must raise typed HoloSphereError, never TypeError
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile(backend=[])  # type: ignore[arg-type]

        tel = build_step_telemetry(
            step_idx=3,
            seq_len=128,
            batch_size=2,
            r_loops=3,
            backend="torch_cpu",
            block_t=64,
            step_ms=10.0,
            norm_drift_max=1e-6,
        )
        self.assertEqual(HoloSphereStepTelemetry.from_dict(tel.to_dict()), tel)
        self.assertAlmostEqual(tel.tok_per_sec, 25600.0, places=3)

        # Fail-closed rejection on bad schemas, booleans, NaN, bad block_t, extra keys
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile(backend="unknown").validate()
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile(block_t=96).validate()
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile(inter_step_cooldown_ms=float("nan")).validate()
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile(microbatch_size=True).validate()  # type: ignore[arg-type]
        with self.assertRaises(HoloSphereError):
            HoloSphereTuningProfile.from_dict({"backend": "auto", "mutate_weights": True})
        with self.assertRaises(HoloSphereError):
            HoloSphereStepTelemetry.from_dict({"step_idx": 0})
        with self.assertRaises(HoloSphereError):
            build_step_telemetry(
                step_idx=0,
                seq_len=16,
                batch_size=2,
                r_loops=1,
                backend="auto",
                block_t=128,
                step_ms=1.0,
            )

    def test_03_stage1a_stage1b_presets_vardepth_and_model_telemetry_hook(self) -> None:
        small_kw = dict(d_model=64, n_heads=4, n_layers=4, base_d_ff=176, n_pre=1, n_core=2, n_suffix=1, r_train=3)
        cfg_1a = build_preset_config("s114_stage1a_cayley", **small_kw)
        cfg_1b = build_preset_config("s114_stage1b_cayley_vardepth", **small_kw)

        # Law-5 verification: surgical Cayley2x2 enabled, dissipative DarcyEMA & BinghamProx strictly excluded!
        for cfg in (cfg_1a, cfg_1b):
            self.assertTrue(cfg.use_cayley)
            self.assertFalse(cfg.use_darcy)
            self.assertFalse(cfg.use_bingham)
            self.assertEqual(cfg.pos_mode, "su2_holonomy")
            self.assertEqual(cfg.residual_mode, "hopf_s3")
        self.assertEqual(cfg_1b.loop_mode, "safe_hopf_cayley_acute")
        self.assertTrue(cfg_1b.var_depth_train)
        self.assertEqual((cfg_1b.r_train_min, cfg_1b.r_train_max), (1, 4))

        # Analytical FLOPs reflect Cayley2x2 cleanly
        cfg_base = build_preset_config("s114_unlooped_covariant", **small_kw)
        fl_base = analytical_forward_flops_per_token(cfg_base, seq_len=128)["core_fwd_flops_per_token"]
        fl_1a = analytical_forward_flops_per_token(cfg_1a, seq_len=128)["core_fwd_flops_per_token"]
        self.assertGreater(fl_1a, fl_base)

        gen = torch.Generator(device="cpu").manual_seed(403)
        x = torch.randint(0, 64, (2, 12), generator=gen)

        # Stage +1a forward + backward + zero-weight-mutation tuning profile + opt-in telemetry
        m1a = HoloSphereTransformer(cfg_1a)
        sha_before = m1a.core_weights_sha256()
        prof = HoloSphereTuningProfile(backend="torch_cpu", block_t=64, microbatch_size=2, inter_step_cooldown_ms=1.0)
        m1a.apply_tuning_profile(prof)
        self.assertEqual(m1a.tuning_profile, prof)
        self.assertEqual(m1a.core_weights_sha256(), sha_before)
        self.assertIsNotNone(m1a.blocks)
        for blk in m1a.blocks or []:
            self.assertEqual(blk.backend, "torch_cpu")
            self.assertEqual(blk.attn.backend, "torch_cpu")

        # Default forward emits ZERO telemetry
        logits_no_tel, trace_no_tel = m1a(x)
        self.assertNotIn("step_telemetry", trace_no_tel)
        self.assertIsNone(m1a.last_step_telemetry)
        logits_no_tel.float().mean().backward()

        # Opt-in emit_telemetry=True produces validated telemetry stamped with profile SHA-256
        logits_tel, trace_tel = m1a(x, emit_telemetry=True, step_ms=5.0)
        self.assertTrue(bool(torch.isfinite(logits_tel).all().item()))
        self.assertIn("step_telemetry", trace_tel)
        self.assertEqual(trace_tel["tuning_profile_sha256"], prof.profile_sha256())
        self.assertIsNotNone(m1a.last_step_telemetry)
        self.assertEqual(m1a.last_step_telemetry.backend, "torch_cpu")
        self.assertEqual(m1a.last_step_telemetry.block_t, 64)
        self.assertEqual(m1a.last_step_telemetry.step_idx, 0)

        # Stage +1b batch-uniform variable-depth training r in {1..4}, deterministic eval=r_train, & R_test=6/12
        m1b = HoloSphereTransformer(cfg_1b)
        m1b.train()
        seen_r: set[int] = set()
        for _ in range(32):
            logits_b, tr_b = m1b(x, return_trace=True)
            r_used = int(tr_b["r_loops_used"])
            self.assertGreaterEqual(r_used, 1)
            self.assertLessEqual(r_used, 4)
            self.assertEqual(len(tr_b["pass_states"]), r_used + 1)
            seen_r.add(r_used)
        self.assertEqual(seen_r, {1, 2, 3, 4})
        logits_b.float().mean().backward()

        # In eval(), defaults cleanly to cfg.r_train (3) unless overridden by tuning_profile or explicit r_loops
        m1b.eval()
        with torch.no_grad():
            _, tr_eval = m1b(x, return_trace=True)
            self.assertEqual(tr_eval["r_loops_used"], 3)
            m1b.apply_tuning_profile(HoloSphereTuningProfile(backend="torch_cpu", block_t=128, r_loops_override=2))
            _, tr_ovr = m1b(x, return_trace=True, emit_telemetry=True, step_ms=4.0)
            self.assertEqual(tr_ovr["r_loops_used"], 2)
            self.assertEqual(tr_ovr["step_telemetry"]["r_loops"], 2)
            # Explicit r_loops=6 & 12 extrapolation (non-antipodal cayley_acute)
            for r_ext in (6, 12):
                logits_ext, tr_ext = m1b(x, r_loops=r_ext, return_trace=True)
                self.assertTrue(bool(torch.isfinite(logits_ext).all().item()))
                self.assertEqual(tr_ext["r_loops_used"], r_ext)
                self.assertEqual(len(tr_ext["pass_states"]), r_ext + 1)

    def test_04_cayley_su2_sfu_free_unit_norm_antipodal_bound_and_gradcheck(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(404)
        # 1. Exact unit-sphere identity ||cayley_su2(omega)|| == 1 for arbitrary omega in R^3
        w = torch.randn((3, 5, 3), generator=gen, dtype=torch.float64) * 1.5
        q = cayley_su2(w)
        self.assertEqual(q.shape, (3, 5, 4))
        norms = torch.linalg.vector_norm(q, dim=-1)
        self.assertLess(float(torch.max(torch.abs(norms - 1.0)).item()), 1e-14)

        # 2. Exact identity cayley_su2(w) == su2_exp(2 * atan(||w||) * w / ||w||)
        r = torch.linalg.vector_norm(w, dim=-1, keepdim=True)
        w_equiv = (2.0 * torch.atan(r) / r) * w
        q_exp = su2_exp(w_equiv, eps=1e-12)
        self.assertLess(float(torch.max(torch.abs(q - q_exp)).item()), 1e-13)

        # 3. Antipodal avoidance: ||w|| <= t_max = 0.8 => w_0 >= (1 - 0.64)/(1 + 0.64) = 9/41 > 0
        t_max = 0.8
        w_bounded = w / torch.clamp(r / t_max, min=1.0)
        q_b = cayley_su2(w_bounded)
        floor = (1.0 - t_max * t_max) / (1.0 + t_max * t_max)
        self.assertGreaterEqual(float(q_b[..., 0].min().item()), floor - 1e-14)

        # 4. Strict FP64 finite-difference gradcheck (< 1e-10)
        w_gc = (torch.randn((2, 4, 3), generator=gen, dtype=torch.float64) * 0.5).detach().requires_grad_(True)
        ok = torch.autograd.gradcheck(cayley_su2, (w_gc,), eps=1e-6, atol=1e-10, rtol=1e-8)
        self.assertTrue(ok)

    def test_05_v2_council_p0_p1_fixes_and_cayley_decay(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(405)

        # 1. Grok-4.7 P0 SLERP near-antipodal O(eps^-3) guard (m2 = 2 * eps^2 must NOT hit ~1e12!)
        eps = 1e-4
        h_near = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        # Choose delta almost antipodal to h so ||mixed||^2 ~= 2 * eps^2 (just above eps^2!)
        d_near = torch.tensor([[-1.0, 2.0 * eps, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        y_near = safe_slerp_step(h_near, d_near, 0.25 * 3.141592653589793, eps=eps)
        (y_near * torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)).sum().backward()
        self.assertIsNotNone(h_near.grad)
        self.assertTrue(bool(torch.isfinite(h_near.grad).all().item()))
        self.assertLess(float(torch.max(torch.abs(h_near.grad)).item()), 2.0 / eps)

        # Exact FP64 gradcheck away from antipode (||mixed||^2 >= 1e-3)
        h_ok = torch.tensor([[1.0, 0.2, -0.3, 0.4]], dtype=torch.float64, requires_grad=True)
        d_ok = torch.tensor([[0.5, 0.6, 0.1, -0.2]], dtype=torch.float64, requires_grad=True)
        al_ok = torch.tensor([0.35], dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b, c: safe_slerp_step(a, b, c, eps=1e-4),
                (h_ok, d_ok, al_ok),
                eps=1e-6,
                atol=1e-10,
                rtol=1e-8,
            )
        )

        # 2. Grok-4.7 P1 non-sticky q_norm(0) -> (1, 0, 0, 0) prevents zero-fiber collapse in scan
        zero_q = torch.zeros((1, 4), dtype=torch.float64)
        id_q = safe_quat_unit_norm(zero_q, eps=1e-6)
        self.assertLess(float(torch.max(torch.abs(id_q - torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))).item()), 1e-14)
        seq = torch.randn((1, 4, 2, 4), generator=gen, dtype=torch.float64)
        seq[:, 1] = 0.0
        scanned = su2_parallel_prefix_scan(seq, seq_dim=1, eps=1e-6)
        self.assertLess(float(torch.max(torch.abs(torch.linalg.vector_norm(scanned, dim=-1) - 1.0)).item()), 1e-12)

        # N8 regression: zero-fallback lane in SU2UnitPrefixScanFunction.backward must yield exact 0.0 gradient
        du_zero = torch.randn((1, 4, 2, 4), generator=gen, dtype=torch.float64)
        du_zero[:, 1] = 0.0
        du_zero.requires_grad_(True)
        pi_zero = SU2UnitPrefixScanFunction.apply(du_zero, 1e-6)
        w_out = torch.randn(pi_zero.shape, generator=gen, dtype=torch.float64)
        (pi_zero * w_out).sum().backward()
        self.assertIsNotNone(du_zero.grad)
        self.assertTrue(bool(torch.isfinite(du_zero.grad).all().item()))
        self.assertEqual(float(torch.max(torch.abs(du_zero.grad[:, 1])).item()), 0.0)

        # 3. ML4 P0 block-4 tied sqk preserves exact SU(2) 4-block equivariance
        attn = CovariantSU2Attention(d_model=32, n_heads=2, pos_mode="su2_holonomy", ngpt_qk_norm=True)
        self.assertIsNotNone(attn.sqk)
        self.assertEqual(tuple(attn.sqk.shape), (1, 2, 1, 4, 1))

        # 4. GPT-6 P0 fail-closed eps > 0 + FP32 pi_path invariant under BF16
        q_bf = torch.randn((1, 2, 8, 16), generator=gen, dtype=torch.bfloat16)
        k_bf = torch.randn((1, 2, 8, 16), generator=gen, dtype=torch.bfloat16)
        v_bf = torch.randn((1, 2, 8, 16), generator=gen, dtype=torch.bfloat16)
        om_bf = torch.randn((1, 2, 8, 4, 3), generator=gen, dtype=torch.bfloat16)
        for map_mode in ("exp", "cayley"):
            qr, kr, vr, pi_p = su2_holonomy_transport(q_bf, k_bf, v_bf, om_bf, transport_v=True, map_mode=map_mode)
            self.assertEqual(qr.dtype, torch.bfloat16)
            self.assertEqual(pi_p.dtype, torch.float32)
            self.assertLess(float(torch.max(torch.abs(torch.linalg.vector_norm(pi_p, dim=-1) - 1.0)).item()), 1e-5)

        for bad_eps in (0.0, -1e-4, float("nan"), float("inf")):
            with self.assertRaises(HoloSphereError):
                safe_unit_norm(h_ok, eps=bad_eps)
            with self.assertRaises(HoloSphereError):
                su2_holonomy_transport(q_bf, k_bf, v_bf, om_bf, eps=bad_eps)
            with self.assertRaises(HoloSphereError):
                fused_spherical_step(h_ok, d_ok, mode="hopf_s3", eps=bad_eps)

        # 5. cayley_decay (phi_r -> 0) + safe_hopf_cayley_decay + s114_cayley_su2_unlooped + hopf_s3_cayley_2x2_step
        prev_phi = 10.0
        for r in range(32):
            c, s = cayley_decay_cos_sin(r)
            self.assertGreaterEqual(c, (1.0 - 0.64) / (1.0 + 0.64) - 1e-12)
            self.assertAlmostEqual(c * c + s * s, 1.0, places=12)
            phi = hopf_clock_angle(r, clock_law="cayley_decay")
            self.assertLess(phi, prev_phi)
            prev_phi = phi
        self.assertLess(hopf_clock_angle(100, clock_law="cayley_decay"), 0.05)
        rot_dec = safe_hopf_clock(torch.randn((2, 16), generator=gen, dtype=torch.float64), pass_idx=5, law="cayley_decay")
        self.assertEqual(rot_dec.shape, (2, 16))

        x_tok = torch.randint(0, 64, (2, 10), generator=gen)
        cfg_csu2 = build_preset_config("s114_cayley_su2_unlooped", d_model=64, n_heads=4, n_layers=2, base_d_ff=176)
        m_csu2 = HoloSphereTransformer(cfg_csu2)
        logits_c, _ = m_csu2(x_tok)
        self.assertTrue(bool(torch.isfinite(logits_c).all().item()))
        logits_c.float().mean().backward()

        cfg_dec = build_preset_config(
            "s114_stage1b_cayley_vardepth",
            loop_mode="safe_hopf_cayley_decay",
            d_model=64,
            n_heads=4,
            n_layers=4,
            base_d_ff=176,
            n_pre=1,
            n_core=2,
            n_suffix=1,
        )
        m_dec = HoloSphereTransformer(cfg_dec)
        logits_d, _ = m_dec(x_tok, r_loops=8)
        self.assertTrue(bool(torch.isfinite(logits_d).all().item()))

        # V2 hopf_s3_cayley_2x2_step & cayley_2x2_step FP64 gradcheck
        h_f = torch.randn((2, 3, 8), generator=gen, dtype=torch.float64, requires_grad=True)
        d_f = torch.randn((2, 3, 8), generator=gen, dtype=torch.float64, requires_grad=True)
        w_f = (torch.randn((2, 3, 4), generator=gen, dtype=torch.float64) * 0.4).detach().requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b, c: hopf_s3_cayley_2x2_step(a, b, c),
                (h_f, d_f, w_f),
                eps=1e-6,
                atol=1e-10,
                rtol=1e-8,
            )
        )
        self.assertEqual(cayley_2x2_step(h_f, w_f).shape, h_f.shape)


if __name__ == "__main__":
    unittest.main()
