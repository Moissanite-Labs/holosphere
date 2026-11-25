#!/usr/bin/env python3
"""Exact algebraic & geometric invariant tests for holosphere.geometry (zero bare asserts)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from holosphere.errors import HoloSphereError
from holosphere.geometry import (
    CayleyBlock2x2SO,
    VectorProximalPoloidalBingham,
    build_s3_width_schedule,
    cayley_acute_cos_sin,
    darboux_intertwining_diagnostic,
    hopf_clock_angle,
    hopf_fiber_rotate,
    hopf_fiber_rotate_cs,
    hopf_s3_residual_step,
    hopf_s3_to_s2,
    measure_dark_mode_energy_fraction,
    quat_conj,
    quat_inverse_untransport,
    quat_left_transport,
    quat_mul,
    run_beltrami_helicity_diagnostic,
    safe_slerp_step,
    safe_unit_norm,
    spearman_rank_corr,
    su2_exp,
    su2_parallel_prefix_scan,
    su2_sequential_prefix_scan,
    verify_su2_interval_holonomy,
)


class TestGeometryExact(unittest.TestCase):
    def test_01_su2_associativity_norm_and_interval_holonomy(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(101)
        p = safe_unit_norm(torch.randn((8, 4), generator=gen, dtype=torch.float64), eps=1e-12)
        q = safe_unit_norm(torch.randn((8, 4), generator=gen, dtype=torch.float64), eps=1e-12)
        r = safe_unit_norm(torch.randn((8, 4), generator=gen, dtype=torch.float64), eps=1e-12)

        # Associativity: (p * q) * r == p * (q * r)
        lhs = quat_mul(quat_mul(p, q), r)
        rhs = quat_mul(p, quat_mul(q, r))
        self.assertLess(float(torch.max(torch.abs(lhs - rhs)).item()), 1e-14)

        # Euler 4-square norm multiplicativity + exact inverse conj(q) * q == (1, 0, 0, 0)
        p_raw = torch.randn((8, 4), generator=gen, dtype=torch.float64)
        q_raw = torch.randn((8, 4), generator=gen, dtype=torch.float64)
        n_prod = (quat_mul(p_raw, q_raw) ** 2).sum(dim=-1)
        n_mul = (p_raw ** 2).sum(dim=-1) * (q_raw ** 2).sum(dim=-1)
        self.assertLess(float(torch.max(torch.abs(n_prod - n_mul)).item()), 1e-12)

        inv_prod = quat_mul(quat_conj(q), q)
        expected_id = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand_as(inv_prod)
        self.assertLess(float(torch.max(torch.abs(inv_prod - expected_id)).item()), 1e-12)

        # Parallel Hillis-Steele scan == sequential scan & exact interval holonomy Pi(s)^{-1} Pi(t)
        omega = torch.randn((2, 19, 3, 3), generator=gen, dtype=torch.float64)
        du = su2_exp(0.75 * torch.tanh(omega), eps=1e-12)
        pi_par = su2_parallel_prefix_scan(du, seq_dim=1, eps=1e-12)
        pi_seq = su2_sequential_prefix_scan(du, seq_dim=1, eps=1e-12)
        self.assertLess(float(torch.max(torch.abs(pi_par - pi_seq)).item()), 1e-12)

        hol = verify_su2_interval_holonomy(du, s=3, t=14, eps=1e-12)
        self.assertLess(hol["max_interval_error"], 1e-12)
        self.assertLess(hol["max_unit_norm_error"], 1e-12)

        # Exact left-transport isometry & inverse untransport recovery
        vec = torch.randn((2, 19, 12), generator=gen, dtype=torch.float64)
        v_rot = quat_left_transport(vec, pi_par)
        v_rec = quat_inverse_untransport(v_rot, pi_par)
        self.assertLess(float(torch.max(torch.abs(v_rec - vec)).item()), 1e-12)
        n_before = torch.linalg.vector_norm(vec, dim=-1)
        n_after = torch.linalg.vector_norm(v_rot, dim=-1)
        self.assertLess(float(torch.max(torch.abs(n_before - n_after)).item()), 1e-12)

    def test_02_antipodal_safe_slerp_and_hopf_dark_fiber_invariance(self) -> None:
        # Exact antipodal cancellation delta = -h at alpha = pi/4 must remain finite (zero NaN)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        out = safe_slerp_step(h, -h, 0.25 * math.pi, eps=1e-4)
        self.assertTrue(bool(torch.isfinite(out).all().item()))
        out.sum().backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(bool(torch.isfinite(h.grad).all().item()))

        # S112 Thm 34: Hopf map S^3 -> S^2 is invariant under right S^1 fiber rotation
        gen = torch.Generator(device="cpu").manual_seed(102)
        q = safe_unit_norm(torch.randn((16, 4), generator=gen, dtype=torch.float64), eps=1e-12)
        q_rot = hopf_fiber_rotate(q, 1.2345)
        self.assertLess(float(torch.max(torch.abs(hopf_s3_to_s2(q) - hopf_s3_to_s2(q_rot))).item()), 1e-12)

        # H6c Native Hopf-S^3 residual preserves zero-delta identity
        z = torch.zeros_like(q)
        self.assertLess(float(torch.max(torch.abs(hopf_s3_residual_step(q, z, eps=1e-12) - q)).item()), 1e-11)

    def test_03_w10_obs31_acute_cone_and_cayley_acute_anti_antipodal(self) -> None:
        # W10-OBS-31 reproduction: unaliased_coprime hits cos(phi_5) < -0.95 (antipodal flip!)
        self.assertLess(math.cos(hopf_clock_angle(5, clock_law="unaliased_coprime")), -0.9)

        # Both acute_cone and SFU-free cayley_acute stay strictly positive cos > 0 for r = 0..64
        prev_ac = -1.0
        prev_ca = -1.0
        for r in range(65):
            phi_ac = hopf_clock_angle(r, clock_law="acute_cone")
            self.assertGreater(math.cos(phi_ac), 0.2)
            self.assertGreater(phi_ac, prev_ac)
            prev_ac = phi_ac

            c, s = cayley_acute_cos_sin(r)
            self.assertGreater(c, 0.2)
            self.assertAlmostEqual(c * c + s * s, 1.0, places=12)
            phi_ca = hopf_clock_angle(r, clock_law="cayley_acute")
            self.assertGreater(phi_ca, prev_ca)
            prev_ca = phi_ca

        # cayley_acute rotation matches hopf_fiber_rotate(atan2(s, c)) identically
        gen = torch.Generator(device="cpu").manual_seed(103)
        x = torch.randn((4, 16), generator=gen, dtype=torch.float64)
        c, s = cayley_acute_cos_sin(6)
        r_cs = hopf_fiber_rotate_cs(x, c, s)
        r_ang = hopf_fiber_rotate(x, math.atan2(s, c))
        self.assertLess(float(torch.max(torch.abs(r_cs - r_ang)).item()), 1e-14)

    def test_04_s3_width_schedule_beltrami_bingham_and_darboux(self) -> None:
        # Exact H7a width sum conservation & low-Spearman permuted nulls
        for prof in ("cylinder", "polar", "equatorial", "permuted_pol", "permuted_eq"):
            w = build_s3_width_schedule(12, 864, prof)
            self.assertEqual(len(w), 12)
            self.assertEqual(sum(w), 12 * 864)
        w_pol = build_s3_width_schedule(12, 864, "polar")
        w_pp = build_s3_width_schedule(12, 864, "permuted_pol")
        self.assertLessEqual(abs(spearman_rank_corr(w_pol, w_pp)), 0.30)

        # Portal-3 Beltrami [C, K] = 0 helicity conservation & O3.5 monotone collapse
        cay = CayleyBlock2x2SO(d_model=16, bottleneck_rank=4, seed=42)
        diag = run_beltrami_helicity_diagnostic(cay, n_steps=8, seed=42)
        self.assertLess(diag["commuting_commutator_frob"], 1e-6)
        self.assertLess(diag["commuting_max_moment_drift_m0_m4"], 1e-5)
        self.assertTrue(diag["o35_symmetric_strictly_monotone_collapse"])

        # S109/S110 VectorProximalPoloidalBingham: forbids post_renorm, SO(k) equivariance, Stagnant-Lid
        with self.assertRaises(HoloSphereError):
            VectorProximalPoloidalBingham(d_model=16, post_renorm=True)
        bing = VectorProximalPoloidalBingham(d_model=16, k_poloidal=8, tau_y=0.5)
        gen = torch.Generator(device="cpu").manual_seed(104)
        u = torch.randn((4, 16), generator=gen, requires_grad=True)
        out, stats = bing(u)
        self.assertGreaterEqual(stats["plug_fraction"], 0.0)
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(u.grad).all().item()))
        # Zero input has exact finite 0 gradient (no 0/0 NaN)
        u0 = torch.zeros((2, 16), requires_grad=True)
        o0, _ = bing(u0)
        o0.sum().backward()
        self.assertTrue(bool(torch.isfinite(u0.grad).all().item()))
        # SO(k) equivariance on poloidal subspace
        q_rot, _ = torch.linalg.qr(torch.randn((8, 8), generator=gen))
        u_test = torch.randn((4, 16), generator=gen)
        u_rot = torch.cat((u_test[:, :8] @ q_rot.T, u_test[:, 8:]), dim=-1)
        y1, _ = bing(u_rot)
        y2_base, _ = bing(u_test)
        y2 = torch.cat((y2_base[:, :8] @ q_rot.T, y2_base[:, 8:]), dim=-1)
        self.assertLess(float(torch.max(torch.abs(y1 - y2)).item()), 1e-5)

        # Portal-5 Darboux intertwining & dark-mode kernel energy
        a_mat = torch.randn((4, 16), generator=gen, dtype=torch.float64)
        db = darboux_intertwining_diagnostic(a_mat, shift_c=0.5)
        self.assertEqual(db["rank"], 4)
        self.assertEqual(db["dark_dim"], 12)
        self.assertLess(db["intertwining_frob_err"], 1e-12)
        self.assertLess(db["dark_annihilation_err"], 1e-12)
        en = measure_dark_mode_energy_fraction(a_mat, torch.randn((8, 16), generator=gen, dtype=torch.float64))
        self.assertGreater(en["mean_dark_energy_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
