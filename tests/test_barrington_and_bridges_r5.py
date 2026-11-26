#!/usr/bin/env python3
"""Pure-stdlib R5 tests for Spinor-Barrington 2I DAG compiler & Cross-Spine Bridges (zero torch, 100% mutant kill)."""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holosphere import barrington_su2 as s119
from holosphere import bridges as s120


class TestBarringtonAndBridgesR5(unittest.TestCase):
    def test_01_barrington_2i_q5_primitives_dag_semantics_and_tamper_controls(self) -> None:
        # Exact Q(sqrt(5)) field axioms, sign edge cases, and N1 overflow/type guards
        phi = s119.Q5(Fraction(1, 2), Fraction(1, 2))
        inv_phi = s119.Q5(Fraction(-1, 2), Fraction(1, 2))
        self.assertEqual(phi * inv_phi, s119.Q5_ONE)
        self.assertEqual(phi.inv(), inv_phi)
        self.assertEqual(phi - inv_phi, s119.Q5_ONE)
        self.assertEqual(phi / phi, s119.Q5_ONE)
        self.assertEqual(phi.as_fractions(), (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(s119.Q5_ZERO.sign(), 0)
        self.assertEqual(s119.Q5_PHI_HALF.sign(), 1)
        self.assertEqual((-s119.Q5_PHI_HALF).sign(), -1)
        self.assertEqual(s119.Q5(3, -1).sign(), 1)
        self.assertEqual(s119.Q5(2, -1).sign(), -1)
        self.assertEqual(s119.Q5(-2, 1).sign(), 1)
        self.assertEqual(s119.Q5(-3, 1).sign(), -1)
        self.assertFalse(s119.Q5_ONE == 1)

        with self.assertRaises(ValueError):
            s119.Q5(float("inf"))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            s119.Q5(0, float("nan"))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            s119.Q5(True, 0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            s119.Q5(0, False)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            s119.Q5(10**400, 0).to_float()
        with self.assertRaises(ValueError):
            s119.Q5(0, 10**308).to_float()

        # Negative controls: Q5 division by zero, Cayley pole, non-unit order, infinite order
        with self.assertRaises(ValueError) as cm_div0:
            s119.Q5_ZERO.inv()
        self.assertIn("Q5 division by zero", str(cm_div0.exception))
        with self.assertRaises(ValueError) as cm_pole:
            s119.cayley_inverse_q5(s119.Q5_NEG_ID)
        self.assertIn("CayleySU2 antipodal pole", str(cm_pole.exception))
        with self.assertRaises(ValueError) as cm_unit:
            s119.q5_order((s119.Q5_TWO, s119.Q5_ZERO, s119.Q5_ZERO, s119.Q5_ZERO))
        self.assertIn("unit norm", str(cm_unit.exception))
        inf_ord_q = s119.cayley_forward_q5((s119.Q5_ONE, s119.Q5_TWO, s119.Q5_ZERO))
        self.assertEqual(s119.q5_norm2(inf_ord_q), s119.Q5_ONE)
        with self.assertRaises(ValueError) as cm_inf:
            s119.q5_order(inf_ord_q)
        self.assertIn("exceeded finite 2I bound", str(cm_inf.exception))

        # Spinor decoders (exact Q5 & FP32 boundary tol == 0.0)
        self.assertEqual(s119.decode_spinor_q5(s119.Q5_ID), 0)
        self.assertEqual(s119.decode_spinor_q5(s119.TAU0), 1)
        with self.assertRaises(ValueError):
            s119.decode_spinor_q5(s119.Q5_NEG_ID)
        with self.assertRaises(ValueError):
            s119.decode_spinor_q5(s119.G_BETA)
        self.assertEqual(s119.decode_spinor_f32(s119.ID_F32, tol=1e-5), 0)
        self.assertEqual(s119.decode_spinor_f32(s119.TAU0_F32, tol=1e-5), 1)
        with self.assertRaises(ValueError):
            s119.decode_spinor_f32(s119.ID_F64, tol=0.0)
        with self.assertRaises(ValueError):
            s119.decode_spinor_f32((0.9, 0.2, 0.2, 0.0), tol=0.05)

        # DAGBuilder wire, output, depth, & VM guards
        b = s119.DAGBuilder("guard_test")
        w0 = b.add_input()
        w1 = b.add_input()
        for bad_call in (
            lambda: b.add_not(-1),
            lambda: b.add_not(2),
            lambda: b.add_and(-1, w0),
            lambda: b.add_and(2, w0),
            lambda: b.add_and(w0, -1),
            lambda: b.add_and(w0, 2),
        ):
            with self.assertRaises(ValueError):
                bad_call()
        w2 = b.add_and(w0, w1)
        for bad_out in ((), (-1,), (3,)):
            with self.assertRaises(ValueError):
                b.build(bad_out)
        dag = b.build((w2,))

        self.assertEqual(s119.compile_dag_to_holosphere_vring(s119.build_shared_diamond_dag(1)).and_depth, 1)
        self.assertEqual(s119.compile_dag_to_holosphere_vring(s119.build_all_ones_lyapunov_chain(1)).and_depth, 1)
        with self.assertRaises(ValueError):
            s119.build_shared_diamond_dag(0)
        with self.assertRaises(ValueError):
            s119.build_all_ones_lyapunov_chain(0)

        for bad_bits in ((0,), (0, 1, 0), (0, 2), (-1, 1)):
            with self.assertRaises(ValueError):
                s119.eval_boolean(dag, bad_bits)
        prog = s119.compile_dag_to_holosphere_vring(dag)
        with self.assertRaises(ValueError):
            s119.run_q5_program(prog, (0,))
        with self.assertRaises(ValueError):
            s119.run_fp32_program(prog, (0, 1), mode="bogus")
        with self.assertRaises(ValueError):
            s119.run_fp32_program(prog, (0,), mode="cayley_reproj")

        # Semantic circuit truth tables (XOR4, 4-bit adder, MAJ5, 3-SAT)
        xor4 = s119.build_xor_chain_4()
        for m in range(16):
            bits = tuple((m >> i) & 1 for i in range(4))
            self.assertEqual(s119.eval_boolean(xor4, bits), (sum(bits) % 2,))
        adder = s119.build_ripple_carry_adder_4bit()
        for a_val in range(16):
            for b_val in range(16):
                bits = tuple((a_val >> i) & 1 for i in range(4)) + tuple((b_val >> i) & 1 for i in range(4))
                out = s119.eval_boolean(adder, bits)
                self.assertEqual(sum(out[i] << i for i in range(5)), a_val + b_val)
        maj5 = s119.build_majority_5()
        for m in range(32):
            bits = tuple((m >> i) & 1 for i in range(5))
            self.assertEqual(s119.eval_boolean(maj5, bits), (1 if sum(bits) >= 3 else 0,))
        sat4 = s119.build_three_sat_4var()
        sat_count = 0
        for m in range(16):
            x0, x1, x2, x3 = ((m >> i) & 1 for i in range(4))
            expected = int(
                (x0 or (not x1) or x2)
                and ((not x0) or x1 or x3)
                and (x1 or (not x2) or (not x3))
                and ((not x0) or (not x2) or x3)
                and (x0 or x2 or (not x3))
            )
            got_bit = s119.eval_boolean(sat4, (x0, x1, x2, x3))[0]
            self.assertEqual(got_bit, expected)
            sat_count += got_bit
        self.assertGreater(sat_count, 0)
        self.assertLess(sat_count, 16)

        # CLI --json, --check, summary, & exhaustive validate_certificate tamper matrix
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = s119.main(["--json"])
        self.assertEqual(rc, 0)
        cert = json.loads(buf.getvalue())
        s119.validate_certificate(cert)
        self.assertIn(cert["payload_sha256"], s119.format_summary(cert))
        buf_chk = io.StringIO()
        with redirect_stdout(buf_chk):
            rc_chk = s119.main(["--check"])
        self.assertEqual(rc_chk, 0)
        self.assertIn(cert["payload_sha256"], buf_chk.getvalue())

        def tamper(mutator: Callable[[dict[str, Any]], None]) -> None:
            bad = copy.deepcopy(cert)
            mutator(bad)
            with self.assertRaises(ValueError):
                s119.validate_certificate(bad)

        for key, val in (
            ("group_order", 119),
            ("unit_norm_all_ok", False),
            ("inverse_closed", False),
            ("closure_all_ok", False),
            ("closure_pairs_checked", 14399),
            ("a5_quotient_order", 59),
            ("cayley_q0_ge_0_count", 74),
            ("cayley_ge0_roundtrip_ok", 74),
            ("cayley_ge0_r2_le1_ok", 74),
            ("cayley_pole_count", 0),
        ):
            tamper(lambda d, k=key, v=val: d["group_2i"].__setitem__(k, v))

        tamper(lambda d: d["spinor_commutator_theorem"].__setitem__("trace_identity_pairs_checked", 100))
        tamper(lambda d: d["spinor_commutator_theorem"].__setitem__("trace_identity_mismatches", 1))
        tamper(lambda d: d["spinor_commutator_theorem"].__setitem__("num_conjugacy_classes", 8))
        tamper(lambda d: next(r for r in d["spinor_commutator_theorem"]["conjugacy_classes"] if r["order"] == 5).__setitem__("pos_hits", 1))
        tamper(lambda d: next(r for r in d["spinor_commutator_theorem"]["conjugacy_classes"] if r["order"] == 5).__setitem__("neg_hits", 9))
        tamper(lambda d: next(r for r in d["spinor_commutator_theorem"]["conjugacy_classes"] if r["order"] == 10 and r["q0_float"] > 0.8).__setitem__("pos_hits", 9))
        tamper(lambda d: next(r for r in d["spinor_commutator_theorem"]["conjugacy_classes"] if r["order"] == 10 and r["q0_float"] > 0.8).__setitem__("neg_hits", 1))
        tamper(lambda d: next(r for r in d["spinor_commutator_theorem"]["conjugacy_classes"] if r["order"] == 10 and r["q0_float"] > 0.8).__setitem__("q0_float", 0.8))

        for field, bad_val in (
            ("tau0_in_2i", False),
            ("tau0_order", 5),
            ("tau0_pow5_is_neg_id", False),
            ("comm_alpha_beta_equals_pos_tau0", False),
            ("sigma0_ord5_pow5_is_pos_id", False),
            ("order5_commutator_equals_neg_sigma0", False),
            ("and_truth_table_exact", False),
            ("not_truth_table_exact", False),
            ("max_intermediate_cayley_norm", 0.8),
        ):
            tamper(lambda d, k=field, v=bad_val: d["spinor_commutator_theorem"]["witness"].__setitem__(k, v))

        tamper(lambda d: d["benchmarks"].pop())
        tamper(lambda d: d["benchmarks"][0].__setitem__("q5_exact_matches", 0))
        tamper(lambda d: d["benchmarks"][0]["fp32_modes"]["s3_norm"].__setitem__("exact_decode_matches", 0))
        tamper(lambda d: d["benchmarks"][0]["fp32_modes"]["cayley_reproj"].__setitem__("overflow_count", 1))
        tamper(lambda d: d["benchmarks"][0]["fp32_modes"]["cayley_reproj"].__setitem__("max_quat_l2_err", 1e-5))
        tamper(lambda d: d["benchmarks"][0]["fp32_modes"]["binary_refresh"].__setitem__("max_quat_l2_err", 1e-6))

        def d16_row(d: dict[str, Any]) -> dict[str, Any]:
            return next(r for r in d["benchmarks"] if r["name"] == "shared_diamond_d16")

        tamper(lambda d: d16_row(d).__setitem__("barrington_exact_tree_len", 100))
        tamper(lambda d: d16_row(d).__setitem__("vring_slots", 200))
        tamper(lambda d: d16_row(d)["fp32_modes"]["unnormalized"].__setitem__("overflow_count", 0))
        tamper(lambda d: d16_row(d)["fp32_modes"]["s3_norm"].__setitem__("max_quat_l2_err", 1e-5))

    def test_02_cross_spine_bridge11_and_bridge12_exact_q5_and_tamper_controls(self) -> None:
        # Exact Q5 primitives, N1/N2/N3 guards, and dual inertia engines
        phi = s120.Q5(Fraction(1, 2), Fraction(1, 2))
        inv_phi = s120.Q5(Fraction(-1, 2), Fraction(1, 2))
        self.assertEqual(phi * inv_phi, s120.Q5_ONE)
        self.assertEqual(phi.inv(), inv_phi)
        self.assertEqual((phi.a, phi.b), (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(s120.Q5_ZERO.sign(), 0)
        self.assertEqual(s120.Q5(3, -1).sign(), 1)
        self.assertEqual(s120.Q5(2, -1).sign(), -1)
        self.assertEqual(s120.Q5(-2, 1).sign(), 1)
        self.assertEqual(s120.Q5(-3, 1).sign(), -1)
        self.assertFalse(s120.Q5_ONE == 1)

        for bad_q5 in (
            lambda: s120.Q5(float("inf")),  # type: ignore[arg-type]
            lambda: s120.Q5(0, float("nan")),  # type: ignore[arg-type]
            lambda: s120.Q5(True, 0),  # type: ignore[arg-type]
            lambda: s120.Q5(0, False),  # type: ignore[arg-type]
            lambda: s120.Q5_ZERO.inv(),
            lambda: s120.rho_su2_q5((s120.Q5_TWO, s120.Q5_ZERO, s120.Q5_ZERO, s120.Q5_ZERO)),
            lambda: s120.holonomy_scalar_block_q5((s120.Q5_ONE,)),  # type: ignore[arg-type]
            lambda: s120.holonomy_scalar_block_q5((s120.Q5_ONE, s120.Q5_ZERO, s120.Q5_ZERO, 0)),  # type: ignore[arg-type]
            lambda: s120.holonomy_scalar_block_q5("bad"),  # type: ignore[arg-type]
        ):
            with self.assertRaises(ValueError):
                bad_q5()

        z, o = s120.Q5_ZERO, s120.Q5_ONE
        hyperbolic = ((z, o, z), (o, z, z), (z, z, z))
        self.assertEqual(s120.inertia_descartes_q5(hyperbolic), (1, 1, 1))
        self.assertEqual(s120.inertia_sylvester_q5(hyperbolic), (1, 1, 1))
        zero_mat = ((z, z, z), (z, z, z), (z, z, z))
        self.assertEqual(s120.inertia_descartes_q5(zero_mat), (0, 0, 3))
        self.assertEqual(s120.inertia_sylvester_q5(zero_mat), (0, 0, 3))
        nonsym = ((o, o, z), (z, o, z), (z, z, o))
        with self.assertRaises(ValueError):
            s120.inertia_descartes_q5(nonsym)
        with self.assertRaises(ValueError):
            s120.inertia_sylvester_q5(nonsym)

        # bent4_quad_f2, bent4_polar_f2 (N3 bool guards + exact 0..15 bounds), & hadamard16_z
        self.assertEqual(s120.bent4_quad_f2(0), 0)
        self.assertEqual(s120.bent4_quad_f2(15), 0)
        self.assertEqual(s120.bent4_polar_f2(0, 15), 0)
        self.assertEqual(s120.bent4_polar_f2(15, 0), 0)
        self.assertEqual(s120.bent4_polar_f2(1, 4), 1)
        for bad_b in (
            lambda: s120.bent4_quad_f2(-1),
            lambda: s120.bent4_quad_f2(16),
            lambda: s120.bent4_quad_f2(True),  # type: ignore[arg-type]
            lambda: s120.bent4_polar_f2(-1, 0),
            lambda: s120.bent4_polar_f2(16, 0),
            lambda: s120.bent4_polar_f2(0, -1),
            lambda: s120.bent4_polar_f2(0, 16),
            lambda: s120.bent4_polar_f2(True, 0),  # type: ignore[arg-type]
            lambda: s120.bent4_polar_f2(0, False),  # type: ignore[arg-type]
            lambda: s120.hadamard16_z((0,) * 15),
            lambda: s120.hadamard16_z((0,) * 17),
        ):
            with self.assertRaises(ValueError):
                bad_b()

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = s120.main(["--json"])
        self.assertEqual(rc, 0)
        cert = json.loads(buf.getvalue())
        s120.validate_certificate(cert)
        self.assertEqual(cert["schema_version"], "holosphere.bridges.s120_certificate.v1")
        self.assertIn(cert["payload_sha256"], s120.format_summary(cert))
        buf_chk = io.StringIO()
        with redirect_stdout(buf_chk):
            rc_chk = s120.main(["--check"])
        self.assertEqual(rc_chk, 0)
        self.assertIn(cert["payload_sha256"], buf_chk.getvalue())

        # Exhaustive fail-closed tamper controls covering EVERY require_ok in validate_certificate
        def tamper(mutator: Callable[[dict[str, Any]], None]) -> None:
            bad = copy.deepcopy(cert)
            mutator(bad)
            with self.assertRaises(ValueError):
                s120.validate_certificate(bad)

        for field, val in (
            ("s117_bit_matches", 15),
            ("polar_identity_matches", 255),
            ("alternating_ok", False),
            ("symplectic_radical_size", 2),
            ("u4_group_order", 31),
            ("u4_unique_matrices", 31),
            ("u4_homomorphism_matches", 1023),
            ("u4_inverse_matches", 31),
            ("u4_commutator_matches", 1023),
            ("gr_u4_lie_bracket_matches", 255),
            ("pauli_4x4_commutator_matches", 255),
            ("wht_all_abs4", False),
            ("wht_self_dual_matches", 15),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_11"]["bent4_symplectic"].__setitem__(k, v))

        for field, val in (
            ("diag_norm_matches", 255),
            ("off_diag_zero_pairs", 32639),
            ("const_warps_bent4", 1),
            ("max_abs_warp_sum_bent4", 16),
            ("const_warps_raw_h256", 0),
            ("max_abs_warp_sum_raw_h256", 16),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_11"]["h256_bent4"].__setitem__(k, v))

        for field, val in (
            ("valid_codewords", 2047),
            ("sec_exhaustive_matches", 32767),
            ("ded_weight2_pairs", 119),
            ("f2_syndrome_linearity_matches", 32767),
            ("bent4_mod2_lift_matches", 255),
            ("upper_nibble_faults_tested", 47),
            ("lsb_only_secded_detected", 1),
            ("integer_h16_shifted", 47),
            ("integer_h16_parseval_exact", 47),
            ("bitplane_rotated_secded_detected", 47),
            ("bridge03_cross_term_z", 0),
            ("bridge03_cross_term_mod2", 1),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_11"]["a16_secded_carry_cover"].__setitem__(k, v))

        tamper(lambda d: d["bridge_12"]["schur_commutant_sym3"].__setitem__("rank", 4))
        tamper(lambda d: d["bridge_12"]["schur_commutant_sym3"].__setitem__("nullity", 2))
        tamper(lambda d: d["bridge_12"]["rotations"].pop())
        for field, val in (
            ("is_exact_so3", False),
            ("dual_method_all_equal", False),
            ("scalar_I3_charpoly_identical", False),
            ("scalar_I3_inertia_UL", [1, 2, 0]),
            ("scalar_I3_inertia_LU", [1, 2, 0]),
            ("G114_n_minus_split", False),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_12"]["rotations"][0].__setitem__(k, v))

        tamper(lambda d: d["bridge_12"]["rotations"][0].__setitem__("G311_inertia_UL", [1, 2, 0]))
        tamper(lambda d: d["bridge_12"]["rotations"][0].__setitem__("G311_inertia_LU", [2, 1, 0]))
        tamper(lambda d: d["bridge_12"]["rotations"][3].__setitem__("G311_inertia_UL", [1, 2, 0]))
        tamper(lambda d: d["bridge_12"]["rotations"][3].__setitem__("G311_inertia_LU", [2, 1, 0]))

        for field, val in (
            ("group_order", 119),
            ("central_neg_count", 1),
            ("zero_count", 1),
            ("nontrivial_pos_count", 117),
            ("pm_id_sign_blind_count", 119),
            ("h_plus_id_exact", "(1+0*sqrt5)"),
            ("h_tau0_exact", "(0+0*sqrt5)"),
            ("min_pos_h_exact", "(1+0*sqrt5)"),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_12"]["holonomy_block_2i"].__setitem__(k, v))

        for field, val in (
            ("dual_method_all_equal", False),
            ("UL_false_I", [3, 1, 0]),
            ("LU_false_I", [2, 2, 0]),
            ("UL_true_tau0", [2, 2, 0]),
            ("LU_true_tau0", [3, 1, 0]),
        ):
            tamper(lambda d, k=field, v=val: d["bridge_12"]["unified_kt_4x4_tau0_G114"].__setitem__(k, v))


if __name__ == "__main__":
    unittest.main()
