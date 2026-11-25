#!/usr/bin/env python3
"""Project HoloSphere R5 Adversarial, Mutation-Hardening & Live-Lean Test Suite.

Zero bare asserts (uses unittest assertions only), zero IEEE-754 floats,
zero external dependencies, zero skipped tests across all Python -I / -O / -W error envs.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

LEAN_DIR = Path(__file__).resolve().parent
if str(LEAN_DIR) not in sys.path:
    sys.path.insert(0, str(LEAN_DIR))

import holosphere_lean_oracle as oracle
import verify_holosphere_lean as ver

CERT_PATH = LEAN_DIR / "holosphere_lean_certificate.json"
LEAN_PATH = LEAN_DIR / oracle.LEAN_FILE_NAME
FROZEN_CERT = json.loads(CERT_PATH.read_text(encoding="utf-8"))


def _resign(cert: dict) -> dict:
    p_sha = oracle.sha256_hex(oracle.canonical_json_bytes(cert["payload"]))
    cert["payload_sha256"] = p_sha
    env = {
        "schema_version": cert["schema_version"],
        "track_id": cert["track_id"],
        "payload_sha256": p_sha,
    }
    cert["certificate_sha256"] = oracle.sha256_hex(oracle.canonical_json_bytes(env))
    return cert


def _walk_leaves(obj, path: list, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_leaves(v, path + [k], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_leaves(v, path + [i], out)
    else:
        out.append((path, obj))


def _set_path(obj, path: list, val) -> None:
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = val


def _tamper_scalar(val):
    if isinstance(val, bool):
        return not val
    if isinstance(val, int):
        return val + 1
    if isinstance(val, str):
        return val + "_X" if val else "X"
    return "TAMPER"


class Test01OraclePrimitivesAndCLI(unittest.TestCase):
    def test_01_oracle_guards_and_math_primitives(self) -> None:
        oracle.require_ok(True, "OK", "ok")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm:
            oracle.require_ok(False, "FAIL_CODE", "fail")
        self.assertEqual(cm.exception.code, "FAIL_CODE")
        self.assertEqual(oracle.require_exact_int(7, "x"), 7)
        with self.assertRaises(oracle.HoloSphereOracleError):
            oracle.require_exact_int(True, "bool")
        with self.assertRaises(oracle.HoloSphereOracleError):
            oracle.require_exact_int("7", "str")
        self.assertEqual(oracle.require_exact_str("abc", "s"), "abc")
        with self.assertRaises(oracle.HoloSphereOracleError):
            oracle.require_exact_str(123, "int")
        self.assertEqual(oracle.q_str(Fraction(5, 1)), "5")
        self.assertEqual(oracle.q_str(Fraction(5, 3)), "5/3")
        with self.assertRaises(oracle.HoloSphereOracleError):
            oracle.q_str(5)  # type: ignore[arg-type]

        with self.assertRaises(oracle.HoloSphereOracleError) as cm_s8:
            oracle.hillis_steele_8_q([oracle.q_make(1, 0, 0, 0)])
        self.assertEqual(cm_s8.exception.code, "SCAN8_LEN")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_es:
            oracle.sequential_scan_q([])
        self.assertEqual(cm_es.exception.code, "EMPTY_SCAN")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_em:
            oracle.mat_mul_q([], [])
        self.assertEqual(cm_em.exception.code, "EMPTY_MATRIX")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_dm:
            oracle.mat_mul_q([[Fraction(1), Fraction(2)]], [[Fraction(1)]])
        self.assertEqual(cm_dm.exception.code, "DIM_MISMATCH")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_mv:
            oracle.mat_vec_q([[Fraction(1), Fraction(2)]], [Fraction(1)])
        self.assertEqual(cm_mv.exception.code, "DIM_MISMATCH")
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_zn:
            oracle.row_concentration_q([[Fraction(0), Fraction(0)]])
        self.assertEqual(cm_zn.exception.code, "ZERO_NORM_MATRIX")

    def test_02_oracle_main_and_live_cert_parity(self) -> None:
        live = oracle.build_certificate()
        self.assertEqual(live, FROZEN_CERT)
        buf_sum = io.StringIO()
        buf_std = io.StringIO()
        old_out = sys.stdout
        try:
            sys.stdout = buf_sum
            self.assertEqual(oracle.main([]), 0)
            sys.stdout = buf_std
            self.assertEqual(oracle.main(["--standalone"]), 0)
        finally:
            sys.stdout = old_out
        summary = json.loads(buf_sum.getvalue())
        self.assertEqual(summary["lean_theorems"], 36)
        self.assertEqual(summary["certificate_sha256"], FROZEN_CERT["certificate_sha256"])
        self.assertEqual(buf_std.getvalue().encode("utf-8"), CERT_PATH.read_bytes())
        with self.assertRaises(oracle.HoloSphereOracleError) as cm_arg:
            oracle.main(["--bad-flag"])
        self.assertEqual(cm_arg.exception.code, "UNKNOWN_ORACLE_ARG")


class Test02ZoneBIOAndStrictJSON(unittest.TestCase):
    def test_03_regular_file_io_and_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory(prefix="holo_io_") as td:
            d = Path(td)
            f = d / "sample.dat"
            ver.write_atomic_0600(f, b"hello-holo")
            self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
            self.assertEqual(ver.read_regular_bytes_capped(f, "s", max_bytes=10), b"hello-holo")
            with self.assertRaises(ver.HoloSphereLeanError) as cm_big:
                ver.read_regular_bytes_capped(f, "s", max_bytes=9)
            self.assertEqual(cm_big.exception.code, "FILE_TOO_LARGE")

            big_data = b"x" * 70000
            big_f = d / "big.dat"
            ver.write_atomic_0600(big_f, big_data)
            self.assertEqual(len(ver.read_regular_bytes_capped(big_f, "big")), 70000)

            sym = d / "link.dat"
            sym.symlink_to(f)
            with self.assertRaises(ver.HoloSphereLeanError) as cm_sym:
                ver.read_regular_bytes_capped(sym, "sym")
            self.assertEqual(cm_sym.exception.code, "SYMLINK_FORBIDDEN")

            sub = d / "real_sub"
            sub.mkdir()
            (sub / "inside.dat").write_bytes(b"abc")
            sym_dir = d / "sym_sub"
            sym_dir.symlink_to(sub)
            with self.assertRaises(ver.HoloSphereLeanError) as cm_anc:
                ver.read_regular_bytes_capped(sym_dir / "inside.dat", "anc")
            self.assertEqual(cm_anc.exception.code, "SYMLINK_FORBIDDEN")

            with self.assertRaises(ver.HoloSphereLeanError) as cm_dir:
                ver.read_regular_bytes_capped(d, "dir")
            self.assertEqual(cm_dir.exception.code, "NON_REGULAR_FILE")

            with self.assertRaises(ver.HoloSphereLeanError) as cm_miss:
                ver.read_regular_bytes_capped(d / "nope", "nope")
            self.assertEqual(cm_miss.exception.code, "IO_OPEN_FAIL")
            with self.assertRaises(ver.HoloSphereLeanError) as cm_wp:
                ver.write_atomic_0600(d / "missing_dir" / "out.json", b"{}")
            self.assertEqual(cm_wp.exception.code, "IO_WRITE_FAIL")

            target_dir = d / "isdir"
            target_dir.mkdir()
            with self.assertRaises(ver.HoloSphereLeanError) as cm_wd:
                ver.write_atomic_0600(target_dir, b"{}")
            self.assertEqual(cm_wd.exception.code, "IO_WRITE_FAIL")
            self.assertEqual(list(d.glob(".holo_lean_tmp_*")), [])

    def test_04_strict_json_dos_and_type_guards(self) -> None:
        self.assertEqual(ver.loads_json_strict(b'{"a": 1, "b": -2}', "ok"), {"a": 1, "b": -2})
        d80 = "9" * 80
        d81 = "9" * 81
        d4500 = "9" * 4500
        self.assertEqual(ver.loads_json_strict(f'{{"n": {d80}}}'.encode(), "ok80")["n"], int(d80))
        self.assertEqual(ver.loads_json_strict(f'{{"n": -{d80}}}'.encode(), "ok80n")["n"], -int(d80))
        for raw, expected_code in (
            (b"\xff\xfe", "INVALID_UTF8"),
            (b'{"a": "\x00"}', "INVALID_JSON"),
            (b'{"a": 1, "a": 2}', "DUPLICATE_JSON_KEY"),
            (b'{"a": 1.5}', "FLOAT_FORBIDDEN"),
            (b'{"a": NaN}', "NON_FINITE_FORBIDDEN"),
            (b'{"a": Infinity}', "NON_FINITE_FORBIDDEN"),
            (f'{{"n": {d81}}}'.encode(), "INT_DIGIT_LIMIT"),
            (f'{{"n": -{d81}}}'.encode(), "INT_DIGIT_LIMIT"),
            (f'{{"n": {d4500}}}'.encode(), "INT_DIGIT_LIMIT"),
            (b"{bad json", "INVALID_JSON"),
        ):
            with self.assertRaises(ver.HoloSphereLeanError) as cm:
                ver.loads_json_strict(raw, "bad")
            self.assertEqual(cm.exception.code, expected_code, f"for {raw[:30]!r}")


class Test03R13bLeanLexerAndStaticGate(unittest.TestCase):
    def test_05_r13b_o1_in_ident_lexer_and_char_literals(self) -> None:
        clean_src = (
            "/- outer /- inner -/ still comment -/\n"
            "-- line comment with sorry \"bad\"\n"
            "def f (x' x1' foo.bar' x'' : Int) := x' + x1' - 1 / 2"
        )
        stripped = ver.strip_lean4_source(clean_src)
        self.assertIn("x' + x1'", stripped)
        self.assertNotIn("sorry", stripped)
        self.assertEqual(ver.lean4_rule_hits(clean_src), [])
        self.assertEqual(ver.lean4_rule_hits("def a := 1 -- sorry"), [])
        self.assertEqual(ver.lean4_rule_hits("/- a - sorry -/"), [])
        self.assertEqual(ver.lean4_rule_hits("def f := a - sorry"), ["R03_SORRY_ADMIT"])
        self.assertIn("x'", ver.strip_lean4_source("def f := x' '"))

        with self.assertRaises(ver.HoloSphereLeanError) as cm_un:
            ver.strip_lean4_source("/- outer /- inner -/ unclosed")
        self.assertEqual(cm_un.exception.code, "LEAN4_UNTERMINATED_COMMENT")

        for atk_name, src, expect_rule in (
            ("X1_bang_char_quote", "def x := !'\"'.isDigit\nsorry", "R03_SORRY_ADMIT"),
            ("X2_digit_char_quote", "def x := 1'\"'\nsorry", "R03_SORRY_ADMIT"),
            ("X3_guil_char_quote", "def x := «x»'\"'\nsorry", "R01_LITERAL_OR_GUILLEMET"),
            ("literal_lf_char", "def c := '\n'\nsorry", "R03_SORRY_ADMIT"),
            ("esc_nl_char", "def c := '\\n'\nsorry", "R03_SORRY_ADMIT"),
            ("esc_hex_char", "def c := '\\x27'\nsorry", "R03_SORRY_ADMIT"),
            ("esc_uni_char", "def c := '\\u0027'\nsorry", "R03_SORRY_ADMIT"),
        ):
            hits = ver.lean4_rule_hits(src)
            self.assertIn(expect_rule, hits, f"{atk_name} missed: {hits}")

    def test_06_all_18_static_lean_rules_and_structural_checks(self) -> None:
        samples = [
            ("R01_LITERAL_OR_GUILLEMET", 'def s := "sorry"'),
            ("R01_LITERAL_OR_GUILLEMET", 'def s := s!"{sorry}"'),
            ("R01_LITERAL_OR_GUILLEMET", 'def s := r#"sorry"#'),
            ("R01_LITERAL_OR_GUILLEMET", "def «bad» := 1"),
            ("R02_IMPORT", "import Init"),
            ("R03_SORRY_ADMIT", "theorem t : True := sorry"),
            ("R04_AXIOM_DECL", "axiom bad : False"),
            ("R05_UNSAFE_EXTERN", "unsafe def f := 1"),
            ("R05_UNSAFE_EXTERN", "@[implemented_by foo] def f := 1"),
            ("R06_OPAQUE_PARTIAL", "partial def f : Nat := f"),
            ("R06_OPAQUE_PARTIAL", "opaque g : Nat"),
            ("R07_NATIVE_DECIDE", "theorem t : True := by native_decide"),
            ("R07_NATIVE_DECIDE", "theorem t : True := by decide +native"),
            ("R08_DBG_TRACE", "def f := dbg_trace 1"),
            ("R08_DBG_TRACE", "def f := dbgSleep 1"),
            ("R09_INIT_EXPORT_ATTR", "@[export foo] def f := 1"),
            ("R09_INIT_EXPORT_ATTR", "initialize x : Nat := pure 1"),
            ("R10_GUARD_MSGS", "#guard_msgs in #check 1"),
            ("R11_WARN_SORRY_OPT", "warn.sorry"),
            ("R12_META_EVAL_CMD", "#eval 1"),
            ("R13_SET_OPTION", "set_option maxRecDepth 1000"),
            ("R14_ELAB_TACTIC_MACRO", "macro_rules | `(x) => `(1)"),
            ("R15_IO_SYSTEM", "def f := IO.println"),
            ("R16_PANIC_UNREACHABLE", "def f := panic!"),
            ("R17_DECIDE_CONFIG", "by decide (config := { native := true })"),
            ("R18_ROOT_SORRYAX", "_root_.sorryAx"),
        ]
        for rule_id, snippet in samples:
            self.assertIn(rule_id, ver.lean4_rule_hits(snippet), f"missed {rule_id}")

        real_src = LEAN_PATH.read_text(encoding="utf-8")
        ver.static_gate_lean4_source(real_src)
        with self.assertRaises(ver.HoloSphereLeanError) as cm_esc:
            ver.static_gate_lean4_source(real_src + "\n-- ok\nsorry\n")
        self.assertEqual(cm_esc.exception.code, "LEAN4_TRUST_ESCAPE")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_cr:
            ver.static_gate_lean4_source(real_src + "\r")
        self.assertEqual(cm_cr.exception.code, "LEAN4_BAD_CHARS")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_nul:
            ver.static_gate_lean4_source(real_src + "\x00")
        self.assertEqual(cm_nul.exception.code, "LEAN4_BAD_CHARS")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_pr:
            ver.static_gate_lean4_source(real_src + "\n#print Nat\n")
        self.assertEqual(cm_pr.exception.code, "LEAN4_BAD_PRINT")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_ps:
            ver.static_gate_lean4_source(real_src.rsplit("\n#print axioms", 1)[0])
        self.assertEqual(cm_ps.exception.code, "LEAN4_PRINT_SET_MISMATCH")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_mt:
            ver.static_gate_lean4_source(real_src.replace("theorem p1_quat_mul_assoc", "def p1_quat_mul_assoc", 1))
        self.assertEqual(cm_mt.exception.code, "LEAN4_MISSING_THEOREM")

    def test_07_axiom_parser_and_env_scrub(self) -> None:
        lines = []
        for idx, name in enumerate(oracle.LEAN_THEOREM_NAMES):
            if idx == 0:
                lines.append(f"'{name}' does not depend on any axioms")
            elif idx == 1:
                lines.append(f"'{name}' depends on axioms: [\n  propext,\n  Quot.sound\n]")
            else:
                lines.append(f"'{name}' depends on axioms: [propext, Quot.sound]")
        parsed = ver.parse_lean_axiom_output("\n".join(lines))
        self.assertEqual(len(parsed), 36)
        self.assertEqual(parsed[oracle.LEAN_THEOREM_NAMES[0]], [])
        self.assertEqual(parsed[oracle.LEAN_THEOREM_NAMES[1]], ["propext", "Quot.sound"])

        with self.assertRaises(ver.HoloSphereLeanError) as cm_unexp:
            ver.parse_lean_axiom_output("unexpected noise line")
        self.assertEqual(cm_unexp.exception.code, "LEAN_UNEXPECTED_STDOUT")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_eof:
            ver.parse_lean_axiom_output("'HoloSphereInvariants.p1_quat_mul_assoc' depends on axioms: [propext")
        self.assertEqual(cm_eof.exception.code, "LEAN_UNEXPECTED_STDOUT")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_dup:
            ver.parse_lean_axiom_output("\n".join(lines + [lines[0]]))
        self.assertEqual(cm_dup.exception.code, "LEAN_DUPLICATE_THEOREM")
        bad_ax = list(lines)
        bad_ax[2] = f"'{oracle.LEAN_THEOREM_NAMES[2]}' depends on axioms: [propext, Classical.choice]"
        with self.assertRaises(ver.HoloSphereLeanError) as cm_ax:
            ver.parse_lean_axiom_output("\n".join(bad_ax))
        self.assertEqual(cm_ax.exception.code, "LEAN_FORBIDDEN_AXIOM")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_set:
            ver.parse_lean_axiom_output("\n".join(lines[:-1]))
        self.assertEqual(cm_set.exception.code, "LEAN_THEOREM_SET_MISMATCH")

        old_env = dict(os.environ)
        try:
            os.environ["LEAN_PATH"] = "/tmp/poison"
            os.environ["ELAN_TOOLCHAIN"] = "poison"
            os.environ["LD_PRELOAD"] = "/tmp/evil.so"
            clean = ver.scrubbed_lean_env()
            for bad_k in ("LEAN_PATH", "ELAN_TOOLCHAIN", "LD_PRELOAD"):
                self.assertNotIn(bad_k, clean)
            os.environ["ALETHEIA_LEAN_BIN"] = "/nonexistent/lean_bin_holo"
            with self.assertRaises(ver.HoloSphereLeanError) as cm_nf:
                ver.resolve_lean_binary()
            self.assertEqual(cm_nf.exception.code, "LEAN_BINARY_NOT_FILE")
            with tempfile.NamedTemporaryFile(prefix="holo_noexec_", delete=False) as tf:
                noexec = Path(tf.name)
            try:
                os.chmod(noexec, 0o600)
                os.environ["ALETHEIA_LEAN_BIN"] = str(noexec)
                with self.assertRaises(ver.HoloSphereLeanError) as cm_nx:
                    ver.resolve_lean_binary()
                self.assertEqual(cm_nx.exception.code, "LEAN_BINARY_NOT_EXECUTABLE")
                os.chmod(noexec, 0o700)
                self.assertEqual(ver.resolve_lean_binary(), str(noexec))
            finally:
                noexec.unlink()
            os.environ.pop("ALETHEIA_LEAN_BIN", None)
            os.environ["PATH"] = ""
            with self.assertRaises(ver.HoloSphereLeanError) as cm_na:
                ver.resolve_lean_binary()
            self.assertEqual(cm_na.exception.code, "LEAN_BINARY_UNAVAILABLE")
        finally:
            os.environ.clear()
            os.environ.update(old_env)


class Test04SemanticAndShapeAudits(unittest.TestCase):
    def test_08_shape_deep_equal_rational_and_aliasing(self) -> None:
        a = ver.expected_payload()
        b = ver.expected_payload()
        a["novelty_grade"] = "MUTATED"
        self.assertEqual(b["novelty_grade"], "N1")
        self.assertEqual(ver.strict_shape_match({"a": [1, True]}, {"a": [2, False]}), "")
        self.assertEqual(ver.strict_shape_match({"a": True}, {"a": 1}), "payload.a")
        self.assertEqual(ver.strict_shape_match({"a": 1}, {"b": 1}), "payload")
        self.assertEqual(ver.strict_shape_match({"a": [1]}, {"a": [1, 2]}), "payload.a")
        self.assertEqual(ver.strict_deep_equal({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}), "")
        self.assertEqual(ver.strict_deep_equal({"a": True}, {"a": 1}), "payload.a")
        self.assertEqual(ver.strict_deep_equal({"a": 1}, {"b": 1}), "payload")
        self.assertEqual(ver.strict_deep_equal({"a": [1]}, {"a": [1, 2]}), "payload.a")
        self.assertEqual(ver.strict_deep_equal({"a": [1, 2]}, {"a": [1, 3]}), "payload.a[1]")

        self.assertEqual(ver.parse_capped_rational("-3/4", "r"), Fraction(-3, 4))
        self.assertEqual(ver.parse_capped_rational("0", "r"), Fraction(0))
        for bad_r in (123, "", "1/0", "1.5", "1/" + "9" * 81, "9" * 81, "NaN"):
            with self.assertRaises(ver.HoloSphereLeanError) as cm:
                ver.parse_capped_rational(bad_r, "r")
            self.assertEqual(cm.exception.code, "INVALID_RATIONAL_STR")

    def test_09_all_independent_semantic_codes_and_exact_boundaries(self) -> None:
        cases = [
            (["novelty_grade"], "N2", "SEC_NOVELTY"),
            (["n2_claims"], 1, "SEC_N2_ZERO"),
            (["lean_companion", "named_theorem_count"], 35, "SEC_LEAN_CNT"),
            (["lean_companion", "total_theorem_declarations"], 99, "SEC_LEAN_TOTAL"),
            (["pillar1_su2_scan_and_suffix_adjoint", "assoc_diff_sq_q"], "1", "SEC_P1_ASSOC"),
            (["pillar1_su2_scan_and_suffix_adjoint", "nonabelian_commutator_norm_sq_q"], "0", "SEC_P1_NONABELIAN"),
            (["pillar1_su2_scan_and_suffix_adjoint", "final_pi7_norm_sq_q"], "2", "SEC_P1_UNIT_NORM"),
            (["pillar1_su2_scan_and_suffix_adjoint", "hillis_vs_seq_8lane_diff_sq_q"], "1", "SEC_P1_HILLIS"),
            (["pillar1_su2_scan_and_suffix_adjoint", "chunked_vs_seq_diff_sq_q"], "1", "SEC_P1_CHUNKED"),
            (["pillar1_su2_scan_and_suffix_adjoint", "per_step_grad_q"], "0", "SEC_P1_GRAD_NZ"),
            (["pillar1_su2_scan_and_suffix_adjoint", "suffix_sum_grad_q"], "1", "SEC_P1_ADJ_EQ"),
            (["pillar1_su2_scan_and_suffix_adjoint", "adjoint_collapse_diff_q"], "1", "SEC_P1_ADJ_DIFF"),
            (["pillar1_su2_scan_and_suffix_adjoint", "cayley_su2_norm_err_q"], "1", "SEC_P1_CSU2_NORM"),
            (["pillar1_su2_scan_and_suffix_adjoint", "cayley_su2_acute_w_q"], "0", "SEC_P1_CSU2_POS"),
            (["pillar1_su2_scan_and_suffix_adjoint", "local_4d_bank_a_bleed_sq_q"], "1", "SEC_P1_LOCAL_ZERO"),
            (["pillar1_su2_scan_and_suffix_adjoint", "global_coupled_bank_a_bleed_sq_q"], "0", "SEC_P1_GLOBAL_BLEED"),
            (["pillar2_covariant_qkv_transport", "transport_isometry_diff_q"], "1", "SEC_P2_ISO"),
            (["pillar2_covariant_qkv_transport", "qk_relative_holonomy_diff_q"], "1", "SEC_P2_QK"),
            (["pillar2_covariant_qkv_transport", "interval_cancel_diff_sq_q"], "1", "SEC_P2_INTERVAL"),
            (["pillar2_covariant_qkv_transport", "weighted_untransport_diff_sq_q"], "1", "SEC_P2_UNTRANS"),
            (["pillar2_covariant_qkv_transport", "self_untransport_diff_sq_q"], "1", "SEC_P2_SELF"),
            (["pillar3_cayley_safe_hopf_and_fiber", "unit_circle_max_err_q"], "1", "SEC_P3_CIRCLE"),
            (["pillar3_cayley_safe_hopf_and_fiber", "cos_floor_tmax_4_5_q"], "0", "SEC_P3_FLOOR_POS"),
            (["pillar3_cayley_safe_hopf_and_fiber", "min_schedule_cos_q"], "8/41", "SEC_P3_FLOOR_BOUND"),
            (["pillar3_cayley_safe_hopf_and_fiber", "strictly_monotone_schedule"], False, "SEC_P3_MONO"),
            (["pillar3_cayley_safe_hopf_and_fiber", "obs31_obtuse_cos_q"], "0", "SEC_P3_OBS31_COS"),
            (["pillar3_cayley_safe_hopf_and_fiber", "obs31_self_align_q"], "0", "SEC_P3_OBS31_ALIGN"),
            (["pillar3_cayley_safe_hopf_and_fiber", "hopf_s2_norm_sq_q"], "2", "SEC_P3_HOPF_NORM"),
            (["pillar3_cayley_safe_hopf_and_fiber", "fiber_rotated_diff_sq_q"], "1", "SEC_P3_HOPF_INV"),
            (["pillar3_cayley_safe_hopf_and_fiber", "off_fiber_diff_sq_q"], "0", "SEC_P3_HOPF_OFF"),
            (["pillar4_beltrami_bingham_darboux", "beltrami_commutator_diff_sq_q"], "1", "SEC_P4_COMM"),
            (["pillar4_beltrami_bingham_darboux", "beltrami_energy_out_q"], "15", "SEC_P4_ENERGY"),
            (["pillar4_beltrami_bingham_darboux", "beltrami_helicity_out_q"], "35", "SEC_P4_HELICITY"),
            (["pillar4_beltrami_bingham_darboux", "cross_swap_energy_out_q"], "15", "SEC_P4_SWAP_E"),
            (["pillar4_beltrami_bingham_darboux", "cross_swap_helicity_out_q"], "36", "SEC_P4_SWAP_H"),
            (["pillar4_beltrami_bingham_darboux", "rayleigh_after_sym_res_q"], "0", "SEC_P4_RAYLEIGH"),
            (["pillar4_beltrami_bingham_darboux", "bingham_pol_in_norm_sq_q"], "10", "SEC_P4_LID_COND"),
            (["pillar4_beltrami_bingham_darboux", "bingham_pol_gated_norm_sq_q"], "1", "SEC_P4_LID_ZERO"),
            (["pillar4_beltrami_bingham_darboux", "bingham_tor_out_norm_sq_q"], "0", "SEC_P4_TOR_POS"),
            (["pillar4_beltrami_bingham_darboux", "darboux_intertwine_diff_sq_q"], "1", "SEC_P4_DARBOUX"),
            (["pillar4_beltrami_bingham_darboux", "darboux_ak_norm_sq_q"], "1", "SEC_P4_KER"),
            (["pillar4_beltrami_bingham_darboux", "darboux_kernel_collision_diff_sq_q"], "1", "SEC_P4_COLL"),
            (["pillar4_beltrami_bingham_darboux", "cayley2x2_closed_form_grad_w_q"], "0", "SEC_P4_CAYLEY_GRAD_NZ"),
            (["pillar4_beltrami_bingham_darboux", "cayley2x2_adjoint_diff_q"], "1", "SEC_P4_CAYLEY_ADJ"),
            (["pillar5_weft_crow_safety_and_rollback", "c_row_patch1_q"], "4/5", "SEC_P5_P1_SAFE"),
            (["pillar5_weft_crow_safety_and_rollback", "c_row_patch2_q"], "4/5", "SEC_P5_P2_SAFE"),
            (["pillar5_weft_crow_safety_and_rollback", "c_row_composed_attack_q"], "3/5", "SEC_P5_COMP_BREACH"),
            (["pillar5_weft_crow_safety_and_rollback", "composed_attack_accepted"], True, "SEC_P5_ATK_BLOCK"),
            (["pillar5_weft_crow_safety_and_rollback", "safe_single_patch_accepted"], False, "SEC_P5_SAFE_ACCEPT"),
            (["pillar5_weft_crow_safety_and_rollback", "c_row_post_isotropic_sleep_q"], "2/5", "SEC_P5_SLEEP_INV"),
            (["pillar5_weft_crow_safety_and_rollback", "lifo_rollback_diff_sq_q"], "1", "SEC_P5_ROLLBACK"),
            (["pillar5_weft_crow_safety_and_rollback", "repoison_err_step1_sq_q"], "0", "SEC_P5_REPOISON_POS"),
            (["pillar5_weft_crow_safety_and_rollback", "repoison_err_step5_sq_q"], "1", "SEC_P5_REPOISON_CONST"),
            (["pillar5_weft_crow_safety_and_rollback", "restored_16b_err_sq_q"], "1", "SEC_P5_RESTORE_16B"),
            (["pillar5_weft_crow_safety_and_rollback", "asym_recovery_excess_sq_q"], "1", "SEC_P5_ASYM_ZERO"),
            (["pillar5_weft_crow_safety_and_rollback", "sym_recovery_dist_sq_q"], "0", "SEC_P5_SYM_POS"),
            (["pillar5_weft_crow_safety_and_rollback", "gate_asymmetric_recovery_accepted"], False, "SEC_P5_ASYM_OK"),
            (["pillar5_weft_crow_safety_and_rollback", "gate_symmetric_zero_headroom_accepted"], True, "SEC_P5_SYM_DEADLOCK"),
            (["pillar5_weft_crow_safety_and_rollback", "left_quant_nullspace_drift_sq_q"], "1", "SEC_P5_LEFT_QUANT"),
            (["pillar5_weft_crow_safety_and_rollback", "right_quant_nullspace_drift_sq_q"], "0", "SEC_P5_RIGHT_QUANT"),
        ]
        for path, bad_val, expected_code in cases:
            p = ver.expected_payload()
            _set_path(p, path, bad_val)
            with self.assertRaises(ver.HoloSphereLeanError) as cm:
                ver.secondary_audit_payload(p)
            self.assertEqual(cm.exception.code, expected_code, f"path={path}")

        with self.assertRaises(ver.HoloSphereLeanError) as cm_sh:
            ver.secondary_audit_payload({"wrong_key": 1})
        self.assertEqual(cm_sh.exception.code, "SCHEMA_SHAPE_MISMATCH")


class Test05ExhaustiveLeafSweepAndEnvelope(unittest.TestCase):
    def test_10_envelope_and_replay_rejection(self) -> None:
        res_fr1 = ver.verify_certificate_dict(FROZEN_CERT, full_replay=True)
        res_fr0 = ver.verify_certificate_dict(FROZEN_CERT, full_replay=False)
        self.assertEqual(res_fr1["status"], "OK")
        self.assertTrue(res_fr1["full_replay"])
        self.assertFalse(res_fr0["full_replay"])

        with self.assertRaises(ver.HoloSphereLeanError) as cm_t:
            ver.verify_certificate_dict([])
        self.assertEqual(cm_t.exception.code, "INVALID_ENVELOPE_TYPE")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_pt:
            ver.secondary_audit_payload([])
        self.assertEqual(cm_pt.exception.code, "INVALID_PAYLOAD_TYPE")

        for mut_fn, exp_code in (
            (lambda c: c.pop("track_id"), "INVALID_ENVELOPE_KEYS"),
            (lambda c: c.__setitem__("schema_version", "wrong"), "SCHEMA_VERSION_MISMATCH"),
            (lambda c: c.__setitem__("track_id", "wrong"), "TRACK_ID_MISMATCH"),
            (lambda c: c.__setitem__("payload_sha256", 123), "INVALID_HASH"),
            (lambda c: c.__setitem__("payload_sha256", "Z" * 64), "INVALID_HASH"),
            (lambda c: c.__setitem__("certificate_sha256", 123), "INVALID_HASH"),
            (lambda c: c.__setitem__("certificate_sha256", "Z" * 64), "INVALID_HASH"),
            (lambda c: c.__setitem__("payload", []), "INVALID_PAYLOAD_TYPE"),
            (lambda c: c.__setitem__("payload_sha256", "0" * 64), "PAYLOAD_HASH_MISMATCH"),
            (lambda c: c.__setitem__("certificate_sha256", "0" * 64), "CERT_HASH_MISMATCH"),
        ):
            c = copy.deepcopy(FROZEN_CERT)
            mut_fn(c)
            with self.assertRaises(ver.HoloSphereLeanError) as cm:
                ver.verify_certificate_dict(c, full_replay=True)
            self.assertEqual(cm.exception.code, exp_code)

        ver._validate_exact_json_tree({"s": "ok", "b": True, "i": (1 << ver.MAX_INT_BITS) - 1, "l": [1, False]})
        with self.assertRaises(ver.HoloSphereLeanError) as cm_k:
            ver._validate_exact_json_tree({1: "bad"})
        self.assertEqual(cm_k.exception.code, "INVALID_JSON_KEY")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_bi:
            ver._validate_exact_json_tree({"n": 1 << ver.MAX_INT_BITS})
        self.assertEqual(cm_bi.exception.code, "INT_DIGIT_LIMIT")
        with self.assertRaises(ver.HoloSphereLeanError) as cm_tp:
            ver._validate_exact_json_tree({"bad": (1, 2)})
        self.assertEqual(cm_tp.exception.code, "INVALID_JSON_TYPE")

        with tempfile.TemporaryDirectory(prefix="holo_canon_") as td:
            d = Path(td)
            crlf_p = d / "crlf.json"
            crlf_p.write_bytes(CERT_PATH.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaises(ver.HoloSphereLeanError) as cm_cr:
                ver.verify_certificate_file(crlf_p)
            self.assertEqual(cm_cr.exception.code, "NON_CANONICAL_BYTES")
            pretty_p = d / "pretty.json"
            pretty_p.write_text(json.dumps(FROZEN_CERT, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(ver.HoloSphereLeanError) as cm_pr:
                ver.verify_certificate_file(pretty_p)
            self.assertEqual(cm_pr.exception.code, "NON_CANONICAL_BYTES")

        orig_build = oracle.build_payload
        try:
            def _mutated_payload():
                p = orig_build()
                p["scope_note"] = "DIFF"
                return p
            oracle.build_payload = _mutated_payload
            with self.assertRaises(ver.HoloSphereLeanError) as cm_rep:
                ver.verify_certificate_dict(FROZEN_CERT, full_replay=True)
            self.assertEqual(cm_rep.exception.code, "REPLAY_MISMATCH")
        finally:
            oracle.build_payload = orig_build

    def test_11_exhaustive_resigned_leaf_sweep_all_3_surfaces(self) -> None:
        leaves: list = []
        _walk_leaves(FROZEN_CERT["payload"], [], leaves)
        self.assertGreaterEqual(len(leaves), 80)
        for path, orig in leaves:
            c = copy.deepcopy(FROZEN_CERT)
            _set_path(c["payload"], path, _tamper_scalar(orig))
            c = _resign(c)
            snap = copy.deepcopy(c)
            with self.assertRaises(ver.HoloSphereLeanError):
                ver.secondary_audit_payload(c["payload"])
            self.assertEqual(c, snap)
            with self.assertRaises(ver.HoloSphereLeanError):
                ver.verify_certificate_dict(c, full_replay=False)
            self.assertEqual(c, snap)
            with self.assertRaises(ver.HoloSphereLeanError):
                ver.verify_certificate_dict(c, full_replay=True)
            self.assertEqual(c, snap)


class Test06CLIFlags(unittest.TestCase):
    def test_12_cli_modes_and_fail_closed_flags(self) -> None:
        def _run_cli(argv: list[str]) -> tuple[int, str, str]:
            out_buf, err_buf = io.StringIO(), io.StringIO()
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = out_buf, err_buf
                rc = ver.main(argv)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            return rc, out_buf.getvalue(), err_buf.getvalue()

        rc, out, _ = _run_cli([])
        self.assertEqual(rc, 0)
        self.assertIn(FROZEN_CERT["certificate_sha256"], out)

        rc, out, _ = _run_cli(["--check"])
        self.assertEqual(rc, 0)
        self.assertIn(FROZEN_CERT["certificate_sha256"], out)

        rc, out, _ = _run_cli(["--check", str(CERT_PATH), "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["status"], "OK")

        with tempfile.TemporaryDirectory(prefix="holo_cli_") as td:
            d = Path(td)
            out_cert = d / "out.json"
            rc, out, _ = _run_cli(["--write", str(out_cert), "--json"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["status"], "WRITTEN")
            self.assertEqual(out_cert.read_bytes(), CERT_PATH.read_bytes())
            self.assertEqual(stat.S_IMODE(out_cert.stat().st_mode), 0o600)

        for bad_argv, code in (
            (["--unknown"], "UNKNOWN_FLAG"),
            (["--check", "--verify-lean"], "CONFLICTING_FLAGS"),
            (["--check", "/nonexistent/holo_cert.json"], "IO_OPEN_FAIL"),
        ):
            rc, _, err = _run_cli(bad_argv)
            self.assertEqual(rc, 2)
            self.assertIn(code, err)


class Test07LiveLeanAndPoisonedEnvHijack(unittest.TestCase):
    def test_13_live_lean_and_cli_verify_lean(self) -> None:
        res = ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
        self.assertEqual(res["theorems_verified"], 36)
        self.assertEqual(res["lean_source_sha256"], oracle.LEAN_SOURCE_SHA256)
        self.assertIn("4.36.0", res["lean_version"])

        with tempfile.TemporaryDirectory(prefix="holo_lean_") as td:
            d = Path(td)
            bad_utf8 = d / "bad.lean"
            bad_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(ver.HoloSphereLeanError) as cm_u:
                ver.verify_lean4_file_live(bad_utf8, check_sha=False)
            self.assertEqual(cm_u.exception.code, "INVALID_UTF8")

            mod_lean = d / "mod.lean"
            mod_lean.write_text(LEAN_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ver.HoloSphereLeanError) as cm_sha:
                ver.verify_lean4_file_live(mod_lean, check_sha=True)
            self.assertEqual(cm_sha.exception.code, "LEAN_SHA_MISMATCH")

            def _make_fake(name: str, body: str) -> Path:
                p = d / name
                p.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
                os.chmod(p, 0o700)
                return p

            old_bin = os.environ.get("ALETHEIA_LEAN_BIN")
            try:
                os.environ["ALETHEIA_LEAN_BIN"] = str(_make_fake("badver.sh", "echo 'Lean (version 4.1.0, x)'"))
                with self.assertRaises(ver.HoloSphereLeanError) as cm_v:
                    ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(cm_v.exception.code, "LEAN_VERSION_MISMATCH")

                os.environ["ALETHEIA_LEAN_BIN"] = str(_make_fake("badverrc.sh", "echo 'Lean (version 4.36.0, x)'; exit 1"))
                with self.assertRaises(ver.HoloSphereLeanError) as cm_vrc:
                    ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(cm_vrc.exception.code, "LEAN_VERSION_MISMATCH")

                orig_run = ver.subprocess.run
                try:
                    def _boom(*_a, **_kw):
                        raise OSError("exec_boom")
                    ver.subprocess.run = _boom
                    with self.assertRaises(ver.HoloSphereLeanError) as cm_ex:
                        ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                    self.assertEqual(cm_ex.exception.code, "LEAN_EXEC_ERROR")
                finally:
                    ver.subprocess.run = orig_run

                os.environ["ALETHEIA_LEAN_BIN"] = str(
                    _make_fake("badrc.sh", "if [ \"$1\" = --version ]; then echo 'Lean (version 4.36.0, x)'; exit 0; fi; exit 1")
                )
                with self.assertRaises(ver.HoloSphereLeanError) as cm_c:
                    ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(cm_c.exception.code, "LEAN_COMPILE_FAIL")

                os.environ["ALETHEIA_LEAN_BIN"] = str(
                    _make_fake("baderr.sh", "if [ \"$1\" = --version ]; then echo 'Lean (version 4.36.0, x)'; exit 0; fi; echo err >&2; exit 0")
                )
                with self.assertRaises(ver.HoloSphereLeanError) as cm_e:
                    ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(cm_e.exception.code, "LEAN_STDERR_NONEMPTY")

                os.environ["ALETHEIA_LEAN_BIN"] = str(
                    _make_fake("badwarn.sh", "if [ \"$1\" = --version ]; then echo 'Lean (version 4.36.0, x)'; exit 0; fi; echo 'warning: bad'; exit 0")
                )
                with self.assertRaises(ver.HoloSphereLeanError) as cm_w:
                    ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(cm_w.exception.code, "LEAN_WARNING_FORBIDDEN")
            finally:
                if old_bin is None:
                    os.environ.pop("ALETHEIA_LEAN_BIN", None)
                else:
                    os.environ["ALETHEIA_LEAN_BIN"] = old_bin

        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(LEAN_DIR / "verify_holosphere_lean.py"), "--verify-lean", "--json"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["theorems_verified"], 36)

    def test_14_live_init_olean_hijack_blocked(self) -> None:
        lean_bin = ver.resolve_lean_binary()
        lean_src = LEAN_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="holo_hijack_") as td:
            fake_dir = Path(td)
            (fake_dir / "Init.olean").write_bytes(b"CORRUPT_FAKE_INIT_OLEAN")
            poison_env = dict(os.environ)
            poison_env["LEAN_PATH"] = str(fake_dir)
            raw_proc = subprocess.run(
                [lean_bin, "--stdin"],
                input=lean_src,
                capture_output=True,
                text=True,
                env=poison_env,
                timeout=120,
                check=False,
            )
            self.assertNotEqual(raw_proc.returncode, 0)
            old_lp = os.environ.get("LEAN_PATH")
            try:
                os.environ["LEAN_PATH"] = str(fake_dir)
                clean_res = ver.verify_lean4_file_live(LEAN_PATH, check_sha=True)
                self.assertEqual(clean_res["theorems_verified"], 36)
            finally:
                if old_lp is None:
                    os.environ.pop("LEAN_PATH", None)
                else:
                    os.environ["LEAN_PATH"] = old_lp


if __name__ == "__main__":
    unittest.main()
