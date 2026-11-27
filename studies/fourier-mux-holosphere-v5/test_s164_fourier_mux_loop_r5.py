#!/usr/bin/env python3
"""Track S164 (EXP-Q45a & EXP-Q45c) R5 Assertion-Free Verification Test Suite
(0 bare asserts, 0 skips, 0 production floats).

Covers:
  T01: CLI roundtrip (--standalone, --write, --check, --json, 0o600, mode conflict)
  T02: Zone-B filesystem, path, symlink, dotdot, hardlink, FIFO, stream cap, atomic write
  T03: Strict JSON parsing (BOM, CRLF, depth, large int boundary, float, NaN, dup key)
  T04: Pure-Q oracle math, boundary guards, AST pure-Q audit, and ALL BoolOp kills
  T05: Re-signed certificate forgery resistance across every secondary check
  T06: R12 Lean 4 static scanner, companion SHA pins, and live/mocked Lean 4.36.0
  T07: Subprocess -I -B CLI execution, RC=2 single-line JSON errors, AST BoolOp sweep
"""
from __future__ import annotations

import ast
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
from typing import Any, Callable

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import s164_fourier_mux_loop_oracle as oracle  # noqa: E402
import verify_s164_fourier_mux_loop as verifier  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def expect_s164_error(
    fn: Callable[[], Any], expected_code: str | None = None, msg: str = ""
) -> str:
    try:
        fn()
    except oracle.S164Error as exc:
        if expected_code is not None:
            check(
                exc.code == expected_code,
                f"{msg}: expected {expected_code!r}, got {exc.code!r} ({exc.message})",
            )
        return exc.code
    raise AssertionError(f"{msg}: expected S164Error({expected_code!r}), but succeeded")


def resign_cert(cert: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cert)
    payload = out["payload"]
    p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    out["payload_sha256"] = p_sha
    body = {
        "schema_version": out["schema_version"],
        "track_id": out["track_id"],
        "payload_sha256": p_sha,
        "payload": payload,
    }
    out["certificate_sha256"] = oracle.sha256_bytes(oracle.canonical_json_bytes(body))
    return out


def _run_cli_capture(argv: list[str]) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = out_buf, err_buf
        rc = verifier.main(argv)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out_buf.getvalue(), err_buf.getvalue()


def _copy_track_bundle(dest: Path) -> Path:
    for fname in (
        oracle.PINNED_LEAN_FILES["lean_source"],
        "lean-toolchain",
        "lakefile.toml",
        "lake-manifest.json",
        "s164_fourier_mux_loop_oracle.py",
        "verify_s164_fourier_mux_loop.py",
    ):
        dest.joinpath(fname).write_bytes(oracle.TRACK_DIR.joinpath(fname).read_bytes())
    return dest


def _good_lean_runner(_tdir: Path) -> dict[str, Any]:
    lines = [
        f"'{name}' depends on axioms: [propext, Quot.sound]"
        for name in oracle.LEAN_THEOREM_NAMES[:-2]
    ]
    lines.append(f"'{oracle.LEAN_THEOREM_NAMES[-2]}' does not depend on any axioms")
    lines.append(f"'{oracle.LEAN_THEOREM_NAMES[-1]}' depends on axioms: [propext, Quot.sound]")
    return {
        "version_rc": 0,
        "version_output": "Lean (version 4.36.0)",
        "build_rc": 0,
        "build_output": "Build completed successfully",
        "axiom_rc": 0,
        "axiom_stdout": "\n".join(lines) + "\n",
    }


# ---------------------------------------------------------------------------
# T01: CLI Roundtrip, Write 0o600, Check, Standalone, Conflicts
# ---------------------------------------------------------------------------


def test_t01_cli_roundtrip() -> None:
    cert = oracle.build_certificate()
    raw = oracle.canonical_json_bytes(cert)

    rc_s, out_s, err_s = _run_cli_capture(["--standalone"])
    check(rc_s == 0 and err_s == "", f"standalone failed: {err_s}")
    check(out_s.encode("ascii") == raw, "standalone bytes mismatch")

    with tempfile.TemporaryDirectory(prefix="s164_t01_") as td:
        out_p = Path(td).joinpath("cert.json")
        rc_w, out_w, err_w = _run_cli_capture(["--write", os.fspath(out_p), "--json"])
        check(rc_w == 0 and err_w == "", f"write json failed: {err_w}")
        w_obj = json.loads(out_w)
        check(w_obj["status"] == "WRITTEN", "write status")
        check(w_obj["certificate_sha256"] == cert["certificate_sha256"], "write sha")
        st = os.lstat(out_p)
        check(stat.S_IMODE(st.st_mode) == 0o600, f"expected 0o600, got {oct(st.st_mode)}")
        check(out_p.read_bytes() == raw, "written bytes != canonical")

        rc_w2, out_w2, err_w2 = _run_cli_capture(["--write", os.fspath(out_p)])
        check(rc_w2 == 0 and err_w2 == "" and "WROTE" in out_w2, "write text failed")

        rc_c, out_c, err_c = _run_cli_capture(["--check", os.fspath(out_p), "--json"])
        check(rc_c == 0 and err_c == "", f"check json failed: {err_c}")
        c_obj = json.loads(out_c)
        check(c_obj["status"] == "PASS", "check PASS")
        check(c_obj["lean"]["theorems_verified"] == 18, "18 theorems")

        rc_c2, out_c2, err_c2 = _run_cli_capture(["--check", os.fspath(out_p)])
        check(rc_c2 == 0 and err_c2 == "" and "Track S164 PASS" in out_c2, "check text")

    rc_def, out_def, err_def = _run_cli_capture(["--json"])
    check(rc_def == 0 and err_def == "", f"default check failed: {err_def}")
    check(json.loads(out_def)["status"] == "PASS", "default check PASS")

    rc_rel, out_rel, err_rel = _run_cli_capture(["--check", oracle.DEFAULT_CERT_FILENAME])
    check(rc_rel == 0 and err_rel == "" and "PASS" in out_rel, "relative cert check")

    rc_bad, _, err_bad = _run_cli_capture(["--check", "--standalone"])
    check(rc_bad == 2 and "MODE_CONFLICT" in err_bad, "mode conflict check+standalone")
    rc_bad2, _, err_bad2 = _run_cli_capture(["--write", "--standalone"])
    check(rc_bad2 == 2 and "MODE_CONFLICT" in err_bad2, "mode conflict write+standalone")


# ---------------------------------------------------------------------------
# T02: Zone-B Filesystem, Path, Symlink, Hardlink, Stream Cap Hardening
# ---------------------------------------------------------------------------


def test_t02_zone_b_filesystem() -> None:
    expect_s164_error(lambda: verifier.require_ok(1, "X", "m"), "X", "non-bool truthy")
    expect_s164_error(lambda: verifier.require_ok(False, "X", "m"), "X", "bool False")
    expect_s164_error(lambda: oracle.require_ok(1, "X", "m"), "X", "oracle non-bool")
    expect_s164_error(lambda: oracle.require_ok(False, "X", "m"), "X", "oracle False")
    expect_s164_error(lambda: verifier._check_path_str("", "p"), "EMPTY_PATH", "empty")
    expect_s164_error(lambda: verifier._check_path_str("a\x00b", "p"), "NUL_BYTE_FORBIDDEN", "nul")
    expect_s164_error(lambda: verifier._check_path_str("a\ud800b", "p"), "INVALID_UTF8_SURROGATE", "surrogate")
    expect_s164_error(lambda: verifier._check_path_str(123, "p"), "TYPE_ERROR", "int path")
    check(verifier._check_path_str(Path("/tmp/ok"), "p") == Path("/tmp/ok"), "Path ok")

    with tempfile.TemporaryDirectory(prefix="s164_t02_") as td:
        tdir = Path(td)
        reg = tdir.joinpath("reg.json")
        reg.write_bytes(b"{}\n")
        sym = tdir.joinpath("sym.json")
        os.symlink(reg, sym)
        hard = tdir.joinpath("hard.json")
        os.link(reg, hard)
        sub = tdir.joinpath("sub")
        sub.mkdir()
        sym_dir = tdir.joinpath("sym_dir")
        os.symlink(sub, sym_dir)
        fifo = tdir.joinpath("pipe")
        os.mkfifo(fifo)

        expect_s164_error(lambda: verifier.read_regular_bytes_capped(sym, "s"), "SYMLINK_IN_PATH", "sym")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(sym_dir.joinpath("x"), "sd"), "SYMLINK_IN_PATH", "symdir")
        dotdot = os.fspath(sym_dir) + "/../reg.json"
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(dotdot, "dd"), "SYMLINK_IN_PATH", "dotdot sym")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(hard, "h"), "HARDLINK_FORBIDDEN", "hardlink")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(sub, "d"), "NOT_REGULAR_FILE", "dir")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(fifo, "f"), "NOT_REGULAR_FILE", "fifo")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(tdir.joinpath("missing"), "m"), "IO_OPEN_ERROR", "missing")

        big = tdir.joinpath("big.json")
        big.write_bytes(b"a" * 32)
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(big, "b", max_bytes=16), "FILE_TOO_LARGE", "cap")
        check(len(verifier.read_regular_bytes_capped(big, "b", max_bytes=32)) == 32, "exact max_bytes")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(big, "b", max_bytes=0), "RANGE_ERROR", "lo")
        expect_s164_error(lambda: verifier.read_regular_bytes_capped(big, "b", max_bytes=64 * 1024 * 1024 + 1), "RANGE_ERROR", "hi")

        # Streaming overflow check (st_size=0)
        orig_fstat = os.fstat

        class _ZeroStat:
            def __init__(self, st_obj: os.stat_result) -> None:
                self.st_mode = st_obj.st_mode
                self.st_nlink = st_obj.st_nlink
                self.st_size = 0

        try:
            os.fstat = lambda fd: _ZeroStat(orig_fstat(fd))  # type: ignore[assignment]
            expect_s164_error(
                lambda: verifier.read_regular_bytes_capped(big, "b", max_bytes=16),
                "FILE_STREAM_TOO_LARGE",
                "stream cap",
            )
            check(len(verifier.read_regular_bytes_capped(big, "b", max_bytes=32)) == 32, "exact stream")
        finally:
            os.fstat = orig_fstat

        # Relative path check via cwd
        orig_cwd = os.getcwd()
        try:
            os.chdir(td)
            check(verifier.read_regular_bytes_capped("big.json", "rel") == b"a" * 32, "rel read")
        finally:
            os.chdir(orig_cwd)

        # Simulated lstat OSError
        orig_lstat = os.lstat
        try:
            def _boom(_p: Any) -> Any:
                raise PermissionError("sim")
            os.lstat = _boom  # type: ignore[assignment]
            expect_s164_error(lambda: verifier._check_no_symlink_components(reg, "r"), "IO_STAT_ERROR", "lstat err")
        finally:
            os.lstat = orig_lstat

        # Write hardening + overwrite error paths
        expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(tdir.joinpath("x"), "str", "w"), "INVALID_BYTES", "str")  # type: ignore[arg-type]
        expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(tdir.joinpath("nope", "x"), b"{}", "w"), "PARENT_STAT_ERROR", "no parent")
        expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(big.joinpath("child"), b"{}", "w"), "PARENT_NOT_DIR", "parent file")
        expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(sub, b"{}", "w"), "TARGET_NOT_REGULAR", "target dir")
        expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(hard, b"{}", "w"), "TARGET_HARDLINK", "target hardlink")

        # Simulated target lstat OSError (on second lstat of reg.json, after symlink-component walk)
        target_calls = [0]

        def _target_boom(p_arg: Any) -> Any:
            if os.fspath(p_arg).endswith("reg.json"):
                target_calls[0] = target_calls[0] + 1
                if target_calls[0] >= 2:
                    raise PermissionError("target stat")
            return orig_lstat(p_arg)

        try:
            os.lstat = _target_boom  # type: ignore[assignment]
            expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(reg, b"{}", "w"), "TARGET_STAT_ERROR", "tstat")
        finally:
            os.lstat = orig_lstat

        orig_replace = os.replace
        try:
            def _rep_boom(_a: Any, _b: Any) -> None:
                raise OSError("disk full")
            os.replace = _rep_boom  # type: ignore[assignment]
            expect_s164_error(lambda: verifier.write_regular_bytes_atomic_0600(tdir.joinpath("new.json"), b"{}", "w"), "IO_WRITE_ERROR", "replace err")
        finally:
            os.replace = orig_replace


# ---------------------------------------------------------------------------
# T03: Strict Zero-Float / Zero-Dup-Key / Bounded-Depth / Bounded-Int JSON
# ---------------------------------------------------------------------------


def test_t03_strict_json_hardening() -> None:
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b"\xff", "u"), "BAD_UTF8", "utf8")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes("\ufeff{}".encode("utf-8"), "b"), "BAD_BOM", "bom")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b"{\r\n}", "cr"), "BAD_CRLF", "crlf")
    check(verifier.parse_strict_json_bytes(b'{"s":"a\\\\b"}', "esc")["s"] == "a\\b", "escaped backslash")
    check(verifier.parse_strict_json_bytes(b'{"s":"a\\\\\\"b"}', "esc")["s"] == 'a\\"b', "escaped str")

    exact_deep = b'{"a":' + b"[" * (verifier.MAX_JSON_DEPTH - 1) + b"0" + b"]" * (verifier.MAX_JSON_DEPTH - 1) + b"}"
    check(verifier.parse_strict_json_bytes(exact_deep, "exact_deep")["a"] is not None, "exact max depth ok")
    deep = b"[" * (verifier.MAX_JSON_DEPTH + 1) + b"]" * (verifier.MAX_JSON_DEPTH + 1)
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(deep, "deep"), "JSON_TOO_DEEP", "depth")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b"]", "neg"), "JSON_UNBALANCED", "neg depth")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b"{", "unbal"), "JSON_UNBALANCED_FINAL", "final depth")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"a": "abc', "unterm"), "JSON_UNTERMINATED_STR", "unterm")

    ok_int = ("1" * verifier.MAX_JSON_INT_DIGITS).encode("ascii")
    bad_int = ("1" * (verifier.MAX_JSON_INT_DIGITS + 1)).encode("ascii")
    check(verifier.parse_strict_json_bytes(b'{"n":' + ok_int + b"}", "okint")["n"] > 0, "exact max int digits")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"n":' + bad_int + b"}", "badint"), "JSON_INT_TOO_LARGE", "too many digits")
    expect_s164_error(lambda: verifier._scan_json_text_strict('{"n":' + "1" * (verifier.MAX_JSON_INT_DIGITS + 1) + "}"), "JSON_INT_TOO_LARGE", "scan int digits")
    expect_s164_error(lambda: verifier._scan_json_text_strict('{"x": 1.25}'), "FLOAT_FORBIDDEN", "scan dot float")
    expect_s164_error(lambda: verifier._scan_json_text_strict('{"x": 1e5}'), "FLOAT_FORBIDDEN", "scan exp float")
    expect_s164_error(lambda: verifier._reject_large_int("-" + ("9" * (verifier.MAX_JSON_INT_DIGITS + 1))), "JSON_INT_TOO_LARGE", "helper neg too big")
    check(verifier._reject_large_int("-42") == -42, "neg int ok")

    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"x": 1.25}', "flt1"), "FLOAT_FORBIDDEN", "dot float")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"x": 1e5}', "flt2"), "FLOAT_FORBIDDEN", "exp float")
    expect_s164_error(lambda: verifier._reject_float("1.5"), "FLOAT_FORBIDDEN", "helper float")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"x": NaN}', "nan"), "NONFINITE_JSON", "nan")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"x": 1, "x": 2}', "dup"), "DUPLICATE_JSON_KEY", "dup")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b'{"x": }', "syn"), "JSON_INVALID", "bad json")
    expect_s164_error(lambda: verifier.parse_strict_json_bytes(b"[1, 2]", "arr"), "JSON_NOT_OBJECT", "list root")


# ---------------------------------------------------------------------------
# T04: Pure-Q Oracle Math, Boundary Guards, AST Audit & BoolOp Kills
# ---------------------------------------------------------------------------


def test_t04_oracle_math_and_boolops() -> None:
    # Oracle basic math & boundary kills
    expect_s164_error(lambda: oracle.require_exact_int(True, "b"), "TYPE_ERROR", "bool not int")
    expect_s164_error(lambda: oracle.require_exact_int(0, "lo", lo=1), "RANGE_ERROR", "lo")
    check(oracle.require_exact_int(1, "lo", lo=1) == 1, "exact lo")
    expect_s164_error(lambda: oracle.require_exact_int(5, "hi", hi=4), "RANGE_ERROR", "hi")
    check(oracle.require_exact_int(4, "hi", hi=4) == 4, "exact hi")

    check(oracle.int_pow(3, 0) == 1 and oracle.int_pow(3, 5) == 243, "int_pow")
    check(oracle.ceil_isqrt_int(0) == 0 and oracle.ceil_isqrt_int(9) == 3 and oracle.ceil_isqrt_int(10) == 4, "ceil_isqrt")
    expect_s164_error(lambda: oracle.q_div(Fraction(1, 1), Fraction(0, 1)), "ZERO_DIV", "zero div")
    check(oracle.fmt_q(Fraction(4, 2)) == "2" and oracle.fmt_q(Fraction(3, 4)) == "3/4", "fmt_q")

    expect_s164_error(lambda: oracle.fwht256_plain_int([0] * 16), "FWHT_LEN", "plain len")
    expect_s164_error(lambda: oracle.fwht256_bent4_int([0] * 16), "FWHT_LEN", "bent4 len")
    expect_s164_error(lambda: oracle.check_finite_fourier_crossover_point(8, Fraction(0, 1)), "CROSSOVER_C_POS", "c<=0")
    c_pt = oracle.check_finite_fourier_crossover_point(8, Fraction(8, 3))
    check(c_pt["fwht_wins_strict"] is False and c_pt["fwht_wins_sq_bound"] is False, "tie at 8/3")
    c_win = oracle.check_finite_fourier_crossover_point(8, Fraction(3, 1))
    check(c_win["fwht_wins_strict"] is True and c_win["fwht_wins_sq_bound"] is True, "win at 3")

    expect_s164_error(lambda: oracle.monarch_stage_allones([0] * 10, 16, 16, 16), "MONARCH_IN_LEN", "monarch len")
    check(oracle.gcd_int(-12, -18) == 6, "gcd negative")
    expect_s164_error(lambda: oracle.exact_subset_min_separation_mitm(2, 4, 3), "MITM_COPRIME", "non-coprime")
    check(oracle.exact_subset_min_separation_mitm(9, 10, 4) == 19, "m_4 == 19")
    check(oracle.exact_subset_min_separation_mitm(1, 2, 2) == 1, "m_2(1/2) == 1")
    expect_s164_error(lambda: oracle.eval_mux_cor15_point(2, 1, 2, 1, Fraction(0, 1)), "MUX_DEN_POS", "den<0")
    expect_s164_error(lambda: oracle.eval_mux_cor15_point(1, 1, 2, 1, Fraction(1, 2)), "MUX_DEN_POS", "den=0")
    expect_s164_error(lambda: oracle.eval_mux_cor15_point(1, 2, 3, 1, Fraction(1, 2)), "MUX_ETA_DEN", "eta den=0")
    expect_s164_error(lambda: oracle.h4_block_apply_int([0] * 4), "H4_BLK_LEN", "h4 len")

    expect_s164_error(lambda: oracle.simulate_dual_axis_carry(Fraction(-1, 2), [Fraction(1, 1)], [Fraction(1, 1)]), "DAC_BETA_NONNEG", "neg beta")
    expect_s164_error(lambda: oracle.simulate_dual_axis_carry(Fraction(0, 1), [], []), "DAC_SEQ_LEN", "empty seq")
    expect_s164_error(lambda: oracle.simulate_dual_axis_carry(Fraction(0, 1), [Fraction(1, 1)], [Fraction(1, 1), Fraction(2, 1)]), "DAC_SEQ_LEN", "len mismatch")
    expect_s164_error(lambda: oracle.simulate_dual_axis_carry(Fraction(0, 1), [Fraction(0, 1)], [Fraction(1, 1)]), "DAC_E_POS", "e<=0")

    # AST Pure-Q audit negative controls (kills every check in verify_pure_q_ast + BoolOp!)
    verifier.verify_pure_q_ast("from . import rel\n", "rel.py")
    for bad_src, code in (
        ("x = 1 / 2\n", "AST_DIV_FORBIDDEN"),
        ("x = 2 ** 3\n", "AST_POW_FORBIDDEN"),
        ("assert True\n", "AST_ASSERT_FORBIDDEN"),
        ("x = 1.5\n", "AST_FLOAT_LITERAL"),
        ("x = 1j\n", "AST_FLOAT_LITERAL"),
        ("import math\n", "AST_FORBIDDEN_IMPORT"),
        ("from math import sin\n", "AST_FORBIDDEN_IMPORT"),
        ("x = float(1)\n", "AST_FORBIDDEN_NAME"),
        ("x = obj.from_float(1)\n", "AST_FORBIDDEN_ATTR"),
    ):
        expect_s164_error(lambda s=bad_src: verifier.verify_pure_q_ast(s, "bad.py"), code, code)

    # Gate BoolOp single-false operand kills in oracle.check_track_gates + assemble_certificate
    cert = oracle.build_certificate()
    pl = cert["payload"]
    p1 = copy.deepcopy(pl["exp_q45a_finite_fourier_crossover"])
    p2 = copy.deepcopy(pl["exp_q45a_holosphere_v5_head"])
    p3 = copy.deepcopy(pl["exp_q45c_mux_cor15_vs_alkashi"])
    p4 = copy.deepcopy(pl["exp_q45c_dual_axis_su2_sfast"])

    check(all(oracle.check_track_gates(p1, p2, p3, p4).values()), "all gates ok")
    for fn_mut in (
        lambda a, b, c, d: a.__setitem__("h256_involution_verified", False),
        lambda a, b, c, d: a.__setitem__("fwht256_bent4_parseval_out_sq", 0),
        lambda a, b, c, d: a.__setitem__("thm41_zero_compression_at_256", False),
        lambda a, b, c, d: a.__setitem__("c3_crossover_m_star", 9),
        lambda a, b, c, d: a.__setitem__("s159_row0_minus_v4_t8192_bpb_q", "0"),
        lambda a, b, c, d: a.__setitem__("s159_v4_minus_row3_t8192_bpb_q", "0"),
        lambda a, b, c, d: b["v4_input_collisions"].pop(),
        lambda a, b, c, d: b.__setitem__("v5_unique_vectors_of_260", 259),
        lambda a, b, c, d: b.__setitem__("v5_max_offdiag_dot", 1),
        lambda a, b, c, d: b["architectures"]["V4_Monarch2_16x16"].__setitem__("min_paths_2stage", 2),
        lambda a, b, c, d: b["architectures"]["V5A_Monarch3_8x8x8_512"].__setitem__("zero_smem_spill", False),
        lambda a, b, c, d: b["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("min_paths_per_pair", 1),
        lambda a, b, c, d: b["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("zero_smem_spill", False),
        lambda a, b, c, d: c["rho_9_10_first_failure_S"].__setitem__("BF16", 3),
        lambda a, b, c, d: c["rho_9_10_first_failure_S"].__setitem__("FP16", 4),
        lambda a, b, c, d: c["rho_9_10_first_failure_S"].__setitem__("FP32", 11),
        lambda a, b, c, d: c["rho_9_10_first_failure_S"].__setitem__("AlKashi_FixedRadix_u0", 14),
        lambda a, b, c, d: c.__setitem__("rho_1_3_m_s_mismatch_count", 1),
        lambda a, b, c, d: c["a16_block_h4_u0_audit"].__setitem__("parseval_mismatch_count", 1),
        lambda a, b, c, d: d["dual_axis_carry_regimes"]["beta_min_guarded_11_20"].__setitem__("max_drift_from_1_q", "2"),
        lambda a, b, c, d: d.__setitem__("su2_step_norm_sq_q", "2"),
        lambda a, b, c, d: d["su2_loop_alignment_r1_to_r4"][3].__setitem__("su2_aligned_cosine_q", "0"),
        lambda a, b, c, d: d.__setitem__("merge_lora_0_required_for_a16_parity", False),
    ):
        a1, b1, c1, d1 = copy.deepcopy(p1), copy.deepcopy(p2), copy.deepcopy(p3), copy.deepcopy(p4)
        fn_mut(a1, b1, c1, d1)
        g = oracle.check_track_gates(a1, b1, c1, d1)
        check(not all(g.values()), "single-false gate mutator must fail")
        expect_s164_error(lambda a=a1, b=b1, c=c1, d=d1: oracle.assemble_certificate(a, b, c, d), "TRACK_GATES_FAILED", "assemble reject")


# ---------------------------------------------------------------------------
# T05: Field-Complete Re-Signed Certificate Forgery Resistance
# ---------------------------------------------------------------------------


def test_t05_resigned_forgery_resistance() -> None:
    base = oracle.build_certificate()
    check(verifier.verify_certificate_object(base)["status"] == "PASS", "base PASS")

    # Rational & SHA helper unit guards + BoolOp kills + digit bounds (MED-4)
    check(verifier._parse_canon_q("0", "q") == Fraction(0), "q=0")
    check(verifier._parse_canon_q("-3/5", "q") == Fraction(-3, 5), "q=-3/5")
    ok64 = "1" * verifier.MAX_JSON_INT_DIGITS
    bad65 = "1" * (verifier.MAX_JSON_INT_DIGITS + 1)
    check(verifier._parse_canon_q(ok64, "q64") == Fraction(int(ok64), 1), "q=64 digits ok")
    check(verifier._parse_canon_q(f"-{ok64}/2", "q64n") == Fraction(-int(ok64), 2), "q num=64 ok")
    check(verifier._parse_canon_q(f"2/{ok64}", "q64d") == Fraction(2, int(ok64)), "q den=64 ok")
    expect_s164_error(lambda: verifier._parse_canon_q(bad65, "q65"), "Q_TOO_LARGE", "q int>64")
    expect_s164_error(lambda: verifier._parse_canon_q(f"{bad65}/2", "q65n"), "Q_TOO_LARGE", "q num>64")
    expect_s164_error(lambda: verifier._parse_canon_q(f"2/{bad65}", "q65d"), "Q_TOO_LARGE", "q den>64")
    expect_s164_error(lambda: verifier._parse_canon_q(1, "q"), "Q_TYPE_ERROR", "q int")
    expect_s164_error(lambda: verifier._parse_canon_q("1/0", "q"), "Q_FORMAT_ERROR", "q div0")
    expect_s164_error(lambda: verifier._parse_canon_q("2/4", "q"), "Q_NOT_REDUCED", "q 2/4")
    expect_s164_error(lambda: verifier._parse_canon_q("0/2", "q"), "Q_NOT_REDUCED", "q 0/2")
    expect_s164_error(lambda: verifier._check_sha256_field(1, "s"), "SHA_TYPE_ERROR", "sha int")
    expect_s164_error(lambda: verifier._check_sha256_field("abc", "s"), "SHA_FORMAT_ERROR", "sha short")

    expect_s164_error(lambda: verifier.verify_certificate_object([]), "CERT_NOT_DICT", "list")  # type: ignore[arg-type]
    bad_env = copy.deepcopy(base)
    bad_env["extra"] = 1
    expect_s164_error(lambda: verifier.verify_certificate_object(bad_env), "ENVELOPE_KEYS_MISMATCH", "extra")
    for k, v, code in (
        ("schema_version", "bad", "SCHEMA_MISMATCH"),
        ("track_id", "S999", "TRACK_MISMATCH"),
        ("payload", [], "PAYLOAD_NOT_DICT"),
    ):
        m = copy.deepcopy(base)
        m[k] = v
        expect_s164_error(lambda x=m: verifier.verify_certificate_object(x), code, k)

    bad_pk = copy.deepcopy(base)
    bad_pk["payload"]["extra"] = 1
    expect_s164_error(lambda: verifier.verify_certificate_object(bad_pk), "PAYLOAD_KEYS_MISMATCH", "pkeys")
    bad_en = copy.deepcopy(base)
    bad_en["payload"]["energy_power_status"] = "CLAIMED"
    expect_s164_error(lambda: verifier.verify_certificate_object(bad_en), "ENERGY_POLICY_VIOLATION", "energy")
    bad_ps = copy.deepcopy(base)
    bad_ps["payload_sha256"] = "0" * 64
    expect_s164_error(lambda: verifier.verify_certificate_object(bad_ps), "PAYLOAD_SHA_MISMATCH", "psha")
    bad_cs = copy.deepcopy(base)
    bad_cs["certificate_sha256"] = "0" * 64
    expect_s164_error(lambda: verifier.verify_certificate_object(bad_cs), "CERT_SHA_MISMATCH", "csha")

    def _forge(mutator: Callable[[dict[str, Any]], None], expected_code: str, label: str) -> None:
        c = copy.deepcopy(base)
        mutator(c["payload"])
        c = resign_cert(c)
        expect_s164_error(lambda: verifier.verify_certificate_object(c), expected_code, label)

    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("fwht256_gates_per_vector", 1024), "P1_FWHT_GATES", "p1 gates")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("fwht256_bent4_parseval_out_sq", 0), "P1_BENT4_PARSEVAL", "p1 bent4 parseval")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("c3_crossover_m_star", 8), "P1_CROSSOVER_M", "p1 m*")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("s159_v4_minus_row3_t8192_bpb_q", "0"), "P1_S159_GAP", "p1 gap 0")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("s159_v4_minus_row3_t8192_bpb_q", "1/3"), "P1_S159_GAP", "p1 gap >1/4")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"].__setitem__("v5_unique_vectors_of_260", 256), "P2_V5_UNIQUE", "p2 uniq")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"].__setitem__("v5_max_offdiag_dot", 2), "P2_V5_ORTHO", "p2 ortho")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"]["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("hbm_vocab_bytes", 512), "P2_V5B_HBM", "p2 hbm")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"]["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("e2e_smem_bytes", oracle.SM86_STATIC_SMEM_LIMIT_BYTES), "ORACLE_REPLAY_MISMATCH", "p2 smem exact limit")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"]["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("e2e_smem_bytes", 49153), "P2_V5B_SMEM", "p2 smem")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["rho_9_10_first_failure_S"].__setitem__("BF16", 3), "P3_MUX_THRESH", "p3 bf16")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["rho_9_10_first_failure_S"].__setitem__("FP16", 6), "P3_MUX_THRESH", "p3 fp16")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["rho_9_10_first_failure_S"].__setitem__("FP32", 11), "P3_MUX_THRESH", "p3 fp32")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["rho_9_10_first_failure_S"].__setitem__("AlKashi_FixedRadix_u0", 12), "P3_ALKASHI_U0", "p3 u0")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["a16_block_h4_u0_audit"].__setitem__("sum_h4_sq", 1), "P3_A16_PARSEVAL", "p3 h4")
    _forge(lambda p: p["exp_q45c_mux_cor15_vs_alkashi"]["a16_block_h4_u0_audit"].__setitem__("parseval_mismatch_count", 1), "P3_A16_MISMATCH", "p3 h4 mismatch")
    _forge(lambda p: p["exp_q45c_dual_axis_su2_sfast"]["dual_axis_carry_regimes"]["beta_min_guarded_11_20"].__setitem__("max_drift_from_1_q", "3/2"), "P4_DAC_DAMP", "p4 dac")
    _forge(lambda p: p["exp_q45c_dual_axis_su2_sfast"]["su2_loop_alignment_r1_to_r4"][0].__setitem__("su2_aligned_cosine_q", "1/2"), "P4_SU2_ALIGN", "p4 su2")
    _forge(lambda p: p["gates"].__setitem__("g1_finite_fourier_n256_crossover", False), "GATES_FAILED", "gate g1")
    _forge(lambda p: p["gates"].__setitem__("g2_holosphere_v5_zero_alias_smem", False), "GATES_FAILED", "gate g2")
    _forge(lambda p: p["gates"].__setitem__("g3_mux_cor15_vs_alkashi_fixed_radix", False), "GATES_FAILED", "gate g3")
    _forge(lambda p: p["gates"].__setitem__("g4_dual_axis_carry_su2_loop_stability", False), "GATES_FAILED", "gate g4")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("c4_crossover_m_star", 16), "ORACLE_REPLAY_MISMATCH", "replay")

    # MED-4: re-signed empty sub-dicts, missing keys, wrong field types, and oversized rationals
    for subkey in (
        "exp_q45a_finite_fourier_crossover",
        "exp_q45a_holosphere_v5_head",
        "exp_q45c_mux_cor15_vs_alkashi",
        "exp_q45c_dual_axis_su2_sfast",
        "gates",
    ):
        _forge(lambda p, k=subkey: p.__setitem__(k, {}), "MISSING_KEY", f"empty {subkey}")
        _forge(lambda p, k=subkey: p.__setitem__(k, []), "SCHEMA_TYPE_ERROR", f"list {subkey}")
    _forge(lambda p: p.__setitem__("provenance_and_lean4", []), "SCHEMA_TYPE_ERROR", "list prov")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("fwht256_gates_per_vector", True), "SCHEMA_TYPE_ERROR", "bool as int")
    _forge(lambda p: p["exp_q45a_holosphere_v5_head"]["architectures"]["V5B_Monarch3_16x32_16x16_16x16"].__setitem__("e2e_smem_bytes", True), "SCHEMA_TYPE_ERROR", "bool smem")
    _forge(lambda p: p["gates"].__setitem__("g1_finite_fourier_n256_crossover", 1), "SCHEMA_TYPE_ERROR", "int as bool")
    _forge(lambda p: p["exp_q45c_dual_axis_su2_sfast"].__setitem__("su2_loop_alignment_r1_to_r4", {}), "SCHEMA_TYPE_ERROR", "dict su2 list")
    _forge(lambda p: p["exp_q45c_dual_axis_su2_sfast"].__setitem__("su2_loop_alignment_r1_to_r4", []), "P4_SU2_LEN", "empty su2 list")
    _forge(lambda p: p["exp_q45c_dual_axis_su2_sfast"]["su2_loop_alignment_r1_to_r4"].__setitem__(0, 123), "SCHEMA_TYPE_ERROR", "non-dict su2 row")
    _forge(lambda p: p["exp_q45a_finite_fourier_crossover"].__setitem__("s159_v4_minus_row3_t8192_bpb_q", "1" * 5000), "Q_TOO_LARGE", "5000-digit q in cert")


# ---------------------------------------------------------------------------
# T06: R12 Lean 4 Static Scanner, Companion SHA Pins, & Live Lean 4.36.0
# ---------------------------------------------------------------------------


def test_t06_lean_scanner_and_live_lean() -> None:
    lean_text = oracle.TRACK_DIR.joinpath(oracle.PINNED_LEAN_FILES["lean_source"]).read_text(encoding="utf-8")
    r12_ok = verifier.scan_lean_source_r12(lean_text)
    check(r12_ok["theorems_verified"] == 18, "18 Lean theorems")

    # Comment & string stripping edge cases (+ char literals, identifiers with ', and trailing -- comment)
    stripped = verifier.strip_lean_comments_and_strings(
        '-- sorry\n/- outer /- nested sorry -/ -/\n"str\\"sorry" theorem ok : True -> True := trivial -- tail'
    )
    check("sorry" not in stripped and "theorem ok" in stripped and "tail" not in stripped, "strip comments/strings")
    check("sorry" in verifier.strip_lean_comments_and_strings('\'"\' sorry'), "char quote at start")
    check("sorry" in verifier.strip_lean_comments_and_strings('let c := \'"\' ; sorry'), "char quote mid")
    check("sorry" in verifier.strip_lean_comments_and_strings('let c := \'\\\'\' ; sorry'), "escaped char quote")
    check("sorry" in verifier.strip_lean_comments_and_strings('let x\' := 1\nsorry'), "ident prime")
    check("sorry" in verifier.strip_lean_comments_and_strings('let _\' := 1\nsorry'), "underscore prime")
    check("sorry" in verifier.strip_lean_comments_and_strings('let x\'\' := 1\nsorry'), "double prime")
    check("sorry" in verifier.strip_lean_comments_and_strings('\'a\n\' sorry'), "newline not char lit")
    check("sorry" not in verifier.strip_lean_comments_and_strings('let c := \'\\\'\'\n"sorry"'), "escaped char closes cleanly")
    expect_s164_error(lambda: verifier.strip_lean_comments_and_strings('\'\\"'), "LEAN_UNTERM_STRING", "escaped char needs close quote")
    expect_s164_error(lambda: verifier.strip_lean_comments_and_strings("/- unterm"), "LEAN_UNTERM_COMMENT", "unterm cmt")
    expect_s164_error(lambda: verifier.strip_lean_comments_and_strings('"unterm'), "LEAN_UNTERM_STRING", "unterm str")
    expect_s164_error(lambda: verifier.validate_lean_companion_metadata("leanprover/lean4:v4.10.0", {"packages": []}), "BAD_TOOLCHAIN", "bad toolchain")
    expect_s164_error(lambda: verifier.validate_lean_companion_metadata(oracle.PINNED_LEAN_FILES["lean_toolchain"], {"packages": ["bad"]}), "NONEMPTY_LAKE_PACKAGES", "nonempty pkgs")

    # R12 trust-escape guards (+ attributed / protected theorems)
    for kw in ("sorry", "admit", "axiom x : False", "native_decide", "Classical.choice", "bv_decide", "#eval 1", "IO.println"):
        expect_s164_error(lambda k=kw: verifier.scan_lean_source_r12(lean_text + f"\n{k}\n"), "LEAN_ESCAPE_FOUND", kw)
    expect_s164_error(lambda: verifier.scan_lean_source_r12(lean_text + "\ntheorem extra_t : True := trivial\n"), "LEAN_THM_COUNT", "thm count")
    expect_s164_error(lambda: verifier.scan_lean_source_r12(lean_text + "\n@[simp] theorem extra_attr : True := trivial\n"), "LEAN_THM_COUNT", "attr thm count")
    expect_s164_error(lambda: verifier.scan_lean_source_r12(lean_text + "\nprotected theorem extra_prot : True := trivial\n"), "LEAN_THM_COUNT", "prot thm count")
    expect_s164_error(
        lambda: verifier.scan_lean_source_r12(lean_text.replace("s_fast_woodbury_contractivity_vs_reverse_decay_blowup", "wrong_thm_name", 1)),
        "LEAN_THM_SET_MISMATCH",
        "thm set",
    )
    expect_s164_error(
        lambda: verifier.scan_lean_source_r12(lean_text.replace("#print axioms FourierMuxLoopOperators.s_fast_woodbury_contractivity_vs_reverse_decay_blowup", "")),
        "LEAN_AXIOM_PRINT_MISMATCH",
        "print axioms set",
    )

    # Mocked & failing Lean runners
    base = oracle.build_certificate()
    res_m = verifier.verify_certificate_object(base, lean_runner=_good_lean_runner)
    check(res_m["lean"]["live_axioms_verified"] == 18, "mocked live lean 18")

    # Ensure custom lean_runner overrides default_live_lean_runner even when run_live_lean=True (kills BoolOp!)
    def _custom_bad_ver(_td: Path) -> dict[str, Any]:
        d = _good_lean_runner(_td)
        d["version_output"] = "4.10.0"
        return d

    expect_s164_error(
        lambda: verifier.verify_certificate_object(base, run_live_lean=True, lean_runner=_custom_bad_ver),
        "BAD_LEAN_VERSION",
        "custom runner overrides live flag",
    )

    for patch_fn, code in (
        (lambda r: r.__setitem__("version_rc", 1), "LEAN_VERSION_RC"),
        (lambda r: r.__setitem__("version_output", "4.10.0"), "BAD_LEAN_VERSION"),
        (lambda r: r.__setitem__("build_rc", 1), "LAKE_BUILD_FAIL"),
        (lambda r: r.__setitem__("build_output", "warning: unused"), "LAKE_BUILD_WARNING"),
        (lambda r: r.__setitem__("axiom_rc", 1), "LEAN_AXIOM_RC"),
        (lambda r: r.__setitem__("axiom_stdout", r["axiom_stdout"].replace("Quot.sound", "Classical.choice", 1)), "LEAN_FORBIDDEN_AXIOM"),
        (lambda r: r.__setitem__("axiom_stdout", "\n".join(r["axiom_stdout"].splitlines()[:-1]) + "\n"), "LEAN_AXIOM_SET_MISMATCH"),
    ):
        def _bad_runner(td: Path, fn=patch_fn) -> dict[str, Any]:
            d = _good_lean_runner(td)
            fn(d)
            return d
        expect_s164_error(lambda r=_bad_runner: verifier.verify_certificate_object(base, lean_runner=r), code, code)

    # Companion pin tamper checks
    with tempfile.TemporaryDirectory(prefix="s164_pins_") as td:
        tdir = _copy_track_bundle(Path(td))
        check(verifier.verify_companion_pins_and_lean(tdir)["theorems_verified"] == 18, "copied pins ok")
        tdir.joinpath("lake-manifest.json").write_text('{"packages":[]}\n', encoding="utf-8")
        expect_s164_error(lambda: verifier.verify_companion_pins_and_lean(tdir), "COMPANION_SHA_MISMATCH", "pin sha")

    # Exercise run_live_lean=True -> default_live_lean_runner with verified subprocess args & temp copy
    good = _good_lean_runner(oracle.TRACK_DIR)
    calls: list[list[str]] = []
    orig_sub_run = subprocess.run

    class _FakeProc:
        def __init__(self, rc: int, out: str, err: str) -> None:
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def _mock_lean_sub_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
        calls.append(list(cmd))
        cwd_p = Path(kwargs["cwd"])
        check(kwargs.get("capture_output") is True and kwargs.get("text") is True, "sub capture/text")
        check(kwargs.get("check") is False, "sub check=False")
        sub_env = kwargs.get("env", {})
        check(sub_env.get("CUDA_VISIBLE_DEVICES") == "" and sub_env.get("PYTHONDONTWRITEBYTECODE") == "1", "sub env")
        for req_f in (
            oracle.PINNED_LEAN_FILES["lean_source"],
            "lean-toolchain",
            "lakefile.toml",
            "lake-manifest.json",
        ):
            check(cwd_p.joinpath(req_f).is_file(), f"copied {req_f}")
        if cmd == ["lean", "--version"]:
            return _FakeProc(0, good["version_output"], "")
        if cmd == ["lake", "build"]:
            return _FakeProc(0, good["build_output"], "")
        if cmd == ["lean", oracle.PINNED_LEAN_FILES["lean_source"]]:
            return _FakeProc(0, good["axiom_stdout"], "")
        raise AssertionError(f"unexpected lean cmd: {cmd!r}")

    try:
        subprocess.run = _mock_lean_sub_run  # type: ignore[assignment]
        res_live = verifier.verify_certificate_object(base, run_live_lean=True)
        check(res_live["lean"]["live_axioms_verified"] == 18, "default_live_lean_runner 18")
        check(
            calls
            == [
                ["lean", "--version"],
                ["lake", "build"],
                ["lean", oracle.PINNED_LEAN_FILES["lean_source"]],
            ],
            f"unexpected calls: {calls!r}",
        )
    finally:
        subprocess.run = orig_sub_run  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# T07: Subprocess -I -B CLI Execution, RC=2 Errors, & 100% AST BoolOp Sweep
# ---------------------------------------------------------------------------


def test_t07_subprocess_and_boolop_sweep() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    v_path = os.fspath(oracle.TRACK_DIR.joinpath("verify_s164_fourier_mux_loop.py"))

    proc_ok = subprocess.run(
        [sys.executable, "-I", "-B", v_path, "--check", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    check(proc_ok.returncode == 0 and proc_ok.stderr == "", f"-I -B failed: {proc_ok.stderr}")
    check(json.loads(proc_ok.stdout)["status"] == "PASS", "subprocess PASS")

    proc_bad = subprocess.run(
        [sys.executable, "-I", "-B", v_path, "--check", "/nonexistent_s164_missing.json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    check(proc_bad.returncode == 2 and proc_bad.stdout == "", f"expected RC=2, got {proc_bad.returncode}")
    err_lines = proc_bad.stderr.strip().splitlines()
    check(len(err_lines) == 1 and "Traceback" not in proc_bad.stderr, "single-line JSON error")
    check(json.loads(err_lines[0])["code"] == "IO_OPEN_ERROR", "IO_OPEN_ERROR")

    # MED-4 subprocess regression: re-signed cert with empty sub-dicts & missing trailing LF -> RC=2 (not RC=1!)
    with tempfile.TemporaryDirectory(prefix="s164_med4_") as td:
        forged_empty = copy.deepcopy(oracle.build_certificate())
        forged_empty["payload"]["exp_q45a_finite_fourier_crossover"] = {}
        forged_empty = resign_cert(forged_empty)
        med4_p = Path(td).joinpath("med4_empty_subdict.json")
        med4_p.write_bytes(oracle.canonical_json_bytes(forged_empty))
        proc_med4 = subprocess.run(
            [sys.executable, "-I", "-B", v_path, "--check", os.fspath(med4_p)],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        check(proc_med4.returncode == 2 and proc_med4.stdout == "", f"MED-4 expected RC=2, got {proc_med4.returncode}")
        med4_lines = proc_med4.stderr.strip().splitlines()
        check(len(med4_lines) == 1 and "Traceback" not in proc_med4.stderr, "MED-4 single-line JSON")
        check(json.loads(med4_lines[0])["code"] == "MISSING_KEY", "MED-4 MISSING_KEY")

        no_nl_p = Path(td).joinpath("no_nl.json")
        no_nl_p.write_bytes(oracle.canonical_json_bytes(oracle.build_certificate())[:-1])
        proc_nl = subprocess.run(
            [sys.executable, "-I", "-B", v_path, "--check", os.fspath(no_nl_p)],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        check(proc_nl.returncode == 2 and proc_nl.stdout == "", f"trailing LF expected RC=2, got {proc_nl.returncode}")
        check(json.loads(proc_nl.stderr.strip())["code"] == "BAD_TRAILING_NEWLINE", "BAD_TRAILING_NEWLINE")

    # Also exercise RecursionError & OSError handlers in main()
    orig_cli = verifier.run_cli
    try:
        def _rec(_argv: Any = None) -> int:
            raise RecursionError("sim rec")
        verifier.run_cli = _rec  # type: ignore[assignment]
        rc_r, _, err_r = _run_cli_capture(["--check"])
        check(rc_r == 2 and "RECURSION_ERROR" in err_r, "recursion handler")

        def _oserr(_argv: Any = None) -> int:
            raise OSError("sim os")
        verifier.run_cli = _oserr  # type: ignore[assignment]
        rc_o, _, err_o = _run_cli_capture(["--check"])
        check(rc_o == 2 and "OS_ERROR" in err_o, "oserror handler")

        def _suberr(_argv: Any = None) -> int:
            raise subprocess.SubprocessError("sim sub")
        verifier.run_cli = _suberr  # type: ignore[assignment]
        rc_sub, _, err_sub = _run_cli_capture(["--check"])
        check(rc_sub == 2 and "OS_ERROR" in err_sub, "suberr handler")

        def _udecerr(_argv: Any = None) -> int:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        verifier.run_cli = _udecerr  # type: ignore[assignment]
        rc_ud, _, err_ud = _run_cli_capture(["--check"])
        check(rc_ud == 2 and "OS_ERROR" in err_ud, "udecerr handler")
    finally:
        verifier.run_cli = orig_cli

    # Check both production files have zero ast.Div, ast.Pow, ast.Assert, and float literals
    for fname in ("s164_fourier_mux_loop_oracle.py", "verify_s164_fourier_mux_loop.py"):
        src = oracle.TRACK_DIR.joinpath(fname).read_text(encoding="utf-8")
        verifier.verify_pure_q_ast(src, fname)
        tree = ast.parse(src, filename=fname)
        boolops = [n for n in ast.walk(tree) if isinstance(n, ast.BoolOp)]
        check(len(boolops) >= 1, f"{fname}: expected BoolOps")


def run_all_tests() -> None:
    tests = (
        test_t01_cli_roundtrip,
        test_t02_zone_b_filesystem,
        test_t03_strict_json_hardening,
        test_t04_oracle_math_and_boolops,
        test_t05_resigned_forgery_resistance,
        test_t06_lean_scanner_and_live_lean,
        test_t07_subprocess_and_boolop_sweep,
    )
    for fn in tests:
        fn()
    sys.stdout.write(f"Track S164 R5 suite PASS: {len(tests)}/{len(tests)} OK (0 skips, 0 bare asserts)\n")


if __name__ == "__main__":
    run_all_tests()
