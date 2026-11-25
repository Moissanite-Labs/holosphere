#!/usr/bin/env python3
"""Project HoloSphere Fail-Closed Verifier & R13b Lean 4.36.0 Gate.

Zero IEEE-754 floats, zero bare asserts, zero external dependencies.
Provides:
  - Hardened Zone-B file I/O (O_NOFOLLOW, symlink ancestor rejection, streaming size cap,
    80-digit int/rational DoS cap, atomic 0o600 write)
  - R13b O(1) in_ident Lean 4 lexer + 18-rule static pre-filter + allowlist scrubbed lean --stdin
  - Field-complete secondary auditor (SEC) across all 5 HoloSphere Pillars + FR0/FR1
  - CLI: --check (default on no-arg), --write, --verify-lean, --json
"""

from __future__ import annotations

import copy
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

LEAN_DIR = Path(__file__).resolve().parent
if str(LEAN_DIR) not in sys.path:
    sys.path.insert(0, str(LEAN_DIR))

import holosphere_lean_oracle as oracle

DEFAULT_CERT_PATH = LEAN_DIR / "holosphere_lean_certificate.json"
DEFAULT_LEAN_PATH = LEAN_DIR / oracle.LEAN_FILE_NAME

MAX_FILE_BYTES = 1_048_576
MAX_INT_DIGITS = 80
MAX_INT_BITS = 266
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_RAT_RE = re.compile(r"^-?[0-9]{1,80}(?:/[1-9][0-9]{0,79})?$")
_ENVELOPE_KEYS = frozenset({"schema_version", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_SAFE_LEAN_ENV_KEYS = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "TERM"})
_ALLOWED_LEAN_AXIOMS = frozenset({"propext", "Quot.sound"})

# O(1) Lean 4 identifier character classes for Char-literal vs prime disambiguation (R13b)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_LEAN4_ID_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_LEAN4_ID_REST = _LEAN4_ID_FIRST | frozenset("0123456789'!?")

# 18-rule static Lean 4 pre-filter (applied to comment- and Char-stripped source)
LEAN4_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("R01_LITERAL_OR_GUILLEMET", re.compile(r'["«»]')),
    ("R02_IMPORT", re.compile(r"(?m)^\s*import\b")),
    ("R03_SORRY_ADMIT", re.compile(r"\b(?:sorry|admit|sorryAx)\b")),
    ("R04_AXIOM_DECL", re.compile(r"(?m)^\s*(?:private\s+|protected\s+)?axiom\b")),
    ("R05_UNSAFE_EXTERN", re.compile(r"\bunsafe\b|@\[[^\]]*\b(?:extern|implemented_by|csimp)\b")),
    ("R06_OPAQUE_PARTIAL", re.compile(r"\b(?:opaque|partial|noncomputable)\b")),
    ("R07_NATIVE_DECIDE", re.compile(r"\bnative_decide\b|\bdecide\s*\+\s*(?:native|revert)\b")),
    ("R08_DBG_TRACE", re.compile(r"\b(?:dbg_trace|dbgTrace|dbgTraceVal|dbgTraceIfShared|dbgSleep|dbgStackTrace)\b")),
    ("R09_INIT_EXPORT_ATTR", re.compile(r"@\[[^\]]*\b(?:export|init|builtin_init)\b|\b(?:initialize|builtin_initialize)\b")),
    ("R10_GUARD_MSGS", re.compile(r"#guard_msgs\b")),
    ("R11_WARN_SORRY_OPT", re.compile(r"\b(?:warn\.sorry|terminalTacticsAsSorry|allowUnsafeReducibility)\b")),
    ("R12_META_EVAL_CMD", re.compile(r"#(?:eval|reduce|synth|check_failure)\b|\brun_cmd\b|\brun_meta\b")),
    ("R13_SET_OPTION", re.compile(r"(?m)^\s*set_option\b")),
    ("R14_ELAB_TACTIC_MACRO", re.compile(r"(?m)^\s*(?:elab|elab_rules|macro|macro_rules|syntax|declare_syntax_cat)\b")),
    ("R15_IO_SYSTEM", re.compile(r"\bIO\.|\bSystem\.|\bLean\.Environment\b|\bLean\.Elab\b")),
    ("R16_PANIC_UNREACHABLE", re.compile(r"\b(?:panic!|unreachable!|assert!)")),
    ("R17_DECIDE_CONFIG", re.compile(r"\bconfig\s*:=\s*\{[^}]*\bnative\b")),
    ("R18_ROOT_SORRYAX", re.compile(r"\b_root_\.sorryAx\b")),
)

_LEAN4_PRINT_LINE_RE = re.compile(r"(?m)^\s*#print\b.*$")
_LEAN4_VALID_PRINT_RE = re.compile(r"^#print\s+axioms\s+(HoloSphereInvariants\.[A-Za-z0-9_]+)$")
_LEAN4_AXIOM_NONE_RE = re.compile(r"^'(HoloSphereInvariants\.[A-Za-z0-9_]+)' does not depend on any axioms$")
_LEAN4_AXIOM_DEP_RE = re.compile(r"^'(HoloSphereInvariants\.[A-Za-z0-9_]+)' depends on axioms: \[([^\]]*)\]$")

_EXPECTED_PAYLOAD_CACHE: dict[str, Any] | None = None


class HoloSphereLeanError(ValueError):
    """Typed fail-closed error for HoloSphere Lean verifier."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise HoloSphereLeanError(code, message)


def _expected_payload_ref() -> dict[str, Any]:
    global _EXPECTED_PAYLOAD_CACHE
    if _EXPECTED_PAYLOAD_CACHE is None:
        _EXPECTED_PAYLOAD_CACHE = oracle.build_payload()
    return _EXPECTED_PAYLOAD_CACHE


def expected_payload() -> dict[str, Any]:
    return copy.deepcopy(_expected_payload_ref())


# ---------------------------------------------------------------------------
# Hardened Zone-B regular-file I/O and strict JSON loader
# ---------------------------------------------------------------------------

def _reject_symlink_ancestors(path: Path, label: str) -> None:
    cur = Path(os.path.abspath(os.fspath(path)))
    for candidate in (cur, *cur.parents):
        if candidate.is_symlink():
            raise HoloSphereLeanError("SYMLINK_FORBIDDEN", f"{label}: symlink path or ancestor forbidden")


def read_regular_bytes_capped(path: Path, label: str, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    _reject_symlink_ancestors(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise HoloSphereLeanError("IO_OPEN_FAIL", f"{label}: cannot open regular file ({exc.__class__.__name__})") from exc
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NON_REGULAR_FILE", f"{label}: not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            blk = os.read(fd, 65536)
            if not blk:
                break
            total += len(blk)
            require_ok(total <= max_bytes, "FILE_TOO_LARGE", f"{label}: exceeds {max_bytes} bytes")
            chunks.append(blk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_atomic_0600(path: Path, data: bytes) -> None:
    parent = path.parent
    require_ok(parent.is_dir(), "IO_WRITE_FAIL", "parent directory missing")
    fd, tmp_name = tempfile.mkstemp(prefix=".holo_lean_tmp_", dir=os.fspath(parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(os.fspath(tmp_path), os.fspath(path))
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        if tmp_path.exists():
            tmp_path.unlink()
        raise HoloSphereLeanError("IO_WRITE_FAIL", f"atomic write failed ({exc.__class__.__name__})") from exc


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "DUPLICATE_JSON_KEY", f"duplicate key {k!r}")
        out[k] = v
    return out


def _reject_float(tok: str) -> Any:
    raise HoloSphereLeanError("FLOAT_FORBIDDEN", f"float literal forbidden: {tok}")


def _reject_const(tok: str) -> Any:
    raise HoloSphereLeanError("NON_FINITE_FORBIDDEN", f"non-finite JSON constant forbidden: {tok}")


def _parse_int_capped(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    require_ok(len(digits) <= MAX_INT_DIGITS, "INT_DIGIT_LIMIT", "integer exceeds 80-digit cap")
    return int(tok)


def loads_json_strict(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise HoloSphereLeanError("INVALID_UTF8", f"{label}: invalid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_dup_keys,
            parse_float=_reject_float,
            parse_constant=_reject_const,
            parse_int=_parse_int_capped,
        )
    except HoloSphereLeanError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise HoloSphereLeanError("INVALID_JSON", f"{label}: malformed JSON ({exc.__class__.__name__})") from exc


# ---------------------------------------------------------------------------
# R13b O(1) in_ident Lean 4 lexer + 18-rule static pre-filter + live Lean
# ---------------------------------------------------------------------------

def _match_lean4_char_literal_end(src: str, i: int, n: int) -> int | None:
    if i + 2 < n:
        ch1 = src[i + 1]
        if ch1 not in ("\\", "'"):
            if src[i + 2] == "'":
                return i + 3
    if i + 3 < n:
        if src[i + 1] == "\\":
            if src[i + 3] == "'":
                if src[i + 2] != "\n":
                    return i + 4
            esc = src[i + 2]
            if esc == "x":
                if i + 5 < n:
                    if src[i + 5] == "'":
                        if all(c in _HEX_DIGITS for c in src[i + 3 : i + 5]):
                            return i + 6
            if esc == "u":
                if i + 7 < n:
                    if src[i + 7] == "'":
                        if all(c in _HEX_DIGITS for c in src[i + 3 : i + 7]):
                            return i + 8
    return None


def strip_lean4_source(src: str) -> str:
    """Strip -- and nested /- -/ comments plus Char literals via O(1) in_ident state machine."""
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    in_ident = False
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if depth > 0:
            if ch == "/" and nxt == "-":
                depth += 1
                out.append("  ")
                i += 2
            elif ch == "-" and nxt == "/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if ch == "/" and nxt == "-":
            in_ident = False
            depth = 1
            out.append("  ")
            i += 2
            continue
        if ch == "-" and nxt == "-":
            in_ident = False
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "'" and not in_ident:
            cend = _match_lean4_char_literal_end(src, i, n)
            if cend is not None:
                out.append(" \n " if src[i + 1] == "\n" else " " * (cend - i))
                i = cend
                continue
        in_ident = (ch in _LEAN4_ID_REST) if in_ident else (ch in _LEAN4_ID_FIRST)
        out.append(ch)
        i += 1
    require_ok(depth == 0, "LEAN4_UNTERMINATED_COMMENT", "unclosed /- block comment")
    return "".join(out)


def lean4_rule_hits(src: str) -> list[str]:
    stripped = strip_lean4_source(src)
    return [rid for rid, rx in LEAN4_RULES if rx.search(stripped) is not None]


def static_gate_lean4_source(src: str) -> None:
    require_ok("\r" not in src, "LEAN4_BAD_CHARS", "CR forbidden in Lean source")
    require_ok("\x00" not in src, "LEAN4_BAD_CHARS", "NUL forbidden in Lean source")
    hits = lean4_rule_hits(src)
    require_ok(hits == [], "LEAN4_TRUST_ESCAPE", f"banned Lean construct(s): {hits}")
    stripped = strip_lean4_source(src)
    seen: list[str] = []
    for pm in _LEAN4_PRINT_LINE_RE.finditer(stripped):
        vm = _LEAN4_VALID_PRINT_RE.match(pm.group(0).strip())
        require_ok(vm is not None, "LEAN4_BAD_PRINT", f"disallowed #print: {pm.group(0)[:80]!r}")
        seen.append(vm.group(1))
    require_ok(seen == list(oracle.LEAN_THEOREM_NAMES), "LEAN4_PRINT_SET_MISMATCH", "#print axioms mismatch")
    for short in oracle.LEAN_THEOREM_SHORT_NAMES:
        require_ok(
            re.search(rf"(?m)^theorem\s+{re.escape(short)}\b", stripped) is not None,
            "LEAN4_MISSING_THEOREM",
            f"missing theorem {short}",
        )


def resolve_lean_binary() -> str:
    env_bin = os.environ.get("ALETHEIA_LEAN_BIN", "")
    if env_bin != "":
        require_ok(os.path.isfile(env_bin), "LEAN_BINARY_NOT_FILE", "ALETHEIA_LEAN_BIN is not a file")
        require_ok(os.access(env_bin, os.X_OK), "LEAN_BINARY_NOT_EXECUTABLE", "ALETHEIA_LEAN_BIN not executable")
        return env_bin
    found = shutil.which("lean")
    require_ok(found is not None, "LEAN_BINARY_UNAVAILABLE", "lean not found on PATH")
    return str(found)


def scrubbed_lean_env() -> dict[str, str]:
    """Allowlist env for live Lean subprocess: drops LEAN_*, ELAN_*, LAKE_*, LD_*, DYLD_*."""
    return {k: v for k, v in os.environ.items() if k in _SAFE_LEAN_ENV_KEYS}


def parse_lean_axiom_output(stdout: str) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    idx = 0
    while idx < len(lines):
        ln = lines[idx]
        idx += 1
        while "depends on axioms: [" in ln and not ln.endswith("]") and idx < len(lines):
            ln = ln + " " + lines[idx]
            idx += 1
        m_none = _LEAN4_AXIOM_NONE_RE.match(ln)
        m_dep = _LEAN4_AXIOM_DEP_RE.match(ln)
        require_ok(m_none is not None or m_dep is not None, "LEAN_UNEXPECTED_STDOUT", f"unexpected line: {ln[:80]!r}")
        m = m_none if m_none is not None else m_dep
        tname = m.group(1)
        require_ok(tname not in seen, "LEAN_DUPLICATE_THEOREM", f"duplicate {tname}")
        axs = [] if m_dep is None else [a.strip() for a in m_dep.group(2).split(",") if a.strip()]
        for ax in axs:
            require_ok(ax in _ALLOWED_LEAN_AXIOMS, "LEAN_FORBIDDEN_AXIOM", f"forbidden axiom {ax}")
        seen[tname] = axs
    require_ok(frozenset(seen) == frozenset(oracle.LEAN_THEOREM_NAMES), "LEAN_THEOREM_SET_MISMATCH", "axiom set mismatch")
    return seen


def verify_lean4_file_live(lean_path: Path, *, check_sha: bool = True) -> dict[str, Any]:
    raw = read_regular_bytes_capped(lean_path, "lean_source")
    try:
        src = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise HoloSphereLeanError("INVALID_UTF8", "Lean file invalid UTF-8") from exc
    static_gate_lean4_source(src)
    src_sha = oracle.sha256_hex(raw)
    if check_sha:
        require_ok(src_sha == oracle.LEAN_SOURCE_SHA256, "LEAN_SHA_MISMATCH", "Lean source SHA-256 mismatch")
    lean_bin = resolve_lean_binary()
    env = scrubbed_lean_env()
    try:
        v_proc = subprocess.run([lean_bin, "--version"], capture_output=True, text=True, timeout=60, check=False, env=env)
        proc = subprocess.run([lean_bin, "--stdin"], input=src, capture_output=True, text=True, timeout=300, check=False, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoloSphereLeanError("LEAN_EXEC_ERROR", f"lean execution failed: {exc.__class__.__name__}") from exc
    require_ok(v_proc.returncode == 0, "LEAN_VERSION_MISMATCH", "lean --version failed")
    require_ok("version 4.36.0," in v_proc.stdout, "LEAN_VERSION_MISMATCH", "expected Lean 4.36.0")
    require_ok(proc.returncode == 0, "LEAN_COMPILE_FAIL", f"lean rc={proc.returncode}")
    require_ok(proc.stderr == "", "LEAN_STDERR_NONEMPTY", "lean stderr non-empty")
    require_ok("warning" not in proc.stdout.lower(), "LEAN_WARNING_FORBIDDEN", "lean warning emitted")
    seen = parse_lean_axiom_output(proc.stdout)
    return {
        "lean_version": v_proc.stdout.strip().splitlines()[0],
        "theorems_verified": len(seen),
        "lean_source_sha256": src_sha,
    }


# ---------------------------------------------------------------------------
# Field-complete secondary auditor (SEC; enforced on SEC, FR0, and FR1)
# ---------------------------------------------------------------------------

def strict_shape_match(a: Any, b: Any) -> str:
    stack: list[tuple[Any, Any, str]] = [(a, b, "payload")]
    while stack:
        x, y, path = stack.pop()
        if type(x) is not type(y):
            return path
        if type(x) is dict:
            if frozenset(x) != frozenset(y):
                return path
            for k in x:
                stack.append((x[k], y[k], f"{path}.{k}"))
        elif type(x) is list:
            if len(x) != len(y):
                return path
            for i, (xi, yi) in enumerate(zip(x, y)):
                stack.append((xi, yi, f"{path}[{i}]"))
    return ""


def strict_deep_equal(a: Any, b: Any) -> str:
    stack: list[tuple[Any, Any, str]] = [(a, b, "payload")]
    while stack:
        x, y, path = stack.pop()
        if type(x) is not type(y):
            return path
        if type(x) is dict:
            if frozenset(x) != frozenset(y):
                return path
            for k in x:
                stack.append((x[k], y[k], f"{path}.{k}"))
        elif type(x) is list:
            if len(x) != len(y):
                return path
            for i, (xi, yi) in enumerate(zip(x, y)):
                stack.append((xi, yi, f"{path}[{i}]"))
        elif x != y:
            return path
    return ""


def parse_capped_rational(val: Any, label: str) -> Fraction:
    require_ok(type(val) is str, "INVALID_RATIONAL_STR", f"{label}: must be str")
    require_ok(_RAT_RE.fullmatch(val) is not None, "INVALID_RATIONAL_STR", f"{label}: invalid rational")
    return Fraction(val)


def _independent_semantics(p: dict[str, Any]) -> None:
    """Independent algebraic checks across all 5 HoloSphere Pillars before leaf-complete match."""
    require_ok(p["novelty_grade"] == "N1", "SEC_NOVELTY", "novelty must be N1")
    require_ok(p["n2_claims"] == 0, "SEC_N2_ZERO", "n2_claims must be 0")
    lc = p["lean_companion"]
    require_ok(lc["named_theorem_count"] == 36, "SEC_LEAN_CNT", "36 named theorems required")
    require_ok(
        lc["named_theorem_count"] + lc["helper_theorem_count"] == lc["total_theorem_declarations"],
        "SEC_LEAN_TOTAL",
        "theorem declaration sum mismatch",
    )

    # Pillar 1: SU(2) Associative Scan & Suffix-Sum Adjoint Collapse + cayley_su2 + Local-S^3
    p1 = p["pillar1_su2_scan_and_suffix_adjoint"]
    require_ok(parse_capped_rational(p1["assoc_diff_sq_q"], "p1_assoc") == 0, "SEC_P1_ASSOC", "associativity diff != 0")
    require_ok(
        parse_capped_rational(p1["nonabelian_commutator_norm_sq_q"], "p1_comm") > 0,
        "SEC_P1_NONABELIAN",
        "SU(2) commutator must be non-zero",
    )
    require_ok(parse_capped_rational(p1["final_pi7_norm_sq_q"], "p1_norm") == 1, "SEC_P1_UNIT_NORM", "||Pi_7||^2 != 1")
    require_ok(
        parse_capped_rational(p1["hillis_vs_seq_8lane_diff_sq_q"], "p1_hs") == 0,
        "SEC_P1_HILLIS",
        "Hillis-Steele vs sequential diff != 0",
    )
    require_ok(
        parse_capped_rational(p1["chunked_vs_seq_diff_sq_q"], "p1_chk") == 0,
        "SEC_P1_CHUNKED",
        "chunked vs sequential diff != 0",
    )
    g_step = parse_capped_rational(p1["per_step_grad_q"], "p1_gstep")
    g_suf = parse_capped_rational(p1["suffix_sum_grad_q"], "p1_gsuf")
    require_ok(g_step != 0, "SEC_P1_GRAD_NZ", "gradient witness must be non-zero")
    require_ok(g_step == g_suf, "SEC_P1_ADJ_EQ", "per-step and suffix-sum adjoint must match")
    require_ok(parse_capped_rational(p1["adjoint_collapse_diff_q"], "p1_gdiff") == 0, "SEC_P1_ADJ_DIFF", "adjoint diff != 0")
    require_ok(parse_capped_rational(p1["cayley_su2_norm_err_q"], "p1_csu2n") == 0, "SEC_P1_CSU2_NORM", "cayley_su2 norm err != 0")
    require_ok(parse_capped_rational(p1["cayley_su2_acute_w_q"], "p1_csu2w") > 0, "SEC_P1_CSU2_POS", "cayley_su2 w must be > 0")
    require_ok(parse_capped_rational(p1["local_4d_bank_a_bleed_sq_q"], "p1_loc") == 0, "SEC_P1_LOCAL_ZERO", "local S^3 bleed != 0")
    require_ok(
        parse_capped_rational(p1["global_coupled_bank_a_bleed_sq_q"], "p1_glb") > 0,
        "SEC_P1_GLOBAL_BLEED",
        "global sphere must bleed across banks",
    )

    # Pillar 2: Covariant (Q, K, V) Parallel Transport & Un-Transport
    p2 = p["pillar2_covariant_qkv_transport"]
    require_ok(parse_capped_rational(p2["transport_isometry_diff_q"], "p2_iso") == 0, "SEC_P2_ISO", "isometry diff != 0")
    require_ok(parse_capped_rational(p2["qk_relative_holonomy_diff_q"], "p2_qk") == 0, "SEC_P2_QK", "QK holonomy diff != 0")
    require_ok(parse_capped_rational(p2["interval_cancel_diff_sq_q"], "p2_int") == 0, "SEC_P2_INTERVAL", "interval cancel != 0")
    require_ok(parse_capped_rational(p2["weighted_untransport_diff_sq_q"], "p2_wun") == 0, "SEC_P2_UNTRANS", "untransport diff != 0")
    require_ok(parse_capped_rational(p2["self_untransport_diff_sq_q"], "p2_self") == 0, "SEC_P2_SELF", "self untransport != 0")

    # Pillar 3: SFU-Free Rational Cayley Safe-Hopf Clock & Hopf S^3 -> S^2 Fiber
    p3 = p["pillar3_cayley_safe_hopf_and_fiber"]
    require_ok(parse_capped_rational(p3["unit_circle_max_err_q"], "p3_uc") == 0, "SEC_P3_CIRCLE", "unit circle err != 0")
    c_floor = parse_capped_rational(p3["cos_floor_tmax_4_5_q"], "p3_fl")
    c_min = parse_capped_rational(p3["min_schedule_cos_q"], "p3_min")
    require_ok(c_floor > 0, "SEC_P3_FLOOR_POS", "cos floor must be > 0")
    require_ok(c_min >= c_floor, "SEC_P3_FLOOR_BOUND", "min schedule cos < floor")
    require_ok(p3["strictly_monotone_schedule"] is True, "SEC_P3_MONO", "schedule must be strictly monotone")
    require_ok(parse_capped_rational(p3["obs31_obtuse_cos_q"], "p3_obc") < 0, "SEC_P3_OBS31_COS", "obtuse cos must be < 0")
    require_ok(parse_capped_rational(p3["obs31_self_align_q"], "p3_oba") < 0, "SEC_P3_OBS31_ALIGN", "obtuse self-align must be < 0")
    require_ok(parse_capped_rational(p3["hopf_s2_norm_sq_q"], "p3_hn") == 1, "SEC_P3_HOPF_NORM", "Hopf S^2 norm != 1")
    require_ok(parse_capped_rational(p3["fiber_rotated_diff_sq_q"], "p3_fib") == 0, "SEC_P3_HOPF_INV", "Hopf S^1 fiber diff != 0")
    require_ok(parse_capped_rational(p3["off_fiber_diff_sq_q"], "p3_off") > 0, "SEC_P3_HOPF_OFF", "off-fiber must change S^2")

    # Pillar 4: Beltrami [C, K] = 0, Poloidal Bingham Stagnant-Lid, & Darboux
    p4 = p["pillar4_beltrami_bingham_darboux"]
    require_ok(parse_capped_rational(p4["beltrami_commutator_diff_sq_q"], "p4_cm") == 0, "SEC_P4_COMM", "[C,K] != 0")
    e0 = parse_capped_rational(p4["beltrami_energy_in_q"], "p4_e0")
    e1 = parse_capped_rational(p4["beltrami_energy_out_q"], "p4_e1")
    h0 = parse_capped_rational(p4["beltrami_helicity_in_q"], "p4_h0")
    h1 = parse_capped_rational(p4["beltrami_helicity_out_q"], "p4_h1")
    require_ok(e0 == e1, "SEC_P4_ENERGY", "Beltrami energy not conserved")
    require_ok(h0 == h1, "SEC_P4_HELICITY", "Beltrami helicity not conserved")
    require_ok(parse_capped_rational(p4["cross_swap_energy_out_q"], "p4_es") == e0, "SEC_P4_SWAP_E", "swap energy != e0")
    require_ok(parse_capped_rational(p4["cross_swap_helicity_out_q"], "p4_hs") != h0, "SEC_P4_SWAP_H", "swap must break helicity")
    require_ok(
        parse_capped_rational(p4["rayleigh_after_sym_res_q"], "p4_r1") > parse_capped_rational(p4["rayleigh_before_q"], "p4_r0"),
        "SEC_P4_RAYLEIGH",
        "symmetric residual must raise Rayleigh ratio",
    )
    pol_in = parse_capped_rational(p4["bingham_pol_in_norm_sq_q"], "p4_bi")
    tau_sq = parse_capped_rational(p4["bingham_tau_sq_q"], "p4_bt")
    require_ok(pol_in <= tau_sq, "SEC_P4_LID_COND", "pol_in must be <= tau_sq")
    require_ok(parse_capped_rational(p4["bingham_pol_gated_norm_sq_q"], "p4_bg") == 0, "SEC_P4_LID_ZERO", "stagnant lid != 0")
    require_ok(parse_capped_rational(p4["bingham_tor_out_norm_sq_q"], "p4_bto") > 0, "SEC_P4_TOR_POS", "toroidal norm must stay > 0")
    require_ok(parse_capped_rational(p4["darboux_intertwine_diff_sq_q"], "p4_di") == 0, "SEC_P4_DARBOUX", "A L1 != L2 A")
    require_ok(parse_capped_rational(p4["darboux_ak_norm_sq_q"], "p4_dak") == 0, "SEC_P4_KER", "Ak != 0")
    require_ok(parse_capped_rational(p4["darboux_kernel_collision_diff_sq_q"], "p4_dc") == 0, "SEC_P4_COLL", "A(x+k) != Ax")
    require_ok(
        parse_capped_rational(p4["cayley2x2_closed_form_grad_w_q"], "p4_cgw") != 0,
        "SEC_P4_CAYLEY_GRAD_NZ",
        "Cayley2x2 grad witness must be non-zero",
    )
    require_ok(
        parse_capped_rational(p4["cayley2x2_adjoint_diff_q"], "p4_cadj") == 0,
        "SEC_P4_CAYLEY_ADJ",
        "Cayley2x2 closed-form adjoint diff != 0",
    )

    # Pillar 5: Weft Composed-DeltaW C_row Gate, 16B Holonomy Rollback, gateAsymmetric & Left-Quant
    p5 = p["pillar5_weft_crow_safety_and_rollback"]
    c_max = parse_capped_rational(p5["c_max_q"], "p5_cm")
    c1 = parse_capped_rational(p5["c_row_patch1_q"], "p5_c1")
    c2 = parse_capped_rational(p5["c_row_patch2_q"], "p5_c2")
    c_atk = parse_capped_rational(p5["c_row_composed_attack_q"], "p5_ca")
    c_slp = parse_capped_rational(p5["c_row_post_isotropic_sleep_q"], "p5_cs")
    require_ok(c1 <= c_max, "SEC_P5_P1_SAFE", "patch 1 must satisfy C_row <= c_max")
    require_ok(c2 <= c_max, "SEC_P5_P2_SAFE", "patch 2 must satisfy C_row <= c_max")
    require_ok(c_atk > c_max, "SEC_P5_COMP_BREACH", "composed attack must breach c_max")
    require_ok(p5["composed_attack_accepted"] is False, "SEC_P5_ATK_BLOCK", "composed attack must be rejected")
    require_ok(p5["safe_single_patch_accepted"] is True, "SEC_P5_SAFE_ACCEPT", "safe single patch must pass")
    require_ok(c_slp == c1, "SEC_P5_SLEEP_INV", "isotropic sleep must preserve C_row")
    require_ok(parse_capped_rational(p5["lifo_rollback_diff_sq_q"], "p5_rb") == 0, "SEC_P5_ROLLBACK", "rollback diff != 0")
    err1 = parse_capped_rational(p5["repoison_err_step1_sq_q"], "p5_rp1")
    err5 = parse_capped_rational(p5["repoison_err_step5_sq_q"], "p5_rp5")
    require_ok(err1 > 0, "SEC_P5_REPOISON_POS", "re-poisoning error must be > 0")
    require_ok(err1 == err5, "SEC_P5_REPOISON_CONST", "infinite-half-life error must stay constant")
    require_ok(parse_capped_rational(p5["restored_16b_err_sq_q"], "p5_r16") == 0, "SEC_P5_RESTORE_16B", "16B restore err != 0")
    require_ok(parse_capped_rational(p5["asym_recovery_excess_sq_q"], "p5_aex") == 0, "SEC_P5_ASYM_ZERO", "asym excess != 0")
    require_ok(parse_capped_rational(p5["sym_recovery_dist_sq_q"], "p5_sd") > 0, "SEC_P5_SYM_POS", "sym dist must be > 0")
    require_ok(p5["gate_asymmetric_recovery_accepted"] is True, "SEC_P5_ASYM_OK", "gateAsymmetric must accept recovery")
    require_ok(
        p5["gate_symmetric_zero_headroom_accepted"] is False,
        "SEC_P5_SYM_DEADLOCK",
        "symmetric gate must deadlock at R=0",
    )
    require_ok(
        parse_capped_rational(p5["left_quant_nullspace_drift_sq_q"], "p5_lq") == 0,
        "SEC_P5_LEFT_QUANT",
        "left-factor quant nullspace drift != 0",
    )
    require_ok(
        parse_capped_rational(p5["right_quant_nullspace_drift_sq_q"], "p5_rq") > 0,
        "SEC_P5_RIGHT_QUANT",
        "right-factor quant witness must breach ker(A)",
    )


def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    require_ok(type(payload) is dict, "INVALID_PAYLOAD_TYPE", "payload must be dict")
    exp = _expected_payload_ref()
    shape_bad = strict_shape_match(payload, exp)
    require_ok(shape_bad == "", "SCHEMA_SHAPE_MISMATCH", f"schema/type mismatch at {shape_bad}")
    _independent_semantics(payload)
    diff_path = strict_deep_equal(payload, exp)
    require_ok(diff_path == "", "SEMANTIC_VALUE_MISMATCH", f"payload mismatch at {diff_path}")
    return {"status": "OK"}


def _validate_exact_json_tree(obj: Any) -> None:
    """Iterative pre-hash tree validator: blocks subclasses, floats, tuples, and >80-digit ints."""
    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                require_ok(type(k) is str, "INVALID_JSON_KEY", "dict key must be exact str")
                stack.append(v)
        elif t is list:
            stack.extend(cur)
        elif t is int:
            require_ok(cur.bit_length() <= MAX_INT_BITS, "INT_DIGIT_LIMIT", "integer exceeds bit limit")
        elif t is str or t is bool:
            continue
        else:
            raise HoloSphereLeanError("INVALID_JSON_TYPE", f"forbidden value type: {t.__name__}")


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    require_ok(type(cert) is dict, "INVALID_ENVELOPE_TYPE", "certificate must be JSON object")
    require_ok(frozenset(cert) == _ENVELOPE_KEYS, "INVALID_ENVELOPE_KEYS", "unexpected envelope keys")
    require_ok(cert["schema_version"] == oracle.SCHEMA_VERSION, "SCHEMA_VERSION_MISMATCH", "schema_version mismatch")
    require_ok(cert["track_id"] == oracle.TRACK_ID, "TRACK_ID_MISMATCH", "track_id mismatch")
    payload_sha = cert["payload_sha256"]
    cert_sha = cert["certificate_sha256"]
    require_ok(type(payload_sha) is str, "INVALID_HASH", "payload_sha256 must be str")
    require_ok(_HEX64_RE.fullmatch(payload_sha) is not None, "INVALID_HASH", "payload_sha256 bad hex")
    require_ok(type(cert_sha) is str, "INVALID_HASH", "certificate_sha256 must be str")
    require_ok(_HEX64_RE.fullmatch(cert_sha) is not None, "INVALID_HASH", "certificate_sha256 bad hex")
    payload = cert["payload"]
    require_ok(type(payload) is dict, "INVALID_PAYLOAD_TYPE", "payload must be dict")
    _validate_exact_json_tree(payload)
    computed_psha = oracle.sha256_hex(oracle.canonical_json_bytes(payload))
    require_ok(computed_psha == payload_sha, "PAYLOAD_HASH_MISMATCH", "payload_sha256 mismatch")
    env_obj = {
        "schema_version": cert["schema_version"],
        "track_id": cert["track_id"],
        "payload_sha256": payload_sha,
    }
    computed_csha = oracle.sha256_hex(oracle.canonical_json_bytes(env_obj))
    require_ok(computed_csha == cert_sha, "CERT_HASH_MISMATCH", "certificate_sha256 mismatch")
    if full_replay:
        fresh = oracle.build_payload()
        replay_bad = strict_deep_equal(payload, fresh)
        require_ok(replay_bad == "", "REPLAY_MISMATCH", f"live replay mismatch at {replay_bad}")
    secondary_audit_payload(payload)
    return {
        "status": "OK",
        "track_id": oracle.TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "full_replay": full_replay,
    }


def verify_certificate_file(cert_path: Path, *, full_replay: bool = True) -> dict[str, Any]:
    raw = read_regular_bytes_capped(cert_path, "certificate")
    cert = loads_json_strict(raw, "certificate")
    res = verify_certificate_dict(cert, full_replay=full_replay)
    require_ok(raw == oracle.canonical_json_bytes(cert) + b"\n", "NON_CANONICAL_BYTES", "certificate bytes must be canonical JSON + LF")
    return res


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = False
    mode = ""
    target: Path | None = None
    i = 0
    try:
        while i < len(args):
            tok = args[i]
            if tok == "--json":
                as_json = True
                i += 1
            elif tok in ("--check", "--write", "--verify-lean"):
                require_ok(mode == "", "CONFLICTING_FLAGS", f"duplicate mode flag {tok}")
                mode = tok
                i += 1
                if i < len(args):
                    if not args[i].startswith("--"):
                        target = Path(args[i])
                        i += 1
            else:
                raise HoloSphereLeanError("UNKNOWN_FLAG", f"unrecognized argument: {tok}")
        if mode == "":
            mode = "--check"
        if mode == "--write":
            out_path = DEFAULT_CERT_PATH if target is None else target
            cert = oracle.build_certificate()
            data = oracle.canonical_json_bytes(cert) + b"\n"
            write_atomic_0600(out_path, data)
            res = {
                "status": "WRITTEN",
                "track_id": oracle.TRACK_ID,
                "certificate_sha256": cert["certificate_sha256"],
                "payload_sha256": cert["payload_sha256"],
            }
        elif mode == "--check":
            cert_path = DEFAULT_CERT_PATH if target is None else target
            res = verify_certificate_file(cert_path, full_replay=True)
        else:
            lean_path = DEFAULT_LEAN_PATH if target is None else target
            res = verify_lean4_file_live(lean_path, check_sha=True)
        if as_json:
            sys.stdout.write(json.dumps(res, sort_keys=True) + "\n")
        elif mode == "--verify-lean":
            sys.stdout.write(f"LEAN_OK theorems={res['theorems_verified']} sha256={res['lean_source_sha256']}\n")
        else:
            sys.stdout.write(f"OK {res['track_id']} cert_sha256={res['certificate_sha256']}\n")
        return 0
    except HoloSphereLeanError as exc:
        sys.stderr.write(f"ERROR {exc.code}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
