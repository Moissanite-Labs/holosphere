#!/usr/bin/env python3
"""Track S164 (EXP-Q45a & EXP-Q45c) Zone-B Hardened Verifier, Pure-Q AST
Auditor, Secondary Mathematical Cross-Auditor, R12 Static Lean 4 Scanner,
and Cold Live Lean 4.36.0 Kernel Verifier.

CLI modes:
  --check [path] [--json] [--live-lean]    Verify certificate + replay + Lean 4.36.0
  --write [path] [--json]                  Atomically write canonical certificate (0o600)
  --standalone                             Emit canonical certificate JSON to stdout
  [--json]                                 Default: check committed certificate in track dir
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import s164_fourier_mux_loop_oracle as oracle  # noqa: E402

MAX_JSON_BYTES = 512 * 1024
MAX_JSON_INT_DIGITS = 64
MAX_JSON_DEPTH = 32
_CANON_Q_RE = re.compile(r"^(?:0|-?[1-9][0-9]*)(?:/[1-9][0-9]*)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FORBIDDEN_AST_MODULES = frozenset({
    "math", "cmath", "decimal", "statistics", "numpy", "scipy", "torch", "random", "builtins",
})
_FORBIDDEN_AST_NAMES = frozenset({
    "float", "complex", "pow", "eval", "exec", "from_float", "from_decimal",
})

_LEAN4_BANNED_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])(?:
        sorry | admit | axiom | unsafe | native_decide | sorryAx
        | Classical | ofReduceBool | ofReduceNat | trustCompiler
        | implemented_by | extern | initialize | builtin_initialize
        | run_cmd | run_meta | run_tac | syntax | declare_syntax_cat
        | macro | macro_rules | elab | elab_rules | notation
        | dbg_trace | dbgTrace | dbgTraceVal | dbgTraceIfShared
        | dbgSleep | dbgStackTrace | panicWithPosWithDecl | throwError
        | csimp | opaque | import | mkSorry | bv_decide | set_option
    )(?![A-Za-z0-9_])
    | (?<![A-Za-z0-9_])(?:panic|unreachable)!
    | (?:\#\s*(?:eval|reduce|exit|check|synth|guard(?:_msgs)?)\b)
    | (?:\bIO\s*\.)
    | (?:\bdebug\s*\.\s*(?:skipKernelTC|byAsSorry|proofAsSorry|sorryAsError)\b)
    | (?:\bwarn\s*\.\s*sorry\b)
    """,
    re.VERBOSE,
)

_PUB_THM_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:@\[[^\]]*\]\s*)*(?:protected[ \t]+)?theorem[ \t]+([A-Za-z0-9_]+)\b"
)
_PRINT_AXIOMS_RE = re.compile(
    r"(?:^|\n)[ \t]*#print[ \t]+axioms[ \t]+FourierMuxLoopOperators\.([A-Za-z0-9_]+)\b"
)
_AXIOM_ENTRY_RE = re.compile(
    r"'([A-Za-z0-9_.]+)'\s+(does not depend on any axioms|depends on axioms:\s*\[([^\]]*)\])",
    re.DOTALL,
)
_ALLOWED_LEAN_AXIOMS = frozenset({"propext", "Quot.sound"})

_ENVELOPE_KEYS = frozenset({
    "schema_version", "track_id", "payload_sha256", "certificate_sha256", "payload",
})
_PAYLOAD_KEYS = frozenset({
    "schema_version",
    "track_id",
    "oracle_class",
    "novelty_classification",
    "energy_power_status",
    "exp_q45a_finite_fourier_crossover",
    "exp_q45a_holosphere_v5_head",
    "exp_q45c_mux_cor15_vs_alkashi",
    "exp_q45c_dual_axis_su2_sfast",
    "gates",
    "provenance_and_lean4",
})


def require_ok(cond: bool, code: str, message: str) -> None:
    if type(cond) is not bool or not cond:
        raise oracle.S164Error(code, message)


# ---------------------------------------------------------------------------
# Pure-Q AST Self-Audit (Zero Floats / Zero ast.Div / Zero ast.Pow)
# ---------------------------------------------------------------------------


def verify_pure_q_ast(source_text: str, label: str) -> None:
    tree = ast.parse(source_text, filename=label)
    for node in ast.walk(tree):
        require_ok(not isinstance(node, ast.Div), "AST_DIV_FORBIDDEN", f"{label}: ast.Div forbidden")
        require_ok(not isinstance(node, ast.Pow), "AST_POW_FORBIDDEN", f"{label}: ast.Pow forbidden")
        require_ok(not isinstance(node, ast.Assert), "AST_ASSERT_FORBIDDEN", f"{label}: bare assert forbidden")
        if isinstance(node, ast.Constant):
            tname = type(node.value).__name__
            require_ok(
                tname != "float" and tname != "complex",
                "AST_FLOAT_LITERAL",
                f"{label}: {tname} literal forbidden",
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".", 1)[0]
                require_ok(
                    root_mod not in _FORBIDDEN_AST_MODULES,
                    "AST_FORBIDDEN_IMPORT",
                    f"{label}: forbidden import {alias.name}",
                )
        if isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".", 1)[0]
            require_ok(
                root_mod not in _FORBIDDEN_AST_MODULES,
                "AST_FORBIDDEN_IMPORT",
                f"{label}: forbidden from-import {node.module}",
            )
        if isinstance(node, ast.Name):
            require_ok(
                node.id not in _FORBIDDEN_AST_NAMES,
                "AST_FORBIDDEN_NAME",
                f"{label}: forbidden name {node.id}",
            )
        if isinstance(node, ast.Attribute):
            require_ok(
                node.attr not in _FORBIDDEN_AST_NAMES,
                "AST_FORBIDDEN_ATTR",
                f"{label}: forbidden attr {node.attr}",
            )


# ---------------------------------------------------------------------------
# Zone-B Path, Filesystem & Strict JSON Hardening
# ---------------------------------------------------------------------------


def _check_path_str(raw: Any, label: str) -> Path:
    if isinstance(raw, Path):
        s = os.fspath(raw)
    else:
        require_ok(type(raw) is str, "TYPE_ERROR", f"{label}: expected str or Path")
        s = raw
    require_ok(len(s) > 0, "EMPTY_PATH", f"{label}: empty path")
    require_ok("\x00" not in s, "NUL_BYTE_FORBIDDEN", f"{label}: NUL byte forbidden")
    try:
        s.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise oracle.S164Error("INVALID_UTF8_SURROGATE", f"{label}: invalid UTF-8") from exc
    return Path(s)


def _unnormalized_parts(path: Path) -> tuple[str, ...]:
    if path.is_absolute():
        return path.parts
    return Path(os.getcwd()).joinpath(path).parts


def _check_no_symlink_components(path: Path, label: str) -> Path:
    parts = _unnormalized_parts(path)
    cur = Path(parts[0])
    for part in parts[1:]:
        cur = cur.joinpath(part)
        try:
            st = os.lstat(cur)
        except (FileNotFoundError, NotADirectoryError):
            break
        except OSError as exc:
            raise oracle.S164Error("IO_STAT_ERROR", f"{label}: {exc}") from exc
        require_ok(
            not stat.S_ISLNK(st.st_mode),
            "SYMLINK_IN_PATH",
            f"{label}: symlink rejected at {cur.name}",
        )
    return Path(os.path.abspath(os.fspath(path)))


def read_regular_bytes_capped(
    path_like: Any, label: str, max_bytes: int = MAX_JSON_BYTES
) -> bytes:
    oracle.require_exact_int(max_bytes, "max_bytes", lo=1, hi=64 * 1024 * 1024)
    p = _check_no_symlink_components(_check_path_str(path_like, label), label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(os.fspath(p), flags)
    except OSError as exc:
        raise oracle.S164Error("IO_OPEN_ERROR", f"{label}: {exc}") from exc
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE", f"{label}: not regular")
        require_ok(st.st_nlink == 1, "HARDLINK_FORBIDDEN", f"{label}: st_nlink={st.st_nlink}")
        require_ok(st.st_size <= max_bytes, "FILE_TOO_LARGE", f"{label}: exceeds {max_bytes}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total = total + len(chunk)
            require_ok(
                total <= max_bytes,
                "FILE_STREAM_TOO_LARGE",
                f"{label}: stream exceeds {max_bytes}",
            )
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_regular_bytes_atomic_0600(path_like: Any, data: bytes, label: str) -> Path:
    require_ok(type(data) is bytes, "INVALID_BYTES", f"{label}: data must be bytes")
    p = _check_no_symlink_components(_check_path_str(path_like, label), label)
    parent = p.parent
    try:
        pst = os.lstat(parent)
    except OSError as exc:
        raise oracle.S164Error("PARENT_STAT_ERROR", f"{label}: {exc}") from exc
    require_ok(stat.S_ISDIR(pst.st_mode), "PARENT_NOT_DIR", f"{label}: parent not dir")
    try:
        tst = os.lstat(p)
    except FileNotFoundError:
        tst = None
    except OSError as exc:
        raise oracle.S164Error("TARGET_STAT_ERROR", f"{label}: {exc}") from exc
    if tst is not None:
        require_ok(stat.S_ISREG(tst.st_mode), "TARGET_NOT_REGULAR", f"{label}: target not regular")
        require_ok(tst.st_nlink == 1, "TARGET_HARDLINK", f"{label}: target st_nlink={tst.st_nlink}")
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".s164_tmp_", dir=os.fspath(parent))
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        written = 0
        while written < len(view):
            written = written + os.write(fd, view[written:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_name, os.fspath(p))
        tmp_name = ""
    except OSError as exc:
        raise oracle.S164Error("IO_WRITE_ERROR", f"{label}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return p


def _scan_json_text_strict(text: str) -> None:
    require_ok(not text.startswith("\ufeff"), "BAD_BOM", "UTF-8 BOM rejected")
    require_ok("\r" not in text, "BAD_CRLF", "CR line ending rejected")
    depth = 0
    in_str = False
    escaped = False
    digit_run = 0
    prev_digit = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            digit_run = 0
            prev_digit = False
            continue
        if ch == '"':
            in_str = True
            digit_run = 0
            prev_digit = False
            continue
        if ch in "{[":
            depth = depth + 1
            require_ok(depth <= MAX_JSON_DEPTH, "JSON_TOO_DEEP", "JSON nesting exceeded")
        elif ch in "}]":
            depth = depth - 1
            require_ok(depth >= 0, "JSON_UNBALANCED", "JSON nesting negative")
        if ch.isdigit():
            digit_run = digit_run + 1
            require_ok(
                digit_run <= MAX_JSON_INT_DIGITS,
                "JSON_INT_TOO_LARGE",
                "JSON integer digit bound exceeded",
            )
            prev_digit = True
        else:
            if prev_digit:
                require_ok(ch not in ".eE", "FLOAT_FORBIDDEN", "JSON float token rejected")
            digit_run = 0
            prev_digit = False
    require_ok(not in_str, "JSON_UNTERMINATED_STR", "unterminated JSON string")
    require_ok(depth == 0, "JSON_UNBALANCED_FINAL", "unbalanced JSON brackets")


def _reject_float(raw: str) -> Any:
    raise oracle.S164Error("FLOAT_FORBIDDEN", f"JSON float rejected: {raw}")


def _reject_constant(raw: str) -> Any:
    raise oracle.S164Error("NONFINITE_JSON", f"non-finite JSON constant rejected: {raw}")


def _reject_large_int(raw: str) -> int:
    digits = raw[1:] if raw.startswith("-") else raw
    require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "JSON_INT_TOO_LARGE", "int digits exceeded")
    return int(raw)


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "DUPLICATE_JSON_KEY", f"duplicate JSON key: {k}")
        out[k] = v
    return out


def parse_strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise oracle.S164Error("BAD_UTF8", f"{label}: invalid UTF-8") from exc
    _scan_json_text_strict(text)
    try:
        obj = json.loads(
            text,
            parse_int=_reject_large_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_dup_keys,
        )
    except oracle.S164Error:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise oracle.S164Error("JSON_INVALID", f"{label}: {exc}") from exc
    require_ok(type(obj) is dict, "JSON_NOT_OBJECT", f"{label}: root must be dict")
    return obj


# ---------------------------------------------------------------------------
# R12-Grade Static Lean 4 Scanner & Cold Live Lean 4.36.0 Verifier
# ---------------------------------------------------------------------------


def strip_lean_comments_and_strings(src: str) -> str:
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    in_str = False
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if (i + 1 < n) else ""
        if depth > 0:
            if ch == "/" and nxt == "-":
                depth = depth + 1
                out.append("  ")
                i = i + 2
                continue
            if ch == "-" and nxt == "/":
                depth = depth - 1
                out.append("  ")
                i = i + 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i = i + 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < n:
                out.append("  ")
                i = i + 2
                continue
            if ch == '"':
                in_str = False
                out.append(" ")
                i = i + 1
                continue
            out.append("\n" if ch == "\n" else " ")
            i = i + 1
            continue
        if ch == "-" and nxt == "-":
            while i < n and src[i] != "\n":
                out.append(" ")
                i = i + 1
            continue
        if ch == "/" and nxt == "-":
            depth = 1
            out.append("  ")
            i = i + 2
            continue
        if ch == "'" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] in "_'")):
            if i + 3 < n and src[i + 1] == "\\" and src[i + 3] == "'":
                out.append("    ")
                i = i + 4
                continue
            if i + 2 < n and src[i + 1] not in ("\\", "'", "\n") and src[i + 2] == "'":
                out.append("   ")
                i = i + 3
                continue
        if ch == '"':
            in_str = True
            out.append(" ")
            i = i + 1
            continue
        out.append(ch)
        i = i + 1
    require_ok(depth == 0, "LEAN_UNTERM_COMMENT", "unterminated Lean block comment")
    require_ok(not in_str, "LEAN_UNTERM_STRING", "unterminated Lean string literal")
    return "".join(out)


def scan_lean_source_r12(src: str) -> dict[str, Any]:
    cleaned = strip_lean_comments_and_strings(src)
    require_ok(
        _LEAN4_BANNED_RE.search(cleaned) is None,
        "LEAN_ESCAPE_FOUND",
        "forbidden Lean 4 trust-escape keyword detected",
    )
    pub_thms = _PUB_THM_RE.findall(cleaned)
    printed = _PRINT_AXIOMS_RE.findall(cleaned)
    expected_short = [name.split(".", 1)[1] for name in oracle.LEAN_THEOREM_NAMES]
    require_ok(
        len(pub_thms) == len(expected_short),
        "LEAN_THM_COUNT",
        f"expected {len(expected_short)} public theorems, found {len(pub_thms)}",
    )
    require_ok(
        set(pub_thms) == set(expected_short),
        "LEAN_THM_SET_MISMATCH",
        "public theorem set does not match pinned names",
    )
    require_ok(
        set(printed) == set(expected_short),
        "LEAN_AXIOM_PRINT_MISMATCH",
        "#print axioms set does not match public theorems",
    )
    return {
        "static_r12_clean": True,
        "theorems_verified": len(pub_thms),
    }


def validate_lean_axiom_stdout(stdout_text: str) -> dict[str, Any]:
    seen: dict[str, list[str]] = {}
    for match in _AXIOM_ENTRY_RE.finditer(stdout_text):
        thm = match.group(1)
        body = match.group(2)
        if body.startswith("does not depend"):
            seen[thm] = []
        else:
            raw_list = match.group(3)
            axioms = [tok.strip() for tok in raw_list.split(",") if tok.strip() != ""]
            for ax in axioms:
                require_ok(
                    ax in _ALLOWED_LEAN_AXIOMS,
                    "LEAN_FORBIDDEN_AXIOM",
                    f"forbidden Lean axiom {ax} in {thm}",
                )
            seen[thm] = axioms
    expected_set = set(oracle.LEAN_THEOREM_NAMES)
    require_ok(
        set(seen.keys()) == expected_set,
        "LEAN_AXIOM_SET_MISMATCH",
        f"axiom audit matched {len(seen)}/{len(expected_set)} theorems",
    )
    return {"live_axioms_verified": len(seen), "zero_classical_choice": True}


def default_live_lean_runner(track_dir: Path) -> dict[str, Any]:
    home = os.environ.get("HOME", "")
    path_parts = [
        os.fspath(Path(home).joinpath(".local", "bin")),
        os.fspath(Path(home).joinpath(".elan", "bin")),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    env = {
        "PATH": ":".join(path_parts),
        "HOME": home,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    lean_files = (
        oracle.PINNED_LEAN_FILES["lean_source"],
        "lean-toolchain",
        "lakefile.toml",
        "lake-manifest.json",
    )
    with tempfile.TemporaryDirectory(prefix="s164_lean_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        for fname in lean_files:
            raw_lean_bytes = read_regular_bytes_capped(track_dir.joinpath(fname), fname)
            tmp_p.joinpath(fname).write_bytes(raw_lean_bytes)
        ver = subprocess.run(
            ["lean", "--version"],
            cwd=tmp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        bld = subprocess.run(
            ["lake", "build"],
            cwd=tmp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        chk = subprocess.run(
            ["lean", oracle.PINNED_LEAN_FILES["lean_source"]],
            cwd=tmp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    return {
        "version_rc": ver.returncode,
        "version_output": ver.stdout + ver.stderr,
        "build_rc": bld.returncode,
        "build_output": bld.stdout + bld.stderr,
        "axiom_rc": chk.returncode,
        "axiom_stdout": chk.stdout,
    }


def validate_lean_companion_metadata(toolchain_text: str, manifest: dict[str, Any]) -> None:
    require_ok(
        toolchain_text.strip() == oracle.PINNED_LEAN_FILES["lean_toolchain"],
        "BAD_TOOLCHAIN",
        "unexpected lean-toolchain value",
    )
    require_ok(
        manifest.get("packages") == [],
        "NONEMPTY_LAKE_PACKAGES",
        "lake-manifest.json packages must be empty",
    )


def verify_companion_pins_and_lean(
    track_dir: Path,
    *,
    run_live_lean: bool = False,
    lean_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pin_checks = (
        ("lean_source", oracle.PINNED_LEAN_FILES["lean_source"], oracle.PINNED_LEAN_FILES["lean_source_sha256"]),
        ("lean_toolchain", "lean-toolchain", oracle.PINNED_LEAN_FILES["lean_toolchain_sha256"]),
        ("lakefile", "lakefile.toml", oracle.PINNED_LEAN_FILES["lakefile_sha256"]),
        ("lake_manifest", "lake-manifest.json", oracle.PINNED_LEAN_FILES["lake_manifest_sha256"]),
    )
    lean_src = ""
    toolchain_text = ""
    manifest: dict[str, Any] = {}
    for key, fname, expected_sha in pin_checks:
        raw = read_regular_bytes_capped(track_dir.joinpath(fname), key)
        got_sha = oracle.sha256_bytes(raw)
        require_ok(got_sha == expected_sha, "COMPANION_SHA_MISMATCH", f"{fname} sha mismatch")
        if key == "lean_source":
            lean_src = raw.decode("utf-8", "strict")
        elif key == "lean_toolchain":
            toolchain_text = raw.decode("utf-8", "strict")
        elif key == "lake_manifest":
            manifest = parse_strict_json_bytes(raw, "lake-manifest.json")
    validate_lean_companion_metadata(toolchain_text, manifest)

    # Pure-Q AST self-audit on oracle & verifier source files
    for py_name in ("s164_fourier_mux_loop_oracle.py", "verify_s164_fourier_mux_loop.py"):
        py_raw = read_regular_bytes_capped(track_dir.joinpath(py_name), py_name)
        verify_pure_q_ast(py_raw.decode("utf-8", "strict"), py_name)

    info = scan_lean_source_r12(lean_src)
    runner = lean_runner
    if run_live_lean and runner is None:
        runner = default_live_lean_runner
    if runner is not None:
        res = runner(track_dir)
        require_ok(res["version_rc"] == 0, "LEAN_VERSION_RC", "lean --version failed")
        require_ok("4.36.0" in res["version_output"], "BAD_LEAN_VERSION", "expected Lean 4.36.0")
        require_ok(res["build_rc"] == 0, "LAKE_BUILD_FAIL", "lake build failed")
        require_ok(
            "warning:" not in res["build_output"].lower(),
            "LAKE_BUILD_WARNING",
            "lake build emitted warnings",
        )
        require_ok(res["axiom_rc"] == 0, "LEAN_AXIOM_RC", "lean axiom check failed")
        live_info = validate_lean_axiom_stdout(res["axiom_stdout"])
        info["live_axioms_verified"] = live_info["live_axioms_verified"]
    return info


# ---------------------------------------------------------------------------
# Independent Secondary Mathematical Cross-Audit
# ---------------------------------------------------------------------------


def _parse_canon_q(raw: Any, label: str) -> Fraction:
    require_ok(type(raw) is str, "Q_TYPE_ERROR", f"{label}: expected str")
    require_ok(_CANON_Q_RE.fullmatch(raw) is not None, "Q_FORMAT_ERROR", f"{label}: {raw}")
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num_digits = num_s[1:] if num_s.startswith("-") else num_s
        require_ok(
            len(num_digits) <= MAX_JSON_INT_DIGITS and len(den_s) <= MAX_JSON_INT_DIGITS,
            "Q_TOO_LARGE",
            f"{label}: rational digits exceeded",
        )
        num_i = int(num_s)
        den_i = int(den_s)
        val = Fraction(num_i, den_i)
        require_ok(
            val.numerator == num_i and val.denominator == den_i,
            "Q_NOT_REDUCED",
            f"{label}: unreduced rational {raw}",
        )
        return val
    digits = raw[1:] if raw.startswith("-") else raw
    require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "Q_TOO_LARGE", f"{label}: rational digits exceeded")
    return Fraction(int(raw), 1)


def _check_sha256_field(val: Any, label: str) -> str:
    require_ok(type(val) is str, "SHA_TYPE_ERROR", f"{label}: expected str")
    require_ok(_SHA256_RE.fullmatch(val) is not None, "SHA_FORMAT_ERROR", f"{label}: bad hex")
    return val


def _require_field(
    container: dict[str, Any], key: str, expected_type: type | None, label: str
) -> Any:
    require_ok(key in container, "MISSING_KEY", f"{label}: missing key {key}")
    val = container[key]
    if expected_type is not None:
        require_ok(
            type(val) is expected_type,
            "SCHEMA_TYPE_ERROR",
            f"{label}.{key}: expected {expected_type.__name__}",
        )
    return val


def verify_certificate_object(
    cert: dict[str, Any],
    *,
    track_dir: Path | None = None,
    run_live_lean: bool = False,
    lean_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require_ok(type(cert) is dict, "CERT_NOT_DICT", "certificate must be dict")
    require_ok(set(cert.keys()) == _ENVELOPE_KEYS, "ENVELOPE_KEYS_MISMATCH", "envelope keys invalid")
    require_ok(cert["schema_version"] == oracle.SCHEMA_VERSION, "SCHEMA_MISMATCH", "bad schema")
    require_ok(cert["track_id"] == oracle.TRACK_ID, "TRACK_MISMATCH", "bad track_id")

    p_sha = _check_sha256_field(cert["payload_sha256"], "payload_sha256")
    c_sha = _check_sha256_field(cert["certificate_sha256"], "certificate_sha256")
    payload = cert["payload"]
    require_ok(type(payload) is dict, "PAYLOAD_NOT_DICT", "payload must be dict")
    require_ok(set(payload.keys()) == _PAYLOAD_KEYS, "PAYLOAD_KEYS_MISMATCH", "payload keys invalid")
    require_ok(
        payload["energy_power_status"] == oracle.ENERGY_POLICY_STATUS,
        "ENERGY_POLICY_VIOLATION",
        "bad energy_power_status",
    )

    got_p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    require_ok(got_p_sha == p_sha, "PAYLOAD_SHA_MISMATCH", "payload sha256 mismatch")
    body = {
        "schema_version": cert["schema_version"],
        "track_id": cert["track_id"],
        "payload_sha256": p_sha,
        "payload": payload,
    }
    got_c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(body))
    require_ok(got_c_sha == c_sha, "CERT_SHA_MISMATCH", "certificate sha256 mismatch")

    # Independent secondary mathematical invariants (survive even against re-signed payload!)
    p1 = _require_field(payload, "exp_q45a_finite_fourier_crossover", dict, "payload")
    require_ok(
        _require_field(p1, "fwht256_gates_per_vector", int, "p1") == 2048,
        "P1_FWHT_GATES",
        "FWHT_256 gates != 2048",
    )
    p1_out_sq = _require_field(p1, "fwht256_bent4_parseval_out_sq", int, "p1")
    p1_in_sq = _require_field(p1, "fwht256_bent4_parseval_in_sq", int, "p1")
    require_ok(p1_out_sq == 256 * p1_in_sq, "P1_BENT4_PARSEVAL", "bent4 Parseval mismatch")
    require_ok(
        _require_field(p1, "c3_crossover_m_star", int, "p1") == 10,
        "P1_CROSSOVER_M",
        "C=3 crossover m* != 10",
    )
    s159_gap = _parse_canon_q(_require_field(p1, "s159_v4_minus_row3_t8192_bpb_q", str, "p1"), "p1.s159_gap")
    require_ok(s159_gap > Fraction(0, 1) and s159_gap <= Fraction(1, 4), "P1_S159_GAP", "bad S159 gap")

    p2 = _require_field(payload, "exp_q45a_holosphere_v5_head", dict, "payload")
    require_ok(
        _require_field(p2, "v5_unique_vectors_of_260", int, "p2") == 260,
        "P2_V5_UNIQUE",
        "V5 unique vectors != 260",
    )
    require_ok(
        _require_field(p2, "v5_max_offdiag_dot", int, "p2") == 0,
        "P2_V5_ORTHO",
        "V5 offdiag dot != 0",
    )
    archs = _require_field(p2, "architectures", dict, "p2")
    v5b = _require_field(archs, "V5B_Monarch3_16x32_16x16_16x16", dict, "p2.architectures")
    require_ok(
        _require_field(v5b, "hbm_vocab_bytes", int, "p2.v5b") == 0,
        "P2_V5B_HBM",
        "V5-B HBM vocab != 0",
    )
    require_ok(
        _require_field(v5b, "e2e_smem_bytes", int, "p2.v5b") <= oracle.SM86_STATIC_SMEM_LIMIT_BYTES,
        "P2_V5B_SMEM",
        "V5-B exceeds 48 KiB sm_86 SMEM limit",
    )

    p3 = _require_field(payload, "exp_q45c_mux_cor15_vs_alkashi", dict, "payload")
    fails = _require_field(p3, "rho_9_10_first_failure_S", dict, "p3")
    bf16_fail = _require_field(fails, "BF16", int, "p3.fails")
    fp16_fail = _require_field(fails, "FP16", int, "p3.fails")
    fp32_fail = _require_field(fails, "FP32", int, "p3.fails")
    require_ok(
        bf16_fail == 4 and fp16_fail == 5 and fp32_fail == 12,
        "P3_MUX_THRESH",
        "MUX rho=9/10 precision collapse thresholds mismatch",
    )
    require_ok(
        _require_field(fails, "AlKashi_FixedRadix_u0", None, "p3.fails") is None,
        "P3_ALKASHI_U0",
        "Al-Kashi u=0 failed",
    )
    a16 = _require_field(p3, "a16_block_h4_u0_audit", dict, "p3")
    a16_h4_sq = _require_field(a16, "sum_h4_sq", int, "p3.a16")
    a16_c_sq = _require_field(a16, "sum_c_sq", int, "p3.a16")
    require_ok(a16_h4_sq == 4 * a16_c_sq, "P3_A16_PARSEVAL", "A_16 Block-H4 Parseval mismatch")
    require_ok(
        _require_field(a16, "parseval_mismatch_count", int, "p3.a16") == 0,
        "P3_A16_MISMATCH",
        "A_16 Block-H4 mismatch count != 0",
    )

    p4 = _require_field(payload, "exp_q45c_dual_axis_su2_sfast", dict, "payload")
    regimes = _require_field(p4, "dual_axis_carry_regimes", dict, "p4")
    reg_guard = _require_field(regimes, "beta_min_guarded_11_20", dict, "p4.regimes")
    reg_last = _require_field(regimes, "carry_last_beta0", dict, "p4.regimes")
    d_guard = _parse_canon_q(_require_field(reg_guard, "max_drift_from_1_q", str, "p4.reg_guard"), "p4.d_guard")
    d_last = _parse_canon_q(_require_field(reg_last, "max_drift_from_1_q", str, "p4.reg_last"), "p4.d_last")
    require_ok(d_guard < d_last, "P4_DAC_DAMP", "beta_min guarded failed to beat carry-last")
    su2_rows = _require_field(p4, "su2_loop_alignment_r1_to_r4", list, "p4")
    require_ok(len(su2_rows) == 4, "P4_SU2_LEN", "expected 4 SU(2) loop rows")
    for row in su2_rows:
        require_ok(type(row) is dict, "SCHEMA_TYPE_ERROR", "p4.su2_row: expected dict")
        require_ok(
            _parse_canon_q(_require_field(row, "su2_aligned_cosine_q", str, "p4.su2_row"), "p4.su2_align")
            == Fraction(1, 1),
            "P4_SU2_ALIGN",
            "SU(2) alignment != 1",
        )

    gates = _require_field(payload, "gates", dict, "payload")
    g1 = _require_field(gates, "g1_finite_fourier_n256_crossover", bool, "gates")
    g2 = _require_field(gates, "g2_holosphere_v5_zero_alias_smem", bool, "gates")
    g3 = _require_field(gates, "g3_mux_cor15_vs_alkashi_fixed_radix", bool, "gates")
    g4 = _require_field(gates, "g4_dual_axis_carry_su2_loop_stability", bool, "gates")
    require_ok(g1 and g2 and g3 and g4, "GATES_FAILED", "certificate gates did not all pass")
    _require_field(payload, "provenance_and_lean4", dict, "payload")

    expected = oracle.build_certificate()
    require_ok(cert == expected, "ORACLE_REPLAY_MISMATCH", "certificate differs from exact replay")

    tdir = oracle.TRACK_DIR if track_dir is None else track_dir
    lean_info = verify_companion_pins_and_lean(
        tdir, run_live_lean=run_live_lean, lean_runner=lean_runner
    )
    return {
        "status": "PASS",
        "track_id": oracle.TRACK_ID,
        "schema_version": oracle.SCHEMA_VERSION,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "lean": lean_info,
    }


def _emit_error(code: str, message: str) -> None:
    line = json.dumps(
        {"status": "FAIL", "code": code, "message": message},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ) + "\n"
    sys.stderr.write(line)


def _resolve_cert_path(given: str | None) -> Path:
    if given is None:
        return oracle.TRACK_DIR.joinpath(oracle.DEFAULT_CERT_FILENAME)
    cand = Path(given)
    if cand.is_absolute():
        return cand
    return oracle.TRACK_DIR.joinpath(cand)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S164 FiniteFourier x MUX x HoloSphere-V5 verifier")
    parser.add_argument("--check", nargs="?", const=oracle.DEFAULT_CERT_FILENAME, default=None)
    parser.add_argument("--write", nargs="?", const=oracle.DEFAULT_CERT_FILENAME, default=None)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--live-lean", action="store_true")
    args = parser.parse_args(argv)

    modes = 0
    if args.check is not None:
        modes = modes + 1
    if args.write is not None:
        modes = modes + 1
    if args.standalone:
        modes = modes + 1
    require_ok(modes <= 1, "MODE_CONFLICT", "specify at most one of --check, --write, --standalone")

    if args.standalone:
        cert = oracle.build_certificate()
        sys.stdout.write(oracle.canonical_json_bytes(cert).decode("ascii"))
        return 0

    if args.write is not None:
        cert = oracle.build_certificate()
        raw = oracle.canonical_json_bytes(cert)
        out_p = write_regular_bytes_atomic_0600(_resolve_cert_path(args.write), raw, "write_cert")
        res = {
            "status": "WRITTEN",
            "path": out_p.name,
            "bytes": len(raw),
            "certificate_sha256": cert["certificate_sha256"],
        }
        if args.json_out:
            sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii"))
        else:
            sys.stdout.write(f"WROTE {out_p.name} ({len(raw)} bytes) sha256={cert['certificate_sha256']}\n")
        return 0

    cert_p = _resolve_cert_path(args.check)
    raw_cert = read_regular_bytes_capped(cert_p, "certificate")
    require_ok(raw_cert.endswith(b"\n"), "BAD_TRAILING_NEWLINE", "certificate must end with LF")
    cert_obj = parse_strict_json_bytes(raw_cert, "certificate")
    summary = verify_certificate_object(cert_obj, run_live_lean=args.live_lean)
    if args.json_out:
        sys.stdout.write(oracle.canonical_json_bytes(summary).decode("ascii"))
    else:
        sys.stdout.write(
            f"Track {oracle.TRACK_ID} PASS cert_sha256={summary['certificate_sha256']} "
            f"theorems={summary['lean']['theorems_verified']}\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except oracle.S164Error as exc:
        _emit_error(exc.code, exc.message)
        return 2
    except RecursionError:
        _emit_error("RECURSION_ERROR", "recursion limit exceeded")
        return 2
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        _emit_error("OS_ERROR", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
