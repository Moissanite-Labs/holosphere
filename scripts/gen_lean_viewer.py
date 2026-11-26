#!/usr/bin/env python3
"""Generate the LEAN_SUITES block of the GitHub Pages explorer from lean/*.lean.

Extracts every top-level public theorem whose line starts with ``theorem p``
(the p1_..p50_ family) from the five Lean suites, keeping the exact source text
of each statement (optionally including the immediately preceding ``/-- ... -/``
docstring) up to but not including the ``:=`` that starts the proof, and writes
the result as a JSON object literal into docs/index.html between the
BEGIN/END GENERATED LEAN_SUITES marker comments.

Usage:
    python3 scripts/gen_lean_viewer.py                 # rewrite docs/index.html in place
    python3 scripts/gen_lean_viewer.py --check         # exit 1 on drift, 0 when up to date
    python3 scripts/gen_lean_viewer.py --html PATH --lean-dir PATH [--check]

Python 3.11+, stdlib only. No bare asserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent

DEFAULT_HTML = REPO_ROOT / "docs" / "index.html"
DEFAULT_LEAN_DIR = REPO_ROOT / "lean"

INDENT = "      "
BEGIN_MARKER = "/* BEGIN GENERATED LEAN_SUITES (scripts/gen_lean_viewer.py; do not edit by hand) */"
END_MARKER = "/* END GENERATED LEAN_SUITES */"

# (suite key, lean file name) in display order.
SUITES = [
    ("invariants", "HoloSphereInvariants.lean"),
    ("barrington", "HoloSphereSpinorBarrington.lean"),
    ("bridges", "HoloSphereCrossSpineBridges.lean"),
    ("cellop", "HoloSphereCellOpInvariants.lean"),
    ("convergence", "HoloSphereConvergenceTheorems.lean"),
]

# A `:=` that directly follows `let <name>` on the same line belongs to a
# let-binder inside the statement term, not to the proof.
LET_BINDER_RE = re.compile(r"let\s+[A-Za-z_][A-Za-z0-9_']*\s*$")

OPENERS = "([{"
CLOSERS = ")]}"

MAX_FILE_BYTES = 16 * 1024 * 1024
_FORBIDDEN_SYS_PREFIXES = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/var",
)


class GeneratorError(Exception):
    """Raised when the Lean sources or the HTML block cannot be processed."""


def _is_forbidden_system_path(path_str: str) -> bool:
    norm = re.sub(r"^/+", "/", path_str)
    if norm == "/":
        return True
    if norm.startswith("/dev/shm/"):
        return False
    return any(norm == pfx or norm.startswith(pfx + "/") for pfx in _FORBIDDEN_SYS_PREFIXES)


def _check_safe_path(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise GeneratorError(f"{label} must be Path")
    raw = str(path)
    if "\x00" in raw or ".." in path.parts or raw.startswith("//"):
        raise GeneratorError(f"{label} contains NUL, '..', or leading '//'")
    try:
        raw.encode("utf-8")
        abs_path = Path(os.path.abspath(raw))
        real_parent = Path(os.path.realpath(str(abs_path.parent)))
    except (OSError, UnicodeEncodeError) as exc:
        raise GeneratorError(f"{label} path resolution failed: {exc}") from exc
    resolved = real_parent / abs_path.name
    if _is_forbidden_system_path(resolved.as_posix()):
        raise GeneratorError(f"{label} targets forbidden system path")
    if resolved.is_symlink():
        raise GeneratorError(f"refusing symlink for {label}: {resolved}")
    return resolved


def _check_dir_safe(path: Path, label: str) -> Path:
    safe = _check_safe_path(path, label)
    dfd = -1
    try:
        dfd = os.open(str(safe), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise GeneratorError(f"{label} must be an existing non-symlink directory: {exc}") from exc
    finally:
        if dfd >= 0:
            os.close(dfd)
    return safe


def _read_bytes_safe(path: Path, label: str) -> bytes:
    safe = _check_safe_path(path, label)
    parent_fd = -1
    fd = -1
    try:
        parent_fd = os.open(str(safe.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = os.open(safe.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise GeneratorError(f"{label} must be a regular file: {path}")
        if st.st_size > MAX_FILE_BYTES:
            raise GeneratorError(f"{label} exceeds {MAX_FILE_BYTES} byte cap: {path}")
        os.set_blocking(fd, True)
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            return fh.read(MAX_FILE_BYTES)
    except GeneratorError:
        raise
    except OSError as exc:
        raise GeneratorError(f"failed reading {label} ({path}): {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _decode_utf8_safe(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GeneratorError(f"{label} is not valid UTF-8: {exc}") from exc


def _write_bytes_atomic(path: Path, data: bytes, label: str) -> None:
    safe = _check_safe_path(path, label)
    parent_fd = -1
    fd = -1
    tmp_name = f".{safe.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    created_tmp = False
    try:
        parent_fd = os.open(str(safe.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            existing = os.stat(safe.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode):
                raise GeneratorError(f"{label} write target must be a regular file")
            if existing.st_nlink != 1:
                raise GeneratorError(f"{label} write target hardlink forbidden")
        fd = os.open(
            tmp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_fd,
        )
        created_tmp = True
        view = memoryview(data)
        pos = 0
        while pos < len(view):
            n = os.write(fd, view[pos:])
            if n <= 0:
                raise GeneratorError(f"short write to {label} temp file")
            pos += n
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_name, safe.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        created_tmp = False
    except GeneratorError:
        raise
    except OSError as exc:
        raise GeneratorError(f"failed writing {label} ({path}): {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if created_tmp and parent_fd >= 0:
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)


def docstring_start(lines: list[str], theorem_idx: int) -> int:
    """Return the first line index of a ``/-- ... -/`` docstring directly above."""
    k = theorem_idx - 1
    if k < 0 or not lines[k].rstrip().endswith("-/"):
        return theorem_idx
    m = k
    while m >= 0 and lines[m].strip() != "" and not lines[m].lstrip().startswith("/--"):
        m -= 1
    if m >= 0 and lines[m].lstrip().startswith("/--"):
        return m
    return theorem_idx


def find_proof_colon(lines: list[str], start_idx: int) -> tuple[int, int]:
    """Locate the first top-level non-let ``:=`` at/after ``start_idx``.

    Returns (line_index, column) of the ``:=`` that starts the proof.
    """
    depth = 0
    j = start_idx
    while j < len(lines):
        line = lines[j]
        col = 0
        while col < len(line):
            ch = line[col]
            if ch in OPENERS:
                depth += 1
            elif ch in CLOSERS:
                if depth > 0:
                    depth -= 1
            elif ch == ":" and col + 1 < len(line) and line[col + 1] == "=":
                if depth == 0 and not LET_BINDER_RE.search(line[:col]):
                    return (j, col)
            col += 1
        j += 1
    raise GeneratorError(
        f"no top-level ':=' (proof start) found for theorem at line {start_idx + 1}"
    )


def extract_statements(text: str, source_name: str) -> list[str]:
    """Extract the exact statement text of every ``theorem p...`` declaration."""
    lines = text.split("\n")
    statements: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("theorem p"):
            proof_line, proof_col = find_proof_colon(lines, i)
            doc_start = docstring_start(lines, i)
            stmt_lines = lines[doc_start:proof_line] + [lines[proof_line][:proof_col]]
            statements.append("\n".join(stmt_lines).rstrip())
            i = proof_line + 1
        else:
            i += 1
    if not statements:
        raise GeneratorError(f"no 'theorem p' declarations found in {source_name}")
    return statements


def build_suites(lean_dir: Path) -> dict[str, dict[str, str]]:
    """Build the LEAN_SUITES object {key: {label, code}} from the lean files."""
    safe_dir = _check_dir_safe(lean_dir, "lean_dir")
    suites: dict[str, dict[str, str]] = {}
    for key, fname in SUITES:
        path = safe_dir / fname
        data = _read_bytes_safe(path, f"lean/{fname}")
        text = _decode_utf8_safe(data, f"lean/{fname}")
        digest = hashlib.sha256(data).hexdigest()[:16]
        statements = extract_statements(text, fname)
        label = (
            f"lean/{fname} ({len(data):,} bytes · {len(statements)} public theorems"
            " · statements only, proofs omitted)"
        )
        header = (
            "-- Generated by scripts/gen_lean_viewer.py from"
            f" lean/{fname} (sha256 {digest}). Statements only; see the file for proofs."
        )
        code = header + "\n\n" + "\n\n".join(statements)
        suites[key] = {"label": label, "code": code}
    return suites


def render_block(suites: dict[str, dict[str, str]]) -> str:
    """Render the marker-delimited ``const LEAN_SUITES = <json>;`` block."""
    js = json.dumps(suites, ensure_ascii=False, indent=2)
    # Escape HTML script-breaking sequences and JS line separators cleanly.
    js = (
        js.replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        INDENT + BEGIN_MARKER + "\n"
        + INDENT + "const LEAN_SUITES = " + js + ";\n"
        + INDENT + END_MARKER
    )


LEAN_SUITES_RE = re.compile(
    r"^(?P<indent>[ \t]*)const LEAN_SUITES = \{.*?^(?P=indent)\};[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def replace_block(html: str, block: str) -> str:
    """Replace the existing LEAN_SUITES block (marker- or brace-delimited)."""
    lines = html.split("\n")
    begin_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if begin_idx is None and BEGIN_MARKER in line:
            begin_idx = idx
        elif begin_idx is not None and end_idx is None and END_MARKER in line:
            end_idx = idx
    if begin_idx is not None and end_idx is not None and end_idx > begin_idx:
        new_lines = lines[:begin_idx] + block.split("\n") + lines[end_idx + 1:]
        return "\n".join(new_lines)
    match = LEAN_SUITES_RE.search(html)
    if match is None:
        raise GeneratorError("could not locate the const LEAN_SUITES block in the HTML")
    return html[: match.start()] + block + html[match.end():]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the LEAN_SUITES block in docs/index.html from lean/*.lean."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the block in the HTML differs from what would be generated",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="path to docs/index.html (default: <repo>/docs/index.html)",
    )
    parser.add_argument(
        "--lean-dir",
        default=None,
        help="path to the lean/ directory (default: <repo>/lean)",
    )
    args = parser.parse_args(argv)

    html_path = Path(args.html) if args.html else DEFAULT_HTML
    lean_dir = Path(args.lean_dir) if args.lean_dir else DEFAULT_LEAN_DIR

    try:
        html_bytes = _read_bytes_safe(html_path, "html")
        html = _decode_utf8_safe(html_bytes, "html")
        suites = build_suites(lean_dir)
        block = render_block(suites)
        new_html = replace_block(html, block)
    except (GeneratorError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if new_html == html:
            total = sum(len(re.findall(r"(?m)^theorem p", s["code"])) for s in suites.values())
            print(
                f"OK: {html_path} LEAN_SUITES block is up to date "
                f"({len(suites)} suites, {total} theorem statements)"
            )
            return 0
        print(
            f"DRIFT: the LEAN_SUITES block in {html_path} differs from what "
            f"scripts/gen_lean_viewer.py would generate from {lean_dir}; "
            "run `python3 scripts/gen_lean_viewer.py` to regenerate",
            file=sys.stderr,
        )
        return 1

    if new_html != html:
        try:
            _write_bytes_atomic(html_path, new_html.encode("utf-8"), "html")
        except (GeneratorError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        total = sum(len(re.findall(r"(?m)^theorem p", s["code"])) for s in suites.values())
        print(f"wrote {html_path}: LEAN_SUITES regenerated ({len(suites)} suites, {total} theorem statements)")
    else:
        print(f"{html_path}: LEAN_SUITES block already up to date; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
