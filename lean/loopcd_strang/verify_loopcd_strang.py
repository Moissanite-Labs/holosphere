#!/usr/bin/env python3
"""
Fail-Closed Pure-Q + Lean 4.36.0 Verifier & Self-Bound Certificate Gate for
`LoopCDStrang` (Spherical Chord Self-Damping Gap, Palindromic Strang Quadratic BCH
Cancellation, Acute-Ball `9/25` & `81/625` Floors, and Commutative Exact Collapse).

Zero IEEE-754 floats in AST, zero `ast.Div` or negative-exponent `ast.Pow` operators,
zero bare asserts, zero external dependencies, and 100% on-disk SHA-256 binding of all
8 Lean package files (`BOUND_PACKAGE_FILES`).
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


MAX_FILE_BYTES: int = 2_097_152
MAX_JSON_DEPTH: int = 32
MAX_JSON_INT_DIGITS: int = 256
PURE_CONSTRUCTIVE_AXIOMS: frozenset[str] = frozenset({"propext", "Quot.sound"})
EXPLICIT_BANNED_AXIOMS: frozenset[str] = frozenset(
    {"Classical.choice", "sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"}
)
EXPECTED_PUBLIC_THEOREMS: int = 38
ALLOWED_MATH_ATTRS: frozenset[str] = frozenset({"gcd", "isqrt"})
ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "argparse",
        "ast",
        "fractions",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "subprocess",
        "sys",
        "tempfile",
        "tomllib",
        "typing",
    }
)
ALLOWED_IMPORT_FROM_MODULES: frozenset[str] = frozenset(
    {"__future__", "fractions", "math", "pathlib", "typing"}
)
FORBIDDEN_DYNAMIC_CALLS: frozenset[str] = frozenset(
    {
        "Decimal",
        "__import__",
        "bytearray",
        "compile",
        "complex",
        "delattr",
        "dir",
        "divmod",
        "eval",
        "exec",
        "float",
        "format",
        "getattr",
        "globals",
        "hasattr",
        "literal_eval",
        "locals",
        "memoryview",
        "pow",
        "round",
        "setattr",
        "type",
        "vars",
    }
)
FORBIDDEN_RATIONAL_METHODS: frozenset[str] = frozenset(
    {
        "as_integer_ratio",
        "cast",
        "conjugate",
        "format",
        "format_map",
        "from_decimal",
        "from_float",
        "fromhex",
        "hex",
        "imag",
        "literal_eval",
        "modules",
        "real",
        "truediv",
        "unpack",
    }
)
ALLOWED_DUNDER_NAMES: frozenset[str] = frozenset({"__file__", "__name__"})
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENT_PRECEDING_RE = re.compile(r"[A-Za-z0-9_']")
_RAW_STR_START_RE = re.compile(r'r(#*)"')

BOUND_PACKAGE_FILES: tuple[str, ...] = (
    "AxiomAudit.lean",
    "LoopCDStrang.lean",
    "LoopCDStrang/IntPoly.lean",
    "LoopCDStrang/QuatStrang.lean",
    "LoopCDStrang/SphereChord.lean",
    "lake-manifest.json",
    "lakefile.toml",
    "lean-toolchain",
)

EXPECTED_AXIOM_AUDIT_LINES: tuple[str, ...] = (
    "'LoopCDStrang.evalPoly_norm' depends on axioms: [propext]",
    "'LoopCDStrang.poly_eq_of_norm_eq' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_const' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_varT' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_neg' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_add' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_sub' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_scale' depends on axioms: [propext]",
    "'LoopCDStrang.Jet3.eval_mul' depends on axioms: [propext]",
    "'LoopCDStrang.int_sq_nonneg' depends on axioms: [propext]",
    "'LoopCDStrang.int_sq_pos_of_pos' depends on axioms: [propext]",
    "'LoopCDStrang.stereo_unit_circle_exact' depends on axioms: [propext]",
    "'LoopCDStrang.loopcd_denom_pos' depends on axioms: [propext]",
    "'LoopCDStrang.loopcd_tan_alpha_trig_eq_rational' depends on axioms: [propext]",
    "'LoopCDStrang.loopcd_raw_norm_sq_exact' depends on axioms: [propext]",
    "'LoopCDStrang.thm1_loopcd_chord_saturation_gap_exact' depends on axioms: [propext]",
    "'LoopCDStrang.loopcd_tan_alpha_strict_mono_omega' depends on axioms: [propext]",
    "'LoopCDStrang.slerp_wraps_antipode_at_90deg_vs_loopcd' does not depend on any axioms",
    "'LoopCDStrang.quat_mul_assoc' depends on axioms: [propext]",
    "'LoopCDStrang.palindromic_conjugation_identity' depends on axioms: [propext]",
    "'LoopCDStrang.quat_norm_sq_mul' depends on axioms: [propext]",
    "'LoopCDStrang.normSq3_nonneg' depends on axioms: [propext]",
    "'LoopCDStrang.cayley_su2_unit_norm_exact' depends on axioms: [propext]",
    "'LoopCDStrang.cayley_compose_su2_exact' depends on axioms: [propext, Quot.sound]",
    "'LoopCDStrang.rightStrangDefectJet_eval_eq' depends on axioms: [propext]",
    "'LoopCDStrang.leftOrderedDefectJet_eval_eq' depends on axioms: [propext]",
    "'LoopCDStrang.thm2a_right_strang_bch_quadratic_zero' depends on axioms: [propext]",
    "'LoopCDStrang.thm2b_left_ordered_bch_quadratic_defect_nonzero' depends on axioms: [propext]",
    "'LoopCDStrang.thm2c_dyadic_su2_chordal_hierarchy_witness' does not depend on any axioms",
    "'LoopCDStrang.acute_ball_cayley_denom_sos_identity' depends on axioms: [propext]",
    "'LoopCDStrang.acute_ball_closed_under_unit_omega' depends on axioms: [propext]",
    "'LoopCDStrang.thm3a_acute_ball_spinor_denom_ge_9_25' depends on axioms: [propext, Quot.sound]",
    "'LoopCDStrang.thm3b_acute_ball_cayley_denom_ge_81_625' depends on axioms: [propext, Quot.sound]",
    "'LoopCDStrang.thm3c_acute_ball_81_625_tight' does not depend on any axioms",
    "'LoopCDStrang.cayley_comp_commutator_exact' depends on axioms: [propext]",
    "'LoopCDStrang.thm4a_commuting_cayley_quat_commute' depends on axioms: [propext, Quot.sound]",
    "'LoopCDStrang.thm4b_commutative_left_right_collapse' depends on axioms: [propext, Quot.sound]",
    "'LoopCDStrang.thm4c_collinear_left_right_cayley_sum_exact' depends on axioms: [propext, Quot.sound]",
)

_FORBIDDEN_LEAN_CODE_RE = re.compile(
    r"[«»]"
    r"|@\["
    r"|\battribute[ \t]*\["
    r"|\br#*\""
    r"|#[ \t]*(?!print[ \t]+axioms\b)[A-Za-z_]"
    r"|\b(?:sorry|admit|sorryAx|unsafe|native_decide|ofReduceBool|trustCompiler|"
    r"bv_decide|bv_check|bv_normalize|Classical|by_cases|grind|"
    r"implemented_by|extern|csimp|macro|macro_rules|"
    r"elab|elab_rules|term_elab|command_elab|tactic_elab|attr_elab|by_elab|"
    r"syntax|syntax_cat|declare_syntax_cat|"
    r"run_cmd|run_elab|run_tac|"
    r"simproc|dsimproc|simproc_decl|dsimproc_decl|builtin_simproc|builtin_dsimproc|"
    r"env_extension|register_builtin_option|register_option|"
    r"initialize|builtin_initialize|opaque|set_option|"
    r"notation|infix|infixl|infixr|prefix|postfix)\b"
    r"|\b(?:IO|System)\."
    r"|\b[msf]!"
    r"|decide[ \t]*\+native"
    r"|native[ \t]*:="
    r"|(?<!\.)\baxiom\b"
    r"|(?<!\.)\bpartial\s+(?:def|abbrev|instance)\b"
    r"|\._native\."
)
_ANY_IMPORT_RE = re.compile(r"(?:^|\n)[ \t]*(?:(?:public|private|meta)[ \t]+)*import(?![A-Za-z0-9_'])([^\n]*)")
_VALID_IMPORT_MOD_RE = re.compile(r"LoopCDStrang(?:\.[A-Za-z0-9_]+)*")
_PUB_THM_RE = re.compile(r"(?:^|\n)[ \t]*theorem[ \t]+([A-Za-z0-9_.']+)")
_PRINT_AXIOMS_RE = re.compile(r"(?:^|\n)[ \t]*#print[ \t]+axioms[ \t]+LoopCDStrang\.([A-Za-z0-9_.']+)")


class LoopCDStrangVerifyError(ValueError):
    """Typed fail-closed verification error for LoopCDStrang."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise LoopCDStrangVerifyError(code, message)


def is_hex64(val: Any) -> bool:
    return isinstance(val, str) and HEX64_RE.fullmatch(val) is not None


def qdiv(num: Fraction, den: Fraction) -> Fraction:
    require_ok(den != Fraction(0), "QDIV_ZERO_DENOM", "Zero denominator in exact rational division")
    return Fraction(num.numerator * den.denominator, num.denominator * den.numerator)


def _assert_no_symlink_ancestors(fpath: Path) -> None:
    cur = fpath.absolute()
    require_ok(not cur.is_symlink(), "SYMLINK_FORBIDDEN", f"Symlink forbidden: {cur.name}")
    for anc in cur.parents:
        require_ok(not anc.is_symlink(), "SYMLINK_FORBIDDEN", f"Ancestor symlink forbidden: {anc.name}")


def read_regular_bytes_nofollow(
    fpath: Path,
    missing_code: str,
    max_bytes: int = MAX_FILE_BYTES,
) -> bytes:
    _assert_no_symlink_ancestors(fpath)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(fpath, flags)
    except FileNotFoundError as exc:
        raise LoopCDStrangVerifyError(missing_code, f"File not found: {fpath.name}") from exc
    except OSError as exc:
        raise LoopCDStrangVerifyError("OPEN_FAILED", f"Cannot open {fpath.name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NON_REGULAR_FILE", f"Not a regular file: {fpath.name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            require_ok(total <= max_bytes, "FILE_TOO_LARGE", f"File exceeds {max_bytes} bytes: {fpath.name}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_regular_bytes_nofollow(
    out_path: Path,
    data: bytes,
    max_bytes: int = MAX_FILE_BYTES,
) -> None:
    require_ok(len(data) <= max_bytes, "FILE_TOO_LARGE", f"Output exceeds {max_bytes} bytes: {out_path.name}")
    _assert_no_symlink_ancestors(out_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(out_path, flags, 0o644)
    except OSError as exc:
        raise LoopCDStrangVerifyError("WRITE_OPEN_FAILED", f"Cannot open {out_path.name} for write: {exc}") from exc
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NON_REGULAR_FILE", f"Destination is not a regular file: {out_path.name}")
        rem = data
        while rem:
            rem = rem[os.write(fd, rem) :]
    finally:
        os.close(fd)


def decode_utf8_strict(raw_bytes: bytes, code: str, label: str) -> str:
    require_ok(not raw_bytes.startswith(b"\xef\xbb\xbf"), code, f"UTF-8 BOM forbidden in {label}")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoopCDStrangVerifyError(code, f"Invalid UTF-8 in {label}") from exc


def _is_frac_type_ann(node: ast.AST | None, fraction_aliases: set[str], fraction_mod_aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in fraction_aliases
    if isinstance(node, ast.Attribute):
        if node.attr == "Fraction":
            if isinstance(node.value, ast.Name):
                return node.value.id in fraction_mod_aliases
    return False


def _is_frac_expr_ast(
    node: ast.AST,
    fraction_aliases: set[str],
    fraction_mod_aliases: set[str],
    seed_vars: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in seed_vars
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id in fraction_aliases
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "Fraction":
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id in fraction_mod_aliases
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            return _is_frac_expr_ast(node.operand, fraction_aliases, fraction_mod_aliases, seed_vars)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            if _is_frac_expr_ast(node.left, fraction_aliases, fraction_mod_aliases, seed_vars):
                return _is_frac_expr_ast(node.right, fraction_aliases, fraction_mod_aliases, seed_vars)
    return False


def _iter_scope_nodes(stmts: list[ast.stmt]) -> list[ast.AST]:
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(reversed(stmts))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out.append(cur)
        stack.extend(reversed(list(ast.iter_child_nodes(cur))))
    return out


def _scope_proven_frac_vars(
    stmts: list[ast.stmt],
    init_vars: set[str],
    fraction_aliases: set[str],
    fraction_mod_aliases: set[str],
) -> frozenset[str]:
    cand: set[str] = set(init_vars)
    tainted: set[str] = set()
    clean_store_ids: set[int] = set()
    scope_nodes = _iter_scope_nodes(stmts)
    for sub in scope_nodes:
        if isinstance(sub, ast.Assign):
            if len(sub.targets) == 1:
                tgt0 = sub.targets[0]
                if isinstance(tgt0, ast.Name):
                    if _is_frac_expr_ast(sub.value, fraction_aliases, fraction_mod_aliases, cand):
                        cand.add(tgt0.id)
                        clean_store_ids.add(id(tgt0))
        elif isinstance(sub, ast.AnnAssign):
            if isinstance(sub.target, ast.Name):
                if sub.value is not None:
                    if _is_frac_expr_ast(sub.value, fraction_aliases, fraction_mod_aliases, cand):
                        cand.add(sub.target.id)
                        clean_store_ids.add(id(sub.target))
    for sub in scope_nodes:
        if isinstance(sub, ast.Name):
            if isinstance(sub.ctx, ast.Store):
                if id(sub) not in clean_store_ids:
                    tainted.add(sub.id)
    return frozenset(cand - tainted)


def _is_safe_int_ast(node: ast.AST, proven_frac_vars: frozenset[str] = frozenset()) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_int_ast(node.operand, proven_frac_vars)
    if isinstance(node, ast.Attribute) and node.attr in ("numerator", "denominator"):
        if isinstance(node.value, ast.Name):
            return node.value.id in proven_frac_vars
        return False
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            return _is_safe_int_ast(node.left, proven_frac_vars) and _is_safe_int_ast(node.right, proven_frac_vars)
        if isinstance(node.op, ast.LShift):
            return _is_safe_int_ast(node.left, proven_frac_vars) and (
                isinstance(node.right, ast.Name) or _is_safe_int_ast(node.right, proven_frac_vars)
            )
    return False


def _collect_pure_type_ids(node: ast.AST | None, out: set[int]) -> None:
    if node is None:
        return
    if isinstance(node, (ast.Name, ast.Constant)):
        out.add(id(node))
    elif isinstance(node, ast.Attribute):
        out.add(id(node))
        _collect_pure_type_ids(node.value, out)
    elif isinstance(node, ast.Subscript):
        _collect_pure_type_ids(node.value, out)
        _collect_pure_type_ids(node.slice, out)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            _collect_pure_type_ids(elt, out)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        _collect_pure_type_ids(node.left, out)
        _collect_pure_type_ids(node.right, out)


def verify_pure_q_ast(source_text: str) -> dict[str, Any]:
    """
    Fail-closed AST guard rejecting `ast.Div` (`/`), `ast.FloorDiv` (`//`), negative/non-int
    `ast.Pow` (`**`), format-spec f-strings, `float`/`complex` literals, forbidden dynamic calls
    or bare Name references (`float`, `Decimal`, `getattr`, `eval`, `format`, `divmod`, etc.),
    all dunder attributes (`.__getattribute__`, `.__class__`, `.__dict__`, `.__truediv__`, etc.),
    `.real`/`.imag`/`.from_float`/`.from_decimal`, wildcard/relative/forbidden imports, `math.*`
    outside `gcd`/`isqrt` (including rebinding), `Fraction` rebinding, custom `.numerator`/
    `.denominator` definitions, and unproven `.numerator`/`.denominator` receivers in `Fraction()`.
    """
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError) as exc:
        raise LoopCDStrangVerifyError("AST_SYNTAX_ERROR", "Python syntax error in AST check") from exc
    math_aliases: set[str] = {"math"}
    json_aliases: set[str] = {"json"}
    fraction_mod_aliases: set[str] = {"fractions"}
    fraction_aliases: set[str] = {"Fraction"}
    stored_names: set[str] = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    raising_fn_names: set[str] = set()
    int_guard_fn_names: set[str] = set()
    for top_stmt in tree.body:
        if isinstance(top_stmt, ast.FunctionDef):
            if top_stmt.name not in stored_names:
                if len(top_stmt.body) == 1:
                    if isinstance(top_stmt.body[0], ast.Raise):
                        raising_fn_names.add(top_stmt.name)
                for sub_call in ast.walk(top_stmt):
                    if isinstance(sub_call, ast.Call):
                        if isinstance(sub_call.func, ast.Name):
                            if sub_call.func.id == "require_ok":
                                int_guard_fn_names.add(top_stmt.name)
    allowed_frac_ref_ids: set[int] = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    allowed_mod_attr_ids: set[int] = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    allowed_typecheck_ids: set[int] = set()
    allowed_type_call_ids: set[int] = set()
    guarded_json_loads_ids: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            _collect_pure_type_ids(n.annotation, allowed_frac_ref_ids)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_pure_type_ids(n.returns, allowed_frac_ref_ids)
        elif isinstance(n, ast.AnnAssign):
            _collect_pure_type_ids(n.annotation, allowed_frac_ref_ids)
        elif isinstance(n, ast.Compare):
            if len(n.ops) == 1:
                if isinstance(n.ops[0], (ast.Is, ast.IsNot)):
                    if isinstance(n.left, ast.Call):
                        if isinstance(n.left.func, ast.Name):
                            if n.left.func.id == "type":
                                if len(n.left.args) == 1:
                                    if len(n.left.keywords) == 0:
                                        allowed_typecheck_ids.add(id(n.left.func))
                                        allowed_type_call_ids.add(id(n.left))
        elif isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id == "isinstance":
                    if len(n.args) == 2:
                        if len(n.keywords) == 0:
                            cls_arg = n.args[1]
                            if isinstance(cls_arg, ast.Name):
                                if cls_arg.id in ("float", "complex"):
                                    allowed_typecheck_ids.add(id(cls_arg))
                            elif isinstance(cls_arg, ast.Tuple):
                                for elt in cls_arg.elts:
                                    if isinstance(elt, ast.Name):
                                        if elt.id in ("float", "complex"):
                                            allowed_typecheck_ids.add(id(elt))
            elif isinstance(n.func, ast.Attribute):
                if n.func.attr == "loads":
                    good_kw_names: set[str] = set()
                    for kw in n.keywords:
                        if kw.arg in ("parse_float", "parse_constant"):
                            if isinstance(kw.value, ast.Name):
                                if kw.value.id in raising_fn_names:
                                    good_kw_names.add(kw.arg)
                        elif kw.arg == "parse_int":
                            if isinstance(kw.value, ast.Name):
                                if kw.value.id in int_guard_fn_names:
                                    good_kw_names.add("parse_int")
                    if "parse_float" in good_kw_names:
                        if "parse_constant" in good_kw_names:
                            if "parse_int" in good_kw_names:
                                guarded_json_loads_ids.add(id(n.func))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                require_ok(root in ALLOWED_IMPORT_ROOTS, "AST_FORBIDDEN_IMPORT", f"Forbidden import {alias.name}")
                require_ok(alias.asname != "Fraction", "AST_FRACTION_REBIND_FORBIDDEN", "Aliasing import to Fraction forbidden")
                if root == "math":
                    math_aliases.add(alias.asname or alias.name)
                elif root == "json":
                    json_aliases.add(alias.asname or alias.name)
                elif root == "fractions":
                    fraction_mod_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            require_ok(node.level == 0, "AST_REL_IMPORT_FORBIDDEN", "Relative imports forbidden")
            mod_root = (node.module or "").split(".")[0]
            require_ok(
                mod_root in ALLOWED_IMPORT_FROM_MODULES,
                "AST_FORBIDDEN_IMPORT",
                f"Forbidden import from {node.module}",
            )
            for alias in node.names:
                require_ok(alias.name != "*", "AST_WILDCARD_IMPORT", f"Wildcard import from {node.module} forbidden")
                if mod_root != "fractions":
                    require_ok(
                        alias.asname != "Fraction",
                        "AST_FRACTION_REBIND_FORBIDDEN",
                        "Aliasing non-Fraction symbol to Fraction forbidden",
                    )
                if mod_root == "__future__":
                    require_ok(alias.name == "annotations", "AST_FORBIDDEN_IMPORT", f"Forbidden __future__ symbol {alias.name}")
                elif mod_root == "math":
                    require_ok(
                        alias.name in ALLOWED_MATH_ATTRS,
                        "AST_FORBIDDEN_MATH_SYM",
                        f"Forbidden math symbol {alias.name}",
                    )
                elif mod_root == "fractions":
                    require_ok(
                        alias.name == "Fraction",
                        "AST_FORBIDDEN_FRACTIONS_SYM",
                        f"Forbidden fractions symbol {alias.name}",
                    )
                    fraction_aliases.add(alias.asname or alias.name)
                elif mod_root == "pathlib":
                    require_ok(alias.name == "Path", "AST_FORBIDDEN_IMPORT", f"Forbidden pathlib symbol {alias.name}")
                elif mod_root == "typing":
                    require_ok(alias.name == "Any", "AST_FORBIDDEN_IMPORT", f"Forbidden typing symbol {alias.name}")

    call_proven_vars: dict[int, frozenset[str]] = {}
    mod_frac_vars = _scope_proven_frac_vars(tree.body, set(), fraction_aliases, fraction_mod_aliases)
    for m_node in _iter_scope_nodes(tree.body):
        if isinstance(m_node, ast.Call):
            call_proven_vars[id(m_node)] = mod_frac_vars
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            require_ok(
                node.name not in ("numerator", "denominator"),
                "AST_CUSTOM_NUMERATOR_DEF",
                f"Custom {node.name} definition forbidden",
            )
            require_ok(
                node.name not in fraction_aliases,
                "AST_FRACTION_REBIND_FORBIDDEN",
                f"Shadowing {node.name} via def is forbidden",
            )
            fn_args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            for arg_node in fn_args:
                require_ok(
                    arg_node.arg not in fraction_aliases,
                    "AST_FRACTION_REBIND_FORBIDDEN",
                    f"Shadowing {arg_node.arg} via parameter is forbidden",
                )
            init_fvars = {
                a.arg for a in fn_args if _is_frac_type_ann(a.annotation, fraction_aliases, fraction_mod_aliases)
            }
            fn_frac_vars = _scope_proven_frac_vars(node.body, init_fvars, fraction_aliases, fraction_mod_aliases)
            for f_sub in _iter_scope_nodes(node.body):
                if isinstance(f_sub, ast.Call):
                    call_proven_vars[id(f_sub)] = fn_frac_vars
        elif isinstance(node, ast.ClassDef):
            require_ok(
                node.name not in fraction_aliases,
                "AST_FRACTION_REBIND_FORBIDDEN",
                f"Shadowing {node.name} via class is forbidden",
            )
            for c_stmt in node.body:
                if isinstance(c_stmt, ast.Assign):
                    for tgt in c_stmt.targets:
                        if isinstance(tgt, ast.Name):
                            require_ok(
                                tgt.id not in ("numerator", "denominator"),
                                "AST_CUSTOM_NUMERATOR_DEF",
                                f"Custom class attribute {tgt.id} forbidden",
                            )
                elif isinstance(c_stmt, ast.AnnAssign):
                    if isinstance(c_stmt.target, ast.Name):
                        require_ok(
                            c_stmt.target.id not in ("numerator", "denominator"),
                            "AST_CUSTOM_NUMERATOR_DEF",
                            f"Custom class attribute {c_stmt.target.id} forbidden",
                        )
        elif isinstance(node, ast.ExceptHandler):
            require_ok(
                node.name not in fraction_aliases,
                "AST_FRACTION_REBIND_FORBIDDEN",
                "Shadowing Fraction in except handler is forbidden",
            )

    for node in ast.walk(tree):
        require_ok(
            not isinstance(node, (ast.Div, ast.FloorDiv, ast.Mod)),
            "AST_DIV_FORBIDDEN",
            "Use qdiv(a, b) or math.gcd instead of '/', '//', or '%' operator",
        )
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            exp_node = node.right if isinstance(node, ast.BinOp) else node.value
            require_ok(
                isinstance(exp_node, ast.Constant)
                and type(exp_node.value) is int
                and exp_node.value >= 0,
                "AST_POW_EXP_FORBIDDEN",
                "Exponent in '**' must be a non-negative integer literal",
            )
        elif isinstance(node, ast.FormattedValue):
            require_ok(
                node.format_spec is None,
                "AST_FORMAT_SPEC_FORBIDDEN",
                "Format specifiers in f-strings forbidden",
            )
        elif isinstance(node, ast.Constant):
            require_ok(
                not isinstance(node.value, (float, complex)),
                "AST_FLOAT_LITERAL",
                f"Forbidden float/complex literal {node.value!r}",
            )
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                require_ok(
                    node.id in ALLOWED_DUNDER_NAMES,
                    "AST_BUILTINS_FORBIDDEN",
                    f"Forbidden dunder reference {node.id}",
                )
            if node.id == "re":
                require_ok(
                    not isinstance(node.ctx, ast.Store),
                    "AST_FLOAT_CALL",
                    "Rebinding module re is forbidden",
                )
            if node.id in FORBIDDEN_DYNAMIC_CALLS:
                require_ok(
                    id(node) in allowed_typecheck_ids,
                    "AST_FLOAT_CALL",
                    f"Forbidden reference to {node.id}",
                )
            if node.id in math_aliases:
                require_ok(
                    id(node) in allowed_mod_attr_ids,
                    "AST_FORBIDDEN_MATH_ATTR",
                    f"Forbidden non-attribute reference to {node.id}",
                )
            if node.id in json_aliases:
                require_ok(
                    id(node) in allowed_mod_attr_ids,
                    "AST_FORBIDDEN_JSON_USE",
                    f"Forbidden non-attribute reference to {node.id}",
                )
            if node.id in fraction_mod_aliases:
                require_ok(
                    id(node) in allowed_mod_attr_ids,
                    "AST_FRACTION_REBIND_FORBIDDEN",
                    f"Forbidden non-attribute reference to {node.id}",
                )
            if node.id in fraction_aliases:
                require_ok(
                    id(node) in allowed_frac_ref_ids,
                    "AST_FRACTION_REBIND_FORBIDDEN",
                    f"Forbidden non-call reference to {node.id}",
                )
        elif isinstance(node, ast.Attribute):
            is_super_init = False
            if node.attr == "__init__":
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "super":
                            is_super_init = True
            if not is_super_init:
                require_ok(
                    not node.attr.startswith("__"),
                    "AST_FROM_FLOAT_FORBIDDEN",
                    f"Forbidden dunder attribute .{node.attr}",
                )
            require_ok(
                node.attr not in FORBIDDEN_RATIONAL_METHODS,
                "AST_FROM_FLOAT_FORBIDDEN",
                f"Forbidden attribute .{node.attr}",
            )
            if node.attr == "Fraction":
                is_valid_frac_mod = False
                if isinstance(node.value, ast.Name):
                    if node.value.id in fraction_mod_aliases:
                        is_valid_frac_mod = True
                require_ok(
                    is_valid_frac_mod,
                    "AST_FRACTION_REBIND_FORBIDDEN",
                    "Receiver of .Fraction must be an imported fractions module",
                )
                require_ok(
                    id(node) in allowed_frac_ref_ids,
                    "AST_FRACTION_REBIND_FORBIDDEN",
                    "Forbidden non-call reference to .Fraction",
                )
            if isinstance(node.value, ast.Name):
                if node.value.id in math_aliases:
                    require_ok(
                        node.attr in ALLOWED_MATH_ATTRS,
                        "AST_FORBIDDEN_MATH_ATTR",
                        f"Forbidden math.{node.attr}",
                    )
                if node.value.id in json_aliases:
                    require_ok(
                        node.attr in ("loads", "dumps", "JSONDecodeError"),
                        "AST_FORBIDDEN_JSON_USE",
                        f"Forbidden json.{node.attr}",
                    )
                    if node.attr == "loads":
                        require_ok(
                            id(node) in guarded_json_loads_ids,
                            "AST_JSON_LOADS_UNGUARDED",
                            "json.loads() must pass safe parse_float and parse_constant",
                        )
        elif isinstance(node, ast.Call):
            require_ok(
                isinstance(node.func, (ast.Name, ast.Attribute)),
                "AST_DYNAMIC_CALL_FUNC",
                "Indirect or computed call target forbidden",
            )
            fn_name = ""
            is_frac_call = False
            if isinstance(node.func, ast.Name):
                if id(node) not in allowed_type_call_ids:
                    fn_name = node.func.id
                if node.func.id in fraction_aliases:
                    is_frac_call = True
            elif isinstance(node.func, ast.Attribute):
                is_re_compile = False
                if node.func.attr == "compile":
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == "re":
                            is_re_compile = True
                if not is_re_compile:
                    fn_name = node.func.attr
                if fn_name == "Fraction":
                    is_frac_call = True
            require_ok(fn_name not in FORBIDDEN_DYNAMIC_CALLS, "AST_FLOAT_CALL", f"Forbidden call to {fn_name}()")
            if is_frac_call:
                require_ok(len(node.keywords) == 0, "AST_FRACTION_KW", "Keyword args to Fraction() forbidden")
                require_ok(1 <= len(node.args) <= 2, "AST_FRACTION_ARITY", "Fraction() must take 1 or 2 integer args")
                scope_fvars = call_proven_vars.get(id(node), frozenset())
                for arg in node.args:
                    require_ok(
                        _is_safe_int_ast(arg, scope_fvars),
                        "AST_FRACTION_ARG_UNSAFE",
                        "Fraction() arguments must be proven integer AST expressions",
                    )
    return {"pure_q_ast_clean": True}


def _match_lean_char_literal_end(src: str, pos: int) -> int:
    """
    If `src[pos] == "'"` starts a valid Lean 4 character literal (`'c'` or `'\\...'`),
    return the index immediately after the closing `'`, else `-1`.
    Identifier primes (`x'`, `ih'`) are excluded via preceding-character check.
    """
    n = len(src)
    if pos > 0 and _IDENT_PRECEDING_RE.fullmatch(src[pos - 1]) is not None:
        return -1
    if pos + 2 >= n:
        return -1
    nxt = src[pos + 1]
    if nxt in ("\n", "'"):
        return -1
    if nxt != "\\":
        return pos + 3 if src[pos + 2] == "'" else -1
    esc = src[pos + 2]
    close_idx = -1
    if esc in ("\\", "'", '"', "n", "t", "r", "0"):
        close_idx = pos + 3
    elif esc == "x":
        close_idx = pos + 5
    elif esc == "u":
        close_idx = pos + 7
    if close_idx > 0 and src[close_idx : close_idx + 1] == "'":
        return close_idx + 1
    return -1


def strip_lean_comments_and_strings(src: str) -> str:
    """
    Blank Lean 4 line comments (`--`), nested block comments (`/- ... -/`),
    character literals (`'"'`, `'\\''`, etc.), and string literals (`"..."`)
    while preserving raw raw-string markers (`r"..."`) so `_FORBIDDEN_LEAN_CODE_RE` catches them.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    in_str = False
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
        elif in_str:
            if ch == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
            elif ch == '"':
                in_str = False
                out.append(" ")
                i += 1
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
        elif ch == "-" and nxt == "-":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
        elif ch == "/" and nxt == "-":
            depth = 1
            out.append("  ")
            i += 2
        elif ch == "'":
            end_pos = _match_lean_char_literal_end(src, i)
            if end_pos > i:
                out.append(" " * (end_pos - i))
                i = end_pos
            else:
                out.append(ch)
                i += 1
        elif ch == "r":
            m_raw = _RAW_STR_START_RE.match(src, i) if _IDENT_PRECEDING_RE.fullmatch(src[i - 1 : i]) is None else None
            if m_raw is not None:
                hashes = m_raw.group(1)
                closer = '"' + hashes
                end_raw = src.find(closer, m_raw.end())
                require_ok(end_raw != -1, "UNTERMINATED_STRING_LITERAL", "Unterminated raw string literal")
                out.append(m_raw.group(0))
                out.append(" " * (end_raw + len(closer) - m_raw.end()))
                i = end_raw + len(closer)
            else:
                out.append(ch)
                i += 1
        elif ch == '"':
            in_str = True
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    require_ok(depth == 0, "UNTERMINATED_BLOCK_COMMENT", "Unterminated Lean block comment")
    require_ok(not in_str, "UNTERMINATED_STRING_LITERAL", "Unterminated Lean string literal")
    return "".join(out)


def dot3_q(a: tuple[Fraction, Fraction, Fraction], b: tuple[Fraction, Fraction, Fraction]) -> Fraction:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3_q(
    a: tuple[Fraction, Fraction, Fraction],
    b: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def quat_conj_q(q: tuple[Fraction, Fraction, Fraction, Fraction]) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_mul_q(
    p: tuple[Fraction, Fraction, Fraction, Fraction],
    q: tuple[Fraction, Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    )


def quat_dist_sq_q(
    p: tuple[Fraction, Fraction, Fraction, Fraction],
    q: tuple[Fraction, Fraction, Fraction, Fraction],
) -> Fraction:
    return sum(((x - y) * (x - y) for x, y in zip(p, q)), Fraction(0))


def cayley_su2_q(w: tuple[Fraction, Fraction, Fraction]) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    nsq = dot3_q(w, w)
    den = Fraction(1) + nsq
    return (
        qdiv(Fraction(1) - nsq, den),
        qdiv(Fraction(2) * w[0], den),
        qdiv(Fraction(2) * w[1], den),
        qdiv(Fraction(2) * w[2], den),
    )


def cayley_compose_q(
    a: tuple[Fraction, Fraction, Fraction],
    b: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    na = dot3_q(a, a)
    nb = dot3_q(b, b)
    den = Fraction(1) + na * nb - Fraction(2) * dot3_q(a, b)
    require_ok(den > Fraction(0), "CAYLEY_DENOM_NONPOS", "Cayley composition hit antipodal pole")
    ca = Fraction(1) - nb
    cb = Fraction(1) - na
    cr = cross3_q(a, b)
    return (
        qdiv(ca * a[0] + cb * b[0] + Fraction(2) * cr[0], den),
        qdiv(ca * a[1] + cb * b[1] + Fraction(2) * cr[1], den),
        qdiv(ca * a[2] + cb * b[2] + Fraction(2) * cr[2], den),
    )


def verify_thm1_chord_saturation_q(
    gap_fn: Any = None,
    tan_fn: Any = None,
    stereo_fn: Any = None,
    s_grid: tuple[Fraction, ...] = (Fraction(1, 20), Fraction(1, 5), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(5)),
    om_grid: tuple[Fraction, ...] = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(8)),
) -> dict[str, Any]:
    if tan_fn is None:
        tan_fn = lambda s, om: qdiv(Fraction(2) * om * s, Fraction(1) + (Fraction(1) + Fraction(2) * om) * s * s)
    if gap_fn is None:
        gap_fn = lambda s, om: qdiv(Fraction(1) + s * s, s * (Fraction(1) + (Fraction(1) + Fraction(2) * om) * s * s))
    if stereo_fn is None:
        stereo_fn = lambda s: (qdiv(Fraction(1) - s * s, Fraction(1) + s * s), qdiv(Fraction(2) * s, Fraction(1) + s * s))

    pairs_checked = 0
    mono_steps = 0
    for s in s_grid:
        cos_t, sin_t = stereo_fn(s)
        require_ok(cos_t * cos_t + sin_t * sin_t == Fraction(1), "STEREO_CIRCLE_FAIL", "cos^2 + sin^2 != 1")
        prev_tan: Fraction | None = None
        for om in om_grid:
            t_rat = tan_fn(s, om)
            t_trig = qdiv(om * sin_t, Fraction(1) + om * (Fraction(1) - cos_t))
            require_ok(t_rat == t_trig, "TAN_ALPHA_LAW_FAIL", f"Trig != rational at {(s, om)}")
            gap = gap_fn(s, om)
            require_ok(qdiv(Fraction(1), s) - t_rat == gap, "SAT_GAP_IDENTITY_FAIL", "cot(theta_half) - tan(alpha) != gap")
            require_ok(gap > Fraction(0), "SAT_GAP_NONPOS", "Saturation gap must be strictly > 0")
            if prev_tan is not None:
                require_ok(t_rat > prev_tan, "OMEGA_MONO_FAIL", "tan(alpha) must strictly increase in omega")
                mono_steps += 1
            prev_tan = t_rat
            pairs_checked += 1

    require_ok(tan_fn(Fraction(1), Fraction(1)) == Fraction(1, 2), "WITNESS_90DEG_FAIL", "Expected tan_alpha(1,1) == 1/2")
    return {"grid_pairs_verified": pairs_checked, "strict_monotonicity_steps": mono_steps}


def _comp_int(
    x: tuple[int, int, int],
    dx: int,
    y: tuple[int, int, int],
    dy: int,
) -> tuple[tuple[int, int, int], int]:
    nx = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    ny = y[0] * y[0] + y[1] * y[1] + y[2] * y[2]
    c1 = dx * (dy * dy - ny)
    c2 = dy * (dx * dx - nx)
    c3 = 2 * dx * dy
    cr = (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])
    num = (
        c1 * x[0] + c2 * y[0] + c3 * cr[0],
        c1 * x[1] + c2 * y[1] + c3 * cr[1],
        c1 * x[2] + c2 * y[2] + c3 * cr[2],
    )
    den = dx * dx * dy * dy + nx * ny - 2 * dx * dy * (x[0] * y[0] + x[1] * y[1] + x[2] * y[2])
    return num, den


def verify_thm2_palindromic_strang_bch_q(
    right_defect_fn: Any = None,
    left_defect_fn: Any = None,
    su2_err_pair_fn: Any = None,
) -> dict[str, Any]:
    def _default_defects(
        t: int, s_scale: int, u: int, v: int, a: tuple[int, int, int], b: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        x = (-t * a[0], -t * a[1], -t * a[2])
        y = (t * b[0], t * b[1], t * b[2])
        nd, dd = _comp_int(x, s_scale, y, s_scale)
        und = (u * nd[0], u * nd[1], u * nd[2])
        nr, dr = _comp_int(y, s_scale, und, v * dd)
        nl, dl = _comp_int(und, v * dd, y, s_scale)
        ng = (t * ((v + u) * b[0] - u * a[0]), t * ((v + u) * b[1] - u * a[1]), t * ((v + u) * b[2] - u * a[2]))
        def_r = (v * s_scale * nr[0] - dr * ng[0], v * s_scale * nr[1] - dr * ng[1], v * s_scale * nr[2] - dr * ng[2])
        def_l = (v * s_scale * nl[0] - dl * ng[0], v * s_scale * nl[1] - dl * ng[1], v * s_scale * nl[2] - dl * ng[2])
        return def_r, def_l

    if right_defect_fn is None:
        right_defect_fn = lambda t, s, u, v, a, b: _default_defects(t, s, u, v, a, b)[0]
    if left_defect_fn is None:
        left_defect_fn = lambda t, s, u, v, a, b: _default_defects(t, s, u, v, a, b)[1]

    samples = (
        (1, 1, 1, (1, 0, 0), (0, 1, 0)),
        (2, 3, 5, (1, 2, -1), (-2, 1, 3)),
        (3, 1, 2, (2, -1, 4), (1, 3, -2)),
    )
    cubic_checks = 0
    for s_scale, u, v, a, b in samples:
        cr = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        require_ok(right_defect_fn(0, s_scale, u, v, a, b) == (0, 0, 0), "STRANG_T0_FAIL", "DefR(0) != 0")
        for t in (2, 3, 5):
            dr_vec = right_defect_fn(t, s_scale, u, v, a, b)
            t3 = t * t * t
            for comp_val in dr_vec:
                require_ok(
                    math.gcd(comp_val, t3) == t3,
                    "STRANG_RIGHT_CUBIC_FAIL",
                    f"Right Strang defect not divisible by t^3 at t={t}",
                )
            dl_vec = left_defect_fn(t, s_scale, u, v, a, b)
            quad_coeff = -4 * (v * v) * u * (s_scale**9) * (t * t)
            rem_l = (dl_vec[0] - quad_coeff * cr[0], dl_vec[1] - quad_coeff * cr[1], dl_vec[2] - quad_coeff * cr[2])
            for comp_val in rem_l:
                require_ok(
                    math.gcd(comp_val, t3) == t3,
                    "STRANG_LEFT_QUAD_FAIL",
                    f"Left defect minus quad not divisible by t^3 at t={t}",
                )
            cubic_checks += 1

    if su2_err_pair_fn is None:
        def _default_err_pair(k: int) -> tuple[Fraction, Fraction]:
            lam = Fraction(1, 1 << k)
            w1 = (lam * Fraction(1, 5), Fraction(0), Fraction(0))
            wr = (Fraction(0), lam * Fraction(1, 4), Fraction(0))
            u1 = cayley_su2_q(w1)
            ur = cayley_su2_q(wr)
            du = quat_mul_q(quat_conj_q(u1), ur)
            u_right = quat_mul_q(ur, du)
            u_left = quat_mul_q(du, ur)
            w_lin = (Fraction(2) * wr[0] - w1[0], Fraction(2) * wr[1] - w1[1], Fraction(2) * wr[2] - w1[2])
            u_lin = cayley_su2_q(w_lin)
            return quat_dist_sq_q(u_lin, u_right), quat_dist_sq_q(u_lin, u_left)
        su2_err_pair_fn = _default_err_pair

    errs = [su2_err_pair_fn(k) for k in (0, 1, 2, 3)]
    for k_idx, (e_r, e_l) in enumerate(errs):
        require_ok(e_r > Fraction(0), "DYADIC_ER_NONPOS", f"Expected e_right > 0 at k={k_idx}")
        require_ok(e_r < e_l, "DYADIC_HIERARCHY_FAIL", f"Expected e_right < e_left at k={k_idx}")

    right_ratio = qdiv(errs[2][0], errs[3][0])
    left_ratio = qdiv(errs[2][1], errs[3][1])
    require_ok(right_ratio > Fraction(60), "RIGHT_HALVING_RATIO_LOW", f"Expected > 60x, got {right_ratio}")
    require_ok(Fraction(15) < left_ratio, "LEFT_HALVING_RATIO_LOW", f"Expected > 15x, got {left_ratio}")
    require_ok(left_ratio < Fraction(17), "LEFT_HALVING_RATIO_HIGH", f"Expected < 17x, got {left_ratio}")
    return {"cubic_divisibility_checks": cubic_checks, "dyadic_scales_verified": 4}


def verify_thm3_acute_ball_floors_q(
    spinor_denom_fn: Any = None,
    cayley_denom_fn: Any = None,
    acute_vectors: tuple[tuple[Fraction, Fraction, Fraction], ...] = (
        (Fraction(4, 5), Fraction(0), Fraction(0)),
        (Fraction(-4, 5), Fraction(0), Fraction(0)),
        (Fraction(2, 5), Fraction(3, 5), Fraction(-1, 5)),
        (Fraction(-3, 5), Fraction(0), Fraction(2, 5)),
        (Fraction(0), Fraction(0), Fraction(0)),
    ),
) -> dict[str, Any]:
    if spinor_denom_fn is None:
        spinor_denom_fn = lambda w1, wr: Fraction(1) + dot3_q(w1, wr)
    if cayley_denom_fn is None:
        cayley_denom_fn = lambda w1, wr: Fraction(1) + dot3_q(w1, w1) * dot3_q(wr, wr) + Fraction(2) * dot3_q(w1, wr)

    omegas = (Fraction(0), Fraction(1, 3), Fraction(4, 5), Fraction(1))
    checks = 0
    min_spinor = Fraction(100)
    min_cayley = Fraction(100)
    for w1 in acute_vectors:
        require_ok(dot3_q(w1, w1) <= Fraction(16, 25), "ACUTE_BALL_W1_OUT", "w1 outside 4_over_5 ball")
        for wr in acute_vectors:
            for om in omegas:
                sw1 = (om * w1[0], om * w1[1], om * w1[2])
                sd = spinor_denom_fn(sw1, wr)
                cd = cayley_denom_fn(sw1, wr)
                require_ok(sd >= Fraction(9, 25), "SPINOR_FLOOR_FAIL", f"1 + <om*w1, wr> = {sd} < 9_over_25")
                require_ok(cd >= Fraction(81, 625), "CAYLEY_FLOOR_FAIL", f"Cayley denom = {cd} < 81_over_625")
                if sd < min_spinor:
                    min_spinor = sd
                if cd < min_cayley:
                    min_cayley = cd
                checks += 1

    require_ok(min_spinor == Fraction(9, 25), "SPINOR_TIGHTNESS_FAIL", f"Expected tight 9_over_25, got {min_spinor}")
    require_ok(min_cayley == Fraction(81, 625), "CAYLEY_TIGHTNESS_FAIL", f"Expected tight 81_over_625, got {min_cayley}")
    return {
        "acute_ball_triples_verified": checks,
        "min_spinor_denom": "9/25",
        "min_cayley_denom": "81/625",
    }


def verify_thm4_commutative_collapse_q(
    compose_fn: Any = cayley_compose_q,
    cross_fn: Any = cross3_q,
    quat_mul_fn: Any = quat_mul_q,
) -> dict[str, Any]:
    axes = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(-2, 3), Fraction(2, 3)),
        (Fraction(-1, 2), Fraction(1, 4), Fraction(1, 5)),
    )
    scales = ((Fraction(1, 3), Fraction(-1, 2)), (Fraction(2, 5), Fraction(3, 7)))
    omegas = (Fraction(-1), Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2))
    checks = 0
    for e in axes:
        for alpha, beta in scales:
            w1 = (alpha * e[0], alpha * e[1], alpha * e[2])
            wr = (beta * e[0], beta * e[1], beta * e[2])
            require_ok(cross_fn(w1, wr) == (Fraction(0), Fraction(0), Fraction(0)), "COLLINEAR_CROSS_NONZERO", "w1 x wr != 0")
            u1 = cayley_su2_q(w1)
            ur = cayley_su2_q(wr)
            require_ok(quat_mul_fn(u1, ur) == quat_mul_fn(ur, u1), "COMM_QUAT_FAIL", "C(w1)*C(wr) != C(wr)*C(w1)")
            nw1 = (-w1[0], -w1[1], -w1[2])
            w_delta = compose_fn(nw1, wr)
            for om in omegas:
                om_d = (om * w_delta[0], om * w_delta[1], om * w_delta[2])
                w_right = compose_fn(wr, om_d)
                w_left = compose_fn(om_d, wr)
                require_ok(w_right == w_left, "COMM_LEFT_RIGHT_MISMATCH", "Right != Left on collinear pair")
                if om == Fraction(1):
                    w_cayley_sum = compose_fn(compose_fn(wr, nw1), wr)
                    require_ok(w_right == w_cayley_sum, "COMM_CAYLEY_SUM_MISMATCH", "Right != Cayley-sum at omega=1")
                checks += 1
    return {"collinear_collapse_checks": checks}


def read_package_snapshot(pkg_dir: Path) -> dict[str, bytes]:
    """Read all 8 bound Lean package files into memory in a single symlink-refusing pass."""
    _assert_no_symlink_ancestors(pkg_dir)
    require_ok(pkg_dir.is_dir(), "MISSING_LEAN_DIR", "loopcd_strang package directory missing")
    for entry in pkg_dir.rglob("*"):
        require_ok(not entry.is_symlink(), "SYMLINK_FORBIDDEN", f"Symlink forbidden inside package: {entry.name}")
    lean_on_disk = sorted(str(p.relative_to(pkg_dir)).replace("\\", "/") for p in pkg_dir.rglob("*.lean"))
    expected_lean = sorted(r for r in BOUND_PACKAGE_FILES if r.endswith(".lean"))
    require_ok(
        lean_on_disk == expected_lean,
        "LEAN_FILE_SET_FAIL",
        f"Expected .lean files {expected_lean}, found {lean_on_disk}",
    )
    snap: dict[str, bytes] = {}
    for rel in BOUND_PACKAGE_FILES:
        fpath = Path(pkg_dir, rel)
        snap[rel] = read_regular_bytes_nofollow(fpath, "MISSING_BOUND_FILE")
    return snap


def compute_bound_source_hashes(pkg_dir: Path, snap: dict[str, bytes] | None = None) -> dict[str, str]:
    data = snap if snap is not None else read_package_snapshot(pkg_dir)
    return {rel: hashlib.sha256(data[rel]).hexdigest() for rel in BOUND_PACKAGE_FILES}


def default_live_lean_runner(snap: dict[str, bytes]) -> dict[str, Any]:
    """
    Materialize the exact SHA-256-verified in-memory `BOUND_PACKAGE_FILES` snapshot
    (including `lake-manifest.json`, `lakefile.toml`, and `lean-toolchain`) into an
    isolated temp directory and run cold `lake build` + `leanchecker --fresh` + `AxiomAudit.lean`.
    """
    home = os.path.expanduser("~")
    elan_bin = str(Path(home, ".elan", "bin"))
    extra_path = os.environ.get("PATH", "")
    env = {
        "PATH": f"{elan_bin}:/usr/local/bin:/usr/bin:/bin:{extra_path}",
        "HOME": home,
        "CUDA_VISIBLE_DEVICES": "",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory() as td:
        tmp_lean = Path(td, "loopcd_strang")
        for rel, raw in snap.items():
            dst = Path(tmp_lean, rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)
        try:
            r_build = subprocess.run(
                ["lake", "build"], cwd=tmp_lean, env=env, capture_output=True, text=True, timeout=300, check=False
            )
            manifest_after = Path(tmp_lean, "lake-manifest.json").read_bytes()
            r_lc = subprocess.run(
                ["lake", "env", "leanchecker", "--fresh", "LoopCDStrang"],
                cwd=tmp_lean,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            r_ax = subprocess.run(
                ["lake", "env", "lean", "AxiomAudit.lean"],
                cwd=tmp_lean,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LoopCDStrangVerifyError("LEAN_TIMEOUT", f"Lean subprocess timed out: {exc.cmd}") from exc
        return {
            "manifest_present": True,
            "manifest_unchanged": manifest_after == snap["lake-manifest.json"],
            "build_rc": r_build.returncode,
            "build_output": r_build.stdout + "\n" + r_build.stderr,
            "leanchecker_rc": r_lc.returncode,
            "leanchecker_output": r_lc.stdout + "\n" + r_lc.stderr,
            "axiom_rc": r_ax.returncode,
            "axiom_stdout": r_ax.stdout,
        }


def validate_axiom_audit_lines(axiom_lines: list[str] | tuple[str, ...], expected_count: int) -> dict[str, int]:
    non_empty = [ln.strip() for ln in axiom_lines if ln.strip()]
    require_ok(
        len(non_empty) == expected_count,
        "AXIOM_LINE_COUNT_FAIL",
        f"Expected {expected_count} #print axioms lines, got {len(non_empty)}",
    )
    propext_only = 0
    propext_quot = 0
    zero_ax = 0
    for ln in non_empty:
        if "does not depend on any axioms" in ln:
            zero_ax += 1
            continue
        require_ok("depends on axioms:" in ln, "UNRECOGNIZED_AXIOM_LINE", f"Unrecognized axiom line: {ln}")
        rhs = ln.split("depends on axioms:", 1)[1].strip()
        require_ok(rhs.startswith("["), "MALFORMED_AXIOM_LBRACKET", f"Missing '[': {ln}")
        require_ok(rhs.endswith("]"), "MALFORMED_AXIOM_RBRACKET", f"Missing ']': {ln}")
        axs = [a.strip() for a in rhs[1:-1].split(",") if a.strip()]
        for ax in axs:
            require_ok(
                ax not in EXPLICIT_BANNED_AXIOMS and "._native." not in ax,
                "BANNED_AXIOM",
                f"Explicitly banned axiom {ax} in {ln}",
            )
            require_ok(ax in PURE_CONSTRUCTIVE_AXIOMS, "FORBIDDEN_AXIOM", f"Forbidden axiom {ax} in {ln}")
        if set(axs) == {"propext"}:
            propext_only += 1
        else:
            propext_quot += 1
    return {"zero_axiom_theorems": zero_ax, "propext_only_theorems": propext_only, "propext_quot_theorems": propext_quot}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "CERT_DUPLICATE_KEY", f"Duplicate JSON key: {k}")
        out[k] = v
    return out


def _reject_json_num_token(tok: str) -> Any:
    raise LoopCDStrangVerifyError("CERT_FORBIDDEN_JSON_NUM", f"Forbidden float or non-finite JSON token: {tok}")


def _parse_bounded_json_int(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "CERT_INT_TOO_LARGE", "JSON integer exceeds digit bound")
    return int(tok)


def _check_json_nesting_depth(raw_text: str, max_depth: int = MAX_JSON_DEPTH) -> None:
    depth = 0
    in_str = False
    esc = False
    for ch in raw_text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in ("{", "["):
            depth += 1
            require_ok(depth <= max_depth, "CERT_JSON_TOO_DEEP", f"JSON nesting exceeds {max_depth}")
        elif ch in ("}", "]"):
            depth -= 1


def strict_json_loads(raw_text: str) -> Any:
    _check_json_nesting_depth(raw_text)
    try:
        return json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_float=_reject_json_num_token,
            parse_int=_parse_bounded_json_int,
            parse_constant=_reject_json_num_token,
        )
    except LoopCDStrangVerifyError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise LoopCDStrangVerifyError("CERT_INVALID_JSON", f"Malformed or oversized JSON document: {exc}") from exc


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def verify_lean_package(
    pkg_dir: Path,
    run_live_lean: bool = False,
    lean_runner: Any = None,
    snap: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    data = snap if snap is not None else read_package_snapshot(pkg_dir)

    toolchain = decode_utf8_strict(data["lean-toolchain"], "SOURCE_INVALID_UTF8", "lean-toolchain").strip()
    require_ok(toolchain == "leanprover/lean4:v4.36.0", "BAD_TOOLCHAIN", f"Unexpected toolchain: {toolchain}")

    manifest = strict_json_loads(decode_utf8_strict(data["lake-manifest.json"], "SOURCE_INVALID_UTF8", "lake-manifest.json"))
    require_ok(isinstance(manifest, dict), "INVALID_LAKE_MANIFEST", "lake-manifest.json root must be a JSON object")
    require_ok(manifest.get("packages") == [], "NONEMPTY_LAKE_PACKAGES", "lake-manifest.json packages must be []")

    try:
        lake_cfg = tomllib.loads(decode_utf8_strict(data["lakefile.toml"], "SOURCE_INVALID_UTF8", "lakefile.toml"))
    except tomllib.TOMLDecodeError as exc:
        raise LoopCDStrangVerifyError("INVALID_LAKEFILE_TOML", f"Malformed lakefile.toml: {exc}") from exc
    require_ok(
        lake_cfg.get("require", []) == [],
        "NONEMPTY_LAKEFILE_REQUIRE",
        "lakefile.toml must declare zero external [[require]] dependencies",
    )

    lean_rels = sorted(r for r in BOUND_PACKAGE_FILES if r.endswith(".lean"))
    total_loc = 0
    pub_thms: list[str] = []
    for rel in lean_rels:
        src = decode_utf8_strict(data[rel], "SOURCE_INVALID_UTF8", rel)
        total_loc += len(src.splitlines())
        cleaned = strip_lean_comments_and_strings(src)
        require_ok(
            _FORBIDDEN_LEAN_CODE_RE.search(cleaned) is None,
            "LEAN_STATIC_ESCAPE",
            f"Forbidden tactic, axiom, or macro escape in {rel}",
        )
        for m_imp in _ANY_IMPORT_RE.finditer(cleaned):
            mod = m_imp.group(1).strip()
            require_ok(
                _VALID_IMPORT_MOD_RE.fullmatch(mod) is not None,
                "FORBIDDEN_IMPORT",
                f"Non-Init or non-project import {mod} in {rel}",
            )
        if rel.startswith("LoopCDStrang/"):
            pub_thms.extend(_PUB_THM_RE.findall(cleaned))

    audit_cleaned = strip_lean_comments_and_strings(
        decode_utf8_strict(data["AxiomAudit.lean"], "SOURCE_INVALID_UTF8", "AxiomAudit.lean")
    )
    audited_thms = _PRINT_AXIOMS_RE.findall(audit_cleaned)
    require_ok(
        len(pub_thms) == EXPECTED_PUBLIC_THEOREMS,
        "PUB_THM_COUNT_FAIL",
        f"Expected {EXPECTED_PUBLIC_THEOREMS} public theorems, found {len(pub_thms)}",
    )
    require_ok(set(pub_thms) == set(audited_thms), "AXIOM_AUDIT_SET_MISMATCH", "AxiomAudit.lean != public theorems")
    require_ok(
        len(audited_thms) == EXPECTED_PUBLIC_THEOREMS,
        "AXIOM_AUDIT_DUP_OR_LEN",
        f"Expected {EXPECTED_PUBLIC_THEOREMS} #print axioms, got {len(audited_thms)}",
    )

    pinned_counts = validate_axiom_audit_lines(EXPECTED_AXIOM_AUDIT_LINES, EXPECTED_PUBLIC_THEOREMS)
    if run_live_lean or lean_runner is not None:
        if lean_runner is not None:
            run_res = lean_runner(dict(data))
        else:
            run_res = default_live_lean_runner(data)
        require_ok(run_res.get("manifest_present") is True, "MISSING_SANDBOX_MANIFEST", "Sandbox omitted lake-manifest.json")
        require_ok(run_res.get("manifest_unchanged") is True, "SANDBOX_MANIFEST_MUTATED", "lake build mutated lake-manifest.json")
        require_ok(run_res["build_rc"] == 0, "LAKE_BUILD_FAIL", "lake build exited non-zero")
        require_ok("warning:" not in run_res["build_output"].lower(), "LAKE_BUILD_WARNING", "lake build emitted warning")
        require_ok(run_res["leanchecker_rc"] == 0, "LEANCHECKER_FAIL", "leanchecker --fresh failed")
        require_ok(run_res["axiom_rc"] == 0, "AXIOM_AUDIT_EXEC_FAIL", "AxiomAudit.lean failed")
        live_lines = [ln.strip() for ln in run_res["axiom_stdout"].splitlines() if ln.strip()]
        ax_counts = validate_axiom_audit_lines(live_lines, EXPECTED_PUBLIC_THEOREMS)
        require_ok(
            tuple(live_lines) == EXPECTED_AXIOM_AUDIT_LINES,
            "AXIOM_PIN_MISMATCH",
            "Live #print axioms output drifted from pinned 38-theorem axiom profile",
        )
    else:
        ax_counts = pinned_counts

    return {
        "lean_toolchain": toolchain,
        "lean_files": len(lean_rels),
        "lean_loc": total_loc,
        "public_theorems_audited": len(pub_thms),
        "axiom_breakdown": ax_counts,
    }


def build_certificate(
    pkg_dir: Path | None = None,
    run_live_lean: bool = False,
    lean_runner: Any = None,
) -> dict[str, Any]:
    base = pkg_dir if pkg_dir is not None else Path(__file__).absolute().parent
    snap = read_package_snapshot(base)
    verifier_bytes = read_regular_bytes_nofollow(Path(base, "verify_loopcd_strang.py"), "MISSING_VERIFIER_FILE")
    verifier_src = decode_utf8_strict(verifier_bytes, "SOURCE_INVALID_UTF8", "verify_loopcd_strang.py")
    ast_info = verify_pure_q_ast(verifier_src)
    t1 = verify_thm1_chord_saturation_q()
    t2 = verify_thm2_palindromic_strang_bch_q()
    t3 = verify_thm3_acute_ball_floors_q()
    t4 = verify_thm4_commutative_collapse_q()
    lean_info = verify_lean_package(base, run_live_lean=run_live_lean, lean_runner=lean_runner, snap=snap)
    source_hashes = compute_bound_source_hashes(base, snap=snap)

    payload: dict[str, Any] = {
        "schema_version": "holosphere.loopcd_strang_lean436.v1",
        "module": "LoopCDStrang",
        "pure_q_ast": ast_info,
        "thm1_chord_saturation": t1,
        "thm2_palindromic_strang_bch": t2,
        "thm3_acute_ball_floors": t3,
        "thm4_commutative_collapse": t4,
        "lean_verification": lean_info,
        "source_files_sha256": source_hashes,
    }
    canon = canonical_json_bytes(payload)
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def validate_certificate(
    cert_obj: Any,
    pkg_dir: Path | None = None,
    run_live_lean: bool = False,
    lean_runner: Any = None,
) -> dict[str, Any]:
    base = pkg_dir if pkg_dir is not None else Path(__file__).absolute().parent
    require_ok(isinstance(cert_obj, dict), "CERT_NOT_OBJECT", "Certificate JSON must be an object")
    claimed_sha = cert_obj.get("payload_sha256")
    require_ok(is_hex64(claimed_sha), "CERT_BAD_PAYLOAD_HEX64", "Missing or malformed 64-hex payload_sha256")

    body = {k: v for k, v in cert_obj.items() if k != "payload_sha256"}
    recomputed_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    require_ok(claimed_sha == recomputed_sha, "CERT_SELF_HASH_MISMATCH", "Certificate payload_sha256 self-hash mismatch")

    src_map = cert_obj.get("source_files_sha256")
    require_ok(isinstance(src_map, dict), "CERT_MISSING_SOURCE_MAP", "Missing source_files_sha256 map")
    require_ok(
        sorted(src_map.keys()) == sorted(BOUND_PACKAGE_FILES),
        "CERT_SOURCE_KEYS_MISMATCH",
        "source_files_sha256 keys do not match BOUND_PACKAGE_FILES",
    )
    snap = read_package_snapshot(base)
    live_hashes = compute_bound_source_hashes(base, snap=snap)
    for rel, digest in src_map.items():
        require_ok(is_hex64(digest), "CERT_BAD_FILE_HEX64", f"Malformed SHA-256 for {rel}")
        require_ok(digest == live_hashes[rel], "SOURCE_SHA256_MISMATCH", f"On-disk file {rel} drifted from certificate")

    fresh = build_certificate(base, run_live_lean=run_live_lean, lean_runner=lean_runner)
    require_ok(
        canonical_json_bytes(cert_obj) == canonical_json_bytes(fresh),
        "CERTIFICATE_DRIFT",
        "Committed certificate drifted from freshly recomputed witnesses",
    )
    live_flag = bool(run_live_lean or lean_runner is not None)
    return {
        "status": "PASS",
        "module": "LoopCDStrang",
        "lean_live_replayed": live_flag,
        "verification_mode": "live_lean_kernel_and_pure_q" if live_flag else "offline_static_and_pure_q",
        "public_theorems_verified": fresh["lean_verification"]["public_theorems_audited"],
        "payload_sha256": fresh["payload_sha256"],
    }


def _run_cli_action(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoopCDStrang Pure-Q + Lean 4.36.0 Self-Bound Certificate Verifier.")
    parser.add_argument("--live-lean", action="store_true", help="Run cold lake build + leanchecker --fresh + AxiomAudit.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="Emit freshly computed certificate JSON to stdout.")
    mode.add_argument("--check", nargs="?", const="", default=None, help="Verify committed certificate JSON on disk.")
    mode.add_argument("--write-cert", nargs="?", const="", default=None, help="Write certificate JSON (requires --live-lean).")
    args = parser.parse_args(argv)

    base = Path(__file__).absolute().parent
    default_cert = Path(base, "loopcd_strang_certificate.json")

    if args.check is not None:
        cpath = Path(args.check) if args.check else default_cert
        raw_bytes = read_regular_bytes_nofollow(cpath, "MISSING_CERT_FILE")
        raw_text = decode_utf8_strict(raw_bytes, "CERT_INVALID_UTF8", cpath.name)
        cert_data = strict_json_loads(raw_text)
        res = validate_certificate(cert_data, pkg_dir=base, run_live_lean=args.live_lean)
        expected_bytes = (json.dumps(cert_data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
        require_ok(raw_bytes == expected_bytes, "CERT_RAW_FORMAT_DRIFT", "Certificate raw file bytes are non-canonical")
        sys.stdout.write(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
        return 0

    if args.write_cert is not None:
        require_ok(
            args.live_lean,
            "WRITE_CERT_REQUIRES_LIVE_LEAN",
            "--write-cert requires --live-lean so committed certificates always reflect a live kernel replay",
        )
        cert = build_certificate(pkg_dir=base, run_live_lean=True)
        rendered = json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        out_path = Path(args.write_cert) if args.write_cert else default_cert
        write_regular_bytes_nofollow(out_path, rendered.encode("ascii"))
        sys.stdout.write(rendered)
        return 0

    cert = build_certificate(pkg_dir=base, run_live_lean=args.live_lean)
    sys.stdout.write(json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run_cli_action(argv)
    except LoopCDStrangVerifyError as err:
        sys.stderr.write(
            json.dumps({"error_code": err.code, "message": str(err), "status": "FAIL"}, sort_keys=True, ensure_ascii=True)
            + "\n"
        )
        return 2
    except OSError as err:
        sys.stderr.write(
            json.dumps({"error_code": "IO_ERROR", "message": str(err), "status": "FAIL"}, sort_keys=True, ensure_ascii=True)
            + "\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
