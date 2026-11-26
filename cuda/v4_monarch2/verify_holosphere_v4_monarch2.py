#!/usr/bin/env python3
"""Pure-Q / Pure-Z Fail-Closed Verifier for Fused HoloSphere-V4 Monarch-2 E2E Megakernel.

Zero-float, zero-ast.Div/FloorDiv/Mod, pure-stdlib, self-bound on-disk verifier enforcing:
  1. Strict canonical-JSON, zero-TOCTOU dir_fd O_NOFOLLOW|O_NONBLOCK + fstat(S_ISREG) refusal
     of symlinks/FIFOs/directories, 1 MiB streaming read cap, duplicate-key refusal, non-finite/
     float/1024-digit integer refusal, closed nested key allowlists, and on-disk SHA-256 +
     comment-stripped contextual scale-constant + study AST bindings.
  2. Closed-allowlist pure-Q AST self-audit blocking ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
     ast.Assert, float/complex literals, format-spec f-strings, unallowlisted imports/from-imports/
     aliases, unguarded json.loads, Fraction/type/module aliasing or shadowing, and dunder escapes.
  3. Exact sm_86 ptxas & cuobjdump SASS zero-spill verification (2/2 kernels, 55 regs, 0 B spills,
     0 LDL/STL, exact ATOMG.E.CAS.STRONG.SYS / RED.E.OR.STRONG.SYS / ATOMS counts).
  4. Pure-Z algebraic oracles: H_256 involution, FWHT_256 x bent4 Parseval + 6-impulse signed-Hadamard
     KATs + BOS-256/byte-33 algebraic equivalence, 1-stage (16/256) vs 2-stage (256/256) Monarch
     reachability, and [16,11,4] SECDED 16/16 SEC heal + 3-pair DED detection.
  5. Exact pure-Q recomputation of fused 1L/4L throughput (1e6 / us), 4-arm x 3-seed x 3-context
     (T=512, T=2048, T=8192) BPB means, FP16 quant deltas, lifts, dense gap closure, gates G1/G2,
     verdict (--require-verdict), and linear-time uncontrolled-laptop telemetry policy guard.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence, Set
import errno
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

SCHEMA_PTXAS = "holosphere.v4_monarch2.ptxas_zero_spill_receipt.v1"
SCHEMA_RESULTS = "holosphere.v4_monarch2.e2e_results.v1"
SCHEMA_VERIFY = "holosphere.v4_monarch2.verification_summary.v1"
ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
EXPECTED_COMPILE_FLAGS = (
    "-O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings "
    "--shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror"
)

KERNEL_CU_REL = "cuda/v4_monarch2/holosphere_v4_monarch2_e2e_sm86.cu"
BENCH_CU_REL = "cuda/v4_monarch2/holosphere_v4_monarch2_live_bench.cu"
BPB_STUDY_REL = "cuda/v4_monarch2/holosphere_v4_monarch2_bpb_study.py"
PTXAS_REL = "receipts/v4_monarch2/holosphere_v4_monarch2_ptxas_receipt.json"
RESULTS_REL = "receipts/v4_monarch2/holosphere_v4_monarch2_results.json"

EXPECTED_CORPUS_ID = "holosphere-v3-flagship-corpus"
EXPECTED_CORPUS_MANIFEST_SHA256 = "a05e415f7d33ef32376ff3d9da1eede0e3f561b64094b5b91147b2696a99e9d4"
EXPECTED_CORPUS_TRAIN_SHA256 = "528c6c792d2a563d82d21ce213b26209441b39fa5e76b8595607bab2ca853b7b"
EXPECTED_CORPUS_EVAL_SHA256 = "31fd132d888a21ecebe9e907b6c4b82e3e68e42b5a29e2839584447dcca631a0"

_KAT_DELTA_INDICES: Tuple[int, ...] = (0, 5, 15, 37, 74, 250)
_DED_BIT_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (3, 12), (7, 15))
MAX_INT_DIGITS = 1024
MAX_JSON_DEPTH = 32
MAX_FILE_BYTES = 1024 * 1024
_OPEN_NOFOLLOW_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_OPEN_DIR_NOFOLLOW_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK

_CPP_STRIP_RE = re.compile(
    r"#\s*if\s+0\b.*?#\s*endif\b|/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"",
    re.DOTALL,
)
_KERNEL_PLAIN_HDR_RE = re.compile(r"\bfwht256_plain_8warp_raw\s*\([^;{}]*\)\s*\{")
_KERNEL_BENT4_HDR_RE = re.compile(r"\bfwht256_bent4_8warp_raw\s*\([^;{}]*\)\s*\{")
_KERNEL_PLAIN_SCALE_RE = re.compile(r"\bv\s*\*=\s*0\.0625f\s*;")
_KERNEL_BENT4_SCALE_RE = re.compile(r"\bv\s*\*=\s*0\.17677669529663688f\s*;")
_KERNEL_V3_GLOBAL_RE = re.compile(
    r"__global__\s+__launch_bounds__\s*\(\s*512\s*,\s*1\s*\)\s+void\s+holosphere_v3_e2e_scalar_fwht_sm86\s*\("
)
_KERNEL_V4_GLOBAL_RE = re.compile(
    r"__global__\s+__launch_bounds__\s*\(\s*512\s*,\s*1\s*\)\s+void\s+holosphere_v4_e2e_monarch2_bias_fwht_sm86\s*\("
)
_BENCH_PLAIN_HDR_RE = re.compile(r"\bfwht256_plain_fp64\s*\([^;{}]*\)\s*\{")
_BENCH_BENT4_HDR_RE = re.compile(r"\bfwht256_bent4_fp64\s*\([^;{}]*\)\s*\{")
_BENCH_PLAIN_SCALE_RE = re.compile(r"\bacc\s*\*\s*0\.0625\s*;")
_BENCH_BENT4_SCALE_RE = re.compile(r"\binv_sqrt32\s*=\s*1\.0\s*/\s*std::sqrt\s*\(\s*32\.0\s*\)\s*;")
_STUDY_EXPECTED_PINS: Dict[str, str] = {
    "CORPUS_ID": EXPECTED_CORPUS_ID,
    "EXPECTED_CORPUS_MANIFEST_SHA256": EXPECTED_CORPUS_MANIFEST_SHA256,
    "EXPECTED_TRAIN_BIN_SHA256": EXPECTED_CORPUS_TRAIN_SHA256,
    "EXPECTED_EVAL_BIN_SHA256": EXPECTED_CORPUS_EVAL_SHA256,
}
_STUDY_FORBIDDEN_IMPORTS: frozenset[str] = frozenset({"pynvml", "subprocess", "ctypes"})

V3_E2E_SYMBOL = (
    "_ZN22holosphere_v4_monarch234holosphere_v3_e2e_scalar_fwht_sm86E"
    "PKhPK5uint4PKfPK6float3S4_S4_PK6__halfPKjP6float4PfSH_PhSH_PjSJ_SJ_ffii"
)
V4_E2E_SYMBOL = (
    "_ZN22holosphere_v4_monarch241holosphere_v4_e2e_monarch2_bias_fwht_sm86E"
    "PKhPK5uint4PKfPK6float3S4_S4_S4_S4_PK6__halfPKjP6float4PfSH_PhSH_PjSJ_SJ_ffii"
)

# Exact expected (registers, barriers, smem_bytes, atom_cas_sys_insns, red_or_sys_insns, atoms_smem_insns)
_EXPECTED_KERNEL_SPECS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    V3_E2E_SYMBOL: (55, 1, 14004, 1, 1, 3),
    V4_E2E_SYMBOL: (55, 1, 26292, 1, 1, 3),
}
_EXPECTED_KERNEL_SYMBOLS: frozenset[str] = frozenset(_EXPECTED_KERNEL_SPECS.keys())

_PTXAS_KEYS: frozenset[str] = frozenset(
    {
        "schema", "energy_power_status", "arch", "nvcc_version", "compile_flags",
        "cu_filename", "cu_sha256", "total_kernels", "zero_spill_kernels",
        "nonzero_spill_kernels", "kernels", "receipt_sha256",
    }
)
_KERNEL_STAT_KEYS: frozenset[str] = frozenset(
    {
        "stack_bytes", "spill_store_bytes", "spill_load_bytes",
        "registers", "barriers", "smem_bytes",
        "ldl_insns", "stl_insns", "atom_cas_sys_insns", "red_or_sys_insns", "atoms_smem_insns",
    }
)
_RESULTS_KEYS: frozenset[str] = frozenset(
    {
        "schema", "energy_power_status", "scope_note",
        "hardware", "source_bindings", "silicon_e2e_bench",
        "e2e_bpb_study", "context_summary", "gates", "verdict", "result_sha256",
    }
)
_HARDWARE_KEYS: frozenset[str] = frozenset({"device", "sm", "nvcc_version"})
_SOURCE_BINDINGS_KEYS: frozenset[str] = frozenset(
    {
        "kernel_cu_filename", "kernel_cu_sha256",
        "bench_cu_filename", "bench_cu_sha256",
        "bpb_study_script_filename", "bpb_study_script_sha256",
        "ptxas_receipt_filename", "ptxas_receipt_sha256",
        "corpus_id", "corpus_manifest_sha256",
        "corpus_train_bin_sha256", "corpus_eval_bin_sha256",
    }
)
_SILICON_KEYS: frozenset[str] = frozenset(
    {
        "clean_packed_u32", "clean_sys_atomics",
        "v3_1l_fp64_max_err_q", "v4_1l_fp64_max_err_q", "v4_4l_fp64_max_err_q",
        "v4_sec16_packed_u32", "v4_sec16_sys_atomics",
        "v4_sec16_hidden_max_abs_err_q", "v4_sec16_logits_max_abs_err_q",
        "v3_scalar_1l_us_per_tok_q", "v4_monarch2_1l_us_per_tok_q", "v4_monarch2_1l_tok_per_s_q",
        "v3_scalar_4l_us_per_tok_q", "v4_monarch2_4l_us_per_tok_q", "v4_monarch2_4l_tok_per_s_q",
    }
)
_STUDY_KEYS: frozenset[str] = frozenset(
    {
        "d_model", "d_ff", "n_heads", "n_layers", "rope_base",
        "batch_size", "train_seq_len", "steps_per_seed", "tokens_per_seed",
        "eval_contexts", "seeds", "h256_ortho_fp64_max_err_q", "arms",
    }
)
_ARM_KEYS: frozenset[str] = frozenset(
    {
        "head_params", "head_fp16_bytes", "hbm_input_vocab_bytes", "hbm_output_vocab_bytes",
        "total_params", "mean_eval_bpb_by_ctx_q", "seeds",
    }
)
_SEED_KEYS: frozenset[str] = frozenset({"seed", "train_wall_s_q", "contexts"})
_CTX_EVAL_KEYS: frozenset[str] = frozenset({"eval_bpb_q", "eval_bpb_bf16_q", "fp16_quant_bpb_diff_q", "eval_tokens"})
_CTX_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "v4_lift_over_scalar_q", "v4_lift_over_lowrank16_q", "v4_minus_dense_q",
        "dense_gap_scalar_minus_dense_q", "v4_gap_closed_fraction_q",
        "v4_lt_scalar_all_seeds", "v4_lt_lowrank16_all_seeds",
    }
)
_GATES_KEYS: frozenset[str] = frozenset(
    {"G1_e2e_multicontext_quality", "G2_fused_sm86_zero_spill_and_latency"}
)

_EXPECTED_ARMS: Tuple[str, ...] = (
    "V3_ScalarFWHT",
    "V4_Monarch2BiasFWHT",
    "V3_LowRank16Bias",
    "V3_DenseHead",
)
# (head_params, head_fp16_bytes, hbm_input_vocab_bytes, hbm_output_vocab_bytes)
_EXPECTED_HEAD_SPECS: Dict[str, Tuple[int, int, int, int]] = {
    "V3_ScalarFWHT": (1, 2, 0, 0),
    "V4_Monarch2BiasFWHT": (8448, 16896, 0, 0),
    "V3_LowRank16Bias": (8448, 16896, 0, 16896),
    "V3_DenseHead": (131329, 262658, 0, 262658),
}
_EXPECTED_CONTEXTS: Tuple[str, ...] = ("T512", "T2048", "T8192")
_EXPECTED_CTX_META: List[Dict[str, int]] = [
    {"seq_len": 512, "chunks": 128, "eval_bytes": 65536},
    {"seq_len": 2048, "chunks": 32, "eval_bytes": 65536},
    {"seq_len": 8192, "chunks": 8, "eval_bytes": 65536},
]

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_Q_STR_RE = re.compile(r"\A-?[0-9]+/[1-9][0-9]*\Z")
_ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_ALLOWED_AST_IMPORT_MODULES: frozenset[str] = frozenset(
    {"argparse", "ast", "errno", "hashlib", "json", "os", "re", "stat", "sys", "unicodedata"}
)
_ALLOWED_AST_FROM_IMPORTS: Dict[str, frozenset[str]] = {
    "__future__": frozenset({"annotations"}),
    "collections.abc": frozenset({"Mapping", "Sequence", "Set"}),
    "fractions": frozenset({"Fraction"}),
    "pathlib": frozenset({"Path"}),
    "typing": frozenset({"Any", "Dict", "List", "Optional", "Tuple"}),
}
_ALLOWED_AST_MODULE_ATTRS: Dict[str, frozenset[str]] = {
    "ast": frozenset(
        {
            "AST", "AnnAssign", "Assert", "Assign", "AsyncFunctionDef", "Attribute", "Call",
            "ClassDef", "Compare", "Constant", "Div", "Eq", "ExceptHandler", "FloorDiv",
            "FormattedValue", "FunctionDef", "GeneratorExp", "Import", "ImportFrom", "Is",
            "JoinedStr", "Mod", "Name", "Pow", "Starred", "Subscript", "Tuple", "arg", "parse",
            "walk",
        }
    ),
    "json": frozenset({"JSONDecodeError", "dumps", "loads"}),
    "os": frozenset({"O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_RDONLY", "close", "fstat", "lstat", "open", "read"}),
    "re": frozenset({"DOTALL", "compile"}),
    "sys": frozenset({"stderr", "stdout"}),
}
_RESTRICTED_AST_SYMBOLS: frozenset[str] = frozenset({"Fraction", "type", "ast", "json", "os", "re", "sys"})
_FORBIDDEN_AST_NAMES: frozenset[str] = frozenset(
    {
        "float", "complex", "pow", "format", "divmod", "round", "open", "memoryview",
        "__import__", "eval", "exec", "compile",
        "getattr", "setattr", "delattr", "globals", "locals", "vars", "builtins", "__builtins__",
    }
)
_FORBIDDEN_AST_ATTRS: frozenset[str] = (_FORBIDDEN_AST_NAMES - frozenset({"compile"})) | frozenset(
    {
        "from_float", "from_decimal", "fromhex", "as_integer_ratio", "format_map",
        "real", "imag", "conjugate", "cast", "literal_eval",
        "truediv", "itruediv", "floordiv", "ifloordiv",
        "__truediv__", "__rtruediv__", "__itruediv__",
        "__floordiv__", "__rfloordiv__", "__ifloordiv__",
        "__mod__", "__rmod__", "__imod__",
        "__float__", "__complex__", "__format__",
        "__pow__", "__rpow__", "__ipow__",
        "__getattr__", "__getattribute__",
        "__class__", "__base__", "__bases__", "__mro__", "__subclasses__", "__globals__",
        "modules", "__dict__", "__loader__", "__spec__",
    }
)

_FORBIDDEN_ENERGY_KEYS: frozenset[str] = frozenset(
    {
        "w", "pw", "nw", "uw", "mw", "kw", "gw", "tw",
        "wh", "whr", "whrs", "mwh", "kwh",
        "j", "pj", "nj", "uj", "mj", "kj", "gj", "tj",
        "tdp", "pwr", "watt", "watts", "wattage", "joule", "joules",
        "millijoule", "millijoules", "energy", "power", "edp",
        "rapl", "nvml", "nvidiasmi", "mah",
    }
)
_FORBIDDEN_COMPACT_KEY_RE = re.compile(
    r"(?:(?:tera|giga|mega|kilo|milli|micro|nano|pico)?(?:watt|joule)"
    r"|(?:avg|max|min|peak|gpu|cpu|sys|pkg|total|board|nvml|rapl)power"
    r"|power(?:draw|cap|limit|use|usage|budget|stat|metric|mw|kw|w\Z)"
    r"|energy|mjpertok|mjtok|jpertok|ujpertok|wpertok|wtok|kwh|mwh|whpertok|tdp|rapl|nvml|nvidiasmi)"
)

# Linear-time, bounded-separator energy/power prose filter (zero nested-star or unbounded separator ReDoS).
_ENERGY_PFX = r"(?:p|n|u|\u03bc|m|k|g|t|pico|nano|micro|milli|kilo|mega|giga|tera)"
_ENERGY_UNIT = r"(?:watt[\s_\-\u2013\u2014]{0,2}hours?|watthours?|watts?|wattage|joules?|whrs?|wh|w|j|tdp)"
_ENERGY_MULTI_UNIT = (
    r"(?:watt[\s_\-\u2013\u2014]{0,2}hours?|watthours?|watts?|wattage|joules?|whrs?|wh|kwh|mwh|tdp|energy|power"
    r"|(?:p|n|u|\u03bc|m|k|g|t|pico|nano|micro|milli|kilo|mega|giga|tera)[\s_\-\u2013\u2014]{0,2}(?:watts?|joules?|w|j))"
)
_UNIT_SEP = r"[\s_\-\u2013\u2014\u00d7\u00b7\u2219\[\(]{0,3}"
_ENERGY_NUM = r"(?<![0-9])\d+(?:[.,]\d+){0,3}(?:e[+-]?\d+)?"
_UPPER_BARE_WJ_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9])(?:[WJ](?:[\s]{0,3}[:=][\s]{0,3}|[\s]{1,3})\d|W[\s]{0,3}/[\s]{0,3}tok)")
_FORBIDDEN_ENERGY_VALUE_RE = re.compile(
    "(?:"
    rf"{_ENERGY_NUM}{_UNIT_SEP}(?:{_ENERGY_PFX}{_UNIT_SEP})?{_ENERGY_UNIT}(?![a-z0-9])(?!/\s|/o(?![a-z0-9])|/\d)(?!\s*=)"
    rf"|(?<![a-z0-9]){_ENERGY_MULTI_UNIT}(?:[\s]{{0,3}}[:=][\s]{{0,3}}|[\s]{{1,3}}){_ENERGY_NUM}(?![a-z0-9])"
    rf"|(?<![a-z0-9])(?:{_ENERGY_PFX})?(?:watt(?:s|age|hours?)?|joules?)(?![a-z0-9])"
    rf"|(?<![a-z0-9_])(?:kwh|mwh|whrs?)(?![a-z0-9_])"
    r"|\A[\s\[\(\-\u2013\u2014]{0,3}(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:w|j|wh)|tdp)[\s\]\)\?!.,\-\u2013\u2014]{0,3}\Z"
    r"|(?:^|[^a-z0-9])(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:joules?|watts?|j|wh)|(?:p|n|u|\u03bc|m|k|g|t)w|kwh|mwh|energy|power)[\s_-]{0,3}(?:per|/)[\s_-]{0,3}tok(?:en)?s?(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])w(?:/|[\s_-]{1,3}per[\s_-]{1,3})tok(?:en)?s?(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])tok(?:en)?s?[\s_-]{0,3}(?:per|/)[\s_-]{0,3}(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:j|w|joules?|watts?)|wh|kwh|mwh|energy|power)(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])per[\s_-]{1,3}(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:watts?|joules?)|w|j)(?:$|[^a-z0-9])"
    r"|(?<![0-9])\d+(?:[.,]\d+){0,2}[\s]{0,2}a[\s]{1,3}(?:at|@|x|\*)[\s]{1,3}\d+(?:[.,]\d+){0,2}[\s]{0,2}v(?![a-z0-9])"
    r"|(?:rapl|nvml|nvidia[\s_-]{0,2}smi|(?:^|[^a-z0-9])mah(?:$|[^a-z0-9]))"
    r"|(?:^|[^a-z0-9])(?:less|lower|reduced|fewer)[\s_-]{1,3}(?:in[\s_-]{1,3})?(?:energy|power)(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])(?:energy|power)[\s_-]{1,3}(?:use|usage|reduction|savings?|dropped|consumption|draw|budget|cap|limit)(?:$|[^a-z0-9])"
    r"|(?:%|percent)[\s_-]{1,3}(?:of[\s_-]{1,3})?(?:(?:stock|baseline|nominal|peak|idle|full|total|gpu|cpu)[\s_-]{1,3})?(?:energy|power)(?:$|[^a-z0-9])"
    ")"
)


class V4Monarch2VerifyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise V4Monarch2VerifyError(code, message)


def is_strict_int(val: Any) -> bool:
    return type(val) is int


def is_hex64(val: Any) -> bool:
    return type(val) is str and bool(_HEX64_RE.match(val))


def _read_fd_regular_bytes(fd: int, label: str, max_bytes: int) -> bytes:
    try:
        require_ok(stat.S_ISREG(os.fstat(fd).st_mode), "NOT_REGULAR_FILE", label)
        chunks: List[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            require_ok(total <= max_bytes, "FILE_TOO_LARGE", label)
            chunks.append(chunk)
        return b"".join(chunks)
    except V4Monarch2VerifyError:
        raise
    except OSError as exc:
        raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    finally:
        os.close(fd)


def read_regular_bytes(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    try:
        fd = os.open(str(path), _OPEN_NOFOLLOW_FLAGS)
    except FileNotFoundError as exc:
        raise V4Monarch2VerifyError("MISSING_FILE", path.name) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise V4Monarch2VerifyError("SYMLINK_FORBIDDEN", path.name) from exc
        raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    return _read_fd_regular_bytes(fd, path.name, max_bytes)


def read_repo_regular_bytes(repo_root: Path, rel_path: str, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    """Zero-TOCTOU dir_fd traversal rejecting symlinks on repo_root, every subdir, and the leaf file."""
    require_ok(not repo_root.is_symlink(), "SYMLINK_FORBIDDEN", repo_root.name)
    parts = rel_path.split("/")
    try:
        cur_fd = os.open(str(repo_root), _OPEN_DIR_NOFOLLOW_FLAGS)
    except FileNotFoundError as exc:
        raise V4Monarch2VerifyError("MISSING_FILE", repo_root.name) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise V4Monarch2VerifyError("SYMLINK_FORBIDDEN", repo_root.name) from exc
        raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    try:
        for part in parts[:-1]:
            try:
                nxt_fd = os.open(part, _OPEN_DIR_NOFOLLOW_FLAGS, dir_fd=cur_fd)
            except FileNotFoundError as exc:
                raise V4Monarch2VerifyError("MISSING_FILE", rel_path) from exc
            except OSError as exc:
                if exc.errno == errno.ELOOP or (
                    exc.errno == errno.ENOTDIR and stat.S_ISLNK(os.lstat(part, dir_fd=cur_fd).st_mode)
                ):
                    raise V4Monarch2VerifyError("SYMLINK_FORBIDDEN", rel_path) from exc
                raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
            os.close(cur_fd)
            cur_fd = nxt_fd
        try:
            leaf_fd = os.open(parts[-1], _OPEN_NOFOLLOW_FLAGS, dir_fd=cur_fd)
        except FileNotFoundError as exc:
            raise V4Monarch2VerifyError("MISSING_FILE", rel_path) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise V4Monarch2VerifyError("SYMLINK_FORBIDDEN", rel_path) from exc
            raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    finally:
        os.close(cur_fd)
    return _read_fd_regular_bytes(leaf_fd, rel_path, max_bytes)


def decode_utf8_bytes(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc


def _no_dup_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "DUPLICATE_JSON_KEY", f"duplicate key: {k!r}")
        out[k] = v
    return out


def _reject_non_finite(token: str) -> Any:
    raise V4Monarch2VerifyError("NON_FINITE_JSON", f"non-finite constant: {token!r}")


def _reject_json_float(token: str) -> Any:
    raise V4Monarch2VerifyError("FLOAT_JSON_FORBIDDEN", f"float literal forbidden: {token!r}")


def _parse_strict_json_int(raw: str) -> int:
    digits = raw[1:] if raw.startswith("-") else raw
    require_ok(len(digits) <= MAX_INT_DIGITS, "INT_DIGIT_LIMIT", f"int exceeds {MAX_INT_DIGITS} digits")
    return int(raw)


def _check_json_raw_depth(text: str, max_depth: int = MAX_JSON_DEPTH) -> None:
    depth = 0
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
            require_ok(depth <= max_depth, "JSON_DEPTH_LIMIT", f"json depth exceeds {max_depth}")
        elif ch in "}]":
            depth -= 1


def loads_strict_canonical_json_bytes(raw_bytes: bytes, label: str) -> Any:
    try:
        text = raw_bytes.decode("ascii")
        _check_json_raw_depth(text)
        parsed = json.loads(
            text,
            object_pairs_hook=_no_dup_object_pairs,
            parse_constant=_reject_non_finite,
            parse_float=_reject_json_float,
            parse_int=_parse_strict_json_int,
        )
        expected_raw = (json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except V4Monarch2VerifyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError, RecursionError) as exc:
        raise V4Monarch2VerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    require_ok(raw_bytes == expected_raw, "RAW_JSON_NOT_CANONICAL", f"non-canonical bytes in {label}")
    return parsed


def load_strict_canonical_json(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> Any:
    return loads_strict_canonical_json_bytes(read_regular_bytes(path, max_bytes=max_bytes), path.name)


def _is_fraction_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Fraction"
    if isinstance(node, ast.Attribute):
        return node.attr == "Fraction"
    return False


def _is_single_arg_type_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and len(node.keywords) == 0
    )


def _collect_safe_type_fraction_ids(node: Optional[ast.AST], out: set[int]) -> None:
    if node is None:
        return
    if isinstance(node, ast.Name):
        if node.id == "Fraction":
            out.add(id(node))
        return
    if isinstance(node, ast.Subscript):
        _collect_safe_type_fraction_ids(node.value, out)
        _collect_safe_type_fraction_ids(node.slice, out)
        return
    if isinstance(node, ast.Tuple):
        for elt in node.elts:
            _collect_safe_type_fraction_ids(elt, out)


def verify_pure_q_ast(source_text: Any) -> None:
    require_ok(type(source_text) is str, "AST_SOURCE_TYPE", "source_text must be str")
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise V4Monarch2VerifyError("AST_SYNTAX_ERROR", str(exc)) from exc

    allowed_restricted_ids: set[int] = set()
    allowed_mro_attr_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_fraction_ref(node.func):
            allowed_restricted_ids.add(id(node.func))
        elif isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Is) and _is_single_arg_type_call(node.left):
                allowed_restricted_ids.add(id(node.left.func))  # type: ignore[attr-defined]
        elif isinstance(node, ast.GeneratorExp):
            if (
                len(node.generators) == 1
                and not node.generators[0].ifs
                and isinstance(node.generators[0].target, ast.Name)
                and isinstance(node.generators[0].iter, ast.Attribute)
                and node.generators[0].iter.attr == "__mro__"
                and _is_single_arg_type_call(node.generators[0].iter.value)
                and isinstance(node.elt, ast.Compare)
                and len(node.elt.ops) == 1
                and isinstance(node.elt.ops[0], ast.Eq)
                and len(node.elt.comparators) == 1
                and isinstance(node.elt.left, ast.Attribute)
                and isinstance(node.elt.left.value, ast.Name)
                and node.elt.left.value.id == node.generators[0].target.id
                and node.elt.left.attr == "__name__"
                and isinstance(node.elt.comparators[0], ast.Constant)
                and type(node.elt.comparators[0].value) is str
            ):
                allowed_mro_attr_ids.add(id(node.generators[0].iter))
                allowed_restricted_ids.add(id(node.generators[0].iter.value.func))  # type: ignore[attr-defined]
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in _ALLOWED_AST_MODULE_ATTRS:
                allowed_restricted_ids.add(id(node.value))
        elif isinstance(node, ast.arg):
            _collect_safe_type_fraction_ids(node.annotation, allowed_restricted_ids)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_safe_type_fraction_ids(node.returns, allowed_restricted_ids)
        elif isinstance(node, ast.AnnAssign):
            _collect_safe_type_fraction_ids(node.annotation, allowed_restricted_ids)

    for node in ast.walk(tree):
        require_ok(not isinstance(node, ast.Div), "AST_DIV_FORBIDDEN", "ast.Div forbidden")
        require_ok(not isinstance(node, ast.FloorDiv), "AST_FLOORDIV_FORBIDDEN", "ast.FloorDiv forbidden")
        require_ok(not isinstance(node, ast.Mod), "AST_MOD_FORBIDDEN", "ast.Mod forbidden")
        require_ok(not isinstance(node, ast.Pow), "AST_POW_FORBIDDEN", "ast.Pow forbidden")
        require_ok(not isinstance(node, ast.Assert), "AST_BARE_ASSERT", "ast.Assert forbidden")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            require_ok(node.name not in _RESTRICTED_AST_SYMBOLS, "AST_SYMBOL_REBIND", node.name)
        if isinstance(node, ast.arg):
            require_ok(node.arg not in _RESTRICTED_AST_SYMBOLS, "AST_SYMBOL_REBIND", node.arg)
        if isinstance(node, ast.ExceptHandler):
            require_ok(node.name not in _RESTRICTED_AST_SYMBOLS, "AST_SYMBOL_REBIND", str(node.name))
        if isinstance(node, ast.FormattedValue):
            require_ok(node.format_spec is None, "AST_FORMAT_SPEC", "format specifiers forbidden")
        if isinstance(node, ast.Constant):
            val = node.value
            require_ok(
                val is None
                or val is Ellipsis
                or type(val) is bool
                or type(val) is int
                or type(val) is str
                or type(val) is bytes,
                "AST_FLOAT_LITERAL",
                "non-exact constant literal",
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                require_ok(alias.name in _ALLOWED_AST_IMPORT_MODULES, "AST_FORBIDDEN_IMPORT", alias.name)
                require_ok(alias.asname is None, "AST_IMPORT_ALIAS", alias.name)
        if isinstance(node, ast.ImportFrom):
            require_ok(node.level == 0 and node.module is not None, "AST_RELATIVE_IMPORT", "relative import forbidden")
            allowed_names = _ALLOWED_AST_FROM_IMPORTS.get(node.module)
            require_ok(allowed_names is not None, "AST_FORBIDDEN_IMPORT", node.module)
            for alias in node.names:
                require_ok(alias.name in allowed_names and alias.asname is None, "AST_FORBIDDEN_IMPORT", alias.name)
        if isinstance(node, ast.Name):
            require_ok(node.id not in _FORBIDDEN_AST_NAMES, "AST_FORBIDDEN_NAME", node.id)
            if node.id in _RESTRICTED_AST_SYMBOLS:
                require_ok(id(node) in allowed_restricted_ids, "AST_SYMBOL_REBIND", node.id)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in _ALLOWED_AST_MODULE_ATTRS:
                require_ok(node.attr in _ALLOWED_AST_MODULE_ATTRS[node.value.id], "AST_FORBIDDEN_MODULE_ATTR", node.attr)
            else:
                if node.attr != "__mro__" or id(node) not in allowed_mro_attr_ids:
                    require_ok(node.attr not in _FORBIDDEN_AST_ATTRS, "AST_FORBIDDEN_ATTR", node.attr)
            if node.attr == "Fraction":
                require_ok(id(node) in allowed_restricted_ids, "AST_SYMBOL_REBIND", "Fraction rebinding forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "json" and node.func.attr == "loads":
                    kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                    has_pf = isinstance(kw_map.get("parse_float"), ast.Name)
                    has_pc = isinstance(kw_map.get("parse_constant"), ast.Name)
                    require_ok(has_pf and has_pc, "AST_UNSAFE_JSON_LOADS", "json.loads requires parse_float and parse_constant")
            if _is_fraction_ref(node.func):
                for arg in node.args:
                    require_ok(not isinstance(arg, ast.Starred), "AST_FRACTION_STARRED", "Fraction(*args) forbidden")
                require_ok(len(node.args) == 2 and not node.keywords, "AST_FRACTION_ARGS", "Fraction requires 2 int args")
                for arg in node.args:
                    require_ok(
                        not isinstance(arg, ast.JoinedStr)
                        and not (isinstance(arg, ast.Constant) and (type(arg.value) is str or type(arg.value) is bytes)),
                        "AST_FRACTION_STR",
                        "Fraction(str) forbidden",
                    )


def canonical_json_bytes(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise V4Monarch2VerifyError("BAD_JSON", str(exc)) from exc


def canonical_sha256(obj: Any, *, exclude_key: Optional[str] = None) -> str:
    payload = obj if exclude_key is None else {k: v for k, v in obj.items() if k != exclude_key}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(read_regular_bytes(path)).hexdigest()


def format_q(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def parse_q(raw: Any, *, code: str = "BAD_RATIONAL") -> Fraction:
    require_ok(type(raw) is str and bool(_Q_STR_RE.match(raw)), code, f"invalid rational: {raw!r}")
    num_s, den_s = raw.split("/", 1)
    num_digits = num_s[1:] if num_s.startswith("-") else num_s
    require_ok(
        len(num_digits) <= MAX_INT_DIGITS and len(den_s) <= MAX_INT_DIGITS,
        code,
        f"rational exceeds {MAX_INT_DIGITS} digits",
    )
    q = Fraction(int(num_s), int(den_s))
    require_ok(format_q(q) == raw, code, f"non-canonical rational: {raw!r}")
    return q


def q_div(a: Fraction, b: Fraction) -> Fraction:
    require_ok(b != Fraction(0, 1), "ZERO_DIV", "division by zero")
    return Fraction(a.numerator * b.denominator, a.denominator * b.numerator)


def q_div_int(q: Fraction, n: int) -> Fraction:
    require_ok(is_strict_int(n) and n > 0, "BAD_DIVISOR", f"n={n!r}")
    return Fraction(q.numerator, q.denominator * n)


def contains_any_float(obj: Any) -> bool:
    if any(base.__name__ == "float" for base in type(obj).__mro__):
        return True
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if contains_any_float(k) or contains_any_float(v):
                return True
        return False
    if isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            if contains_any_float(item):
                return True
    return False


def _strip_nfkd_marks(raw: str) -> str:
    nfkd = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", raw))
    return "".join(ch for ch in nfkd if unicodedata.category(ch) not in ("Cf", "Mn", "Cc", "Cs", "Co"))


def _has_non_ascii_letter(norm: str) -> bool:
    return any(not ch.isascii() and ch.isalpha() and ch != "\u03bc" for ch in norm)


def is_forbidden_energy_value(raw: str) -> bool:
    cleaned = _strip_nfkd_marks(raw)
    if _UPPER_BARE_WJ_PREFIX_RE.search(cleaned):
        return True
    norm = cleaned.casefold()
    return _has_non_ascii_letter(norm) or bool(_FORBIDDEN_ENERGY_VALUE_RE.search(norm))


def is_forbidden_energy_key_name(raw: str) -> bool:
    cleaned = _strip_nfkd_marks(raw)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", cleaned).replace("_", " ")
    if is_forbidden_energy_value(spaced):
        return True
    tokens = _ALNUM_TOKEN_RE.findall(spaced.casefold())
    for tok in tokens:
        if tok in _FORBIDDEN_ENERGY_KEYS:
            return True
    return bool(_FORBIDDEN_COMPACT_KEY_RE.search("".join(tokens)))


def has_forbidden_energy_claim(obj: Any, *, _path: Tuple[str, ...] = ()) -> bool:
    if isinstance(obj, str):
        return is_forbidden_energy_value(obj)
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str):
                child = _path + (k,)
                if child == ("energy_power_status",) and v == ENERGY_POLICY_STATUS:
                    continue
                if len(child) == 2 and child[0] == "kernels" and k in _EXPECTED_KERNEL_SYMBOLS:
                    if has_forbidden_energy_claim(v, _path=child):
                        return True
                    continue
                if is_forbidden_energy_key_name(k):
                    return True
                if has_forbidden_energy_claim(v, _path=child):
                    return True
            else:
                if has_forbidden_energy_claim(v, _path=_path):
                    return True
        return False
    if isinstance(obj, (Sequence, Set)) and not isinstance(obj, (bytes, bytearray)):
        for item in obj:
            if has_forbidden_energy_claim(item, _path=_path):
                return True
    return False


def _strip_cpp_comments_and_strings(src: str) -> str:
    return _CPP_STRIP_RE.sub(" ", src)


def _extract_cpp_func_body(clean_src: str, header_re: Any) -> str:
    m = header_re.search(clean_src)
    if m is None:
        return ""
    start = m.end() - 1
    depth = 0
    for idx in range(start, len(clean_src)):
        ch = clean_src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return clean_src[start : idx + 1]
    return ""


def verify_cuda_scale_tokens(kernel_text: str, bench_text: str, study_text: str) -> Dict[str, bool]:
    k_clean = _strip_cpp_comments_and_strings(kernel_text)
    k_plain_body = _extract_cpp_func_body(k_clean, _KERNEL_PLAIN_HDR_RE)
    k_bent4_body = _extract_cpp_func_body(k_clean, _KERNEL_BENT4_HDR_RE)
    require_ok(
        bool(_KERNEL_PLAIN_SCALE_RE.search(k_plain_body))
        and bool(_KERNEL_BENT4_SCALE_RE.search(k_bent4_body))
        and bool(_KERNEL_V3_GLOBAL_RE.search(k_clean))
        and bool(_KERNEL_V4_GLOBAL_RE.search(k_clean)),
        "KERNEL_SCALE_MISMATCH",
        "kernel executable FWHT scale or entrypoint missing",
    )
    b_clean = _strip_cpp_comments_and_strings(bench_text)
    b_plain_body = _extract_cpp_func_body(b_clean, _BENCH_PLAIN_HDR_RE)
    b_bent4_body = _extract_cpp_func_body(b_clean, _BENCH_BENT4_HDR_RE)
    require_ok(
        bool(_BENCH_PLAIN_SCALE_RE.search(b_plain_body))
        and bool(_BENCH_BENT4_SCALE_RE.search(b_bent4_body)),
        "BENCH_SCALE_MISMATCH",
        "bench executable FWHT scale missing",
    )
    try:
        study_tree = ast.parse(study_text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise V4Monarch2VerifyError("STUDY_AST_INVALID", str(exc)) from exc

    pin_counts: Dict[str, int] = {k: 0 for k in _STUDY_EXPECTED_PINS}
    pin_values: Dict[str, str] = {}
    has_study_h256 = False
    has_study_sqrt32 = False
    for stmt in study_tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            tname = stmt.targets[0].id
            if tname in _STUDY_EXPECTED_PINS and isinstance(stmt.value, ast.Constant) and type(stmt.value.value) is str:
                pin_values[tname] = stmt.value.value
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "run_e2e_bpb_study":
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Constant) and repr(sub.value) == "0.0625":
                    has_study_h256 = True
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "math"
                    and sub.func.attr == "sqrt"
                    and len(sub.args) == 1
                    and isinstance(sub.args[0], ast.Constant)
                    and repr(sub.args[0].value) == "32.0"
                ):
                    has_study_sqrt32 = True

    for node in ast.walk(study_tree):
        require_ok(not isinstance(node, ast.Assert), "STUDY_BARE_ASSERT", "bare assert in BPB study")
        if isinstance(node, ast.Import):
            for alias in node.names:
                require_ok(alias.name.split(".", 1)[0] not in _STUDY_FORBIDDEN_IMPORTS, "STUDY_FORBIDDEN_IMPORT", alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            require_ok(node.module.split(".", 1)[0] not in _STUDY_FORBIDDEN_IMPORTS, "STUDY_FORBIDDEN_IMPORT", node.module)
        if isinstance(node, ast.Name):
            require_ok(node.id != "__import__", "STUDY_FORBIDDEN_IMPORT", "__import__")
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in pin_counts:
                    pin_counts[tgt.id] += 1
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in pin_counts:
            pin_counts[node.target.id] += 1

    require_ok(
        has_study_h256
        and has_study_sqrt32
        and all(pin_counts[k] == 1 and pin_values.get(k) == exp for k, exp in _STUDY_EXPECTED_PINS.items()),
        "STUDY_SCALE_MISMATCH",
        "BPB study executable scale or pinned corpus constants mismatch",
    )
    return {"cuda_and_study_scale_tokens_verified": True}


# ============================================================================
# Pure-Z Exact Oracles (H_256, FWHT_256 x bent4 + KATs, Monarch-2, SECDED)
# ============================================================================

def fwht256_plain_int(u256: Sequence[int]) -> List[int]:
    require_ok(len(u256) == 256, "BAD_FWHT_LEN", f"len={len(u256)}")
    buf = list(u256)
    step = 1
    while step < 256:
        stride = step << 1
        for base in range(0, 256, stride):
            for off in range(step):
                a = buf[base + off]
                b = buf[base + off + step]
                buf[base + off] = a + b
                buf[base + off + step] = a - b
        step = stride
    return buf


def fwht256_bent4_int(u256: Sequence[int]) -> List[int]:
    """Unnormalized integer FWHT_256 x bent4 (||y||_2^2 = 256 ||u||_2^2; CUDA multiplies by 1/sqrt(32))."""
    require_ok(len(u256) == 256, "BAD_FWHT_LEN", f"len={len(u256)}")
    buf: List[int] = [0] * 256
    for idx in range(256):
        w = idx >> 5
        lane = idx & 31
        src_w = w ^ (lane & 7)
        src_idx = (src_w << 5) | lane
        b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
        val = u256[src_idx]
        buf[idx] = -val if b4 else val
    return fwht256_plain_int(buf)


def monarch16_stage_allones_int(z_in: Sequence[int]) -> List[int]:
    require_ok(len(z_in) == 256, "BAD_MONARCH_LEN", f"len={len(z_in)}")
    z_out = [0] * 256
    for g in range(16):
        col_sum = sum(z_in[c * 16 + g] for c in range(16))
        for r in range(16):
            z_out[g * 16 + r] = col_sum
    return z_out


def a16_encode_u16(data11: int) -> int:
    require_ok(is_strict_int(data11) and 0 <= data11 < 2048, "BAD_DATA11", f"data11={data11!r}")
    dpos = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    w = 0
    for i, pos in enumerate(dpos):
        if (data11 >> i) & 1:
            w |= 1 << pos
    p1 = (w & 0xAAAA).bit_count() & 1
    p2 = (w & 0xCCCC).bit_count() & 1
    p3 = (w & 0xF0F0).bit_count() & 1
    p4 = (w & 0xFF00).bit_count() & 1
    w |= (p1 << 1) | (p2 << 2) | (p3 << 4) | (p4 << 8)
    p0 = w.bit_count() & 1
    return w | p0


def a16_syndrome_u5(w16: int) -> int:
    require_ok(is_strict_int(w16) and 0 <= w16 <= 0xFFFF, "BAD_W16", f"w16={w16!r}")
    p0 = w16.bit_count() & 1
    p1 = (w16 & 0xAAAA).bit_count() & 1
    p2 = (w16 & 0xCCCC).bit_count() & 1
    p3 = (w16 & 0xF0F0).bit_count() & 1
    p4 = (w16 & 0xFF00).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def verify_holosphere_v4_oracles(
    *,
    plain_fn: Any = fwht256_plain_int,
    bent4_fn: Any = fwht256_bent4_int,
    stage_fn: Any = monarch16_stage_allones_int,
    encode_fn: Any = a16_encode_u16,
    syndrome_fn: Any = a16_syndrome_u5,
    bos_byte: int = 33,
) -> Dict[str, Any]:
    u: List[int] = []
    rem = 3
    for _ in range(256):
        u.append(rem - 5)
        rem += 19
        while rem >= 11:
            rem -= 11
    h2 = plain_fn(plain_fn(u))
    require_ok(h2 == [256 * x for x in u], "ORACLE_H256_INVOLUTION", "H_256^2 != 256 I")

    y = bent4_fn(u)
    energy_u = sum(x * x for x in u)
    energy_y = sum(v * v for v in y)
    require_ok(energy_y == 256 * energy_u, "ORACLE_PARSEVAL_FAIL", "Parseval mismatch")

    kat_verified = 0
    for src_idx in _KAT_DELTA_INDICES:
        delta = [1 if i == src_idx else 0 for i in range(256)]
        got_kat = bent4_fn(delta)
        lane = src_idx & 31
        src_w = src_idx >> 5
        dst_idx = ((src_w ^ (lane & 7)) << 5) | lane
        b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
        expected_kat = [(-1 if (((k & dst_idx).bit_count() ^ b4) & 1) else 1) for k in range(256)]
        require_ok(got_kat == expected_kat, "ORACLE_BENT4_KAT", f"src_idx={src_idx}")
        kat_verified += 1

    # Algebraic equivalence between study BOS=256 (sp_flip=(d>>5)&1) and uint8_t byte 33 (0x21)
    bos_equiv = all(
        ((bos_byte & (d ^ ((d & 7) << 5))).bit_count() & 1) == ((d >> 5) & 1)
        for d in range(256)
    )
    require_ok(bos_equiv, "ORACLE_BOS_EQUIV", "BOS-256 != byte-33")

    e0 = [1 if i == 0 else 0 for i in range(256)]
    z1 = stage_fn(e0)
    r1 = sum(1 for v in z1 if v != 0)
    z2 = stage_fn(z1)
    r2 = sum(1 for v in z2 if v != 0)
    require_ok(r1 == 16 and r2 == 256, "ORACLE_MONARCH2_REACH", f"reach=({r1},{r2})")

    cw = encode_fn(0x5A3)
    require_ok(syndrome_fn(cw) == 0, "ORACLE_CLEAN_SYND", "clean syndrome != 0")
    sec_healed = 0
    for bit in range(16):
        cor = cw ^ (1 << bit)
        s = syndrome_fn(cor)
        require_ok((s & 1) == 1 and (cor ^ (1 << ((s >> 1) & 15))) == cw, "ORACLE_SEC_HEAL", f"bit={bit}")
        sec_healed += 1
    ded_detected = 0
    for b1, b2 in _DED_BIT_PAIRS:
        s_ded = syndrome_fn(cw ^ (1 << b1) ^ (1 << b2))
        require_ok((s_ded & 1) == 0 and s_ded != 0, "ORACLE_DED_DETECT", f"pair=({b1},{b2})")
        ded_detected += 1
    return {
        "h256_involution_verified": True,
        "bent4_parseval_input_l2_sq": energy_u,
        "bent4_parseval_output_l2_sq": energy_y,
        "bent4_impulse_kats_verified": kat_verified,
        "bos_256_matches_byte_33": bos_equiv,
        "monarch1_reach_count": r1,
        "monarch2_full_reach_count": r2,
        "secded_sec_healed": sec_healed,
        "secded_ded_detected": ded_detected,
        "v4_hbm_vocab_bytes": 0,
    }


# ============================================================================
# Self-Bound On-Disk Receipt Verifier (PTXAS + Live Results + Gate Re-Grading)
# ============================================================================

def verify_ptxas_receipt(ptx: Any, *, expected_cu_sha: Optional[str] = None) -> Dict[str, Any]:
    require_ok(type(ptx) is dict, "PTXAS_TYPE", "ptxas must be dict")
    require_ok(ptx.get("schema") == SCHEMA_PTXAS, "PTXAS_SCHEMA", "bad ptxas schema")
    require_ok(
        ptx.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_claim(ptx),
        "PTXAS_ENERGY_POLICY",
        "ptxas violates energy policy",
    )
    require_ok(not contains_any_float(ptx), "PTXAS_FLOAT_FORBIDDEN", "ptxas contains float")
    require_ok(frozenset(ptx.keys()) == _PTXAS_KEYS, "PTXAS_KEYS", "bad ptxas keys")
    rsha = ptx["receipt_sha256"]
    require_ok(is_hex64(rsha), "PTXAS_SHA_FORMAT", "bad receipt_sha256")
    require_ok(canonical_sha256(ptx, exclude_key="receipt_sha256") == rsha, "PTXAS_SHA_MISMATCH", "sha mismatch")
    require_ok(ptx["arch"] == "sm_86", "PTXAS_ARCH", "arch != sm_86")
    require_ok(ptx["compile_flags"] == EXPECTED_COMPILE_FLAGS, "PTXAS_FLAGS", "compile_flags mismatch")
    require_ok(type(ptx["nvcc_version"]) is str and len(ptx["nvcc_version"]) > 0, "PTXAS_NVCC", "bad nvcc_version")
    require_ok(ptx["cu_filename"] == KERNEL_CU_REL, "PTXAS_CU_FILE", "bad cu_filename")
    cu_sha = ptx["cu_sha256"]
    require_ok(is_hex64(cu_sha), "PTXAS_CU_SHA", "bad cu_sha256")
    if expected_cu_sha is not None:
        require_ok(cu_sha == expected_cu_sha, "PTXAS_CU_SHA_DISK", "cu_sha256 disk mismatch")

    kernels = ptx["kernels"]
    require_ok(type(kernels) is dict and len(kernels) == 2, "PTXAS_KERNELS", "expected 2 kernels")
    require_ok(frozenset(kernels.keys()) == _EXPECTED_KERNEL_SYMBOLS, "PTXAS_REQUIRED_SYMBOLS", "symbols mismatch")

    nonzero = 0
    for name, st in kernels.items():
        require_ok(type(st) is dict and frozenset(st.keys()) == _KERNEL_STAT_KEYS, "PTXAS_KSTAT", name)
        for field in _KERNEL_STAT_KEYS:
            require_ok(is_strict_int(st[field]) and st[field] >= 0, "PTXAS_FIELD_TYPE", f"{name}.{field}")
        if (
            st["stack_bytes"] != 0
            or st["spill_store_bytes"] != 0
            or st["spill_load_bytes"] != 0
            or st["ldl_insns"] != 0
            or st["stl_insns"] != 0
            or st["registers"] <= 0
        ):
            nonzero += 1
        exp_regs, exp_bar, exp_smem, exp_cas, exp_or, exp_atoms = _EXPECTED_KERNEL_SPECS[name]
        require_ok(
            st["registers"] == exp_regs
            and st["barriers"] == exp_bar
            and st["smem_bytes"] == exp_smem
            and st["atom_cas_sys_insns"] == exp_cas
            and st["red_or_sys_insns"] == exp_or
            and st["atoms_smem_insns"] == exp_atoms,
            "PTXAS_KERNEL_SPEC",
            name,
        )
    require_ok(
        ptx["total_kernels"] == 2
        and ptx["zero_spill_kernels"] == 2
        and ptx["nonzero_spill_kernels"] == 0
        and nonzero == 0,
        "PTXAS_SPILL_FAIL",
        f"nonzero={nonzero}",
    )
    return ptx


def verify_results_receipt(
    res: Any,
    ptx: Dict[str, Any],
    *,
    expected_bench_cu_sha: Optional[str] = None,
    expected_bpb_study_sha: Optional[str] = None,
) -> Dict[str, Any]:
    require_ok(type(res) is dict, "RESULTS_TYPE", "results must be dict")
    require_ok(res.get("schema") == SCHEMA_RESULTS, "RESULTS_SCHEMA", "bad schema")
    require_ok(
        res.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_claim(res),
        "RESULTS_ENERGY_POLICY",
        "results violates energy policy",
    )
    require_ok(not contains_any_float(res), "RESULTS_FLOAT_FORBIDDEN", "results contains float")
    require_ok(frozenset(res.keys()) == _RESULTS_KEYS, "RESULTS_KEYS", "bad results keys")

    rsha = res["result_sha256"]
    require_ok(is_hex64(rsha), "RESULTS_SHA_FORMAT", "bad result_sha256")
    require_ok(canonical_sha256(res, exclude_key="result_sha256") == rsha, "RESULTS_SHA_MISMATCH", "sha mismatch")
    require_ok(type(res["scope_note"]) is str and len(res["scope_note"]) > 0, "RESULTS_SCOPE", "bad scope_note")

    hw = res["hardware"]
    require_ok(
        type(hw) is dict
        and frozenset(hw.keys()) == _HARDWARE_KEYS
        and type(hw["device"]) is str
        and len(hw["device"]) > 0
        and hw["sm"] == "8.6"
        and hw["nvcc_version"] == ptx["nvcc_version"],
        "RESULTS_HARDWARE",
        "invalid hardware block",
    )

    sb = res["source_bindings"]
    require_ok(type(sb) is dict and frozenset(sb.keys()) == _SOURCE_BINDINGS_KEYS, "RESULTS_BINDINGS", "bad source_bindings")
    require_ok(
        sb["kernel_cu_filename"] == KERNEL_CU_REL
        and sb["bench_cu_filename"] == BENCH_CU_REL
        and sb["bpb_study_script_filename"] == BPB_STUDY_REL
        and sb["ptxas_receipt_filename"] == PTXAS_REL,
        "RESULTS_BINDING_FILES",
        "bad binding filenames",
    )
    require_ok(sb["kernel_cu_sha256"] == ptx["cu_sha256"], "RESULTS_CU_BIND", "kernel cu sha mismatch")
    require_ok(sb["ptxas_receipt_sha256"] == ptx["receipt_sha256"], "RESULTS_PTX_BIND", "ptxas sha mismatch")
    require_ok(is_hex64(sb["bench_cu_sha256"]), "RESULTS_BENCH_SHA", "bad bench_cu_sha256")
    if expected_bench_cu_sha is not None:
        require_ok(sb["bench_cu_sha256"] == expected_bench_cu_sha, "RESULTS_BENCH_DISK", "bench disk mismatch")
    require_ok(is_hex64(sb["bpb_study_script_sha256"]), "RESULTS_STUDY_SCRIPT_SHA", "bad bpb_study_script_sha256")
    if expected_bpb_study_sha is not None:
        require_ok(
            sb["bpb_study_script_sha256"] == expected_bpb_study_sha,
            "RESULTS_STUDY_SCRIPT_DISK",
            "bpb study script disk mismatch",
        )
    require_ok(
        sb["corpus_id"] == EXPECTED_CORPUS_ID
        and sb["corpus_manifest_sha256"] == EXPECTED_CORPUS_MANIFEST_SHA256
        and sb["corpus_train_bin_sha256"] == EXPECTED_CORPUS_TRAIN_SHA256
        and sb["corpus_eval_bin_sha256"] == EXPECTED_CORPUS_EVAL_SHA256,
        "RESULTS_CORPUS_BIND",
        "corpus SHA-256 binding mismatch",
    )

    sil = res["silicon_e2e_bench"]
    require_ok(type(sil) is dict and frozenset(sil.keys()) == _SILICON_KEYS, "RESULTS_SILICON", "bad silicon_e2e_bench")
    for k in ("clean_packed_u32", "clean_sys_atomics", "v4_sec16_packed_u32", "v4_sec16_sys_atomics"):
        require_ok(is_strict_int(sil[k]) and sil[k] >= 0, "RESULTS_SIL_INT", k)
    e_v3_1 = parse_q(sil["v3_1l_fp64_max_err_q"], code="RESULTS_SIL_Q")
    e_v4_1 = parse_q(sil["v4_1l_fp64_max_err_q"], code="RESULTS_SIL_Q")
    e_v4_4 = parse_q(sil["v4_4l_fp64_max_err_q"], code="RESULTS_SIL_Q")
    e_sec_h = parse_q(sil["v4_sec16_hidden_max_abs_err_q"], code="RESULTS_SIL_Q")
    e_sec_l = parse_q(sil["v4_sec16_logits_max_abs_err_q"], code="RESULTS_SIL_Q")
    us_v3_1 = parse_q(sil["v3_scalar_1l_us_per_tok_q"], code="RESULTS_SIL_Q")
    us_v4_1 = parse_q(sil["v4_monarch2_1l_us_per_tok_q"], code="RESULTS_SIL_Q")
    tps_v4_1 = parse_q(sil["v4_monarch2_1l_tok_per_s_q"], code="RESULTS_SIL_Q")
    us_v3_4 = parse_q(sil["v3_scalar_4l_us_per_tok_q"], code="RESULTS_SIL_Q")
    us_v4_4 = parse_q(sil["v4_monarch2_4l_us_per_tok_q"], code="RESULTS_SIL_Q")
    tps_v4_4 = parse_q(sil["v4_monarch2_4l_tok_per_s_q"], code="RESULTS_SIL_Q")
    for ev in (e_v3_1, e_v4_1, e_v4_4):
        require_ok(Fraction(0, 1) <= ev < Fraction(1, 100000), "RESULTS_PARITY_BOUND", f"err={ev}")
    for uv in (us_v3_1, us_v4_1, us_v3_4, us_v4_4):
        require_ok(uv > Fraction(0, 1), "RESULTS_US_POS", f"us={uv}")
    require_ok(tps_v4_1 == q_div(Fraction(1000000, 1), us_v4_1), "RESULTS_TPS_1L", "1L tok/s mismatch")
    require_ok(tps_v4_4 == q_div(Fraction(1000000, 1), us_v4_4), "RESULTS_TPS_4L", "4L tok/s mismatch")

    study = res["e2e_bpb_study"]
    require_ok(type(study) is dict and frozenset(study.keys()) == _STUDY_KEYS, "RESULTS_STUDY", "bad e2e_bpb_study")
    expected_seeds = study["seeds"]
    require_ok(
        expected_seeds == [42, 137, 2026]
        and all(is_strict_int(s) for s in expected_seeds)
        and study["d_model"] == 512
        and study["rope_base"] == 500000
        and study["eval_contexts"] == _EXPECTED_CTX_META
        and study["tokens_per_seed"] == study["batch_size"] * study["train_seq_len"] * study["steps_per_seed"],
        "RESULTS_STUDY_META",
        "invalid study metadata",
    )
    h256_err = parse_q(study["h256_ortho_fp64_max_err_q"], code="RESULTS_H256_ERR")
    require_ok(Fraction(0, 1) <= h256_err <= Fraction(1, 100000000000), "RESULTS_H256_BOUND", f"err={h256_err}")

    arms = study["arms"]
    require_ok(type(arms) is dict and frozenset(arms.keys()) == frozenset(_EXPECTED_ARMS), "RESULTS_ARMS", "bad arms")
    arm_means: Dict[str, Dict[str, Fraction]] = {}
    arm_seeds: Dict[str, Dict[str, List[Fraction]]] = {}
    for arm_name in _EXPECTED_ARMS:
        entry = arms[arm_name]
        require_ok(type(entry) is dict and frozenset(entry.keys()) == _ARM_KEYS, "RESULTS_ARM_TYPE", arm_name)
        exp_p, exp_b, exp_in, exp_out = _EXPECTED_HEAD_SPECS[arm_name]
        require_ok(
            entry["head_params"] == exp_p
            and entry["head_fp16_bytes"] == exp_b
            and entry["hbm_input_vocab_bytes"] == exp_in
            and entry["hbm_output_vocab_bytes"] == exp_out
            and is_strict_int(entry["total_params"])
            and entry["total_params"] > 0,
            "RESULTS_ARM_BYTES",
            arm_name,
        )
        seeds_list = entry["seeds"]
        require_ok(type(seeds_list) is list and len(seeds_list) == 3, "RESULTS_ARM_SEEDS", arm_name)
        row_seeds: List[int] = []
        per_ctx_vals: Dict[str, List[Fraction]] = {ck: [] for ck in _EXPECTED_CONTEXTS}
        for row in seeds_list:
            require_ok(
                type(row) is dict and frozenset(row.keys()) == _SEED_KEYS and is_strict_int(row["seed"]),
                "RESULTS_SEED_ROW",
                arm_name,
            )
            row_seeds.append(row["seed"])
            wall_s = parse_q(row["train_wall_s_q"], code="RESULTS_SEED_WALL")
            require_ok(wall_s > Fraction(0, 1), "RESULTS_SEED_WALL_POS", arm_name)
            ctxs = row["contexts"]
            require_ok(type(ctxs) is dict and frozenset(ctxs.keys()) == frozenset(_EXPECTED_CONTEXTS), "RESULTS_SEED_CTXS", arm_name)
            for ck in _EXPECTED_CONTEXTS:
                c_row = ctxs[ck]
                require_ok(
                    type(c_row) is dict
                    and frozenset(c_row.keys()) == _CTX_EVAL_KEYS
                    and is_strict_int(c_row["eval_tokens"])
                    and c_row["eval_tokens"] == 65536,
                    "RESULTS_CTX_ROW",
                    f"{arm_name}.{ck}",
                )
                v_fp16 = parse_q(c_row["eval_bpb_q"], code="RESULTS_CTX_BPB")
                v_bf16 = parse_q(c_row["eval_bpb_bf16_q"], code="RESULTS_CTX_BPB")
                qdiff = parse_q(c_row["fp16_quant_bpb_diff_q"], code="RESULTS_CTX_BPB")
                require_ok(v_fp16 > Fraction(0, 1) and v_bf16 > Fraction(0, 1), "RESULTS_CTX_BPB_POS", f"{arm_name}.{ck}")
                require_ok(qdiff == abs(v_fp16 - v_bf16), "RESULTS_CTX_QDIFF", f"{arm_name}.{ck}")
                per_ctx_vals[ck].append(v_fp16)
        require_ok(row_seeds == expected_seeds, "RESULTS_SEED_ORDER", arm_name)
        stored_means = entry["mean_eval_bpb_by_ctx_q"]
        require_ok(type(stored_means) is dict and frozenset(stored_means.keys()) == frozenset(_EXPECTED_CONTEXTS), "RESULTS_MEANS_KEYS", arm_name)
        arm_means[arm_name] = {}
        for ck in _EXPECTED_CONTEXTS:
            calc_m = q_div_int(sum(per_ctx_vals[ck], Fraction(0, 1)), 3)
            require_ok(parse_q(stored_means[ck], code="RESULTS_MEAN_Q") == calc_m, "RESULTS_MEAN_MISMATCH", f"{arm_name}.{ck}")
            arm_means[arm_name][ck] = calc_m
        arm_seeds[arm_name] = per_ctx_vals

    cs = res["context_summary"]
    require_ok(type(cs) is dict and frozenset(cs.keys()) == frozenset(_EXPECTED_CONTEXTS), "RESULTS_CS", "bad context_summary")
    g1_calc = True
    for ck in _EXPECTED_CONTEXTS:
        crow = cs[ck]
        require_ok(type(crow) is dict and frozenset(crow.keys()) == _CTX_SUMMARY_KEYS, "RESULTS_CS_ROW", ck)
        m_sc = arm_means["V3_ScalarFWHT"][ck]
        m_v4 = arm_means["V4_Monarch2BiasFWHT"][ck]
        m_lr = arm_means["V3_LowRank16Bias"][ck]
        m_dn = arm_means["V3_DenseHead"][ck]
        lift_sc = m_sc - m_v4
        lift_lr = m_lr - m_v4
        gap_dn = m_v4 - m_dn
        dense_span = m_sc - m_dn
        gap_closed = q_div(lift_sc, dense_span) if dense_span > Fraction(0, 1) else Fraction(0, 1)
        lt_sc = all(a < b for a, b in zip(arm_seeds["V4_Monarch2BiasFWHT"][ck], arm_seeds["V3_ScalarFWHT"][ck]))
        lt_lr = all(a < b for a, b in zip(arm_seeds["V4_Monarch2BiasFWHT"][ck], arm_seeds["V3_LowRank16Bias"][ck]))

        require_ok(parse_q(crow["v4_lift_over_scalar_q"]) == lift_sc, "RESULTS_CS_LIFT_SC", ck)
        require_ok(parse_q(crow["v4_lift_over_lowrank16_q"]) == lift_lr, "RESULTS_CS_LIFT_LR", ck)
        require_ok(parse_q(crow["v4_minus_dense_q"]) == gap_dn, "RESULTS_CS_GAP_DN", ck)
        require_ok(parse_q(crow["dense_gap_scalar_minus_dense_q"]) == dense_span, "RESULTS_CS_SPAN", ck)
        require_ok(parse_q(crow["v4_gap_closed_fraction_q"]) == gap_closed, "RESULTS_CS_GCLOSED", ck)
        require_ok(crow["v4_lt_scalar_all_seeds"] is lt_sc and crow["v4_lt_lowrank16_all_seeds"] is lt_lr, "RESULTS_CS_FLAGS", ck)

        if not (lift_sc >= Fraction(1, 2) and lt_lr):
            g1_calc = False
        if ck in ("T2048", "T8192") and not (gap_dn <= Fraction(1, 5)):
            g1_calc = False

    g2_calc = (
        ptx["nonzero_spill_kernels"] == 0
        and ptx["total_kernels"] == 2
        and sil["clean_packed_u32"] == 0
        and sil["clean_sys_atomics"] == 0
        and sil["v4_sec16_packed_u32"] == 16
        and sil["v4_sec16_sys_atomics"] == 1
        and e_sec_h == Fraction(0, 1)
        and e_sec_l == Fraction(0, 1)
        and us_v4_1 <= Fraction(25, 1)
    )
    gates = res["gates"]
    require_ok(
        type(gates) is dict
        and frozenset(gates.keys()) == _GATES_KEYS
        and gates["G1_e2e_multicontext_quality"] is g1_calc
        and gates["G2_fused_sm86_zero_spill_and_latency"] is g2_calc,
        "RESULTS_GATES_MISMATCH",
        "recomputed gates mismatch",
    )
    if g1_calc and g2_calc:
        expected_verdict = "CONFIRMED"
    elif g1_calc or g2_calc:
        expected_verdict = "PARTIAL"
    else:
        expected_verdict = "FALSIFIED"
    require_ok(res["verdict"] == expected_verdict, "RESULTS_VERDICT_MISMATCH", f"expected={expected_verdict}")
    return res


def verify_on_disk(repo_root: Path) -> Dict[str, Any]:
    verify_pure_q_ast(decode_utf8_bytes(read_regular_bytes(Path(__file__))))
    kernel_bytes = read_repo_regular_bytes(repo_root, KERNEL_CU_REL)
    bench_bytes = read_repo_regular_bytes(repo_root, BENCH_CU_REL)
    study_bytes = read_repo_regular_bytes(repo_root, BPB_STUDY_REL)
    ptx_bytes = read_repo_regular_bytes(repo_root, PTXAS_REL)
    res_bytes = read_repo_regular_bytes(repo_root, RESULTS_REL)

    kernel_sha = hashlib.sha256(kernel_bytes).hexdigest()
    bench_sha = hashlib.sha256(bench_bytes).hexdigest()
    study_sha = hashlib.sha256(study_bytes).hexdigest()
    scale_check = verify_cuda_scale_tokens(
        decode_utf8_bytes(kernel_bytes),
        decode_utf8_bytes(bench_bytes),
        decode_utf8_bytes(study_bytes),
    )
    ptx = loads_strict_canonical_json_bytes(ptx_bytes, PTXAS_REL)
    res = loads_strict_canonical_json_bytes(res_bytes, RESULTS_REL)

    verify_ptxas_receipt(ptx, expected_cu_sha=kernel_sha)
    verify_results_receipt(
        res,
        ptx,
        expected_bench_cu_sha=bench_sha,
        expected_bpb_study_sha=study_sha,
    )
    oracles = verify_holosphere_v4_oracles()

    summary: Dict[str, Any] = {
        "schema": SCHEMA_VERIFY,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "status": "PASS",
        "verdict": res["verdict"],
        "gates": res["gates"],
        "kernel_cu_sha256": kernel_sha,
        "bench_cu_sha256": bench_sha,
        "bpb_study_script_sha256": study_sha,
        "ptxas_file_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
        "results_file_sha256": hashlib.sha256(res_bytes).hexdigest(),
        "ptxas_receipt_sha256": ptx["receipt_sha256"],
        "result_sha256": res["result_sha256"],
        "scale_alignment": scale_check,
        "holosphere_v4_oracles": oracles,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pure-Q / Pure-Z Verifier for HoloSphere-V4 Monarch-2 E2E")
    parser.add_argument("--check", action="store_true", help="Verify on-disk receipts and oracles")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary")
    parser.add_argument("--repo-root", type=str, default=None, help="Override repository root directory")
    parser.add_argument("--require-verdict", type=str, default=None, help="Fail closed unless verdict matches")
    args = parser.parse_args(argv)

    root = Path(args.repo_root) if args.repo_root is not None else Path(__file__).absolute().parent.parent.parent
    try:
        summary = verify_on_disk(root)
        if args.require_verdict is not None:
            require_ok(
                summary["verdict"] == args.require_verdict,
                "VERDICT_REQUIREMENT_FAILED",
                f"expected={args.require_verdict} got={summary['verdict']}",
            )
    except V4Monarch2VerifyError as exc:
        sys.stderr.write(json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        return 2
    except (OSError, ValueError, RecursionError, MemoryError) as exc:
        sys.stderr.write(json.dumps({"status": "ERROR", "code": "UNEXPECTED_RUNTIME_ERROR", "message": str(exc)}, sort_keys=True) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"PASS {summary['verdict']} result_sha256={summary['result_sha256']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
