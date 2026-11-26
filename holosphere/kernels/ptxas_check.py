#!/usr/bin/env python3
"""Offline 40-kernel Ampere sm_86 ptxas zero-spill compiler & receipt checker (pure stdlib).

Covers the 3 core HoloSphere CUDA translation units (40 kernels total):
  1. `su2_holonomy_scan_sm86.cu`        (27 kernels)
  2. `holosphere_h256_chunk32_sm86.cu`  (11 kernels, receipt-pinned in `ptxas_s117_zero_spill_receipt.json`)
  3. `holosphere_v3_frontier_sm86.cu`   ( 2 kernels, receipt-pinned in `ptxas_v3_frontier_zero_spill_receipt.json`)

Zero GPU touch, zero torch dependency, zero bare asserts. Compiles into an ephemeral
temporary directory so no `.so`/`.o` binaries are ever left in the repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holosphere.chunk32_ops import (
    DEFAULT_CSRC_DIR,
    verify_s117_ptxas_receipt,
    verify_v3_frontier_ptxas_receipt,
)
from holosphere.errors import HoloSphereError, require_ok

DEFAULT_INCLUDE_DIR = REPO_ROOT / "include"

CORE_40_CU_SPECS: Tuple[Tuple[str, int], ...] = (
    ("su2_holonomy_scan_sm86.cu", 27),
    ("holosphere_h256_chunk32_sm86.cu", 11),
    ("holosphere_v3_frontier_sm86.cu", 2),
)

_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)'")
_SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_REGS_RE = re.compile(r"Used (\d+) registers")


def parse_ptxas_verbose(stderr: str, *, expected_kernels: int | None = None) -> Dict[str, Any]:
    """Parse `nvcc --ptxas-options=-v` output and fail closed on any stack frame or spill."""
    require_ok(isinstance(stderr, str), "ptxas stderr must be str", code="PTXAS_BAD_LOG")
    kernels: List[Dict[str, Any]] = []
    cur: Dict[str, Any] | None = None
    for line in stderr.splitlines():
        m_ent = _ENTRY_RE.search(line)
        if m_ent is not None:
            cur = {
                "symbol": m_ent.group(1),
                "registers": None,
                "stack_frame_bytes": None,
                "spill_store_bytes": None,
                "spill_load_bytes": None,
            }
            kernels.append(cur)
            continue
        m_sp = _SPILL_RE.search(line)
        if m_sp is not None:
            sf, ss, sl = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
            require_ok(
                sf == 0 and ss == 0 and sl == 0,
                f"Non-zero stack/spill detected in {cur['symbol'] if cur is not None else '<orphan>'}: {sf}/{ss}/{sl}",
                code="PTXAS_SPILL_DETECTED",
            )
            require_ok(cur is not None, "Spill stats appeared before any entry function header", code="PTXAS_ORPHAN_STATS")
            require_ok(
                cur["stack_frame_bytes"] is None,
                f"Duplicate spill stats for {cur['symbol']}",
                code="PTXAS_DUPLICATE_STATS",
            )
            cur["stack_frame_bytes"] = sf
            cur["spill_store_bytes"] = ss
            cur["spill_load_bytes"] = sl
        m_rg = _REGS_RE.search(line)
        if m_rg is not None:
            require_ok(cur is not None, "Register stats appeared before any entry function header", code="PTXAS_ORPHAN_STATS")
            require_ok(
                cur["registers"] is None,
                f"Duplicate register stats for {cur['symbol']}",
                code="PTXAS_DUPLICATE_STATS",
            )
            cur["registers"] = int(m_rg.group(1))

    require_ok(len(kernels) > 0, "No compiled kernels found in ptxas output", code="PTXAS_NO_KERNELS")
    if expected_kernels is not None:
        require_ok(
            len(kernels) == expected_kernels,
            f"Expected {expected_kernels} kernels, parsed {len(kernels)}",
            code="PTXAS_COUNT_MISMATCH",
        )

    zero_spill = 0
    max_regs = 0
    for k in kernels:
        require_ok(
            k["stack_frame_bytes"] is not None and k["registers"] is not None,
            f"Incomplete ptxas stats for {k['symbol']}",
            code="PTXAS_INCOMPLETE_STATS",
        )
        zero_spill += 1
        if k["registers"] > max_regs:
            max_regs = k["registers"]

    return {
        "total_kernels": len(kernels),
        "zero_spill_kernels": zero_spill,
        "max_registers": max_regs,
        "kernels": kernels,
    }


def verify_all_40_receipts(csrc_dir: Path = DEFAULT_CSRC_DIR) -> Dict[str, Any]:
    """Verify the pinned S117 (11) & V3 Frontier (2) receipts and core SU(2) (27) CUDA source presence."""
    su2_path = csrc_dir / "su2_holonomy_scan_sm86.cu"
    require_ok(su2_path.is_file(), f"Missing {su2_path}", code="PTXAS_NO_CU")
    su2_sha = hashlib.sha256(su2_path.read_bytes()).hexdigest()

    s117 = verify_s117_ptxas_receipt(csrc_dir)
    v3 = verify_v3_frontier_ptxas_receipt(csrc_dir)

    receipt_zero = int(s117["zero_spill_kernels"]) + int(v3["zero_spill_kernels"])
    return {
        "schema": "holosphere.ptxas_40_receipts_verified.v1",
        "receipt_toolchain": "nvcc 13.4.92 (Build cuda_13.4.r13.4/compiler.38855100_0, sm_86)",
        "toolchain_scope_note": (
            "Pinned ptxas register/SASS counts are scoped to nvcc 13.4.92; other CUDA 13.x "
            "toolchains (e.g. 13.0/13.2) may vary register counts by +/-1 while preserving 0 B spill."
        ),
        "total_kernels_catalogued": 40,
        "receipt_verified_zero_spill_kernels": receipt_zero,
        "core_su2_cu_sha256": su2_sha,
        "core_su2_expected_kernels": 27,
        "s117_receipt": s117,
        "v3_frontier_receipt": v3,
    }


def compile_all_40_kernels(
    csrc_dir: Path = DEFAULT_CSRC_DIR,
    include_dir: Path = DEFAULT_INCLUDE_DIR,
    *,
    nvcc: str = "nvcc",
    arch: str = "sm_86",
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Compile all 3 HoloSphere CUDA files (40 kernels) into a tempdir and enforce 40/40 0 B spills."""
    require_ok(isinstance(nvcc, str) and nvcc.strip() != "", "nvcc path must be non-empty str", code="NVCC_BAD_ARG")
    require_ok(isinstance(arch, str) and arch.startswith("sm_"), f"invalid arch {arch!r}", code="NVCC_BAD_ARCH")
    try:
        v_proc = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30.0, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        require_ok(False, f"Failed to execute {nvcc!r}: {exc}", code="NVCC_UNAVAILABLE")
        return {}
    require_ok(v_proc.returncode == 0, f"nvcc --version exited {v_proc.returncode}", code="NVCC_VERSION_FAIL")
    version_lines = [ln.strip() for ln in v_proc.stdout.splitlines() if ln.strip()]
    nvcc_version = version_lines[-1] if version_lines else "unknown"

    units: List[Dict[str, Any]] = []
    total_kernels = 0
    total_zero = 0
    with tempfile.TemporaryDirectory(prefix="holosphere_ptxas40_") as td:
        for cu_name, exp_n in CORE_40_CU_SPECS:
            cu_path = csrc_dir / cu_name
            require_ok(cu_path.is_file(), f"Missing {cu_path}", code="PTXAS_NO_CU")
            cu_sha = hashlib.sha256(cu_path.read_bytes()).hexdigest()
            out_so = Path(td) / f"{cu_path.stem}.so"
            cmd = [
                nvcc,
                "-O3",
                "-std=c++17",
                f"-arch={arch}",
                f"-I{include_dir}",
                "--ptxas-options=-v",
                "-Xcompiler",
                "-fPIC",
                "-shared",
                str(cu_path),
                "-o",
                str(out_so),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                require_ok(False, f"nvcc compile failed for {cu_name}: {exc}", code="NVCC_COMPILE_ERROR")
                return {}
            require_ok(
                proc.returncode == 0,
                f"nvcc exited {proc.returncode} on {cu_name}: {proc.stderr[:400]}",
                code="NVCC_COMPILE_FAIL",
            )
            parsed = parse_ptxas_verbose(proc.stderr, expected_kernels=exp_n)
            total_kernels += parsed["total_kernels"]
            total_zero += parsed["zero_spill_kernels"]
            units.append(
                {
                    "cu_filename": cu_name,
                    "cu_sha256": cu_sha,
                    "total_kernels": parsed["total_kernels"],
                    "zero_spill_kernels": parsed["zero_spill_kernels"],
                    "max_registers": parsed["max_registers"],
                }
            )

    return {
        "schema": "holosphere.ptxas_40_compile_verified.v1",
        "arch": arch,
        "nvcc_version": nvcc_version,
        "total_kernels": total_kernels,
        "zero_spill_kernels": total_zero,
        "units": units,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = False
    do_compile = False
    do_receipts = False
    nvcc = "nvcc"
    arch = "sm_86"
    i = 0
    try:
        while i < len(args):
            tok = args[i]
            if tok == "--json":
                as_json = True
                i += 1
            elif tok == "--compile":
                do_compile = True
                i += 1
            elif tok == "--verify-receipts":
                do_receipts = True
                i += 1
            elif tok == "--nvcc":
                require_ok(i + 1 < len(args), "--nvcc requires a value", code="CLI_BAD_ARG")
                nvcc = args[i + 1]
                i += 2
            elif tok == "--arch":
                require_ok(i + 1 < len(args), "--arch requires a value", code="CLI_BAD_ARG")
                arch = args[i + 1]
                i += 2
            else:
                require_ok(False, f"Unrecognized argument: {tok}", code="CLI_BAD_ARG")
        if not do_compile and not do_receipts:
            do_receipts = True
        out: Dict[str, Any] = {}
        if do_receipts:
            out["receipts"] = verify_all_40_receipts()
        if do_compile:
            out["compile"] = compile_all_40_kernels(nvcc=nvcc, arch=arch)
        if as_json:
            sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write("OK ptxas_check\n")
        return 0
    except HoloSphereError as exc:
        sys.stderr.write(f"ERROR {exc.code}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
