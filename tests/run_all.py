#!/usr/bin/env python3
"""Deterministic 3-mode isolated (-I) test runner for HoloSphere (zero bare asserts, CPU-only)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py", top_level_dir=str(REPO_ROOT))
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)
    ok = result.wasSuccessful() and len(result.skipped) == 0
    summary = {
        "schema": "holosphere.tests.run_all.v1",
        "passed": ok,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    summary["summary_sha256"] = hashlib.sha256(canon).hexdigest()
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    if not ok:
        sys.stderr.write(stream.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
