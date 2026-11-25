#!/usr/bin/env python3
"""Repository hygiene & 3-mode isolated -I parity gate for HoloSphere (zero bare asserts)."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FORBIDDEN_WEIGHT_EXTS = frozenset({".pt", ".pth", ".safetensors", ".bin", ".gguf", ".ckpt"})


class TestRepoHygiene(unittest.TestCase):
    def test_01_zero_bare_asserts_zero_leaks_zero_weights_zero_pycache(self) -> None:
        home_re = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")
        audit_tag_re = re.compile(r"\bAUDIT-[0-9]+[A-Za-z0-9_-]*\b")
        checked_py = 0
        for p in sorted(REPO_ROOT.rglob("*")):
            if ".git" in p.parts:
                continue
            self.assertNotEqual(p.name, "__pycache__", f"__pycache__ found at {p}")
            if not p.is_file():
                continue
            self.assertNotIn(p.suffix.lower(), FORBIDDEN_WEIGHT_EXTS, f"weight file forbidden: {p}")
            if p.suffix in (".py", ".md", ".toml", ".yml", ".yaml", ".cu", ".h", ".c"):
                txt = p.read_text(encoding="utf-8")
                self.assertNotIn("\r\n", txt, f"CRLF detected in {p}")
                if p.name != "test_repo_hygiene.py":
                    self.assertIsNone(home_re.search(txt), f"home path leak in {p}")
                    self.assertIsNone(audit_tag_re.search(txt), f"internal AUDIT tag leak in {p}")
            if p.suffix == ".py":
                checked_py += 1
                tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
                bare_asserts = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
                self.assertEqual(bare_asserts, [], f"bare assert in {p}: {bare_asserts}")
        self.assertGreaterEqual(checked_py, 15)

    def test_02_prereg_s114_preflight_3mode_isolated_bit_identical(self) -> None:
        script = str(REPO_ROOT / "holosphere" / "benchmarks" / "prereg_s114_suite.py")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        digests: list[str] = []
        for flags in (["-I", "-B"], ["-I", "-O", "-B"], ["-I", "-B", "-W", "error"]):
            proc = subprocess.run(
                [sys.executable, *flags, script, "--cpu-preflight", "--json"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            digests.append(hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest())
        self.assertEqual(len(set(digests)), 1)


if __name__ == "__main__":
    unittest.main()
