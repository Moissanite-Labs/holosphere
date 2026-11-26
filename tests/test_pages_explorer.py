#!/usr/bin/env python3
"""Tests for the GitHub Pages explorer: generated Lean viewer, live oracle, Pages artifact staging.

CPU-only, deterministic, no network, no pytest, no bare asserts (self.assert* only),
no skips (tests/run_all.py treats skips as failures).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent

GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_lean_viewer.py"
HTML_PATH = REPO_ROOT / "docs" / "index.html"
LEAN_DIR = REPO_ROOT / "lean"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"

LEAN_FILES = {
    "invariants": "HoloSphereInvariants.lean",
    "barrington": "HoloSphereSpinorBarrington.lean",
    "bridges": "HoloSphereCrossSpineBridges.lean",
    "cellop": "HoloSphereCellOpInvariants.lean",
    "convergence": "HoloSphereConvergenceTheorems.lean",
}

FABRICATED_NAMES = [
    "su2_mul_norm_invariant",
    "hopf_fiber_phase_invariant",
    "cayley_unit_sphere_exact",
    "chunk32_tree_scan_equiv_seq",
    "holosphere_isometric_jacobian_bound",
]

LEAN_BEGIN_MARKER = "/* BEGIN GENERATED LEAN_SUITES (scripts/gen_lean_viewer.py; do not edit by hand) */"
LEAN_END_MARKER = "/* END GENERATED LEAN_SUITES */"
ORACLE_BEGIN_MARKER = "/* BEGIN ORACLE_EVAL */"
ORACLE_END_MARKER = "/* END ORACLE_EVAL */"


def read_html() -> str:
    return HTML_PATH.read_bytes().decode("utf-8")


def extract_between(html: str, begin_marker: str, end_marker: str) -> str:
    i = html.find(begin_marker)
    j = html.find(end_marker)
    if i < 0 or j < 0 or j < i:
        raise RuntimeError(f"markers not found or out of order: {begin_marker!r} / {end_marker!r}")
    return html[i + len(begin_marker): j]


def lean_suites_from_html(html: str) -> dict:
    block = extract_between(html, LEAN_BEGIN_MARKER, LEAN_END_MARKER).strip()
    prefix = "const LEAN_SUITES ="
    if not block.startswith(prefix):
        raise RuntimeError("const LEAN_SUITES declaration not found between the generated markers")
    js = block[len(prefix):].strip()
    if not js.endswith(";"):
        raise RuntimeError("const LEAN_SUITES declaration is missing its trailing semicolon")
    return json.loads(js[:-1])


def run_gen(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GEN_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        check=False,
    )


class TestPagesExplorer(unittest.TestCase):
    def test_01_lean_viewer_matches_source(self) -> None:
        proc = run_gen(["--check"], REPO_ROOT)
        self.assertEqual(
            proc.returncode,
            0,
            f"gen_lean_viewer.py --check exited {proc.returncode}: {proc.stdout}{proc.stderr}",
        )

    def test_02_lean_viewer_drift_detected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            lean_dir = tmp / "lean"
            lean_dir.mkdir()
            html_path = tmp / "index.html"
            shutil.copy2(HTML_PATH, html_path)
            for fname in LEAN_FILES.values():
                shutil.copy2(LEAN_DIR / fname, lean_dir / fname)

            # Case 1: a new theorem appended to a copied lean file must be detected.
            invariants = lean_dir / "HoloSphereInvariants.lean"
            text = invariants.read_bytes().decode("utf-8")
            if not text.endswith("\n"):
                text += "\n"
            text += "theorem p99_drift_probe : True := trivial\n"
            invariants.write_bytes(text.encode("utf-8"))
            proc = run_gen(
                ["--check", "--html", str(html_path), "--lean-dir", str(lean_dir)], tmp
            )
            self.assertNotEqual(
                proc.returncode,
                0,
                f"drift via appended theorem not detected (exit {proc.returncode}): {proc.stdout}{proc.stderr}",
            )

            # Case 2: a single-character edit inside the generated HTML block must be detected.
            shutil.copy2(LEAN_DIR / "HoloSphereInvariants.lean", invariants)
            html = html_path.read_bytes().decode("utf-8")
            self.assertIn("p1_quat_mul_assoc", html)
            tampered = html.replace("p1_quat_mul_assoc", "p1_quat_mul_assoX", 1)
            self.assertNotEqual(tampered, html)
            html_path.write_bytes(tampered.encode("utf-8"))
            proc = run_gen(
                ["--check", "--html", str(html_path), "--lean-dir", str(lean_dir)], tmp
            )
            self.assertNotEqual(
                proc.returncode,
                0,
                f"drift via edited theorem name not detected (exit {proc.returncode}): {proc.stdout}{proc.stderr}",
            )

    def test_03_lean_viewer_has_51_real_theorems(self) -> None:
        html = read_html()
        suites = lean_suites_from_html(html)
        self.assertEqual(sorted(suites.keys()), sorted(LEAN_FILES.keys()))
        total = 0
        for key, fname in LEAN_FILES.items():
            suite = suites[key]
            self.assertEqual(sorted(suite.keys()), ["code", "label"])
            code = suite["code"]
            total += code.count("theorem p")
            names = re.findall(r"(?m)^theorem (p\w+)", code)
            lean_text = (LEAN_DIR / fname).read_bytes().decode("utf-8")
            for name in names:
                self.assertIn(
                    f"theorem {name}",
                    lean_text,
                    f"{name} from suite {key!r} not found in lean/{fname}",
                )
            size = (LEAN_DIR / fname).stat().st_size
            expected_label = (
                f"lean/{fname} ({size:,} bytes · {len(names)} public theorems"
                " · statements only, proofs omitted)"
            )
            self.assertEqual(suite["label"], expected_label)
            header = code.split("\n", 1)[0]
            self.assertTrue(
                header.startswith("-- Generated by scripts/gen_lean_viewer.py from lean/"),
                f"suite {key!r} missing generated header comment: {header!r}",
            )
        self.assertEqual(total, 51, f"expected 51 'theorem p' statements across suites, got {total}")
        for name in FABRICATED_NAMES:
            self.assertNotIn(name, html, f"fabricated theorem name still present: {name}")

    def test_04_oracle_eval_pass_and_fail(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.fail("node executable not found on PATH; the oracle eval tests require Node.js")
        html = read_html()
        snippet = extract_between(html, ORACLE_BEGIN_MARKER, ORACLE_END_MARKER)
        self.assertIn("const ORACLE_TOLERANCE = 1e-12;", snippet)
        self.assertIn("function evaluateOracle(items, tol)", snippet)
        self.assertNotIn("document.", snippet, "ORACLE_EVAL block must be DOM-free")
        harness = (
            snippet
            + "\nconst cases = [];\n"
            + "cases.push(evaluateOracle([\n"
            + "  { name: 'Cayley', err: 2.2e-16 },\n"
            + "  { name: 'Closure', err: 1.1e-16 },\n"
            + "  { name: 'Assoc', err: 0.0 },\n"
            + "  { name: 'Hopf', err: 3.3e-16 }\n"
            + "], 1e-12));\n"
            + "cases.push(evaluateOracle([\n"
            + "  { name: 'Cayley', err: 2.2e-16 },\n"
            + "  { name: 'Closure', err: 1.0 },\n"
            + "  { name: 'Assoc', err: 0.0 },\n"
            + "  { name: 'Hopf', err: 3.3e-16 }\n"
            + "], 1e-12));\n"
            + "cases.push(evaluateOracle([\n"
            + "  { name: 'Cayley', err: NaN },\n"
            + "  { name: 'Closure', err: 0.0 },\n"
            + "  { name: 'Assoc', err: 0.0 },\n"
            + "  { name: 'Hopf', err: 0.0 }\n"
            + "], 1e-12));\n"
            + "console.log(JSON.stringify(cases));\n"
        )
        with tempfile.TemporaryDirectory() as td:
            harness_path = Path(td) / "oracle_harness.js"
            harness_path.write_bytes(harness.encode("utf-8"))
            proc = subprocess.run(
                [node, str(harness_path)], capture_output=True, text=True, check=False
            )
        self.assertEqual(proc.returncode, 0, f"node harness failed: {proc.stderr}")
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1, f"expected exactly one JSON line, got: {proc.stdout!r}")
        cases = json.loads(lines[0])
        self.assertEqual(len(cases), 3)

        # (i) four tiny finite errors -> pass, all rows pass.
        self.assertTrue(cases[0]["pass"], "case (i) should pass")
        self.assertEqual(len(cases[0]["rows"]), 4)
        for row in cases[0]["rows"]:
            self.assertTrue(row["pass"], f"case (i) row {row['name']} should pass")
        self.assertAlmostEqual(cases[0]["maxErr"], 3.3e-16)

        # (ii) one error of 1.0 -> fail, exactly that row fails, maxErr == 1.0.
        self.assertFalse(cases[1]["pass"], "case (ii) should fail")
        failing = [row for row in cases[1]["rows"] if not row["pass"]]
        self.assertEqual(len(failing), 1)
        self.assertEqual(failing[0]["name"], "Closure")
        self.assertEqual(cases[1]["maxErr"], 1.0)

        # (iii) NaN -> fail.
        self.assertFalse(cases[2]["pass"], "case (iii) with NaN should fail")
        self.assertFalse(cases[2]["rows"][0]["pass"])

    def test_05_oracle_no_hardcoded_pass(self) -> None:
        html = read_html()
        self.assertNotIn("~1e-16", html)
        self.assertNotIn("Checks Passed (", html)
        self.assertNotIn("' ✓</span>'", html, "unconditional ' ✓</span>' row pattern still present")
        # The pass/fail rendering must be conditional and the wiring must be real.
        self.assertIn("evaluateOracle(items, ORACLE_TOLERANCE)", html)
        self.assertIn("return result;", html)
        self.assertIn("✗ FAIL", html)
        self.assertIn("text-rose-300", html)
        self.assertIn("const ORACLE_SAMPLES = 1250;", html)
        self.assertIn("runLiveOracleChecks(ORACLE_SAMPLES)", html)
        self.assertIn("Run 1,250-Draw Check", html)
        self.assertIn(
            "Runs 1,250 random draws × 4 IEEE-754 invariant checks (5,000 evaluations) in your browser, pass/fail at tolerance 1e-12",
            html,
        )
        self.assertIn('id="btn-run-oracle"', html)

    def test_06_pages_artifact_excludes_manifesto(self) -> None:
        workflow = WORKFLOW_PATH.read_bytes().decode("utf-8")
        self.assertIn("Stage Pages site (explorer only; MANIFESTO.md excluded)", workflow)
        self.assertIn("cp docs/index.html docs/.nojekyll _site/", workflow)
        self.assertIn("test ! -e _site/MANIFESTO.md", workflow)
        self.assertIn("path: _site", workflow)
        self.assertNotIn("path: docs", workflow)

        # Actually execute the three staging shell lines in a temp copy of docs/.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            docs_copy = tmp / "docs"
            docs_copy.mkdir()
            for name in ("index.html", ".nojekyll", "MANIFESTO.md"):
                src = REPO_ROOT / "docs" / name
                if not src.is_file():
                    self.fail(f"expected repo file missing: docs/{name}")
                shutil.copy2(src, docs_copy / name)
            staging = (
                "rm -rf _site && mkdir _site\n"
                "cp docs/index.html docs/.nojekyll _site/\n"
                "test ! -e _site/MANIFESTO.md\n"
            )
            proc = subprocess.run(
                ["bash", "-c", staging],
                cwd=str(tmp),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"staging failed: {proc.stderr}")
            site_entries = sorted(p.name for p in (tmp / "_site").iterdir())
            self.assertEqual(site_entries, [".nojekyll", "index.html"])
            self.assertFalse((tmp / "_site" / "MANIFESTO.md").exists())


if __name__ == "__main__":
    unittest.main()
