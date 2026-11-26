#!/usr/bin/env python3
"""Tests for the GitHub Pages explorer: generated Lean viewer, live oracle, Pages artifact staging.

CPU-only, deterministic, no network, no pytest, no bare asserts (self.assert* only),
no skips (tests/run_all.py treats skips as failures).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
HOLOSCRIPT_BEGIN_MARKER = "/* BEGIN HOLOSCRIPT_EVAL */"
HOLOSCRIPT_END_MARKER = "/* END HOLOSCRIPT_EVAL */"

PINNED_CDN_SRI = {
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css": "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
    "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js": "sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8",
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js": "sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu",
    "https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js": "sha384-c6Uo4N9c3SOEigMVzP6IshUG1wQ5uMp3xeoQFiHWAQ86joWdgyajkvopySyKy/Z6",
}

PINNED_GITHUB_ACTIONS = {
    "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "actions/configure-pages@983d7736d9b0ae728b81ab479565c72886d7745b",
    "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa",
    "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e",
}


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

    def test_07_hygiene_sri_action_pins_and_scorecard_qualifiers(self) -> None:
        html = read_html()
        # No static PASS claim, no stale 213.7 KB / zip header, no 60 FPS claims anywhere, no Zero-Spill overclaim
        for forbidden in (
            "holosphere_lean_certificate.json · PASS",
            "213.7 KB",
            "holosphere-main.zip",
            "85 Entries",
            "Zero-Spill Silicon Receipts",
            "cdn.tailwindcss.com",
            "new Function",
        ):
            self.assertNotIn(forbidden, html, f"forbidden/stale string still present in docs/index.html: {forbidden!r}")
        self.assertIsNone(re.search(r"\b60\s*fps\b", html, re.I), "residual '60 FPS' reference found in docs/index.html")
        self.assertIsNone(re.search(r"\beval\s*\(", html), "forbidden eval() call found in docs/index.html")

        # Strict CSP meta (zero unsafe-eval, zero cdn.tailwindcss.com) and precompiled static Tailwind CSS
        self.assertIn('<meta http-equiv="Content-Security-Policy"', html)
        self.assertNotIn("unsafe-eval", html)
        self.assertIn('<style id="tailwind-static-css">', html)

        # Safe textContent rendering in selectLeanTab (never innerHTML for Lean labels/code)
        self.assertIn("document.getElementById('lean-file-label').textContent = suite.label;", html)
        self.assertIn("document.getElementById('lean-code-view').textContent = suite.code;", html)

        # Exact Lean suite total matches 213.8 KB on disk
        lean_bytes = sum((LEAN_DIR / fname).stat().st_size for fname in LEAN_FILES.values())
        self.assertIn(f"{lean_bytes / 1000:.1f} KB", html)

        # Every curated REPO_MODULES entry exists on disk with exact matching byte size
        atlas_entries = re.findall(r"path:\s*'([^']+)',\s*size:\s*'([^']+)'", html)
        self.assertEqual(len(atlas_entries), 24)
        for rel_path, size_label in atlas_entries:
            fp = REPO_ROOT / rel_path
            self.assertTrue(fp.is_file(), f"atlas entry missing on disk: {rel_path}")
            self.assertEqual(size_label, f"{fp.stat().st_size:,} B", f"stale atlas byte size for {rel_path}")

        # su2_holonomy_scan_sm86.cu card discloses local build / no committed ptxas receipt
        self.assertIn("No committed receipt", html)
        self.assertIn("Local nvcc build", html)

        # Every external <script src=...> carries an exact pinned SHA-384 SRI integrity hash + crossorigin="anonymous"
        ext_scripts = re.findall(r'<script\b[^>]*\bsrc="([^"]+)"[^>]*>', html)
        self.assertEqual(len(ext_scripts), 3, f"expected exactly 3 external scripts, got {ext_scripts}")
        for cdn_url, expected_sri in PINNED_CDN_SRI.items():
            pat = re.compile(
                re.escape(cdn_url)
                + r'"\s+integrity="'
                + re.escape(expected_sri)
                + r'"\s+crossorigin="anonymous"'
            )
            self.assertIsNotNone(pat.search(html), f"missing exact pinned SHA-384 SRI for {cdn_url}")

        # All GitHub Actions uses: directives in pages.yml match exact pinned 40-char commit SHAs and least-privilege permissions
        workflow = WORKFLOW_PATH.read_bytes().decode("utf-8")
        uses_lines = set(re.findall(r"(?m)^\s*uses:\s*(\S+)", workflow))
        self.assertEqual(uses_lines, PINNED_GITHUB_ACTIONS)
        wf_top = workflow.split("jobs:", 1)[0]
        self.assertNotIn("pages: write", wf_top, "workflow-level permissions must not grant pages: write to build job")
        self.assertNotIn("id-token: write", wf_top, "workflow-level permissions must not grant id-token: write to build job")
        build_block = workflow.split("build:", 1)[1].split("deploy:", 1)[0]
        self.assertNotIn("pages: write", build_block)
        self.assertNotIn("id-token: write", build_block)

        # Scorecard & Manifesto: zero "Zone-B Clean", C3 explicitly DEFERRED/PROVISIONAL, every numeric mJ/tok & W marked †
        scorecard = (REPO_ROOT / "research" / "SCORECARD_S106_S114.md").read_bytes().decode("utf-8")
        self.assertNotIn("Zone-B Clean", scorecard)
        self.assertNotIn("passed ALL locked gates (`C1`", scorecard)
        self.assertNotIn("failed `C3`", scorecard)
        self.assertIn("DEFERRED / PROVISIONAL", scorecard)
        preamble = scorecard.split("---", 1)[0]
        self.assertIn("NVML", preamble)
        self.assertIn("uncontrolled", preamble.lower())
        self.assertIn("provisional", preamble.lower())
        for m in re.finditer(r"[0-9][0-9.,/×x%\- ]*\s*(?:mJ/tok|W)\b([^`\n]*)", scorecard):
            span = scorecard[m.start(): m.end() + 4]
            self.assertIn("†", span, f"unmarked energy/power figure in SCORECARD: {span!r}")

        manifesto = (REPO_ROOT / "docs" / "MANIFESTO.md").read_bytes().decode("utf-8")
        self.assertIn("uncontrolled", manifesto.lower())
        self.assertIn("provisional", manifesto.lower())
        self.assertIn("deferred", manifesto.lower())
        self.assertIn("NVML", manifesto)
        self.assertNotIn("and `mJ/tok†` kill gates", manifesto)
        for m in re.finditer(r"[0-9][0-9.,/×x%\- ]*\s*mJ/tok([^`\n]*)", manifesto):
            span = manifesto[m.start(): m.end() + 4]
            self.assertIn("†", span, f"unmarked mJ/tok figure in MANIFESTO: {span!r}")

    def test_08_holoscript_sandbox_evaluator_and_escape_refusal(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.fail("node executable not found on PATH; HoloScript evaluator tests require Node.js")
        html = read_html()
        eval_snippet = extract_between(html, HOLOSCRIPT_BEGIN_MARKER, HOLOSCRIPT_END_MARKER)
        self.assertNotIn("new Function", eval_snippet)
        self.assertNotIn("eval(", eval_snippet)
        # Extract all 5 preset code strings from PRESETS block
        presets_start = html.find("const PRESETS = {")
        presets_end = html.find("/* BEGIN HOLOSCRIPT_EVAL */", presets_start)
        self.assertGreater(presets_start, 0)
        self.assertGreater(presets_end, presets_start)
        preset_codes = re.findall(r"code:\s*`([^`]+)`", html[presets_start:presets_end])
        self.assertEqual(len(preset_codes), 5, f"expected 5 HoloScript presets, got {len(preset_codes)}")

        harness = (
            eval_snippet
            + "\nconst std = {\n"
            + "  PI: Math.PI, TAU: Math.PI * 2, sin: Math.sin, cos: Math.cos, tan: Math.tan,\n"
            + "  asin: Math.asin, acos: Math.acos, atan2: Math.atan2, sqrt: Math.sqrt, exp: Math.exp,\n"
            + "  abs: Math.abs, pow: Math.pow, min: Math.min, max: Math.max, tanh: Math.tanh,\n"
            + "  hopf: (th, ph, ps) => [Math.cos(th/2)*Math.cos(ps+ph/2), Math.cos(th/2)*Math.sin(ps+ph/2), Math.sin(th/2)*Math.cos(ps-ph/2), Math.sin(th/2)*Math.sin(ps-ph/2)],\n"
            + "  stereo: (q, a) => [q[1]/(1-a*q[0]+0.1), q[2]/(1-a*q[0]+0.1), q[3]/(1-a*q[0]+0.1)],\n"
            + "  beltrami: (q) => [q[1]/(Math.abs(q[0])+0.1), q[2]/(Math.abs(q[0])+0.1), q[3]/(Math.abs(q[0])+0.1)],\n"
            + "  cayley: (x, y, z) => { const n = 1 + x*x + y*y + z*z; return [(2-n)/n, 2*x/n, 2*y/n, 2*z/n]; },\n"
            + "  su2Mul: (a, b) => [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]],\n"
            + "  su2Conj: (q) => [q[0], -q[1], -q[2], -q[3]],\n"
            + "  su2Exp: (x, y, z, ang) => { const n = Math.hypot(x,y,z) || 1; const s = Math.sin(ang/2)/n; return [Math.cos(ang/2), x*s, y*s, z*s]; },\n"
            + "  commutator: (u, v) => [1, 0, 0, 0],\n"
            + "  slerp: (a, b, u) => [a[0]*(1-u)+b[0]*u, a[1]*(1-u)+b[1]*u, a[2]*(1-u)+b[2]*u, a[3]*(1-u)+b[3]*u],\n"
            + "  e8Root: (idx, spin) => [Math.cos(spin), Math.sin(spin), 0, 0],\n"
            + "  binghamWeight: (q, l1, l2, l3) => 1.0,\n"
            + "  basePoints: [{ theta: 1.05, phi: 0.4, hue: 0.12 }, { theta: 1.55, phi: 2.55, hue: 0.55 }, { theta: 2.05, phi: 4.65, hue: 0.90 }]\n"
            + "};\n"
            + "const presets = " + json.dumps(preset_codes) + ";\n"
            + "const presetOk = presets.map((code) => {\n"
            + "  const fn = buildHoloScriptEvaluator(code, std);\n"
            + "  const out0 = fn(0, 32, 0.25, 0.5, { alpha: 0.72, p1: 1.2, p2: 2.0, p3: 0.5 });\n"
            + "  const out5 = fn(5, 32, 0.50, 0.5, { alpha: 0.72, p1: 1.2, p2: 2.0, p3: 0.5 });\n"
            + "  return Array.isArray(out0.pos) && out0.pos.every(Number.isFinite) && Array.isArray(out5.pos) && out5.pos.every(Number.isFinite);\n"
            + "});\n"
            + "const escapes = [\n"
            + "  'return globalThis;',\n"
            + "  'return window;',\n"
            + "  'return document;',\n"
            + "  'return fetch(1);',\n"
            + "  'return process;',\n"
            + "  'return ({}).constructor;',\n"
            + "  'return params.__proto__;',\n"
            + "  'return sin.constructor;',\n"
            + "  'return Function(1);',\n"
            + "  'return eval(1);'\n"
            + "];\n"
            + "const escapesBlocked = escapes.map((src) => {\n"
            + "  try {\n"
            + "    const fn = buildHoloScriptEvaluator(src, std);\n"
            + "    fn(0, 32, 0.25, 0.0, { alpha: 0.7, p1: 1, p2: 1, p3: 1 });\n"
            + "    return false;\n"
            + "  } catch (_e) {\n"
            + "    return true;\n"
            + "  }\n"
            + "});\n"
            + "console.log(JSON.stringify({ presetOk, escapesBlocked }));\n"
        )
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "holoscript_harness.js"
            hp.write_bytes(harness.encode("utf-8"))
            proc = subprocess.run([node, str(hp)], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, f"HoloScript node harness failed: {proc.stderr}")
        res = json.loads(proc.stdout.strip())
        self.assertEqual(res["presetOk"], [True] * 5, f"preset evaluation failed: {res['presetOk']}")
        self.assertEqual(res["escapesBlocked"], [True] * 10, f"escape probe not blocked: {res['escapesBlocked']}")

    def test_09_gen_lean_viewer_symlink_atomic_utf8_and_escape_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            lean_dir = tmp / "lean"
            lean_dir.mkdir()
            html_path = tmp / "index.html"
            shutil.copy2(HTML_PATH, html_path)
            for fname in LEAN_FILES.values():
                shutil.copy2(LEAN_DIR / fname, lean_dir / fname)

            # 1. Symlinked --html refused with RC=2
            sym_html = tmp / "sym_index.html"
            sym_html.symlink_to(html_path)
            p_sym_h = run_gen(["--check", "--html", str(sym_html), "--lean-dir", str(lean_dir)], tmp)
            self.assertEqual(p_sym_h.returncode, 2)
            self.assertIn("symlink", p_sym_h.stderr.lower())

            # 2. Symlinked --lean-dir refused with RC=2
            sym_lean = tmp / "sym_lean"
            sym_lean.symlink_to(lean_dir)
            p_sym_d = run_gen(["--check", "--html", str(html_path), "--lean-dir", str(sym_lean)], tmp)
            self.assertEqual(p_sym_d.returncode, 2)
            self.assertIn("symlink", p_sym_d.stderr.lower())

            # 3. Symlinked .lean file inside lean_dir refused with RC=2
            lean_dir_sym = tmp / "lean_with_sym"
            lean_dir_sym.mkdir()
            for fname in LEAN_FILES.values():
                shutil.copy2(LEAN_DIR / fname, lean_dir_sym / fname)
            inv_sym = lean_dir_sym / "HoloSphereInvariants.lean"
            inv_sym.unlink()
            inv_sym.symlink_to(LEAN_DIR / "HoloSphereInvariants.lean")
            p_sym_f = run_gen(["--check", "--html", str(html_path), "--lean-dir", str(lean_dir_sym)], tmp)
            self.assertEqual(p_sym_f.returncode, 2)
            self.assertIn("symlink", p_sym_f.stderr.lower())

            # 4. Invalid UTF-8 in HTML or Lean returns RC=2 cleanly with zero Python traceback
            bad_utf8_html = tmp / "bad_utf8.html"
            bad_utf8_html.write_bytes(b"\xff\xfe\x00not_utf8")
            p_utf8 = run_gen(["--check", "--html", str(bad_utf8_html), "--lean-dir", str(lean_dir)], tmp)
            self.assertEqual(p_utf8.returncode, 2)
            self.assertNotIn("Traceback", p_utf8.stderr)
            self.assertIn("UTF-8", p_utf8.stderr)

            # 5. Atomic regeneration writes cleanly with zero stray .tmp files and escapes </script> and <!--
            inv = lean_dir / "HoloSphereInvariants.lean"
            inv_text = inv.read_bytes().decode("utf-8")
            inv.write_bytes(
                (inv_text + "\n/-- escape probe </script><!--comment--> -/\ntheorem p99_escape_probe : True := trivial\n").encode("utf-8")
            )
            p_write = run_gen(["--html", str(html_path), "--lean-dir", str(lean_dir)], tmp)
            self.assertEqual(p_write.returncode, 0, msg=p_write.stderr)
            self.assertEqual(list(tmp.glob("*.tmp")), [])
            written_html = html_path.read_bytes().decode("utf-8")
            self.assertIn("<\\/script>", written_html)
            self.assertIn("<\\!--comment-->", written_html)
            self.assertNotIn("</script><!--comment-->", written_html)

            # 6. Direct in-process guard, boundary, FIFO, hardlink, short-write, and parser error coverage
            spec = importlib.util.spec_from_file_location("gen_lean_viewer_mod", GEN_SCRIPT)
            if spec is None or spec.loader is None:
                self.fail("Could not load scripts/gen_lean_viewer.py")
            gen_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_mod)

            self.assertTrue(gen_mod._is_forbidden_system_path("/"))
            self.assertTrue(gen_mod._is_forbidden_system_path("//etc/passwd"))
            self.assertTrue(gen_mod._is_forbidden_system_path("/dev/null"))
            self.assertFalse(gen_mod._is_forbidden_system_path("/dev/shm/ok.html"))
            self.assertFalse(gen_mod._is_forbidden_system_path("/etcetera/ok.html"))

            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._check_safe_path("/tmp/not_a_path", "p")  # type: ignore[arg-type]
            for bad_p in (
                Path("a\x00b"),
                Path("../escape"),
                Path("//tmp/x"),
                Path("/tmp/bad\udcffname"),
                Path("/etc/passwd"),
                sym_html,
            ):
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._check_safe_path(bad_p, "p")

            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._check_dir_safe(tmp / "no_such_dir", "lean_dir")
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._check_dir_safe(html_path, "lean_dir")

            # _read_bytes_safe: dir, non-blocking FIFO, missing file, exact MAX_FILE_BYTES boundary
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._read_bytes_safe(lean_dir, "html")
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._read_bytes_safe(tmp / "missing.html", "html")
            fifo_p = tmp / "fifo_probe"
            os.mkfifo(fifo_p)
            try:
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._read_bytes_safe(fifo_p, "html")
                # Non-regular FIFO (st_nlink == 1) write target must be refused by S_ISREG before atomic replace
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._write_bytes_atomic(fifo_p, b"x", "html")
            finally:
                fifo_p.unlink()

            probe_f = tmp / "probe.txt"
            probe_f.write_bytes(b"12345")
            with mock.patch.object(gen_mod, "MAX_FILE_BYTES", 5):
                self.assertEqual(gen_mod._read_bytes_safe(probe_f, "probe"), b"12345")
            with mock.patch.object(gen_mod, "MAX_FILE_BYTES", 4):
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._read_bytes_safe(probe_f, "probe")

            # _write_bytes_atomic: dir refusal, hardlink refusal, short write cleanup, OSError, clean atomic write
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod._write_bytes_atomic(lean_dir, b"x", "html")
            hl_alias = tmp / "probe_hl.txt"
            os.link(probe_f, hl_alias)
            try:
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._write_bytes_atomic(probe_f, b"new", "probe")
                self.assertEqual(probe_f.read_bytes(), b"12345")
            finally:
                hl_alias.unlink()
            with mock.patch.object(gen_mod.os, "write", return_value=0):
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._write_bytes_atomic(probe_f, b"short", "probe")
            self.assertEqual(list(tmp.glob("*.tmp")), [])
            self.assertEqual(probe_f.read_bytes(), b"12345")
            with mock.patch.object(gen_mod.os, "open", side_effect=OSError("perm denied")):
                with self.assertRaises(gen_mod.GeneratorError):
                    gen_mod._write_bytes_atomic(probe_f, b"fail", "probe")
            gen_mod._write_bytes_atomic(probe_f, b"ok", "probe")
            self.assertEqual(probe_f.read_bytes(), b"ok")
            self.assertEqual(list(tmp.glob("*.tmp")), [])

            # extract_statements & replace_block error paths
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod.extract_statements("-- no theorems here\n", "empty.lean")
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod.extract_statements("theorem p1_broken : True\n", "broken.lean")
            with self.assertRaises(gen_mod.GeneratorError):
                gen_mod.replace_block("<html><body>no block</body></html>", "block")


if __name__ == "__main__":
    unittest.main()
