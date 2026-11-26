#!/usr/bin/env python3
"""R5 Causality, Tamper-Refusal, Pure-Q AST, & Energy-Policy Test Suite for HoloSphere-V4 Monarch-2 E2E.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import contextlib
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_DIR = REPO_ROOT.joinpath("cuda", "v4_monarch2")
if str(V4_DIR) not in sys.path:
    sys.path.insert(0, str(V4_DIR))

import verify_holosphere_v4_monarch2 as v4m


def _resign(doc: dict, key: str) -> dict:
    doc = copy.deepcopy(doc)
    doc[key] = v4m.canonical_sha256(doc, exclude_key=key)
    return doc


class TestHoloSphereV4Monarch2R5(unittest.TestCase):
    def assertCode(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(v4m.V4Monarch2VerifyError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_hardened_pure_q_ast_audit_and_negatives(self) -> None:
        src = V4_DIR.joinpath("verify_holosphere_v4_monarch2.py").read_text(encoding="utf-8")
        v4m.verify_pure_q_ast(src)
        v4m.verify_pure_q_ast(
            "from fractions import Fraction\n"
            "import json\n"
            "import os\n"
            "import re\n"
            "x = Fraction(1, 2)\n"
            "y = mod.Fraction(3, 4)\n"
            "z = fn(1)\n"
            "a = json.dumps({})\n"
            "b = json.loads('{}', parse_float=fn, parse_constant=fn)\n"
            "c = other.loads('{}')\n"
            "d = fn().loads('{}')\n"
            "e = type(x) is int\n"
            "f = other(x) is int\n"
            "g = any(base.__name__ == 'float' for base in type(x).__mro__)\n"
            "k = os.O_RDONLY\n"
            "p = re.compile('a', re.DOTALL)\n"
            "n = (None, ..., True, 1, 's', b'b')\n"
        )

        self.assertCode("AST_SOURCE_TYPE", v4m.verify_pure_q_ast, 123)
        self.assertCode("AST_SYNTAX_ERROR", v4m.verify_pure_q_ast, "x = 1\x00\n")

        for code, bad_code in (
            ("AST_SYNTAX_ERROR", "def (\n"),
            ("AST_DIV_FORBIDDEN", "x = 1 / 2\n"),
            ("AST_DIV_FORBIDDEN", "x /= 2\n"),
            ("AST_FLOORDIV_FORBIDDEN", "x = 1 // 2\n"),
            ("AST_FLOORDIV_FORBIDDEN", "x //= 2\n"),
            ("AST_MOD_FORBIDDEN", "from fractions import Fraction\nx = '%.20f' % Fraction(1, 3)\n"),
            ("AST_MOD_FORBIDDEN", "x = fmt % 1\n"),
            ("AST_MOD_FORBIDDEN", "x %= 2\n"),
            ("AST_POW_FORBIDDEN", "x = 2 ** -1\n"),
            ("AST_BARE_ASSERT", "assert True\n"),
            ("AST_FORMAT_SPEC", "x = f'{1:.3f}'\n"),
            ("AST_FLOAT_LITERAL", "x = 1.5\n"),
            ("AST_FLOAT_LITERAL", "x = 1j\n"),
            ("AST_FORBIDDEN_IMPORT", "import math\n"),
            ("AST_FORBIDDEN_IMPORT", "import decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import _decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import struct\n"),
            ("AST_FORBIDDEN_IMPORT", "import random\n"),
            ("AST_FORBIDDEN_IMPORT", "import fractions\n"),
            ("AST_IMPORT_ALIAS", "import json as j\n"),
            ("AST_IMPORT_ALIAS", "import sys as s\n"),
            ("AST_FORBIDDEN_IMPORT", "from math import sqrt\n"),
            ("AST_FORBIDDEN_IMPORT", "from _operator import truediv\n"),
            ("AST_FORBIDDEN_IMPORT", "from _operator import truediv as Fraction\n"),
            ("AST_FORBIDDEN_IMPORT", "from importlib import import_module\n"),
            ("AST_FORBIDDEN_IMPORT", "from json import loads as l\n"),
            ("AST_RELATIVE_IMPORT", "from . import helper\n"),
            ("AST_RELATIVE_IMPORT", "from .foo import bar\n"),
            ("AST_FORBIDDEN_IMPORT", "from os import *\n"),
            ("AST_FORBIDDEN_IMPORT", "from sys import modules\n"),
            ("AST_FORBIDDEN_IMPORT", "from fractions import Fraction as F\n"),
            ("AST_FORBIDDEN_IMPORT", "from fractions import gcd\n"),
            ("AST_FORBIDDEN_IMPORT", "from typing import *\n"),
            ("AST_FORBIDDEN_MODULE_ATTR", "import os\nos.system('id')\n"),
            ("AST_FORBIDDEN_MODULE_ATTR", "import ast\nast.literal_eval('1.5')\n"),
            ("AST_FORBIDDEN_MODULE_ATTR", "import json\njson.JSONEncoder()\n"),
            ("AST_FORBIDDEN_MODULE_ATTR", "import re\nre.sub('a', 'b', 'a')\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5')\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5', parse_float=None, parse_constant=fn)\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5', parse_float=False, parse_constant=fn)\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5', parse_float=fn, parse_constant=None)\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5', parse_float=fn)\n"),
            ("AST_UNSAFE_JSON_LOADS", "import json\njson.loads('1.5', **kw)\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nF = Fraction\nx = F('0.5')\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nF, G = Fraction, 1\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nF: object = Fraction\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nx: (F := Fraction) = 1\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nx = (F := Fraction)('0.5')\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nx = [Fraction][0]('0.5')\n"),
            ("AST_SYMBOL_REBIND", "F = mod.Fraction\n"),
            ("AST_SYMBOL_REBIND", "def Fraction(a, b):\n    return a\n"),
            ("AST_SYMBOL_REBIND", "async def Fraction(a, b):\n    return a\n"),
            ("AST_SYMBOL_REBIND", "class Fraction:\n    pass\n"),
            ("AST_SYMBOL_REBIND", "def f(Fraction):\n    return Fraction\n"),
            ("AST_SYMBOL_REBIND", "def f(Fraction):\n    pass\n"),
            ("AST_SYMBOL_REBIND", "try:\n    pass\nexcept Exception as Fraction:\n    pass\n"),
            ("AST_SYMBOL_REBIND", "import json\nj = json\n"),
            ("AST_SYMBOL_REBIND", "from fractions import Fraction\nx = type(Fraction(1, 1))('1.5')\n"),
            ("AST_SYMBOL_REBIND", "t = type\n"),
            ("AST_SYMBOL_REBIND", "x = type(1) == int\n"),
            ("AST_SYMBOL_REBIND", "x = type(1) is int is True\n"),
            ("AST_SYMBOL_REBIND", "x = type(1, 2) is int\n"),
            ("AST_SYMBOL_REBIND", "x = type(1, k=2) is int\n"),
            ("AST_SYMBOL_REBIND", "x = type(1).__name__\n"),
            ("AST_FORBIDDEN_ATTR", "x = type(1, 2).__mro__\n"),
            ("AST_FORBIDDEN_ATTR", "x = type(1, k=2).__mro__\n"),
            ("AST_FORBIDDEN_ATTR", "from fractions import Fraction\nx = type(Fraction(1, 1)).__mro__[0]('0.5')\n"),
            ("AST_FORBIDDEN_ATTR", "from fractions import Fraction\nx = [c('0.5') for c in type(Fraction(1, 1)).__mro__]\n"),
            ("AST_FORBIDDEN_ATTR", "x = any(b.__name__ == 'float' for b in type(1).__mro__ if b)\n"),
            ("AST_FORBIDDEN_ATTR", "x = any(b.__name__ != 'float' for b in type(1).__mro__)\n"),
            ("AST_FORBIDDEN_ATTR", "x = any(b.__name__ == 1 for b in type(1).__mro__)\n"),
            ("AST_FORBIDDEN_ATTR", "x = any(b.__name__ == 'float' for b in type(1).__mro__ for c in [1])\n"),
            ("AST_FORBIDDEN_ATTR", "x = obj.__mro__\n"),
            ("AST_FORBIDDEN_ATTR", "x = Path('a').open()\n"),
            ("AST_FORBIDDEN_NAME", "x = float(1)\n"),
            ("AST_FORBIDDEN_NAME", "x = pow(2, -1)\n"),
            ("AST_FORBIDDEN_NAME", "x = format(1, '.2f')\n"),
            ("AST_FORBIDDEN_NAME", "x = divmod(5, 2)\n"),
            ("AST_FORBIDDEN_NAME", "x = round(1)\n"),
            ("AST_FORBIDDEN_NAME", "x = open('a')\n"),
            ("AST_FORBIDDEN_NAME", "x = memoryview(b'12345678')\n"),
            ("AST_FORBIDDEN_NAME", "x = __import__('math')\n"),
            ("AST_FORBIDDEN_NAME", "k = 'float'\nx = getattr(__builtins__, k)(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = Fraction.from_float(1)\n"),
            ("AST_FORBIDDEN_ATTR", "from fractions import Fraction\nx = Fraction(1, 2).__class__('0.5')\n"),
            ("AST_FORBIDDEN_ATTR", "x = ().__class__.__base__.__subclasses__()\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).real\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).imag\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).conjugate()\n"),
            ("AST_FORBIDDEN_ATTR", "x = '{a:.2f}'.format_map({'a': 1})\n"),
            ("AST_FORBIDDEN_ATTR", "x = mv.cast('d')\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__truediv__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__floordiv__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__mod__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__pow__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = obj.__getattribute__('float')\n"),
            ("AST_FORBIDDEN_ATTR", "x = '{:.2f}'.format(1)\n"),
            ("AST_FORBIDDEN_MODULE_ATTR", "import sys\nm = sys.modules['math']\n"),
            ("AST_FORBIDDEN_ATTR", "x = b.float(1)\n"),
            ("AST_FRACTION_STARRED", "from fractions import Fraction\nx = Fraction(*a)\n"),
            ("AST_FRACTION_STARRED", "from fractions import Fraction\nx = Fraction(*a, *b)\n"),
            ("AST_FRACTION_ARGS", "from fractions import Fraction\nx = Fraction(1)\n"),
            ("AST_FRACTION_ARGS", "from fractions import Fraction\nx = Fraction(1, 2, k=3)\n"),
            ("AST_FRACTION_ARGS", "x = mod.Fraction('1.5')\n"),
            ("AST_FRACTION_STR", "from fractions import Fraction\nx = Fraction('1', 2)\n"),
            ("AST_FRACTION_STR", "from fractions import Fraction\nx = Fraction(b'1', 2)\n"),
            ("AST_FRACTION_STR", "from fractions import Fraction\nx = Fraction(f'{1}', 2)\n"),
            ("AST_FRACTION_STR", "x = mod.Fraction('1', 2)\n"),
        ):
            self.assertCode(code, v4m.verify_pure_q_ast, bad_code)

        self.assertFalse(v4m.contains_any_float({"a": [1, "2/3", {"b": (3, 4)}], "s": frozenset({"ok"})}))
        self.assertTrue(v4m.contains_any_float({"a": [1, 1.5]}))
        self.assertTrue(v4m.contains_any_float({1.2: "x"}))
        self.assertTrue(v4m.contains_any_float({"a": (1.0,)}))

    def test_02_rational_kat_digit_limits_redos_and_energy_policy_guard(self) -> None:
        self.assertEqual(v4m.parse_q("3/4"), Fraction(3, 4))
        self.assertEqual(v4m.parse_q("-5/2"), Fraction(-5, 2))
        self.assertEqual(v4m.parse_q("0/1"), Fraction(0, 1))
        self.assertEqual(v4m.parse_q("9" * 1024 + "/1"), Fraction(int("9" * 1024), 1))
        self.assertEqual(v4m.parse_q("-" + "9" * 1024 + "/1"), Fraction(-int("9" * 1024), 1))
        self.assertEqual(v4m.parse_q("1/" + "9" * 1024), Fraction(1, int("9" * 1024)))
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "9" * 1025 + "/1")
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "-" + "9" * 1025 + "/1")
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "1/" + "9" * 1025)
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "9" * 4400 + "/1")
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "1/" + "9" * 4400)
        self.assertEqual(v4m._parse_strict_json_int("9" * 1024), int("9" * 1024))
        self.assertEqual(v4m._parse_strict_json_int("-" + "9" * 1024), -int("9" * 1024))
        self.assertCode("INT_DIGIT_LIMIT", v4m._parse_strict_json_int, "9" * 1025)
        self.assertCode("INT_DIGIT_LIMIT", v4m._parse_strict_json_int, "-" + "9" * 1025)
        self.assertCode("INT_DIGIT_LIMIT", v4m._parse_strict_json_int, "9" * 4400)

        self.assertEqual(v4m.q_div(Fraction(3, 4), Fraction(2, 5)), Fraction(15, 8))
        self.assertCode("ZERO_DIV", v4m.q_div, Fraction(1, 2), Fraction(0, 1))
        self.assertCode("BAD_RATIONAL", v4m.parse_q, 1)
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "2/4")
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "1/0")
        self.assertCode("BAD_RATIONAL", v4m.parse_q, "3/4\n")
        self.assertCode("BAD_DIVISOR", v4m.q_div_int, Fraction(1, 2), 0)
        self.assertCode("BAD_DIVISOR", v4m.q_div_int, Fraction(1, 2), True)
        self.assertCode("BAD_JSON", v4m.canonical_json_bytes, {"x": object()})
        self.assertFalse(v4m.is_hex64("z" * 64))
        self.assertFalse(v4m.is_hex64("a" * 64 + "\n"))
        self.assertFalse(v4m.is_hex64(123))

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            p = tdp.joinpath("kat.bin")
            p.write_bytes(b"abc")
            self.assertEqual(
                v4m.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(v4m.read_regular_bytes(p, max_bytes=3), b"abc")
            self.assertCode("FILE_TOO_LARGE", v4m.read_regular_bytes, p, max_bytes=2)
            self.assertCode("NOT_REGULAR_FILE", v4m.read_regular_bytes, tdp)
            fifo_p = tdp.joinpath("pipe.fifo")
            os.mkfifo(str(fifo_p))
            self.assertCode("NOT_REGULAR_FILE", v4m.read_regular_bytes, fifo_p)
            self.assertCode("READ_OR_PARSE_ERROR", v4m.decode_utf8_bytes, b"\xff\xfe")
            with patch("os.open", side_effect=PermissionError("denied")):
                self.assertCode("READ_OR_PARSE_ERROR", v4m.read_regular_bytes, p)
            with patch("os.open", side_effect=OSError(40, "Too many levels of symbolic links")):
                self.assertCode("SYMLINK_FORBIDDEN", v4m.read_regular_bytes, p)
            with patch("os.read", side_effect=OSError("io fail")):
                self.assertCode("READ_OR_PARSE_ERROR", v4m.read_regular_bytes, p)

        # Pre-json.loads raw depth guard (32 OK, 33 & 5000 fail closed; escaped quotes inside string ignored)
        v4m._check_json_raw_depth("[" * 32 + "]" * 32)
        v4m._check_json_raw_depth('{"a": "\\"' + "{" * 50 + '\\""}')
        self.assertCode("JSON_DEPTH_LIMIT", v4m._check_json_raw_depth, "[" * 33 + "]" * 33)
        self.assertCode("JSON_DEPTH_LIMIT", v4m._check_json_raw_depth, "{" * 33 + "}" * 33)
        self.assertCode("JSON_DEPTH_LIMIT", v4m.loads_strict_canonical_json_bytes, ("[" * 5000 + "]" * 5000).encode("ascii"), "deep.json")

        bad_values = (
            "45.2 W", "45W", "45.2 W.", "45.2W!", "W?", "[W]", "45.2\u00d7W", "45.2-W", "45.2 [W]",
            "W: 45.2", "W=45.2", "45.2 kilowatts", "45.2\u200bW", "45.2 k\u200bW", "5 GW", "5 TW",
            "5 pW", "5 kilo watts", "5 kilo-watts", "5 m W", "15 watthours", "15 watt hours",
            "15 watt-hour", "5\u2013W", "12\u00b7W", "3.9e2 W", "1.5E-3W", "0.5 kWh", "1.2 J/tok",
            "joules per token", "MJ per token", "mJ/tok", "15 TDP", "TDP 15", "TDP",
            "70% less energy", "70 percent less power", "energy use dropped", "power savings",
            "5 \u051d", "\u0435nergy", "3 \u043a\u0412\u0442", "\uff11\uff12\uff37", "45 milliwatts",
            "1.2 megawatts", "12.5 kilojoules", "W: 45.2 drain", "45.2[W] drain", "J: 5",
            "cap_45_w", "45_watts", "limit_45_w", "draw_45w", "45 Whr", "130w_psu",
            "max_45w", "45w", "5 PW", "45.2 \u03c9", "\u051datt", "45.2\u2219W",
            "measured with RAPL", "nvml query", "nvidia-smi draw", "10 tok/J", "tokens per joule",
            "50 tok/kWh", "score per watt", "500 mAh", "45\u200cW", "45\u200dW", "45\u2060W",
            "45\ufeffW", "45\u00adW",
            "j/tok", "1 j/ tok", "tokens_per_joule", "x_nvml_y", "wa\u0000tt",
            "45.0% of stock power", "energy per token", "power-savings", "W 45.2",
            "forty-five watts", "fortyfive watts", "fifty watts", "ten joules",
            "energy: 12", "energy 12", "power: 45", "power 45", "power = 120",
            "wattage: 45", "2.25 A at 20 V", "W/tok", "250 W/ token", "250 W / token",
            "mW/tok", "tok_per_mwh", "rapluj", "nvmlDeviceGetTotalEnergyConsumption", "wattage",
        )
        good_values = (
            "5 ms", "latency 12", "70 C", "78\u00b0C", "70/78/85 C knees", "5 C deadband",
            "5 deg C deadband", "135/2", "45.0%", "0.5 mm", "ac", "battery", "unknown",
            "temp_c", "gpu_temp_exceeded", "not_on_ac_power", "UNCONTROLLED_LAPTOP_NOT_CLAIMED",
            "grad_W_q", "stdlib_sha256_chain", "cpu_jiffies_total", "a" * 64, "129,279.00 tok/s",
            "469.41 us/tok", "8.113 MiB", "sm_86", "T10-TAG-61", "wmma", "window 20", "warp 32",
            "16 warps/SM", "h256", "rows 1024", "tdp_like_free_text_no", "T13-TAG-DECIDE",
            "J2000 epoch-like", "J2000.0", "worker T10", "2024-W05", "sprint-W12", "node-w10",
            "q4-w3", "W-2", "W12", "kW12", "bench w/ perf counters", "taskset -c 3 w/ 5 reps",
            "12 ms/tok", "sm_clock 2100 MHz", "0.9 V", "1024 MiB/s",
            "j = 3", "i = 1, j = 3", "w = 7", "3 w/o pinning", "run 3 W/O cache",
            "w/ tokens", "3 w/5 reps", "j = 3 w/o pinning",
        )
        self.assertEqual(len(bad_values), 102)
        self.assertEqual(len(good_values), 58)
        for bad in bad_values:
            self.assertTrue(v4m.is_forbidden_energy_value(bad), f"must reject {bad!r}")
        for good in good_values:
            self.assertFalse(v4m.is_forbidden_energy_value(good), f"false positive on {good!r}")

        self.assertFalse(v4m.is_forbidden_energy_key_name("empowerment_score"))
        self.assertTrue(v4m.is_forbidden_energy_key_name("powerdraw"))
        self.assertTrue(v4m.is_forbidden_energy_key_name("avgpower"))

        # Linear-time ReDoS stress guard across adversarial separator/digit inputs (CPU process_time)
        t0 = time.process_time()
        for probe in (
            "1" + "." * 10000,
            "1" + "," * 10000,
            "." * 10000,
            "9" * 10000,
            "1.2." * 3000,
            "1 " * 5000 + "pinning",
            "energy" + " " * 10000 + "ok",
        ):
            self.assertFalse(v4m.is_forbidden_energy_value(probe))
        elapsed = time.process_time() - t0
        self.assertLess(elapsed, 0.25, f"energy regex exceeded linear-time CPU budget: {elapsed:.4f}s")

        self.assertFalse(
            v4m.has_forbidden_energy_claim(
                {
                    "energy_power_status": v4m.ENERGY_POLICY_STATUS,
                    "empowerment_score": 100,
                    "kernels": {v4m.V4_E2E_SYMBOL: {"registers": 55}},
                    "ok": list(good_values),
                    "t": ("sm_86",),
                    "f": frozenset({"16896 B"}),
                    1: "clean",
                }
            )
        )
        for bad_obj in (
            {"mJ_per_tok": 1},
            {"avgPower": 45},
            {"po\u0301wer": 45},
            {"\u0435nergy": 1},
            {"max_45w": 1},
            {"pwr": 1},
            {"rapl": "ok"},
            {"rapluj": 1},
            {"nvml": "ok"},
            {"nvidia_smi": "ok"},
            {"w_per_tok": 1},
            {"tok_per_mwh": 1},
            {"nested": {"energy_power_status": v4m.ENERGY_POLICY_STATUS}},
            {"kernels": {v4m.V4_E2E_SYMBOL: {"note": "45.2 W"}}},
            {"kernels": {"fake_pj_symbol": {"registers": 8}}},
            {"t": ("45.2 W",)},
            {1: "45.2 W"},
        ):
            self.assertTrue(v4m.has_forbidden_energy_claim(bad_obj), f"must reject {bad_obj!r}")

    def test_03_pure_z_holosphere_v4_oracles_and_kat_negatives(self) -> None:
        o = v4m.verify_holosphere_v4_oracles()
        self.assertTrue(o["h256_involution_verified"])
        self.assertTrue(o["bos_256_matches_byte_33"])
        self.assertEqual(o["bent4_impulse_kats_verified"], 6)
        self.assertEqual(o["monarch1_reach_count"], 16)
        self.assertEqual(o["monarch2_full_reach_count"], 256)
        self.assertEqual(o["secded_sec_healed"], 16)
        self.assertEqual(o["secded_ded_detected"], 3)

        self.assertCode("BAD_FWHT_LEN", v4m.fwht256_plain_int, [0] * 16)
        self.assertCode("BAD_FWHT_LEN", v4m.fwht256_bent4_int, [0] * 16)
        self.assertCode("BAD_MONARCH_LEN", v4m.monarch16_stage_allones_int, [0] * 16)
        self.assertCode("BAD_DATA11", v4m.a16_encode_u16, -1)
        self.assertCode("BAD_DATA11", v4m.a16_encode_u16, 2048)
        self.assertCode("BAD_DATA11", v4m.a16_encode_u16, True)
        self.assertCode("BAD_W16", v4m.a16_syndrome_u5, -1)
        self.assertCode("BAD_W16", v4m.a16_syndrome_u5, 65536)

        self.assertCode("ORACLE_H256_INVOLUTION", v4m.verify_holosphere_v4_oracles, plain_fn=lambda u: [0] * 256)
        self.assertCode("ORACLE_PARSEVAL_FAIL", v4m.verify_holosphere_v4_oracles, bent4_fn=lambda u: [0] * 256)
        self.assertCode("ORACLE_BOS_EQUIV", v4m.verify_holosphere_v4_oracles, bos_byte=34)

        # Energy-preserving wrong transforms (pass Parseval, MUST fail 6-impulse signed-Hadamard KAT)
        def _bent4_no_sign(u):
            buf = [0] * 256
            for idx in range(256):
                w = idx >> 5
                lane = idx & 31
                buf[idx] = u[((w ^ (lane & 7)) << 5) | lane]
            return v4m.fwht256_plain_int(buf)

        def _bent4_no_warp_xor(u):
            buf = [0] * 256
            for idx in range(256):
                lane = idx & 31
                b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
                buf[idx] = -u[idx] if b4 else u[idx]
            return v4m.fwht256_plain_int(buf)

        for wrong_fn in (
            v4m.fwht256_plain_int,
            _bent4_no_sign,
            _bent4_no_warp_xor,
            lambda u: list(reversed(v4m.fwht256_bent4_int(u))),
            lambda u: [-v for v in v4m.fwht256_bent4_int(u)],
        ):
            self.assertCode("ORACLE_BENT4_KAT", v4m.verify_holosphere_v4_oracles, bent4_fn=wrong_fn)

        self.assertCode("ORACLE_MONARCH2_REACH", v4m.verify_holosphere_v4_oracles, stage_fn=lambda u: [0] * 256)
        calls = {"n": 0}

        def _stage1_only(u):
            calls["n"] += 1
            return v4m.monarch16_stage_allones_int(u) if calls["n"] == 1 else [1] * 16 + [0] * 240

        self.assertCode("ORACLE_MONARCH2_REACH", v4m.verify_holosphere_v4_oracles, stage_fn=_stage1_only)

        cw_ref = v4m.a16_encode_u16(0x5A3)
        self.assertCode("ORACLE_CLEAN_SYND", v4m.verify_holosphere_v4_oracles, syndrome_fn=lambda w: 1)
        self.assertCode(
            "ORACLE_SEC_HEAL",
            v4m.verify_holosphere_v4_oracles,
            syndrome_fn=lambda w: 0 if w == cw_ref else 2,
        )
        self.assertCode(
            "ORACLE_SEC_HEAL",
            v4m.verify_holosphere_v4_oracles,
            syndrome_fn=lambda w: 0 if w == cw_ref else (3 if w == (cw_ref ^ 1) else v4m.a16_syndrome_u5(w)),
        )
        self.assertCode(
            "ORACLE_DED_DETECT",
            v4m.verify_holosphere_v4_oracles,
            syndrome_fn=lambda w: 0 if w == (cw_ref ^ 3) else v4m.a16_syndrome_u5(w),
        )
        self.assertCode(
            "ORACLE_DED_DETECT",
            v4m.verify_holosphere_v4_oracles,
            syndrome_fn=lambda w: 1 if w == (cw_ref ^ 3) else v4m.a16_syndrome_u5(w),
        )

    def test_04_live_on_disk_receipts_scale_tokens_and_tamper_refusal(self) -> None:
        summary = v4m.verify_on_disk(REPO_ROOT)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["verdict"], "CONFIRMED")

        k_txt = REPO_ROOT.joinpath(v4m.KERNEL_CU_REL).read_text(encoding="utf-8")
        b_txt = REPO_ROOT.joinpath(v4m.BENCH_CU_REL).read_text(encoding="utf-8")
        s_txt = REPO_ROOT.joinpath(v4m.BPB_STUDY_REL).read_text(encoding="utf-8")
        self.assertTrue(v4m.verify_cuda_scale_tokens(k_txt, b_txt, s_txt)["cuda_and_study_scale_tokens_verified"])

        # Comment-only, #if 0, string-literal, and missing-entrypoint fakes MUST fail closed
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, "// 0.0625f\n// 0.17677669529663688f\n__global__ void fake() {}", b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, "float fwht256_plain_8warp_raw(float v) { v *= 0.0625f;", b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, "#if 0\n" + k_txt + "\n#endif\n", b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt.replace("0.0625f", "0.25f"), b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt.replace("0.17677669529663688f", "0.25f"), b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt.replace("holosphere_v3_e2e_scalar_fwht_sm86(", "fake_v3("), b_txt, s_txt)
        self.assertCode("KERNEL_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt.replace("holosphere_v4_e2e_monarch2_bias_fwht_sm86(", "fake_v4("), b_txt, s_txt)
        self.assertCode("BENCH_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, 'const char* s = "acc * 0.0625; inv_sqrt32 = 1.0 / std::sqrt(32.0);";', s_txt)
        self.assertCode("BENCH_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt.replace("0.0625", "0.25"), s_txt)
        self.assertCode("BENCH_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt.replace("std::sqrt(32.0)", "std::sqrt(16.0)"), s_txt)
        self.assertCode("STUDY_AST_INVALID", v4m.verify_cuda_scale_tokens, k_txt, b_txt, "def (\n")
        self.assertCode("STUDY_BARE_ASSERT", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nassert True\n")
        self.assertCode("STUDY_FORBIDDEN_IMPORT", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nimport subprocess\n")
        self.assertCode("STUDY_FORBIDDEN_IMPORT", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nfrom ctypes import CDLL\n")
        self.assertCode("STUDY_FORBIDDEN_IMPORT", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nx = __import__('os')\n")
        self.assertCode("STUDY_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nif False:\n    CORPUS_ID = 'fake'\n")
        self.assertCode("STUDY_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt + "\nCORPUS_ID: str = 'fake'\n")
        self.assertCode("STUDY_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt.replace("0.0625", "0.25"))
        self.assertCode("STUDY_SCALE_MISMATCH", v4m.verify_cuda_scale_tokens, k_txt, b_txt, s_txt.replace("math.sqrt(32.0)", "math.sqrt(16.0)"))

        ptx = v4m.load_strict_canonical_json(REPO_ROOT.joinpath(v4m.PTXAS_REL))
        res = v4m.load_strict_canonical_json(REPO_ROOT.joinpath(v4m.RESULTS_REL))

        self.assertCode("PTXAS_TYPE", v4m.verify_ptxas_receipt, [])
        for code, resign, mut in (
            ("PTXAS_SCHEMA", False, lambda d: d.__setitem__("schema", "bad")),
            ("PTXAS_ENERGY_POLICY", True, lambda d: d.__setitem__("energy_power_status", "BAD")),
            ("PTXAS_ENERGY_POLICY", True, lambda d: d.__setitem__("note", "45.2 W")),
            ("PTXAS_FLOAT_FORBIDDEN", True, lambda d: d.__setitem__("x", 1.5)),
            ("PTXAS_KEYS", True, lambda d: d.__setitem__("extra_key", "0")),
            ("PTXAS_SHA_FORMAT", False, lambda d: d.__setitem__("receipt_sha256", "bad")),
            ("PTXAS_SHA_MISMATCH", False, lambda d: d.__setitem__("arch", "sm_90")),
            ("PTXAS_ARCH", True, lambda d: d.__setitem__("arch", "sm_90")),
            ("PTXAS_FLAGS", True, lambda d: d.__setitem__("compile_flags", "-O0")),
            ("PTXAS_NVCC", True, lambda d: d.__setitem__("nvcc_version", "")),
            ("PTXAS_CU_FILE", True, lambda d: d.__setitem__("cu_filename", "wrong.cu")),
            ("PTXAS_CU_SHA", True, lambda d: d.__setitem__("cu_sha256", "bad")),
            ("PTXAS_KERNELS", True, lambda d: d.__setitem__("kernels", {})),
            ("PTXAS_REQUIRED_SYMBOLS", True, lambda d: (d["kernels"].pop(v4m.V3_E2E_SYMBOL), d["kernels"].__setitem__("fake", copy.deepcopy(ptx["kernels"][v4m.V3_E2E_SYMBOL])))),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"].__setitem__(v4m.V3_E2E_SYMBOL, [])),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("extra", 0)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("registers", -1)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("registers", True)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("registers", 60)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("barriers", 2)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("smem_bytes", 1024)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("atom_cas_sys_insns", 0)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("red_or_sys_insns", 0)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V4_E2E_SYMBOL].__setitem__("atoms_smem_insns", 0)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("stack_bytes", 8)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("spill_store_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("spill_load_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("ldl_insns", 1)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("stl_insns", 1)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][v4m.V3_E2E_SYMBOL].__setitem__("registers", 0)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("total_kernels", 1)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("zero_spill_kernels", 1)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("nonzero_spill_kernels", 1)),
        ):
            p2 = copy.deepcopy(ptx)
            mut(p2)
            if resign:
                p2 = _resign(p2, "receipt_sha256")
            self.assertCode(code, v4m.verify_ptxas_receipt, p2)
        self.assertCode("PTXAS_CU_SHA_DISK", v4m.verify_ptxas_receipt, ptx, expected_cu_sha="0" * 64)

        self.assertCode("RESULTS_TYPE", v4m.verify_results_receipt, [], ptx)
        for code, resign, mut in (
            ("RESULTS_SCHEMA", False, lambda d: d.__setitem__("schema", "bad")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("energy_power_status", "BAD")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("mj_per_tok", "1/2")),
            ("RESULTS_FLOAT_FORBIDDEN", True, lambda d: d.__setitem__("x", 0.5)),
            ("RESULTS_KEYS", True, lambda d: d.__setitem__("extra_key", "0")),
            ("RESULTS_SHA_FORMAT", False, lambda d: d.__setitem__("result_sha256", "bad")),
            ("RESULTS_SHA_MISMATCH", False, lambda d: d.__setitem__("verdict", "FALSIFIED")),
            ("RESULTS_SCOPE", True, lambda d: d.__setitem__("scope_note", "")),
            ("RESULTS_SCOPE", True, lambda d: d.__setitem__("scope_note", 123)),
            ("RESULTS_HARDWARE", True, lambda d: d.__setitem__("hardware", [])),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("extra", 1)),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("device", 1)),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("sm", "9.0")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("nvcc_version", "other")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("device", "")),
            ("RESULTS_BINDINGS", True, lambda d: d.__setitem__("source_bindings", [])),
            ("RESULTS_BINDING_FILES", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_filename", "wrong.cu")),
            ("RESULTS_BINDING_FILES", True, lambda d: d["source_bindings"].__setitem__("bench_cu_filename", "wrong.cu")),
            ("RESULTS_BINDING_FILES", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_filename", "wrong.py")),
            ("RESULTS_BINDING_FILES", True, lambda d: d["source_bindings"].__setitem__("ptxas_receipt_filename", "wrong.json")),
            ("RESULTS_CU_BIND", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_sha256", "0" * 64)),
            ("RESULTS_PTX_BIND", True, lambda d: d["source_bindings"].__setitem__("ptxas_receipt_sha256", "0" * 64)),
            ("RESULTS_BENCH_SHA", True, lambda d: d["source_bindings"].__setitem__("bench_cu_sha256", "bad")),
            ("RESULTS_STUDY_SCRIPT_SHA", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_sha256", "bad")),
            ("RESULTS_CORPUS_BIND", True, lambda d: d["source_bindings"].__setitem__("corpus_id", "wrong")),
            ("RESULTS_CORPUS_BIND", True, lambda d: d["source_bindings"].__setitem__("corpus_manifest_sha256", "0" * 64)),
            ("RESULTS_CORPUS_BIND", True, lambda d: d["source_bindings"].__setitem__("corpus_train_bin_sha256", "0" * 64)),
            ("RESULTS_CORPUS_BIND", True, lambda d: d["source_bindings"].__setitem__("corpus_eval_bin_sha256", "0" * 64)),
            ("RESULTS_SILICON", True, lambda d: d.__setitem__("silicon_e2e_bench", [])),
            ("RESULTS_SIL_INT", True, lambda d: d["silicon_e2e_bench"].__setitem__("clean_packed_u32", -1)),
            ("RESULTS_SIL_INT", True, lambda d: d["silicon_e2e_bench"].__setitem__("clean_packed_u32", True)),
            ("RESULTS_SIL_Q", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_1l_fp64_max_err_q", "9" * 4400 + "/1")),
            ("RESULTS_PARITY_BOUND", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_1l_fp64_max_err_q", "1/100000")),
            ("RESULTS_PARITY_BOUND", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_1l_fp64_max_err_q", "-1/1000000")),
            ("RESULTS_US_POS", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_monarch2_1l_us_per_tok_q", "0/1")),
            ("RESULTS_TPS_1L", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_monarch2_1l_tok_per_s_q", "1/1")),
            ("RESULTS_TPS_4L", True, lambda d: d["silicon_e2e_bench"].__setitem__("v4_monarch2_4l_tok_per_s_q", "1/1")),
            ("RESULTS_STUDY", True, lambda d: d.__setitem__("e2e_bpb_study", [])),
            ("RESULTS_STUDY_META", True, lambda d: d["e2e_bpb_study"].__setitem__("seeds", [42, 137, 999])),
            ("RESULTS_STUDY_META", True, lambda d: d["e2e_bpb_study"].__setitem__("d_model", 256)),
            ("RESULTS_STUDY_META", True, lambda d: d["e2e_bpb_study"].__setitem__("rope_base", 10000)),
            ("RESULTS_STUDY_META", True, lambda d: d["e2e_bpb_study"].__setitem__("eval_contexts", [])),
            ("RESULTS_STUDY_META", True, lambda d: d["e2e_bpb_study"].__setitem__("tokens_per_seed", 1)),
            ("RESULTS_H256_BOUND", True, lambda d: d["e2e_bpb_study"].__setitem__("h256_ortho_fp64_max_err_q", "1/10")),
            ("RESULTS_H256_BOUND", True, lambda d: d["e2e_bpb_study"].__setitem__("h256_ortho_fp64_max_err_q", "-1/10")),
            ("RESULTS_ARMS", True, lambda d: d["e2e_bpb_study"].__setitem__("arms", {})),
            ("RESULTS_ARM_TYPE", True, lambda d: d["e2e_bpb_study"]["arms"].__setitem__("V3_ScalarFWHT", [])),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("head_params", 1)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("head_fp16_bytes", 2)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("hbm_input_vocab_bytes", 16)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("hbm_output_vocab_bytes", 16)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("total_params", 0)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"].__setitem__("total_params", True)),
            ("RESULTS_ARM_SEEDS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"].__setitem__("seeds", [])),
            ("RESULTS_SEED_ROW", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"].__setitem__(0, "bad")),
            ("RESULTS_SEED_ROW", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0].__setitem__("seed", True)),
            ("RESULTS_SEED_WALL_POS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0].__setitem__("train_wall_s_q", "0/1")),
            ("RESULTS_SEED_CTXS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0].__setitem__("contexts", {})),
            ("RESULTS_CTX_ROW", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0]["contexts"]["T512"].__setitem__("eval_tokens", 100)),
            ("RESULTS_CTX_ROW", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0]["contexts"]["T512"].__setitem__("eval_tokens", True)),
            ("RESULTS_CTX_BPB_POS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0]["contexts"]["T512"].__setitem__("eval_bpb_q", "0/1")),
            ("RESULTS_CTX_BPB_POS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0]["contexts"]["T512"].__setitem__("eval_bpb_bf16_q", "0/1")),
            ("RESULTS_CTX_QDIFF", True, lambda d: d["e2e_bpb_study"]["arms"]["V4_Monarch2BiasFWHT"]["seeds"][0]["contexts"]["T512"].__setitem__("fp16_quant_bpb_diff_q", "1/2")),
            ("RESULTS_SEED_ORDER", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["seeds"][0].__setitem__("seed", 99)),
            ("RESULTS_MEANS_KEYS", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"].__setitem__("mean_eval_bpb_by_ctx_q", {})),
            ("RESULTS_MEAN_MISMATCH", True, lambda d: d["e2e_bpb_study"]["arms"]["V3_ScalarFWHT"]["mean_eval_bpb_by_ctx_q"].__setitem__("T512", "99/1")),
            ("RESULTS_CS", True, lambda d: d.__setitem__("context_summary", {})),
            ("RESULTS_CS_ROW", True, lambda d: d["context_summary"].__setitem__("T512", [])),
            ("RESULTS_CS_LIFT_SC", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_lift_over_scalar_q", "99/1")),
            ("RESULTS_CS_LIFT_LR", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_lift_over_lowrank16_q", "99/1")),
            ("RESULTS_CS_GAP_DN", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_minus_dense_q", "99/1")),
            ("RESULTS_CS_SPAN", True, lambda d: d["context_summary"]["T512"].__setitem__("dense_gap_scalar_minus_dense_q", "99/1")),
            ("RESULTS_CS_GCLOSED", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_gap_closed_fraction_q", "99/1")),
            ("RESULTS_CS_FLAGS", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_lt_scalar_all_seeds", not d["context_summary"]["T512"]["v4_lt_scalar_all_seeds"])),
            ("RESULTS_CS_FLAGS", True, lambda d: d["context_summary"]["T512"].__setitem__("v4_lt_lowrank16_all_seeds", not d["context_summary"]["T512"]["v4_lt_lowrank16_all_seeds"])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d.__setitem__("gates", [])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("extra", True)),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("G1_e2e_multicontext_quality", not d["gates"]["G1_e2e_multicontext_quality"])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("G2_fused_sm86_zero_spill_and_latency", not d["gates"]["G2_fused_sm86_zero_spill_and_latency"])),
            ("RESULTS_VERDICT_MISMATCH", True, lambda d: d.__setitem__("verdict", "PARTIAL" if d["verdict"] != "PARTIAL" else "FALSIFIED")),
        ):
            r2 = copy.deepcopy(res)
            mut(r2)
            if resign:
                r2 = _resign(r2, "result_sha256")
            self.assertCode(code, v4m.verify_results_receipt, r2, ptx)

        self.assertCode("RESULTS_BENCH_DISK", v4m.verify_results_receipt, res, ptx, expected_bench_cu_sha="0" * 64)
        self.assertCode("RESULTS_STUDY_SCRIPT_DISK", v4m.verify_results_receipt, res, ptx, expected_bpb_study_sha="0" * 64)

        def _build_synth(sc_v: str, v4_v: str, lr_v: str, dn_v: str, v4_us: str, sec_ok: bool) -> dict:
            rs = copy.deepcopy(res)
            vals = {"V3_ScalarFWHT": sc_v, "V4_Monarch2BiasFWHT": v4_v, "V3_LowRank16Bias": lr_v, "V3_DenseHead": dn_v}
            for arm_n, qv in vals.items():
                for row in rs["e2e_bpb_study"]["arms"][arm_n]["seeds"]:
                    for ck in v4m._EXPECTED_CONTEXTS:
                        row["contexts"][ck]["eval_bpb_q"] = qv
                        row["contexts"][ck]["eval_bpb_bf16_q"] = qv
                        row["contexts"][ck]["fp16_quant_bpb_diff_q"] = "0/1"
                for ck in v4m._EXPECTED_CONTEXTS:
                    rs["e2e_bpb_study"]["arms"][arm_n]["mean_eval_bpb_by_ctx_q"][ck] = qv
            qsc, qv4, qlr, qdn = (v4m.parse_q(vals[k]) for k in v4m._EXPECTED_ARMS)
            span = qsc - qdn
            gcl = v4m.q_div(qsc - qv4, span) if span > Fraction(0, 1) else Fraction(0, 1)
            for ck in v4m._EXPECTED_CONTEXTS:
                rs["context_summary"][ck] = {
                    "v4_lift_over_scalar_q": v4m.format_q(qsc - qv4),
                    "v4_lift_over_lowrank16_q": v4m.format_q(qlr - qv4),
                    "v4_minus_dense_q": v4m.format_q(qv4 - qdn),
                    "dense_gap_scalar_minus_dense_q": v4m.format_q(span),
                    "v4_gap_closed_fraction_q": v4m.format_q(gcl),
                    "v4_lt_scalar_all_seeds": qv4 < qsc,
                    "v4_lt_lowrank16_all_seeds": qv4 < qlr,
                }
            qu = v4m.parse_q(v4_us)
            rs["silicon_e2e_bench"]["v4_monarch2_1l_us_per_tok_q"] = v4_us
            rs["silicon_e2e_bench"]["v4_monarch2_1l_tok_per_s_q"] = v4m.format_q(v4m.q_div(Fraction(1000000, 1), qu))
            rs["silicon_e2e_bench"]["v4_sec16_packed_u32"] = 16 if sec_ok else 0
            g1 = (qsc - qv4 >= Fraction(1, 2)) and (qv4 < qlr) and (qv4 - qdn <= Fraction(1, 5))
            g2 = sec_ok and (qu <= Fraction(25, 1))
            rs["gates"]["G1_e2e_multicontext_quality"] = g1
            rs["gates"]["G2_fused_sm86_zero_spill_and_latency"] = g2
            rs["verdict"] = "CONFIRMED" if (g1 and g2) else ("PARTIAL" if (g1 or g2) else "FALSIFIED")
            return _resign(rs, "result_sha256")

        self.assertEqual(v4m.verify_results_receipt(_build_synth("9/2", "4/1", "21/5", "19/5", "25/1", True), ptx)["verdict"], "CONFIRMED")
        self.assertEqual(v4m.verify_results_receipt(_build_synth("4/1", "4/1", "9/2", "4/1", "10/1", True), ptx)["verdict"], "PARTIAL")
        self.assertEqual(v4m.verify_results_receipt(_build_synth("9/2", "4/1", "4/1", "19/5", "10/1", True), ptx)["verdict"], "PARTIAL")
        self.assertEqual(v4m.verify_results_receipt(_build_synth("5/1", "4/1", "9/2", "19/5", "26/1", True), ptx)["verdict"], "PARTIAL")
        self.assertEqual(v4m.verify_results_receipt(_build_synth("5/1", "4/1", "9/2", "7/2", "10/1", True), ptx)["verdict"], "PARTIAL")
        self.assertEqual(v4m.verify_results_receipt(_build_synth("4/1", "4/1", "4/1", "7/2", "30/1", False), ptx)["verdict"], "FALSIFIED")

        # T512 vs T2048/T8192 dense-gap scope check: T512 > 1/5 with T2048/T8192 <= 1/5 stays G1=True
        r_t512 = _build_synth("5/1", "4/1", "9/2", "19/5", "10/1", True)
        for row in r_t512["e2e_bpb_study"]["arms"]["V3_DenseHead"]["seeds"]:
            row["contexts"]["T512"]["eval_bpb_q"] = "7/2"
            row["contexts"]["T512"]["eval_bpb_bf16_q"] = "7/2"
        r_t512["e2e_bpb_study"]["arms"]["V3_DenseHead"]["mean_eval_bpb_by_ctx_q"]["T512"] = "7/2"
        r_t512["context_summary"]["T512"]["v4_minus_dense_q"] = "1/2"
        r_t512["context_summary"]["T512"]["dense_gap_scalar_minus_dense_q"] = "3/2"
        r_t512["context_summary"]["T512"]["v4_gap_closed_fraction_q"] = "2/3"
        r_t512 = _resign(r_t512, "result_sha256")
        self.assertEqual(v4m.verify_results_receipt(r_t512, ptx)["verdict"], "CONFIRMED")

        r_bnd = _build_synth("5/1", "4/1", "9/2", "19/5", "10/1", True)
        r_bnd["silicon_e2e_bench"]["v4_1l_fp64_max_err_q"] = "0/1"
        r_bnd["e2e_bpb_study"]["h256_ortho_fp64_max_err_q"] = "1/100000000000"
        r_bnd = _resign(r_bnd, "result_sha256")
        self.assertEqual(v4m.verify_results_receipt(r_bnd, ptx)["verdict"], "CONFIRMED")

        for g2_mut in (
            lambda p, r: p.__setitem__("nonzero_spill_kernels", 1),
            lambda p, r: p.__setitem__("total_kernels", 1),
            lambda p, r: r["silicon_e2e_bench"].__setitem__("clean_packed_u32", 1),
            lambda p, r: r["silicon_e2e_bench"].__setitem__("clean_sys_atomics", 1),
            lambda p, r: r["silicon_e2e_bench"].__setitem__("v4_sec16_sys_atomics", 0),
            lambda p, r: r["silicon_e2e_bench"].__setitem__("v4_sec16_hidden_max_abs_err_q", "1/10"),
            lambda p, r: r["silicon_e2e_bench"].__setitem__("v4_sec16_logits_max_abs_err_q", "1/10"),
        ):
            p_bad = copy.deepcopy(ptx)
            r_bad = copy.deepcopy(res)
            g2_mut(p_bad, r_bad)
            r_bad = _resign(r_bad, "result_sha256")
            self.assertCode("RESULTS_GATES_MISMATCH", v4m.verify_results_receipt, r_bad, p_bad)

    def test_05_cli_subprocess_symlinks_fifos_and_raw_byte_tamper_handling(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(v4m.main(["--check", "--json"]), 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["status"], "PASS")

        buf_req = io.StringIO()
        with contextlib.redirect_stdout(buf_req):
            self.assertEqual(v4m.main(["--check", "--require-verdict", "CONFIRMED"]), 0)
        self.assertTrue(buf_req.getvalue().startswith("PASS CONFIRMED "))

        # Real subprocess invocation covering `if __name__ == "__main__":`
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(V4_DIR.joinpath("verify_holosphere_v4_monarch2.py")),
                "--check",
                "--require-verdict",
                "CONFIRMED",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("PASS CONFIRMED "))

        err_v = io.StringIO()
        with contextlib.redirect_stderr(err_v):
            self.assertEqual(v4m.main(["--check", "--require-verdict", "FALSIFIED"]), 2)
        self.assertEqual(json.loads(err_v.getvalue())["code"], "VERDICT_REQUIREMENT_FAILED")

        # Unexpected runtime error in main() exits RC=2 with structured JSON (zero traceback)
        with patch.object(v4m, "verify_on_disk", side_effect=OSError("disk io fail")):
            err_u = io.StringIO()
            with contextlib.redirect_stderr(err_u):
                self.assertEqual(v4m.main(["--check"]), 2)
            self.assertEqual(json.loads(err_u.getvalue())["code"], "UNEXPECTED_RUNTIME_ERROR")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self.assertCode("MISSING_FILE", v4m.load_strict_canonical_json, tdp.joinpath("missing.json"))
            self.assertCode("MISSING_FILE", v4m.read_repo_regular_bytes, tdp.joinpath("no_such_root"), v4m.PTXAS_REL)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(v4m.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err.getvalue())["code"], "MISSING_FILE")

            tracked_rels = (
                v4m.KERNEL_CU_REL,
                v4m.BENCH_CU_REL,
                v4m.BPB_STUDY_REL,
                v4m.PTXAS_REL,
                v4m.RESULTS_REL,
            )
            for rel in tracked_rels:
                dst = tdp.joinpath(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(REPO_ROOT.joinpath(rel).read_bytes())
            self.assertEqual(v4m.verify_on_disk(tdp)["status"], "PASS")
            self.assertCode("MISSING_FILE", v4m.read_repo_regular_bytes, tdp, "receipts/v4_monarch2/nope.json")
            with patch("os.open", side_effect=PermissionError("denied")):
                self.assertCode("READ_OR_PARSE_ERROR", v4m.read_repo_regular_bytes, tdp, v4m.PTXAS_REL)
            with patch("os.open", side_effect=OSError(40, "Too many levels of symbolic links")):
                self.assertCode("SYMLINK_FORBIDDEN", v4m.read_repo_regular_bytes, tdp, v4m.PTXAS_REL)

            # 1-byte tamper on standalone BPB study script must fail closed
            study_dst = tdp.joinpath(v4m.BPB_STUDY_REL)
            orig_study = study_dst.read_bytes()
            study_dst.write_bytes(orig_study + b"\n")
            self.assertCode("RESULTS_STUDY_SCRIPT_DISK", v4m.verify_on_disk, tdp)
            study_dst.write_bytes(orig_study)

            # Symlink tamper across every tracked artifact + repo_root symlink + subdir symlink -> RC=2 SYMLINK_FORBIDDEN
            ext_target = tdp.joinpath("external_copy.bin")
            for rel in tracked_rels:
                dst = tdp.joinpath(rel)
                saved = dst.read_bytes()
                ext_target.write_bytes(saved)
                dst.unlink()
                os.symlink(str(ext_target), str(dst))
                self.assertCode("SYMLINK_FORBIDDEN", v4m.read_regular_bytes, dst)
                err_sym = io.StringIO()
                with contextlib.redirect_stderr(err_sym):
                    self.assertEqual(v4m.main(["--check", "--repo-root", str(tdp)]), 2)
                self.assertEqual(json.loads(err_sym.getvalue())["code"], "SYMLINK_FORBIDDEN")
                dst.unlink()
                dst.write_bytes(saved)

            sym_root = tdp.joinpath("sym_repo_root")
            os.symlink(str(tdp), str(sym_root))
            self.assertCode("SYMLINK_FORBIDDEN", v4m.verify_on_disk, sym_root)
            sym_root.unlink()

            real_sub = tdp.joinpath("real_receipts_v4")
            tdp.joinpath("receipts", "v4_monarch2").rename(real_sub)
            os.symlink(str(real_sub), str(tdp.joinpath("receipts", "v4_monarch2")))
            self.assertCode("SYMLINK_FORBIDDEN", v4m.verify_on_disk, tdp)
            tdp.joinpath("receipts", "v4_monarch2").unlink()
            tdp.joinpath("receipts", "v4_monarch2").write_bytes(b"not a dir")
            self.assertCode("READ_OR_PARSE_ERROR", v4m.verify_on_disk, tdp)
            tdp.joinpath("receipts", "v4_monarch2").unlink()
            real_sub.rename(tdp.joinpath("receipts", "v4_monarch2"))

            # Non-blocking FIFO refusal inside repo tree (must not hang!)
            res_dst = tdp.joinpath(v4m.RESULTS_REL)
            saved_res = res_dst.read_bytes()
            res_dst.unlink()
            os.mkfifo(str(res_dst))
            self.assertCode("NOT_REGULAR_FILE", v4m.verify_on_disk, tdp)
            res_dst.unlink()
            res_dst.write_bytes(saved_res)

            # Permission errors on subdir / leaf open inside dir_fd traversal
            orig_open = os.open

            def _fail_subdir(path, flags, *a, **kw):
                if path == "v4_monarch2":
                    raise PermissionError("subdir denied")
                return orig_open(path, flags, *a, **kw)

            def _fail_leaf(path, flags, *a, **kw):
                if path == "holosphere_v4_monarch2_ptxas_receipt.json":
                    raise PermissionError("leaf denied")
                return orig_open(path, flags, *a, **kw)

            with patch("os.open", side_effect=_fail_subdir):
                self.assertCode("READ_OR_PARSE_ERROR", v4m.read_repo_regular_bytes, tdp, v4m.PTXAS_REL)
            with patch("os.open", side_effect=_fail_leaf):
                self.assertCode("READ_OR_PARSE_ERROR", v4m.read_repo_regular_bytes, tdp, v4m.PTXAS_REL)

            res_raw = REPO_ROOT.joinpath(v4m.RESULTS_REL).read_text(encoding="utf-8")
            dup_raw = "{\n  \"energy_power_status\": \"45.2 W claimed via RAPL\",\n" + res_raw[2:]
            res_dst.write_text(dup_raw, encoding="utf-8")
            self.assertCode("DUPLICATE_JSON_KEY", v4m.verify_on_disk, tdp)

            res_dst.write_text("{\"x\": NaN}\n", encoding="utf-8")
            self.assertCode("NON_FINITE_JSON", v4m.verify_on_disk, tdp)

            res_dst.write_text("{\"x\": 1.5}\n", encoding="utf-8")
            self.assertCode("FLOAT_JSON_FORBIDDEN", v4m.verify_on_disk, tdp)

            res_dst.write_text("{\"x\": " + ("9" * 4400) + "}\n", encoding="utf-8")
            err_big = io.StringIO()
            with contextlib.redirect_stderr(err_big):
                self.assertEqual(v4m.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err_big.getvalue())["code"], "INT_DIGIT_LIMIT")

            res_dst.write_text("{\"x\": 1e400}\n", encoding="utf-8")
            self.assertCode("FLOAT_JSON_FORBIDDEN", v4m.verify_on_disk, tdp)

            res_dst.write_bytes(REPO_ROOT.joinpath(v4m.RESULTS_REL).read_bytes() + b"\n")
            self.assertCode("RAW_JSON_NOT_CANONICAL", v4m.verify_on_disk, tdp)

            res_dst.write_text("{bad", encoding="utf-8")
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                self.assertEqual(v4m.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err2.getvalue())["code"], "READ_OR_PARSE_ERROR")


if __name__ == "__main__":
    unittest.main()
