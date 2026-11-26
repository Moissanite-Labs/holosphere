#!/usr/bin/env python3
"""
Deterministic Negative-Control & Certificate-Binding Test Suite for `LoopCDStrang`
(`lean/loopcd_strang/verify_loopcd_strang.py` and `loopcd_strang_certificate.json`).

Pure Python 3 standard library. Zero floats in AST, zero bare asserts, zero test skips.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = Path(REPO_ROOT, "lean", "loopcd_strang")
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
try:
    import verify_loopcd_strang as vmod
except SystemExit as _imp_exit:
    raise RuntimeError("verify_loopcd_strang triggered SystemExit on import") from _imp_exit


def _rehash_cert(cert_dict: dict[str, object]) -> dict[str, object]:
    body = {k: v for k, v in cert_dict.items() if k != "payload_sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    out = dict(body)
    out["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return out


class TestLoopCDStrangCert(unittest.TestCase):
    def test_01_pure_q_ast_guard_and_qdiv_controls(self) -> None:
        self.assertEqual(vmod.qdiv(Fraction(3, 5), Fraction(9, 10)), Fraction(2, 3))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^QDIV_ZERO_DENOM:"):
            vmod.qdiv(Fraction(1), Fraction(0))

        good_src = (
            "from __future__ import annotations\n"
            "import fractions as fr\n"
            "from fractions import Fraction\n"
            "import json as j\n"
            "import math\n"
            "from math import gcd, isqrt\n"
            "from pathlib import Path\n"
            "import re\n"
            "from typing import Any\n"
            "RX = re.compile('a')\n"
            "class _E(ValueError):\n"
            "    def __init__(self, m: str) -> None:\n"
            "        super().__init__(m)\n"
            "def _bad_num(tok: Any) -> Path:\n"
            "    raise ValueError(tok)\n"
            "def _int_ok(tok: str) -> int:\n"
            "    require_ok(len(tok) <= 10)\n"
            "    return int(tok)\n"
            "def _helper(a: Fraction, b: tuple[Fraction, fr.Fraction], u=1) -> Fraction | None:\n"
            "    return Fraction(a.numerator, a.denominator)\n"
            "def _plain(v):\n"
            "    ok_t = (type(v) is int) or (type(v) is not str)\n"
            "    _ = j.loads('1', parse_float=_bad_num, parse_constant=_bad_num, parse_int=_int_ok)\n"
            "    _ = j.dumps({'a': 1})\n"
            "    _ = j.JSONDecodeError\n"
            "    return ok_t or isinstance(v, (float, complex)) or isinstance(v, float)\n"
            "z: Fraction | None = None\n"
            "w_ann: Fraction = Fraction(1, 3)\n"
            "w_mod = fr.Fraction(1, 3)\n"
            "k = 2\n"
            "x = Fraction(3, 5) * Fraction(-2) + (-Fraction(+1, 1 << 3)) - Fraction(1, 1 << k)\n"
            "p = (2 ** 9) + (1 << 4)\n"
            "p2 = 2\n"
            "p2 **= 3\n"
            "y = Fraction(x.numerator * 2 + w_mod.numerator - 0, x.denominator + w_ann.numerator)\n"
            "g = math.gcd(6, 9) + isqrt(16) + gcd(4, 8)\n"
        )
        res_ast = vmod.verify_pure_q_ast(good_src)
        self.assertTrue(res_ast["pure_q_ast_clean"])

        bad_snippets = (
            ("def (\n", r"^AST_SYNTAX_ERROR:"),
            ("x = 1\x00\n", r"^AST_SYNTAX_ERROR:"),
            ("x = 1 / 2\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = 1 // 2\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = 4\nx //= 2\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = '%f' % 1\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = 5 % 2\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = 5\nx %= 2\n", r"^AST_DIV_FORBIDDEN:"),
            ("x = 2 ** -1\n", r"^AST_POW_EXP_FORBIDDEN:"),
            ("x = 2 ** (-2)\n", r"^AST_POW_EXP_FORBIDDEN:"),
            ("x = 2\nx **= -1\n", r"^AST_POW_EXP_FORBIDDEN:"),
            ("k = 3\nx = 2 ** k\n", r"^AST_POW_EXP_FORBIDDEN:"),
            ("x = 2 ** True\n", r"^AST_POW_EXP_FORBIDDEN:"),
            ("x = f'{1:.3f}'\n", r"^AST_FORMAT_SPEC_FORBIDDEN:"),
            ("x = '{:.3f}'.format(1)\n", r"^AST_FLOAT_CALL:"),
            ("f_m = '{:.3f}'.format\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = format(1, '.3f')\n", r"^AST_FLOAT_CALL:"),
            ("re = 1\n", r"^AST_FLOAT_CALL:"),
            ("x = compile('1', '<s >', 'eval')\n", r"^AST_FLOAT_CALL:"),
            ("x = mod.compile('1')\n", r"^AST_FLOAT_CALL:"),
            ("t = type\n", r"^AST_FLOAT_CALL:"),
            ("t = type(1)\n", r"^AST_FLOAT_CALL:"),
            ("b = type(1, 2) is int\n", r"^AST_FLOAT_CALL:"),
            ("b = type(1, k=2) is int\n", r"^AST_FLOAT_CALL:"),
            ("b = type(1) == int\n", r"^AST_FLOAT_CALL:"),
            ("b = type(1) is int == True\n", r"^AST_FLOAT_CALL:"),
            ("from fractions import Fraction\nx = type(Fraction(1, 2))('0.5')\n", r"^AST_(?:DYNAMIC_CALL_FUNC|FLOAT_CALL):"),
            ("x = (f)()[0]()\n", r"^AST_DYNAMIC_CALL_FUNC:"),
            ("from json import loads\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import json\nj = json\n", r"^AST_FORBIDDEN_JSON_USE:"),
            ("import json\nx = json.load(fp)\n", r"^AST_FORBIDDEN_JSON_USE:"),
            ("import json as j\nx = j.load(fp)\n", r"^AST_FORBIDDEN_JSON_USE:"),
            ("import json\nf = json.loads\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            ("import json\nx = json.loads('1')\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            ("import json\nx = json.loads('1', parse_float=fn)\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            ("import json\nx = json.loads('1', parse_constant=fn)\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            ("import json\nx = json.loads('1', parse_float=1, parse_constant=1)\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            ("import json\nx = json.loads('1', parse_float=float, parse_constant=fn)\n", r"^AST_(?:JSON_LOADS_UNGUARDED|FLOAT_CALL):"),
            ("import json\nf = None\nx = json.loads('1', parse_float=f, parse_constant=f)\n", r"^AST_JSON_LOADS_UNGUARDED:"),
            (
                "import json\ndef nop(t):\n    return 0\nx = json.loads('1', parse_float=nop, parse_constant=nop)\n",
                r"^AST_JSON_LOADS_UNGUARDED:",
            ),
            (
                "import json\ndef two(t):\n    a = 1\n    raise ValueError(t)\nx = json.loads('1', parse_float=two, parse_constant=two)\n",
                r"^AST_JSON_LOADS_UNGUARDED:",
            ),
            (
                "import json\ndef _b(t):\n    raise ValueError(t)\n_b = None\nx = json.loads('1', parse_float=_b, parse_constant=_b)\n",
                r"^AST_JSON_LOADS_UNGUARDED:",
            ),
            (
                "import json\ndef _b(t):\n    raise ValueError(t)\nx = json.loads('1', parse_float=_b, parse_constant=_b)\n",
                r"^AST_JSON_LOADS_UNGUARDED:",
            ),
            (
                "import json\ndef _b(t):\n    raise ValueError(t)\nx = json.loads('1', parse_float=_b, parse_constant=_b, parse_int=int)\n",
                r"^AST_JSON_LOADS_UNGUARDED:",
            ),
            ("import ast\nx = ast.literal_eval('1.5')\n", r"^AST_FLOAT_CALL:"),
            ("import ast\nf = ast.literal_eval\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("import _decimal\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import _operator\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from _operator import truediv\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from sys import modules\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from __future__ import division\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from pathlib import PurePath\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from typing import cast\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import math as Fraction\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("from math import gcd as Fraction\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("import fractions\nr = fractions\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("x = obj.Fraction(1, 2)\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            (
                "from fractions import Fraction\ndef bad_ann(a: obj.Fraction):\n    return Fraction(a.numerator, 1)\n",
                r"^AST_FRACTION_ARG_UNSAFE:",
            ),
            (
                "from fractions import Fraction\nw = obj.Fraction(1, 2)\nx = Fraction(w.numerator, 1)\n",
                r"^AST_FRACTION_ARG_UNSAFE:",
            ),
            ("def Fraction(x):\n    return x\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("def f(Fraction):\n    return 1\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("class Fraction:\n    pass\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("try:\n    x = 1\nexcept ValueError as Fraction:\n    pass\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("x = 0.25\n", r"^AST_FLOAT_LITERAL:"),
            (f"x = {chr(49)}{chr(106)}\n", r"^AST_FLOAT_LITERAL:"),
            ('f = __builtins__["float"]\n', r"^AST_BUILTINS_FORBIDDEN:"),
            ('x = __builtins__["float"]("1e308")\n', r"^AST_DYNAMIC_CALL_FUNC:"),
            ("x = (1).__init__\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ('from fractions import Fraction\nx: Fraction\nF = __annotations__["x"]\n', r"^AST_BUILTINS_FORBIDDEN:"),
            ("x = __loader__\n", r"^AST_BUILTINS_FORBIDDEN:"),
            ("x = (1).__float__()\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("g = (1).__getattribute__('__truediv__')\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = (1).__getattribute__('__truediv__')(2)\n", r"^AST_DYNAMIC_CALL_FUNC:"),
            ("x = (1).__class__\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = type(1).__dict__\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = (1).real\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = (1).imag\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            (
                'from fractions import Fraction\ndef f(x: Fraction):\n    pass\nF = f.__annotations__["x"]\n',
                r"^AST_FROM_FLOAT_FORBIDDEN:",
            ),
            ("x = float(1)\n", r"^AST_FLOAT_CALL:"),
            ("f = float\nx = f(1)\n", r"^AST_FLOAT_CALL:"),
            ("g = getattr\n", r"^AST_FLOAT_CALL:"),
            ("x = divmod(5, 2)\n", r"^AST_FLOAT_CALL:"),
            ("x = vars()\n", r"^AST_FLOAT_CALL:"),
            ("x = memoryview(b'0')\n", r"^AST_FLOAT_CALL:"),
            ("x = buf.cast('d')\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("x = Decimal(1)\n", r"^AST_FLOAT_CALL:"),
            ("x = mod.Decimal(1)\n", r"^AST_FLOAT_CALL:"),
            ("x = getattr(obj, 'a')\n", r"^AST_FLOAT_CALL:"),
            ("x = eval('1')\n", r"^AST_FLOAT_CALL:"),
            ("x = pow(2, -1)\n", r"^AST_FLOAT_CALL:"),
            ("x = (1).__truediv__(2)\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("import sys\nm = sys.modules['math']\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("import struct\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import random\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import time\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from . import foo\n", r"^AST_REL_IMPORT_FORBIDDEN:"),
            ("import builtins\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import operator\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import importlib\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from fractions import Fraction\ny = Fraction.from_float(1)\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ("from fractions import Fraction\ny = Fraction.from_decimal(1)\n", r"^AST_FROM_FLOAT_FORBIDDEN:"),
            ('from fractions import Fraction\nF = Fraction\nx = F("0.5")\n', r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ('from fractions import Fraction as F\nG = F\nx = G("0.5")\n', r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ('import fractions as fr\nF = fr.Fraction\nx = F("0.5")\n', r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("from fractions import Fraction\nx: (F := Fraction)\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("from fractions import Fraction\nx: 1 + Fraction\n", r"^AST_FRACTION_REBIND_FORBIDDEN:"),
            ("import decimal\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import sympy\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import mpmath\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("import numpy as np\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from decimal import Decimal\n", r"^AST_FORBIDDEN_IMPORT:"),
            ("from math import *\n", r"^AST_WILDCARD_IMPORT:"),
            ("from math import cos\n", r"^AST_FORBIDDEN_MATH_SYM:"),
            ("from fractions import gcd\n", r"^AST_FORBIDDEN_FRACTIONS_SYM:"),
            ("import math\nx = math.sin(0)\n", r"^AST_FORBIDDEN_MATH_ATTR:"),
            ("import math\nm = math\n", r"^AST_FORBIDDEN_MATH_ATTR:"),
            ("import math as m\nx = m.log(2)\n", r"^AST_FORBIDDEN_MATH_ATTR:"),
            ("class C:\n    numerator = 1\n", r"^AST_CUSTOM_NUMERATOR_DEF:"),
            ("class C:\n    denominator: int = 1\n", r"^AST_CUSTOM_NUMERATOR_DEF:"),
            ("def numerator():\n    return 1\n", r"^AST_CUSTOM_NUMERATOR_DEF:"),
            ("from fractions import Fraction\ndef bad_u(u):\n    return Fraction(u.numerator, 1)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(foo().numerator, 1)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            (
                "from fractions import Fraction\nx = Fraction(1, 2)\nx = other\ny = Fraction(x.numerator, 1)\n",
                r"^AST_FRACTION_ARG_UNSAFE:",
            ),
            ("from fractions import Fraction\np = (1, 2)\nx = Fraction(*p)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction('1/2')\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction as F\nx = F('1/2')\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("import fractions as fr\nx = fr.Fraction('1/2')\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\ns = '1/2'\nx = Fraction(s)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(~1)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(y.real)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(y.other)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(1 + s)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(s + 1)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(True, 2)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(-True, 2)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(s << 1)\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction(1 << 'a')\n", r"^AST_FRACTION_ARG_UNSAFE:"),
            ("from fractions import Fraction\nx = Fraction()\n", r"^AST_FRACTION_ARITY:"),
            ("from fractions import Fraction\nx = Fraction(1, 2, 3)\n", r"^AST_FRACTION_ARITY:"),
            ("from fractions import Fraction\nx = Fraction(numerator=1)\n", r"^AST_FRACTION_KW:"),
        )
        for snippet, pat in bad_snippets:
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, pat, msg=snippet):
                vmod.verify_pure_q_ast(snippet)

    def test_02_thm1_and_thm2_negative_controls(self) -> None:
        r1 = vmod.verify_thm1_chord_saturation_q()
        self.assertEqual(r1["grid_pairs_verified"], 36)
        self.assertEqual(r1["strict_monotonicity_steps"], 30)

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^STEREO_CIRCLE_FAIL:"):
            vmod.verify_thm1_chord_saturation_q(stereo_fn=lambda _s: (Fraction(0), Fraction(0)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^TAN_ALPHA_LAW_FAIL:"):
            vmod.verify_thm1_chord_saturation_q(tan_fn=lambda _s, _om: Fraction(99))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SAT_GAP_IDENTITY_FAIL:"):
            vmod.verify_thm1_chord_saturation_q(gap_fn=lambda _s, _om: Fraction(99))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SAT_GAP_NONPOS:"):
            vmod.verify_thm1_chord_saturation_q(
                s_grid=(Fraction(7, 4),),
                om_grid=(Fraction(1),),
                stereo_fn=lambda _s: (Fraction(3, 5), Fraction(4, 5)),
                tan_fn=lambda _s, _om: Fraction(4, 7),
                gap_fn=lambda _s, _om: Fraction(0),
            )
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^OMEGA_MONO_FAIL:"):
            vmod.verify_thm1_chord_saturation_q(om_grid=(Fraction(1), Fraction(1)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^OMEGA_MONO_FAIL:"):
            vmod.verify_thm1_chord_saturation_q(om_grid=(Fraction(1), Fraction(0)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^WITNESS_90DEG_FAIL:"):
            calls = {"n": 0}

            def _tan_bad_last(s: Fraction, om: Fraction) -> Fraction:
                calls["n"] += 1
                val = vmod.qdiv(Fraction(2) * om * s, Fraction(1) + (Fraction(1) + Fraction(2) * om) * s * s)
                return Fraction(1, 3) if calls["n"] == 37 else val

            vmod.verify_thm1_chord_saturation_q(tan_fn=_tan_bad_last)

        r2 = vmod.verify_thm2_palindromic_strang_bch_q()
        self.assertEqual(r2["cubic_divisibility_checks"], 9)
        self.assertEqual(r2["dyadic_scales_verified"], 4)

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CAYLEY_DENOM_NONPOS:"):
            vmod.cayley_compose_q((Fraction(1), Fraction(0), Fraction(0)), (Fraction(1), Fraction(0), Fraction(0)))

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^STRANG_T0_FAIL:"):
            vmod.verify_thm2_palindromic_strang_bch_q(
                right_defect_fn=lambda t, s, u, v, a, b: (1, 0, 0) if t == 0 else (0, 0, 0)
            )
        for idx in (0, 1, 2):
            bad_vec = tuple(1 if j == idx else 0 for j in range(3))
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^STRANG_RIGHT_CUBIC_FAIL:"):
                vmod.verify_thm2_palindromic_strang_bch_q(
                    right_defect_fn=lambda t, s, u, v, a, b, bv=bad_vec: (0, 0, 0) if t == 0 else bv
                )
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^STRANG_LEFT_QUAD_FAIL:"):
                vmod.verify_thm2_palindromic_strang_bch_q(
                    left_defect_fn=lambda t, s, u, v, a, b, bv=bad_vec: bv
                )

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^DYADIC_ER_NONPOS:"):
            vmod.verify_thm2_palindromic_strang_bch_q(su2_err_pair_fn=lambda _k: (Fraction(0), Fraction(1)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^DYADIC_HIERARCHY_FAIL:"):
            vmod.verify_thm2_palindromic_strang_bch_q(su2_err_pair_fn=lambda _k: (Fraction(2), Fraction(2)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^RIGHT_HALVING_RATIO_LOW:"):
            vmod.verify_thm2_palindromic_strang_bch_q(
                su2_err_pair_fn=lambda k: (Fraction(1, 60**k), Fraction(2, 16**k))
            )
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEFT_HALVING_RATIO_LOW:"):
            vmod.verify_thm2_palindromic_strang_bch_q(
                su2_err_pair_fn=lambda k: (Fraction(1, 64**k), Fraction(2, 15**k))
            )
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEFT_HALVING_RATIO_HIGH:"):
            vmod.verify_thm2_palindromic_strang_bch_q(
                su2_err_pair_fn=lambda k: (Fraction(1, 64**k), Fraction(2, 17**k))
            )

    def test_03_thm3_and_thm4_negative_controls(self) -> None:
        r3 = vmod.verify_thm3_acute_ball_floors_q()
        self.assertEqual(r3["min_spinor_denom"], "9/25")
        self.assertEqual(r3["min_cayley_denom"], "81/625")

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^ACUTE_BALL_W1_OUT:"):
            vmod.verify_thm3_acute_ball_floors_q(acute_vectors=((Fraction(1), Fraction(0), Fraction(0)),))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SPINOR_FLOOR_FAIL:"):
            vmod.verify_thm3_acute_ball_floors_q(spinor_denom_fn=lambda _w1, _wr: Fraction(8, 25))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CAYLEY_FLOOR_FAIL:"):
            vmod.verify_thm3_acute_ball_floors_q(cayley_denom_fn=lambda _w1, _wr: Fraction(80, 625))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SPINOR_TIGHTNESS_FAIL:"):
            vmod.verify_thm3_acute_ball_floors_q(spinor_denom_fn=lambda _w1, _wr: Fraction(10, 25))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CAYLEY_TIGHTNESS_FAIL:"):
            vmod.verify_thm3_acute_ball_floors_q(cayley_denom_fn=lambda _w1, _wr: Fraction(82, 625))

        r4 = vmod.verify_thm4_commutative_collapse_q()
        self.assertEqual(r4["collinear_collapse_checks"], 30)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^COLLINEAR_CROSS_NONZERO:"):
            vmod.verify_thm4_commutative_collapse_q(cross_fn=lambda _a, _b: (Fraction(1), Fraction(0), Fraction(0)))
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^COMM_QUAT_FAIL:"):
            vmod.verify_thm4_commutative_collapse_q(quat_mul_fn=lambda p, _q: p)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^COMM_LEFT_RIGHT_MISMATCH:"):
            vmod.verify_thm4_commutative_collapse_q(compose_fn=lambda a, _b: a)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^COMM_CAYLEY_SUM_MISMATCH:"):
            c_cnt = {"n": 0}

            def _bad_cayley_sum(
                a: tuple[Fraction, Fraction, Fraction], b: tuple[Fraction, Fraction, Fraction]
            ) -> tuple[Fraction, Fraction, Fraction]:
                c_cnt["n"] += 1
                res_c = vmod.cayley_compose_q(a, b)
                return (Fraction(999), Fraction(0), Fraction(0)) if c_cnt["n"] == 11 else res_c

            vmod.verify_thm4_commutative_collapse_q(compose_fn=_bad_cayley_sum)

    def test_04_lean_char_lexer_and_axiom_validator_controls(self) -> None:
        cleaned = vmod.strip_lean_comments_and_strings(
            "-- sorry in line comment\n/- nested /- bv_decide -/ comment -/\ndef s := \"quote \\\" sorry\"\ndef xr\"sorry\" := 1\n"
        )
        self.assertNotIn("sorry", cleaned)
        self.assertNotIn("bv_decide", cleaned)
        sub_cleaned = vmod.strip_lean_comments_and_strings("theorem t : 2 - 1 = 1 := by bv_decide\n")
        self.assertIn("bv_decide", sub_cleaned)

        # Char-literal quote desync regression controls ('"', '\'', '\\', '\x41', '\u0041', identifier primes)
        char_desync_payloads = (
            "def x := '\"'\ntheorem cheat : 1 = 2 := by sorry\ndef y := '\"'\n",
            "def x := '\\\"'\ntheorem cheat : 1 = 2 := by sorry\ndef y := '\\''\n",
            "def x := '\\x22'\ntheorem cheat : 1 = 2 := by sorry\ndef y := '\\u0022'\n",
            "theorem t' (x' : Nat) : x' = x' := by sorry\n",
            'def raw := r#"sorry"#\n',
        )
        for payload in char_desync_payloads:
            sc = vmod.strip_lean_comments_and_strings(payload)
            self.assertIsNotNone(vmod._FORBIDDEN_LEAN_CODE_RE.search(sc), msg=payload)

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNTERMINATED_BLOCK_COMMENT:"):
            vmod.strip_lean_comments_and_strings("/- open comment")
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNTERMINATED_STRING_LITERAL:"):
            vmod.strip_lean_comments_and_strings('def x := "open')
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNTERMINATED_STRING_LITERAL:"):
            vmod.strip_lean_comments_and_strings("def x' := \"open")
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNTERMINATED_STRING_LITERAL:"):
            vmod.strip_lean_comments_and_strings('def p := \'\\"x"ok"\n')
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNTERMINATED_STRING_LITERAL:"):
            vmod.strip_lean_comments_and_strings('def raw := r#"open\n')

        good_lines = [
            "'A.t0' does not depend on any axioms",
            "'A.t1' depends on axioms: [propext]",
            "'A.t2' depends on axioms: [propext, Quot.sound]",
        ]
        counts = vmod.validate_axiom_audit_lines(good_lines, 3)
        self.assertEqual(counts["zero_axiom_theorems"], 1)
        self.assertEqual(counts["propext_only_theorems"], 1)
        self.assertEqual(counts["propext_quot_theorems"], 1)

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^AXIOM_LINE_COUNT_FAIL:"):
            vmod.validate_axiom_audit_lines(good_lines, 4)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^UNRECOGNIZED_AXIOM_LINE:"):
            vmod.validate_axiom_audit_lines(["garbage line"], 1)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MALFORMED_AXIOM_LBRACKET:"):
            vmod.validate_axiom_audit_lines(["'A.t' depends on axioms: propext]"], 1)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MALFORMED_AXIOM_RBRACKET:"):
            vmod.validate_axiom_audit_lines(["'A.t' depends on axioms: [propext"], 1)
        for banned_ax in ("Classical.choice", "Foo._native.bv_decide.ax_1", "sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"):
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^BANNED_AXIOM:"):
                vmod.validate_axiom_audit_lines([f"'A.t' depends on axioms: [propext, {banned_ax}]"], 1)
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^FORBIDDEN_AXIOM:"):
            vmod.validate_axiom_audit_lines(["'A.t' depends on axioms: [propext, customAx]"], 1)

    def test_05_on_disk_certificate_self_binding_and_tamper_controls(self) -> None:
        cert_path = Path(PKG_DIR, "loopcd_strang_certificate.json")
        self.assertTrue(cert_path.is_file(), msg="Committed loopcd_strang_certificate.json missing on disk")
        on_disk_cert = json.loads(cert_path.read_text(encoding="utf-8"))

        check_res = vmod.validate_certificate(on_disk_cert, pkg_dir=PKG_DIR)
        self.assertEqual(check_res["status"], "PASS")
        self.assertFalse(check_res["lean_live_replayed"])
        self.assertEqual(check_res["verification_mode"], "offline_static_and_pure_q")
        self.assertEqual(check_res["public_theorems_verified"], 38)

        buf_check = io.StringIO()
        with patch("sys.stdout", buf_check):
            rc_check = vmod.main(["--check"])
        self.assertEqual(rc_check, 0)
        parsed_check = json.loads(buf_check.getvalue())
        self.assertFalse(parsed_check["lean_live_replayed"])
        self.assertEqual(parsed_check["payload_sha256"], on_disk_cert["payload_sha256"])

        buf_json = io.StringIO()
        with patch("sys.stdout", buf_json):
            rc_json = vmod.main(["--json"])
        self.assertEqual(rc_json, 0)
        self.assertEqual(json.loads(buf_json.getvalue()), on_disk_cert)

        def _assert_cli_fail(argv_fail: list[str], expected_code: str) -> None:
            err_buf = io.StringIO()
            with patch("sys.stderr", err_buf):
                rc_err = vmod.main(argv_fail)
            self.assertEqual(rc_err, 2, msg=f"argv={argv_fail} stderr={err_buf.getvalue()}")
            err_obj = json.loads(err_buf.getvalue())
            self.assertEqual(err_obj["status"], "FAIL")
            self.assertEqual(err_obj["error_code"], expected_code)

        # --write-cert without --live-lean & CLI bad inputs must fail closed with rc == 2 and zero tracebacks
        _assert_cli_fail(["--write-cert"], "WRITE_CERT_REQUIRES_LIVE_LEAN")
        _assert_cli_fail(["--check", str(Path(PKG_DIR, "nonexistent_cert.json"))], "MISSING_CERT_FILE")
        with tempfile.TemporaryDirectory() as td_c:
            dup_p = Path(td_c, "dup.json")
            dup_p.write_text('{"module": "A", "module": "B"}\n', encoding="utf-8")
            _assert_cli_fail(["--check", str(dup_p)], "CERT_DUPLICATE_KEY")

            bad_json_p = Path(td_c, "bad.json")
            bad_json_p.write_text("{bad\n", encoding="utf-8")
            _assert_cli_fail(["--check", str(bad_json_p)], "CERT_INVALID_JSON")

            for fname, bad_num_txt in (
                ("flt.json", '{"a": 0.5}\n'),
                ("ovf.json", '{"a": 1e400}\n'),
                ("nan.json", '{"a": NaN}\n'),
            ):
                p_num = Path(td_c, fname)
                p_num.write_text(bad_num_txt, encoding="utf-8")
                _assert_cli_fail(["--check", str(p_num)], "CERT_FORBIDDEN_JSON_NUM")

            bad_utf8_p = Path(td_c, "utf8.json")
            bad_utf8_p.write_bytes(b"\xff\xfe")
            _assert_cli_fail(["--check", str(bad_utf8_p)], "CERT_INVALID_UTF8")

            bom_p = Path(td_c, "bom.json")
            bom_p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n')
            _assert_cli_fail(["--check", str(bom_p)], "CERT_INVALID_UTF8")

            big_int_p = Path(td_c, "big4400.json")
            big_int_p.write_text('{"n": ' + ("9" * 4400) + "}\n", encoding="utf-8")
            _assert_cli_fail(["--check", str(big_int_p)], "CERT_INT_TOO_LARGE")

            self.assertEqual(vmod.strict_json_loads("9" * 256), int("9" * 256))
            self.assertEqual(vmod.strict_json_loads("-" + ("9" * 256)), -int("9" * 256))
            for over_int in ("9" * 257, "-" + ("9" * 257)):
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_INT_TOO_LARGE:"):
                    vmod.strict_json_loads(over_int)

            vmod._check_json_nesting_depth("[[],[],[]]", max_depth=2)
            vmod._check_json_nesting_depth('{"a": "[[[[[[", "b": "\\"[[[["}', max_depth=2)
            for deep_txt in ("[[[]]]", '["\\"", [[[]]]]'):
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_JSON_TOO_DEEP:"):
                    vmod._check_json_nesting_depth(deep_txt, max_depth=2)
            self.assertIsInstance(vmod.strict_json_loads(("[" * 32) + ("]" * 32)), list)
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_JSON_TOO_DEEP:"):
                vmod.strict_json_loads(("[" * 33) + ("]" * 33))

            # HIGH #2: Symlinked certificate (leaf & parent), non-regular directory/FIFO, and byte-cap bounds MUST fail closed
            sym_cert_p = Path(td_c, "sym_cert.json")
            os.symlink(str(cert_path), str(sym_cert_p))
            _assert_cli_fail(["--check", str(sym_cert_p)], "SYMLINK_FORBIDDEN")

            sym_parent_d = Path(td_c, "sym_parent_dir")
            os.symlink(str(PKG_DIR), str(sym_parent_d))
            _assert_cli_fail(["--check", str(Path(sym_parent_d, "loopcd_strang_certificate.json"))], "SYMLINK_FORBIDDEN")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SYMLINK_FORBIDDEN:"):
                vmod.write_regular_bytes_nofollow(Path(sym_parent_d, "out.json"), b"{}")

            real_gp = Path(td_c, "real_gp")
            real_sub = Path(real_gp, "sub")
            real_sub.mkdir(parents=True)
            Path(real_sub, "c.json").write_bytes(cert_path.read_bytes())
            sym_gp = Path(td_c, "sym_gp")
            os.symlink(str(real_gp), str(sym_gp))
            _assert_cli_fail(["--check", str(Path(sym_gp, "sub", "c.json"))], "SYMLINK_FORBIDDEN")

            dir_cert_p = Path(td_c, "dir_cert.json")
            dir_cert_p.mkdir()
            _assert_cli_fail(["--check", str(dir_cert_p)], "NON_REGULAR_FILE")

            sz_p = Path(td_c, "sz.bin")
            vmod.write_regular_bytes_nofollow(sz_p, b"a" * 16, max_bytes=16)
            self.assertEqual(vmod.read_regular_bytes_nofollow(sz_p, "MISSING_CERT_FILE", max_bytes=16), b"a" * 16)
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^FILE_TOO_LARGE:"):
                vmod.write_regular_bytes_nofollow(sz_p, b"a" * 17, max_bytes=16)
            sz_p.write_bytes(b"a" * 17)
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^FILE_TOO_LARGE:"):
                vmod.read_regular_bytes_nofollow(sz_p, "MISSING_CERT_FILE", max_bytes=16)
            sz_p.write_bytes(b"b" * 70000)
            self.assertEqual(vmod.read_regular_bytes_nofollow(sz_p, "MISSING_CERT_FILE", max_bytes=70000), b"b" * 70000)
            sz_p.write_bytes(b"b" * 70001)
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^FILE_TOO_LARGE:"):
                vmod.read_regular_bytes_nofollow(sz_p, "MISSING_CERT_FILE", max_bytes=70000)

            fifo_p = Path(td_c, "test_pipe.fifo")
            os.mkfifo(fifo_p)
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^WRITE_OPEN_FAILED:"):
                vmod.write_regular_bytes_nofollow(fifo_p, b"abc")
            with patch.object(vmod.os, "open", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^OPEN_FAILED:"):
                    vmod.read_regular_bytes_nofollow(cert_path, "MISSING_CERT_FILE")
            r_fd = os.open(fifo_p, os.O_RDONLY | os.O_NONBLOCK)
            try:
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^NON_REGULAR_FILE:"):
                    vmod.read_regular_bytes_nofollow(fifo_p, "MISSING_CERT_FILE")
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^NON_REGULAR_FILE:"):
                    vmod.write_regular_bytes_nofollow(fifo_p, b"abc")
            finally:
                os.close(r_fd)
            with patch.object(vmod, "_run_cli_action", side_effect=OSError("disk error")):
                _assert_cli_fail(["--check"], "IO_ERROR")

            noncanon_p = Path(td_c, "noncanon.json")
            noncanon_p.write_bytes(cert_path.read_bytes().replace(b"\n", b"\r\n"))
            _assert_cli_fail(["--check", str(noncanon_p)], "CERT_RAW_FORMAT_DRIFT")

            # Live subprocess check: zero Python tracebacks even under PYTHONINTMAXSTRDIGITS=0
            sub_env = {**os.environ, "PYTHONINTMAXSTRDIGITS": "0"}
            for sub_p in (big_int_p, sym_cert_p):
                r_sub = subprocess.run(
                    [sys.executable, "-B", str(Path(PKG_DIR, "verify_loopcd_strang.py")), "--check", str(sub_p)],
                    env=sub_env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(r_sub.returncode, 2)
                self.assertNotIn("Traceback", r_sub.stderr)

        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                vmod.main(["--check", "--json"])

        # Certificate structural & forged re-hash negative controls
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_NOT_OBJECT:"):
            vmod.validate_certificate(["not", "a", "dict"], pkg_dir=PKG_DIR)
        for bad_hex in (None, 123, "abc", "Z" * 64, "a" * 63):
            bad_c = dict(on_disk_cert)
            bad_c["payload_sha256"] = bad_hex
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_BAD_PAYLOAD_HEX64:"):
                vmod.validate_certificate(bad_c, pkg_dir=PKG_DIR)

        bad_self = dict(on_disk_cert)
        bad_self["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_SELF_HASH_MISMATCH:"):
            vmod.validate_certificate(bad_self, pkg_dir=PKG_DIR)

        no_src = _rehash_cert({k: v for k, v in on_disk_cert.items() if k != "source_files_sha256"})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_MISSING_SOURCE_MAP:"):
            vmod.validate_certificate(no_src, pkg_dir=PKG_DIR)

        extra_src = copy.deepcopy(on_disk_cert)
        extra_src["source_files_sha256"]["extra.lean"] = "a" * 64
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_SOURCE_KEYS_MISMATCH:"):
            vmod.validate_certificate(_rehash_cert(extra_src), pkg_dir=PKG_DIR)

        for bad_fhex in (123, "abc"):
            bad_file_hex = copy.deepcopy(on_disk_cert)
            bad_file_hex["source_files_sha256"]["LoopCDStrang.lean"] = bad_fhex
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERT_BAD_FILE_HEX64:"):
                vmod.validate_certificate(_rehash_cert(bad_file_hex), pkg_dir=PKG_DIR)

        wrong_file_sha = copy.deepcopy(on_disk_cert)
        wrong_file_sha["source_files_sha256"]["LoopCDStrang.lean"] = "f" * 64
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SOURCE_SHA256_MISMATCH:"):
            vmod.validate_certificate(_rehash_cert(wrong_file_sha), pkg_dir=PKG_DIR)

        forged_witness = copy.deepcopy(on_disk_cert)
        forged_witness["thm3_acute_ball_floors"]["min_spinor_denom"] = "1/2"
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^CERTIFICATE_DRIFT:"):
            vmod.validate_certificate(_rehash_cert(forged_witness), pkg_dir=PKG_DIR)

    def test_06_lean_package_sandbox_manifest_and_negative_controls(self) -> None:
        ax_stdout = "\n".join(vmod.EXPECTED_AXIOM_AUDIT_LINES) + "\n"

        def _ok_runner(snap_arg: dict[str, bytes]) -> dict[str, object]:
            self.assertIsInstance(snap_arg, dict)
            self.assertIn("lake-manifest.json", snap_arg)
            return {
                "manifest_present": True,
                "manifest_unchanged": True,
                "build_rc": 0,
                "build_output": "Build completed successfully.",
                "leanchecker_rc": 0,
                "leanchecker_output": "",
                "axiom_rc": 0,
                "axiom_stdout": ax_stdout,
            }

        on_disk_cert = json.loads(Path(PKG_DIR, "loopcd_strang_certificate.json").read_text(encoding="utf-8"))
        live_check = vmod.validate_certificate(on_disk_cert, pkg_dir=PKG_DIR, run_live_lean=False, lean_runner=_ok_runner)
        self.assertTrue(live_check["lean_live_replayed"])
        self.assertEqual(live_check["verification_mode"], "live_lean_kernel_and_pure_q")

        # Verify default_live_lean_runner materializes lake-manifest.json from in-memory snapshot
        observed_files: list[str] = []

        def _fake_subproc_run(cmd: list[str], cwd: Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
            _ = kwargs
            if cmd[:2] == ["lake", "build"]:
                observed_files.extend(sorted(str(p.relative_to(cwd)).replace("\\", "/") for p in cwd.rglob("*") if p.is_file()))
                return subprocess.CompletedProcess(cmd, 0, stdout="Build completed.", stderr="")
            if "leanchecker" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=ax_stdout, stderr="")

        with patch.object(vmod.subprocess, "run", side_effect=_fake_subproc_run):
            res_sandbox = vmod.default_live_lean_runner(vmod.read_package_snapshot(PKG_DIR))
        self.assertIn("lake-manifest.json", observed_files)
        self.assertEqual(observed_files, sorted(vmod.BOUND_PACKAGE_FILES))
        self.assertTrue(res_sandbox["manifest_unchanged"])

        with patch.object(vmod.subprocess, "run", side_effect=subprocess.TimeoutExpired(["lake", "build"], 90)):
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEAN_TIMEOUT:"):
                vmod.default_live_lean_runner(vmod.read_package_snapshot(PKG_DIR))

        with patch.object(vmod, "default_live_lean_runner", _ok_runner):
            with tempfile.TemporaryDirectory() as td_w:
                out_cert_tmp = Path(td_w, "loopcd_test_write_cert.json")
                with patch("sys.stdout", io.StringIO()):
                    rc_w = vmod.main(["--live-lean", "--write-cert", str(out_cert_tmp)])
                self.assertEqual(rc_w, 0)
                self.assertTrue(out_cert_tmp.is_file())

                sym_out = Path(td_w, "sym_write_cert.json")
                os.symlink(str(out_cert_tmp), str(sym_out))
                err_w = io.StringIO()
                with patch("sys.stderr", err_w):
                    rc_sw = vmod.main(["--live-lean", "--write-cert", str(sym_out)])
                self.assertEqual(rc_sw, 2)
                self.assertEqual(json.loads(err_w.getvalue())["error_code"], "SYMLINK_FORBIDDEN")

        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MISSING_SANDBOX_MANIFEST:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "manifest_present": False})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MISSING_SANDBOX_MANIFEST:"):
            vmod.verify_lean_package(
                PKG_DIR, lean_runner=lambda s: {k: v for k, v in _ok_runner(s).items() if k != "manifest_present"}
            )
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SANDBOX_MANIFEST_MUTATED:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "manifest_unchanged": False})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SANDBOX_MANIFEST_MUTATED:"):
            vmod.verify_lean_package(
                PKG_DIR, lean_runner=lambda s: {k: v for k, v in _ok_runner(s).items() if k != "manifest_unchanged"}
            )
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LAKE_BUILD_FAIL:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "build_rc": 1})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LAKE_BUILD_WARNING:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "build_output": "Warning: unused"})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEANCHECKER_FAIL:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "leanchecker_rc": 1})
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^AXIOM_AUDIT_EXEC_FAIL:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "axiom_rc": 1})
        drifted_ax = list(vmod.EXPECTED_AXIOM_AUDIT_LINES)
        drifted_ax[0] = "'LoopCDStrang.evalPoly_norm' depends on axioms: [propext, Quot.sound]"
        with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^AXIOM_PIN_MISMATCH:"):
            vmod.verify_lean_package(PKG_DIR, lean_runner=lambda s: {**_ok_runner(s), "axiom_stdout": "\n".join(drifted_ax)})

        with tempfile.TemporaryDirectory() as td:
            missing_dir = Path(td, "missing_pkg")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MISSING_LEAN_DIR:"):
                vmod.verify_lean_package(missing_dir)

            cp = Path(td, "pkg")
            shutil.copytree(PKG_DIR, cp)

            sym_pkg = Path(td, "sym_pkg")
            os.symlink(str(cp), str(sym_pkg))
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SYMLINK_FORBIDDEN:"):
                vmod.verify_lean_package(sym_pkg)

            extra_sym = Path(cp, "extra_link.txt")
            os.symlink(str(Path(cp, "README.md")), str(extra_sym))
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SYMLINK_FORBIDDEN:"):
                vmod.verify_lean_package(cp)
            extra_sym.unlink()

            orig_cert = json.loads(Path(cp, "loopcd_strang_certificate.json").read_text(encoding="utf-8"))
            root_lean = Path(cp, "LoopCDStrang.lean")
            orig_root = root_lean.read_text(encoding="utf-8")
            root_lean.write_text(orig_root + "-- comment edit\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SOURCE_SHA256_MISMATCH:"):
                vmod.validate_certificate(orig_cert, pkg_dir=cp)
            root_lean.write_text(orig_root, encoding="utf-8")

            tc_p = Path(cp, "lean-toolchain")
            tc_p.unlink()
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^MISSING_BOUND_FILE:"):
                vmod.verify_lean_package(cp)
            tc_p.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^SOURCE_INVALID_UTF8:"):
                vmod.verify_lean_package(cp)
            tc_p.write_text("leanprover/lean4:v4.35.0\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^BAD_TOOLCHAIN:"):
                vmod.verify_lean_package(cp)
            tc_p.write_text("leanprover/lean4:v4.36.0\n", encoding="utf-8")

            mf_p = Path(cp, "lake-manifest.json")
            orig_mf = mf_p.read_text(encoding="utf-8")
            mf_p.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^INVALID_LAKE_MANIFEST:"):
                vmod.verify_lean_package(cp)
            mf_p.write_text('{"packages": [{"name": "mathlib"}]}', encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^NONEMPTY_LAKE_PACKAGES:"):
                vmod.verify_lean_package(cp)
            mf_p.write_text(orig_mf, encoding="utf-8")

            lk_p = Path(cp, "lakefile.toml")
            orig_lk = lk_p.read_text(encoding="utf-8")
            lk_p.write_text("[broken\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^INVALID_LAKEFILE_TOML:"):
                vmod.verify_lean_package(cp)
            lk_p.write_text(orig_lk + '\n[[require]]\nname = "mathlib"\n', encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^NONEMPTY_LAKEFILE_REQUIRE:"):
                vmod.verify_lean_package(cp)
            lk_p.write_text(orig_lk, encoding="utf-8")

            ex = Path(cp, "Extra.lean")
            ex.write_text("def x : Nat := 0\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEAN_FILE_SET_FAIL:"):
                vmod.verify_lean_package(cp)
            ex.unlink()

            for esc_snippet in (
                "#eval 1 + 1\n",
                "#reduce 1 + 1\n",
                "#synth Inhabited Nat\n",
                "#guard_msgs in theorem g : 1 = 1 := rfl\n",
                "#guard 1 == 1\n",
                "#exit\n",
                "#check_failure 1 + true\n",
                "#print Nat\n",
                "elab_rules : tactic | `(tactic| foo) => pure ()\n",
                "term_elab myElab : term => fun _ _ => pure default\n",
                "command_elab myCmd : command => fun _ => pure ()\n",
                "tactic_elab myTac : tactic => fun _ => pure ()\n",
                "declare_syntax_cat my_cat\n",
                "syntax_cat\n",
                "theorem rt : 1 = 1 := by run_tac pure ()\n",
                "simproc mySimp (_) := fun _ => pure .continue\n",
                "dsimproc myDSimp (_) := fun _ => pure .continue\n",
                "attribute [simp] Nat.add_zero\n",
                "@[init foo] def initHook : IO Unit := pure ()\n",
                "@[env_extension] def extHook : Nat := 0\n",
                "theorem bad : 1 = 1 := by bv_decide\n",
                "theorem bad : 1 = 1 := by decide +native\n",
                'def bad := m!"{sorry}"\n',
                "import«Std».Tactic.BVDecide\n",
                "@[implemented_by foo] def bar : Nat := 0\n",
                "@[csimp] theorem bad_c : 1 = 1 := rfl\n",
                "open Classical\n",
                "def x := '\"'\ntheorem cheat : 1 = 2 := by sorry\ndef y := '\"'\n",
                "run_cmd pure ()\n",
                "opaque secretVal : Nat\n",
                "set_option maxRecDepth 999999\n",
            ):
                root_lean.write_text(esc_snippet, encoding="utf-8")
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^LEAN_STATIC_ESCAPE:", msg=esc_snippet):
                    vmod.verify_lean_package(cp)

            # Live subprocess check of HIGH #1 repro: `#eval 1 + 1` in LoopCDStrang.lean MUST exit RC=2 on `--json`
            root_lean.write_text(orig_root + "\n#eval 1 + 1\n", encoding="utf-8")
            r_eval = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(cp, "verify_loopcd_strang.py")), "--json"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(r_eval.returncode, 2)
            self.assertEqual(json.loads(r_eval.stderr)["error_code"], "LEAN_STATIC_ESCAPE")
            root_lean.write_text(orig_root, encoding="utf-8")

            for bad_imp in ("import Std.Tactic.BVDecide\n", "import LoopCDStrang Std\n"):
                root_lean.write_text(bad_imp, encoding="utf-8")
                with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^FORBIDDEN_IMPORT:"):
                    vmod.verify_lean_package(cp)
            root_lean.write_text(orig_root, encoding="utf-8")

            ax_p = Path(cp, "AxiomAudit.lean")
            orig_ax_lean = ax_p.read_text(encoding="utf-8")
            ax_p.write_text(orig_ax_lean + "\n#print axioms LoopCDStrang.evalPoly_norm\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^AXIOM_AUDIT_DUP_OR_LEN:"):
                vmod.verify_lean_package(cp)

            lines_ax = [ln for ln in orig_ax_lean.splitlines() if ln.strip().startswith("#print")]
            lines_ax[0] = "#print axioms LoopCDStrang.nonexistent_thm"
            ax_p.write_text("import LoopCDStrang\n" + "\n".join(lines_ax) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^AXIOM_AUDIT_SET_MISMATCH:"):
                vmod.verify_lean_package(cp)
            ax_p.write_text(orig_ax_lean, encoding="utf-8")

            sc_p = Path(cp, "LoopCDStrang", "SphereChord.lean")
            sc_txt = sc_p.read_text(encoding="utf-8")
            sc_p.write_text(sc_txt.replace("end LoopCDStrang", "theorem extra_t : 1 = 1 := rfl\nend LoopCDStrang"), encoding="utf-8")
            ax_p.write_text(orig_ax_lean + "\n#print axioms LoopCDStrang.extra_t\n", encoding="utf-8")
            with self.assertRaisesRegex(vmod.LoopCDStrangVerifyError, r"^PUB_THM_COUNT_FAIL:"):
                vmod.verify_lean_package(cp)


if __name__ == "__main__":
    unittest.main()
