#!/usr/bin/env python3
"""Track S119: Exact Q(sqrt(5)) Spinor-Barrington 2I subset SU(2) DAG-to-HoloSphere Compiler.

Proves and constructs:
1. Exact Q(sqrt(5)) binary icosahedral group 2I ~= SL(2, 5) subset SU(2) (120 elements,
   120x120 closure, exact unit norm, 2I/{+-I} ~= A_5 quotient of order 60, and exact
   CayleySU2 roundtrip across all 75 elements with q0 >= 0, ||omega||^2 <= 1).
2. Exact SL(2, 5) = 2I Spinor-Barrington Commutator Sign Dichotomy across all 9 conjugacy
   classes via Re([a, b]) = 1 - 2 ||vec(a) x vec(b)||^2:
   - Strict Z_2 spinor obstruction on both order-5 classes (0 positive / 10 negative hits).
   - Sign-free exact spinor closure on acute order-10 class C_10^+ (q0 = (1+sqrt(5))/4,
     tau0^5 = -I, tau0^10 = +I: 10 positive / 0 negative hits, zero sign canonicalization,
     zero -I leakage, and all intermediate AND/NOT states strictly inside ||omega|| <= tan(pi/5) < 0.8).
3. Constructive O(S) DAG-to-HoloSphere (CayleySU2 + 16 B Relational V_ring) compiler collapsing
   Barrington's 4^d formula blowup to O(S) gates, verified across 6 non-trivial Boolean DAGs
   (including d=6 and d=16 shared diamond DAGs where 4^16 = 4,294,967,296) in both exact
   Q(sqrt(5)) and strict IEEE-754 FP32 arithmetic across 4 numerical regimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from typing import Any


def require_ok(cond: bool, msg: str) -> None:
    """Fail-closed invariant guard (zero bare asserts, mutation-auditable)."""
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# 1. Exact Q(sqrt(5)) Field Arithmetic (Zero Floats)
# ---------------------------------------------------------------------------

class Q5:
    """Exact element a + b*sqrt(5) in Q(sqrt(5)), zero floating-point arithmetic.

    Internally stores canonical exact integers (na, nb, den) with den > 0 and
    gcd(|na|, |nb|, den) == 1, exposing exact (Fraction, Fraction) coordinates
    via .a, .b, and .as_fractions().
    """

    __slots__ = ("_na", "_nb", "_den")

    def __init__(self, a: Fraction | int = 0, b: Fraction | int = 0) -> None:
        require_ok(
            isinstance(a, (int, Fraction)) and not isinstance(a, bool),
            "Q5 real component must be int or Fraction (not bool)",
        )
        require_ok(
            isinstance(b, (int, Fraction)) and not isinstance(b, bool),
            "Q5 sqrt5 component must be int or Fraction (not bool)",
        )
        fa = a if isinstance(a, Fraction) else Fraction(a)
        fb = b if isinstance(b, Fraction) else Fraction(b)
        da, db = fa.denominator, fb.denominator
        den = (da // math.gcd(da, db)) * db
        na = fa.numerator * (den // da)
        nb = fb.numerator * (den // db)
        g = math.gcd(math.gcd(abs(na), abs(nb)), den)
        self._na = na // g
        self._nb = nb // g
        self._den = den // g

    @classmethod
    def _from_raw(cls, na: int, nb: int, den: int) -> Q5:
        if den < 0:
            na, nb, den = -na, -nb, -den
        g = math.gcd(math.gcd(abs(na), abs(nb)), den)
        obj = cls.__new__(cls)
        obj._na = na // g
        obj._nb = nb // g
        obj._den = den // g
        return obj

    @property
    def a(self) -> Fraction:
        return Fraction(self._na, self._den)

    @property
    def b(self) -> Fraction:
        return Fraction(self._nb, self._den)

    def as_fractions(self) -> tuple[Fraction, Fraction]:
        return (self.a, self.b)

    def is_zero(self) -> bool:
        return self._na == 0 and self._nb == 0

    def sign(self) -> int:
        """Exact sign in {-1, 0, +1} via integer comparison of na^2 vs 5*nb^2."""
        na, nb = self._na, self._nb
        if na == 0 and nb == 0:
            return 0
        if na >= 0 and nb >= 0:
            return 1
        if na <= 0 and nb <= 0:
            return -1
        lhs = na * na
        rhs = 5 * nb * nb
        if na > 0:
            return 1 if lhs > rhs else -1
        return 1 if rhs > lhs else -1

    def __neg__(self) -> Q5:
        obj = Q5.__new__(Q5)
        obj._na = -self._na
        obj._nb = -self._nb
        obj._den = self._den
        return obj

    def __add__(self, other: Q5) -> Q5:
        na = self._na * other._den + other._na * self._den
        nb = self._nb * other._den + other._nb * self._den
        return Q5._from_raw(na, nb, self._den * other._den)

    def __sub__(self, other: Q5) -> Q5:
        na = self._na * other._den - other._na * self._den
        nb = self._nb * other._den - other._nb * self._den
        return Q5._from_raw(na, nb, self._den * other._den)

    def __mul__(self, other: Q5) -> Q5:
        na = self._na * other._na + 5 * self._nb * other._nb
        nb = self._na * other._nb + self._nb * other._na
        return Q5._from_raw(na, nb, self._den * other._den)

    def inv(self) -> Q5:
        norm = self._na * self._na - 5 * self._nb * self._nb
        require_ok(norm != 0, "Q5 division by zero")
        return Q5._from_raw(self._na * self._den, -self._nb * self._den, norm)

    def __truediv__(self, other: Q5) -> Q5:
        return self * other.inv()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Q5):
            return False
        return (
            self._na == other._na
            and self._nb == other._nb
            and self._den == other._den
        )

    def __hash__(self) -> int:
        return hash((self._na, self._nb, self._den))

    def sort_key(self) -> tuple[int, int, int]:
        return (self._den, self._na, self._nb)

    def to_float(self) -> float:
        try:
            val = (self._na + self._nb * math.sqrt(5.0)) / self._den
        except OverflowError:
            require_ok(False, "Q5.to_float() overflowed IEEE-754 float64")
            return 0.0
        require_ok(math.isfinite(val), "Q5.to_float() produced non-finite float64")
        return val

    def canonical_str(self) -> str:
        return f"({self.a}+{self.b}*sqrt5)"

    def __repr__(self) -> str:
        return self.canonical_str()


Q5_ZERO = Q5(0, 0)
Q5_ONE = Q5(1, 0)
Q5_TWO = Q5(2, 0)
Q5_HALF = Q5(Fraction(1, 2), 0)
Q5_PHI_HALF = Q5(Fraction(1, 4), Fraction(1, 4))        # (1 + sqrt(5)) / 4 = cos(pi/5)
Q5_INVPHI_HALF = Q5(Fraction(-1, 4), Fraction(1, 4))    # (sqrt(5) - 1) / 4 = cos(2*pi/5)

QuatQ5 = tuple[Q5, Q5, Q5, Q5]
Vec3Q5 = tuple[Q5, Q5, Q5]

Q5_ID: QuatQ5 = (Q5_ONE, Q5_ZERO, Q5_ZERO, Q5_ZERO)
Q5_NEG_ID: QuatQ5 = (-Q5_ONE, Q5_ZERO, Q5_ZERO, Q5_ZERO)


def q5_mul(p: QuatQ5, q: QuatQ5) -> QuatQ5:
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = q
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def q5_conj(p: QuatQ5) -> QuatQ5:
    return (p[0], -p[1], -p[2], -p[3])


def q5_neg(p: QuatQ5) -> QuatQ5:
    return (-p[0], -p[1], -p[2], -p[3])


def q5_norm2(p: QuatQ5) -> Q5:
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]


def q5_comm(a: QuatQ5, b: QuatQ5) -> QuatQ5:
    return q5_mul(q5_mul(a, b), q5_mul(q5_conj(a), q5_conj(b)))


def q5_vec_cross_norm2(a: QuatQ5, b: QuatQ5) -> Q5:
    cx = a[2] * b[3] - a[3] * b[2]
    cy = a[3] * b[1] - a[1] * b[3]
    cz = a[1] * b[2] - a[2] * b[1]
    return cx * cx + cy * cy + cz * cz


def q5_order(q: QuatQ5) -> int:
    require_ok(q5_norm2(q) == Q5_ONE, "Quaternion must have unit norm")
    cur = q
    found = 0
    for k in range(1, 13):
        if cur == Q5_ID:
            found = k
            break
        cur = q5_mul(cur, q)
    require_ok(found > 0, "Quaternion order exceeded finite 2I bound 12")
    return found


def cayley_forward_q5(omega: Vec3Q5) -> QuatQ5:
    w1, w2, w3 = omega
    r2 = w1 * w1 + w2 * w2 + w3 * w3
    den = Q5_ONE + r2
    return (
        (Q5_ONE - r2) / den,
        (Q5_TWO * w1) / den,
        (Q5_TWO * w2) / den,
        (Q5_TWO * w3) / den,
    )


def cayley_inverse_q5(q: QuatQ5) -> Vec3Q5:
    denom = Q5_ONE + q[0]
    require_ok(not denom.is_zero(), "CayleySU2 antipodal pole at q0 = -1")
    return (q[1] / denom, q[2] / denom, q[3] / denom)


def cayley_radius2_q5(omega: Vec3Q5) -> Q5:
    return omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]


def quat_q5_sort_key(q: QuatQ5) -> tuple[tuple[int, int, int], ...]:
    return tuple(c.sort_key() for c in q)


def quat_q5_str(q: QuatQ5) -> list[str]:
    return [c.canonical_str() for c in q]


# Canonical acute order-10 spinor-Barrington witness in C_10^+ subset 2I:
TAU0: QuatQ5 = (Q5_PHI_HALF, Q5_INVPHI_HALF, Q5_HALF, Q5_ZERO)
G_ALPHA: QuatQ5 = (Q5_PHI_HALF, -Q5_INVPHI_HALF, Q5_HALF, Q5_ZERO)
G_BETA: QuatQ5 = (Q5_HALF, -Q5_HALF, Q5_HALF, Q5_HALF)
G_ALPHA_INV: QuatQ5 = q5_conj(G_ALPHA)
G_BETA_INV: QuatQ5 = q5_conj(G_BETA)
# Corresponding order-5 element sigma0 = -tau0 (sigma0^5 == +I) illustrating Z_2 obstruction:
SIGMA0_ORD5: QuatQ5 = q5_neg(TAU0)


# ---------------------------------------------------------------------------
# 2. Exact 2I ~= SL(2, 5) Construction & Spinor Commutator Sign Theorem
# ---------------------------------------------------------------------------

def _is_even_perm(perm: tuple[int, ...]) -> bool:
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return (inv % 2) == 0


def build_2i_elements() -> tuple[QuatQ5, ...]:
    """Construct all 120 unit quaternions of the binary icosahedral group 2I subset SU(2)."""
    elems: set[QuatQ5] = set()
    # 8 axis units (+-1, +-i, +-j, +-k)
    for idx in range(4):
        for s in (Q5_ONE, -Q5_ONE):
            v = [Q5_ZERO, Q5_ZERO, Q5_ZERO, Q5_ZERO]
            v[idx] = s
            elems.add((v[0], v[1], v[2], v[3]))
    # 16 half-integer Hurwitz units (+-1/2, +-1/2, +-1/2, +-1/2)
    for signs in product((Q5_HALF, -Q5_HALF), repeat=4):
        elems.add(signs)
    # 96 golden-ratio even permutations of (0, +-1/2, +-(sqrt(5)-1)/4, +-(1+sqrt(5))/4)
    for perm in permutations(range(4)):
        if not _is_even_perm(perm):
            continue
        for s1, s2, s3 in product((1, -1), repeat=3):
            raw = (
                Q5_ZERO,
                Q5_HALF if s1 == 1 else -Q5_HALF,
                Q5_INVPHI_HALF if s2 == 1 else -Q5_INVPHI_HALF,
                Q5_PHI_HALF if s3 == 1 else -Q5_PHI_HALF,
            )
            elems.add((raw[perm[0]], raw[perm[1]], raw[perm[2]], raw[perm[3]]))
    return tuple(sorted(elems, key=quat_q5_sort_key))


def _canonical_projective(q: QuatQ5) -> QuatQ5:
    for c in q:
        s = c.sign()
        if s > 0:
            return q
        if s < 0:
            return q5_neg(q)
    return q


def verify_2i_group(elems: tuple[QuatQ5, ...]) -> dict[str, Any]:
    """Verify exact 2I group axioms, A_5 quotient, and CayleySU2 bounded-ball roundtrip."""
    elem_set = set(elems)
    unit_norm_ok = all(q5_norm2(q) == Q5_ONE for q in elems)
    inv_closed = all(q5_conj(q) in elem_set for q in elems)
    closure_pairs = 0
    closure_ok = True
    for p in elems:
        for q in elems:
            closure_pairs += 1
            if q5_mul(p, q) not in elem_set:
                closure_ok = False

    a5_proj = {_canonical_projective(q) for q in elems}

    q0_ge_0_count = 0
    q0_gt_0_count = 0
    q0_eq_0_count = 0
    cayley_ge0_roundtrip_ok = 0
    cayley_ge0_r2_le1_ok = 0
    cayley_lt0_nonpole_ok = 0
    pole_count = 0
    max_r2_ge0 = Q5_ZERO
    max_r2_acute = Q5_ZERO

    for q in elems:
        if q == Q5_NEG_ID:
            pole_count += 1
            continue
        w = cayley_inverse_q5(q)
        rec = cayley_forward_q5(w)
        r2 = cayley_radius2_q5(w)
        s0 = q[0].sign()
        if s0 >= 0:
            q0_ge_0_count += 1
            if s0 > 0:
                q0_gt_0_count += 1
                if (r2 - max_r2_acute).sign() > 0:
                    max_r2_acute = r2
            else:
                q0_eq_0_count += 1
            if rec == q:
                cayley_ge0_roundtrip_ok += 1
            if (Q5_ONE - r2).sign() >= 0:
                cayley_ge0_r2_le1_ok += 1
            if (r2 - max_r2_ge0).sign() > 0:
                max_r2_ge0 = r2
        else:
            if rec == q and (r2 - Q5_ONE).sign() > 0:
                cayley_lt0_nonpole_ok += 1

    return {
        "group_order": len(elems),
        "unit_norm_all_ok": unit_norm_ok,
        "inverse_closed": inv_closed,
        "closure_pairs_checked": closure_pairs,
        "closure_all_ok": closure_ok,
        "a5_quotient_order": len(a5_proj),
        "cayley_q0_ge_0_count": q0_ge_0_count,
        "cayley_q0_gt_0_count": q0_gt_0_count,
        "cayley_q0_eq_0_count": q0_eq_0_count,
        "cayley_ge0_roundtrip_ok": cayley_ge0_roundtrip_ok,
        "cayley_ge0_r2_le1_ok": cayley_ge0_r2_le1_ok,
        "cayley_lt0_nonpole_ok": cayley_lt0_nonpole_ok,
        "cayley_pole_count": pole_count,
        "max_r2_q0_ge_0": max_r2_ge0.canonical_str(),
        "max_r2_acute": max_r2_acute.canonical_str(),
        "max_norm_acute_float": round(math.sqrt(max_r2_acute.to_float()), 9),
    }


def verify_sl25_spinor_commutator_theorem(elems: tuple[QuatQ5, ...]) -> dict[str, Any]:
    """Exhaustively prove the SL(2,5) = 2I Spinor-Barrington Commutator Sign Dichotomy."""
    elem_set = set(elems)
    # 1. Exhaustive verification of Re([a, b]) == 1 - 2 ||vec(a) x vec(b)||^2 across 14,400 pairs
    trace_identity_mismatches = 0
    for a in elems:
        for b in elems:
            lhs = q5_comm(a, b)[0]
            rhs = Q5_ONE - Q5_TWO * q5_vec_cross_norm2(a, b)
            if lhs != rhs:
                trace_identity_mismatches += 1

    # 2. Exact partition into all 9 conjugacy classes of 2I
    remaining = set(elems)
    raw_classes: list[tuple[int, Q5, list[QuatQ5]]] = []
    while remaining:
        rep = min(remaining, key=quat_q5_sort_key)
        cls_set = {q5_mul(q5_mul(g, rep), q5_conj(g)) for g in elems}
        cls_sorted = sorted(cls_set, key=quat_q5_sort_key)
        raw_classes.append((q5_order(rep), rep[0], cls_sorted))
        remaining -= cls_set
    raw_classes.sort(key=lambda item: (item[0], item[1].to_float()))

    class_table: list[dict[str, Any]] = []
    for order_c, q0_c, cls_list in raw_classes:
        rep = cls_list[0]
        neg_rep = q5_neg(rep)
        pos_hits = 0
        neg_hits = 0
        landing_orders: set[int] = set()
        for a in cls_list:
            for b in cls_list:
                cm = q5_comm(a, b)
                landing_orders.add(q5_order(cm))
                if cm == rep:
                    pos_hits += 1
                elif cm == neg_rep:
                    neg_hits += 1
        class_table.append(
            {
                "order": order_c,
                "size": len(cls_list),
                "q0_exact": q0_c.canonical_str(),
                "q0_float": round(q0_c.to_float(), 9),
                "pos_hits": pos_hits,
                "neg_hits": neg_hits,
                "commutator_orders": sorted(landing_orders),
            }
        )

    # 3. Verify canonical acute witness (TAU0, G_ALPHA, G_BETA) and all intermediate gate states
    alpha = q5_mul(q5_mul(G_ALPHA, TAU0), G_ALPHA_INV)
    beta = q5_mul(q5_mul(G_BETA, TAU0), G_BETA_INV)
    comm_tau0 = q5_comm(alpha, beta)

    # Verify order-5 obstruction directly on sigma0 = -tau0
    alpha_5 = q5_mul(q5_mul(G_ALPHA, SIGMA0_ORD5), G_ALPHA_INV)
    beta_5 = q5_mul(q5_mul(G_BETA, SIGMA0_ORD5), G_BETA_INV)
    comm_sigma0 = q5_comm(alpha_5, beta_5)
    sigma0_pow5 = q5_mul(q5_mul(q5_mul(q5_mul(SIGMA0_ORD5, SIGMA0_ORD5), SIGMA0_ORD5), SIGMA0_ORD5), SIGMA0_ORD5)
    tau0_pow5 = q5_mul(q5_mul(q5_mul(q5_mul(TAU0, TAU0), TAU0), TAU0), TAU0)

    # Verify all 4 truth inputs for AND + 2 for NOT and track min intermediate q0 / max ||omega||
    min_interm_q0 = Q5_ONE
    max_interm_r2 = Q5_ZERO
    and_table_exact = True
    for ba, bb in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ua = TAU0 if ba else Q5_ID
        ub = TAU0 if bb else Q5_ID
        t1 = q5_mul(G_ALPHA, ua)
        aa = q5_mul(t1, G_ALPHA_INV)
        t2 = q5_mul(G_BETA, ub)
        bb_q = q5_mul(t2, G_BETA_INV)
        ab = q5_mul(aa, bb_q)
        abai = q5_mul(ab, q5_conj(aa))
        out = q5_mul(abai, q5_conj(bb_q))
        expected = TAU0 if (ba & bb) else Q5_ID
        if out != expected:
            and_table_exact = False
        for state in (ua, ub, G_ALPHA, G_BETA, t1, aa, t2, bb_q, ab, abai, out):
            if (min_interm_q0 - state[0]).sign() > 0:
                min_interm_q0 = state[0]
            r2 = cayley_radius2_q5(cayley_inverse_q5(state))
            if (r2 - max_interm_r2).sign() > 0:
                max_interm_r2 = r2

    not_table_exact = (
        q5_mul(TAU0, q5_conj(Q5_ID)) == TAU0
        and q5_mul(TAU0, q5_conj(TAU0)) == Q5_ID
    )

    return {
        "trace_identity_pairs_checked": len(elems) * len(elems),
        "trace_identity_mismatches": trace_identity_mismatches,
        "num_conjugacy_classes": len(class_table),
        "conjugacy_classes": class_table,
        "witness": {
            "tau0": quat_q5_str(TAU0),
            "tau0_in_2i": TAU0 in elem_set,
            "tau0_order": q5_order(TAU0),
            "tau0_pow5_is_neg_id": tau0_pow5 == Q5_NEG_ID,
            "tau0_cayley_norm": round(math.sqrt(cayley_radius2_q5(cayley_inverse_q5(TAU0)).to_float()), 9),
            "g_alpha": quat_q5_str(G_ALPHA),
            "g_alpha_order": q5_order(G_ALPHA),
            "g_alpha_cayley_norm": round(math.sqrt(cayley_radius2_q5(cayley_inverse_q5(G_ALPHA)).to_float()), 9),
            "g_beta": quat_q5_str(G_BETA),
            "g_beta_order": q5_order(G_BETA),
            "g_beta_cayley_norm": round(math.sqrt(cayley_radius2_q5(cayley_inverse_q5(G_BETA)).to_float()), 9),
            "comm_alpha_beta_equals_pos_tau0": comm_tau0 == TAU0,
            "sigma0_ord5_pow5_is_pos_id": sigma0_pow5 == Q5_ID,
            "order5_commutator_equals_neg_sigma0": comm_sigma0 == q5_neg(SIGMA0_ORD5),
            "and_truth_table_exact": and_table_exact,
            "not_truth_table_exact": not_table_exact,
            "min_intermediate_q0_exact": min_interm_q0.canonical_str(),
            "min_intermediate_q0_float": round(min_interm_q0.to_float(), 9),
            "max_intermediate_cayley_norm": round(math.sqrt(max_interm_r2.to_float()), 9),
        },
    }


# ---------------------------------------------------------------------------
# 3. Boolean DAG IR & Constructive O(S) HoloSphere V_ring Compiler
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    op: str     # "INPUT", "NOT", "AND"
    a: int      # input index for INPUT, wire index for NOT/AND
    b: int = -1 # second wire index for AND


@dataclass(frozen=True)
class BooleanDAG:
    name: str
    num_inputs: int
    gates: tuple[Gate, ...]
    outputs: tuple[int, ...]


class DAGBuilder:
    """Strict fan-in-2 (NOT, AND) Boolean DAG builder with derived macros."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._num_inputs = 0
        self._gates: list[Gate] = []

    def add_input(self) -> int:
        idx = len(self._gates)
        self._gates.append(Gate("INPUT", self._num_inputs))
        self._num_inputs += 1
        return idx

    def add_not(self, a: int) -> int:
        require_ok(0 <= a < len(self._gates), "Invalid wire index for NOT")
        idx = len(self._gates)
        self._gates.append(Gate("NOT", a))
        return idx

    def add_and(self, a: int, b: int) -> int:
        require_ok(
            0 <= a < len(self._gates) and 0 <= b < len(self._gates),
            "Invalid wire index for AND",
        )
        idx = len(self._gates)
        self._gates.append(Gate("AND", a, b))
        return idx

    def add_nand(self, a: int, b: int) -> int:
        return self.add_not(self.add_and(a, b))

    def add_or(self, a: int, b: int) -> int:
        return self.add_not(self.add_and(self.add_not(a), self.add_not(b)))

    def add_xor(self, a: int, b: int) -> int:
        # Shared half-adder XOR: (a | b) & ~(a & b) = 2 ANDs + 4 NOTs!
        ab = self.add_and(a, b)
        a_or_b = self.add_or(a, b)
        return self.add_and(a_or_b, self.add_not(ab))

    def add_half_adder(self, a: int, b: int) -> tuple[int, int]:
        carry = self.add_and(a, b)
        a_or_b = self.add_or(a, b)
        s = self.add_and(a_or_b, self.add_not(carry))
        return s, carry

    def add_full_adder(self, a: int, b: int, cin: int) -> tuple[int, int]:
        s1, c1 = self.add_half_adder(a, b)
        s2, c2 = self.add_half_adder(s1, cin)
        cout = self.add_or(c1, c2)
        return s2, cout

    def build(self, outputs: tuple[int, ...]) -> BooleanDAG:
        require_ok(len(outputs) > 0, "DAG must declare at least one output wire")
        require_ok(
            all(0 <= o < len(self._gates) for o in outputs),
            "Invalid output wire index",
        )
        return BooleanDAG(
            name=self._name,
            num_inputs=self._num_inputs,
            gates=tuple(self._gates),
            outputs=outputs,
        )


def eval_boolean(dag: BooleanDAG, bits: tuple[int, ...]) -> tuple[int, ...]:
    require_ok(len(bits) == dag.num_inputs, "Input bit width mismatch")
    require_ok(all(b in (0, 1) for b in bits), "Input bits must be 0 or 1")
    wires: list[int] = []
    for g in dag.gates:
        if g.op == "INPUT":
            wires.append(bits[g.a])
        elif g.op == "NOT":
            wires.append(1 - wires[g.a])
        else:
            wires.append(wires[g.a] & wires[g.b])
    return tuple(wires[o] for o in dag.outputs)


@dataclass(frozen=True)
class VringInstr:
    op: str     # "LOAD_INPUT", "NOT_GATE", "AND_GATE"
    dst: int
    a: int
    b: int = -1


@dataclass(frozen=True)
class HoloSphereProgram:
    name: str
    num_inputs: int
    vring_slots: int
    vring_bytes: int
    outputs: tuple[int, ...]
    instructions: tuple[VringInstr, ...]
    num_and: int
    num_not: int
    quat_muls_total: int
    and_depth: int
    full_depth: int
    barrington_exact_tree_len: int
    barrington_4_pow_and_depth: int
    compression_ratio_vs_barrington_exact: float


def compile_dag_to_holosphere_vring(dag: BooleanDAG) -> HoloSphereProgram:
    """Compile a fan-in-2 Boolean DAG to O(S) HoloSphere CayleySU2 + 16 B V_ring ops."""
    instrs: list[VringInstr] = []
    and_depths: list[int] = []
    full_depths: list[int] = []
    # Exact classic Barrington (1989) branching-program recurrence on unfolded formula:
    #   L(INPUT) = 1, L(NOT(a)) = L(a), L(AND(a, b)) = 2*L(a) + 2*L(b)
    barrington_lens: list[int] = []
    num_and = 0
    num_not = 0
    quat_muls = 0

    for dst, g in enumerate(dag.gates):
        if g.op == "INPUT":
            instrs.append(VringInstr("LOAD_INPUT", dst, g.a))
            and_depths.append(0)
            full_depths.append(0)
            barrington_lens.append(1)
        elif g.op == "NOT":
            instrs.append(VringInstr("NOT_GATE", dst, g.a))
            and_depths.append(and_depths[g.a])
            full_depths.append(full_depths[g.a] + 1)
            barrington_lens.append(barrington_lens[g.a])
            num_not += 1
            quat_muls += 1
        else:
            instrs.append(VringInstr("AND_GATE", dst, g.a, g.b))
            and_depths.append(max(and_depths[g.a], and_depths[g.b]) + 1)
            full_depths.append(max(full_depths[g.a], full_depths[g.b]) + 1)
            barrington_lens.append(2 * barr_len(barrington_lens, g.a) + 2 * barr_len(barrington_lens, g.b))
            num_and += 1
            quat_muls += 7  # 2 (conj A) + 2 (conj B) + 3 (A*B*A^*B^*)

    max_and_depth = max(and_depths[o] for o in dag.outputs)
    max_full_depth = max(full_depths[o] for o in dag.outputs)
    total_barrington_len = sum(barrington_lens[o] for o in dag.outputs)
    slots = len(dag.gates)
    # Compare total Barrington steps vs executed V_ring Hamilton products (min 1)
    vring_steps = max(1, quat_muls)
    return HoloSphereProgram(
        name=dag.name,
        num_inputs=dag.num_inputs,
        vring_slots=slots,
        vring_bytes=slots * 16,
        outputs=dag.outputs,
        instructions=tuple(instrs),
        num_and=num_and,
        num_not=num_not,
        quat_muls_total=quat_muls,
        and_depth=max_and_depth,
        full_depth=max_full_depth,
        barrington_exact_tree_len=total_barrington_len,
        barrington_4_pow_and_depth=4 ** max_and_depth,
        compression_ratio_vs_barrington_exact=round(total_barrington_len / vring_steps, 4),
    )


def barr_len(lens: list[int], idx: int) -> int:
    return lens[idx]


# ---------------------------------------------------------------------------
# 4. Exact Q(sqrt(5)) V_ring VM & Strict IEEE-754 FP32 CayleySU2 V_ring VM
# ---------------------------------------------------------------------------

def decode_spinor_q5(q: QuatQ5) -> int:
    require_ok(q == Q5_ID or q == TAU0, "Spinor value outside {+I, +tau0}")
    return 1 if q == TAU0 else 0


def run_q5_program(prog: HoloSphereProgram, bits: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[QuatQ5, ...]]:
    """Execute compiled HoloSphere V_ring program in exact Q(sqrt(5)) arithmetic."""
    require_ok(len(bits) == prog.num_inputs, "Input bit width mismatch")
    vring: list[QuatQ5] = [Q5_ID] * prog.vring_slots
    # Materialize static constants through exact CayleySU2 roundtrip
    tau0_c = cayley_forward_q5(cayley_inverse_q5(TAU0))
    ga_c = cayley_forward_q5(cayley_inverse_q5(G_ALPHA))
    gai_c = cayley_forward_q5(cayley_inverse_q5(G_ALPHA_INV))
    gb_c = cayley_forward_q5(cayley_inverse_q5(G_BETA))
    gbi_c = cayley_forward_q5(cayley_inverse_q5(G_BETA_INV))
    id_c = cayley_forward_q5((Q5_ZERO, Q5_ZERO, Q5_ZERO))

    for ins in prog.instructions:
        if ins.op == "LOAD_INPUT":
            vring[ins.dst] = tau0_c if bits[ins.a] else id_c
        elif ins.op == "NOT_GATE":
            vring[ins.dst] = q5_mul(tau0_c, q5_conj(vring[ins.a]))
        else:
            aa = q5_mul(q5_mul(ga_c, vring[ins.a]), gai_c)
            bb = q5_mul(q5_mul(gb_c, vring[ins.b]), gbi_c)
            vring[ins.dst] = q5_mul(q5_mul(q5_mul(aa, bb), q5_conj(aa)), q5_conj(bb))
    out_quats = tuple(vring[o] for o in prog.outputs)
    out_bits = tuple(decode_spinor_q5(q) for q in out_quats)
    return out_bits, out_quats


# Strict IEEE-754 FP32 arithmetic (every operation rounded to binary32 via struct)
QuatF32 = tuple[float, float, float, float]
Vec3F32 = tuple[float, float, float]


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f32_add(a: float, b: float) -> float:
    return _f32(a + b)


def _f32_sub(a: float, b: float) -> float:
    return _f32(a - b)


def _f32_mul(a: float, b: float) -> float:
    return _f32(a * b)


def _f32_div(a: float, b: float) -> float:
    return _f32(a / b)


def cayley_forward_f32(omega: Vec3F32) -> QuatF32:
    w1, w2, w3 = _f32(omega[0]), _f32(omega[1]), _f32(omega[2])
    r2 = _f32_add(_f32_add(_f32_mul(w1, w1), _f32_mul(w2, w2)), _f32_mul(w3, w3))
    one = _f32(1.0)
    two = _f32(2.0)
    den = _f32_add(one, r2)
    return (
        _f32_div(_f32_sub(one, r2), den),
        _f32_div(_f32_mul(two, w1), den),
        _f32_div(_f32_mul(two, w2), den),
        _f32_div(_f32_mul(two, w3), den),
    )


def cayley_reproj_f32(q: QuatF32) -> QuatF32:
    den = _f32_add(_f32(1.0), q[0])
    return cayley_forward_f32((_f32_div(q[1], den), _f32_div(q[2], den), _f32_div(q[3], den)))


def qnorm_f32(q: QuatF32) -> QuatF32:
    n2 = _f32_add(
        _f32_add(_f32_mul(q[0], q[0]), _f32_mul(q[1], q[1])),
        _f32_add(_f32_mul(q[2], q[2]), _f32_mul(q[3], q[3])),
    )
    inv_n = _f32_div(_f32(1.0), _f32(math.sqrt(n2)))
    return (_f32_mul(q[0], inv_n), _f32_mul(q[1], inv_n), _f32_mul(q[2], inv_n), _f32_mul(q[3], inv_n))


def qconj_f32(q: QuatF32) -> QuatF32:
    return (q[0], _f32(-q[1]), _f32(-q[2]), _f32(-q[3]))


def qmul_f32(p: QuatF32, q: QuatF32) -> QuatF32:
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = q
    c0 = _f32_sub(_f32_sub(_f32_sub(_f32_mul(a0, b0), _f32_mul(a1, b1)), _f32_mul(a2, b2)), _f32_mul(a3, b3))
    c1 = _f32_sub(_f32_add(_f32_add(_f32_mul(a0, b1), _f32_mul(a1, b0)), _f32_mul(a2, b3)), _f32_mul(a3, b2))
    c2 = _f32_add(_f32_add(_f32_sub(_f32_mul(a0, b2), _f32_mul(a1, b3)), _f32_mul(a2, b0)), _f32_mul(a3, b1))
    c3 = _f32_add(_f32_sub(_f32_add(_f32_mul(a0, b3), _f32_mul(a1, b2)), _f32_mul(a2, b1)), _f32_mul(a3, b0))
    return (c0, c1, c2, c3)


def _q5_to_f64(q: QuatQ5) -> QuatF32:
    return (q[0].to_float(), q[1].to_float(), q[2].to_float(), q[3].to_float())


def _w5_to_f32(q: QuatQ5) -> Vec3F32:
    w = cayley_inverse_q5(q)
    return (_f32(w[0].to_float()), _f32(w[1].to_float()), _f32(w[2].to_float()))


TAU0_F64: QuatF32 = _q5_to_f64(TAU0)
ID_F64: QuatF32 = (1.0, 0.0, 0.0, 0.0)
ID_F32: QuatF32 = cayley_forward_f32((0.0, 0.0, 0.0))
TAU0_F32: QuatF32 = cayley_forward_f32(_w5_to_f32(TAU0))
GA_F32: QuatF32 = cayley_forward_f32(_w5_to_f32(G_ALPHA))
GAI_F32: QuatF32 = cayley_forward_f32(_w5_to_f32(G_ALPHA_INV))
GB_F32: QuatF32 = cayley_forward_f32(_w5_to_f32(G_BETA))
GBI_F32: QuatF32 = cayley_forward_f32(_w5_to_f32(G_BETA_INV))
# Exact midpoint between q0(+I)=1 and q0(+tau0)=(1+sqrt(5))/4: (5+sqrt(5))/8 ~= 0.9045085
BINARY_REFRESH_THRESHOLD_F32: float = _f32(0.9045085)


def quat_l2_err(q: QuatF32, ref: QuatF32) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(q, ref)))


def so3_frobenius_err(q: QuatF32, ref: QuatF32) -> float:
    """Exact Frobenius norm ||R(q) - R(ref)||_F on SO(3) (with unit projection)."""
    def to_so3(u: QuatF32) -> tuple[float, ...]:
        n = math.sqrt(sum(x * x for x in u))
        if n == 0.0 or not math.isfinite(n):
            return ()
        w, x, y, z = (c / n for c in u)
        return (
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        )
    rq = to_so3(q)
    rr = to_so3(ref)
    if not rq or not rr:
        return math.sqrt(12.0)  # Maximal SO(3) collapse indicator when ||q|| underflows to 0
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(rq, rr)))


def decode_spinor_f32(q: QuatF32, tol: float = 0.05) -> int:
    d_id = quat_l2_err(q, ID_F64)
    d_tau = quat_l2_err(q, TAU0_F64)
    require_ok(min(d_id, d_tau) < tol, "FP32 spinor drift exceeded decode tolerance")
    return 0 if d_id < d_tau else 1


VALID_FP32_MODES = ("unnormalized", "s3_norm", "cayley_reproj", "binary_refresh")


def run_fp32_program(
    prog: HoloSphereProgram,
    bits: tuple[int, ...],
    mode: str = "cayley_reproj",
) -> tuple[tuple[QuatF32, ...], bool]:
    """Run compiled HoloSphere V_ring program in strict IEEE-754 FP32 arithmetic.

    Returns (output_quaternions, overflowed_flag).
    """
    require_ok(mode in VALID_FP32_MODES, f"Unknown FP32 execution mode: {mode}")
    require_ok(len(bits) == prog.num_inputs, "Input bit width mismatch")
    vring: list[QuatF32] = [ID_F32] * prog.vring_slots
    try:
        for ins in prog.instructions:
            if ins.op == "LOAD_INPUT":
                vring[ins.dst] = TAU0_F32 if bits[ins.a] else ID_F32
            elif ins.op == "NOT_GATE":
                res = qmul_f32(TAU0_F32, qconj_f32(vring[ins.a]))
                if mode == "s3_norm":
                    res = qnorm_f32(res)
                elif mode == "cayley_reproj":
                    res = cayley_reproj_f32(res)
                elif mode == "binary_refresh":
                    res = ID_F32 if res[0] > BINARY_REFRESH_THRESHOLD_F32 else TAU0_F32
                vring[ins.dst] = res
            else:
                aa = qmul_f32(qmul_f32(GA_F32, vring[ins.a]), GAI_F32)
                bb = qmul_f32(qmul_f32(GB_F32, vring[ins.b]), GBI_F32)
                res = qmul_f32(qmul_f32(qmul_f32(aa, bb), qconj_f32(aa)), qconj_f32(bb))
                if mode == "s3_norm":
                    res = qnorm_f32(res)
                elif mode == "cayley_reproj":
                    res = cayley_reproj_f32(res)
                elif mode == "binary_refresh":
                    res = ID_F32 if res[0] > BINARY_REFRESH_THRESHOLD_F32 else TAU0_F32
                vring[ins.dst] = res
    except OverflowError:
        return (), True
    return tuple(vring[o] for o in prog.outputs), False


# ---------------------------------------------------------------------------
# 5. Benchmark Boolean DAG Suite & Exhaustive 2^n Verification
# ---------------------------------------------------------------------------

def build_xor_chain_4() -> BooleanDAG:
    b = DAGBuilder("xor_chain_4")
    xs = [b.add_input() for _ in range(4)]
    acc = b.add_xor(xs[0], xs[1])
    acc = b.add_xor(acc, xs[2])
    acc = b.add_xor(acc, xs[3])
    return b.build((acc,))


def build_ripple_carry_adder_4bit() -> BooleanDAG:
    b = DAGBuilder("ripple_carry_adder_4bit")
    a = [b.add_input() for _ in range(4)]
    bb = [b.add_input() for _ in range(4)]
    sums: list[int] = []
    s0, c = b.add_half_adder(a[0], bb[0])
    sums.append(s0)
    for i in range(1, 4):
        si, c = b.add_full_adder(a[i], bb[i], c)
        sums.append(si)
    return b.build((sums[0], sums[1], sums[2], sums[3], c))


def build_shared_diamond_dag(depth: int) -> BooleanDAG:
    """4-wire non-degenerate shared diamond DAG of exact AND-depth d.

    Every layer applies w'_i = AND(NOT(w_i), w_{(i+1) mod 4}) using shared wires from
    layer k-1, giving exact Barrington formula length = 4^d per output wire while
    Relational V_ring stays strictly O(d) (4 AND + 4 NOT per layer).
    """
    require_ok(depth >= 1, "Diamond DAG depth must be >= 1")
    b = DAGBuilder(f"shared_diamond_d{depth}")
    layer = [b.add_input() for _ in range(4)]
    for _ in range(depth):
        nots = [b.add_not(w) for w in layer]
        layer = [b.add_and(nots[i], layer[(i + 1) % 4]) for i in range(4)]
    return b.build((layer[0],))


def build_three_sat_4var() -> BooleanDAG:
    """4-variable 5-clause 3-SAT CNF circuit."""
    b = DAGBuilder("three_sat_4var")
    x0, x1, x2, x3 = (b.add_input() for _ in range(4))
    nx0, nx1, nx2, nx3 = b.add_not(x0), b.add_not(x1), b.add_not(x2), b.add_not(x3)
    c1 = b.add_or(b.add_or(x0, nx1), x2)
    c2 = b.add_or(b.add_or(nx0, x1), x3)
    c3 = b.add_or(b.add_or(x1, nx2), nx3)
    c4 = b.add_or(b.add_or(nx0, nx2), x3)
    c5 = b.add_or(b.add_or(x0, x2), nx3)
    out = b.add_and(b.add_and(b.add_and(c1, c2), b.add_and(c3, c4)), c5)
    return b.build((out,))


def build_majority_5() -> BooleanDAG:
    """5-input majority circuit MAJ5(x0..x4) = [sum(x) >= 3] with shared subterms."""
    b = DAGBuilder("majority_5")
    x = [b.add_input() for _ in range(5)]
    # Shared pairwise AND/OR thresholds for (x0,x1,x2) and (x3,x4)
    a01 = b.add_and(x[0], x[1])
    o01 = b.add_or(x[0], x[1])
    # Exact counts on (x0, x1, x2): >=1, >=2, >=3
    ge1_012 = b.add_or(o01, x[2])
    ge2_012 = b.add_or(a01, b.add_and(o01, x[2]))
    ge3_012 = b.add_and(a01, x[2])
    # Exact counts on (x3, x4): >=1, >=2
    ge1_34 = b.add_or(x[3], x[4])
    ge2_34 = b.add_and(x[3], x[4])
    # Total >= 3 iff (>=3 in first3) | (>=2 in first3 & >=1 in last2) | (>=1 in first3 & >=2 in last2)
    out = b.add_or(ge3_012, b.add_or(b.add_and(ge2_012, ge1_34), b.add_and(ge1_012, ge2_34)))
    return b.build((out,))


def build_all_ones_lyapunov_chain(depth: int = 16) -> BooleanDAG:
    """Adversarial self-AND chain exposing exact phi^d tangential Lyapunov growth at q=tau0."""
    require_ok(depth >= 1, "Chain depth must be >= 1")
    b = DAGBuilder(f"lyapunov_self_and_d{depth}")
    w = b.add_input()
    for _ in range(depth):
        w = b.add_and(w, w)
    return b.build((w,))


def verify_single_benchmark(dag: BooleanDAG) -> dict[str, Any]:
    """Exhaustively verify a DAG over all 2^n assignments in Boolean, Q(sqrt(5)), and FP32."""
    prog = compile_dag_to_holosphere_vring(dag)
    total_assignments = 1 << dag.num_inputs
    q5_matches = 0
    fp32_stats: dict[str, dict[str, Any]] = {}
    for mode in VALID_FP32_MODES:
        fp32_stats[mode] = {
            "exact_decode_matches": 0,
            "max_quat_l2_err": 0.0,
            "max_so3_frob_err": 0.0,
            "overflow_count": 0,
        }

    ones_on_out0 = 0
    for mask in range(total_assignments):
        bits = tuple((mask >> i) & 1 for i in range(dag.num_inputs))
        truth = eval_boolean(dag, bits)
        ones_on_out0 += truth[0]
        q5_bits, _ = run_q5_program(prog, bits)
        if q5_bits == truth:
            q5_matches += 1

        for mode in VALID_FP32_MODES:
            st = fp32_stats[mode]
            out_quats, overflowed = run_fp32_program(prog, bits, mode=mode)
            if overflowed:
                st["overflow_count"] += 1
                continue
            decoded_all_ok = True
            for bit_ref, q_f32 in zip(truth, out_quats):
                ref_f64 = TAU0_F64 if bit_ref else ID_F64
                l2_e = quat_l2_err(q_f32, ref_f64)
                so3_e = so3_frobenius_err(q_f32, ref_f64)
                if l2_e > st["max_quat_l2_err"]:
                    st["max_quat_l2_err"] = l2_e
                if so3_e > st["max_so3_frob_err"]:
                    st["max_so3_frob_err"] = so3_e
                if l2_e >= 0.05 or (0 if quat_l2_err(q_f32, ID_F64) < quat_l2_err(q_f32, TAU0_F64) else 1) != bit_ref:
                    decoded_all_ok = False
            if decoded_all_ok:
                st["exact_decode_matches"] += 1

    # Round floats deterministically for canonical JSON output
    for mode in VALID_FP32_MODES:
        st = fp32_stats[mode]
        st["max_quat_l2_err"] = float(f"{st['max_quat_l2_err']:.6e}")
        st["max_so3_frob_err"] = float(f"{st['max_so3_frob_err']:.6e}")

    return {
        "name": dag.name,
        "num_inputs": dag.num_inputs,
        "num_outputs": len(dag.outputs),
        "total_assignments": total_assignments,
        "ones_on_out0": ones_on_out0,
        "vring_slots": prog.vring_slots,
        "vring_bytes": prog.vring_bytes,
        "num_and": prog.num_and,
        "num_not": prog.num_not,
        "quat_muls_total": prog.quat_muls_total,
        "and_depth": prog.and_depth,
        "full_depth": prog.full_depth,
        "barrington_exact_tree_len": prog.barrington_exact_tree_len,
        "barrington_4_pow_and_depth": prog.barrington_4_pow_and_depth,
        "compression_ratio_vs_barrington_exact": prog.compression_ratio_vs_barrington_exact,
        "q5_exact_matches": q5_matches,
        "fp32_modes": fp32_stats,
    }


def build_full_certificate() -> dict[str, Any]:
    """Run all algebraic proofs and DAG benchmarks and return a deterministic certificate."""
    elems = build_2i_elements()
    group_report = verify_2i_group(elems)
    comm_report = verify_sl25_spinor_commutator_theorem(elems)
    dags = (
        build_xor_chain_4(),
        build_ripple_carry_adder_4bit(),
        build_shared_diamond_dag(6),
        build_shared_diamond_dag(16),
        build_three_sat_4var(),
        build_majority_5(),
        build_all_ones_lyapunov_chain(16),
    )
    benchmarks = [verify_single_benchmark(d) for d in dags]
    payload: dict[str, Any] = {
        "track": "S119",
        "title": "Exact Q(sqrt(5)) Spinor-Barrington 2I subset SU(2) DAG-to-HoloSphere Compiler",
        "group_2i": group_report,
        "spinor_commutator_theorem": comm_report,
        "benchmarks": benchmarks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_certificate(cert: dict[str, Any]) -> None:
    """Fail-closed theorem & numerical envelope audit over the S119 certificate."""
    g = cert["group_2i"]
    require_ok(g["group_order"] == 120, "2I group order must be 120")
    require_ok(g["unit_norm_all_ok"] is True, "All 2I elements must have unit norm")
    require_ok(g["inverse_closed"] is True, "2I must be closed under inversion")
    require_ok(g["closure_all_ok"] is True and g["closure_pairs_checked"] == 14400, "2I 120x120 closure failed")
    require_ok(g["a5_quotient_order"] == 60, "2I/{+-I} quotient must have order 60")
    require_ok(
        g["cayley_q0_ge_0_count"] == 75
        and g["cayley_ge0_roundtrip_ok"] == 75
        and g["cayley_ge0_r2_le1_ok"] == 75
        and g["cayley_pole_count"] == 1,
        "CayleySU2 hemisphere roundtrip invariant failed",
    )

    c = cert["spinor_commutator_theorem"]
    require_ok(
        c["trace_identity_pairs_checked"] == 14400 and c["trace_identity_mismatches"] == 0,
        "Re([a,b]) == 1 - 2||vec(a) x vec(b)||^2 identity mismatch",
    )
    require_ok(c["num_conjugacy_classes"] == 9, "2I must have exactly 9 conjugacy classes")
    ord5_classes = [row for row in c["conjugacy_classes"] if row["order"] == 5]
    require_ok(
        len(ord5_classes) == 2 and all(row["pos_hits"] == 0 and row["neg_hits"] == 10 for row in ord5_classes),
        "Order-5 Z_2 spinor obstruction theorem violated",
    )
    ord10_acute = [row for row in c["conjugacy_classes"] if row["order"] == 10 and row["q0_float"] > 0.8]
    require_ok(
        len(ord10_acute) == 1 and ord10_acute[0]["pos_hits"] == 10 and ord10_acute[0]["neg_hits"] == 0,
        "Acute order-10 sign-free spinor closure theorem violated",
    )

    w = c["witness"]
    require_ok(
        w["tau0_in_2i"] is True
        and w["tau0_order"] == 10
        and w["tau0_pow5_is_neg_id"] is True
        and w["comm_alpha_beta_equals_pos_tau0"] is True
        and w["sigma0_ord5_pow5_is_pos_id"] is True
        and w["order5_commutator_equals_neg_sigma0"] is True
        and w["and_truth_table_exact"] is True
        and w["not_truth_table_exact"] is True,
        "Canonical acute spinor-Barrington witness verification failed",
    )
    require_ok(
        w["max_intermediate_cayley_norm"] < 0.8,
        "Intermediate AND/NOT CayleySU2 radius exceeded t_max = 0.8",
    )

    by_name = {row["name"]: row for row in cert["benchmarks"]}
    require_ok(len(by_name) == 7, "Expected all 7 benchmark DAGs")
    for name, row in by_name.items():
        n_tot = row["total_assignments"]
        require_ok(row["q5_exact_matches"] == n_tot, f"Exact Q(sqrt(5)) mismatch on {name}")
        for proj_mode in ("s3_norm", "cayley_reproj", "binary_refresh"):
            st = row["fp32_modes"][proj_mode]
            require_ok(
                st["exact_decode_matches"] == n_tot and st["overflow_count"] == 0,
                f"FP32 {proj_mode} decode failure on {name}",
            )
        require_ok(
            row["fp32_modes"]["cayley_reproj"]["max_quat_l2_err"] < 1e-5,
            f"FP32 cayley_reproj drift >= 1e-5 on {name}",
        )
        require_ok(
            row["fp32_modes"]["binary_refresh"]["max_quat_l2_err"] < 1e-6,
            f"FP32 binary_refresh drift >= 1e-6 on {name}",
        )

    # Specific compression & honest unnormalized-overflow checks on depth-16 diamond
    d16 = by_name["shared_diamond_d16"]
    require_ok(
        d16["barrington_exact_tree_len"] == 4 ** 16 and d16["vring_slots"] < 200,
        "Depth-16 shared diamond O(S) vs 4^16 collapse invariant failed",
    )
    require_ok(
        d16["fp32_modes"]["unnormalized"]["overflow_count"] > 0,
        "Expected unnormalized FP32 radial 4^d overflow control on shared_diamond_d16",
    )
    require_ok(
        d16["fp32_modes"]["s3_norm"]["max_quat_l2_err"] < 1e-5,
        "FP32 s3_norm drift >= 1e-5 on shared_diamond_d16",
    )


def format_summary(cert: dict[str, Any]) -> str:
    g = cert["group_2i"]
    c = cert["spinor_commutator_theorem"]
    w = c["witness"]
    lines = [
        "Track S119: Exact Q(sqrt(5)) Spinor-Barrington 2I subset SU(2) DAG-to-HoloSphere Compiler",
        f"2I Group: |2I|={g['group_order']} closure={g['closure_pairs_checked']}/{g['closure_pairs_checked']} "
        f"|2I/{{+-I}}|={g['a5_quotient_order']} Cayley(q0>=0)={g['cayley_ge0_roundtrip_ok']}/{g['cayley_q0_ge_0_count']} "
        f"poles={g['cayley_pole_count']}",
        f"SL(2,5) Commutator Identity: Re([a,b]) == 1 - 2||vec(a) x vec(b)||^2 "
        f"mismatches={c['trace_identity_mismatches']}/{c['trace_identity_pairs_checked']}",
        f"Witness C_10^+: ord(tau0)={w['tau0_order']} tau0^5=-I ||w(tau0)||={w['tau0_cayley_norm']:.6f} "
        f"||w(g_a)||={w['g_alpha_cayley_norm']:.6f} ||w(g_b)||={w['g_beta_cayley_norm']:.6f} "
        f"max_interm||w||={w['max_intermediate_cayley_norm']:.6f} < 0.8",
        "Benchmarks (all 2^n assignments):",
    ]
    for row in cert["benchmarks"]:
        s3 = row["fp32_modes"]["s3_norm"]
        cr = row["fp32_modes"]["cayley_reproj"]
        un = row["fp32_modes"]["unnormalized"]
        un_txt = f"OVERFLOW({un['overflow_count']})" if un["overflow_count"] else f"{un['max_quat_l2_err']:.2e}"
        lines.append(
            f"  {row['name']:24s} n={row['num_inputs']} d_and={row['and_depth']:2d} "
            f"V_ring={row['vring_bytes']:4d}B muls={row['quat_muls_total']:3d} "
            f"Barr={row['barrington_exact_tree_len']:11d} "
            f"ratio={row['compression_ratio_vs_barrington_exact']:12.2f}x "
            f"Q5={row['q5_exact_matches']}/{row['total_assignments']} "
            f"FP32(cayley={cr['max_quat_l2_err']:.2e}, s3={s3['max_quat_l2_err']:.2e}, raw={un_txt})"
        )
    lines.append(f"payload_sha256: {cert['payload_sha256']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S119 Spinor-Barrington 2I DAG-to-HoloSphere compiler.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate.")
    parser.add_argument("--check", action="store_true", help="Run fail-closed theorem & numerical check.")
    args = parser.parse_args(argv)

    cert = build_full_certificate()
    validate_certificate(cert)
    if args.json:
        sys.stdout.write(json.dumps(cert, sort_keys=True, indent=2) + "\n")
    elif args.check:
        sys.stdout.write(f"PASS S119 payload_sha256={cert['payload_sha256']}\n")
    else:
        sys.stdout.write(format_summary(cert) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
