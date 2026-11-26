#!/usr/bin/env python3
"""Track S120: Pure-Z/Q/Q(sqrt(5)) Cross-Spine Atlas Certifier for BRIDGE_11 & BRIDGE_12.

Proves and certifies two cross-spine exact operator bridges (zero floats, zero bare asserts,
zero RNG, ready for integration into UNIFIED_OPERATOR_ATLAS.md):

1. BRIDGE_11 (Portal 78 M3 vp-vnp Lem F.2 <-> Track S117 H_256 x bent4 & A_16 [16,11,4] SECDED):
   - Exact F_2^4 Maiorana-McFarland quadratic form q(x) = x1*x3 + x2*x4 and symplectic polar
     form B(x, y) = q(x+y) + q(x) + q(y) = x1*y3 + x3*y1 + x2*y4 + x4*y2 (mod 2).
   - Exact group isomorphism Heis_2(F_2) ~= H subset U_4(F_2) (order 32, 1024/1024 products),
     matching both the 2-qubit Pauli commutator P(x)P(y) = (-1)^B(x,y) P(y)P(x) and the M3
     gr U_4(F_2) rank-1 shear Lie bracket [sum x_i G_i, sum y_j G_j] = B(x,y) E_{0,3}.
   - Exact self-dual flat Walsh-Hadamard spectrum W_q(u) = 4*(-1)^q(u) (16/16) and bit-exact
     S117 H_256 x bent4 orthogonality with 0/2048 constant warps (vs 64/2048 raw H_256).
   - N2-candidate [16,11,4] A_16 SECDED Carry-Cover Theorem & explicit BRIDGE_03 ker(pi)=2Z^16
     obstruction: 2048 codewords, 32768/32768 SEC, 120/120 DED, quotient factorization through
     pi: Z^16 -> F_2^16, explicit 0/48 LSB-only detection vs 48/48 integer H_16 perturbation on
     upper nibble bits 1..3, and 48/48 recovery via bit-plane-rotated SECDED.

2. BRIDGE_12 (Spine I Krein Inertia OP3_F3 <-> Spine IV Non-Abelian SU(2) Holonomy):
   - Exact constructive Schur commutant proof on Sym_3(Q(sqrt(5))) for rho(2I) subset SO(3):
     rank 5/6 -> nullity 1 = span{I_3}.
   - Exact scalar blindness obstruction (G = I_3): charpoly(M_UL) == charpoly(M_LU) and
     inertia (2, 1, 0) == (2, 1, 0) across U_cyc=(0 1 2) and Track S119 2I generators
     rho(tau0), rho(g_alpha), rho(g_beta).
   - Dual-method exact inertia (Descartes real-root charpoly == Sylvester congruence LDL^T,
     zero floats) proving strict non-scalar order separation:
     * Canonical Q witness G_3 = diag(3,1,1), L = diag(3/5,6/5,3/5): U_cyc & rho(g_beta)
       jump n_-(UL)=1 -> n_-(LU)=2.
     * Universal anisotropic Q(sqrt(5)) witness G_114 = diag(1,1,4): splits ALL 4/4 rotations
       (U_cyc: 2 vs 1, rho(tau0): 1 vs 2, rho(g_alpha): 1 vs 2, rho(g_beta): 2 vs 1).
   - Exact 4th scalar block h(Delta U) = 3 - |tr(Delta U)|^2 across all 120 elements of 2I:
     h = -1 < 0 on central {+-I} (2/120), h >= (3-sqrt5)/2 > 0 on all 118/120 nontrivial
     elements (exact S119 FALSE/TRUE separator, with explicit Law-5 +-I sign-blindness bound).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from itertools import permutations, product
from typing import Any


def require_ok(cond: bool, msg: str) -> None:
    """Fail-closed invariant guard (zero bare asserts, mutation-auditable)."""
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# 1. Exact Q(sqrt(5)) Field & 3x3 Exact Linear Algebra (Zero Floats)
# ---------------------------------------------------------------------------

class Q5:
    """Exact element (na + nb*sqrt(5)) / den in Q(sqrt(5)), zero floating-point ops."""

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

    def is_zero(self) -> bool:
        return self._na == 0 and self._nb == 0

    def sign(self) -> int:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Q5):
            return False
        return self._na == other._na and self._nb == other._nb and self._den == other._den

    def __hash__(self) -> int:
        return hash((self._na, self._nb, self._den))

    def sort_key(self) -> tuple[int, int, int]:
        return (self._den, self._na, self._nb)

    def canonical_str(self) -> str:
        return f"({self.a}+{self.b}*sqrt5)"


Q5_ZERO = Q5(0, 0)
Q5_ONE = Q5(1, 0)
Q5_TWO = Q5(2, 0)
Q5_THREE = Q5(3, 0)
Q5_HALF = Q5(Fraction(1, 2), 0)
Q5_PHI_HALF = Q5(Fraction(1, 4), Fraction(1, 4))
Q5_INVPHI_HALF = Q5(Fraction(-1, 4), Fraction(1, 4))

QuatQ5 = tuple[Q5, Q5, Q5, Q5]
Mat3Q5 = tuple[tuple[Q5, Q5, Q5], tuple[Q5, Q5, Q5], tuple[Q5, Q5, Q5]]

Q5_ID: QuatQ5 = (Q5_ONE, Q5_ZERO, Q5_ZERO, Q5_ZERO)
Q5_NEG_ID: QuatQ5 = (-Q5_ONE, Q5_ZERO, Q5_ZERO, Q5_ZERO)

# Exact Track S119 acute C_10^+ spinor-Barrington generators in 2I subset SU(2):
TAU0: QuatQ5 = (Q5_PHI_HALF, Q5_INVPHI_HALF, Q5_HALF, Q5_ZERO)
G_ALPHA: QuatQ5 = (Q5_PHI_HALF, -Q5_INVPHI_HALF, Q5_HALF, Q5_ZERO)
G_BETA: QuatQ5 = (Q5_HALF, -Q5_HALF, Q5_HALF, Q5_HALF)


def _is_even_perm(perm: tuple[int, ...]) -> bool:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return (inv % 2) == 0


def build_2i_elements() -> tuple[QuatQ5, ...]:
    """Construct all 120 unit quaternions of 2I ~= SL(2, 5) subset SU(2)."""
    elems: set[QuatQ5] = set()
    for idx in range(4):
        for s in (Q5_ONE, -Q5_ONE):
            v = [Q5_ZERO, Q5_ZERO, Q5_ZERO, Q5_ZERO]
            v[idx] = s
            elems.add((v[0], v[1], v[2], v[3]))
    for signs in product((Q5_HALF, -Q5_HALF), repeat=4):
        elems.add(signs)
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
    return tuple(sorted(elems, key=lambda q: tuple(c.sort_key() for c in q)))


def rho_su2_q5(q: QuatQ5) -> Mat3Q5:
    """Exact adjoint homomorphism rho: SU(2) -> SO(3) over Q(sqrt(5))."""
    w, x, y, z = q
    norm2 = w * w + x * x + y * y + z * z
    require_ok(norm2 == Q5_ONE, "rho_su2_q5 requires unit quaternion")
    return (
        (Q5_ONE - Q5_TWO * (y * y + z * z), Q5_TWO * (x * y - w * z), Q5_TWO * (x * z + w * y)),
        (Q5_TWO * (x * y + w * z), Q5_ONE - Q5_TWO * (x * x + z * z), Q5_TWO * (y * z - w * x)),
        (Q5_TWO * (x * z - w * y), Q5_TWO * (y * z + w * x), Q5_ONE - Q5_TWO * (x * x + y * y)),
    )


def mat3_mul(a: Mat3Q5, b: Mat3Q5) -> Mat3Q5:
    rows: list[tuple[Q5, Q5, Q5]] = []
    for i in range(3):
        c0 = a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0]
        c1 = a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1]
        c2 = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2]
        rows.append((c0, c1, c2))
    return (rows[0], rows[1], rows[2])


def mat3_transpose(a: Mat3Q5) -> Mat3Q5:
    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )


def mat3_sub(a: Mat3Q5, b: Mat3Q5) -> Mat3Q5:
    return (
        (a[0][0] - b[0][0], a[0][1] - b[0][1], a[0][2] - b[0][2]),
        (a[1][0] - b[1][0], a[1][1] - b[1][1], a[1][2] - b[1][2]),
        (a[2][0] - b[2][0], a[2][1] - b[2][1], a[2][2] - b[2][2]),
    )


def mat3_det(a: Mat3Q5) -> Q5:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def mat3_diag(d0: Q5, d1: Q5, d2: Q5) -> Mat3Q5:
    return (
        (d0, Q5_ZERO, Q5_ZERO),
        (Q5_ZERO, d1, Q5_ZERO),
        (Q5_ZERO, Q5_ZERO, d2),
    )


def mat3_is_symmetric(m: Mat3Q5) -> bool:
    return m[0][1] == m[1][0] and m[0][2] == m[2][0] and m[1][2] == m[2][1]


def mat3_charpoly(m: Mat3Q5) -> tuple[Q5, Q5, Q5, Q5]:
    """Exact characteristic polynomial det(lam*I - M) = lam^3 - c1*lam^2 + c2*lam - c3."""
    c1 = m[0][0] + m[1][1] + m[2][2]
    m01 = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    m02 = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    m12 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    c2 = m01 + m02 + m12
    c3 = mat3_det(m)
    return (Q5_ONE, -c1, c2, -c3)


def _sign_changes(signs: list[int]) -> int:
    nz = [s for s in signs if s != 0]
    return sum(1 for i in range(len(nz) - 1) if nz[i] * nz[i + 1] < 0)


def inertia_descartes_q5(m: Mat3Q5) -> tuple[int, int, int]:
    """Exact symmetric 3x3 inertia (n_pos, n_neg, n_zero) via Descartes' rule on real-rooted charpoly."""
    require_ok(mat3_is_symmetric(m), "inertia_descartes_q5 requires symmetric 3x3 matrix")
    coeffs = list(mat3_charpoly(m))
    n_zero = 0
    while coeffs and coeffs[-1].is_zero():
        coeffs.pop()
        n_zero += 1
    deg = len(coeffs) - 1
    signs_pos = [c.sign() for c in coeffs]
    signs_neg = [
        (c.sign() if ((deg - i) % 2) == 0 else -c.sign())
        for i, c in enumerate(coeffs)
    ]
    n_pos = _sign_changes(signs_pos)
    n_neg = _sign_changes(signs_neg)
    return (n_pos, n_neg, n_zero)


def inertia_sylvester_q5(m: Mat3Q5) -> tuple[int, int, int]:
    """Exact symmetric 3x3 inertia (n_pos, n_neg, n_zero) via exact Sylvester congruence diagonalization."""
    require_ok(mat3_is_symmetric(m), "inertia_sylvester_q5 requires symmetric 3x3 matrix")
    a = [[m[i][j] for j in range(3)] for i in range(3)]
    diag: list[Q5] = []
    idx = 0
    while idx < 3:
        piv = -1
        for k in range(idx, 3):
            if not a[k][k].is_zero():
                piv = k
                break
        if piv == -1:
            pair: tuple[int, int] | None = None
            for r in range(idx, 3):
                for c in range(r + 1, 3):
                    if not a[r][c].is_zero():
                        pair = (r, c)
                        break
                if pair is not None:
                    break
            if pair is None:
                for _ in range(idx, 3):
                    diag.append(Q5_ZERO)
                break
            r_idx, c_idx = pair
            # Exact congruence e_r <- e_r + e_c produces diagonal 2*a[r][c] != 0
            for k in range(3):
                a[r_idx][k] = a[r_idx][k] + a[c_idx][k]
            for k in range(3):
                a[k][r_idx] = a[k][r_idx] + a[k][c_idx]
            piv = r_idx
        if piv != idx:
            a[idx], a[piv] = a[piv], a[idx]
            for k in range(3):
                a[k][idx], a[k][piv] = a[k][piv], a[k][idx]
        d_val = a[idx][idx]
        diag.append(d_val)
        inv_d = d_val.inv()
        for r in range(idx + 1, 3):
            fac = a[r][idx] * inv_d
            if not fac.is_zero():
                for c in range(idx + 1, 3):
                    a[r][c] = a[r][c] - fac * a[idx][c]
                a[r][idx] = Q5_ZERO
                a[idx][r] = Q5_ZERO
        idx += 1
    return (
        sum(1 for d in diag if d.sign() > 0),
        sum(1 for d in diag if d.sign() < 0),
        sum(1 for d in diag if d.sign() == 0),
    )


# ---------------------------------------------------------------------------
# 2. BRIDGE_11: bent4 Polar Form === M3 gr U_4(F_2) Bracket + A_16 Carry-Cover
# ---------------------------------------------------------------------------

def bent4_quad_f2(lane: int) -> int:
    """Bit-exact S117 Maiorana-McFarland quadratic form q(x) = x1*x3 ^ x2*x4 on F_2^4."""
    require_ok(
        isinstance(lane, int) and not isinstance(lane, bool) and 0 <= lane < 16,
        "bent4_quad_f2 lane must be int in 0..15",
    )
    x1 = (lane >> 0) & 1
    x2 = (lane >> 1) & 1
    x3 = (lane >> 2) & 1
    x4 = (lane >> 3) & 1
    return (x1 & x3) ^ (x2 & x4)


def bent4_polar_f2(x: int, y: int) -> int:
    """Exact symplectic polar bilinear form B(x, y) = x1*y3 + x3*y1 + x2*y4 + x4*y2 (mod 2)."""
    require_ok(
        isinstance(x, int)
        and not isinstance(x, bool)
        and isinstance(y, int)
        and not isinstance(y, bool)
        and 0 <= x < 16
        and 0 <= y < 16,
        "bent4_polar_f2 inputs must be int in 0..15",
    )
    x1, x2, x3, x4 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1
    y1, y2, y3, y4 = (y >> 0) & 1, (y >> 1) & 1, (y >> 2) & 1, (y >> 3) & 1
    return (x1 & y3) ^ (x3 & y1) ^ (x2 & y4) ^ (x4 & y2)


def _u4_f2_from_heis(x: int, s: int) -> tuple[tuple[int, ...], ...]:
    """Exact 4x4 unipotent matrix M(x, s) in U_4(F_2) realizing Heis_2(F_2) via M3 rank-1 shears."""
    x1, x2, x3, x4 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1
    return (
        (1, x1, x2, s & 1),
        (0, 1, 0, x3),
        (0, 0, 1, x4),
        (0, 0, 0, 1),
    )


def _mat4_mul_f2(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(a[i][k] & b[k][j] for k in range(4)) & 1 for j in range(4))
        for i in range(4)
    )


def _u4_inv_f2(m: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    # For M(x, s) in H subset U_4(F_2), M(x, s)^-1 = M(x, s ^ (x1*x3 ^ x2*x4))
    x1, x2, s = m[0][1], m[0][2], m[0][3]
    x3, x4 = m[1][3], m[2][3]
    return (
        (1, x1, x2, s ^ (x1 & x3) ^ (x2 & x4)),
        (0, 1, 0, x3),
        (0, 0, 1, x4),
        (0, 0, 0, 1),
    )


def _mat4_mul_z(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _pauli_4x4_z(x: int) -> tuple[tuple[int, ...], ...]:
    """Exact 4x4 integer Weyl-Pauli matrix P(x) = X1^x1 X2^x2 Z1^x3 Z2^x4 over {-1, 0, +1}."""
    x1, x2, x3, x4 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1
    rows: list[tuple[int, ...]] = []
    for r in range(4):
        b1 = (r >> 0) & 1
        b2 = (r >> 1) & 1
        c = ((b1 ^ x1) << 0) | ((b2 ^ x2) << 1)
        # Apply Z1^x3 Z2^x4 to basis state |c>, then X1^x1 X2^x2 maps |c> -> |r>
        c_b1 = (c >> 0) & 1
        c_b2 = (c >> 1) & 1
        sign = -1 if ((x3 & c_b1) ^ (x4 & c_b2)) else 1
        row = [0, 0, 0, 0]
        row[c] = sign
        rows.append(tuple(row))
    return tuple(rows)


def h256_bent4_byte_inject_s8(utf8_byte: int, warp_id: int, lane_id: int, use_bent4: bool = True) -> int:
    """Bit-exact integer mirror of S117 H_256 x bent4 byte injection (and raw H_256 baseline)."""
    d = ((warp_id & 7) << 5) | (lane_id & 31)
    db = d ^ ((lane_id & 7) << 5)
    bent4 = ((lane_id & 3) & ((lane_id >> 2) & 3)).bit_count() & 1 if use_bent4 else 0
    parity = (((utf8_byte & 0xFF) & db).bit_count() ^ bent4) & 1
    return -127 if parity else 127


def a16_secded_syndrome(w16: int) -> int:
    """Bit-exact mirror of S117 [16, 11, 4] extended Hamming A_16 SECDED syndrome."""
    w = w16 & 0xFFFF
    p0 = w.bit_count() & 1
    p1 = (w & 0xAAAA).bit_count() & 1
    p2 = (w & 0xCCCC).bit_count() & 1
    p3 = (w & 0xF0F0).bit_count() & 1
    p4 = (w & 0xFF00).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def hadamard16_z(vec: tuple[int, ...]) -> tuple[int, ...]:
    """Exact integer 16-point Walsh-Hadamard transform H_16 @ vec."""
    require_ok(len(vec) == 16, "hadamard16_z requires 16-element vector")
    out: list[int] = []
    for u in range(16):
        acc = 0
        for v in range(16):
            acc += -vec[v] if ((u & v).bit_count() & 1) else vec[v]
        out.append(acc)
    return tuple(out)


def verify_bridge_11() -> dict[str, Any]:
    """Prove BRIDGE_11: bent4 polar form === M3 gr U_4(F_2) shear bracket + A_16 carry-cover."""
    # 1. Exact S117 bit-match, polarization, alternating, and symplectic non-degeneracy
    s117_bit_matches = sum(
        1 for lane in range(16)
        if bent4_quad_f2(lane) == (((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1)
    )
    polar_matches = 0
    for x in range(16):
        for y in range(16):
            lhs = bent4_quad_f2(x ^ y) ^ bent4_quad_f2(x) ^ bent4_quad_f2(y)
            if lhs == bent4_polar_f2(x, y):
                polar_matches += 1
    alternating_ok = all(bent4_polar_f2(x, x) == 0 for x in range(16))
    radical = [x for x in range(16) if all(bent4_polar_f2(x, y) == 0 for y in range(16))]

    # 2. Exact Heis_2(F_2) ~= H subset U_4(F_2) group isomorphism & commutator == I + B(x,y)*E_{0,3}
    u4_id = _u4_f2_from_heis(0, 0)
    heis_elems = [(x, s) for x in range(16) for s in (0, 1)]
    u4_map = {(x, s): _u4_f2_from_heis(x, s) for x, s in heis_elems}
    u4_hom_matches = 0
    u4_inv_matches = 0
    u4_comm_matches = 0
    for x, s in heis_elems:
        mx = u4_map[(x, s)]
        ix = _u4_inv_f2(mx)
        if _mat4_mul_f2(mx, ix) == u4_id:
            u4_inv_matches += 1
        for y, t in heis_elems:
            my = u4_map[(y, t)]
            cocycle = (((x >> 0) & 1) & ((y >> 2) & 1)) ^ (((x >> 1) & 1) & ((y >> 3) & 1))
            expected_prod = u4_map[(x ^ y, s ^ t ^ cocycle)]
            if _mat4_mul_f2(mx, my) == expected_prod:
                u4_hom_matches += 1
            comm = _mat4_mul_f2(_mat4_mul_f2(mx, my), _mat4_mul_f2(ix, _u4_inv_f2(my)))
            expected_comm = _u4_f2_from_heis(0, bent4_polar_f2(x, y))
            if comm == expected_comm:
                u4_comm_matches += 1

    # 3. Exact gr U_4(F_2) degree-1 shear Lie bracket & 2-qubit Pauli commutator
    # Degree-1 generators G_0=E_{0,1}, G_1=E_{0,2}, G_2=E_{1,3}, G_3=E_{2,3}; central C=E_{0,3}
    def gr_elem(x: int) -> tuple[tuple[int, ...], ...]:
        x1, x2, x3, x4 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1
        return ((0, x1, x2, 0), (0, 0, 0, x3), (0, 0, 0, x4), (0, 0, 0, 0))

    gr_bracket_matches = 0
    pauli_comm_matches = 0
    paulis = [_pauli_4x4_z(x) for x in range(16)]
    for x in range(16):
        gx = gr_elem(x)
        px = paulis[x]
        for y in range(16):
            gy = gr_elem(y)
            py = paulis[y]
            ab = _mat4_mul_f2(gx, gy)
            ba = _mat4_mul_f2(gy, gx)
            bracket = tuple(tuple(ab[i][j] ^ ba[i][j] for j in range(4)) for i in range(4))
            expected_b = ((0, 0, 0, bent4_polar_f2(x, y)), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
            if bracket == expected_b:
                gr_bracket_matches += 1
            p_ab = _mat4_mul_z(px, py)
            p_ba = _mat4_mul_z(py, px)
            sgn = -1 if bent4_polar_f2(x, y) else 1
            scaled = tuple(tuple(sgn * p_ba[i][j] for j in range(4)) for i in range(4))
            if p_ab == scaled:
                pauli_comm_matches += 1

    # 4. Exact self-dual Walsh-Hadamard flat spectrum W_q(u) == 4 * (-1)^q(u)
    wht_vals = []
    self_dual_matches = 0
    for u in range(16):
        acc = 0
        for x in range(16):
            exp = bent4_quad_f2(x) ^ ((u & x).bit_count() & 1)
            acc += -1 if exp else 1
        wht_vals.append(acc)
        if acc == (-4 if bent4_quad_f2(u) else 4):
            self_dual_matches += 1
    wht_all_abs4 = all(abs(v) == 4 for v in wht_vals)

    # 5. Bit-exact S117 H_256 x bent4 orthogonality & 0/2048 constant warps vs raw H_256
    vecs_bent = [[h256_bent4_byte_inject_s8(b, d >> 5, d & 31, True) for d in range(256)] for b in range(256)]
    diag_expected = 256 * 127 * 127
    diag_ok = sum(1 for b in range(256) if sum(v * v for v in vecs_bent[b]) == diag_expected)
    off_diag_zero = 0
    off_diag_total = 0
    for b1 in range(256):
        v1 = vecs_bent[b1]
        for b2 in range(b1 + 1, 256):
            off_diag_total += 1
            if sum(a * c for a, c in zip(v1, vecs_bent[b2])) == 0:
                off_diag_zero += 1

    const_warps_bent = 0
    max_abs_warp_bent = 0
    const_warps_raw = 0
    max_abs_warp_raw = 0
    for b in range(256):
        for w in range(8):
            s_b = sum(1 if h256_bent4_byte_inject_s8(b, w, ln, True) > 0 else -1 for ln in range(32))
            s_r = sum(1 if h256_bent4_byte_inject_s8(b, w, ln, False) > 0 else -1 for ln in range(32))
            if abs(s_b) > max_abs_warp_bent:
                max_abs_warp_bent = abs(s_b)
            if abs(s_b) == 32:
                const_warps_bent += 1
            if abs(s_r) > max_abs_warp_raw:
                max_abs_warp_raw = abs(s_r)
            if abs(s_r) == 32:
                const_warps_raw += 1

    # 6. [16, 11, 4] A_16 SECDED exhaustive verification + Carry-Cover & BRIDGE_03 ker(pi)=2Z^16 obstruction
    codewords = [w for w in range(65536) if a16_secded_syndrome(w) == 0]
    sec_exact = 0
    for cw in codewords:
        for j in range(16):
            syn = a16_secded_syndrome(cw ^ (1 << j))
            if (syn & 1) == 1 and ((syn >> 1) & 0xF) == j:
                sec_exact += 1
    ded_weight2_exact = 0
    for i in range(16):
        for j in range(i + 1, 16):
            syn = a16_secded_syndrome((1 << i) ^ (1 << j))
            if syn != 0 and (syn & 1) == 0:
                ded_weight2_exact += 1
    # Verify exact F_2 syndrome linearity on 2048 codewords x 16 unit basis -> DED extends to all 2048*120!
    linearity_exact = sum(
        1 for cw in codewords for j in range(16)
        if a16_secded_syndrome(cw ^ (1 << j)) == a16_secded_syndrome(1 << j)
    )

    # Carry-cover factorization of bent4 mod 2: (-1)^q(x + 2y) == (-1)^q(x)
    bent4_mod2_lift_matches = 0
    for x in range(16):
        qx = bent4_quad_f2(x)
        x_bits = [(x >> k) & 1 for k in range(4)]
        for y in range(16):
            z = [x_bits[k] + 2 * ((y >> k) & 1) for k in range(4)]
            qz_mod2 = (z[0] * z[2] + z[1] * z[3]) & 1
            if qz_mod2 == qx:
                bent4_mod2_lift_matches += 1

    # Concrete signed int4 A_16-valid codeword z in {-7..+7}^16 (LSB bit-vector = codeword 0x0000 -> all even!)
    # Use a non-trivial valid codeword 0x000F (lanes 0..3 have LSB=1, syndrome=0) lifted to signed integers:
    base_cw = 0x000F  # Check: popcount=4 (p0=0), 0x000F&0xAAAA=0xA (p1=0), &0xCCCC=0xC (p2=0), &0xF0F0=0, &0xFF00=0
    base_int4 = tuple(((j % 3) * 2) + ((base_cw >> j) & 1) for j in range(16))
    base_lsb_word = sum(((base_int4[j] & 1) << j) for j in range(16))
    base_h16 = hadamard16_z(base_int4)

    # Test ALL 48 upper-nibble faults delta = 2^b * e_j (b in {1,2,3}, j in 0..15) in ker(pi) = 2Z^16:
    lsb_only_detected = 0
    h16_energy_shifted = 0
    h16_parseval_exact = 0
    bitplane_rotated_detected = 0
    for b_plane in (1, 2, 3):
        # Construct codeword whose bit-plane b_plane is ALSO a valid A_16 codeword (plane=0x0000 -> clean baseline)
        clean_plane_vec = tuple((base_int4[j] & 1) for j in range(16))
        clean_h16 = hadamard16_z(clean_plane_vec)
        step = 1 << b_plane
        for j in range(16):
            pert = list(clean_plane_vec)
            pert[j] += step  # perturbation in ker(pi) = 2*Z^16!
            pert_t = tuple(pert)
            # 1. LSB-only SECDED sees zero syndrome!
            lsb_w = sum(((pert_t[k] & 1) << k) for k in range(16))
            if a16_secded_syndrome(lsb_w) != 0:
                lsb_only_detected += 1
            # 2. Integer H_16 sees exact Parseval shift ||H_16(z+delta) - H_16(z)||^2 == 16 * 4^b
            pert_h16 = hadamard16_z(pert_t)
            diff_sq = sum((a - c) * (a - c) for a, c in zip(pert_h16, clean_h16))
            if diff_sq > 0:
                h16_energy_shifted += 1
            if diff_sq == 16 * step * step:
                h16_parseval_exact += 1
            # 3. Bit-plane-rotated SECDED (pi_b(z) = (z >> b) & 1) detects & locates exact lane j!
            plane_w = sum((((pert_t[k] >> b_plane) & 1) << k) for k in range(16))
            syn_b = a16_secded_syndrome(plane_w)
            if (syn_b & 1) == 1 and ((syn_b >> 1) & 0xF) == j:
                bitplane_rotated_detected += 1

    # Verify exact Frobenius characteristic-2 vs integer cross-term seam (BRIDGE_03):
    # For z = base_int4, k = e_0: (z_0 + 2)^2 - (z_0^2 + 4) == 4*z_0 != 0 in Z, while == 0 mod 2!
    frob_cross_z = (base_int4[0] + 2) ** 2 - (base_int4[0] ** 2 + 4)
    frob_cross_mod2 = frob_cross_z & 1

    return {
        "bridge_id": "BRIDGE_11_BENT4_POLAR_M3_SHEAR_AND_A16_SECDED_CARRY_COVER",
        "tracks": ["Portal_78_M3", "S22c", "S40", "S117", "S118"],
        "bent4_symplectic": {
            "s117_bit_matches": s117_bit_matches,
            "polar_identity_matches": polar_matches,
            "alternating_ok": alternating_ok,
            "symplectic_radical_size": len(radical),
            "u4_group_order": len(u4_map),
            "u4_unique_matrices": len(set(u4_map.values())),
            "u4_homomorphism_matches": u4_hom_matches,
            "u4_inverse_matches": u4_inv_matches,
            "u4_commutator_matches": u4_comm_matches,
            "gr_u4_lie_bracket_matches": gr_bracket_matches,
            "pauli_4x4_commutator_matches": pauli_comm_matches,
            "wht_all_abs4": wht_all_abs4,
            "wht_self_dual_matches": self_dual_matches,
        },
        "h256_bent4": {
            "diag_norm_matches": diag_ok,
            "off_diag_zero_pairs": off_diag_zero,
            "off_diag_total_pairs": off_diag_total,
            "const_warps_bent4": const_warps_bent,
            "max_abs_warp_sum_bent4": max_abs_warp_bent,
            "const_warps_raw_h256": const_warps_raw,
            "max_abs_warp_sum_raw_h256": max_abs_warp_raw,
            "total_byte_warps": 2048,
        },
        "a16_secded_carry_cover": {
            "valid_codewords": len(codewords),
            "sec_exhaustive_matches": sec_exact,
            "ded_weight2_pairs": ded_weight2_exact,
            "f2_syndrome_linearity_matches": linearity_exact,
            "bent4_mod2_lift_matches": bent4_mod2_lift_matches,
            "base_codeword_hex": f"0x{base_lsb_word:04x}",
            "base_h16_l2_sq": sum(v * v for v in base_h16),
            "upper_nibble_faults_tested": 48,
            "lsb_only_secded_detected": lsb_only_detected,
            "integer_h16_shifted": h16_energy_shifted,
            "integer_h16_parseval_exact": h16_parseval_exact,
            "bitplane_rotated_secded_detected": bitplane_rotated_detected,
            "bridge03_cross_term_z": frob_cross_z,
            "bridge03_cross_term_mod2": frob_cross_mod2,
        },
    }


# ---------------------------------------------------------------------------
# 3. BRIDGE_12: Non-Scalar Krein-Holonomy Phase Detector K_t (Spine I <-> IV)
# ---------------------------------------------------------------------------

def compute_schur_commutant_nullity_q5(gens: tuple[Mat3Q5, ...]) -> tuple[int, int]:
    """Exact Gaussian elimination over Q(sqrt(5)) for U^T G U - G = 0 on Sym_3 (dim 6)."""
    basis: list[Mat3Q5] = []
    for i in range(3):
        m = [[Q5_ZERO] * 3 for _ in range(3)]
        m[i][i] = Q5_ONE
        basis.append(((m[0][0], m[0][1], m[0][2]), (m[1][0], m[1][1], m[1][2]), (m[2][0], m[2][1], m[2][2])))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = [[Q5_ZERO] * 3 for _ in range(3)]
        m[i][j] = Q5_ONE
        m[j][i] = Q5_ONE
        basis.append(((m[0][0], m[0][1], m[0][2]), (m[1][0], m[1][1], m[1][2]), (m[2][0], m[2][1], m[2][2])))

    rows: list[list[Q5]] = []
    for u in gens:
        ut = mat3_transpose(u)
        diffs = [mat3_sub(mat3_mul(ut, mat3_mul(e, u)), e) for e in basis]
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            rows.append([diffs[k][i][j] for k in range(6)])

    a = [list(r) for r in rows]
    rank = 0
    for col in range(6):
        piv = -1
        for r in range(rank, len(a)):
            if not a[r][col].is_zero():
                piv = r
                break
        if piv == -1:
            continue
        a[rank], a[piv] = a[piv], a[rank]
        inv_p = a[rank][col].inv()
        a[rank] = [x * inv_p for x in a[rank]]
        for r in range(len(a)):
            if r != rank and not a[r][col].is_zero():
                fac = a[r][col]
                a[r] = [a[r][c] - fac * a[rank][c] for c in range(6)]
        rank += 1
    return rank, 6 - rank


def krein_defect_3x3(g: Mat3Q5, op: Mat3Q5) -> Mat3Q5:
    """Exact 3x3 Krein defect block G - op^T G op over Q(sqrt(5))."""
    return mat3_sub(g, mat3_mul(mat3_transpose(op), mat3_mul(g, op)))


def holonomy_scalar_block_q5(delta_u: QuatQ5) -> Q5:
    """Exact 4th scalar block h(Delta U) = 3 - |tr(Delta U)|^2 = 3 - 4*q0^2 over Q(sqrt(5))."""
    require_ok(
        isinstance(delta_u, (tuple, list)) and len(delta_u) == 4 and all(isinstance(c, Q5) for c in delta_u),
        "holonomy_scalar_block_q5 requires a 4-element QuatQ5",
    )
    tr = Q5_TWO * delta_u[0]
    return Q5_THREE - tr * tr


def unified_kt_4x4_inertia(
    g: Mat3Q5,
    op: Mat3Q5,
    delta_u: QuatQ5,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Compute exact 4x4 inertia of K_t = diag(G - op^T G op, 3 - |tr(Delta U)|^2) via both methods."""
    m3 = krein_defect_3x3(g, op)
    d3 = inertia_descartes_q5(m3)
    s3 = inertia_sylvester_q5(m3)
    h4 = holonomy_scalar_block_q5(delta_u).sign()
    dp = (d3[0] + (1 if h4 > 0 else 0), d3[1] + (1 if h4 < 0 else 0), d3[2] + (1 if h4 == 0 else 0))
    sp = (s3[0] + (1 if h4 > 0 else 0), s3[1] + (1 if h4 < 0 else 0), s3[2] + (1 if h4 == 0 else 0))
    return dp, sp


def verify_bridge_12() -> dict[str, Any]:
    """Prove BRIDGE_12: Non-scalar Krein-Holonomy Phase Detector K_t over Q and Q(sqrt(5))."""
    id3 = mat3_diag(Q5_ONE, Q5_ONE, Q5_ONE)
    g_311 = mat3_diag(Q5_THREE, Q5_ONE, Q5_ONE)
    g_114 = mat3_diag(Q5_ONE, Q5_ONE, Q5(4))
    l_diag = mat3_diag(Q5(Fraction(3, 5)), Q5(Fraction(6, 5)), Q5(Fraction(3, 5)))
    u_cyc: Mat3Q5 = (
        (Q5_ZERO, Q5_ONE, Q5_ZERO),
        (Q5_ZERO, Q5_ZERO, Q5_ONE),
        (Q5_ONE, Q5_ZERO, Q5_ZERO),
    )

    rotations: tuple[tuple[str, Mat3Q5], ...] = (
        ("U_cyc_012", u_cyc),
        ("rho_tau0", rho_su2_q5(TAU0)),
        ("rho_g_alpha", rho_su2_q5(G_ALPHA)),
        ("rho_g_beta", rho_su2_q5(G_BETA)),
    )

    # 1. Exact constructive Schur commutant proof for rho(2I) on Sym_3(Q(sqrt(5)))
    schur_rank, schur_nullity = compute_schur_commutant_nullity_q5(
        (rho_su2_q5(TAU0), rho_su2_q5(G_BETA))
    )

    # 2. Evaluate scalar G = I_3, canonical G_3 = diag(3,1,1), and universal G_114 = diag(1,1,4)
    rotation_results: list[dict[str, Any]] = []
    for name, u in rotations:
        is_so3 = (mat3_mul(mat3_transpose(u), u) == id3) and (mat3_det(u) == Q5_ONE)
        ul = mat3_mul(u, l_diag)
        lu = mat3_mul(l_diag, u)

        # Scalar blindness G = I_3
        m_ul_i = krein_defect_3x3(id3, ul)
        m_lu_i = krein_defect_3x3(id3, lu)
        cp_equal_i = mat3_charpoly(m_ul_i) == mat3_charpoly(m_lu_i)
        d_ul_i, s_ul_i = inertia_descartes_q5(m_ul_i), inertia_sylvester_q5(m_ul_i)
        d_lu_i, s_lu_i = inertia_descartes_q5(m_lu_i), inertia_sylvester_q5(m_lu_i)

        # Canonical G_3 = diag(3, 1, 1)
        m_ul_311 = krein_defect_3x3(g_311, ul)
        m_lu_311 = krein_defect_3x3(g_311, lu)
        d_ul_311, s_ul_311 = inertia_descartes_q5(m_ul_311), inertia_sylvester_q5(m_ul_311)
        d_lu_311, s_lu_311 = inertia_descartes_q5(m_lu_311), inertia_sylvester_q5(m_lu_311)

        # Universal anisotropic G_114 = diag(1, 1, 4)
        m_ul_114 = krein_defect_3x3(g_114, ul)
        m_lu_114 = krein_defect_3x3(g_114, lu)
        d_ul_114, s_ul_114 = inertia_descartes_q5(m_ul_114), inertia_sylvester_q5(m_ul_114)
        d_lu_114, s_lu_114 = inertia_descartes_q5(m_lu_114), inertia_sylvester_q5(m_lu_114)

        dual_method_all_equal = (
            d_ul_i == s_ul_i
            and d_lu_i == s_lu_i
            and d_ul_311 == s_ul_311
            and d_lu_311 == s_lu_311
            and d_ul_114 == s_ul_114
            and d_lu_114 == s_lu_114
        )

        rotation_results.append(
            {
                "name": name,
                "is_exact_so3": is_so3,
                "dual_method_all_equal": dual_method_all_equal,
                "scalar_I3_charpoly_identical": cp_equal_i,
                "scalar_I3_inertia_UL": list(d_ul_i),
                "scalar_I3_inertia_LU": list(d_lu_i),
                "G311_inertia_UL": list(d_ul_311),
                "G311_inertia_LU": list(d_lu_311),
                "G311_n_minus_split": d_ul_311[1] != d_lu_311[1],
                "G114_inertia_UL": list(d_ul_114),
                "G114_inertia_LU": list(d_lu_114),
                "G114_n_minus_split": d_ul_114[1] != d_lu_114[1],
            }
        )

    # 3. Exhaustive 4th scalar holonomy block h(Delta U) = 3 - |tr(Delta U)|^2 across all 120 elements of 2I
    elems_2i = build_2i_elements()
    h_neg_count = 0
    h_zero_count = 0
    h_pos_count = 0
    h_sign_blind_count = 0
    min_pos_h = Q5(100)
    for q in elems_2i:
        hq = holonomy_scalar_block_q5(q)
        neg_q: QuatQ5 = (-q[0], -q[1], -q[2], -q[3])
        if hq == holonomy_scalar_block_q5(neg_q):
            h_sign_blind_count += 1
        s = hq.sign()
        if s < 0:
            h_neg_count += 1
        elif s > 0:
            h_pos_count += 1
            if (min_pos_h - hq).sign() > 0:
                min_pos_h = hq
        else:
            h_zero_count += 1

    # 4. Full 4x4 unified K_t detector combining G_114 + rho(tau0) with S119 FALSE (+I) vs TRUE (+tau0)
    u_tau0 = rho_su2_q5(TAU0)
    kt_ul_false_d, kt_ul_false_s = unified_kt_4x4_inertia(g_114, mat3_mul(u_tau0, l_diag), Q5_ID)
    kt_lu_false_d, kt_lu_false_s = unified_kt_4x4_inertia(g_114, mat3_mul(l_diag, u_tau0), Q5_ID)
    kt_ul_true_d, kt_ul_true_s = unified_kt_4x4_inertia(g_114, mat3_mul(u_tau0, l_diag), TAU0)
    kt_lu_true_d, kt_lu_true_s = unified_kt_4x4_inertia(g_114, mat3_mul(l_diag, u_tau0), TAU0)

    return {
        "bridge_id": "BRIDGE_12_NONSCALAR_KREIN_HOLONOMY_PHASE_DETECTOR",
        "tracks": ["S24", "S37", "S115", "S116", "S119", "OP3_F3"],
        "schur_commutant_sym3": {
            "dimension_sym3": 6,
            "rank": schur_rank,
            "nullity": schur_nullity,
            "unique_invariant_metric": "span{I_3}",
        },
        "rotations": rotation_results,
        "holonomy_block_2i": {
            "group_order": len(elems_2i),
            "central_neg_count": h_neg_count,
            "zero_count": h_zero_count,
            "nontrivial_pos_count": h_pos_count,
            "pm_id_sign_blind_count": h_sign_blind_count,
            "h_plus_id_exact": holonomy_scalar_block_q5(Q5_ID).canonical_str(),
            "h_tau0_exact": holonomy_scalar_block_q5(TAU0).canonical_str(),
            "min_pos_h_exact": min_pos_h.canonical_str(),
        },
        "unified_kt_4x4_tau0_G114": {
            "dual_method_all_equal": (
                kt_ul_false_d == kt_ul_false_s
                and kt_lu_false_d == kt_lu_false_s
                and kt_ul_true_d == kt_ul_true_s
                and kt_lu_true_d == kt_lu_true_s
            ),
            "UL_false_I": list(kt_ul_false_d),
            "LU_false_I": list(kt_lu_false_d),
            "UL_true_tau0": list(kt_ul_true_d),
            "LU_true_tau0": list(kt_lu_true_d),
        },
    }


# ---------------------------------------------------------------------------
# 4. Certificate Assembly, Fail-Closed Validator & Deterministic CLI
# ---------------------------------------------------------------------------

def build_full_certificate() -> dict[str, Any]:
    b11 = verify_bridge_11()
    b12 = verify_bridge_12()
    payload: dict[str, Any] = {
        "schema_version": "holosphere.bridges.s120_certificate.v1",
        "track": "S120",
        "title": "Exact Pure-Z/Q/Q(sqrt(5)) Cross-Spine Atlas Bridges BRIDGE_11 & BRIDGE_12",
        "bridge_11": b11,
        "bridge_12": b12,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_certificate(cert: dict[str, Any]) -> None:
    """Fail-closed theorem audit over BRIDGE_11 and BRIDGE_12."""
    b11 = cert["bridge_11"]
    bs = b11["bent4_symplectic"]
    require_ok(
        bs["s117_bit_matches"] == 16
        and bs["polar_identity_matches"] == 256
        and bs["alternating_ok"] is True
        and bs["symplectic_radical_size"] == 1,
        "BRIDGE_11 bent4 symplectic polar form invariant failed",
    )
    require_ok(
        bs["u4_group_order"] == 32
        and bs["u4_unique_matrices"] == 32
        and bs["u4_homomorphism_matches"] == 1024
        and bs["u4_inverse_matches"] == 32
        and bs["u4_commutator_matches"] == 1024,
        "BRIDGE_11 Heis_2(F_2) ~= H subset U_4(F_2) isomorphism invariant failed",
    )
    require_ok(
        bs["gr_u4_lie_bracket_matches"] == 256
        and bs["pauli_4x4_commutator_matches"] == 256
        and bs["wht_all_abs4"] is True
        and bs["wht_self_dual_matches"] == 16,
        "BRIDGE_11 M3 gr U_4(F_2) / Pauli / self-dual WHT invariant failed",
    )

    h256 = b11["h256_bent4"]
    require_ok(
        h256["diag_norm_matches"] == 256
        and h256["off_diag_zero_pairs"] == 32640
        and h256["const_warps_bent4"] == 0
        and h256["max_abs_warp_sum_bent4"] == 8
        and h256["const_warps_raw_h256"] == 64
        and h256["max_abs_warp_sum_raw_h256"] == 32,
        "BRIDGE_11 H_256 x bent4 orthogonality / constant-warp elimination failed",
    )

    cc = b11["a16_secded_carry_cover"]
    require_ok(
        cc["valid_codewords"] == 2048
        and cc["sec_exhaustive_matches"] == 32768
        and cc["ded_weight2_pairs"] == 120
        and cc["f2_syndrome_linearity_matches"] == 32768
        and cc["bent4_mod2_lift_matches"] == 256,
        "BRIDGE_11 A_16 [16,11,4] SECDED quotient factorization failed",
    )
    require_ok(
        cc["upper_nibble_faults_tested"] == 48
        and cc["lsb_only_secded_detected"] == 0
        and cc["integer_h16_shifted"] == 48
        and cc["integer_h16_parseval_exact"] == 48
        and cc["bitplane_rotated_secded_detected"] == 48
        and cc["bridge03_cross_term_z"] != 0
        and cc["bridge03_cross_term_mod2"] == 0,
        "BRIDGE_11 ker(pi)=2Z^16 carry-cover & BRIDGE_03 obstruction check failed",
    )

    b12 = cert["bridge_12"]
    sc = b12["schur_commutant_sym3"]
    require_ok(
        sc["rank"] == 5 and sc["nullity"] == 1,
        "BRIDGE_12 Schur commutant nullity != 1",
    )

    rots = {r["name"]: r for r in b12["rotations"]}
    require_ok(len(rots) == 4, "BRIDGE_12 expected 4 SO(3) rotation witnesses")
    for name, r in rots.items():
        require_ok(
            r["is_exact_so3"] is True
            and r["dual_method_all_equal"] is True
            and r["scalar_I3_charpoly_identical"] is True
            and r["scalar_I3_inertia_UL"] == [2, 1, 0]
            and r["scalar_I3_inertia_LU"] == [2, 1, 0]
            and r["G114_n_minus_split"] is True,
            f"BRIDGE_12 rotation invariant failed for {name}",
        )

    require_ok(
        rots["U_cyc_012"]["G311_inertia_UL"] == [2, 1, 0]
        and rots["U_cyc_012"]["G311_inertia_LU"] == [1, 2, 0]
        and rots["rho_g_beta"]["G311_inertia_UL"] == [2, 1, 0]
        and rots["rho_g_beta"]["G311_inertia_LU"] == [1, 2, 0],
        "BRIDGE_12 canonical G_3 = diag(3,1,1) order-split (1 -> 2) invariant failed",
    )

    hb = b12["holonomy_block_2i"]
    require_ok(
        hb["group_order"] == 120
        and hb["central_neg_count"] == 2
        and hb["zero_count"] == 0
        and hb["nontrivial_pos_count"] == 118
        and hb["pm_id_sign_blind_count"] == 120
        and hb["h_plus_id_exact"] == "(-1+0*sqrt5)"
        and hb["h_tau0_exact"] == "(3/2+-1/2*sqrt5)"
        and hb["min_pos_h_exact"] == "(3/2+-1/2*sqrt5)",
        "BRIDGE_12 2I 4th scalar holonomy block invariant failed",
    )

    kt = b12["unified_kt_4x4_tau0_G114"]
    require_ok(
        kt["dual_method_all_equal"] is True
        and kt["UL_false_I"] == [2, 2, 0]
        and kt["LU_false_I"] == [1, 3, 0]
        and kt["UL_true_tau0"] == [3, 1, 0]
        and kt["LU_true_tau0"] == [2, 2, 0],
        "BRIDGE_12 unified 4x4 K_t inertia invariant failed",
    )


def format_summary(cert: dict[str, Any]) -> str:
    b11 = cert["bridge_11"]
    bs = b11["bent4_symplectic"]
    h256 = b11["h256_bent4"]
    cc = b11["a16_secded_carry_cover"]
    b12 = cert["bridge_12"]
    sc = b12["schur_commutant_sym3"]
    hb = b12["holonomy_block_2i"]
    kt = b12["unified_kt_4x4_tau0_G114"]
    lines = [
        "Track S120: Pure-Z/Q/Q(sqrt(5)) Cross-Spine Atlas Bridges BRIDGE_11 & BRIDGE_12",
        f"BRIDGE_11 bent4/M3: polar={bs['polar_identity_matches']}/256 rad={bs['symplectic_radical_size']} "
        f"|Heis_2~=U_4|={bs['u4_unique_matrices']} hom={bs['u4_homomorphism_matches']}/1024 "
        f"comm={bs['u4_commutator_matches']}/1024 grU4={bs['gr_u4_lie_bracket_matches']}/256 "
        f"Pauli={bs['pauli_4x4_commutator_matches']}/256 WHT_self_dual={bs['wht_self_dual_matches']}/16",
        f"BRIDGE_11 H256/A16: orth={h256['off_diag_zero_pairs']}/{h256['off_diag_total_pairs']} "
        f"const_warps={h256['const_warps_bent4']}/2048 (raw={h256['const_warps_raw_h256']}/2048) "
        f"A16={cc['valid_codewords']}cw SEC={cc['sec_exhaustive_matches']}/32768 DED={cc['ded_weight2_pairs']}/120 "
        f"ker_pi_upper48(LSB={cc['lsb_only_secded_detected']}/48, H16_shift={cc['integer_h16_shifted']}/48, "
        f"rot_SECDED={cc['bitplane_rotated_secded_detected']}/48)",
        f"BRIDGE_12 Schur/K_t: Sym3_comm_rank={sc['rank']}/6 nullity={sc['nullity']} "
        f"2I_h_block(neg={hb['central_neg_count']}, pos={hb['nontrivial_pos_count']}, min_pos={hb['min_pos_h_exact']})",
    ]
    for r in b12["rotations"]:
        lines.append(
            f"  {r['name']:12s} I3: UL={r['scalar_I3_inertia_UL']} LU={r['scalar_I3_inertia_LU']} | "
            f"G311: UL={r['G311_inertia_UL']} LU={r['G311_inertia_LU']} | "
            f"G114: UL={r['G114_inertia_UL']} LU={r['G114_inertia_LU']}"
        )
    lines.append(
        f"  K_t_4x4(tau0,G114): UL_F={kt['UL_false_I']} LU_F={kt['LU_false_I']} "
        f"UL_T={kt['UL_true_tau0']} LU_T={kt['LU_true_tau0']}"
    )
    lines.append(f"payload_sha256: {cert['payload_sha256']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S120 Pure-Z/Q/Q(sqrt(5)) BRIDGE_11 & BRIDGE_12 certifier.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate.")
    parser.add_argument("--check", action="store_true", help="Run fail-closed theorem verification.")
    args = parser.parse_args(argv)

    cert = build_full_certificate()
    validate_certificate(cert)
    if args.json:
        sys.stdout.write(json.dumps(cert, sort_keys=True, indent=2) + "\n")
    elif args.check:
        sys.stdout.write(f"PASS S120 payload_sha256={cert['payload_sha256']}\n")
    else:
        sys.stdout.write(format_summary(cert) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
