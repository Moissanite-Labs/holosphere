#!/usr/bin/env python3
"""Project HoloSphere Pure-Z/Q Exact Witness Oracle for Lean 4.36.0 Invariants.

Zero IEEE-754 floats, zero bare asserts, zero external dependencies, zero RNG.
Computes exact integer/Fraction witnesses across all 5 HoloSphere Pillars:
  - Pillar 1: SU(2) ~= S^3 associative prefix scan, 8-lane Hillis-Steele warp scan,
              Euler four-square norm conservation, non-Abelian witness, and exact
              reverse-mode 4-lane additive suffix-sum adjoint collapse.
  - Pillar 2: Covariant (Q, K, V) left parallel transport isometry, relative holonomy
              attention score, prefix cancellation, and weighted value un-transport.
  - Pillar 3: SFU-free rational Cayley acute-cone Safe-Hopf clock (curing W10-OBS-31
              antipodal collapse) and Hopf S^3 -> S^2 right-S^1 fiber invariance.
  - Pillar 4: Beltrami [C, K] = 0 skew-Cayley helicity conservation, Poloidal Bingham
              below-yield stagnant-lid arrest, and rectangular Darboux intertwining.
  - Pillar 5: Weft composed-DeltaW C_row safety gate (blocking W10-OBS-20 multi-patch
              cancellation attack), isotropic sleep invariance, and LIFO rollback.

Companions `HoloSphereInvariants.lean` (30 public + 36 private = 66 theorem decls).
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import sys
from typing import Any, Sequence

SCHEMA_VERSION = "holosphere.lean_invariants.v1"
TRACK_ID = "holosphere-lean-invariants"
LEAN_FILE_NAME = "HoloSphereInvariants.lean"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.36.0"
LEAN_SOURCE_SHA256 = "059cb938f0dbd6feb732f59ebb30776a808829f2e5edcd77c8d54ca71bb61e4e"
LEAN_NAMED_THEOREM_COUNT = 30
LEAN_HELPER_THEOREM_COUNT = 36
LEAN_TOTAL_THEOREM_DECLS = 66

LEAN_THEOREM_SHORT_NAMES: tuple[str, ...] = (
    "p1_quat_mul_assoc",
    "p1_quat_identity_and_conj_antihom",
    "p1_quat_noncommutative_su2_holonomy_witness",
    "p1_quat_euler_four_square_norm",
    "p1_quat_prefix_scan_unit_norm",
    "p1_quat_chunked_scan_eq_sequential",
    "p1_quat_hillis_steele_8lane_eq_sequential",
    "p1_quat_bilinear_two_sided_adjoint",
    "p1_quat_scan_step_backward_factorization",
    "p1_quat_scan_suffix_sum_adjoint_collapse",
    "p2_covariant_left_right_inverse_cancel",
    "p2_covariant_left_transport_isometry",
    "p2_covariant_qk_score_relative_holonomy",
    "p2_covariant_interval_path_cancellation",
    "p2_covariant_value_untransport_single",
    "p2_covariant_value_untransport_weighted_sum",
    "p3_cayley_rational_unit_circle_identity",
    "p3_cayley_acute_cone_strict_pos_and_floor",
    "p3_cayley_acute_strict_monotone_distinct",
    "p3_w10_obs31_obtuse_antipodal_sign_flip",
    "p3_hopf_s3_to_s2_norm_and_right_fiber_invariance",
    "p4_beltrami_cayley_2x2_orthogonality_and_det",
    "p4_beltrami_paired_commutator_and_helicity_conservation",
    "p4_beltrami_symmetric_residual_rayleigh_drift",
    "p4_poloidal_bingham_hodge_stagnant_lid",
    "p4_darboux_isospectral_intertwining_and_dark_kernel",
    "p5_weft_w10_obs20_cancellation_attack_blocked",
    "p5_weft_tri_conjunctive_commit_soundness",
    "p5_weft_isotropic_sleep_preserves_crow_gate",
    "p5_weft_atomic_rollback_exact_involution",
)

LEAN_THEOREM_NAMES: tuple[str, ...] = tuple(
    f"HoloSphereInvariants.{short}" for short in LEAN_THEOREM_SHORT_NAMES
)

QuatQ = tuple[Fraction, Fraction, Fraction, Fraction]


class HoloSphereOracleError(ValueError):
    """Typed fail-closed error for HoloSphere pure-Q Lean oracle."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise HoloSphereOracleError(code, message)


def require_exact_int(val: Any, label: str) -> int:
    require_ok(type(val) is int, "TYPE_INT_REQUIRED", f"{label}: exact int required")
    return val


def require_exact_str(val: Any, label: str) -> str:
    require_ok(type(val) is str, "TYPE_STR_REQUIRED", f"{label}: exact str required")
    return val


def q_str(val: Fraction) -> str:
    require_ok(type(val) is Fraction, "TYPE_FRACTION_REQUIRED", "Fraction required")
    return str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Pure-Fraction Hamilton quaternion & matrix primitives (zero floats)
# ---------------------------------------------------------------------------

def q_make(w: Fraction | int, x: Fraction | int, y: Fraction | int, z: Fraction | int) -> QuatQ:
    return (Fraction(w), Fraction(x), Fraction(y), Fraction(z))


def q_add(p: QuatQ, q: QuatQ) -> QuatQ:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3])


def q_sub(p: QuatQ, q: QuatQ) -> QuatQ:
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3])


def q_scale(s: Fraction, p: QuatQ) -> QuatQ:
    return (s * p[0], s * p[1], s * p[2], s * p[3])


def q_conj(p: QuatQ) -> QuatQ:
    return (p[0], -p[1], -p[2], -p[3])


def q_mul(p: QuatQ, q: QuatQ) -> QuatQ:
    return (
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    )


def q_inner4(p: QuatQ, q: QuatQ) -> Fraction:
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]


def q_norm_sq(p: QuatQ) -> Fraction:
    return q_inner4(p, p)


def hillis_steele_8_q(us: Sequence[QuatQ]) -> list[QuatQ]:
    require_ok(len(us) == 8, "SCAN8_LEN", "hillis_steele_8_q requires exactly 8 elements")
    cur = list(us)
    for stride in (1, 2, 4):
        nxt = list(cur)
        for idx in range(stride, 8):
            nxt[idx] = q_mul(cur[idx - stride], cur[idx])
        cur = nxt
    return cur


def sequential_scan_q(us: Sequence[QuatQ]) -> list[QuatQ]:
    require_ok(len(us) > 0, "EMPTY_SCAN", "sequential_scan_q requires non-empty sequence")
    out = [us[0]]
    for u in us[1:]:
        out.append(q_mul(out[-1], u))
    return out


def cayley_cos_sin_q(t: Fraction) -> tuple[Fraction, Fraction]:
    t2 = t * t
    den = Fraction(1) + t2
    return ((Fraction(1) - t2) / den, (Fraction(2) * t) / den)


def hopf_s3_to_s2_q(q: QuatQ) -> tuple[Fraction, Fraction, Fraction]:
    w, x, y, z = q
    return (
        w * w + x * x - y * y - z * z,
        Fraction(2) * (x * y + w * z),
        Fraction(2) * (x * z - w * y),
    )


def mat_mul_q(a: Sequence[Sequence[Fraction]], b: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    require_ok(len(a) > 0 and len(b) > 0, "EMPTY_MATRIX", "matrix must be non-empty")
    k = len(a[0])
    require_ok(len(b) == k, "DIM_MISMATCH", "inner matrix dimensions must match")
    cols = len(b[0])
    return [
        [sum((a[i][r] * b[r][j] for r in range(k)), Fraction(0)) for j in range(cols)]
        for i in range(len(a))
    ]


def mat_vec_q(a: Sequence[Sequence[Fraction]], x: Sequence[Fraction]) -> list[Fraction]:
    require_ok(len(a) > 0, "EMPTY_MATRIX", "matrix must be non-empty")
    require_ok(len(a[0]) == len(x), "DIM_MISMATCH", "matrix-vector dimension mismatch")
    return [sum((row[j] * x[j] for j in range(len(x))), Fraction(0)) for row in a]


def row_concentration_q(mat: Sequence[Sequence[Fraction]]) -> Fraction:
    require_ok(len(mat) > 0 and len(mat[0]) > 0, "EMPTY_MATRIX", "matrix must be non-empty")
    row_sqs = [sum((v * v for v in row), Fraction(0)) for row in mat]
    frob_sq = sum(row_sqs, Fraction(0))
    require_ok(frob_sq > 0, "ZERO_NORM_MATRIX", "Frobenius norm must be positive")
    return max(row_sqs) / frob_sq


# ---------------------------------------------------------------------------
# Exact 5-Pillar Witness Builders
# ---------------------------------------------------------------------------

def _unit_stream_8() -> list[QuatQ]:
    f35 = Fraction(3, 5)
    f45 = Fraction(4, 5)
    f5_13 = Fraction(5, 13)
    f12_13 = Fraction(12, 13)
    return [
        q_make(f35, f45, 0, 0),
        q_make(f35, 0, f45, 0),
        q_make(0, f35, 0, f45),
        q_make(f45, 0, -f35, 0),
        q_make(f5_13, f12_13, 0, 0),
        q_make(f5_13, 0, 0, f12_13),
        q_make(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        q_make(f12_13, 0, -f5_13, 0),
    ]


def _build_pillar1() -> dict[str, Any]:
    us = _unit_stream_8()
    hs = hillis_steele_8_q(us)
    seq = sequential_scan_q(us)
    hs_diff = sum((q_norm_sq(q_sub(a, b)) for a, b in zip(hs, seq)), Fraction(0))

    left_chunk = sequential_scan_q(us[:4])[-1]
    right_from_left = [left_chunk]
    for u in us[4:]:
        right_from_left.append(q_mul(right_from_left[-1], u))
    chunk_diff = q_norm_sq(q_sub(right_from_left[-1], seq[-1]))

    assoc_diff = q_norm_sq(q_sub(q_mul(q_mul(us[0], us[1]), us[2]), q_mul(us[0], q_mul(us[1], us[2]))))
    comm_norm = q_norm_sq(q_sub(q_mul(us[0], us[1]), q_mul(us[1], us[0])))

    # Reverse-mode adjoint collapse at step t = 1 across future steps j in {1, 2, 3}
    gs = [
        q_make(1, -2, 3, -1),
        q_make(2, 1, -1, 4),
        q_make(-3, 2, 2, 1),
        q_make(1, 3, -2, 2),
    ]
    du1 = q_make(2, -1, 3, 5)
    pi0 = seq[0]
    pi1 = seq[1]
    r1_list = [
        q_make(1, 0, 0, 0),
        us[2],
        q_mul(us[2], us[3]),
    ]
    per_step_grad = Fraction(0)
    sigma_1 = q_make(0, 0, 0, 0)
    for idx, r1j in enumerate(r1_list, start=1):
        dpi_j = q_mul(q_mul(pi0, du1), r1j)
        per_step_grad += q_inner4(gs[idx], dpi_j)
        sigma_1 = q_add(sigma_1, q_mul(gs[idx], q_conj(seq[idx])))
    collapsed_grad_vec = q_mul(q_mul(q_conj(pi0), sigma_1), pi1)
    suffix_sum_grad = q_inner4(collapsed_grad_vec, du1)

    return {
        "assoc_diff_sq_q": q_str(assoc_diff),
        "nonabelian_commutator_norm_sq_q": q_str(comm_norm),
        "final_pi7_norm_sq_q": q_str(q_norm_sq(seq[-1])),
        "hillis_vs_seq_8lane_diff_sq_q": q_str(hs_diff),
        "chunked_vs_seq_diff_sq_q": q_str(chunk_diff),
        "per_step_grad_q": q_str(per_step_grad),
        "suffix_sum_grad_q": q_str(suffix_sum_grad),
        "adjoint_collapse_diff_q": q_str(per_step_grad - suffix_sum_grad),
    }


def _build_pillar2() -> dict[str, Any]:
    us = _unit_stream_8()
    seq = sequential_scan_q(us)
    pi_s = seq[1]
    delta_st = q_mul(us[2], us[3])
    pi_t = seq[3]
    q_vec = q_make(2, -3, 5, 1)
    k_vec = q_make(-1, 4, 2, -2)

    iso_diff = q_inner4(q_mul(pi_t, q_vec), q_mul(pi_t, k_vec)) - q_inner4(q_vec, k_vec)
    rel_hol = q_mul(q_conj(pi_t), pi_s)
    qk_diff = q_inner4(q_mul(pi_t, q_vec), q_mul(pi_s, k_vec)) - q_inner4(q_vec, q_mul(rel_hol, k_vec))
    interval_diff = q_norm_sq(q_sub(q_mul(q_conj(pi_s), pi_t), delta_st))

    weights = [Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)]
    vs = [q_make(1, 2, -1, 3), q_make(-2, 1, 4, 0), q_make(3, -1, 2, -2)]
    pis = [seq[0], seq[1], pi_t]
    trans_sum = q_make(0, 0, 0, 0)
    rel_sum = q_make(0, 0, 0, 0)
    for w, p_s, v_s in zip(weights, pis, vs):
        trans_sum = q_add(trans_sum, q_scale(w, q_mul(p_s, v_s)))
        rel_sum = q_add(rel_sum, q_scale(w, q_mul(q_mul(q_conj(pi_t), p_s), v_s)))
    untrans_out = q_mul(q_conj(pi_t), trans_sum)
    untrans_diff = q_norm_sq(q_sub(untrans_out, rel_sum))
    self_diff = q_norm_sq(q_sub(q_mul(q_conj(pi_t), q_mul(pi_t, vs[2])), vs[2]))

    return {
        "transport_isometry_diff_q": q_str(iso_diff),
        "qk_relative_holonomy_diff_q": q_str(qk_diff),
        "interval_cancel_diff_sq_q": q_str(interval_diff),
        "weighted_untransport_diff_sq_q": q_str(untrans_diff),
        "self_untransport_diff_sq_q": q_str(self_diff),
    }


def _build_pillar3() -> dict[str, Any]:
    t_max = Fraction(4, 5)
    c_floor, _ = cayley_cos_sin_q(t_max)
    r_steps = (0, 1, 2, 3, 5, 11)
    cos_vals: list[Fraction] = []
    max_circle_err = Fraction(0)
    for r in r_steps:
        t_r = t_max * Fraction(r, r + 2)
        c_r, s_r = cayley_cos_sin_q(t_r)
        err = abs(c_r * c_r + s_r * s_r - Fraction(1))
        if err > max_circle_err:
            max_circle_err = err
        cos_vals.append(c_r)
    strictly_decreasing = all(cos_vals[i + 1] < cos_vals[i] for i in range(len(cos_vals) - 1))

    # W10-OBS-31 obtuse/antipodal regime: t = 5 > 1 -> c = (1 - 25)/(1 + 25) = -12/13 < 0
    c_obtuse, s_obtuse = cayley_cos_sin_q(Fraction(5, 1))
    vx, vy = Fraction(2), Fraction(-3)
    rx = c_obtuse * vx - s_obtuse * vy
    ry = s_obtuse * vx + c_obtuse * vy
    self_align_obtuse = vx * rx + vy * ry

    # Hopf S^3 -> S^2 right-S^1 fiber invariance
    q0 = q_make(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    h0 = hopf_s3_to_s2_q(q0)
    h0_norm_sq = h0[0] * h0[0] + h0[1] * h0[1] + h0[2] * h0[2]
    q_s1 = q_mul(q0, q_make(Fraction(3, 5), Fraction(4, 5), 0, 0))
    h_s1 = hopf_s3_to_s2_q(q_s1)
    s1_diff = sum(((a - b) * (a - b) for a, b in zip(h0, h_s1)), Fraction(0))
    q_off = q_mul(q0, q_make(Fraction(3, 5), 0, Fraction(4, 5), 0))
    h_off = hopf_s3_to_s2_q(q_off)
    off_diff = sum(((a - b) * (a - b) for a, b in zip(h0, h_off)), Fraction(0))

    return {
        "unit_circle_max_err_q": q_str(max_circle_err),
        "cos_floor_tmax_4_5_q": q_str(c_floor),
        "min_schedule_cos_q": q_str(min(cos_vals)),
        "strictly_monotone_schedule": strictly_decreasing,
        "obs31_obtuse_cos_q": q_str(c_obtuse),
        "obs31_self_align_q": q_str(self_align_obtuse),
        "hopf_s2_norm_sq_q": q_str(h0_norm_sq),
        "fiber_rotated_diff_sq_q": q_str(s1_diff),
        "off_fiber_diff_sq_q": q_str(off_diff),
    }


def _build_pillar4() -> dict[str, Any]:
    # Paired 4x4 Beltrami Cayley block: c = 3/5, s = 4/5, lam = 3
    c, s = Fraction(3, 5), Fraction(4, 5)
    lam = Fraction(3)
    v = [Fraction(2), Fraction(3), Fraction(1), Fraction(0)]
    y = [
        c * v[0] + s * v[1],
        -s * v[0] + c * v[1],
        c * v[2] + s * v[3],
        -s * v[2] + c * v[3],
    ]
    kv = [lam * v[0], lam * v[1], -lam * v[2], -lam * v[3]]
    ck_v = [
        c * kv[0] + s * kv[1],
        -s * kv[0] + c * kv[1],
        c * kv[2] + s * kv[3],
        -s * kv[2] + c * kv[3],
    ]
    kc_v = [lam * y[0], lam * y[1], -lam * y[2], -lam * y[3]]
    comm_diff = sum(((a - b) * (a - b) for a, b in zip(ck_v, kc_v)), Fraction(0))
    e_in = sum((x * x for x in v), Fraction(0))
    e_out = sum((x * x for x in y), Fraction(0))
    h_in = lam * (v[0] * v[0] + v[1] * v[1] - v[2] * v[2] - v[3] * v[3])
    h_out = lam * (y[0] * y[0] + y[1] * y[1] - y[2] * y[2] - y[3] * y[3])
    v_swap = [v[0], v[2], v[1], v[3]]
    e_swap = sum((x * x for x in v_swap), Fraction(0))
    h_swap = lam * (v_swap[0] ** 2 + v_swap[1] ** 2 - v_swap[2] ** 2 - v_swap[3] ** 2)

    # O3.5 symmetric commuting residual diag(2, 2, 1, 1) on (1, 0, 1, 0)
    ray_before = Fraction(0, 2)
    ray_after = (lam * (Fraction(4) - Fraction(1))) / (Fraction(4) + Fraction(1))

    # Poloidal Bingham stagnant lid: ||u_pol||^2 = 5 <= tau^2 = 9, ||u_tor||^2 = 25
    u_pol = [Fraction(1), Fraction(2)]
    u_tor = [Fraction(3), Fraction(-4)]
    tau_sq = Fraction(9)
    pol_sq = sum((x * x for x in u_pol), Fraction(0))
    tor_sq = sum((x * x for x in u_tor), Fraction(0))
    pol_gated = [Fraction(0) if pol_sq <= tau_sq else x for x in u_pol]
    pol_gated_sq = sum((x * x for x in pol_gated), Fraction(0))

    # Rectangular 2x3 Darboux intertwining with shift c_shift = 2 and kernel (1, 1, 1)
    a_mat = [
        [Fraction(1), Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(-1)],
    ]
    at_mat = [[a_mat[r][col] for r in range(2)] for col in range(3)]
    c_shift = Fraction(2)
    ata = mat_mul_q(at_mat, a_mat)
    aat = mat_mul_q(a_mat, at_mat)
    l1 = [[ata[i][j] + (c_shift if i == j else Fraction(0)) for j in range(3)] for i in range(3)]
    l2 = [[aat[i][j] + (c_shift if i == j else Fraction(0)) for j in range(2)] for i in range(2)]
    al1 = mat_mul_q(a_mat, l1)
    l2a = mat_mul_q(l2, a_mat)
    intertwine_diff = sum(
        ((al1[i][j] - l2a[i][j]) ** 2 for i in range(2) for j in range(3)),
        Fraction(0),
    )
    k_vec = [Fraction(1), Fraction(1), Fraction(1)]
    x_vec = [Fraction(4), Fraction(-1), Fraction(2)]
    ak = mat_vec_q(a_mat, k_vec)
    ak_sq = sum((v_i * v_i for v_i in ak), Fraction(0))
    ax = mat_vec_q(a_mat, x_vec)
    axk = mat_vec_q(a_mat, [x_vec[j] + k_vec[j] for j in range(3)])
    col_diff = sum(((ax[i] - axk[i]) ** 2 for i in range(2)), Fraction(0))

    return {
        "beltrami_commutator_diff_sq_q": q_str(comm_diff),
        "beltrami_energy_in_q": q_str(e_in),
        "beltrami_energy_out_q": q_str(e_out),
        "beltrami_helicity_in_q": q_str(h_in),
        "beltrami_helicity_out_q": q_str(h_out),
        "cross_swap_energy_out_q": q_str(e_swap),
        "cross_swap_helicity_out_q": q_str(h_swap),
        "rayleigh_before_q": q_str(ray_before),
        "rayleigh_after_sym_res_q": q_str(ray_after),
        "bingham_pol_in_norm_sq_q": q_str(pol_sq),
        "bingham_tau_sq_q": q_str(tau_sq),
        "bingham_pol_gated_norm_sq_q": q_str(pol_gated_sq),
        "bingham_tor_out_norm_sq_q": q_str(tor_sq),
        "darboux_intertwine_diff_sq_q": q_str(intertwine_diff),
        "darboux_ak_norm_sq_q": q_str(ak_sq),
        "darboux_kernel_collision_diff_sq_q": q_str(col_diff),
    }


def _build_pillar5() -> dict[str, Any]:
    c_max = Fraction(3, 5)
    dw1 = [[Fraction(3), Fraction(0)], [Fraction(3), Fraction(0)]]
    dw2 = [[Fraction(3), Fraction(0)], [Fraction(-3), Fraction(0)]]
    dw_comp = [[dw1[i][j] + dw2[i][j] for j in range(2)] for i in range(2)]
    c1 = row_concentration_q(dw1)
    c2 = row_concentration_q(dw2)
    c_comp = row_concentration_q(dw_comp)

    lam_sleep = Fraction(3, 4)
    dw1_sleep = [[lam_sleep * dw1[i][j] for j in range(2)] for i in range(2)]
    c_sleep = row_concentration_q(dw1_sleep)

    w0 = [[Fraction(5), Fraction(-2)], [Fraction(1), Fraction(4)]]
    w_after = [[w0[i][j] + dw1[i][j] + dw2[i][j] for j in range(2)] for i in range(2)]
    w_rolled = [[w_after[i][j] - dw2[i][j] - dw1[i][j] for j in range(2)] for i in range(2)]
    rb_diff = sum(((w_rolled[i][j] - w0[i][j]) ** 2 for i in range(2) for j in range(2)), Fraction(0))

    return {
        "c_max_q": q_str(c_max),
        "c_row_patch1_q": q_str(c1),
        "c_row_patch2_q": q_str(c2),
        "c_row_composed_attack_q": q_str(c_comp),
        "composed_attack_accepted": bool(c_comp <= c_max),
        "safe_single_patch_accepted": bool(c1 <= c_max),
        "c_row_post_isotropic_sleep_q": q_str(c_sleep),
        "lifo_rollback_diff_sq_q": q_str(rb_diff),
    }


def build_payload() -> dict[str, Any]:
    return {
        "novelty_grade": "N1",
        "n2_claims": 0,
        "scope_note": (
            "Pure-Init Lean 4.36.0 machine-checked algebraic invariants and exact rational "
            "witnesses for Project HoloSphere across SU(2) holonomy scan, covariant QKV "
            "transport, SFU-free Cayley Safe-Hopf, Beltrami/Bingham/Darboux, and Weft safety."
        ),
        "lean_companion": {
            "lean_file": require_exact_str(LEAN_FILE_NAME, "lean_file"),
            "lean_toolchain": require_exact_str(LEAN_TOOLCHAIN, "lean_toolchain"),
            "lean_source_sha256": require_exact_str(LEAN_SOURCE_SHA256, "lean_source_sha256"),
            "named_theorem_count": require_exact_int(LEAN_NAMED_THEOREM_COUNT, "named_cnt"),
            "helper_theorem_count": require_exact_int(LEAN_HELPER_THEOREM_COUNT, "helper_cnt"),
            "total_theorem_declarations": require_exact_int(LEAN_TOTAL_THEOREM_DECLS, "total_cnt"),
            "theorem_names": list(LEAN_THEOREM_NAMES),
        },
        "pillar1_su2_scan_and_suffix_adjoint": _build_pillar1(),
        "pillar2_covariant_qkv_transport": _build_pillar2(),
        "pillar3_cayley_safe_hopf_and_fiber": _build_pillar3(),
        "pillar4_beltrami_bingham_darboux": _build_pillar4(),
        "pillar5_weft_crow_safety_and_rollback": _build_pillar5(),
    }


def build_certificate() -> dict[str, Any]:
    payload = build_payload()
    payload_sha = sha256_hex(canonical_json_bytes(payload))
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
    }
    cert_sha = sha256_hex(canonical_json_bytes(envelope))
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cert = build_certificate()
    if args == ["--standalone"]:
        sys.stdout.write(canonical_json_bytes(cert).decode("ascii") + "\n")
        return 0
    require_ok(args == [], "UNKNOWN_ORACLE_ARG", f"unrecognized args: {args}")
    summary = {
        "track_id": TRACK_ID,
        "lean_theorems": LEAN_NAMED_THEOREM_COUNT,
        "payload_sha256": cert["payload_sha256"],
        "certificate_sha256": cert["certificate_sha256"],
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
