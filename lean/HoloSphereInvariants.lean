-- Project HoloSphere — Pure-Init Lean 4.36.0 Formalization of Core Silicon-Geometry Invariants
-- Zero external imports, zero Mathlib, zero sorry, zero Classical.choice, zero warnings.
-- Covers Pillars 1 (SU(2) Associative Scan, Euler Norm & Suffix-Sum Adjoint Collapse),
-- 2 (Covariant Q/K/V Parallel Transport & Un-Transport),
-- 3 (SFU-Free Rational Cayley Acute-Cone Safe-Hopf Clock & Hopf S^3 -> S^2 Fiber),
-- 4 (Beltrami [C, K] = 0 Helicity, Poloidal Bingham Stagnant-Lid & Darboux Intertwining),
-- and 5 (Weft Composed-DeltaW C_row Gate, Cancellation Attack W10-OBS-20 & Rollback).

namespace HoloSphereInvariants

-- ============================================================================
-- 0. Constructive Integer Helpers & Quaternion / Matrix Definitions (Pure Init)
-- ============================================================================

private theorem int_sq_nonneg (a : Int) : 0 ≤ a * a := by
  if h : 0 ≤ a then
    exact Int.mul_nonneg h h
  else
    have hna : 0 ≤ -a := by omega
    have hmul : 0 ≤ (-a) * (-a) := Int.mul_nonneg hna hna
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem int_sq_pos_of_ne_zero (a : Int) (ha : a ≠ 0) : 0 < a * a := by
  if h : 0 < a then
    exact Int.mul_pos h h
  else
    have hna : 0 < -a := by omega
    have hmul : 0 < (-a) * (-a) := Int.mul_pos hna hna
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem int_diff_sq_identity (a b : Int) :
    a * a - b * b = (a - b) * (a + b) := by
  simp only [Int.sub_mul, Int.mul_add, Int.mul_comm b a]
  omega

private theorem int_sq_le_sq_of_nonneg (a b : Int) (ha : 0 ≤ a) (hab : a ≤ b) :
    a * a ≤ b * b := by
  have h1 : 0 ≤ b - a := by omega
  have h2 : 0 ≤ b + a := by omega
  have hprod : 0 ≤ (b - a) * (b + a) := Int.mul_nonneg h1 h2
  have hid := int_diff_sq_identity b a
  omega

private theorem int_sq_lt_sq_of_nonneg (a b : Int) (ha : 0 ≤ a) (hab : a < b) :
    a * a < b * b := by
  have h1 : 0 < b - a := by omega
  have h2 : 0 < b + a := by omega
  have hprod : 0 < (b - a) * (b + a) := Int.mul_pos h1 h2
  have hid := int_diff_sq_identity b a
  omega

private theorem int_mul_le_iff_of_pos_left (s x y : Int) (hs : 0 < s) :
    s * x ≤ s * y ↔ x ≤ y := by
  constructor
  · intro hle
    if hxy : x ≤ y then
      exact hxy
    else
      have hgt : 0 < x - y := by omega
      have hp : 0 < s * (x - y) := Int.mul_pos hs hgt
      rw [Int.mul_sub] at hp
      omega
  · intro hxy
    have hge : 0 ≤ y - x := by omega
    have hp : 0 ≤ s * (y - x) := Int.mul_nonneg (Int.le_of_lt hs) hge
    rw [Int.mul_sub] at hp
    omega

-- Hamilton quaternion over Int matching holosphere/geometry/su2.py (w, x, y, z)
structure QuatZ where
  w : Int
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

def qzero : QuatZ := ⟨0, 0, 0, 0⟩
def qone : QuatZ := ⟨1, 0, 0, 0⟩
def qi : QuatZ := ⟨0, 1, 0, 0⟩
def qj : QuatZ := ⟨0, 0, 1, 0⟩
def qk : QuatZ := ⟨0, 0, 0, 1⟩

def qadd (p q : QuatZ) : QuatZ :=
  ⟨p.w + q.w, p.x + q.x, p.y + q.y, p.z + q.z⟩

def qsub (p q : QuatZ) : QuatZ :=
  ⟨p.w - q.w, p.x - q.x, p.y - q.y, p.z - q.z⟩

def qneg (p : QuatZ) : QuatZ :=
  ⟨-p.w, -p.x, -p.y, -p.z⟩

def qscale (s : Int) (p : QuatZ) : QuatZ :=
  ⟨s * p.w, s * p.x, s * p.y, s * p.z⟩

def qconj (p : QuatZ) : QuatZ :=
  ⟨p.w, -p.x, -p.y, -p.z⟩

def qmul (p q : QuatZ) : QuatZ :=
  ⟨p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
   p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
   p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
   p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w⟩

def inner4 (p q : QuatZ) : Int :=
  p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z

def normSq (p : QuatZ) : Int :=
  inner4 p p

private theorem quat_ext {p q : QuatZ}
    (hw : p.w = q.w) (hx : p.x = q.x) (hy : p.y = q.y) (hz : p.z = q.z) : p = q := by
  cases p
  cases q
  simp only at hw hx hy hz
  subst hw hx hy hz
  rfl

private theorem qconj_involutive (p : QuatZ) : qconj (qconj p) = p := by
  apply quat_ext <;> simp only [qconj] <;> omega

private theorem inner4_symm (p q : QuatZ) : inner4 p q = inner4 q p := by
  simp only [inner4, Int.mul_comm]

private theorem inner4_qadd_left (p q r : QuatZ) :
    inner4 (qadd p q) r = inner4 p r + inner4 q r := by
  simp only [inner4, qadd, Int.add_mul]
  omega

private theorem inner4_qscale_left (s : Int) (p q : QuatZ) :
    inner4 (qscale s p) q = s * inner4 p q := by
  simp only [inner4, qscale, Int.mul_add, Int.mul_assoc]

private theorem qmul_qadd_left (p q r : QuatZ) :
    qmul (qadd p q) r = qadd (qmul p r) (qmul q r) := by
  apply quat_ext <;> simp only [qmul, qadd, Int.add_mul] <;> omega

private theorem qmul_qadd_right (p q r : QuatZ) :
    qmul p (qadd q r) = qadd (qmul p q) (qmul p r) := by
  apply quat_ext <;> simp only [qmul, qadd, Int.mul_add] <;> omega

private theorem qmul_qscale_left (s : Int) (p q : QuatZ) :
    qmul (qscale s p) q = qscale s (qmul p q) := by
  apply quat_ext <;>
    simp only [qmul, qscale, Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.mul_assoc]

private theorem qmul_qscale_right (s : Int) (p q : QuatZ) :
    qmul p (qscale s q) = qscale s (qmul p q) := by
  apply quat_ext <;>
    simp only [qmul, qscale, Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg,
               Int.mul_comm, Int.mul_left_comm]

private theorem qscale_one (p : QuatZ) : qscale 1 p = p := by
  apply quat_ext <;> simp only [qscale, Int.one_mul]

private theorem two_mul_eq_add (x : Int) : 2 * x = x + x := by
  omega

private theorem quat_conj_mul_self (pi : QuatZ) :
    qmul (qconj pi) pi = ⟨normSq pi, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, inner4, Int.sub_eq_add_neg,
               Int.mul_neg, Int.neg_neg, Int.mul_comm] <;>
    omega

private theorem quat_mul_conj_self (pi : QuatZ) :
    qmul pi (qconj pi) = ⟨normSq pi, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, inner4, Int.mul_neg, Int.mul_comm] <;>
    omega

private theorem qmul_scalar_left (s : Int) (v : QuatZ) :
    qmul ⟨s, 0, 0, 0⟩ v = qscale s v := by
  apply quat_ext <;> simp only [qmul, qscale, Int.zero_mul] <;> omega

private theorem qmul_scalar_right (v : QuatZ) (s : Int) :
    qmul v ⟨s, 0, 0, 0⟩ = qscale s v := by
  apply quat_ext <;> simp only [qmul, qscale, Int.mul_zero, Int.mul_comm] <;> omega

private theorem quat_left_adj (a b x : QuatZ) :
    inner4 a (qmul b x) = inner4 (qmul (qconj b) a) x := by
  simp only [inner4, qmul, qconj, Int.sub_eq_add_neg, Int.mul_add,
             Int.mul_neg, Int.neg_neg, Int.mul_comm, Int.mul_left_comm]
  omega

private theorem quat_right_adj (a x c : QuatZ) :
    inner4 a (qmul x c) = inner4 (qmul a (qconj c)) x := by
  simp only [inner4, qmul, qconj, Int.sub_eq_add_neg, Int.mul_add,
             Int.mul_neg, Int.neg_neg, Int.mul_comm, Int.mul_left_comm]
  omega

-- ============================================================================
-- Pillar 1: SU(2) Associative Scan, Euler Norm & Reverse-Mode Adjoint Collapse
-- ============================================================================

-- Theorem 1: Universal Hamilton quaternion associativity
theorem p1_quat_mul_assoc (p q r : QuatZ) :
    qmul (qmul p q) r = qmul p (qmul q r) := by
  apply quat_ext <;>
    simp only [qmul, Int.sub_eq_add_neg, Int.add_mul, Int.mul_add,
               Int.neg_add, Int.mul_neg, Int.neg_mul, Int.neg_neg, Int.mul_assoc] <;>
    omega

-- Theorem 2: Identity element and conjugation anti-homomorphism + norm invariance
theorem p1_quat_identity_and_conj_antihom (p q : QuatZ) :
    qmul qone p = p ∧
    qmul p qone = p ∧
    qconj (qmul p q) = qmul (qconj q) (qconj p) ∧
    normSq (qconj p) = normSq p := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [qone, qmul_scalar_left, qscale_one]
  · rw [qone, qmul_scalar_right, qscale_one]
  · apply quat_ext <;>
      simp only [qmul, qconj, Int.sub_eq_add_neg, Int.neg_add,
                 Int.mul_neg, Int.neg_neg, Int.mul_comm] <;>
      omega
  · simp only [normSq, inner4, qconj, Int.neg_mul_neg]

-- Theorem 3: Non-Abelian SU(2) holonomy witness separating SU(2) from Abelian RoPE/LieRE
theorem p1_quat_noncommutative_su2_holonomy_witness :
    qmul qi qj = qk ∧ qmul qj qi = qneg qk ∧ qmul qi qj ≠ qmul qj qi := by
  decide

-- Theorem 4: Euler four-square multiplicative norm conservation ||p * q||^2 = ||p||^2 * ||q||^2
theorem p1_quat_euler_four_square_norm (p q : QuatZ) :
    normSq (qmul p q) = normSq p * normSq q := by
  rw [normSq, quat_left_adj, ← p1_quat_mul_assoc, quat_conj_mul_self,
      qmul_scalar_left, inner4_qscale_left]
  rfl

-- Theorem 5: Arbitrary-length S^3 prefix scan preserves unit norm identically
theorem p1_quat_prefix_scan_unit_norm (us : List QuatZ) (init : QuatZ)
    (hinit : normSq init = 1)
    (hus : ∀ u, u ∈ us → normSq u = 1) :
    normSq (us.foldl qmul init) = 1 := by
  induction us generalizing init with
  | nil => exact hinit
  | cons u rest ih =>
    simp only [List.foldl]
    apply ih
    · rw [p1_quat_euler_four_square_norm, hinit, hus u (List.Mem.head rest)]
      rfl
    · intro v hv
      exact hus v (List.Mem.tail u hv)

-- Theorem 6: Chunked / segmented prefix scan equals sequential O(1)-state recurrence
theorem p1_quat_chunked_scan_eq_sequential (xs ys : List QuatZ) (init : QuatZ) :
    (xs ++ ys).foldl qmul init = ys.foldl qmul (xs.foldl qmul init) := by
  induction xs generalizing init with
  | nil => rfl
  | cons x rest ih =>
    simp only [List.cons_append, List.foldl]
    exact ih (qmul init x)

-- Explicit 8-lane Hillis-Steele 3-round binary-tree warp scan vs sequential scan
def hillisSteele8 (u0 u1 u2 u3 u4 u5 u6 u7 : QuatZ) :
    QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ :=
  let a0 := u0
  let a1 := qmul u0 u1
  let a2 := qmul u1 u2
  let a3 := qmul u2 u3
  let a4 := qmul u3 u4
  let a5 := qmul u4 u5
  let a6 := qmul u5 u6
  let a7 := qmul u6 u7
  let b0 := a0
  let b1 := a1
  let b2 := qmul a0 a2
  let b3 := qmul a1 a3
  let b4 := qmul a2 a4
  let b5 := qmul a3 a5
  let b6 := qmul a4 a6
  let b7 := qmul a5 a7
  let c0 := b0
  let c1 := b1
  let c2 := b2
  let c3 := b3
  let c4 := qmul b0 b4
  let c5 := qmul b1 b5
  let c6 := qmul b2 b6
  let c7 := qmul b3 b7
  (c0, c1, c2, c3, c4, c5, c6, c7)

def sequentialScan8 (u0 u1 u2 u3 u4 u5 u6 u7 : QuatZ) :
    QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ × QuatZ :=
  let s0 := u0
  let s1 := qmul s0 u1
  let s2 := qmul s1 u2
  let s3 := qmul s2 u3
  let s4 := qmul s3 u4
  let s5 := qmul s4 u5
  let s6 := qmul s5 u6
  let s7 := qmul s6 u7
  (s0, s1, s2, s3, s4, s5, s6, s7)

-- Theorem 7: 8-lane Hillis-Steele parallel warp scan equals sequential inference scan
theorem p1_quat_hillis_steele_8lane_eq_sequential
    (u0 u1 u2 u3 u4 u5 u6 u7 : QuatZ) :
    hillisSteele8 u0 u1 u2 u3 u4 u5 u6 u7 = sequentialScan8 u0 u1 u2 u3 u4 u5 u6 u7 := by
  simp only [hillisSteele8, sequentialScan8, p1_quat_mul_assoc]

-- Theorem 8: Universal two-sided quaternion multiplication reverse-mode adjoint
theorem p1_quat_bilinear_two_sided_adjoint (a b x c : QuatZ) :
    inner4 a (qmul (qmul b x) c) = inner4 (qmul (qmul (qconj b) a) (qconj c)) x := by
  rw [quat_right_adj, quat_left_adj, ← p1_quat_mul_assoc]

-- Theorem 9: Exact prefix-scan backward step factorization (eliminates r_{t->j})
theorem p1_quat_scan_step_backward_factorization
    (pi_prev pi_t r_tj g du : QuatZ) :
    let pi_j := qmul pi_t r_tj
    normSq pi_t * inner4 g (qmul (qmul pi_prev du) r_tj) =
      inner4 (qmul (qmul (qconj pi_prev) (qmul g (qconj pi_j))) pi_t) du ∧
    (normSq pi_t = 1 →
      inner4 g (qmul (qmul pi_prev du) r_tj) =
        inner4 (qmul (qmul (qconj pi_prev) (qmul g (qconj pi_j))) pi_t) du) := by
  have hrel : qmul (qconj (qmul pi_t r_tj)) pi_t = qscale (normSq pi_t) (qconj r_tj) := by
    rw [(p1_quat_identity_and_conj_antihom pi_t r_tj).2.2.1,
        p1_quat_mul_assoc, quat_conj_mul_self, qmul_scalar_right]
  have hmain :
      normSq pi_t * inner4 g (qmul (qmul pi_prev du) r_tj) =
        inner4 (qmul (qmul (qconj pi_prev) (qmul g (qconj (qmul pi_t r_tj)))) pi_t) du := by
    rw [p1_quat_bilinear_two_sided_adjoint, ← inner4_qscale_left]
    rw [p1_quat_mul_assoc (qconj pi_prev) (qmul g (qconj (qmul pi_t r_tj))) pi_t,
        p1_quat_mul_assoc g (qconj (qmul pi_t r_tj)) pi_t,
        hrel, qmul_qscale_right, qmul_qscale_right, ← p1_quat_mul_assoc]
  refine ⟨hmain, ?_⟩
  intro hunit
  rw [← hmain, hunit, Int.one_mul]

def perStepGradSum (pi_prev pi_t du : QuatZ) : List (QuatZ × QuatZ) → Int
  | [] => 0
  | (g_j, pi_j) :: rest =>
      inner4 (qmul (qmul (qconj pi_prev) (qmul g_j (qconj pi_j))) pi_t) du +
        perStepGradSum pi_prev pi_t du rest

def suffixRightAdjointSum : List (QuatZ × QuatZ) → QuatZ
  | [] => qzero
  | (g_j, pi_j) :: rest =>
      qadd (qmul g_j (qconj pi_j)) (suffixRightAdjointSum rest)

-- Theorem 10: Non-Abelian backward scan collapses to a 4-lane additive suffix sum
theorem p1_quat_scan_suffix_sum_adjoint_collapse
    (pi_prev pi_t du : QuatZ) (suffix : List (QuatZ × QuatZ)) :
    perStepGradSum pi_prev pi_t du suffix =
      inner4 (qmul (qmul (qconj pi_prev) (suffixRightAdjointSum suffix)) pi_t) du := by
  induction suffix with
  | nil =>
    simp only [perStepGradSum, suffixRightAdjointSum]
    have hz1 : qmul (qconj pi_prev) qzero = qzero := by
      apply quat_ext <;> simp only [qmul, qzero, Int.mul_zero] <;> omega
    have hz2 : qmul qzero pi_t = qzero := by
      apply quat_ext <;> simp only [qmul, qzero, Int.zero_mul] <;> omega
    rw [hz1, hz2]
    simp only [inner4, qzero, Int.zero_mul]
    omega
  | cons hd tl ih =>
    cases hd with
    | mk g_j pi_j =>
      simp only [perStepGradSum, suffixRightAdjointSum]
      rw [ih, qmul_qadd_right, qmul_qadd_left, inner4_qadd_left]

-- Homogenized SFU-free rational SU(2) Cayley map matching holosphere/geometry/su2.py (cayley_su2)
def cayleySU2Homog (d x y z : Int) : QuatZ :=
  ⟨d * d - x * x - y * y - z * z, 2 * d * x, 2 * d * y, 2 * d * z⟩

private theorem two_prod_sq_factor (d v : Int) :
    (2 * d * v) * (2 * d * v) = (2 * d) * (2 * d) * (v * v) := by
  simp only [Int.mul_assoc, Int.mul_left_comm]

private theorem cayley_su2_quad_identity (d s : Int) :
    (d * d - s) * (d * d - s) + (2 * d) * (2 * d) * s =
      (d * d + s) * (d * d + s) := by
  rw [two_mul_eq_add d]
  simp only [Int.sub_eq_add_neg, Int.mul_add,
             Int.mul_neg, Int.neg_neg, Int.mul_comm, Int.mul_left_comm]
  omega

-- Theorem 11: SFU-free rational SU(2) Cayley map exact Euler norm and acute -I pole immunity
theorem p1_cayley_su2_unit_norm_and_acute_positivity (d x y z : Int) :
    normSq (cayleySU2Homog d x y z) =
      (d * d + x * x + y * y + z * z) * (d * d + x * x + y * y + z * z) ∧
    normSq (cayleySU2Homog 1 x y z) =
      (1 + x * x + y * y + z * z) * (1 + x * x + y * y + z * z) ∧
    (x * x + y * y + z * z < d * d →
      0 < (cayleySU2Homog d x y z).w ∧ 0 < d * d + x * x + y * y + z * z) := by
  have hgen :
      normSq (cayleySU2Homog d x y z) =
        (d * d + x * x + y * y + z * z) * (d * d + x * x + y * y + z * z) := by
    simp only [normSq, inner4, cayleySU2Homog]
    have hw : d * d - x * x - y * y - z * z = d * d - (x * x + y * y + z * z) := by omega
    have hp : d * d + x * x + y * y + z * z = d * d + (x * x + y * y + z * z) := by omega
    have him :
        (2 * d * x) * (2 * d * x) + (2 * d * y) * (2 * d * y) + (2 * d * z) * (2 * d * z) =
          (2 * d) * (2 * d) * (x * x + y * y + z * z) := by
      rw [two_prod_sq_factor d x, two_prod_sq_factor d y, two_prod_sq_factor d z,
          ← Int.mul_add, ← Int.mul_add]
    have hassoc4 : ∀ a b c e : Int, a + b + c + e = a + (b + c + e) := by
      intro a b c e; omega
    rw [hassoc4, hw, hp, him]
    exact cayley_su2_quad_identity d (x * x + y * y + z * z)
  have h1 :
      normSq (cayleySU2Homog 1 x y z) =
        (1 + x * x + y * y + z * z) * (1 + x * x + y * y + z * z) := by
    have h11 : (1 : Int) * 1 = 1 := rfl
    have hspec :
        normSq (cayleySU2Homog 1 x y z) =
          (1 * 1 + x * x + y * y + z * z) * (1 * 1 + x * x + y * y + z * z) := by
      simp only [normSq, inner4, cayleySU2Homog]
      have hw : 1 * 1 - x * x - y * y - z * z = 1 * 1 - (x * x + y * y + z * z) := by omega
      have hp : 1 * 1 + x * x + y * y + z * z = 1 * 1 + (x * x + y * y + z * z) := by omega
      have him :
          (2 * 1 * x) * (2 * 1 * x) + (2 * 1 * y) * (2 * 1 * y) + (2 * 1 * z) * (2 * 1 * z) =
            (2 * 1) * (2 * 1) * (x * x + y * y + z * z) := by
        rw [two_prod_sq_factor 1 x, two_prod_sq_factor 1 y, two_prod_sq_factor 1 z,
            ← Int.mul_add, ← Int.mul_add]
      have hassoc4 : ∀ a b c e : Int, a + b + c + e = a + (b + c + e) := by
        intro a b c e; omega
      rw [hassoc4, hw, hp, him]
      exact cayley_su2_quad_identity 1 (x * x + y * y + z * z)
    rw [h11] at hspec
    exact hspec
  refine ⟨hgen, h1, ?_⟩
  intro hlt
  have hx2 := int_sq_nonneg x
  have hy2 := int_sq_nonneg y
  have hz2 := int_sq_nonneg z
  simp only [cayleySU2Homog]
  refine ⟨by omega, by omega⟩

private theorem qsub_self (p : QuatZ) : qsub p p = qzero := by
  apply quat_ext <;> simp only [qsub, qzero] <;> omega

private theorem qsub_qscale_left (s1 s2 : Int) (p : QuatZ) :
    qsub (qscale s1 p) (qscale s2 p) = qscale (s1 - s2) p := by
  apply quat_ext <;> simp only [qsub, qscale, Int.sub_mul]

private theorem normSq_qscale (s : Int) (p : QuatZ) :
    normSq (qscale s p) = (s * s) * normSq p := by
  unfold normSq
  rw [inner4_qscale_left, inner4_symm, inner4_qscale_left, inner4_symm, ← Int.mul_assoc]

-- Theorem 12: Local block-4 S^3 normalization has zero cross-bank bleed vs global sphere coupling
theorem p1_local_4d_s3_zero_cross_bank_bleed
    (qA qB qB_new : QuatZ) (hA : 0 < normSq qA) (hB : normSq qB ≠ normSq qB_new) :
    qsub (qscale (normSq qA) qA) (qscale (normSq qA) qA) = qzero ∧
    qsub (qscale (normSq qA + normSq qB_new) qA) (qscale (normSq qA + normSq qB) qA) =
      qscale (normSq qB_new - normSq qB) qA ∧
    0 < normSq (qsub (qscale (normSq qA + normSq qB_new) qA)
                     (qscale (normSq qA + normSq qB) qA)) := by
  have hdiff : (normSq qA + normSq qB_new) - (normSq qA + normSq qB) =
      normSq qB_new - normSq qB := by omega
  have hglobal :
      qsub (qscale (normSq qA + normSq qB_new) qA) (qscale (normSq qA + normSq qB) qA) =
        qscale (normSq qB_new - normSq qB) qA := by
    rw [qsub_qscale_left, hdiff]
  have hne : normSq qB_new - normSq qB ≠ 0 := by omega
  have hsq_pos : 0 < (normSq qB_new - normSq qB) * (normSq qB_new - normSq qB) :=
    int_sq_pos_of_ne_zero (normSq qB_new - normSq qB) hne
  refine ⟨qsub_self (qscale (normSq qA) qA), hglobal, ?_⟩
  rw [hglobal, normSq_qscale]
  exact Int.mul_pos hsq_pos hA

-- ============================================================================
-- Pillar 2: Covariant (Q, K, V) Parallel Transport & Un-Transport (H14c)
-- ============================================================================

-- Theorem 11: Exact left/right inverse cancellation by conjugation on S^3
theorem p2_covariant_left_right_inverse_cancel (pi v : QuatZ) :
    qmul (qconj pi) (qmul pi v) = qscale (normSq pi) v ∧
    qmul (qmul v pi) (qconj pi) = qscale (normSq pi) v ∧
    (normSq pi = 1 → qmul (qconj pi) (qmul pi v) = v ∧ qmul (qmul v pi) (qconj pi) = v) := by
  have hl : qmul (qconj pi) (qmul pi v) = qscale (normSq pi) v := by
    rw [← p1_quat_mul_assoc, quat_conj_mul_self, qmul_scalar_left]
  have hr : qmul (qmul v pi) (qconj pi) = qscale (normSq pi) v := by
    rw [p1_quat_mul_assoc, quat_mul_conj_self, qmul_scalar_right]
  refine ⟨hl, hr, ?_⟩
  intro h1
  rw [hl, hr, h1, qscale_one]
  exact ⟨rfl, rfl⟩

-- Theorem 12: Left parallel transport preserves Euclidean 4-vector inner products
theorem p2_covariant_left_transport_isometry (pi u v : QuatZ) :
    inner4 (qmul pi u) (qmul pi v) = normSq pi * inner4 u v := by
  rw [quat_left_adj, (p2_covariant_left_right_inverse_cancel pi u).1, inner4_qscale_left]

-- Theorem 13: Transported Q-K dot product computes exact relative holonomy conj(pi_t) * pi_s
theorem p2_covariant_qk_score_relative_holonomy (pi_t pi_s q k : QuatZ) :
    inner4 (qmul pi_t q) (qmul pi_s k) =
      inner4 q (qmul (qmul (qconj pi_t) pi_s) k) := by
  rw [inner4_symm, quat_left_adj, inner4_symm, ← p1_quat_mul_assoc]

-- Theorem 14: Prefix history pi_s cancels identically leaving pure interval holonomy delta
theorem p2_covariant_interval_path_cancellation (pi_s delta : QuatZ)
    (hunit : normSq pi_s = 1) :
    qmul (qconj pi_s) (qmul pi_s delta) = delta ∧
    qmul (qconj (qmul pi_s delta)) pi_s = qconj delta := by
  refine ⟨(p2_covariant_left_right_inverse_cancel pi_s delta).2.2 hunit |>.1, ?_⟩
  rw [(p1_quat_identity_and_conj_antihom pi_s delta).2.2.1,
      p1_quat_mul_assoc, quat_conj_mul_self, hunit, qmul_scalar_right, qscale_one]

-- Theorem 15: Single-value covariant un-transport and exact self-identity recovery
theorem p2_covariant_value_untransport_single (pi_t pi_s v : QuatZ) :
    qmul (qconj pi_t) (qmul pi_s v) = qmul (qmul (qconj pi_t) pi_s) v ∧
    (normSq pi_t = 1 → qmul (qconj pi_t) (qmul pi_t v) = v) := by
  refine ⟨(p1_quat_mul_assoc (qconj pi_t) pi_s v).symm, ?_⟩
  intro h1
  exact (p2_covariant_left_right_inverse_cancel pi_t v).2.2 h1 |>.1

def weightedTransportedSum : List (Int × QuatZ × QuatZ) → QuatZ
  | [] => qzero
  | (alpha_s, pi_s, v_s) :: rest =>
      qadd (qscale alpha_s (qmul pi_s v_s)) (weightedTransportedSum rest)

def weightedRelativeHolonomySum (pi_t : QuatZ) : List (Int × QuatZ × QuatZ) → QuatZ
  | [] => qzero
  | (alpha_s, pi_s, v_s) :: rest =>
      qadd (qscale alpha_s (qmul (qmul (qconj pi_t) pi_s) v_s))
           (weightedRelativeHolonomySum pi_t rest)

-- Theorem 16: Attention value un-transport distributes over arbitrary weighted sums
theorem p2_covariant_value_untransport_weighted_sum
    (pi_t : QuatZ) (kvs : List (Int × QuatZ × QuatZ)) :
    qmul (qconj pi_t) (weightedTransportedSum kvs) =
      weightedRelativeHolonomySum pi_t kvs := by
  induction kvs with
  | nil =>
    apply quat_ext <;> simp only [weightedTransportedSum, weightedRelativeHolonomySum,
                                  qmul, qzero, Int.mul_zero] <;> omega
  | cons hd tl ih =>
    cases hd with
    | mk alpha_s pv =>
      cases pv with
      | mk pi_s v_s =>
        simp only [weightedTransportedSum, weightedRelativeHolonomySum]
        rw [qmul_qadd_right, qmul_qscale_right, p1_quat_mul_assoc (qconj pi_t) pi_s v_s, ih]

-- ============================================================================
-- Pillar 3: SFU-Free Rational Cayley Safe-Hopf Clock & Hopf S^3 -> S^2 Fiber
-- ============================================================================

-- Theorem 17: Exact SFU-free rational Cayley unit-circle identity (homogeneous & d=1)
theorem p3_cayley_rational_unit_circle_identity (d n : Int) :
    (d * d - n * n) * (d * d - n * n) + (2 * d * n) * (2 * d * n) =
      (d * d + n * n) * (d * d + n * n) ∧
    (1 - n * n) * (1 - n * n) + (2 * n) * (2 * n) =
      (1 + n * n) * (1 + n * n) := by
  have hgen :
      (d * d - n * n) * (d * d - n * n) + (2 * d * n) * (2 * d * n) =
        (d * d + n * n) * (d * d + n * n) := by
    rw [Int.mul_assoc 2 d n, two_mul_eq_add (d * n)]
    simp only [Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.neg_neg,
               Int.mul_comm, Int.mul_left_comm]
    omega
  have h1 :
      (1 - n * n) * (1 - n * n) + (2 * n) * (2 * n) =
        (1 + n * n) * (1 + n * n) := by
    have h11 : (1 : Int) * 1 = 1 := rfl
    have h21 : (2 : Int) * 1 * n = 2 * n := by omega
    have hspec := hgen
    have hsub :
        (1 * 1 - n * n) * (1 * 1 - n * n) + (2 * 1 * n) * (2 * 1 * n) =
          (1 * 1 + n * n) * (1 * 1 + n * n) := by
      rw [h11, h21, two_mul_eq_add n]
      simp only [Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.neg_neg, Int.mul_one,
                 Int.mul_comm, Int.mul_left_comm]
      omega
    rw [h11, h21] at hsub
    exact hsub
  exact ⟨hgen, h1⟩

private theorem cayley_cross_diff_identity (d n m : Int) :
    (d * d - n * n) * (d * d + m * m) - (d * d - m * m) * (d * d + n * n) =
      2 * (d * d) * (m * m - n * n) := by
  rw [two_mul_eq_add (d * d)]
  simp only [Int.sub_eq_add_neg, Int.mul_add, Int.neg_add, Int.mul_neg, Int.neg_neg,
             Int.mul_comm, Int.mul_left_comm]
  omega

-- Theorem 18: Acute-cone Safe-Hopf clock strictly avoids -I and obeys exact positive floor
theorem p3_cayley_acute_cone_strict_pos_and_floor (n n_max d : Int)
    (hn0 : 0 ≤ n) (hn_le : n ≤ n_max) (hmax_lt : n_max < d) :
    0 < d * d + n * n ∧
    0 < d * d - n_max * n_max ∧
    0 < d * d - n * n ∧
    (d * d - n_max * n_max) * (d * d + n * n) ≤
      (d * d - n * n) * (d * d + n_max * n_max) := by
  have hnm0 : 0 ≤ n_max := by omega
  have hd_pos : 0 < d := by omega
  have hdd_pos : 0 < d * d := Int.mul_pos hd_pos hd_pos
  have hnn_nonneg : 0 ≤ n * n := int_sq_nonneg n
  have h_n_le_m : n * n ≤ n_max * n_max := int_sq_le_sq_of_nonneg n n_max hn0 hn_le
  have h_m_lt_d : n_max * n_max < d * d := int_sq_lt_sq_of_nonneg n_max d hnm0 hmax_lt
  have h2dd_pos : 0 < 2 * (d * d) := by omega
  have hdiff_nonneg : 0 ≤ n_max * n_max - n * n := by omega
  have hprod_nonneg : 0 ≤ 2 * (d * d) * (n_max * n_max - n * n) :=
    Int.mul_nonneg (Int.le_of_lt h2dd_pos) hdiff_nonneg
  have hcross := cayley_cross_diff_identity d n n_max
  refine ⟨by omega, by omega, by omega, by omega⟩

-- Theorem 19: Acute-cone Safe-Hopf schedule is strictly monotone (zero phase aliasing)
theorem p3_cayley_acute_strict_monotone_distinct (n1 n2 d : Int)
    (hn1 : 0 ≤ n1) (hlt : n1 < n2) (hn2 : n2 < d) :
    (d * d - n2 * n2) * (d * d + n1 * n1) <
      (d * d - n1 * n1) * (d * d + n2 * n2) := by
  have hd_pos : 0 < d := by omega
  have hdd_pos : 0 < d * d := Int.mul_pos hd_pos hd_pos
  have h2dd_pos : 0 < 2 * (d * d) := by omega
  have hsq_lt : n1 * n1 < n2 * n2 := int_sq_lt_sq_of_nonneg n1 n2 hn1 hlt
  have hdiff_pos : 0 < n2 * n2 - n1 * n1 := by omega
  have hprod_pos : 0 < 2 * (d * d) * (n2 * n2 - n1 * n1) := Int.mul_pos h2dd_pos hdiff_pos
  have hcross := cayley_cross_diff_identity d n1 n2
  omega

-- Theorem 20: W10-OBS-31 Obtuse/Antipodal Sign Flip vs Acute Self-Alignment Dichotomy
theorem p3_w10_obs31_obtuse_antipodal_sign_flip (n d vx vy : Int)
    (hd : 0 < d) (hobtuse : d < n) (hv : vx ≠ 0 ∨ vy ≠ 0) :
    d * d - n * n < 0 ∧
    vx * ((d * d - n * n) * vx - (2 * d * n) * vy) +
      vy * ((2 * d * n) * vx + (d * d - n * n) * vy) < 0 := by
  have hsq_lt : d * d < n * n := int_sq_lt_sq_of_nonneg d n (Int.le_of_lt hd) hobtuse
  have hv_pos : 0 < vx * vx + vy * vy := by
    cases hv with
    | inl hx =>
      have hx2 := int_sq_pos_of_ne_zero vx hx
      have hy2 := int_sq_nonneg vy
      omega
    | inr hy =>
      have hx2 := int_sq_nonneg vx
      have hy2 := int_sq_pos_of_ne_zero vy hy
      omega
  have hquad :
      vx * ((d * d - n * n) * vx - (2 * d * n) * vy) +
        vy * ((2 * d * n) * vx + (d * d - n * n) * vy) =
      (d * d - n * n) * (vx * vx + vy * vy) := by
    simp only [Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.mul_comm, Int.mul_left_comm]
    omega
  have hneg_prod : 0 < (n * n - d * d) * (vx * vx + vy * vy) :=
    Int.mul_pos (by omega) hv_pos
  have hflip : (d * d - n * n) * (vx * vx + vy * vy) =
      -((n * n - d * d) * (vx * vx + vy * vy)) := by
    simp only [Int.sub_mul]
    omega
  refine ⟨by omega, ?_⟩
  rw [hquad, hflip]
  omega

-- Hopf S^3 -> S^2 map h(q) = q * i * conj(q) matching fixed holosphere/geometry/hopf.py
def hopfHx (q : QuatZ) : Int := q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
def hopfHy (q : QuatZ) : Int := 2 * (q.x * q.y + q.w * q.z)
def hopfHz (q : QuatZ) : Int := 2 * (q.x * q.z - q.w * q.y)

def quatRightFiber (q : QuatZ) (a b : Int) : QuatZ :=
  qmul q ⟨a, b, 0, 0⟩

-- Theorem 21: Hopf S^3 -> S^2 conjugation formula, S^2 norm, and right-S^1 fiber invariance
theorem p3_hopf_s3_to_s2_norm_and_right_fiber_invariance (q : QuatZ) (a b : Int) :
    qmul (qmul q qi) (qconj q) = ⟨0, hopfHx q, hopfHy q, hopfHz q⟩ ∧
    hopfHx q * hopfHx q + hopfHy q * hopfHy q + hopfHz q * hopfHz q =
      normSq q * normSq q ∧
    hopfHx (quatRightFiber q a b) = (a * a + b * b) * hopfHx q ∧
    hopfHy (quatRightFiber q a b) = (a * a + b * b) * hopfHy q ∧
    hopfHz (quatRightFiber q a b) = (a * a + b * b) * hopfHz q := by
  have hform : ∀ r : QuatZ, qmul (qmul r qi) (qconj r) = ⟨0, hopfHx r, hopfHy r, hopfHz r⟩ := by
    intro r
    have hri : qmul r qi = ⟨-r.x, r.w, r.z, -r.y⟩ := by
      apply quat_ext <;> simp only [qmul, qi] <;> omega
    rw [hri]
    apply quat_ext
    · simp only [qmul, qconj, Int.sub_eq_add_neg, Int.mul_neg, Int.neg_neg, Int.mul_comm]
      omega
    · simp only [qmul, qconj, hopfHx, Int.sub_eq_add_neg, Int.mul_neg, Int.neg_mul, Int.neg_neg]
      omega
    · simp only [qmul, qconj, hopfHy, two_mul_eq_add (r.x * r.y + r.w * r.z),
                 Int.sub_eq_add_neg, Int.mul_neg, Int.neg_neg, Int.mul_comm]
      omega
    · simp only [qmul, qconj, hopfHz, Int.sub_eq_add_neg, Int.mul_neg,
                 Int.neg_neg, Int.mul_comm]
      omega
  have hnorm :
      hopfHx q * hopfHx q + hopfHy q * hopfHy q + hopfHz q * hopfHz q =
        normSq q * normSq q := by
    have hn1 : normSq (qmul (qmul q qi) (qconj q)) = normSq q * normSq q := by
      rw [p1_quat_euler_four_square_norm, p1_quat_euler_four_square_norm,
          (p1_quat_identity_and_conj_antihom q q).2.2.2]
      have hqi : normSq qi = 1 := by decide
      rw [hqi, Int.mul_one]
    rw [hform q] at hn1
    simp only [normSq, inner4, Int.zero_mul, Int.zero_add] at hn1
    exact hn1
  have hcomm_fi : qmul ⟨a, b, 0, 0⟩ qi = qmul qi ⟨a, b, 0, 0⟩ := by
    apply quat_ext <;>
      simp only [qmul, qi, Int.mul_zero, Int.zero_mul, Int.mul_one, Int.one_mul] <;>
      omega
  have hfib_quat :
      qmul (qmul (quatRightFiber q a b) qi) (qconj (quatRightFiber q a b)) =
        qscale (a * a + b * b) (qmul (qmul q qi) (qconj q)) := by
    unfold quatRightFiber
    rw [(p1_quat_identity_and_conj_antihom q ⟨a, b, 0, 0⟩).2.2.1,
        p1_quat_mul_assoc q ⟨a, b, 0, 0⟩ qi, hcomm_fi,
        ← p1_quat_mul_assoc q qi ⟨a, b, 0, 0⟩,
        p1_quat_mul_assoc (qmul q qi) ⟨a, b, 0, 0⟩ (qmul (qconj ⟨a, b, 0, 0⟩) (qconj q)),
        ← p1_quat_mul_assoc ⟨a, b, 0, 0⟩ (qconj ⟨a, b, 0, 0⟩) (qconj q),
        quat_mul_conj_self ⟨a, b, 0, 0⟩, qmul_scalar_left, qmul_qscale_right]
    have hnorm_ab : normSq ⟨a, b, 0, 0⟩ = a * a + b * b := by
      simp only [normSq, inner4, Int.zero_mul]
      omega
    rw [hnorm_ab]
  rw [hform (quatRightFiber q a b), hform q] at hfib_quat
  simp only [qscale, Int.mul_zero] at hfib_quat
  injection hfib_quat with _ hx hy hz
  exact ⟨hform q, hnorm, hx, hy, hz⟩

-- ============================================================================
-- Pillar 4: Beltrami [C, K] = 0 Helicity, Poloidal Bingham & Darboux Intertwining
-- ============================================================================

-- Homogeneous 2x2 SO(2) Cayley block matching holosphere/geometry/beltrami.py
def cayley2x2X (d n x0 x1 : Int) : Int := (d * d - n * n) * x0 + (2 * d * n) * x1
def cayley2x2Y (d n x0 x1 : Int) : Int := -(2 * d * n) * x0 + (d * d - n * n) * x1

private theorem brahmagupta_rot2 (c s x0 x1 : Int) :
    (c * x0 + s * x1) * (c * x0 + s * x1) + (-s * x0 + c * x1) * (-s * x0 + c * x1) =
      (c * c + s * s) * (x0 * x0 + x1 * x1) := by
  simp only [Int.mul_add, Int.mul_neg, Int.neg_neg, Int.mul_comm, Int.mul_left_comm]
  omega

-- Theorem 22: 2x2 skew-Cayley orthogonality (norm isometry) and positive SO(2) determinant
theorem p4_beltrami_cayley_2x2_orthogonality_and_det (d n x0 x1 : Int) :
    cayley2x2X d n x0 x1 * cayley2x2X d n x0 x1 +
      cayley2x2Y d n x0 x1 * cayley2x2Y d n x0 x1 =
      (d * d + n * n) * (d * d + n * n) * (x0 * x0 + x1 * x1) ∧
    (d * d - n * n) * (d * d - n * n) - (2 * d * n) * (-(2 * d * n)) =
      (d * d + n * n) * (d * d + n * n) := by
  have hiso :
      cayley2x2X d n x0 x1 * cayley2x2X d n x0 x1 +
        cayley2x2Y d n x0 x1 * cayley2x2Y d n x0 x1 =
        ((d * d - n * n) * (d * d - n * n) + (2 * d * n) * (2 * d * n)) *
          (x0 * x0 + x1 * x1) := by
    unfold cayley2x2X cayley2x2Y
    exact brahmagupta_rot2 (d * d - n * n) (2 * d * n) x0 x1
  rw [hiso, (p3_cayley_rational_unit_circle_identity d n).1]
  refine ⟨rfl, ?_⟩
  have hdet :
      (d * d - n * n) * (d * d - n * n) - (2 * d * n) * (-(2 * d * n)) =
        (d * d - n * n) * (d * d - n * n) + (2 * d * n) * (2 * d * n) := by
    simp only [Int.mul_neg]
    omega
  rw [hdet, (p3_cayley_rational_unit_circle_identity d n).1]

-- Paired 4-state chiral Beltrami curl diag(lam, lam, -lam, -lam) and Cayley mixer
def beltramiCurl4 (lam : Int) (v : Int × Int × Int × Int) : Int × Int × Int × Int :=
  match v with
  | (v0, v1, v2, v3) => (lam * v0, lam * v1, -lam * v2, -lam * v3)

def beltramiStep4 (d n : Int) (v : Int × Int × Int × Int) : Int × Int × Int × Int :=
  match v with
  | (v0, v1, v2, v3) =>
    (cayley2x2X d n v0 v1, cayley2x2Y d n v0 v1,
     cayley2x2X d n v2 v3, cayley2x2Y d n v2 v3)

def energy4 (v : Int × Int × Int × Int) : Int :=
  match v with
  | (v0, v1, v2, v3) => v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3

def helicity4 (lam : Int) (v : Int × Int × Int × Int) : Int :=
  match v with
  | (v0, v1, v2, v3) => lam * (v0 * v0 + v1 * v1 - v2 * v2 - v3 * v3)

def crossSectorSwap4 (v : Int × Int × Int × Int) : Int × Int × Int × Int :=
  match v with
  | (v0, v1, v2, v3) => (v0, v2, v1, v3)

private theorem rot2_linear_scale (c s mu v0 v1 : Int) :
    c * (mu * v0) + s * (mu * v1) = mu * (c * v0 + s * v1) ∧
    (-s) * (mu * v0) + c * (mu * v1) = mu * ((-s) * v0 + c * v1) := by
  refine ⟨?_, ?_⟩
  · simp only [Int.mul_add, Int.mul_left_comm]
  · simp only [Int.mul_add, Int.mul_neg, Int.mul_comm, Int.mul_left_comm]

-- Theorem 23: Paired Beltrami [C, K] = 0 exact commutation, energy & helicity conservation,
-- and O3.2 cross-sector helicity breach control
theorem p4_beltrami_paired_commutator_and_helicity_conservation
    (lam d n v0 v1 v2 v3 : Int) :
    let v := (v0, v1, v2, v3)
    let s := d * d + n * n
    beltramiStep4 d n (beltramiCurl4 lam v) = beltramiCurl4 lam (beltramiStep4 d n v) ∧
    energy4 (beltramiStep4 d n v) = s * s * energy4 v ∧
    helicity4 lam (beltramiStep4 d n v) = s * s * helicity4 lam v ∧
    energy4 (crossSectorSwap4 v) = energy4 v ∧
    helicity4 lam (crossSectorSwap4 v) - helicity4 lam v =
      2 * lam * (v2 * v2 - v1 * v1) := by
  have h1 := (p4_beltrami_cayley_2x2_orthogonality_and_det d n v0 v1).1
  have h2 := (p4_beltrami_cayley_2x2_orthogonality_and_det d n v2 v3).1
  have hc_pos := rot2_linear_scale (d * d - n * n) (2 * d * n) lam v0 v1
  have hc_neg := rot2_linear_scale (d * d - n * n) (2 * d * n) (-lam) v2 v3
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [beltramiStep4, beltramiCurl4, cayley2x2X, cayley2x2Y]
    rw [hc_pos.1, hc_pos.2, hc_neg.1, hc_neg.2]
  · simp only [energy4, beltramiStep4]
    have hsum :
        cayley2x2X d n v0 v1 * cayley2x2X d n v0 v1 +
          cayley2x2Y d n v0 v1 * cayley2x2Y d n v0 v1 +
          cayley2x2X d n v2 v3 * cayley2x2X d n v2 v3 +
          cayley2x2Y d n v2 v3 * cayley2x2Y d n v2 v3 =
        (cayley2x2X d n v0 v1 * cayley2x2X d n v0 v1 +
          cayley2x2Y d n v0 v1 * cayley2x2Y d n v0 v1) +
        (cayley2x2X d n v2 v3 * cayley2x2X d n v2 v3 +
          cayley2x2Y d n v2 v3 * cayley2x2Y d n v2 v3) := by omega
    rw [hsum, h1, h2, ← Int.mul_add]
    have hsplit : v0 * v0 + v1 * v1 + (v2 * v2 + v3 * v3) =
        v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 := by omega
    rw [hsplit]
  · simp only [helicity4, beltramiStep4]
    have hdiff :
        cayley2x2X d n v0 v1 * cayley2x2X d n v0 v1 +
          cayley2x2Y d n v0 v1 * cayley2x2Y d n v0 v1 -
          cayley2x2X d n v2 v3 * cayley2x2X d n v2 v3 -
          cayley2x2Y d n v2 v3 * cayley2x2Y d n v2 v3 =
        (cayley2x2X d n v0 v1 * cayley2x2X d n v0 v1 +
          cayley2x2Y d n v0 v1 * cayley2x2Y d n v0 v1) -
        (cayley2x2X d n v2 v3 * cayley2x2X d n v2 v3 +
          cayley2x2Y d n v2 v3 * cayley2x2Y d n v2 v3) := by omega
    rw [hdiff, h1, h2, ← Int.mul_sub]
    have hsplit : v0 * v0 + v1 * v1 - (v2 * v2 + v3 * v3) =
        v0 * v0 + v1 * v1 - v2 * v2 - v3 * v3 := by omega
    rw [hsplit, ← Int.mul_assoc, Int.mul_comm lam ((d * d + n * n) * (d * d + n * n)),
        Int.mul_assoc]
  · simp only [energy4, crossSectorSwap4]
    omega
  · simp only [helicity4, crossSectorSwap4]
    rw [two_mul_eq_add lam]
    simp only [Int.add_mul, Int.mul_add, Int.mul_sub]
    omega

private theorem rayleigh_diff_id (lam Gp Gn Ep En : Int) :
    (lam * (Gp * Ep - Gn * En)) * (Ep + En) -
      (lam * (Ep - En)) * (Gp * Ep + Gn * En) =
    2 * (lam * ((Gp - Gn) * (Ep * En))) := by
  rw [two_mul_eq_add]
  simp only [Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.neg_add, Int.neg_neg,
             Int.mul_comm, Int.mul_left_comm]
  omega

-- Theorem 24: O3.5 Symmetric commuting residual alters the Rayleigh helicity-to-energy ratio
theorem p4_beltrami_symmetric_residual_rayleigh_drift
    (lam g_pos g_neg e_pos e_neg : Int)
    (hlam : 0 < lam) (he_pos : 0 < e_pos) (he_neg : 0 < e_neg)
    (hg : 0 < g_pos * g_pos - g_neg * g_neg) :
    helicity4 lam (e_pos, 0, e_neg, 0) *
        (g_pos * g_pos * (e_pos * e_pos) + g_neg * g_neg * (e_neg * e_neg)) <
      (lam * (g_pos * g_pos * (e_pos * e_pos) - g_neg * g_neg * (e_neg * e_neg))) *
        energy4 (e_pos, 0, e_neg, 0) := by
  simp only [helicity4, energy4, Int.zero_mul, Int.add_zero, Int.sub_zero]
  have he1 : 0 < e_pos * e_pos := Int.mul_pos he_pos he_pos
  have he2 : 0 < e_neg * e_neg := Int.mul_pos he_neg he_neg
  have heprod : 0 < (e_pos * e_pos) * (e_neg * e_neg) := Int.mul_pos he1 he2
  have hdiff_pos :
      0 < 2 * (lam * ((g_pos * g_pos - g_neg * g_neg) * ((e_pos * e_pos) * (e_neg * e_neg)))) :=
    Int.mul_pos (by omega) (Int.mul_pos hlam (Int.mul_pos hg heprod))
  have hid := rayleigh_diff_id lam (g_pos * g_pos) (g_neg * g_neg) (e_pos * e_pos) (e_neg * e_neg)
  omega

-- Finite-dimensional sum/norm/matrix helpers over Int for Bingham & Darboux
def sumN : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumN n f + f n

def normSqN (n : Nat) (u : Nat → Int) : Int :=
  sumN n (fun i => u i * u i)

private theorem sumN_congr (n : Nat) (f g : Nat → Int)
    (h : ∀ i, i < n → f i = g i) : sumN n f = sumN n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [sumN]
    have h1 := ih (fun i hi => h i (Nat.lt_succ_of_lt hi))
    have h2 := h k (Nat.lt_succ_self k)
    omega

private theorem sumN_zero (n : Nat) : sumN n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [sumN]; omega

private theorem sumN_add (n : Nat) (f g : Nat → Int) :
    sumN n (fun i => f i + g i) = sumN n f + sumN n g := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [sumN]; omega

private theorem sumN_mul_left (n : Nat) (c : Int) (f : Nat → Int) :
    sumN n (fun i => c * f i) = c * sumN n f := by
  induction n with
  | zero => simp only [sumN, Int.mul_zero]
  | succ k ih => simp only [sumN, ih, Int.mul_add]

private theorem sumN_mul_right (n : Nat) (f : Nat → Int) (c : Int) :
    sumN n (fun i => f i * c) = sumN n f * c := by
  induction n with
  | zero => simp only [sumN, Int.zero_mul]
  | succ k ih => simp only [sumN, ih, Int.add_mul]

private theorem sumN_swap (m n : Nat) (f : Nat → Nat → Int) :
    sumN m (fun i => sumN n (fun j => f i j)) =
      sumN n (fun j => sumN m (fun i => f i j)) := by
  induction m with
  | zero =>
    simp only [sumN]
    exact (sumN_zero n).symm
  | succ k ih =>
    simp only [sumN, ih, ← sumN_add]

private theorem sumN_delta_right (n j : Nat) (hj : j < n) (g : Nat → Int) :
    sumN n (fun r => g r * (if r = j then 1 else 0)) = g j := by
  induction n with
  | zero => omega
  | succ k ih =>
    simp only [sumN]
    if hkj : k = j then
      subst hkj
      have hz : sumN k (fun r => g r * (if r = k then 1 else 0)) = 0 := by
        rw [sumN_congr k _ (fun _ => 0)]
        · exact sumN_zero k
        · intro r hr
          have hne : r ≠ k := by omega
          simp only [hne, ite_false, Int.mul_zero]
      simp only [hz, ite_true, Int.mul_one, Int.zero_add]
    else
      have hlt : j < k := by omega
      rw [ih hlt]
      simp only [hkj, ite_false, Int.mul_zero, Int.add_zero]

private theorem sumN_delta_left (m i : Nat) (hi : i < m) (g : Nat → Int) :
    sumN m (fun r => (if i = r then 1 else 0) * g r) = g i := by
  induction m with
  | zero => omega
  | succ k ih =>
    simp only [sumN]
    if hik : k = i then
      subst hik
      have hz : sumN k (fun r => (if k = r then 1 else 0) * g r) = 0 := by
        rw [sumN_congr k _ (fun _ => 0)]
        · exact sumN_zero k
        · intro r hr
          have hne : k ≠ r := by omega
          simp only [hne, ite_false, Int.zero_mul]
      rw [hz]
      simp only [ite_true, Int.one_mul, Int.zero_add]
    else
      have hne_ik : i ≠ k := by omega
      have hlt : i < k := by omega
      rw [ih hlt]
      simp only [hne_ik, ite_false, Int.zero_mul, Int.add_zero]

private theorem normSqN_nonneg (n : Nat) (u : Nat → Int) : 0 ≤ normSqN n u := by
  unfold normSqN
  induction n with
  | zero => simp only [sumN]; omega
  | succ k ih =>
    simp only [sumN]
    have hsq := int_sq_nonneg (u k)
    omega

-- Poloidal Bingham vector-proximal gate matching holosphere/geometry/bingham.py
def binghamGatePol (n : Nat) (tauSq : Int) (u_pol : Nat → Int) (i : Nat) : Int :=
  if normSqN n u_pol ≤ tauSq then 0 else u_pol i

-- Theorem 25: Poloidal Bingham below-yield stagnant-lid zeroing & toroidal preservation
theorem p4_poloidal_bingham_hodge_stagnant_lid
    (n m : Nat) (tauSq : Int) (u_pol u_tor : Nat → Int)
    (hlid : normSqN n u_pol ≤ tauSq) :
    (∀ i, i < n → binghamGatePol n tauSq u_pol i = 0) ∧
    normSqN n (binghamGatePol n tauSq u_pol) = 0 ∧
    normSqN n (binghamGatePol n tauSq u_pol) + normSqN m u_tor = normSqN m u_tor ∧
    normSqN m u_tor ≤ normSqN n u_pol + normSqN m u_tor := by
  have hzero_vec : ∀ i, binghamGatePol n tauSq u_pol i = 0 := by
    intro i
    simp only [binghamGatePol, hlid, ite_true]
  have hnorm0 : normSqN n (binghamGatePol n tauSq u_pol) = 0 := by
    unfold normSqN
    rw [sumN_congr n _ (fun _ => 0)]
    · exact sumN_zero n
    · intro i _
      rw [hzero_vec i, Int.zero_mul]
  have hpol_nn := normSqN_nonneg n u_pol
  refine ⟨fun i _ => hzero_vec i, hnorm0, by omega, by omega⟩

-- Rectangular Darboux partner operators L1 = A^T A + c I_n and L2 = A A^T + c I_m
def matVecRect (_m n : Nat) (A : Nat → Nat → Int) (x : Nat → Int) (i : Nat) : Int :=
  sumN n (fun j => A i j * x j)

def matMulRect (_m k _n : Nat) (A B : Nat → Nat → Int) (i j : Nat) : Int :=
  sumN k (fun r => A i r * B r j)

def darbouxL1 (m _n : Nat) (A : Nat → Nat → Int) (c : Int) (i j : Nat) : Int :=
  sumN m (fun r => A r i * A r j) + (if i = j then c else 0)

def darbouxL2 (_m n : Nat) (A : Nat → Nat → Int) (c : Int) (i j : Nat) : Int :=
  sumN n (fun r => A i r * A j r) + (if i = j then c else 0)

-- Theorem 26: Rectangular Darboux intertwining A L1 = L2 A and dark-mode kernel collision
theorem p4_darboux_isospectral_intertwining_and_dark_kernel
    (m n : Nat) (A : Nat → Nat → Int) (c : Int) (x k : Nat → Int)
    (hker : ∀ i, i < m → matVecRect m n A k i = 0) :
    (∀ i j, i < m → j < n →
      matMulRect m n n A (darbouxL1 m n A c) i j =
        matMulRect m m n (darbouxL2 m n A c) A i j) ∧
    (∀ i, i < m → matVecRect m n A (fun j => x j + k j) i = matVecRect m n A x i) := by
  refine ⟨?_, ?_⟩
  · intro i j hi hj
    unfold matMulRect darbouxL1 darbouxL2
    have hlhs :
        sumN n (fun r => A i r * (sumN m (fun s => A s r * A s j) + (if r = j then c else 0))) =
          sumN n (fun r => sumN m (fun s => A i r * (A s r * A s j))) + A i j * c := by
      rw [sumN_congr n _ (fun r => A i r * sumN m (fun s => A s r * A s j) +
            A i r * (if r = j then c else 0))]
      · rw [sumN_add]
        have h1 : sumN n (fun r => A i r * sumN m (fun s => A s r * A s j)) =
            sumN n (fun r => sumN m (fun s => A i r * (A s r * A s j))) := by
          apply sumN_congr
          intro r _
          exact (sumN_mul_left m (A i r) _).symm
        have h2 : sumN n (fun r => A i r * (if r = j then c else 0)) = A i j * c := by
          have hc : sumN n (fun r => A i r * (if r = j then c else 0)) =
              sumN n (fun r => (A i r * (if r = j then 1 else 0)) * c) := by
            apply sumN_congr
            intro r _
            if hrj : r = j then
              simp only [hrj, ite_true, Int.mul_one]
            else
              simp only [hrj, ite_false, Int.mul_zero, Int.zero_mul]
          rw [hc, sumN_mul_right, sumN_delta_right n j hj]
        rw [h1, h2]
      · intro r _
        exact Int.mul_add (A i r) _ _
    have hrhs :
        sumN m (fun s => (sumN n (fun r => A i r * A s r) + (if i = s then c else 0)) * A s j) =
          sumN m (fun s => sumN n (fun r => (A i r * A s r) * A s j)) + c * A i j := by
      rw [sumN_congr m _ (fun s => sumN n (fun r => A i r * A s r) * A s j +
            (if i = s then c else 0) * A s j)]
      · rw [sumN_add]
        have h1 : sumN m (fun s => sumN n (fun r => A i r * A s r) * A s j) =
            sumN m (fun s => sumN n (fun r => (A i r * A s r) * A s j)) := by
          apply sumN_congr
          intro s _
          exact (sumN_mul_right n _ (A s j)).symm
        have h2 : sumN m (fun s => (if i = s then c else 0) * A s j) = c * A i j := by
          have hc : sumN m (fun s => (if i = s then c else 0) * A s j) =
              sumN m (fun s => c * ((if i = s then 1 else 0) * A s j)) := by
            apply sumN_congr
            intro s _
            if his : i = s then
              simp only [his, ite_true, Int.one_mul]
            else
              simp only [his, ite_false, Int.zero_mul, Int.mul_zero]
          rw [hc, sumN_mul_left, sumN_delta_left m i hi]
        rw [h1, h2]
      · intro s _
        exact Int.add_mul _ _ (A s j)
    rw [hlhs, hrhs, sumN_swap n m]
    have hassoc :
        sumN m (fun s => sumN n (fun r => A i r * (A s r * A s j))) =
          sumN m (fun s => sumN n (fun r => (A i r * A s r) * A s j)) := by
      apply sumN_congr
      intro s _
      apply sumN_congr
      intro r _
      exact (Int.mul_assoc (A i r) (A s r) (A s j)).symm
    rw [hassoc, Int.mul_comm (A i j) c]
  · intro i hi
    unfold matVecRect
    rw [sumN_congr n _ (fun j => A i j * x j + A i j * k j)]
    · rw [sumN_add]
      have hk0 := hker i hi
      unfold matVecRect at hk0
      omega
    · intro j _
      exact Int.mul_add (A i j) (x j) (k j)

private theorem rot2_input_adjoint_id (c s g0 g1 x0 x1 : Int) :
    g0 * (c * x0 + s * x1) + g1 * (-s * x0 + c * x1) =
      (c * g0 - s * g1) * x0 + (s * g0 + c * g1) * x1 := by
  simp only [Int.sub_eq_add_neg, Int.mul_add,
             Int.mul_neg, Int.mul_comm, Int.mul_left_comm]
  omega

private theorem cayley2x2_dy0_num (w x0 x1 : Int) :
    ((-2 * w) * x0 + 2 * x1) * (1 + w * w) -
      ((1 - w * w) * x0 + (2 * w) * x1) * (2 * w) =
    2 * ((-2 * w) * x0 + (1 - w * w) * x1) := by
  have hm2 : (-2 : Int) * w = -(2 * w) := by omega
  rw [hm2, two_mul_eq_add w, two_mul_eq_add x1, two_mul_eq_add]
  simp only [Int.sub_eq_add_neg, Int.mul_add,
             Int.neg_add, Int.mul_neg, Int.neg_neg,
             Int.mul_one, Int.mul_comm, Int.mul_left_comm]
  omega

private theorem cayley2x2_dy1_num (w x0 x1 : Int) :
    (-2 * x0 - (2 * w) * x1) * (1 + w * w) -
      ((-2 * w) * x0 + (1 - w * w) * x1) * (2 * w) =
    -(2 * ((1 - w * w) * x0 + (2 * w) * x1)) := by
  have hm2w : (-2 : Int) * w = -(2 * w) := by omega
  have hm2x : (-2 : Int) * x0 = -(x0 + x0) := by omega
  rw [hm2w, hm2x, two_mul_eq_add w, two_mul_eq_add]
  simp only [Int.sub_eq_add_neg, Int.mul_add,
             Int.neg_add, Int.mul_neg, Int.neg_neg,
             Int.mul_one, Int.mul_comm, Int.mul_left_comm]
  omega

-- Theorem 29: Closed-form 2x2 Cayley SO(d) input & parameter reverse-mode adjoint
-- matching Cayley2x2AutogradFn: (1 + w^2) <g, dy/dw> = 2 * (g0 * y1 - g1 * y0)
theorem p4_cayley2x2_closed_form_adjoint (d n w g0 g1 x0 x1 : Int) :
    g0 * cayley2x2X d n x0 x1 + g1 * cayley2x2Y d n x0 x1 =
      ((d * d - n * n) * g0 - (2 * d * n) * g1) * x0 +
        ((2 * d * n) * g0 + (d * d - n * n) * g1) * x1 ∧
    g0 * (((-2 * w) * x0 + 2 * x1) * (1 + w * w) - cayley2x2X 1 w x0 x1 * (2 * w)) +
      g1 * ((-2 * x0 - (2 * w) * x1) * (1 + w * w) - cayley2x2Y 1 w x0 x1 * (2 * w)) =
      2 * (g0 * cayley2x2Y 1 w x0 x1 - g1 * cayley2x2X 1 w x0 x1) := by
  have hin :
      g0 * cayley2x2X d n x0 x1 + g1 * cayley2x2Y d n x0 x1 =
        ((d * d - n * n) * g0 - (2 * d * n) * g1) * x0 +
          ((2 * d * n) * g0 + (d * d - n * n) * g1) * x1 := by
    unfold cayley2x2X cayley2x2Y
    exact rot2_input_adjoint_id (d * d - n * n) (2 * d * n) g0 g1 x0 x1
  refine ⟨hin, ?_⟩
  have hx1 : cayley2x2X 1 w x0 x1 = (1 - w * w) * x0 + (2 * w) * x1 := by
    unfold cayley2x2X
    have h11 : (1 : Int) * 1 = 1 := rfl
    have h21 : (2 : Int) * 1 * w = 2 * w := by omega
    rw [h11, h21]
  have hy1 : cayley2x2Y 1 w x0 x1 = (-2 * w) * x0 + (1 - w * w) * x1 := by
    unfold cayley2x2Y
    have h11 : (1 : Int) * 1 = 1 := rfl
    have h21 : -((2 : Int) * 1 * w) = -2 * w := by omega
    rw [h11, h21]
  rw [hx1, hy1, cayley2x2_dy0_num w x0 x1, cayley2x2_dy1_num w x0 x1]
  have set_or_lin : ∀ a b : Int, g0 * (2 * a) + g1 * -(2 * b) = 2 * (g0 * a - g1 * b) := by
    intro a b
    rw [two_mul_eq_add a, two_mul_eq_add b, two_mul_eq_add (g0 * a - g1 * b)]
    simp only [Int.mul_add, Int.mul_neg]
    omega
  exact set_or_lin ((-2 * w) * x0 + (1 - w * w) * x1) ((1 - w * w) * x0 + (2 * w) * x1)

-- ============================================================================
-- Pillar 5: Weft Composed-DeltaW C_row Gate, W10-OBS-20 & Atomic Rollback
-- ============================================================================

structure Mat2x2 where
  m00 : Int
  m01 : Int
  m10 : Int
  m11 : Int
  deriving DecidableEq, Repr

def addMat2x2 (A B : Mat2x2) : Mat2x2 :=
  ⟨A.m00 + B.m00, A.m01 + B.m01, A.m10 + B.m10, A.m11 + B.m11⟩

def scaleMat2x2 (lam : Int) (A : Mat2x2) : Mat2x2 :=
  ⟨lam * A.m00, lam * A.m01, lam * A.m10, lam * A.m11⟩

def row0Sq2x2 (A : Mat2x2) : Int := A.m00 * A.m00 + A.m01 * A.m01
def row1Sq2x2 (A : Mat2x2) : Int := A.m10 * A.m10 + A.m11 * A.m11

def maxRowSq2x2 (A : Mat2x2) : Int :=
  if row0Sq2x2 A ≤ row1Sq2x2 A then row1Sq2x2 A else row0Sq2x2 A

def frobSq2x2 (A : Mat2x2) : Int :=
  row0Sq2x2 A + row1Sq2x2 A

def weftTriGate (c_num c_den : Int) (D_prev dW : Mat2x2) (receipt holdout : Bool) : Bool :=
  let total := addMat2x2 D_prev dW
  decide (0 < frobSq2x2 total) &&
    decide (c_den * maxRowSq2x2 total ≤ c_num * frobSq2x2 total) &&
    receipt &&
    holdout

def obs20Patch1 (a : Int) : Mat2x2 := ⟨a, 0, a, 0⟩
def obs20Patch2 (a : Int) : Mat2x2 := ⟨a, 0, -a, 0⟩

-- Theorem 27: W10-OBS-20 Multi-patch cancellation attack passes per-patch C_max = 3/5
-- yet strictly breaches composed C_row and is 100% blocked by weftTriGate
theorem p5_weft_w10_obs20_cancellation_attack_blocked (a : Int) (ha : a ≠ 0) :
    5 * maxRowSq2x2 (obs20Patch1 a) ≤ 3 * frobSq2x2 (obs20Patch1 a) ∧
    5 * maxRowSq2x2 (obs20Patch2 a) ≤ 3 * frobSq2x2 (obs20Patch2 a) ∧
    3 * frobSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) <
      5 * maxRowSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) ∧
    weftTriGate 3 5 (obs20Patch1 a) (obs20Patch2 a) true true = false := by
  have ha2_pos : 0 < a * a := int_sq_pos_of_ne_zero a ha
  have hp1_max : maxRowSq2x2 (obs20Patch1 a) = a * a := by
    simp only [maxRowSq2x2, row0Sq2x2, row1Sq2x2, obs20Patch1, Int.zero_mul, Int.add_zero,
               Int.le_refl, ite_true]
  have hp1_frob : frobSq2x2 (obs20Patch1 a) = 2 * (a * a) := by
    simp only [frobSq2x2, row0Sq2x2, row1Sq2x2, obs20Patch1, Int.zero_mul]
    omega
  have hp2_max : maxRowSq2x2 (obs20Patch2 a) = a * a := by
    simp only [maxRowSq2x2, row0Sq2x2, row1Sq2x2, obs20Patch2, Int.neg_mul_neg,
               Int.zero_mul, Int.add_zero, Int.le_refl, ite_true]
  have hp2_frob : frobSq2x2 (obs20Patch2 a) = 2 * (a * a) := by
    simp only [frobSq2x2, row0Sq2x2, row1Sq2x2, obs20Patch2, Int.neg_mul_neg, Int.zero_mul]
    omega
  have hsum_r0 : row0Sq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) = 4 * (a * a) := by
    simp only [row0Sq2x2, addMat2x2, obs20Patch1, obs20Patch2, Int.add_mul, Int.mul_add, Int.zero_mul]
    omega
  have hsum_r1 : row1Sq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) = 0 := by
    simp only [row1Sq2x2, addMat2x2, obs20Patch1, obs20Patch2]
    have hz : a + -a = 0 := by omega
    rw [hz]
    decide
  have hsum_max : maxRowSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) = 4 * (a * a) := by
    unfold maxRowSq2x2
    rw [hsum_r0, hsum_r1]
    have hnot : ¬ (4 * (a * a) ≤ 0) := by omega
    simp only [hnot, ite_false]
  have hsum_frob : frobSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) = 4 * (a * a) := by
    unfold frobSq2x2
    rw [hsum_r0, hsum_r1]
    omega
  have hbreach :
      3 * frobSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) <
        5 * maxRowSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) := by
    rw [hsum_frob, hsum_max]
    omega
  refine ⟨by rw [hp1_max, hp1_frob]; omega,
          by rw [hp2_max, hp2_frob]; omega,
          hbreach,
          ?_⟩
  unfold weftTriGate
  have hnot_gate :
      ¬ (5 * maxRowSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a)) ≤
          3 * frobSq2x2 (addMat2x2 (obs20Patch1 a) (obs20Patch2 a))) := by
    omega
  simp only [hnot_gate, decide_false, Bool.and_false, Bool.false_and]

-- Theorem 28: Tri-conjunctive commit gate accepts iff composed C_row, receipt, and holdout hold
theorem p5_weft_tri_conjunctive_commit_soundness
    (c_num c_den : Int) (D_prev dW : Mat2x2) (receipt holdout : Bool) :
    weftTriGate c_num c_den D_prev dW receipt holdout = true ↔
      (0 < frobSq2x2 (addMat2x2 D_prev dW) ∧
       c_den * maxRowSq2x2 (addMat2x2 D_prev dW) ≤ c_num * frobSq2x2 (addMat2x2 D_prev dW) ∧
       receipt = true ∧
       holdout = true) := by
  unfold weftTriGate
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · intro h
    exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩
  · intro h
    exact ⟨⟨⟨h.1, h.2.1⟩, h.2.2.1⟩, h.2.2.2⟩

-- Theorem 29: Isotropic multiplicative sleep scaling preserves C_row gate iff pre-sleep passed
theorem p5_weft_isotropic_sleep_preserves_crow_gate
    (lam c_num c_den : Int) (M : Mat2x2) (hlam : lam ≠ 0) :
    c_den * maxRowSq2x2 (scaleMat2x2 lam M) ≤ c_num * frobSq2x2 (scaleMat2x2 lam M) ↔
      c_den * maxRowSq2x2 M ≤ c_num * frobSq2x2 M := by
  have hl2_pos : 0 < lam * lam := int_sq_pos_of_ne_zero lam hlam
  have hr0 : row0Sq2x2 (scaleMat2x2 lam M) = (lam * lam) * row0Sq2x2 M := by
    simp only [row0Sq2x2, scaleMat2x2, Int.mul_add, Int.mul_comm, Int.mul_left_comm]
  have hr1 : row1Sq2x2 (scaleMat2x2 lam M) = (lam * lam) * row1Sq2x2 M := by
    simp only [row1Sq2x2, scaleMat2x2, Int.mul_add, Int.mul_comm, Int.mul_left_comm]
  have hmax : maxRowSq2x2 (scaleMat2x2 lam M) = (lam * lam) * maxRowSq2x2 M := by
    unfold maxRowSq2x2
    rw [hr0, hr1]
    have hiff := int_mul_le_iff_of_pos_left (lam * lam) (row0Sq2x2 M) (row1Sq2x2 M) hl2_pos
    if hle : row0Sq2x2 M ≤ row1Sq2x2 M then
      have hle' : (lam * lam) * row0Sq2x2 M ≤ (lam * lam) * row1Sq2x2 M := hiff.mpr hle
      simp only [hle, hle', ite_true]
    else
      have hle' : ¬ ((lam * lam) * row0Sq2x2 M ≤ (lam * lam) * row1Sq2x2 M) :=
        fun hc => hle (hiff.mp hc)
      simp only [hle, hle', ite_false]
  have hfrob : frobSq2x2 (scaleMat2x2 lam M) = (lam * lam) * frobSq2x2 M := by
    unfold frobSq2x2
    rw [hr0, hr1, ← Int.mul_add]
  rw [hmax, hfrob]
  have hcomm1 : c_den * ((lam * lam) * maxRowSq2x2 M) =
      (lam * lam) * (c_den * maxRowSq2x2 M) := by
    rw [← Int.mul_assoc, Int.mul_comm c_den (lam * lam), Int.mul_assoc]
  have hcomm2 : c_num * ((lam * lam) * frobSq2x2 M) =
      (lam * lam) * (c_num * frobSq2x2 M) := by
    rw [← Int.mul_assoc, Int.mul_comm c_num (lam * lam), Int.mul_assoc]
  rw [hcomm1, hcomm2]
  exact int_mul_le_iff_of_pos_left (lam * lam) (c_den * maxRowSq2x2 M) (c_num * frobSq2x2 M) hl2_pos

def commitPatches (W : Nat → Nat → Int) : List (Nat → Nat → Int) → Nat → Nat → Int
  | [] => W
  | dW :: rest => commitPatches (fun i j => W i j + dW i j) rest

def rollbackPatches (W : Nat → Nat → Int) : List (Nat → Nat → Int) → Nat → Nat → Int
  | [] => W
  | dW :: rest => fun i j => rollbackPatches W rest i j - dW i j

-- Theorem 30: Exact single-patch and multi-patch LIFO atomic rollback involution
theorem p5_weft_atomic_rollback_exact_involution
    (W dW : Nat → Nat → Int) (patches : List (Nat → Nat → Int)) (i j : Nat) :
    (W i j + dW i j) - dW i j = W i j ∧
    rollbackPatches (commitPatches W patches) patches i j = W i j := by
  refine ⟨by omega, ?_⟩
  induction patches generalizing W with
  | nil => rfl
  | cons p rest ih =>
    simp only [commitPatches, rollbackPatches]
    rw [ih (fun r c => W r c + p r c)]
    omega

private theorem quat_foldl_left_mul (us : List QuatZ) (a b : QuatZ) :
    us.foldl qmul (qmul a b) = qmul a (us.foldl qmul b) := by
  induction us generalizing b with
  | nil => rfl
  | cons u rest ih =>
    simp only [List.foldl]
    rw [p1_quat_mul_assoc a b u]
    exact ih (qmul b u)

private theorem qmul_qsub_left (p q r : QuatZ) :
    qmul (qsub p q) r = qsub (qmul p r) (qmul q r) := by
  apply quat_ext <;> simp only [qmul, qsub, Int.sub_mul] <;> omega

private theorem eq_zero_of_sq_eq_zero (a : Int) (h : a * a = 0) : a = 0 := by
  cases Int.mul_eq_zero.mp h with
  | inl h0 => exact h0
  | inr h0 => exact h0

private theorem normSq_pos_of_ne_qzero (p : QuatZ) (hp : p ≠ qzero) :
    0 < normSq p := by
  have hw := int_sq_nonneg p.w
  have hx := int_sq_nonneg p.x
  have hy := int_sq_nonneg p.y
  have hz := int_sq_nonneg p.z
  have hge : 0 ≤ normSq p := by unfold normSq inner4; omega
  if hpos : 0 < normSq p then
    exact hpos
  else
    have heq0 : normSq p = 0 := by omega
    unfold normSq inner4 at heq0
    have hw0 : p.w * p.w = 0 := by omega
    have hx0 : p.x * p.x = 0 := by omega
    have hy0 : p.y * p.y = 0 := by omega
    have hz0 : p.z * p.z = 0 := by omega
    have hpeq : p = qzero :=
      quat_ext (eq_zero_of_sq_eq_zero p.w hw0) (eq_zero_of_sq_eq_zero p.x hx0)
               (eq_zero_of_sq_eq_zero p.y hy0) (eq_zero_of_sq_eq_zero p.z hz0)
    exact absurd hpeq hp

def restoreHolonomy16B (pi_ckpt _pi_cur : QuatZ) : QuatZ :=
  pi_ckpt

-- Theorem 34: Infinite-Half-Life SU(2) Holonomy Re-Poisoning Law & 16B Checkpoint Cure
theorem p5_su2_holonomy_repoisoning_and_16b_rollback
    (pi_t0 e_err : QuatZ) (clean_future : List QuatZ)
    (h0 : normSq pi_t0 = 1) (herr_norm : normSq e_err = 1) (herr_ne : e_err ≠ qone)
    (hclean : ∀ u, u ∈ clean_future → normSq u = 1) :
    let delta := clean_future.foldl qmul qone
    let pi_clean := clean_future.foldl qmul pi_t0
    let pi_stale := clean_future.foldl qmul (qmul pi_t0 e_err)
    let pi_restored := clean_future.foldl qmul (restoreHolonomy16B pi_t0 (qmul pi_t0 e_err))
    normSq pi_stale = 1 ∧
    qmul (qconj pi_t0) pi_clean = delta ∧
    qmul (qconj pi_t0) pi_stale = qmul e_err delta ∧
    normSq (qsub (qmul (qconj pi_t0) pi_stale) delta) = normSq (qsub e_err qone) ∧
    0 < normSq (qsub e_err qone) ∧
    pi_stale ≠ pi_clean ∧
    pi_restored = pi_clean ∧
    qmul (qconj pi_t0) pi_restored = delta := by
  have hinit_stale : normSq (qmul pi_t0 e_err) = 1 := by
    rw [p1_quat_euler_four_square_norm, h0, herr_norm]
    rfl
  have hstale_unit : normSq (clean_future.foldl qmul (qmul pi_t0 e_err)) = 1 :=
    p1_quat_prefix_scan_unit_norm clean_future (qmul pi_t0 e_err) hinit_stale hclean
  have hqone_norm : normSq qone = 1 := by decide
  have hdelta_unit : normSq (clean_future.foldl qmul qone) = 1 :=
    p1_quat_prefix_scan_unit_norm clean_future qone hqone_norm hclean
  have hclean_eq :
      clean_future.foldl qmul pi_t0 = qmul pi_t0 (clean_future.foldl qmul qone) := by
    have h1 := quat_foldl_left_mul clean_future pi_t0 qone
    rw [(p1_quat_identity_and_conj_antihom pi_t0 qone).2.1] at h1
    exact h1
  have hstale_eq :
      clean_future.foldl qmul (qmul pi_t0 e_err) =
        qmul pi_t0 (qmul e_err (clean_future.foldl qmul qone)) := by
    have h1 := quat_foldl_left_mul clean_future (qmul pi_t0 e_err) qone
    rw [(p1_quat_identity_and_conj_antihom (qmul pi_t0 e_err) qone).2.1] at h1
    rw [h1, p1_quat_mul_assoc]
  have hrel_clean :
      qmul (qconj pi_t0) (clean_future.foldl qmul pi_t0) = clean_future.foldl qmul qone := by
    rw [hclean_eq, (p2_covariant_left_right_inverse_cancel pi_t0 _).2.2 h0 |>.1]
  have hrel_stale :
      qmul (qconj pi_t0) (clean_future.foldl qmul (qmul pi_t0 e_err)) =
        qmul e_err (clean_future.foldl qmul qone) := by
    rw [hstale_eq, (p2_covariant_left_right_inverse_cancel pi_t0 _).2.2 h0 |>.1]
  have hsub_err :
      qsub (qmul e_err (clean_future.foldl qmul qone)) (clean_future.foldl qmul qone) =
        qmul (qsub e_err qone) (clean_future.foldl qmul qone) := by
    rw [qmul_qsub_left, (p1_quat_identity_and_conj_antihom (clean_future.foldl qmul qone) qone).1]
  have herr_const :
      normSq (qsub (qmul (qconj pi_t0) (clean_future.foldl qmul (qmul pi_t0 e_err)))
                   (clean_future.foldl qmul qone)) =
        normSq (qsub e_err qone) := by
    rw [hrel_stale, hsub_err, p1_quat_euler_four_square_norm, hdelta_unit, Int.mul_one]
  have hdiff_ne : qsub e_err qone ≠ qzero := by
    intro hzero
    apply herr_ne
    apply quat_ext
    · have hw := congrArg QuatZ.w hzero; simp only [qsub, qzero] at hw; omega
    · have hx := congrArg QuatZ.x hzero; simp only [qsub, qzero] at hx; omega
    · have hy := congrArg QuatZ.y hzero; simp only [qsub, qzero] at hy; omega
    · have hz := congrArg QuatZ.z hzero; simp only [qsub, qzero] at hz; omega
  have herr_pos : 0 < normSq (qsub e_err qone) := normSq_pos_of_ne_qzero (qsub e_err qone) hdiff_ne
  have hstale_ne_clean :
      clean_future.foldl qmul (qmul pi_t0 e_err) ≠ clean_future.foldl qmul pi_t0 := by
    intro heq
    have hrel_eq :
        qmul (qconj pi_t0) (clean_future.foldl qmul (qmul pi_t0 e_err)) =
          clean_future.foldl qmul qone := by
      rw [heq, hrel_clean]
    have hzero_norm : normSq (qsub e_err qone) = 0 := by
      rw [← herr_const, hrel_eq, qsub_self]
      decide
    omega
  refine ⟨hstale_unit, hrel_clean, hrel_stale, herr_const, herr_pos, hstale_ne_clean, rfl, hrel_clean⟩

-- Asymmetric safety recovery gate vs symmetric trust-region deadlock (optifab / SiliSense)
def posPart (x : Int) : Int :=
  if 0 ≤ x then x else 0

def asymExcessSq (n : Nat) (u u0 : Nat → Int) : Int :=
  sumN n (fun i => posPart (u i - u0 i) * posPart (u i - u0 i))

def symDistSq (n : Nat) (u u0 : Nat → Int) : Int :=
  sumN n (fun i => (u i - u0 i) * (u i - u0 i))

def gateAsymmetric (n : Nat) (u u0 : Nat → Int) (R : Int) : Bool :=
  decide (asymExcessSq n u u0 ≤ R * R)

def gateSymmetricSigned (n : Nat) (u u0 : Nat → Int) (R : Int) : Bool :=
  decide (0 ≤ R) && decide (symDistSq n u u0 ≤ R * R)

private theorem sumN_nonneg (n : Nat) (f : Nat → Int)
    (hnn : ∀ i, i < n → 0 ≤ f i) : 0 ≤ sumN n f := by
  induction n with
  | zero => simp only [sumN]; omega
  | succ k ih =>
    simp only [sumN]
    have h1 := ih (fun j hj => hnn j (Nat.lt_succ_of_lt hj))
    have h2 := hnn k (Nat.lt_succ_self k)
    omega

private theorem sumN_pos_of_nonneg_of_pos (n i0 : Nat) (f : Nat → Int)
    (hi0 : i0 < n) (hnn : ∀ i, i < n → 0 ≤ f i) (hpos : 0 < f i0) :
    0 < sumN n f := by
  induction n with
  | zero => omega
  | succ k ih =>
    simp only [sumN]
    if hik : k = i0 then
      subst hik
      have hprev := sumN_nonneg k f (fun j hj => hnn j (Nat.lt_succ_of_lt hj))
      omega
    else
      have hlt : i0 < k := by omega
      have h1 := ih hlt (fun j hj => hnn j (Nat.lt_succ_of_lt hj))
      have h2 := hnn k (Nat.lt_succ_self k)
      omega

-- Theorem 35: gateAsymmetric admits strict safety recovery at R <= 0 while symmetric gate deadlocks
theorem p5_gate_asymmetric_recovery_vs_symmetric_deadlock
    (n i0 : Nat) (u u0 : Nat → Int) (R : Int)
    (hi0 : i0 < n) (hdesc : ∀ i, i < n → u i ≤ u0 i) (hstrict : u i0 < u0 i0) :
    asymExcessSq n u u0 = 0 ∧
    gateAsymmetric n u u0 R = true ∧
    0 < symDistSq n u u0 ∧
    (R ≤ 0 → gateSymmetricSigned n u u0 R = false) := by
  have hasym0 : asymExcessSq n u u0 = 0 := by
    unfold asymExcessSq
    rw [sumN_congr n _ (fun _ => 0)]
    · exact sumN_zero n
    · intro i hi
      have hle := hdesc i hi
      unfold posPart
      if hge0 : 0 ≤ u i - u0 i then
        have hz : u i - u0 i = 0 := by omega
        rw [hz]
        decide
      else
        simp only [hge0, ite_false, Int.zero_mul]
  have hgate_ok : gateAsymmetric n u u0 R = true := by
    unfold gateAsymmetric
    rw [hasym0]
    simp only [decide_eq_true_eq]
    exact int_sq_nonneg R
  have hsym_pos : 0 < symDistSq n u u0 := by
    unfold symDistSq
    apply sumN_pos_of_nonneg_of_pos n i0 _ hi0
    · intro i _
      exact int_sq_nonneg (u i - u0 i)
    · apply int_sq_pos_of_ne_zero
      omega
  refine ⟨hasym0, hgate_ok, hsym_pos, ?_⟩
  intro hRle
  unfold gateSymmetricSigned
  if hR0 : 0 ≤ R then
    have hReq0 : R = 0 := by omega
    rw [hReq0]
    have hnot : ¬ (symDistSq n u u0 ≤ 0 * 0) := by omega
    simp only [hnot, decide_false, Bool.and_false]
  else
    simp only [hR0, decide_false, Bool.false_and]

def rightOrigA (i j : Nat) : Int :=
  match i, j with
  | 0, 0 => 2
  | 0, 1 => -2
  | _, _ => 0

def rightQuantA (i j : Nat) : Int :=
  match i, j with
  | 0, 0 => 2
  | 0, 1 => -1
  | _, _ => 0

def rightNullX : Nat → Int
  | 0 => 1
  | 1 => 1
  | _ => 0

-- Theorem 36: Asymmetric left-factor quantization Q(B) A x = 0 has zero null-space drift
-- whereas right-factor quantization Q(A) x != 0 shatters ker(A)
theorem p5_asymmetric_left_factor_zero_nullspace_drift
    (p m n : Nat) (Q_B A : Nat → Nat → Int) (x : Nat → Int)
    (hker : ∀ r, r < m → matVecRect m n A x r = 0) :
    (∀ i, i < p → matVecRect p m Q_B (matVecRect m n A x) i = 0) ∧
    (∀ i, i < p → matVecRect p n (matMulRect p m n Q_B A) x i = 0) ∧
    matVecRect 1 2 rightOrigA rightNullX 0 = 0 ∧
    matVecRect 1 2 rightQuantA rightNullX 0 ≠ 0 := by
  have hstaged : ∀ i, i < p → matVecRect p m Q_B (matVecRect m n A x) i = 0 := by
    intro i _
    change sumN m (fun r => Q_B i r * matVecRect m n A x r) = 0
    rw [sumN_congr m _ (fun _ => 0)]
    · exact sumN_zero m
    · intro r hr
      rw [hker r hr, Int.mul_zero]
  have hfused : ∀ i, i < p → matVecRect p n (matMulRect p m n Q_B A) x i = 0 := by
    intro i hi
    have hswap_eq :
        matVecRect p n (matMulRect p m n Q_B A) x i =
          matVecRect p m Q_B (matVecRect m n A x) i := by
      unfold matVecRect matMulRect
      have hl :
          sumN n (fun j => sumN m (fun r => Q_B i r * A r j) * x j) =
            sumN n (fun j => sumN m (fun r => Q_B i r * (A r j * x j))) := by
        apply sumN_congr
        intro j _
        rw [← sumN_mul_right]
        apply sumN_congr
        intro r _
        exact Int.mul_assoc (Q_B i r) (A r j) (x j)
      have hr :
          sumN m (fun r => Q_B i r * sumN n (fun j => A r j * x j)) =
            sumN m (fun r => sumN n (fun j => Q_B i r * (A r j * x j))) := by
        apply sumN_congr
        intro r _
        exact (sumN_mul_left n (Q_B i r) _).symm
      rw [hl, hr, sumN_swap n m]
    rw [hswap_eq]
    exact hstaged i hi
  refine ⟨hstaged, hfused, by decide, by decide⟩

end HoloSphereInvariants

#print axioms HoloSphereInvariants.p1_quat_mul_assoc
#print axioms HoloSphereInvariants.p1_quat_identity_and_conj_antihom
#print axioms HoloSphereInvariants.p1_quat_noncommutative_su2_holonomy_witness
#print axioms HoloSphereInvariants.p1_quat_euler_four_square_norm
#print axioms HoloSphereInvariants.p1_quat_prefix_scan_unit_norm
#print axioms HoloSphereInvariants.p1_quat_chunked_scan_eq_sequential
#print axioms HoloSphereInvariants.p1_quat_hillis_steele_8lane_eq_sequential
#print axioms HoloSphereInvariants.p1_quat_bilinear_two_sided_adjoint
#print axioms HoloSphereInvariants.p1_quat_scan_step_backward_factorization
#print axioms HoloSphereInvariants.p1_quat_scan_suffix_sum_adjoint_collapse
#print axioms HoloSphereInvariants.p1_cayley_su2_unit_norm_and_acute_positivity
#print axioms HoloSphereInvariants.p1_local_4d_s3_zero_cross_bank_bleed
#print axioms HoloSphereInvariants.p2_covariant_left_right_inverse_cancel
#print axioms HoloSphereInvariants.p2_covariant_left_transport_isometry
#print axioms HoloSphereInvariants.p2_covariant_qk_score_relative_holonomy
#print axioms HoloSphereInvariants.p2_covariant_interval_path_cancellation
#print axioms HoloSphereInvariants.p2_covariant_value_untransport_single
#print axioms HoloSphereInvariants.p2_covariant_value_untransport_weighted_sum
#print axioms HoloSphereInvariants.p3_cayley_rational_unit_circle_identity
#print axioms HoloSphereInvariants.p3_cayley_acute_cone_strict_pos_and_floor
#print axioms HoloSphereInvariants.p3_cayley_acute_strict_monotone_distinct
#print axioms HoloSphereInvariants.p3_w10_obs31_obtuse_antipodal_sign_flip
#print axioms HoloSphereInvariants.p3_hopf_s3_to_s2_norm_and_right_fiber_invariance
#print axioms HoloSphereInvariants.p4_beltrami_cayley_2x2_orthogonality_and_det
#print axioms HoloSphereInvariants.p4_beltrami_paired_commutator_and_helicity_conservation
#print axioms HoloSphereInvariants.p4_beltrami_symmetric_residual_rayleigh_drift
#print axioms HoloSphereInvariants.p4_poloidal_bingham_hodge_stagnant_lid
#print axioms HoloSphereInvariants.p4_darboux_isospectral_intertwining_and_dark_kernel
#print axioms HoloSphereInvariants.p4_cayley2x2_closed_form_adjoint
#print axioms HoloSphereInvariants.p5_weft_w10_obs20_cancellation_attack_blocked
#print axioms HoloSphereInvariants.p5_weft_tri_conjunctive_commit_soundness
#print axioms HoloSphereInvariants.p5_weft_isotropic_sleep_preserves_crow_gate
#print axioms HoloSphereInvariants.p5_weft_atomic_rollback_exact_involution
#print axioms HoloSphereInvariants.p5_su2_holonomy_repoisoning_and_16b_rollback
#print axioms HoloSphereInvariants.p5_gate_asymmetric_recovery_vs_symmetric_deadlock
#print axioms HoloSphereInvariants.p5_asymmetric_left_factor_zero_nullspace_drift
