-- Project HoloSphere (PR #2 Candidate / Track S116 CellOp Co-Design)
-- Pure-Init Lean 4.36.0 Formalization of Theorems #37-#40 + #40b (silikernels#2 Co-Design)
-- Zero external imports, zero Mathlib, zero sorry, zero Classical.choice,
-- zero Lean.ofReduceBool (no native_decide), zero compiler warnings.
--
-- Public Theorems proved:
--   #37  p1_blockh4_involutory_and_int8_bounds       (OBS-ORDA-01 / OBS-ORDA-04)
--   #38  p2_cayley_unit_ball_so3_involution_and_even (W10-OBS-48 / OBS-ORDA-09)
--   #39  p3_cayley_exact_inverse_reversibility       (W10-OBS-52 / OBS-ORDA-13)
--   #40  p4_hadamard256_byte_orthogonality           (W10-OBS-51 / OBS-ORDA-12)
--   #40b p4b_bent_phase_preserves_orthogonality      (W10-OBS-55 / OBS-ORDA-16)

namespace HoloSphereCellOpInvariants

-- ============================================================================
-- 0. Exact Integer Vector, Quaternion, and Matrix Definitions (Pure Init)
-- ============================================================================

/-- 4-lane integer vector for Block-H4 signed 4-bit nibble groups and int8 outputs. -/
structure Vec4Z where
  u0 : Int
  u1 : Int
  u2 : Int
  u3 : Int
  deriving DecidableEq, Repr

def vec4Scale (s : Int) (v : Vec4Z) : Vec4Z :=
  ⟨s * v.u0, s * v.u1, s * v.u2, s * v.u3⟩

def vec4NormSq (v : Vec4Z) : Int :=
  v.u0 * v.u0 + v.u1 * v.u1 + v.u2 * v.u2 + v.u3 * v.u3

/-- Exact 4x4 Sylvester-Hadamard butterfly H_4 matching s116_a16_hamming_ste.py:41-46
    and holosphere_megakernel_sm86.cu:
      r0 = u0 + u1 + u2 + u3
      r1 = u0 - u1 + u2 - u3
      r2 = u0 + u1 - u2 - u3
      r3 = u0 - u1 - u2 + u3 -/
def blockH4 (u : Vec4Z) : Vec4Z :=
  ⟨u.u0 + u.u1 + u.u2 + u.u3,
   u.u0 - u.u1 + u.u2 - u.u3,
   u.u0 + u.u1 - u.u2 - u.u3,
   u.u0 - u.u1 - u.u2 + u.u3⟩

private theorem vec4_ext {a b : Vec4Z}
    (h0 : a.u0 = b.u0) (h1 : a.u1 = b.u1) (h2 : a.u2 = b.u2) (h3 : a.u3 = b.u3) :
    a = b := by
  cases a
  cases b
  simp only at h0 h1 h2 h3
  subst h0 h1 h2 h3
  rfl

/-- 3D integer / homogeneous-rational Lie generator vector w in su(2) ~ R^3. -/
structure Vec3 where
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

def vec3Neg (w : Vec3) : Vec3 :=
  ⟨-w.x, -w.y, -w.z⟩

def vec3Dot (u v : Vec3) : Int :=
  u.x * v.x + u.y * v.y + u.z * v.z

def vec3NormSq (w : Vec3) : Int :=
  vec3Dot w w

/-- Hamilton quaternion over Int matching HoloSphereInvariants.QuatZ (w, x, y, z). -/
structure QuatZ where
  w : Int
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

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

def normSq (p : QuatZ) : Int :=
  p.w * p.w + p.x * p.x + p.y * p.y + p.z * p.z

private theorem quat_ext {p q : QuatZ}
    (hw : p.w = q.w) (hx : p.x = q.x) (hy : p.y = q.y) (hz : p.z = q.z) : p = q := by
  cases p
  cases q
  simp only at hw hx hy hz
  subst hw hx hy hz
  rfl

private theorem qmul_assoc (p q r : QuatZ) :
    qmul (qmul p q) r = qmul p (qmul q r) := by
  apply quat_ext <;>
    simp only [qmul, Int.sub_eq_add_neg, Int.add_mul, Int.mul_add,
               Int.neg_add, Int.mul_neg, Int.neg_mul, Int.neg_neg, Int.mul_assoc] <;>
    omega

private theorem qmul_scalar_left (s : Int) (v : QuatZ) :
    qmul ⟨s, 0, 0, 0⟩ v = qscale s v := by
  apply quat_ext <;> simp only [qmul, qscale, Int.zero_mul] <;> omega

private theorem quat_conj_mul_self (p : QuatZ) :
    qmul (qconj p) p = ⟨normSq p, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, Int.sub_eq_add_neg,
               Int.mul_neg, Int.neg_neg, Int.mul_comm] <;>
    omega

private theorem quat_mul_conj_self (p : QuatZ) :
    qmul p (qconj p) = ⟨normSq p, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, Int.mul_neg, Int.mul_comm] <;>
    omega

/-- Unnormalized rational Cayley numerator for w in R^3 (d = 1):
    cayleyNum(w) = (1 - ||w||^2, 2*w), with strict denominator 1 + ||w||^2 > 0. -/
def cayleyNum (w : Vec3) : QuatZ :=
  ⟨1 - vec3NormSq w, 2 * w.x, 2 * w.y, 2 * w.z⟩

def cayleyDen (w : Vec3) : Int :=
  1 + vec3NormSq w

/-- Homogeneous rational Cayley numerator for rational generator w / d in Q^3:
    cayleyHomNum(d, w) = (d^2 - ||w||^2, 2*d*w), avoiding all IEEE-754 floats. -/
def cayleyHomNum (d : Int) (w : Vec3) : QuatZ :=
  ⟨d * d - vec3NormSq w, 2 * d * w.x, 2 * d * w.y, 2 * d * w.z⟩

def cayleyHomDen (d : Int) (w : Vec3) : Int :=
  d * d + vec3NormSq w

/-- Exact 3x3 homogeneous SO(3) quadratic adjoint matrix R(q) matching
    su2_to_so3_adjoint_9d in s116_h16e_fmnmx_ste.py:267-277. -/
structure Mat3Z where
  r00 : Int
  r01 : Int
  r02 : Int
  r10 : Int
  r11 : Int
  r12 : Int
  r20 : Int
  r21 : Int
  r22 : Int
  deriving DecidableEq, Repr

def so3Adj (q : QuatZ) : Mat3Z :=
  ⟨q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
   2 * (q.x * q.y - q.w * q.z),
   2 * (q.x * q.z + q.w * q.y),
   2 * (q.x * q.y + q.w * q.z),
   q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
   2 * (q.y * q.z - q.w * q.x),
   2 * (q.x * q.z - q.w * q.y),
   2 * (q.y * q.z + q.w * q.x),
   q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z⟩

def mat3Vec (m : Mat3Z) (v : Vec3) : Vec3 :=
  ⟨m.r00 * v.x + m.r01 * v.y + m.r02 * v.z,
   m.r10 * v.x + m.r11 * v.y + m.r12 * v.z,
   m.r20 * v.x + m.r21 * v.y + m.r22 * v.z⟩

def so3Act (q : QuatZ) (v : Vec3) : Vec3 :=
  mat3Vec (so3Adj q) v

private theorem mat3_ext {a b : Mat3Z}
    (h00 : a.r00 = b.r00) (h01 : a.r01 = b.r01) (h02 : a.r02 = b.r02)
    (h10 : a.r10 = b.r10) (h11 : a.r11 = b.r11) (h12 : a.r12 = b.r12)
    (h20 : a.r20 = b.r20) (h21 : a.r21 = b.r21) (h22 : a.r22 = b.r22) : a = b := by
  cases a
  cases b
  simp only at h00 h01 h02 h10 h11 h12 h20 h21 h22
  subst h00 h01 h02 h10 h11 h12 h20 h21 h22
  rfl

private theorem two_mul_all (x : Int) : 2 * x = x + x := by
  omega

private theorem int_sq_nonneg (a : Int) : 0 ≤ a * a := by
  if h : 0 ≤ a then
    exact Int.mul_nonneg h h
  else
    have hna : 0 ≤ -a := by omega
    have hmul : 0 ≤ (-a) * (-a) := Int.mul_nonneg hna hna
    rw [Int.neg_mul_neg] at hmul
    exact hmul

private theorem vec3NormSq_nonneg (w : Vec3) : 0 ≤ vec3NormSq w := by
  have hx := int_sq_nonneg w.x
  have hy := int_sq_nonneg w.y
  have hz := int_sq_nonneg w.z
  simp only [vec3NormSq, vec3Dot]
  omega

private theorem cayleyNum_neg (w : Vec3) :
    cayleyNum (vec3Neg w) = qconj (cayleyNum w) := by
  apply quat_ext
  · simp only [cayleyNum, vec3NormSq, vec3Dot, vec3Neg, qconj, Int.neg_mul_neg]
  · simp only [cayleyNum, vec3Neg, qconj, Int.mul_neg]
  · simp only [cayleyNum, vec3Neg, qconj, Int.mul_neg]
  · simp only [cayleyNum, vec3Neg, qconj, Int.mul_neg]

private theorem cayleyHomNum_neg (d : Int) (w : Vec3) :
    cayleyHomNum d (vec3Neg w) = qconj (cayleyHomNum d w) := by
  apply quat_ext
  · simp only [cayleyHomNum, vec3NormSq, vec3Dot, vec3Neg, qconj, Int.neg_mul_neg]
  · simp only [cayleyHomNum, vec3Neg, qconj, Int.mul_neg]
  · simp only [cayleyHomNum, vec3Neg, qconj, Int.mul_neg]
  · simp only [cayleyHomNum, vec3Neg, qconj, Int.mul_neg]

private theorem cayleyHom_norm_identity (d : Int) (w : Vec3) :
    normSq (cayleyHomNum d w) = cayleyHomDen d w * cayleyHomDen d w := by
  have h4dw :
      (2 * d * w.x) * (2 * d * w.x) + (2 * d * w.y) * (2 * d * w.y) +
        (2 * d * w.z) * (2 * d * w.z) =
        4 * (d * d) * vec3NormSq w := by
    simp only [vec3NormSq, vec3Dot]
    rw [Int.mul_assoc 2 d w.x, Int.mul_assoc 2 d w.y, Int.mul_assoc 2 d w.z,
        two_mul_all (d * w.x), two_mul_all (d * w.y), two_mul_all (d * w.z)]
    have h4 : (4 : Int) * (d * d) = d * d + d * d + d * d + d * d := by omega
    rw [h4]
    simp only [Int.mul_add, Int.mul_comm, Int.mul_left_comm]
    omega
  have hsq :
      (d * d - vec3NormSq w) * (d * d - vec3NormSq w) + 4 * (d * d) * vec3NormSq w =
        (d * d + vec3NormSq w) * (d * d + vec3NormSq w) := by
    have h4 : (4 : Int) * (d * d) = d * d + d * d + d * d + d * d := by omega
    rw [h4]
    simp only [Int.sub_eq_add_neg, Int.add_mul, Int.mul_add, Int.mul_neg,
               Int.neg_neg, Int.mul_comm, Int.mul_left_comm]
    omega
  simp only [normSq, cayleyHomNum, cayleyHomDen]
  omega

private theorem cayleyNum_norm_identity (w : Vec3) :
    normSq (cayleyNum w) = cayleyDen w * cayleyDen w := by
  have h1 : cayleyNum w = cayleyHomNum 1 w := by
    apply quat_ext <;> simp only [cayleyNum, cayleyHomNum, Int.mul_one]
  have hd : cayleyDen w = cayleyHomDen 1 w := by
    simp only [cayleyDen, cayleyHomDen, Int.mul_one]
  rw [h1, hd]
  exact cayleyHom_norm_identity 1 w

-- ============================================================================
-- Theorem #37: Block-H4 Involutory Identity & Tight Signed-Nibble int8 Bounds
-- ============================================================================

/-- Theorem #37 (OBS-ORDA-01 / OBS-ORDA-04 — Block-H4 Involutory Identity & int8 Bounds):
    For any 4-tuple of signed 4-bit nibbles u = (u0, u1, u2, u3) with -8 <= u_i <= 7:
      1. Exact involutory Hadamard law: H_4 (H_4 u) = 4 * u (justifying the exact 0.5 * scale
         in SiliquantA16BlockH4STE so (0.5 * H_4)^2 = I_4),
      2. Exact Parseval norm law: ||H_4 u||^2 = 4 * ||u||^2,
      3. Tight componentwise signed int8 bounds: every component r_k of H_4 u satisfies
         -32 <= r_k <= 30 (strictly inside int8 [-128, 127]),
      4. Exact boundary witnesses: (-8,-8,-8,-8) -> (-32,0,0,0), (7,7,7,7) -> (28,0,0,0),
         and alternating parity nibble (7,-8,7,-8) -> (-2,30,0,0) attaining +30. -/
theorem p1_blockh4_involutory_and_int8_bounds (u : Vec4Z)
    (h0_lo : -8 ≤ u.u0) (h0_hi : u.u0 ≤ 7)
    (h1_lo : -8 ≤ u.u1) (h1_hi : u.u1 ≤ 7)
    (h2_lo : -8 ≤ u.u2) (h2_hi : u.u2 ≤ 7)
    (h3_lo : -8 ≤ u.u3) (h3_hi : u.u3 ≤ 7) :
    blockH4 (blockH4 u) = vec4Scale 4 u ∧
    vec4NormSq (blockH4 u) = 4 * vec4NormSq u ∧
    (-32 ≤ (blockH4 u).u0 ∧ (blockH4 u).u0 ≤ 30) ∧
    (-32 ≤ (blockH4 u).u1 ∧ (blockH4 u).u1 ≤ 30) ∧
    (-32 ≤ (blockH4 u).u2 ∧ (blockH4 u).u2 ≤ 30) ∧
    (-32 ≤ (blockH4 u).u3 ∧ (blockH4 u).u3 ≤ 30) ∧
    blockH4 ⟨-8, -8, -8, -8⟩ = ⟨-32, 0, 0, 0⟩ ∧
    blockH4 ⟨7, 7, 7, 7⟩ = ⟨28, 0, 0, 0⟩ ∧
    blockH4 ⟨7, -8, 7, -8⟩ = ⟨-2, 30, 0, 0⟩ := by
  have hinv : blockH4 (blockH4 u) = vec4Scale 4 u := by
    apply vec4_ext <;> simp only [blockH4, vec4Scale] <;> omega
  have hnorm : vec4NormSq (blockH4 u) = 4 * vec4NormSq u := by
    have h4 : (4 : Int) * vec4NormSq u =
        vec4NormSq u + vec4NormSq u + vec4NormSq u + vec4NormSq u := by omega
    rw [h4]
    simp only [vec4NormSq, blockH4, Int.sub_eq_add_neg, Int.mul_add,
               Int.neg_add, Int.mul_neg, Int.neg_neg, Int.mul_comm]
    omega
  refine ⟨hinv, hnorm, ?_, ?_, ?_, ?_, by decide, by decide, by decide⟩
  · refine ⟨?_, ?_⟩ <;> simp only [blockH4] <;> omega
  · refine ⟨?_, ?_⟩ <;> simp only [blockH4] <;> omega
  · refine ⟨?_, ?_⟩ <;> simp only [blockH4] <;> omega
  · refine ⟨?_, ?_⟩ <;> simp only [blockH4] <;> omega

-- ============================================================================
-- Theorem #38: Single-Step Cayley Unit-Ball SO(3) Involution & +-I Evenness
-- ============================================================================

/-- Theorem #38 (W10-OBS-48 / OBS-ORDA-09 — Cayley Unit-Ball SO(3) Involution & +-I Readout):
    For any integer/rational generator w : Vec3, homogeneous denominator d : Int,
    quaternion q : QuatZ, and vector v : Vec3:
      1. Unit-norm generator ||w||^2 = 1 gives exact scalar part 0 in cayleyNum(w) = (0, 2*w)
         and exact quaternion square cayleyNum(w)^2 = (-4, 0, 0, 0) (pure 180-deg SO(3) involution
         at finite ||w|| = 1 with zero singularity and positive denominator cayleyDen(w) = 2 > 0),
      2. Homogeneous rational unit sphere ||w||^2 = d^2 (for any rational generator w/d in Q^3)
         likewise gives (cayleyHomNum d w).w = 0 and cayleyHomNum(d, w)^2 = (-4*d^4, 0, 0, 0),
      3. Exact +-I-invariant readout: so3Adj(-q) = so3Adj(q) identically for all q : QuatZ
         (quotienting SU(2)/{+-I} = SO(3), covering A_5 = 2I/{+-I} inside ||w|| <= 1 in 1 step),
      4. Exact conjugation bridge: q * (0, v) * conj(q) = (0, so3Act q v). -/
theorem p2_cayley_unit_ball_so3_involution_and_even
    (w : Vec3) (d : Int) (q : QuatZ) (v : Vec3) :
    (vec3NormSq w = 1 →
      (cayleyNum w).w = 0 ∧ qmul (cayleyNum w) (cayleyNum w) = ⟨-4, 0, 0, 0⟩) ∧
    (vec3NormSq w = d * d →
      (cayleyHomNum d w).w = 0 ∧
      qmul (cayleyHomNum d w) (cayleyHomNum d w) = ⟨-(4 * (d * d) * (d * d)), 0, 0, 0⟩) ∧
    so3Adj (qneg q) = so3Adj q ∧
    qmul (qmul q ⟨0, v.x, v.y, v.z⟩) (qconj q) =
      ⟨0, (so3Act q v).x, (so3Act q v).y, (so3Act q v).z⟩ := by
  have hunithom : ∀ d0 : Int, vec3NormSq w = d0 * d0 →
      (cayleyHomNum d0 w).w = 0 ∧
      qmul (cayleyHomNum d0 w) (cayleyHomNum d0 w) = ⟨-(4 * (d0 * d0) * (d0 * d0)), 0, 0, 0⟩ := by
    intro d0 hwd
    have hscalar : (cayleyHomNum d0 w).w = 0 := by
      simp only [cayleyHomNum, hwd]
      omega
    have hconj_pure : qconj (cayleyHomNum d0 w) = qneg (cayleyHomNum d0 w) := by
      apply quat_ext <;> simp only [qconj, qneg, cayleyHomNum, hwd] <;> omega
    have hmul_neg :
        qmul (cayleyHomNum d0 w) (qneg (cayleyHomNum d0 w)) =
          qneg (qmul (cayleyHomNum d0 w) (cayleyHomNum d0 w)) := by
      apply quat_ext <;> simp only [qmul, qneg, Int.mul_neg, Int.sub_eq_add_neg, Int.neg_add, Int.neg_neg] <;> omega
    have hself := quat_mul_conj_self (cayleyHomNum d0 w)
    rw [hconj_pure, hmul_neg, cayleyHom_norm_identity d0 w] at hself
    have hden : cayleyHomDen d0 w = 2 * (d0 * d0) := by
      simp only [cayleyHomDen, hwd]
      omega
    have hden_sq : cayleyHomDen d0 w * cayleyHomDen d0 w = 4 * (d0 * d0) * (d0 * d0) := by
      rw [hden, two_mul_all (d0 * d0)]
      have h4 : (4 : Int) * (d0 * d0) = d0 * d0 + d0 * d0 + d0 * d0 + d0 * d0 := by omega
      rw [h4]
      simp only [Int.add_mul, Int.mul_add]
      omega
    rw [hden_sq] at hself
    have hsq : qmul (cayleyHomNum d0 w) (cayleyHomNum d0 w) =
        ⟨-(4 * (d0 * d0) * (d0 * d0)), 0, 0, 0⟩ := by
      have hneg2 := congrArg qneg hself
      have hinv_neg : ∀ p : QuatZ, qneg (qneg p) = p := by
        intro p
        apply quat_ext <;> simp only [qneg, Int.neg_neg]
      rw [hinv_neg] at hneg2
      rw [hneg2]
      apply quat_ext <;> simp only [qneg] <;> omega
    exact ⟨hscalar, hsq⟩
  have hunit1 : vec3NormSq w = 1 →
      (cayleyNum w).w = 0 ∧ qmul (cayleyNum w) (cayleyNum w) = ⟨-4, 0, 0, 0⟩ := by
    intro hw1
    have h1 : cayleyNum w = cayleyHomNum 1 w := by
      apply quat_ext <;> simp only [cayleyNum, cayleyHomNum, Int.mul_one]
    have hw11 : vec3NormSq w = 1 * 1 := by omega
    have hres := hunithom 1 hw11
    rw [← h1] at hres
    exact hres
  have heven : so3Adj (qneg q) = so3Adj q := by
    apply mat3_ext <;> simp only [so3Adj, qneg, Int.neg_mul_neg]
  have hconj :
      qmul (qmul q ⟨0, v.x, v.y, v.z⟩) (qconj q) =
        ⟨0, (so3Act q v).x, (so3Act q v).y, (so3Act q v).z⟩ := by
    apply quat_ext
    · simp only [qmul, qconj, Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg,
                 Int.neg_neg, Int.mul_zero, Int.mul_comm, Int.mul_left_comm]
      omega
    · simp only [so3Act, mat3Vec, so3Adj, two_mul_all]
      simp only [qmul, qconj, Int.sub_eq_add_neg, Int.mul_add, Int.neg_add,
                 Int.mul_neg, Int.neg_neg, Int.mul_zero,
                 Int.mul_comm, Int.mul_left_comm]
      omega
    · simp only [so3Act, mat3Vec, so3Adj, two_mul_all]
      simp only [qmul, qconj, Int.sub_eq_add_neg, Int.mul_add, Int.neg_add,
                 Int.mul_neg, Int.neg_neg, Int.mul_zero,
                 Int.mul_comm, Int.mul_left_comm]
      omega
    · simp only [so3Act, mat3Vec, so3Adj, two_mul_all]
      simp only [qmul, qconj, Int.sub_eq_add_neg, Int.mul_add, Int.neg_add,
                 Int.mul_neg, Int.neg_neg, Int.mul_zero,
                 Int.mul_comm, Int.mul_left_comm]
      omega
  exact ⟨hunit1, hunithom d, heven, hconj⟩

-- ============================================================================
-- Theorem #39: Exact Algebraic Cayley Inverse & O(1)-VRAM Backward Reversibility
-- ============================================================================

/-- Theorem #39 (W10-OBS-52 / OBS-ORDA-13 — Cayley Exact Inverse & O(1)-VRAM Reversibility):
    For any generator w : Vec3, rational denominator d : Int, and holonomy state U : QuatZ:
      1. Strict positivity of unit denominator: 0 < cayleyDen w = 1 + ||w||^2,
      2. Unit-denominator left & right inverse:
         qmul (cayleyNum w) (cayleyNum (-w)) = (cayleyDen(w)^2, 0, 0, 0) =
         qmul (cayleyNum (-w)) (cayleyNum w),
      3. Homogeneous rational inverse & Euler norm:
         qmul (cayleyHomNum d w) (cayleyHomNum d (-w)) = (cayleyHomDen(d, w)^2, 0, 0, 0)
         and normSq (cayleyHomNum d w) = cayleyHomDen(d, w)^2,
      4. Exact O(1)-VRAM backward holonomy step reconstruction (kappa = 1):
         qmul (cayleyHomNum d (-w)) (qmul (cayleyHomNum d w) U) = cayleyHomDen(d, w)^2 * U. -/
theorem p3_cayley_exact_inverse_reversibility (w : Vec3) (d : Int) (U : QuatZ) :
    0 < cayleyDen w ∧
    qmul (cayleyNum w) (cayleyNum (vec3Neg w)) =
      ⟨cayleyDen w * cayleyDen w, 0, 0, 0⟩ ∧
    qmul (cayleyNum (vec3Neg w)) (cayleyNum w) =
      ⟨cayleyDen w * cayleyDen w, 0, 0, 0⟩ ∧
    qmul (cayleyHomNum d w) (cayleyHomNum d (vec3Neg w)) =
      ⟨cayleyHomDen d w * cayleyHomDen d w, 0, 0, 0⟩ ∧
    normSq (cayleyHomNum d w) =
      cayleyHomDen d w * cayleyHomDen d w ∧
    qmul (cayleyHomNum d (vec3Neg w)) (qmul (cayleyHomNum d w) U) =
      qscale (cayleyHomDen d w * cayleyHomDen d w) U := by
  have hden_pos : 0 < cayleyDen w := by
    have hnn := vec3NormSq_nonneg w
    simp only [cayleyDen]
    omega
  have h1_lr :
      qmul (cayleyNum w) (cayleyNum (vec3Neg w)) = ⟨cayleyDen w * cayleyDen w, 0, 0, 0⟩ := by
    rw [cayleyNum_neg, quat_mul_conj_self, cayleyNum_norm_identity]
  have h1_rl :
      qmul (cayleyNum (vec3Neg w)) (cayleyNum w) = ⟨cayleyDen w * cayleyDen w, 0, 0, 0⟩ := by
    rw [cayleyNum_neg, quat_conj_mul_self, cayleyNum_norm_identity]
  have hd_lr :
      qmul (cayleyHomNum d w) (cayleyHomNum d (vec3Neg w)) =
        ⟨cayleyHomDen d w * cayleyHomDen d w, 0, 0, 0⟩ := by
    rw [cayleyHomNum_neg, quat_mul_conj_self, cayleyHom_norm_identity]
  have hd_rl :
      qmul (cayleyHomNum d (vec3Neg w)) (cayleyHomNum d w) =
        ⟨cayleyHomDen d w * cayleyHomDen d w, 0, 0, 0⟩ := by
    rw [cayleyHomNum_neg, quat_conj_mul_self, cayleyHom_norm_identity]
  have hrev :
      qmul (cayleyHomNum d (vec3Neg w)) (qmul (cayleyHomNum d w) U) =
        qscale (cayleyHomDen d w * cayleyHomDen d w) U := by
    rw [← qmul_assoc, hd_rl, qmul_scalar_left]
  exact ⟨hden_pos, h1_lr, h1_rl, hd_lr, cayleyHom_norm_identity d w, hrev⟩

-- ============================================================================
-- Theorem #40 & #40b: H_256 Byte Orthogonality & Bent-Phase Invariance
-- ============================================================================

/-- Recursive k-bit population count (number of set bits among the low k bits of x). -/
def popcBits : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, x => (x % 2) + popcBits k (x / 2)

/-- Exact 8-bit popcount on Nat (low 8 bits) and UInt8. -/
def popc8 (x : Nat) : Nat := popcBits 8 x
def popcUInt8 (b : UInt8) : Nat := popcBits 8 b.toNat

/-- Maiorana-McFarland 4-bit bent phase sign (-1)^(popc((lane & 3) & ((lane >> 2) & 3)) % 2)
    from W10-OBS-55 / silikernels#2#issuecomment-7178686206 (eliminates 64/2048 -> 0/2048
    constant warps while preserving exact 256x256 Hadamard orthogonality). -/
def bent4PhaseFin (d : Fin 256) : Int :=
  (-1 : Int) ^ ((popc8 ((d.val &&& 3) &&& ((d.val / 4) &&& 3))) % 2)

def bent4PhaseUInt8 (d : UInt8) : Int :=
  (-1 : Int) ^ ((popcUInt8 ((d &&& (3 : UInt8)) &&& ((d >>> (2 : UInt8)) &&& (3 : UInt8)))) % 2)

/-- Sum f(0) + f(1) + ... + f(n - 1). -/
def sumRange : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumRange n f + f n

/-- Sum over Fin n: sum_{d : Fin n} f(d). -/
def sumFin : (n : Nat) → (Fin n → Int) → Int
  | 0, _ => 0
  | n + 1, f =>
      sumFin n (fun i => f ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩) +
        f ⟨n, Nat.lt_succ_self n⟩

/-- Sum over all 256 UInt8 byte/channel indices d in 0..255. -/
def sumUInt8 (f : UInt8 → Int) : Int :=
  sumFin 256 (fun d => f (UInt8.ofNat d.val))

private theorem uint8_eq_of_toNat_eq {a b : UInt8} (h : a.toNat = b.toNat) : a = b := by
  cases a with
  | ofBitVec va =>
    cases b with
    | ofBitVec vb =>
      have hv : va.toNat = vb.toNat := h
      have hbv : va = vb := BitVec.eq_of_toNat_eq hv
      subst hbv
      rfl

private theorem uint8_ofNat_toNat_lt (d : Nat) (hd : d < 256) :
    (UInt8.ofNat d).toNat = d := by
  rw [UInt8.toNat_ofNat']
  exact Nat.mod_eq_of_lt hd

private theorem sumFin_eq_sumRange (n : Nat) (g : Nat → Int) :
    sumFin n (fun i => g i.val) = sumRange n g := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumFin, sumRange]
    rw [ih]

private theorem sumFin_congr (n : Nat) (f g : Fin n → Int)
    (hfg : ∀ d : Fin n, f d = g d) : sumFin n f = sumFin n g := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumFin]
    rw [ih _ _ (fun i => hfg ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩),
        hfg ⟨m, Nat.lt_succ_self m⟩]

private theorem sumRange_congr (n : Nat) (f g : Nat → Int)
    (hfg : ∀ i, i < n → f i = g i) : sumRange n f = sumRange n g := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumRange]
    rw [ih (fun i hi => hfg i (by omega)), hfg m (by omega)]

private theorem sumRange_mul_const (n : Nat) (c : Int) (f : Nat → Int) :
    sumRange n (fun i => c * f i) = c * sumRange n f := by
  induction n with
  | zero =>
    simp only [sumRange, Int.mul_zero]
  | succ m ih =>
    simp only [sumRange, ih, Int.mul_add]

private theorem sumRange_const_one (n : Nat) :
    sumRange n (fun _ => 1) = (n : Int) := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumRange, ih]
    omega

private theorem sumRange_even_odd (m : Nat) (f : Nat → Int) :
    sumRange (2 * m) f = sumRange m (fun d => f (2 * d) + f (2 * d + 1)) := by
  induction m with
  | zero => rfl
  | succ k ih =>
    have h2 : 2 * (k + 1) = (2 * k + 1) + 1 := by omega
    rw [h2]
    simp only [sumRange]
    rw [ih]
    omega

private theorem neg_one_pow_add (a b : Nat) :
    ((-1 : Int) ^ (a + b)) = ((-1 : Int) ^ a) * ((-1 : Int) ^ b) := by
  induction b with
  | zero =>
    simp only [Nat.add_zero, Int.pow_zero, Int.mul_one]
  | succ m ih =>
    rw [Nat.add_succ, Int.pow_succ, ih, Int.pow_succ, Int.mul_assoc]

private theorem neg_one_pow_sq_eq_one (n : Nat) :
    ((-1 : Int) ^ n) * ((-1 : Int) ^ n) = 1 := by
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [Int.pow_succ]
    have hcomm :
        ((-1 : Int) ^ m * (-1)) * ((-1 : Int) ^ m * (-1)) =
          ((-1 : Int) ^ m * (-1 : Int) ^ m) * ((-1 : Int) * (-1)) := by
      simp only [Int.mul_assoc, Int.mul_left_comm]
    rw [hcomm, ih]
    rfl

private theorem and_two_mul_mod2 (a d : Nat) : (a &&& (2 * d)) % 2 = 0 := by
  have h1 : (a &&& (2 * d)) % 2 ^ 1 = (a % 2 ^ 1) &&& ((2 * d) % 2 ^ 1) := Nat.and_mod_two_pow
  have hp : 2 ^ 1 = 2 := rfl
  rw [hp] at h1
  have h2d : (2 * d) % 2 = 0 := by omega
  rw [h1, h2d, Nat.and_zero]

private theorem and_two_mul_add_one_mod2 (a d : Nat) : (a &&& (2 * d + 1)) % 2 = a % 2 := by
  have h1 : (a &&& (2 * d + 1)) % 2 ^ 1 = (a % 2 ^ 1) &&& ((2 * d + 1) % 2 ^ 1) :=
    Nat.and_mod_two_pow
  have hp : 2 ^ 1 = 2 := rfl
  rw [hp] at h1
  have h2d : (2 * d + 1) % 2 = 1 := by omega
  rw [h1, h2d]
  if ha : a % 2 = 0 then
    rw [ha]
    rfl
  else
    have ha1 : a % 2 = 1 := by omega
    rw [ha1]
    rfl

private theorem and_div2_even (a d : Nat) : (a &&& (2 * d)) / 2 = (a / 2) &&& d := by
  have h1 : (a &&& (2 * d)) / 2 ^ 1 = (a / 2 ^ 1) &&& ((2 * d) / 2 ^ 1) := Nat.and_div_two_pow
  have hp : 2 ^ 1 = 2 := rfl
  rw [hp] at h1
  have hd : (2 * d) / 2 = d := by omega
  rw [h1, hd]

private theorem and_div2_odd (a d : Nat) : (a &&& (2 * d + 1)) / 2 = (a / 2) &&& d := by
  have h1 : (a &&& (2 * d + 1)) / 2 ^ 1 = (a / 2 ^ 1) &&& ((2 * d + 1) / 2 ^ 1) :=
    Nat.and_div_two_pow
  have hp : 2 ^ 1 = 2 := rfl
  rw [hp] at h1
  have hd : (2 * d + 1) / 2 = d := by omega
  rw [h1, hd]

private theorem lsb_factor (a b : Nat) :
    1 + ((-1 : Int) ^ (a % 2)) * ((-1 : Int) ^ (b % 2)) =
      if a % 2 = b % 2 then 2 else 0 := by
  if ha : a % 2 = 0 then
    if hb : b % 2 = 0 then
      rw [ha, hb]
      rfl
    else
      have hb1 : b % 2 = 1 := by omega
      rw [ha, hb1]
      rfl
  else
    have ha1 : a % 2 = 1 := by omega
    if hb : b % 2 = 0 then
      rw [ha1, hb]
      rfl
    else
      have hb1 : b % 2 = 1 := by omega
      rw [ha1, hb1]
      rfl

/-- General k-bit Walsh-Hadamard character orthogonality over 0 .. 2^k - 1
    proved purely by bit-peeling induction (zero native_decide / zero brute force). -/
private theorem walsh_orthogonality_bits (k a b : Nat)
    (ha : a < 2 ^ k) (hb : b < 2 ^ k) :
    sumRange (2 ^ k)
      (fun d => ((-1 : Int) ^ popcBits k (a &&& d)) * ((-1 : Int) ^ popcBits k (b &&& d))) =
    if a = b then ((2 ^ k : Nat) : Int) else 0 := by
  induction k generalizing a b with
  | zero =>
    have ha0 : a = 0 := by omega
    have hb0 : b = 0 := by omega
    subst ha0 hb0
    rfl
  | succ m ih =>
    have hpow : 2 ^ (m + 1) = 2 * (2 ^ m) := by
      rw [Nat.pow_succ, Nat.mul_comm]
    rw [hpow, sumRange_even_odd (2 ^ m)]
    have hterm : ∀ d, d < 2 ^ m →
        ((-1 : Int) ^ popcBits (m + 1) (a &&& (2 * d))) *
          ((-1 : Int) ^ popcBits (m + 1) (b &&& (2 * d))) +
        ((-1 : Int) ^ popcBits (m + 1) (a &&& (2 * d + 1))) *
          ((-1 : Int) ^ popcBits (m + 1) (b &&& (2 * d + 1))) =
        (1 + ((-1 : Int) ^ (a % 2)) * ((-1 : Int) ^ (b % 2))) *
          (((-1 : Int) ^ popcBits m ((a / 2) &&& d)) *
           ((-1 : Int) ^ popcBits m ((b / 2) &&& d))) := by
      intro d _
      simp only [popcBits, and_two_mul_mod2, and_two_mul_add_one_mod2,
                 and_div2_even, and_div2_odd, Nat.zero_add]
      rw [neg_one_pow_add (a % 2), neg_one_pow_add (b % 2)]
      simp only [Int.add_mul, Int.one_mul, Int.mul_assoc, Int.mul_left_comm]
    rw [sumRange_congr (2 ^ m) _ _ hterm, sumRange_mul_const]
    have ha_div : a / 2 < 2 ^ m := by omega
    have hb_div : b / 2 < 2 ^ m := by omega
    rw [ih (a / 2) (b / 2) ha_div hb_div, lsb_factor a b]
    if hmod : a % 2 = b % 2 then
      simp only [hmod, ↓reduceIte]
      if hdiv : a / 2 = b / 2 then
        have hab : a = b := by omega
        simp only [hab, ↓reduceIte]
        omega
      else
        have hne : a ≠ b := by omega
        simp only [hdiv, hne, ↓reduceIte, Int.mul_zero]
    else
      have hne : a ≠ b := by omega
      simp only [hmod, hne, ↓reduceIte, Int.zero_mul]

/-- Theorem #40 (W10-OBS-51 / OBS-ORDA-12 — H_256 Byte Injection Pairwise Orthogonality):
    Proves by exact k-bit character-sum induction (zero native_decide, zero Lean.ofReduceBool,
    zero Classical.choice) that the 0-VRAM H_256 Sylvester-Hadamard byte embedding
    e(b)_d = (1/16) * (-1)^popc(b &&& d) is 100% pairwise orthogonal on S^255:
      1. Over Fin 256: sum_{d : Fin 256} (-1)^popc8(b1 &&& d) * (-1)^popc8(b2 &&& d) =
         if b1 = b2 then 256 else 0,
      2. Over UInt8: sum_{d : UInt8} (-1)^popcUInt8(b1 &&& d) * (-1)^popcUInt8(b2 &&& d) =
         if b1 = b2 then 256 else 0,
      3. Exact int8 quantization bound & norm: for every byte b and channel d,
         127 * (-1)^popcUInt8(b &&& d) has square 16129 = 127^2, lies strictly inside
         signed int8 [-128, 127] with zero quantization error, and sums to 256 * 127^2 = 4129024. -/
theorem p4_hadamard256_byte_orthogonality
    (f1 f2 : Fin 256) (b1 b2 d8 : UInt8) :
    sumFin 256 (fun d =>
      ((-1 : Int) ^ popc8 (f1.val &&& d.val)) *
      ((-1 : Int) ^ popc8 (f2.val &&& d.val))) =
      (if f1 = f2 then 256 else 0) ∧
    sumUInt8 (fun d =>
      ((-1 : Int) ^ popcUInt8 (b1 &&& d)) *
      ((-1 : Int) ^ popcUInt8 (b2 &&& d))) =
      (if b1 = b2 then 256 else 0) ∧
    (-128 ≤ 127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) ∧
      127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) ≤ 127 ∧
      (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) *
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) = 16129) ∧
    sumUInt8 (fun d =>
      (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d))) *
      (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d)))) = 4129024 := by
  have h256 : (2 ^ 8 : Nat) = 256 := rfl
  have hfin :
      sumFin 256 (fun d =>
        ((-1 : Int) ^ popc8 (f1.val &&& d.val)) *
        ((-1 : Int) ^ popc8 (f2.val &&& d.val))) =
      (if f1 = f2 then 256 else 0) := by
    rw [sumFin_eq_sumRange 256
          (fun d => ((-1 : Int) ^ popc8 (f1.val &&& d)) * ((-1 : Int) ^ popc8 (f2.val &&& d)))]
    unfold popc8
    have hf1 : f1.val < 2 ^ 8 := by rw [h256]; exact f1.isLt
    have hf2 : f2.val < 2 ^ 8 := by rw [h256]; exact f2.isLt
    have hw := walsh_orthogonality_bits 8 f1.val f2.val hf1 hf2
    rw [h256] at hw
    rw [hw]
    if hfeq : f1 = f2 then
      subst hfeq
      simp only [↓reduceIte]
      rfl
    else
      have hvne : f1.val ≠ f2.val := fun hv => hfeq (Fin.ext hv)
      simp only [hfeq, hvne, ↓reduceIte]
  have huint :
      sumUInt8 (fun d =>
        ((-1 : Int) ^ popcUInt8 (b1 &&& d)) *
        ((-1 : Int) ^ popcUInt8 (b2 &&& d))) =
      (if b1 = b2 then 256 else 0) := by
    unfold sumUInt8
    rw [sumFin_eq_sumRange 256
          (fun d =>
            ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d)) *
            ((-1 : Int) ^ popcUInt8 (b2 &&& UInt8.ofNat d)))]
    have hcong : ∀ d, d < 256 →
        ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d)) *
          ((-1 : Int) ^ popcUInt8 (b2 &&& UInt8.ofNat d)) =
        ((-1 : Int) ^ popcBits 8 (b1.toNat &&& d)) *
          ((-1 : Int) ^ popcBits 8 (b2.toNat &&& d)) := by
      intro d hd
      unfold popcUInt8
      rw [UInt8.toNat_and, UInt8.toNat_and, uint8_ofNat_toNat_lt d hd]
    rw [sumRange_congr 256 _ _ hcong]
    have hb1 : b1.toNat < 2 ^ 8 := b1.toNat_lt
    have hb2 : b2.toNat < 2 ^ 8 := b2.toNat_lt
    have hw := walsh_orthogonality_bits 8 b1.toNat b2.toNat hb1 hb2
    rw [h256] at hw
    rw [hw]
    if hbeq : b1 = b2 then
      subst hbeq
      simp only [↓reduceIte]
      rfl
    else
      have hvne : b1.toNat ≠ b2.toNat := fun hv => hbeq (uint8_eq_of_toNat_eq hv)
      simp only [hbeq, hvne, ↓reduceIte]
  have hsq_unit :
      ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) * ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) = 1 :=
    neg_one_pow_sq_eq_one (popcUInt8 (b1 &&& d8))
  have hentry_sq :
      (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) *
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) = 16129 := by
    have hcomm :
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) *
          (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) =
        (127 * 127) *
          (((-1 : Int) ^ popcUInt8 (b1 &&& d8)) * ((-1 : Int) ^ popcUInt8 (b1 &&& d8))) := by
      simp only [Int.mul_assoc, Int.mul_left_comm]
    rw [hcomm, hsq_unit]
    rfl
  have hsign_bounds :
      -1 ≤ ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) ∧
        ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) ≤ 1 := by
    if hlo : -1 ≤ ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) then
      if hhi : ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) ≤ 1 then
        exact ⟨hlo, hhi⟩
      else
        have hd2 : 0 ≤ ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) - 2 := by omega
        have hp := Int.mul_nonneg hd2 (by omega : 0 ≤ ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) + 2)
        simp only [Int.sub_mul, Int.mul_add] at hp
        omega
    else
      have hd2 : 0 ≤ -2 - ((-1 : Int) ^ popcUInt8 (b1 &&& d8)) := by omega
      have hp := Int.mul_nonneg hd2 (by omega : 0 ≤ 2 - ((-1 : Int) ^ popcUInt8 (b1 &&& d8)))
      simp only [Int.sub_mul, Int.mul_sub] at hp
      omega
  have hs255_norm :
      sumUInt8 (fun d =>
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d))) *
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& d)))) = 4129024 := by
    unfold sumUInt8
    rw [sumFin_eq_sumRange 256
          (fun d =>
            (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) *
            (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))))]
    have hconst : ∀ d, d < 256 →
        (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) *
          (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) = 16129 * 1 := by
      intro d _
      have hs := neg_one_pow_sq_eq_one (popcUInt8 (b1 &&& UInt8.ofNat d))
      have hc :
          (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) *
            (127 * ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) =
          (127 * 127) *
            (((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d)) *
             ((-1 : Int) ^ popcUInt8 (b1 &&& UInt8.ofNat d))) := by
        simp only [Int.mul_assoc, Int.mul_left_comm]
      rw [hc, hs]
      rfl
    rw [sumRange_congr 256 _ _ hconst, sumRange_mul_const, sumRange_const_one]
    rfl
  refine ⟨hfin, huint, ⟨by omega, by omega, hentry_sq⟩, hs255_norm⟩

/-- Theorem #40b (W10-OBS-55 / OBS-ORDA-16 — Bent-Phase Sign Modulation Preserves Orthogonality):
    For ANY fixed +-1 diagonal phase modulation s : Fin 256 -> Int with forall d, s d * s d = 1
    (and concretely for the Maiorana-McFarland bent phase bent4PhaseFin / bent4PhaseUInt8):
      1. Universal sign-modulation isometry: sum_d (s d * e1 d) * (s d * e2 d) = sum_d e1 d * e2 d,
      2. Exact +-1 unit-square law for bent4PhaseFin and bent4PhaseUInt8 across all 256 channels,
      3. Bent-phase-modulated H_256 byte embeddings remain 100% pairwise orthogonal over Fin 256
         and UInt8 (= 256 iff b1 = b2, else 0). -/
theorem p4b_bent_phase_preserves_orthogonality
    (s : Fin 256 → Int) (hs : ∀ d : Fin 256, s d * s d = 1)
    (e1 e2 : Fin 256 → Int)
    (f1 f2 dFin : Fin 256) (b1 b2 d8 : UInt8) :
    sumFin 256 (fun d => (s d * e1 d) * (s d * e2 d)) =
      sumFin 256 (fun d => e1 d * e2 d) ∧
    bent4PhaseFin dFin * bent4PhaseFin dFin = 1 ∧
    bent4PhaseUInt8 d8 * bent4PhaseUInt8 d8 = 1 ∧
    sumFin 256 (fun d =>
      (bent4PhaseFin d * ((-1 : Int) ^ popc8 (f1.val &&& d.val))) *
      (bent4PhaseFin d * ((-1 : Int) ^ popc8 (f2.val &&& d.val)))) =
      (if f1 = f2 then 256 else 0) ∧
    sumUInt8 (fun d =>
      (bent4PhaseUInt8 d * ((-1 : Int) ^ popcUInt8 (b1 &&& d))) *
      (bent4PhaseUInt8 d * ((-1 : Int) ^ popcUInt8 (b2 &&& d)))) =
      (if b1 = b2 then 256 else 0) := by
  have hgen :
      ∀ (s0 eA eB : Fin 256 → Int), (∀ d : Fin 256, s0 d * s0 d = 1) →
        sumFin 256 (fun d => (s0 d * eA d) * (s0 d * eB d)) =
          sumFin 256 (fun d => eA d * eB d) := by
    intro s0 eA eB hs0
    apply sumFin_congr 256
    intro d
    have hcomm :
        (s0 d * eA d) * (s0 d * eB d) = (s0 d * s0 d) * (eA d * eB d) := by
      simp only [Int.mul_assoc, Int.mul_left_comm]
    rw [hcomm, hs0 d, Int.one_mul]
  have hbent_fin : ∀ d : Fin 256, bent4PhaseFin d * bent4PhaseFin d = 1 := by
    intro d
    exact neg_one_pow_sq_eq_one ((popc8 ((d.val &&& 3) &&& ((d.val / 4) &&& 3))) % 2)
  have hbent_u8 : ∀ d : UInt8, bent4PhaseUInt8 d * bent4PhaseUInt8 d = 1 := by
    intro d
    exact neg_one_pow_sq_eq_one
      ((popcUInt8 ((d &&& (3 : UInt8)) &&& ((d >>> (2 : UInt8)) &&& (3 : UInt8)))) % 2)
  have hortho := p4_hadamard256_byte_orthogonality f1 f2 b1 b2 d8
  have hfin_bent :
      sumFin 256 (fun d =>
        (bent4PhaseFin d * ((-1 : Int) ^ popc8 (f1.val &&& d.val))) *
        (bent4PhaseFin d * ((-1 : Int) ^ popc8 (f2.val &&& d.val)))) =
      (if f1 = f2 then 256 else 0) := by
    rw [hgen bent4PhaseFin _ _ hbent_fin]
    exact hortho.1
  have hu8_bent :
      sumUInt8 (fun d =>
        (bent4PhaseUInt8 d * ((-1 : Int) ^ popcUInt8 (b1 &&& d))) *
        (bent4PhaseUInt8 d * ((-1 : Int) ^ popcUInt8 (b2 &&& d)))) =
      (if b1 = b2 then 256 else 0) := by
    unfold sumUInt8
    rw [hgen (fun d => bent4PhaseUInt8 (UInt8.ofNat d.val)) _ _
          (fun d => hbent_u8 (UInt8.ofNat d.val))]
    exact hortho.2.1
  exact ⟨hgen s e1 e2 hs, hbent_fin dFin, hbent_u8 d8, hfin_bent, hu8_bent⟩

-- ============================================================================
-- Machine-Checked Axiom Audit (#print axioms)
-- ============================================================================

#print axioms HoloSphereCellOpInvariants.p1_blockh4_involutory_and_int8_bounds
#print axioms HoloSphereCellOpInvariants.p2_cayley_unit_ball_so3_involution_and_even
#print axioms HoloSphereCellOpInvariants.p3_cayley_exact_inverse_reversibility
#print axioms HoloSphereCellOpInvariants.p4_hadamard256_byte_orthogonality
#print axioms HoloSphereCellOpInvariants.p4b_bent_phase_preserves_orthogonality

end HoloSphereCellOpInvariants
