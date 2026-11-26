/-
  LoopCDStrang/QuatStrang.lean

  Theorems 2, 3, & 4 (Palindromic Strang `SU(2, Q)` Holonomy & Acute-Ball Bounds):
  - Exact integer-scaled quaternion & `SU(2, Q)` Cayley-BCH algebra (`cayley_compose_su2_exact`)
  - **Theorem 2 (`thm2a_right_strang_bch_quadratic_zero` & `thm2b_left_ordered_bch_quadratic_defect_nonzero`)**:
      Under Cayley-parameter scaling `ω · w_Δ` (`w_Δ = (-w_1) ⊕_C w_R`), right-ordered palindromic
      composition `C(w_R) · C(ω w_Δ)` cancels the degree-0, degree-1, and degree-2 BCH commutator
      defects identically (`O(t^3)` jet with exact `t^3` cubic remainder) universally for ALL `a, b ∈ Z^3`,
      ALL scales `S`, and ALL `ω = u/v`, whereas non-palindromic left ordering `C(ω w_Δ) · C(w_R)` retains
      exact nonzero `-4 v^2 u S^9 (a × b)` quadratic defect (with `(0, 0, -4)` orthogonal witness + dyadic
      `SU(2, Q)` squared-chordal-error hierarchy `0 < E_right < E_left`)!
  - **Theorem 3 (`thm3a_acute_ball_spinor_denom_ge_9_25`, `thm3b_acute_ball_cayley_denom_ge_81_625`,
      `thm3c_acute_ball_81_625_tight`)**:
      Inside the acute spinor ball `||w_1||, ||w_R|| ≤ 4/5` with `ω ∈ [0, 1]`, `1 + <w_1, w_R> ≥ 9/25 > 0`
      and Cayley composition denominator `≥ 81/625 > 0` (tight at `(±4/5, 0, 0)`).
  - **Theorem 4 (`thm4a_commuting_cayley_quat_commute`, `thm4b_commutative_left_right_collapse`,
      `thm4c_collinear_left_right_cayley_sum_exact`)**:
      When `[w_1, w_R] = 0` (`w_1 × w_R = 0`), right-ordered, left-ordered, and Cayley-sum
      extrapolators collapse identically to all orders!

  Pure `Init` (`Lean 4.36.0`), `0` `Classical.choice`, `0` `bv_decide`, `0` `native_decide`, `0` `sorryAx`.
-/

import LoopCDStrang.SphereChord

namespace LoopCDStrang

/-! ## 1. Exact Integer-Scaled `Im(H)` & `SU(2, Q)` Quaternion Cayley Algebra -/

abbrev Vec3Z := Int × Int × Int
abbrev QuatZ := Int × Int × Int × Int

def dot3 (a b : Vec3Z) : Int :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

def normSq3 (a : Vec3Z) : Int :=
  dot3 a a

def cross3 (a b : Vec3Z) : Vec3Z :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1   - a.1   * b.2.2,
   a.1   * b.2.1 - a.2.1 * b.1)

def vecAdd3 (a b : Vec3Z) : Vec3Z :=
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2)

def vecSub3 (a b : Vec3Z) : Vec3Z :=
  (a.1 - b.1, a.2.1 - b.2.1, a.2.2 - b.2.2)

def vecNeg3 (a : Vec3Z) : Vec3Z :=
  (-a.1, -a.2.1, -a.2.2)

def vecScale3 (c : Int) (a : Vec3Z) : Vec3Z :=
  (c * a.1, c * a.2.1, c * a.2.2)

def quatConj (q : QuatZ) : QuatZ :=
  (q.1, -q.2.1, -q.2.2.1, -q.2.2.2)

def quatMul (p q : QuatZ) : QuatZ :=
  (p.1 * q.1     - p.2.1 * q.2.1   - p.2.2.1 * q.2.2.1 - p.2.2.2 * q.2.2.2,
   p.1 * q.2.1   + p.2.1 * q.1     + p.2.2.1 * q.2.2.2 - p.2.2.2 * q.2.2.1,
   p.1 * q.2.2.1 - p.2.1 * q.2.2.2 + p.2.2.1 * q.1     + p.2.2.2 * q.2.1,
   p.1 * q.2.2.2 + p.2.1 * q.2.2.1 - p.2.2.1 * q.2.1   + p.2.2.2 * q.1)

def quatNormSq (q : QuatZ) : Int :=
  q.1 * q.1 + q.2.1 * q.2.1 + q.2.2.1 * q.2.2.1 + q.2.2.2 * q.2.2.2

/-- Homogeneous numerator `(d^2 - ||w||^2, 2 d w)` of `C(w / d) ∈ SU(2, Q)`. -/
def cayleyQuatNum (w : Vec3Z) (d : Int) : QuatZ :=
  (d * d - normSq3 w, 2 * d * w.1, 2 * d * w.2.1, 2 * d * w.2.2)

/-- Homogeneous denominator `d^2 + ||w||^2` of `C(w / d) ∈ SU(2, Q)`. -/
def cayleyQuatDen (w : Vec3Z) (d : Int) : Int :=
  d * d + normSq3 w

/-- Homogeneous numerator of non-Abelian Cayley-BCH composition `(x / dx) ⊕_C (y / dy)`. -/
def cayleyCompNum (x : Vec3Z) (dx : Int) (y : Vec3Z) (dy : Int) : Vec3Z :=
  let c1 := dx * (dy * dy - normSq3 y)
  let c2 := dy * (dx * dx - normSq3 x)
  let c3 := 2 * dx * dy
  vecAdd3 (vecAdd3 (vecScale3 c1 x) (vecScale3 c2 y)) (vecScale3 c3 (cross3 x y))

/-- Homogeneous denominator `dx^2 dy^2 + ||x||^2 ||y||^2 - 2 dx dy <x, y>` of `(x / dx) ⊕_C (y / dy)`. -/
def cayleyCompDen (x : Vec3Z) (dx : Int) (y : Vec3Z) (dy : Int) : Int :=
  dx * dx * (dy * dy) + normSq3 x * normSq3 y - 2 * dx * dy * dot3 x y

/-! ### Top-Level Reified `PExpr` Algebra (Exact Identical Operator Associativity) -/

private abbrev Vec3E := PExpr × PExpr × PExpr
private abbrev QuatE := PExpr × PExpr × PExpr × PExpr

private def dot3E (a b : Vec3E) : PExpr :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

private def normSq3E (a : Vec3E) : PExpr := dot3E a a

private def cross3E (a b : Vec3E) : Vec3E :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1   - a.1   * b.2.2,
   a.1   * b.2.1 - a.2.1 * b.1)

private def vadd3E (a b : Vec3E) : Vec3E := (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2)
private def vsub3E (a b : Vec3E) : Vec3E := (a.1 - b.1, a.2.1 - b.2.1, a.2.2 - b.2.2)
private def vneg3E (a : Vec3E) : Vec3E := (-a.1, -a.2.1, -a.2.2)
private def vscale3E (c : PExpr) (a : Vec3E) : Vec3E := (c * a.1, c * a.2.1, c * a.2.2)

private def qmulE (p q : QuatE) : QuatE :=
  (p.1 * q.1     - p.2.1 * q.2.1   - p.2.2.1 * q.2.2.1 - p.2.2.2 * q.2.2.2,
   p.1 * q.2.1   + p.2.1 * q.1     + p.2.2.1 * q.2.2.2 - p.2.2.2 * q.2.2.1,
   p.1 * q.2.2.1 - p.2.1 * q.2.2.2 + p.2.2.1 * q.1     + p.2.2.2 * q.2.1,
   p.1 * q.2.2.2 + p.2.1 * q.2.2.1 - p.2.2.1 * q.2.1   + p.2.2.2 * q.1)

private def qnormE (q : QuatE) : PExpr :=
  q.1 * q.1 + q.2.1 * q.2.1 + q.2.2.1 * q.2.2.1 + q.2.2.2 * q.2.2.2

private def cayleyQuatNumE (w : Vec3E) (d : PExpr) : QuatE :=
  (d * d - normSq3E w, 2 * d * w.1, 2 * d * w.2.1, 2 * d * w.2.2)

private def cayleyQuatDenE (w : Vec3E) (d : PExpr) : PExpr :=
  d * d + normSq3E w

private def cayleyCompNumE (x : Vec3E) (dx : PExpr) (y : Vec3E) (dy : PExpr) : Vec3E :=
  let c1 := dx * (dy * dy - normSq3E y)
  let c2 := dy * (dx * dx - normSq3E x)
  let c3 := 2 * dx * dy
  vadd3E (vadd3E (vscale3E c1 x) (vscale3E c2 y)) (vscale3E c3 (cross3E x y))

private def cayleyCompDenE (x : Vec3E) (dx : PExpr) (y : Vec3E) (dy : PExpr) : PExpr :=
  dx * dx * (dy * dy) + normSq3E x * normSq3E y - 2 * dx * dy * dot3E x y

private def qAssocPair : QuatE × QuatE :=
  let p : QuatE := (.var 0, .var 1, .var 2, .var 3)
  let q : QuatE := (.var 4, .var 5, .var 6, .var 7)
  let r : QuatE := (.var 8, .var 9, .var 10, .var 11)
  (qmulE (qmulE p q) r, qmulE p (qmulE q r))

private def qNormMulPair : PExpr × PExpr :=
  let p : QuatE := (.var 0, .var 1, .var 2, .var 3)
  let q : QuatE := (.var 4, .var 5, .var 6, .var 7)
  (qnormE (qmulE p q), qnormE p * qnormE q)

private def cayleyUnitPair : PExpr × PExpr :=
  let w : Vec3E := (.var 0, .var 1, .var 2)
  let d := PExpr.var 3
  let den := cayleyQuatDenE w d
  (qnormE (cayleyQuatNumE w d), den * den)

/-- Degree-4 quaternion product relation `qProd = (2 * d - dProd, 2 * n)` (~16 monomials). -/
private def cayleyProdDeg4E : QuatE × QuatE :=
  let x : Vec3E := (.var 0, .var 1, .var 2); let dx := PExpr.var 3
  let y : Vec3E := (.var 4, .var 5, .var 6); let dy := PExpr.var 7
  let n := cayleyCompNumE x dx y dy
  let d := cayleyCompDenE x dx y dy
  let qProd := qmulE (cayleyQuatNumE x dx) (cayleyQuatNumE y dy)
  let dProd := cayleyQuatDenE x dx * cayleyQuatDenE y dy
  (qProd, (2 * d - dProd, 2 * n.1, 2 * n.2.1, 2 * n.2.2))

private def cayleyNormBridgePair : PExpr × PExpr :=
  let d := PExpr.var 0; let dProd := PExpr.var 1
  let n : Vec3E := (.var 2, .var 3, .var 4)
  (qnormE (2 * d - dProd, 2 * n.1, 2 * n.2.1, 2 * n.2.2) - dProd * dProd,
   4 * (cayleyQuatDenE n d - d * dProd))

private def sqMulCommPair : PExpr × PExpr :=
  let a := PExpr.var 0; let b := PExpr.var 1
  ((a * a) * (b * b), (a * b) * (a * b))

private def qComp0BridgePair : PExpr × PExpr :=
  let d := PExpr.var 0; let dProd := PExpr.var 1; let nsq := PExpr.var 2
  ((d * d - nsq) + (d * d + nsq - d * dProd), d * (2 * d - dProd))

private def qCompVecBridgePair : PExpr × PExpr :=
  let d := PExpr.var 0; let ni := PExpr.var 1
  (2 * d * ni, d * (2 * ni))

/-- Exact associativity of Hamilton quaternion multiplication over `Int`. -/
theorem quat_mul_assoc (p q r : QuatZ) :
    quatMul (quatMul p q) r = quatMul p (quatMul q r) := by
  let env := [p.1, p.2.1, p.2.2.1, p.2.2.2, q.1, q.2.1, q.2.2.1, q.2.2.2, r.1, r.2.1, r.2.2.1, r.2.2.2]
  have h0 : (quatMul (quatMul p q) r).1 = (quatMul p (quatMul q r)).1 :=
    poly_eq_of_norm_eq env qAssocPair.1.1 qAssocPair.2.1 rfl
  have h1 : (quatMul (quatMul p q) r).2.1 = (quatMul p (quatMul q r)).2.1 :=
    poly_eq_of_norm_eq env qAssocPair.1.2.1 qAssocPair.2.2.1 rfl
  have h2 : (quatMul (quatMul p q) r).2.2.1 = (quatMul p (quatMul q r)).2.2.1 :=
    poly_eq_of_norm_eq env qAssocPair.1.2.2.1 qAssocPair.2.2.2.1 rfl
  have h3 : (quatMul (quatMul p q) r).2.2.2 = (quatMul p (quatMul q r)).2.2.2 :=
    poly_eq_of_norm_eq env qAssocPair.1.2.2.2 qAssocPair.2.2.2.2 rfl
  rcases hL : quatMul (quatMul p q) r with ⟨a, b, c, d⟩
  rcases hR : quatMul p (quatMul q r) with ⟨a', b', c', d'⟩
  rw [hL, hR] at h0 h1 h2 h3
  dsimp only at h0 h1 h2 h3
  rw [h0, h1, h2, h3]

/-- Exact palindromic conjugation identity `U_R * (U_1^† * U_R) = (U_R * U_1^†) * U_R`. -/
theorem palindromic_conjugation_identity (u1 ur : QuatZ) :
    quatMul ur (quatMul (quatConj u1) ur) = quatMul (quatMul ur (quatConj u1)) ur :=
  (quat_mul_assoc ur (quatConj u1) ur).symm

/-- Euler's four-square multiplicative norm identity `||p * q||^2 = ||p||^2 * ||q||^2`. -/
theorem quat_norm_sq_mul (p q : QuatZ) :
    quatNormSq (quatMul p q) = quatNormSq p * quatNormSq q :=
  poly_eq_of_norm_eq [p.1, p.2.1, p.2.2.1, p.2.2.2, q.1, q.2.1, q.2.2.1, q.2.2.2]
    qNormMulPair.1 qNormMulPair.2 rfl

/-- Nonnegativity of `normSq3 w = w_x^2 + w_y^2 + w_z^2 ≥ 0`. -/
theorem normSq3_nonneg (w : Vec3Z) : 0 ≤ normSq3 w :=
  Int.add_nonneg (Int.add_nonneg (int_sq_nonneg w.1) (int_sq_nonneg w.2.1)) (int_sq_nonneg w.2.2)

/-- Exact `SU(2, Q)` unit-norm property of the rational Cayley map `C(w / d)`:
    `0 < d^2 + ||w||^2` and `||cayleyQuatNum(w, d)||^2 = (d^2 + ||w||^2)^2`. -/
theorem cayley_su2_unit_norm_exact (w : Vec3Z) (d : Int) (hd : 0 < d) :
    0 < cayleyQuatDen w d ∧
    quatNormSq (cayleyQuatNum w d) = cayleyQuatDen w d * cayleyQuatDen w d :=
  ⟨Int.add_pos_of_pos_of_nonneg (int_sq_pos_of_pos hd) (normSq3_nonneg w),
   poly_eq_of_norm_eq [w.1, w.2.1, w.2.2, d] cayleyUnitPair.1 cayleyUnitPair.2 rfl⟩

/-- Exact non-Abelian `SU(2, Q)` Cayley-BCH group homomorphism:
    `C((x/dx) ⊕_C (y/dy))` equals `C(x/dx) * C(y/dy)` on all 4 quaternion components! -/
theorem cayley_compose_su2_exact (x : Vec3Z) (dx : Int) (y : Vec3Z) (dy : Int) :
    let n := cayleyCompNum x dx y dy
    let d := cayleyCompDen x dx y dy
    let qComp := cayleyQuatNum n d
    let dComp := cayleyQuatDen n d
    let qProd := quatMul (cayleyQuatNum x dx) (cayleyQuatNum y dy)
    let dProd := cayleyQuatDen x dx * cayleyQuatDen y dy
    dComp = d * dProd ∧
    QEq ⟨qComp.1, dComp⟩ ⟨qProd.1, dProd⟩ ∧
    QEq ⟨qComp.2.1, dComp⟩ ⟨qProd.2.1, dProd⟩ ∧
    QEq ⟨qComp.2.2.1, dComp⟩ ⟨qProd.2.2.1, dProd⟩ ∧
    QEq ⟨qComp.2.2.2, dComp⟩ ⟨qProd.2.2.2, dProd⟩ := by
  dsimp only
  let env := [x.1, x.2.1, x.2.2, dx, y.1, y.2.1, y.2.2, dy]
  let n := cayleyCompNum x dx y dy
  let d := cayleyCompDen x dx y dy
  let qProd := quatMul (cayleyQuatNum x dx) (cayleyQuatNum y dy)
  let dProd := cayleyQuatDen x dx * cayleyQuatDen y dy
  have hp0 : qProd.1 = 2 * d - dProd :=
    poly_eq_of_norm_eq env cayleyProdDeg4E.1.1 cayleyProdDeg4E.2.1 rfl
  have hp1 : qProd.2.1 = 2 * n.1 :=
    poly_eq_of_norm_eq env cayleyProdDeg4E.1.2.1 cayleyProdDeg4E.2.2.1 rfl
  have hp2 : qProd.2.2.1 = 2 * n.2.1 :=
    poly_eq_of_norm_eq env cayleyProdDeg4E.1.2.2.1 cayleyProdDeg4E.2.2.2.1 rfl
  have hp3 : qProd.2.2.2 = 2 * n.2.2 :=
    poly_eq_of_norm_eq env cayleyProdDeg4E.1.2.2.2 cayleyProdDeg4E.2.2.2.2 rfl
  have hqp_eq : qProd = (2 * d - dProd, 2 * n.1, 2 * n.2.1, 2 * n.2.2) := by
    rcases hq : qProd with ⟨q0, q1, q2, q3⟩
    rw [hq] at hp0 hp1 hp2 hp3
    dsimp only at hp0 hp1 hp2 hp3
    rw [hp0, hp1, hp2, hp3]
  have hnorm_prod : quatNormSq qProd = dProd * dProd := by
    rw [quat_norm_sq_mul]
    have ux : quatNormSq (cayleyQuatNum x dx) = cayleyQuatDen x dx * cayleyQuatDen x dx :=
      poly_eq_of_norm_eq [x.1, x.2.1, x.2.2, dx] cayleyUnitPair.1 cayleyUnitPair.2 rfl
    have uy : quatNormSq (cayleyQuatNum y dy) = cayleyQuatDen y dy * cayleyQuatDen y dy :=
      poly_eq_of_norm_eq [y.1, y.2.1, y.2.2, dy] cayleyUnitPair.1 cayleyUnitPair.2 rfl
    rw [ux, uy]
    exact poly_eq_of_norm_eq [cayleyQuatDen x dx, cayleyQuatDen y dy] sqMulCommPair.1 sqMulCommPair.2 rfl
  have hbridge : quatNormSq (2 * d - dProd, 2 * n.1, 2 * n.2.1, 2 * n.2.2) - dProd * dProd =
      4 * (cayleyQuatDen n d - d * dProd) :=
    poly_eq_of_norm_eq [d, dProd, n.1, n.2.1, n.2.2] cayleyNormBridgePair.1 cayleyNormBridgePair.2 rfl
  rw [← hqp_eq, hnorm_prod, Int.sub_self] at hbridge
  have hd : cayleyQuatDen n d = d * dProd := by omega
  have hc0 : (cayleyQuatNum n d).1 = d * qProd.1 := by
    have hb0 : (d * d - normSq3 n) + (d * d + normSq3 n - d * dProd) = d * (2 * d - dProd) :=
      poly_eq_of_norm_eq [d, dProd, normSq3 n] qComp0BridgePair.1 qComp0BridgePair.2 rfl
    have hden_zero : d * d + normSq3 n - d * dProd = 0 := by
      change cayleyQuatDen n d - d * dProd = 0
      omega
    rw [hden_zero, Int.add_zero, ← hp0] at hb0
    exact hb0
  have hc1 : (cayleyQuatNum n d).2.1 = d * qProd.2.1 := by
    change 2 * d * n.1 = d * qProd.2.1
    rw [hp1]
    exact poly_eq_of_norm_eq [d, n.1] qCompVecBridgePair.1 qCompVecBridgePair.2 rfl
  have hc2 : (cayleyQuatNum n d).2.2.1 = d * qProd.2.2.1 := by
    change 2 * d * n.2.1 = d * qProd.2.2.1
    rw [hp2]
    exact poly_eq_of_norm_eq [d, n.2.1] qCompVecBridgePair.1 qCompVecBridgePair.2 rfl
  have hc3 : (cayleyQuatNum n d).2.2.2 = d * qProd.2.2.2 := by
    change 2 * d * n.2.2 = d * qProd.2.2.2
    rw [hp3]
    exact poly_eq_of_norm_eq [d, n.2.2] qCompVecBridgePair.1 qCompVecBridgePair.2 rfl
  refine ⟨hd, ?_, ?_, ?_, ?_⟩ <;> unfold QEq <;> dsimp only
  · rw [hc0, hd]; change d * qProd.1 * dProd = qProd.1 * (d * dProd); rw [Int.mul_comm d qProd.1, Int.mul_assoc]
  · rw [hc1, hd]; change d * qProd.2.1 * dProd = qProd.2.1 * (d * dProd); rw [Int.mul_comm d qProd.2.1, Int.mul_assoc]
  · rw [hc2, hd]; change d * qProd.2.2.1 * dProd = qProd.2.2.1 * (d * dProd); rw [Int.mul_comm d qProd.2.2.1, Int.mul_assoc]
  · rw [hc3, hd]; change d * qProd.2.2.2 * dProd = qProd.2.2.2 * (d * dProd); rw [Int.mul_comm d qProd.2.2.2, Int.mul_assoc]

/-! ## 2. Theorem 2: Palindromic Strang Quadratic BCH Cancellation (`Jet3` Non-Perturbative Proof) -/

abbrev Jet3Vec := Jet3 × Jet3 × Jet3

def Jet3.lin (c : Int) : Jet3 := ⟨0, c, 0, 0⟩

def Jet3Vec.eval (t : Int) (v : Jet3Vec) : Vec3Z :=
  (v.1.eval t, v.2.1.eval t, v.2.2.eval t)

def dot3J (t : Int) (a b : Jet3Vec) : Jet3 :=
  (Jet3.mul t a.1 b.1).add ((Jet3.mul t a.2.1 b.2.1).add (Jet3.mul t a.2.2 b.2.2))

def cross3J (t : Int) (a b : Jet3Vec) : Jet3Vec :=
  ((Jet3.mul t a.2.1 b.2.2).sub (Jet3.mul t a.2.2 b.2.1),
   (Jet3.mul t a.2.2 b.1).sub   (Jet3.mul t a.1   b.2.2),
   (Jet3.mul t a.1   b.2.1).sub (Jet3.mul t a.2.1 b.1))

def vadd3J (a b : Jet3Vec) : Jet3Vec := (a.1.add b.1, a.2.1.add b.2.1, a.2.2.add b.2.2)
def vsub3J (a b : Jet3Vec) : Jet3Vec := (a.1.sub b.1, a.2.1.sub b.2.1, a.2.2.sub b.2.2)
def vscale3J (t : Int) (c : Jet3) (a : Jet3Vec) : Jet3Vec :=
  (Jet3.mul t c a.1, Jet3.mul t c a.2.1, Jet3.mul t c a.2.2)

def cayleyCompJ (t : Int) (x : Jet3Vec) (dx : Jet3) (y : Jet3Vec) (dy : Jet3) : Jet3Vec × Jet3 :=
  let nx := dot3J t x x
  let ny := dot3J t y y
  let c1 := Jet3.mul t dx ((Jet3.mul t dy dy).sub ny)
  let c2 := Jet3.mul t dy ((Jet3.mul t dx dx).sub nx)
  let c3 := Jet3.scale 2 (Jet3.mul t dx dy)
  let nJ := vadd3J (vadd3J (vscale3J t c1 x) (vscale3J t c2 y)) (vscale3J t c3 (cross3J t x y))
  let dJ := ((Jet3.mul t (Jet3.mul t dx dx) (Jet3.mul t dy dy)).add (Jet3.mul t nx ny)).sub
            (Jet3.mul t c3 (dot3J t x y))
  (nJ, dJ)

/-- Exact cross-multiplied numerator defect `v * S * N_right - D_right * N_guide` between
    right-ordered `U_R (U_1^{-1} U_R)^ω` (`w_R ⊕_C (ω ((-w_1) ⊕_C w_R))`) and
    linear Cayley guidance `w_guide = (1 + ω) w_R - ω w_1` for `w_1 = t a / S, w_R = t b / S, ω = u / v`. -/
def rightStrangDefectVec (t S u v : Int) (a b : Vec3Z) : Vec3Z :=
  let x := vecScale3 (-t) a
  let y := vecScale3 t b
  let nd := cayleyCompNum x S y S
  let dd := cayleyCompDen x S y S
  let nr := cayleyCompNum y S (vecScale3 u nd) (v * dd)
  let dr := cayleyCompDen y S (vecScale3 u nd) (v * dd)
  let nGuide := vecScale3 t (vecSub3 (vecScale3 (v + u) b) (vecScale3 u a))
  vecSub3 (vecScale3 (v * S) nr) (vecScale3 dr nGuide)

/-- Exact cross-multiplied numerator defect `v * S * N_left - D_left * N_guide` for
    non-palindromic left ordering `(U_1^{-1} U_R)^ω U_R` (`(ω ((-w_1) ⊕_C w_R)) ⊕_C w_R`). -/
def leftOrderedDefectVec (t S u v : Int) (a b : Vec3Z) : Vec3Z :=
  let x := vecScale3 (-t) a
  let y := vecScale3 t b
  let nd := cayleyCompNum x S y S
  let dd := cayleyCompDen x S y S
  let nl := cayleyCompNum (vecScale3 u nd) (v * dd) y S
  let dl := cayleyCompDen (vecScale3 u nd) (v * dd) y S
  let nGuide := vecScale3 t (vecSub3 (vecScale3 (v + u) b) (vecScale3 u a))
  vecSub3 (vecScale3 (v * S) nl) (vecScale3 dl nGuide)

def rightStrangDefectJet (t S u v : Int) (a b : Vec3Z) : Jet3Vec :=
  let sJ := Jet3.const S
  let uJ := Jet3.const u
  let vJ := Jet3.const v
  let xJ : Jet3Vec := (Jet3.lin (-a.1), Jet3.lin (-a.2.1), Jet3.lin (-a.2.2))
  let yJ : Jet3Vec := (Jet3.lin b.1, Jet3.lin b.2.1, Jet3.lin b.2.2)
  let comp1 := cayleyCompJ t xJ sJ yJ sJ
  let comp2 := cayleyCompJ t yJ sJ (vscale3J t uJ comp1.1) (Jet3.mul t vJ comp1.2)
  let gDir := vecSub3 (vecScale3 (v + u) b) (vecScale3 u a)
  let nGuideJ : Jet3Vec := (Jet3.lin gDir.1, Jet3.lin gDir.2.1, Jet3.lin gDir.2.2)
  vsub3J (vscale3J t (Jet3.const (v * S)) comp2.1) (vscale3J t comp2.2 nGuideJ)

def leftOrderedDefectJet (t S u v : Int) (a b : Vec3Z) : Jet3Vec :=
  let sJ := Jet3.const S
  let uJ := Jet3.const u
  let vJ := Jet3.const v
  let xJ : Jet3Vec := (Jet3.lin (-a.1), Jet3.lin (-a.2.1), Jet3.lin (-a.2.2))
  let yJ : Jet3Vec := (Jet3.lin b.1, Jet3.lin b.2.1, Jet3.lin b.2.2)
  let comp1 := cayleyCompJ t xJ sJ yJ sJ
  let comp2 := cayleyCompJ t (vscale3J t uJ comp1.1) (Jet3.mul t vJ comp1.2) yJ sJ
  let gDir := vecSub3 (vecScale3 (v + u) b) (vecScale3 u a)
  let nGuideJ : Jet3Vec := (Jet3.lin gDir.1, Jet3.lin gDir.2.1, Jet3.lin gDir.2.2)
  vsub3J (vscale3J t (Jet3.const (v * S)) comp2.1) (vscale3J t comp2.2 nGuideJ)

private def linJetLHS : PExpr := (0 : PExpr) + PExpr.var 0 * PExpr.var 1 + (PExpr.var 0 * PExpr.var 0) * 0 + (PExpr.var 0 * PExpr.var 0 * PExpr.var 0) * 0
private def linJetRHS : PExpr := PExpr.var 0 * PExpr.var 1

private theorem Jet3_eval_lin (t c : Int) : (Jet3.lin c).eval t = t * c :=
  poly_eq_of_norm_eq [t, c] linJetLHS linJetRHS rfl

private theorem dot3J_eval (t : Int) (a b : Jet3Vec) :
    (dot3J t a b).eval t = dot3 (a.eval t) (b.eval t) := by
  simp only [dot3J, dot3, Jet3Vec.eval, Jet3.eval_add, Jet3.eval_mul, Int.add_assoc]

private theorem cross3J_eval (t : Int) (a b : Jet3Vec) :
    (cross3J t a b).eval t = cross3 (a.eval t) (b.eval t) := by
  simp only [cross3J, cross3, Jet3Vec.eval, Jet3.eval_sub, Jet3.eval_mul]

private theorem vadd3J_eval (t : Int) (a b : Jet3Vec) :
    (vadd3J a b).eval t = vecAdd3 (a.eval t) (b.eval t) := by
  simp only [vadd3J, vecAdd3, Jet3Vec.eval, Jet3.eval_add]

private theorem vsub3J_eval (t : Int) (a b : Jet3Vec) :
    (vsub3J a b).eval t = vecSub3 (a.eval t) (b.eval t) := by
  simp only [vsub3J, vecSub3, Jet3Vec.eval, Jet3.eval_sub]

private theorem vscale3J_eval (t : Int) (c : Jet3) (a : Jet3Vec) :
    (vscale3J t c a).eval t = vecScale3 (c.eval t) (a.eval t) := by
  simp only [vscale3J, vecScale3, Jet3Vec.eval, Jet3.eval_mul]

private theorem cayleyCompJ_fst_eval (t : Int) (x : Jet3Vec) (dx : Jet3) (y : Jet3Vec) (dy : Jet3) :
    (cayleyCompJ t x dx y dy).1.eval t = cayleyCompNum (x.eval t) (dx.eval t) (y.eval t) (dy.eval t) := by
  simp only [cayleyCompJ, cayleyCompNum, normSq3, vadd3J_eval, vscale3J_eval, cross3J_eval, dot3J_eval,
    Jet3.eval_sub, Jet3.eval_mul, Jet3.eval_scale, Int.mul_assoc]

private theorem cayleyCompJ_snd_eval (t : Int) (x : Jet3Vec) (dx : Jet3) (y : Jet3Vec) (dy : Jet3) :
    (cayleyCompJ t x dx y dy).2.eval t = cayleyCompDen (x.eval t) (dx.eval t) (y.eval t) (dy.eval t) := by
  simp only [cayleyCompJ, cayleyCompDen, normSq3, dot3J_eval,
    Jet3.eval_add, Jet3.eval_sub, Jet3.eval_mul, Jet3.eval_scale, Int.mul_assoc]

private theorem jet3Vec_lin_eval (t : Int) (v : Vec3Z) :
    Jet3Vec.eval t (Jet3.lin v.1, Jet3.lin v.2.1, Jet3.lin v.2.2) = vecScale3 t v := by
  simp only [Jet3Vec.eval, Jet3_eval_lin, vecScale3]

theorem rightStrangDefectJet_eval_eq (t S u v : Int) (a b : Vec3Z) :
    (rightStrangDefectJet t S u v a b).eval t = rightStrangDefectVec t S u v a b := by
  unfold rightStrangDefectJet rightStrangDefectVec
  dsimp only
  have hx : Jet3Vec.eval t (Jet3.lin (-a.1), Jet3.lin (-a.2.1), Jet3.lin (-a.2.2)) = vecScale3 (-t) a := by
    simp only [Jet3Vec.eval, Jet3_eval_lin, vecScale3, Int.mul_neg, Int.neg_mul]
  simp only [vsub3J_eval, vscale3J_eval, cayleyCompJ_fst_eval, cayleyCompJ_snd_eval,
    Jet3.eval_mul, Jet3.eval_const, hx, jet3Vec_lin_eval]

theorem leftOrderedDefectJet_eval_eq (t S u v : Int) (a b : Vec3Z) :
    (leftOrderedDefectJet t S u v a b).eval t = leftOrderedDefectVec t S u v a b := by
  unfold leftOrderedDefectJet leftOrderedDefectVec
  dsimp only
  have hx : Jet3Vec.eval t (Jet3.lin (-a.1), Jet3.lin (-a.2.1), Jet3.lin (-a.2.2)) = vecScale3 (-t) a := by
    simp only [Jet3Vec.eval, Jet3_eval_lin, vecScale3, Int.mul_neg, Int.neg_mul]
  simp only [vsub3J_eval, vscale3J_eval, cayleyCompJ_fst_eval, cayleyCompJ_snd_eval,
    Jet3.eval_mul, Jet3.eval_const, hx, jet3Vec_lin_eval]

/-! ### Two-Stage Truncated `Jet2` Homomorphism (Zero Remainder Blowup) -/

private structure Jet2 where
  c0 : Int
  c1 : Int
  c2 : Int
  deriving Repr, DecidableEq

private abbrev Jet2Vec := Jet2 × Jet2 × Jet2

private def Jet3.trunc (j : Jet3) : Jet2 := ⟨j.c0, j.c1, j.c2⟩
private def Jet3Vec.trunc (v : Jet3Vec) : Jet2Vec := (v.1.trunc, v.2.1.trunc, v.2.2.trunc)

private def Jet2.const (c : Int) : Jet2 := ⟨c, 0, 0⟩
private def Jet2.lin (c : Int) : Jet2 := ⟨0, c, 0⟩
private def Jet2.add (x y : Jet2) : Jet2 := ⟨x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2⟩
private def Jet2.sub (x y : Jet2) : Jet2 := ⟨x.c0 - y.c0, x.c1 - y.c1, x.c2 - y.c2⟩
private def Jet2.scale (k : Int) (x : Jet2) : Jet2 := ⟨k * x.c0, k * x.c1, k * x.c2⟩
private def Jet2.mul (x y : Jet2) : Jet2 :=
  ⟨x.c0 * y.c0,
   x.c0 * y.c1 + x.c1 * y.c0,
   x.c0 * y.c2 + x.c1 * y.c1 + x.c2 * y.c0⟩

private def dot3J2Z (a b : Jet2Vec) : Jet2 :=
  (Jet2.mul a.1 b.1).add ((Jet2.mul a.2.1 b.2.1).add (Jet2.mul a.2.2 b.2.2))

private def cross3J2Z (a b : Jet2Vec) : Jet2Vec :=
  ((Jet2.mul a.2.1 b.2.2).sub (Jet2.mul a.2.2 b.2.1),
   (Jet2.mul a.2.2 b.1).sub   (Jet2.mul a.1   b.2.2),
   (Jet2.mul a.1   b.2.1).sub (Jet2.mul a.2.1 b.1))

private def vadd3J2Z (a b : Jet2Vec) : Jet2Vec := (a.1.add b.1, a.2.1.add b.2.1, a.2.2.add b.2.2)
private def vsub3J2Z (a b : Jet2Vec) : Jet2Vec := (a.1.sub b.1, a.2.1.sub b.2.1, a.2.2.sub b.2.2)
private def vscale3J2Z (c : Jet2) (a : Jet2Vec) : Jet2Vec :=
  (Jet2.mul c a.1, Jet2.mul c a.2.1, Jet2.mul c a.2.2)

private def cayleyCompJ2Z (x : Jet2Vec) (dx : Jet2) (y : Jet2Vec) (dy : Jet2) : Jet2Vec × Jet2 :=
  let nx := dot3J2Z x x
  let ny := dot3J2Z y y
  let c1 := Jet2.mul dx ((Jet2.mul dy dy).sub ny)
  let c2 := Jet2.mul dy ((Jet2.mul dx dx).sub nx)
  let c3 := Jet2.scale 2 (Jet2.mul dx dy)
  let nJ := vadd3J2Z (vadd3J2Z (vscale3J2Z c1 x) (vscale3J2Z c2 y)) (vscale3J2Z c3 (cross3J2Z x y))
  let dJ := ((Jet2.mul (Jet2.mul dx dx) (Jet2.mul dy dy)).add (Jet2.mul nx ny)).sub
            (Jet2.mul c3 (dot3J2Z x y))
  (nJ, dJ)

private theorem cayleyCompJ_trunc_fst (t : Int) (x : Jet3Vec) (dx : Jet3) (y : Jet3Vec) (dy : Jet3) :
    (cayleyCompJ t x dx y dy).1.trunc = (cayleyCompJ2Z x.trunc dx.trunc y.trunc dy.trunc).1 := rfl

private theorem cayleyCompJ_trunc_snd (t : Int) (x : Jet3Vec) (dx : Jet3) (y : Jet3Vec) (dy : Jet3) :
    (cayleyCompJ t x dx y dy).2.trunc = (cayleyCompJ2Z x.trunc dx.trunc y.trunc dy.trunc).2 := rfl

private def relHolClosedNum (S : Int) (a b : Vec3Z) : Jet2Vec :=
  let s3 := S * S * S
  let m2s2 := -2 * S * S
  let cr := cross3 a b
  (⟨0, s3 * (b.1 - a.1), m2s2 * cr.1⟩,
   ⟨0, s3 * (b.2.1 - a.2.1), m2s2 * cr.2.1⟩,
   ⟨0, s3 * (b.2.2 - a.2.2), m2s2 * cr.2.2⟩)

private def relHolClosedDen (S : Int) (a b : Vec3Z) : Jet2 :=
  ⟨S * S * (S * S), 0, 2 * S * S * dot3 a b⟩

private abbrev Jet2E := PExpr × PExpr × PExpr
private def j2Const (e : PExpr) : Jet2E := (e, 0, 0)
private def j2Add (x y : Jet2E) : Jet2E := (x.1 + y.1, x.2.1 + y.2.1, x.2.2 + y.2.2)
private def j2Sub (x y : Jet2E) : Jet2E := (x.1 - y.1, x.2.1 - y.2.1, x.2.2 - y.2.2)
private def j2Scale (c : PExpr) (x : Jet2E) : Jet2E := (c * x.1, c * x.2.1, c * x.2.2)
private def j2Mul (x y : Jet2E) : Jet2E :=
  (x.1 * y.1,
   x.1 * y.2.1 + x.2.1 * y.1,
   x.1 * y.2.2 + x.2.1 * y.2.1 + x.2.2 * y.1)

private abbrev Jet2VecE := Jet2E × Jet2E × Jet2E
private def dot3J2 (a b : Jet2VecE) : Jet2E :=
  j2Add (j2Mul a.1 b.1) (j2Add (j2Mul a.2.1 b.2.1) (j2Mul a.2.2 b.2.2))
private def cross3J2 (a b : Jet2VecE) : Jet2VecE :=
  (j2Sub (j2Mul a.2.1 b.2.2) (j2Mul a.2.2 b.2.1),
   j2Sub (j2Mul a.2.2 b.1)   (j2Mul a.1   b.2.2),
   j2Sub (j2Mul a.1   b.2.1) (j2Mul a.2.1 b.1))
private def vadd3J2 (a b : Jet2VecE) : Jet2VecE := (j2Add a.1 b.1, j2Add a.2.1 b.2.1, j2Add a.2.2 b.2.2)
private def vsub3J2 (a b : Jet2VecE) : Jet2VecE := (j2Sub a.1 b.1, j2Sub a.2.1 b.2.1, j2Sub a.2.2 b.2.2)
private def vscale3J2 (c : Jet2E) (a : Jet2VecE) : Jet2VecE := (j2Mul c a.1, j2Mul c a.2.1, j2Mul c a.2.2)

private def compJ2 (x : Jet2VecE) (dx : Jet2E) (y : Jet2VecE) (dy : Jet2E) : Jet2VecE × Jet2E :=
  let nx := dot3J2 x x
  let ny := dot3J2 y y
  let c1 := j2Mul dx (j2Sub (j2Mul dy dy) ny)
  let c2 := j2Mul dy (j2Sub (j2Mul dx dx) nx)
  let c3 := j2Scale 2 (j2Mul dx dy)
  let nJ := vadd3J2 (vadd3J2 (vscale3J2 c1 x) (vscale3J2 c2 y)) (vscale3J2 c3 (cross3J2 x y))
  let dJ := j2Sub (j2Add (j2Mul (j2Mul dx dx) (j2Mul dy dy)) (j2Mul nx ny)) (j2Mul c3 (dot3J2 x y))
  (nJ, dJ)

private def stage1RelHolE : (Jet2VecE × Jet2E) × (Jet2VecE × Jet2E) :=
  let sE := PExpr.var 0
  let aE : Vec3E := (.var 1, .var 2, .var 3)
  let bE : Vec3E := (.var 4, .var 5, .var 6)
  let xJ : Jet2VecE := ((0, -aE.1, 0), (0, -aE.2.1, 0), (0, -aE.2.2, 0))
  let yJ : Jet2VecE := ((0, bE.1, 0), (0, bE.2.1, 0), (0, bE.2.2, 0))
  let sJ := j2Const sE
  let s3 := sE * sE * sE
  let m2s2 := (-2 : PExpr) * sE * sE
  let cr := cross3E aE bE
  let ndC : Jet2VecE :=
    ((0, s3 * (bE.1 - aE.1), m2s2 * cr.1),
     (0, s3 * (bE.2.1 - aE.2.1), m2s2 * cr.2.1),
     (0, s3 * (bE.2.2 - aE.2.2), m2s2 * cr.2.2))
  let ddC : Jet2E := (sE * sE * (sE * sE), 0, 2 * sE * sE * dot3E aE bE)
  (compJ2 xJ sJ yJ sJ, (ndC, ddC))

private theorem jet2_ext {p q : Jet2} (h0 : p.c0 = q.c0) (h1 : p.c1 = q.c1) (h2 : p.c2 = q.c2) : p = q := by
  cases p; cases q; dsimp only at h0 h1 h2; rw [h0, h1, h2]

private theorem vsub3J_trunc (a b : Jet3Vec) : (vsub3J a b).trunc = vsub3J2Z a.trunc b.trunc := rfl
private theorem vscale3J_trunc (t : Int) (c : Jet3) (a : Jet3Vec) :
    (vscale3J t c a).trunc = vscale3J2Z c.trunc a.trunc := rfl
private theorem Jet3_mul_trunc (t : Int) (x y : Jet3) : (Jet3.mul t x y).trunc = Jet2.mul x.trunc y.trunc := rfl
private theorem Jet3_const_trunc (c : Int) : (Jet3.const c).trunc = Jet2.const c := rfl
private theorem Jet3_lin_trunc (c : Int) : (Jet3.lin c).trunc = Jet2.lin c := rfl
private theorem Jet3Vec_trunc_mk (x y z : Jet3) : Jet3Vec.trunc (x, y, z) = (x.trunc, y.trunc, z.trunc) := rfl

private theorem relHolonomyJ2_eq (S : Int) (a b : Vec3Z) :
    let xJ : Jet2Vec := (Jet2.lin (-a.1), Jet2.lin (-a.2.1), Jet2.lin (-a.2.2))
    let yJ : Jet2Vec := (Jet2.lin b.1, Jet2.lin b.2.1, Jet2.lin b.2.2)
    cayleyCompJ2Z xJ (Jet2.const S) yJ (Jet2.const S) = (relHolClosedNum S a b, relHolClosedDen S a b) := by
  dsimp only
  let env := [S, a.1, a.2.1, a.2.2, b.1, b.2.1, b.2.2]
  let lhs := cayleyCompJ2Z (Jet2.lin (-a.1), Jet2.lin (-a.2.1), Jet2.lin (-a.2.2)) (Jet2.const S)
               (Jet2.lin b.1, Jet2.lin b.2.1, Jet2.lin b.2.2) (Jet2.const S)
  have n1 : lhs.1.1 = (relHolClosedNum S a b).1 :=
    jet2_ext
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.1.1 stage1RelHolE.2.1.1.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.1.2.1 stage1RelHolE.2.1.1.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.1.2.2 stage1RelHolE.2.1.1.2.2 rfl)
  have n2 : lhs.1.2.1 = (relHolClosedNum S a b).2.1 :=
    jet2_ext
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.1.1 stage1RelHolE.2.1.2.1.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.1.2.1 stage1RelHolE.2.1.2.1.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.1.2.2 stage1RelHolE.2.1.2.1.2.2 rfl)
  have n3 : lhs.1.2.2 = (relHolClosedNum S a b).2.2 :=
    jet2_ext
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.2.1 stage1RelHolE.2.1.2.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.2.2.1 stage1RelHolE.2.1.2.2.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.1.2.2.2.2 stage1RelHolE.2.1.2.2.2.2 rfl)
  have d0 : lhs.2 = relHolClosedDen S a b :=
    jet2_ext
      (poly_eq_of_norm_eq env stage1RelHolE.1.2.1 stage1RelHolE.2.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.2.2.1 stage1RelHolE.2.2.2.1 rfl)
      (poly_eq_of_norm_eq env stage1RelHolE.1.2.2.2 stage1RelHolE.2.2.2.2 rfl)
  change lhs = (relHolClosedNum S a b, relHolClosedDen S a b)
  rcases hc : lhs with ⟨⟨v1, v2, v3⟩, vd⟩
  rw [hc] at n1 n2 n3 d0
  dsimp only at n1 n2 n3 d0
  rw [n1, n2, n3, d0]

private def rightStage2Z (S u v : Int) (a b : Vec3Z) : Jet2Vec :=
  let sJ := Jet2.const S
  let uJ := Jet2.const u
  let vJ := Jet2.const v
  let yJ : Jet2Vec := (Jet2.lin b.1, Jet2.lin b.2.1, Jet2.lin b.2.2)
  let ndC := relHolClosedNum S a b
  let ddC := relHolClosedDen S a b
  let comp2 := cayleyCompJ2Z yJ sJ (vscale3J2Z uJ ndC) (Jet2.mul vJ ddC)
  let gDir := vecSub3 (vecScale3 (v + u) b) (vecScale3 u a)
  let gJ : Jet2Vec := (Jet2.lin gDir.1, Jet2.lin gDir.2.1, Jet2.lin gDir.2.2)
  vsub3J2Z (vscale3J2Z (Jet2.const (v * S)) comp2.1) (vscale3J2Z comp2.2 gJ)

private def leftStage2Z (S u v : Int) (a b : Vec3Z) : Jet2Vec :=
  let sJ := Jet2.const S
  let uJ := Jet2.const u
  let vJ := Jet2.const v
  let yJ : Jet2Vec := (Jet2.lin b.1, Jet2.lin b.2.1, Jet2.lin b.2.2)
  let ndC := relHolClosedNum S a b
  let ddC := relHolClosedDen S a b
  let comp2 := cayleyCompJ2Z (vscale3J2Z uJ ndC) (Jet2.mul vJ ddC) yJ sJ
  let gDir := vecSub3 (vecScale3 (v + u) b) (vecScale3 u a)
  let gJ : Jet2Vec := (Jet2.lin gDir.1, Jet2.lin gDir.2.1, Jet2.lin gDir.2.2)
  vsub3J2Z (vscale3J2Z (Jet2.const (v * S)) comp2.1) (vscale3J2Z comp2.2 gJ)

private theorem rightDefect_trunc_eq (t S u v : Int) (a b : Vec3Z) :
    (rightStrangDefectJet t S u v a b).trunc = rightStage2Z S u v a b := by
  unfold rightStrangDefectJet rightStage2Z
  dsimp only
  simp only [vsub3J_trunc, vscale3J_trunc, cayleyCompJ_trunc_fst, cayleyCompJ_trunc_snd,
    Jet3_mul_trunc, Jet3_const_trunc, Jet3_lin_trunc, Jet3Vec_trunc_mk]
  have hr := relHolonomyJ2_eq S a b
  dsimp only at hr
  rw [hr]

private theorem leftDefect_trunc_eq (t S u v : Int) (a b : Vec3Z) :
    (leftOrderedDefectJet t S u v a b).trunc = leftStage2Z S u v a b := by
  unfold leftOrderedDefectJet leftStage2Z
  dsimp only
  simp only [vsub3J_trunc, vscale3J_trunc, cayleyCompJ_trunc_fst, cayleyCompJ_trunc_snd,
    Jet3_mul_trunc, Jet3_const_trunc, Jet3_lin_trunc, Jet3Vec_trunc_mk]
  have hr := relHolonomyJ2_eq S a b
  dsimp only at hr
  rw [hr]

private def rightStage2E : Jet2VecE :=
  let sE := PExpr.var 0; let uE := PExpr.var 1; let vE := PExpr.var 2
  let aE : Vec3E := (.var 3, .var 4, .var 5); let bE : Vec3E := (.var 6, .var 7, .var 8)
  let sJ := j2Const sE; let uJ := j2Const uE; let vJ := j2Const vE
  let yJ : Jet2VecE := ((0, bE.1, 0), (0, bE.2.1, 0), (0, bE.2.2, 0))
  let s3 := sE * sE * sE; let m2s2 := (-2 : PExpr) * sE * sE; let cr := cross3E aE bE
  let ndC : Jet2VecE :=
    ((0, s3 * (bE.1 - aE.1), m2s2 * cr.1),
     (0, s3 * (bE.2.1 - aE.2.1), m2s2 * cr.2.1),
     (0, s3 * (bE.2.2 - aE.2.2), m2s2 * cr.2.2))
  let ddC : Jet2E := (sE * sE * (sE * sE), 0, 2 * sE * sE * dot3E aE bE)
  let comp2 := compJ2 yJ sJ (vscale3J2 uJ ndC) (j2Mul vJ ddC)
  let gx := (vE + uE) * bE.1 - uE * aE.1
  let gy := (vE + uE) * bE.2.1 - uE * aE.2.1
  let gz := (vE + uE) * bE.2.2 - uE * aE.2.2
  let gJ : Jet2VecE := ((0, gx, 0), (0, gy, 0), (0, gz, 0))
  vsub3J2 (vscale3J2 (j2Const (vE * sE)) comp2.1) (vscale3J2 comp2.2 gJ)

private def leftStage2E : Jet2VecE :=
  let sE := PExpr.var 0; let uE := PExpr.var 1; let vE := PExpr.var 2
  let aE : Vec3E := (.var 3, .var 4, .var 5); let bE : Vec3E := (.var 6, .var 7, .var 8)
  let sJ := j2Const sE; let uJ := j2Const uE; let vJ := j2Const vE
  let yJ : Jet2VecE := ((0, bE.1, 0), (0, bE.2.1, 0), (0, bE.2.2, 0))
  let s3 := sE * sE * sE; let m2s2 := (-2 : PExpr) * sE * sE; let cr := cross3E aE bE
  let ndC : Jet2VecE :=
    ((0, s3 * (bE.1 - aE.1), m2s2 * cr.1),
     (0, s3 * (bE.2.1 - aE.2.1), m2s2 * cr.2.1),
     (0, s3 * (bE.2.2 - aE.2.2), m2s2 * cr.2.2))
  let ddC : Jet2E := (sE * sE * (sE * sE), 0, 2 * sE * sE * dot3E aE bE)
  let comp2 := compJ2 (vscale3J2 uJ ndC) (j2Mul vJ ddC) yJ sJ
  let gx := (vE + uE) * bE.1 - uE * aE.1
  let gy := (vE + uE) * bE.2.1 - uE * aE.2.1
  let gz := (vE + uE) * bE.2.2 - uE * aE.2.2
  let gJ : Jet2VecE := ((0, gx, 0), (0, gy, 0), (0, gz, 0))
  vsub3J2 (vscale3J2 (j2Const (vE * sE)) comp2.1) (vscale3J2 comp2.2 gJ)

def pow9 (S : Int) : Int :=
  let s2 := S * S
  let s4 := s2 * s2
  let s8 := s4 * s4
  s8 * S

private def leftQuadExpectedE : Vec3E :=
  let sE := PExpr.var 0; let uE := PExpr.var 1; let vE := PExpr.var 2
  let aE : Vec3E := (.var 3, .var 4, .var 5); let bE : Vec3E := (.var 6, .var 7, .var 8)
  let s2 := sE * sE; let s4 := s2 * s2; let s8 := s4 * s4
  let s9 := s8 * sE
  vscale3E ((-4 : PExpr) * vE * vE * uE * s9) (cross3E aE bE)

/-- **THEOREM 2A (Palindromic Strang Degree-0..2 BCH Cancellation & Exact `O(t^3)` Remainder)**:
    For EVERY tangent pair `a, b ∈ Z^3`, EVERY guidance numerator/denominator `(u, v)` (`ω = u / v`),
    EVERY scale `S`, and EVERY step `t`, right-ordered Cayley-scaled composition `C(w_R) · C(ω w_Δ)`
    (`w_Δ = (-w_1) ⊕_C w_R`, modeling `ω`-scaled relative Cayley tangent rather than Lie fractional power
    `(U_1^{-1} U_R)^ω`) vs linear Cayley guidance `w_guide = (1 + ω) w_R - ω w_1` has:
    - Exact `0` degree-0, `0` degree-1, AND **`0` degree-2 BCH commutator defect**, and
    - Exact `t^3` cubic remainder factorization `rightStrangDefectVec t S u v a b = t^3 • rem`! -/
theorem thm2a_right_strang_bch_quadratic_zero (t S u v : Int) (a b : Vec3Z) :
    let j := rightStrangDefectJet t S u v a b
    (j.1.c0 = 0 ∧ j.2.1.c0 = 0 ∧ j.2.2.c0 = 0) ∧
    (j.1.c1 = 0 ∧ j.2.1.c1 = 0 ∧ j.2.2.c1 = 0) ∧
    (j.1.c2 = 0 ∧ j.2.1.c2 = 0 ∧ j.2.2.c2 = 0) ∧
    ∃ rem : Vec3Z, rightStrangDefectVec t S u v a b = vecScale3 (t * t * t) rem := by
  intro j
  let env := [S, u, v, a.1, a.2.1, a.2.2, b.1, b.2.1, b.2.2]
  have htr : j.trunc = rightStage2Z S u v a b := rightDefect_trunc_eq t S u v a b
  have s10 : (rightStage2Z S u v a b).1.c0 = 0 := poly_eq_of_norm_eq env rightStage2E.1.1 0 rfl
  have s20 : (rightStage2Z S u v a b).2.1.c0 = 0 := poly_eq_of_norm_eq env rightStage2E.2.1.1 0 rfl
  have s30 : (rightStage2Z S u v a b).2.2.c0 = 0 := poly_eq_of_norm_eq env rightStage2E.2.2.1 0 rfl
  have s11 : (rightStage2Z S u v a b).1.c1 = 0 := poly_eq_of_norm_eq env rightStage2E.1.2.1 0 rfl
  have s21 : (rightStage2Z S u v a b).2.1.c1 = 0 := poly_eq_of_norm_eq env rightStage2E.2.1.2.1 0 rfl
  have s31 : (rightStage2Z S u v a b).2.2.c1 = 0 := poly_eq_of_norm_eq env rightStage2E.2.2.2.1 0 rfl
  have s12 : (rightStage2Z S u v a b).1.c2 = 0 := poly_eq_of_norm_eq env rightStage2E.1.2.2 0 rfl
  have s22 : (rightStage2Z S u v a b).2.1.c2 = 0 := poly_eq_of_norm_eq env rightStage2E.2.1.2.2 0 rfl
  have s32 : (rightStage2Z S u v a b).2.2.c2 = 0 := poly_eq_of_norm_eq env rightStage2E.2.2.2.2 0 rfl
  have h10 : j.1.c0 = 0 := by have e := congrArg (fun p => p.1.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s10]
  have h20 : j.2.1.c0 = 0 := by have e := congrArg (fun p => p.2.1.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s20]
  have h30 : j.2.2.c0 = 0 := by have e := congrArg (fun p => p.2.2.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s30]
  have h11 : j.1.c1 = 0 := by have e := congrArg (fun p => p.1.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s11]
  have h21 : j.2.1.c1 = 0 := by have e := congrArg (fun p => p.2.1.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s21]
  have h31 : j.2.2.c1 = 0 := by have e := congrArg (fun p => p.2.2.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s31]
  have h12 : j.1.c2 = 0 := by have e := congrArg (fun p => p.1.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s12]
  have h22 : j.2.1.c2 = 0 := by have e := congrArg (fun p => p.2.1.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s22]
  have h32 : j.2.2.c2 = 0 := by have e := congrArg (fun p => p.2.2.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s32]
  refine ⟨⟨h10, h20, h30⟩, ⟨h11, h21, h31⟩, ⟨h12, h22, h32⟩, ⟨(j.1.rem, j.2.1.rem, j.2.2.rem), ?_⟩⟩
  rw [← rightStrangDefectJet_eval_eq]
  change (j.1.eval t, j.2.1.eval t, j.2.2.eval t) = vecScale3 (t * t * t) (j.1.rem, j.2.1.rem, j.2.2.rem)
  unfold Jet3.eval vecScale3
  rw [h10, h20, h30, h11, h21, h31, h12, h22, h32]
  simp only [Int.mul_zero, Int.zero_add]

/-- **THEOREM 2B (Non-Palindromic Left Ordering Retains Exact Nonzero `-4 v^2 u S^9 (a × b)` Degree-2 Defect)**:
    Under left-ordered Cayley-scaled composition `C(ω w_Δ) · C(w_R)`, the degree-0 and degree-1 defects vanish,
    the exact degree-2 BCH commutator defect equals `-4 v^2 u S^9 (a × b)`, and on the
    orthogonal witness `a = (1, 0, 0), b = (0, 1, 0), u = 1, v = 1, S = 1` the full quadratic defect tuple is `(0, 0, -4)`! -/
theorem thm2b_left_ordered_bch_quadratic_defect_nonzero (t S u v : Int) (a b : Vec3Z) :
    let j := leftOrderedDefectJet t S u v a b
    let quadExpected := vecScale3 (-4 * v * v * u * pow9 S) (cross3 a b)
    let w := leftOrderedDefectJet 1 1 1 1 (1, 0, 0) (0, 1, 0)
    (j.1.c0 = 0 ∧ j.2.1.c0 = 0 ∧ j.2.2.c0 = 0) ∧
    (j.1.c1 = 0 ∧ j.2.1.c1 = 0 ∧ j.2.2.c1 = 0) ∧
    (j.1.c2, j.2.1.c2, j.2.2.c2) = quadExpected ∧
    ((w.1.c2, w.2.1.c2, w.2.2.c2) = (0, 0, -4)) := by
  intro j quadExpected w
  let env := [S, u, v, a.1, a.2.1, a.2.2, b.1, b.2.1, b.2.2]
  have htr : j.trunc = leftStage2Z S u v a b := leftDefect_trunc_eq t S u v a b
  have s10 : (leftStage2Z S u v a b).1.c0 = 0 := poly_eq_of_norm_eq env leftStage2E.1.1 0 rfl
  have s20 : (leftStage2Z S u v a b).2.1.c0 = 0 := poly_eq_of_norm_eq env leftStage2E.2.1.1 0 rfl
  have s30 : (leftStage2Z S u v a b).2.2.c0 = 0 := poly_eq_of_norm_eq env leftStage2E.2.2.1 0 rfl
  have s11 : (leftStage2Z S u v a b).1.c1 = 0 := poly_eq_of_norm_eq env leftStage2E.1.2.1 0 rfl
  have s21 : (leftStage2Z S u v a b).2.1.c1 = 0 := poly_eq_of_norm_eq env leftStage2E.2.1.2.1 0 rfl
  have s31 : (leftStage2Z S u v a b).2.2.c1 = 0 := poly_eq_of_norm_eq env leftStage2E.2.2.2.1 0 rfl
  have s12 : (leftStage2Z S u v a b).1.c2 = quadExpected.1 := poly_eq_of_norm_eq env leftStage2E.1.2.2 leftQuadExpectedE.1 rfl
  have s22 : (leftStage2Z S u v a b).2.1.c2 = quadExpected.2.1 := poly_eq_of_norm_eq env leftStage2E.2.1.2.2 leftQuadExpectedE.2.1 rfl
  have s32 : (leftStage2Z S u v a b).2.2.c2 = quadExpected.2.2 := poly_eq_of_norm_eq env leftStage2E.2.2.2.2 leftQuadExpectedE.2.2 rfl
  have h10 : j.1.c0 = 0 := by have e := congrArg (fun p => p.1.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s10]
  have h20 : j.2.1.c0 = 0 := by have e := congrArg (fun p => p.2.1.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s20]
  have h30 : j.2.2.c0 = 0 := by have e := congrArg (fun p => p.2.2.c0) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s30]
  have h11 : j.1.c1 = 0 := by have e := congrArg (fun p => p.1.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s11]
  have h21 : j.2.1.c1 = 0 := by have e := congrArg (fun p => p.2.1.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s21]
  have h31 : j.2.2.c1 = 0 := by have e := congrArg (fun p => p.2.2.c1) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s31]
  have h12 : j.1.c2 = quadExpected.1 := by have e := congrArg (fun p => p.1.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s12]
  have h22 : j.2.1.c2 = quadExpected.2.1 := by have e := congrArg (fun p => p.2.1.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s22]
  have h32 : j.2.2.c2 = quadExpected.2.2 := by have e := congrArg (fun p => p.2.2.c2) htr; dsimp [Jet3Vec.trunc, Jet3.trunc] at e; rw [e, s32]
  refine ⟨⟨h10, h20, h30⟩, ⟨h11, h21, h31⟩, ?_, rfl⟩
  rw [h12, h22, h32]

/-- Squared rational quaternion distance `||p / dp - q / dq||^2` represented as an exact `QPair`. -/
def quatDistSqQ (p : QuatZ) (dp : Int) (q : QuatZ) (dq : Int) : QPair :=
  let diff : QuatZ :=
    (p.1 * dq - q.1 * dp,
     p.2.1 * dq - q.2.1 * dp,
     p.2.2.1 * dq - q.2.2.1 * dp,
     p.2.2.2 * dq - q.2.2.2 * dp)
  ⟨quatNormSq diff, (dp * dq) * (dp * dq)⟩

/-- Constructive `SU(2, Q)` witness (`w_1 = (1/(5N),0,0), w_R = (0,1/(4N),0)` at `ω = 1`):
    Across dyadic scales `N ∈ {1, 2, 4, 8}`, `0 < E_right < E_left` AND `60 * E_right(1/8) < E_right(1/4)`! -/
theorem thm2c_dyadic_su2_chordal_hierarchy_witness :
    let evalErrs (N : Int) :=
      let w1 : Vec3Z := (4, 0, 0)
      let wr : Vec3Z := (0, 5, 0)
      let d  : Int   := 20 * N
      let u1 := cayleyQuatNum w1 d; let du1 := cayleyQuatDen w1 d
      let ur := cayleyQuatNum wr d; let dur := cayleyQuatDen wr d
      let dU := quatMul (quatConj u1) ur; let ddU := du1 * dur
      let uRight := quatMul ur dU; let duRight := dur * ddU
      let uLeft  := quatMul dU ur; let duLeft  := ddU * dur
      let wGuide : Vec3Z := vecSub3 (vecScale3 2 wr) w1
      let uLin   := cayleyQuatNum wGuide d; let duLin := cayleyQuatDen wGuide d
      (quatDistSqQ uLin duLin uRight duRight, quatDistSqQ uLin duLin uLeft duLeft)
    let (er1, el1) := evalErrs 1
    let (er2, el2) := evalErrs 2
    let (er4, el4) := evalErrs 4
    let (er8, el8) := evalErrs 8
    QLt ⟨0, 1⟩ er1 ∧ QLt er1 el1 ∧
    QLt ⟨0, 1⟩ er2 ∧ QLt er2 el2 ∧
    QLt ⟨0, 1⟩ er4 ∧ QLt er4 el4 ∧
    QLt ⟨0, 1⟩ er8 ∧ QLt er8 el8 ∧
    QLt (qmul ⟨60, 1⟩ er8) er4 := by
  decide

/-! ## 3. Theorem 3: Acute-Ball Denominator Safety Floors (`9/25 > 0` & `81/625 > 0`) -/

private theorem nonneg_of_pos_mul_nonneg {a b : Int} (ha : 0 < a) (hab : 0 ≤ a * b) : 0 ≤ b :=
  if hb : 0 ≤ b then hb
  else
    have hbn : b < 0 := by omega
    have hneg : a * b < 0 := Int.mul_neg_of_pos_of_neg ha hbn
    by omega

private def sosPlusPair : PExpr × PExpr :=
  let a : Vec3E := (.var 0, .var 1, .var 2); let da := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let db := PExpr.var 7
  let deltaA := 16 * da * da - 25 * normSq3E a
  let deltaB := 16 * db * db - 25 * normSq3E b
  (625 * (da * da * (db * db) + normSq3E a * normSq3E b + 2 * da * db * dot3E a b) - 81 * (da * da) * (db * db),
   625 * normSq3E (vadd3E (vscale3E db a) (vscale3E da b)) +
   9 * (da * da) * deltaB + 9 * (db * db) * deltaA + deltaA * deltaB)

private def sosMinusPair : PExpr × PExpr :=
  let a : Vec3E := (.var 0, .var 1, .var 2); let da := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let db := PExpr.var 7
  let deltaA := 16 * da * da - 25 * normSq3E a
  let deltaB := 16 * db * db - 25 * normSq3E b
  (625 * (da * da * (db * db) + normSq3E a * normSq3E b - 2 * da * db * dot3E a b) - 81 * (da * da) * (db * db),
   625 * normSq3E (vsub3E (vscale3E db a) (vscale3E da b)) +
   9 * (da * da) * deltaB + 9 * (db * db) * deltaA + deltaA * deltaB)

/-- Exact square-root-free algebraic sum-of-nonnegatives identity underlying `81/625 > 0`! -/
theorem acute_ball_cayley_denom_sos_identity (a : Vec3Z) (da : Int) (b : Vec3Z) (db : Int) :
    let deltaA := 16 * da * da - 25 * normSq3 a
    let deltaB := 16 * db * db - 25 * normSq3 b
    (625 * (da * da * (db * db) + normSq3 a * normSq3 b + 2 * da * db * dot3 a b) - 81 * (da * da) * (db * db) =
      625 * normSq3 (vecAdd3 (vecScale3 db a) (vecScale3 da b)) +
      9 * (da * da) * deltaB + 9 * (db * db) * deltaA + deltaA * deltaB) ∧
    (625 * (da * da * (db * db) + normSq3 a * normSq3 b - 2 * da * db * dot3 a b) - 81 * (da * da) * (db * db) =
      625 * normSq3 (vecSub3 (vecScale3 db a) (vecScale3 da b)) +
      9 * (da * da) * deltaB + 9 * (db * db) * deltaA + deltaA * deltaB) :=
  ⟨poly_eq_of_norm_eq [a.1, a.2.1, a.2.2, da, b.1, b.2.1, b.2.2, db] sosPlusPair.1 sosPlusPair.2 rfl,
   poly_eq_of_norm_eq [a.1, a.2.1, a.2.2, da, b.1, b.2.1, b.2.2, db] sosMinusPair.1 sosMinusPair.2 rfl⟩

private def normScalePair : PExpr × PExpr :=
  let a : Vec3E := (.var 0, .var 1, .var 2); let u := PExpr.var 3
  (25 * normSq3E (vscale3E u a), (u * u) * (25 * normSq3E a))

private def rhsScalePair : PExpr × PExpr :=
  let v := PExpr.var 0; let da := PExpr.var 1
  (16 * (v * da) * (v * da), (v * v) * (16 * da * da))

/-- Scaling a vector in the `4/5` acute spinor ball by `ω = u / v ∈ [0, 1]` stays inside the `4/5` ball! -/
theorem acute_ball_closed_under_unit_omega (a : Vec3Z) (da u v : Int)
    (_hda : 0 < da) (hu0 : 0 ≤ u) (huv : u ≤ v)
    (ha : 25 * normSq3 a ≤ 16 * da * da) :
    25 * normSq3 (vecScale3 u a) ≤ 16 * (v * da) * (v * da) := by
  have hnorm_sc : 25 * normSq3 (vecScale3 u a) = (u * u) * (25 * normSq3 a) :=
    poly_eq_of_norm_eq [a.1, a.2.1, a.2.2, u] normScalePair.1 normScalePair.2 rfl
  have hrhs_sc : 16 * (v * da) * (v * da) = (v * v) * (16 * da * da) :=
    poly_eq_of_norm_eq [v, da] rhsScalePair.1 rhsScalePair.2 rfl
  rw [hnorm_sc, hrhs_sc]
  have hu2_nonneg : 0 ≤ u * u := int_sq_nonneg u
  have hv_nonneg : 0 ≤ v := Int.le_trans hu0 huv
  have hu2_le_v2 : u * u ≤ v * v :=
    Int.le_trans (Int.mul_le_mul_of_nonneg_left huv hu0) (Int.mul_le_mul_of_nonneg_right huv hv_nonneg)
  have h16_nonneg : 0 ≤ 16 * da * da := by
    have h1 : 16 * da * da = 16 * (da * da) := Int.mul_assoc 16 da da
    rw [h1]
    exact Int.mul_nonneg (by decide : (0 : Int) ≤ 16) (int_sq_nonneg da)
  exact Int.le_trans
    (Int.mul_le_mul_of_nonneg_left ha hu2_nonneg)
    (Int.mul_le_mul_of_nonneg_right hu2_le_v2 h16_nonneg)

private def spinor925Pair : PExpr × PExpr :=
  let x : Vec3E := (.var 0, .var 1, .var 2); let dx := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let dR := PExpr.var 7
  ((2 * dx * dR) * ((dx * dR + dot3E x b) * 25 - 9 * (dx * dR)),
   25 * normSq3E (vadd3E (vscale3E dR x) (vscale3E dx b)) +
   (dR * dR) * (16 * dx * dx - 25 * normSq3E x) +
   (dx * dx) * (16 * dR * dR - 25 * normSq3E b))

/-- **THEOREM 3A (Acute-Ball Spinor Half-Tangent Denominator Floor `1 + <w_1, w_R> ≥ 9/25 > 0`)**:
    For ALL `w_1 = a / d1, w_R = b / dR` with `||w_1||^2 ≤ 16/25, ||w_R||^2 ≤ 16/25` and
    ALL `ω = u / v ∈ [0, 1]`, BOTH `1 + <w_1, w_R> ≥ 9/25 > 0` AND `1 + <ω w_1, w_R> ≥ 9/25 > 0`! -/
theorem thm3a_acute_ball_spinor_denom_ge_9_25 (a : Vec3Z) (d1 : Int) (b : Vec3Z) (dR : Int) (u v : Int)
    (hd1 : 0 < d1) (hdR : 0 < dR) (hu0 : 0 ≤ u) (huv : u ≤ v) (hv : 0 < v)
    (ha : 25 * normSq3 a ≤ 16 * d1 * d1)
    (hb : 25 * normSq3 b ≤ 16 * dR * dR) :
    QLe ⟨9, 25⟩ ⟨d1 * dR + dot3 a b, d1 * dR⟩ ∧
    0 < d1 * dR + dot3 a b ∧
    QLe ⟨9, 25⟩ ⟨(v * d1) * dR + dot3 (vecScale3 u a) b, (v * d1) * dR⟩ ∧
    0 < (v * d1) * dR + dot3 (vecScale3 u a) b := by
  have core : ∀ (x : Vec3Z) (dx : Int), 0 < dx → 25 * normSq3 x ≤ 16 * dx * dx →
      QLe ⟨9, 25⟩ ⟨dx * dR + dot3 x b, dx * dR⟩ ∧ 0 < dx * dR + dot3 x b := by
    intro x dx hdx hx
    unfold QLe; dsimp only
    have hid : (2 * dx * dR) * ((dx * dR + dot3 x b) * 25 - 9 * (dx * dR)) =
        25 * normSq3 (vecAdd3 (vecScale3 dR x) (vecScale3 dx b)) +
        (dR * dR) * (16 * dx * dx - 25 * normSq3 x) +
        (dx * dx) * (16 * dR * dR - 25 * normSq3 b) :=
      poly_eq_of_norm_eq [x.1, x.2.1, x.2.2, dx, b.1, b.2.1, b.2.2, dR] spinor925Pair.1 spinor925Pair.2 rfl
    have hnonneg_rhs : 0 ≤ (2 * dx * dR) * ((dx * dR + dot3 x b) * 25 - 9 * (dx * dR)) := by
      rw [hid]
      have t1 : 0 ≤ 25 * normSq3 (vecAdd3 (vecScale3 dR x) (vecScale3 dx b)) :=
        Int.mul_nonneg (by decide : (0 : Int) ≤ 25) (normSq3_nonneg _)
      have t2 : 0 ≤ (dR * dR) * (16 * dx * dx - 25 * normSq3 x) :=
        Int.mul_nonneg (int_sq_nonneg dR) (Int.sub_nonneg_of_le hx)
      have t3 : 0 ≤ (dx * dx) * (16 * dR * dR - 25 * normSq3 b) :=
        Int.mul_nonneg (int_sq_nonneg dx) (Int.sub_nonneg_of_le hb)
      exact Int.add_nonneg (Int.add_nonneg t1 t2) t3
    have h2pos : 0 < 2 * dx * dR := Int.mul_pos (Int.mul_pos (by decide : (0 : Int) < 2) hdx) hdR
    have hdiff : 0 ≤ (dx * dR + dot3 x b) * 25 - 9 * (dx * dR) :=
      nonneg_of_pos_mul_nonneg h2pos hnonneg_rhs
    have hpos_dxdR : 0 < dx * dR := Int.mul_pos hdx hdR
    refine ⟨by omega, by omega⟩
  rcases core a d1 hd1 ha with ⟨h1, h2⟩
  have hua := acute_ball_closed_under_unit_omega a d1 u v hd1 hu0 huv ha
  rcases core (vecScale3 u a) (v * d1) (Int.mul_pos hv hd1) hua with ⟨h3, h4⟩
  exact ⟨h1, h2, h3, h4⟩

private def negDenPair : PExpr × PExpr :=
  let x : Vec3E := (.var 0, .var 1, .var 2); let dx := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let dR := PExpr.var 7
  (cayleyCompDenE (vneg3E x) dx b dR,
   dx * dx * (dR * dR) + normSq3E x * normSq3E b + 2 * dx * dR * dot3E x b)

/-- **THEOREM 3B (Acute-Ball `SU(2)` Cayley Composition Denominator Floor `≥ 81/625 > 0`)**:
    For ALL `w_1 = a / d1, w_R = b / dR` in the acute spinor ball `||w|| ≤ 4/5` and ALL `ω = u/v ∈ [0, 1]`:
    1. Relative holonomy `C(w_1)^† C(w_R)` denominator `cayleyCompDen (-a) d1 b dR` satisfies `≥ 81/625 > 0`,
    2. Forward composition `C(w_1) C(w_R)` denominator `cayleyCompDen a d1 b dR` satisfies `≥ 81/625 > 0`,
    3. `ω`-scaled `C(ω w_1)^† C(w_R)` denominator `cayleyCompDen (-(u a)) (v d1) b dR` satisfies `≥ 81/625 > 0`! -/
theorem thm3b_acute_ball_cayley_denom_ge_81_625 (a : Vec3Z) (d1 : Int) (b : Vec3Z) (dR : Int) (u v : Int)
    (hd1 : 0 < d1) (hdR : 0 < dR) (hu0 : 0 ≤ u) (huv : u ≤ v) (hv : 0 < v)
    (ha : 25 * normSq3 a ≤ 16 * d1 * d1)
    (hb : 25 * normSq3 b ≤ 16 * dR * dR) :
    QLe ⟨81, 625⟩ ⟨cayleyCompDen (vecNeg3 a) d1 b dR, (d1 * d1) * (dR * dR)⟩ ∧
    0 < cayleyCompDen (vecNeg3 a) d1 b dR ∧
    QLe ⟨81, 625⟩ ⟨cayleyCompDen a d1 b dR, (d1 * d1) * (dR * dR)⟩ ∧
    0 < cayleyCompDen a d1 b dR ∧
    QLe ⟨81, 625⟩ ⟨cayleyCompDen (vecNeg3 (vecScale3 u a)) (v * d1) b dR, ((v * d1) * (v * d1)) * (dR * dR)⟩ ∧
    0 < cayleyCompDen (vecNeg3 (vecScale3 u a)) (v * d1) b dR := by
  have core : ∀ (x : Vec3Z) (dx : Int), 0 < dx → 25 * normSq3 x ≤ 16 * dx * dx →
      QLe ⟨81, 625⟩ ⟨cayleyCompDen (vecNeg3 x) dx b dR, (dx * dx) * (dR * dR)⟩ ∧
      0 < cayleyCompDen (vecNeg3 x) dx b dR ∧
      QLe ⟨81, 625⟩ ⟨cayleyCompDen x dx b dR, (dx * dx) * (dR * dR)⟩ ∧
      0 < cayleyCompDen x dx b dR := by
    intro x dx hdx hx
    rcases acute_ball_cayley_denom_sos_identity x dx b dR with ⟨hsosPlus, hsosMinus⟩
    have hneg_den : cayleyCompDen (vecNeg3 x) dx b dR =
        dx * dx * (dR * dR) + normSq3 x * normSq3 b + 2 * dx * dR * dot3 x b :=
      poly_eq_of_norm_eq [x.1, x.2.1, x.2.2, dx, b.1, b.2.1, b.2.2, dR] negDenPair.1 negDenPair.2 rfl
    have hdelA : 0 ≤ 16 * dx * dx - 25 * normSq3 x := Int.sub_nonneg_of_le hx
    have hdelB : 0 ≤ 16 * dR * dR - 25 * normSq3 b := Int.sub_nonneg_of_le hb
    have p1 : 0 ≤ 9 * (dx * dx) * (16 * dR * dR - 25 * normSq3 b) :=
      Int.mul_nonneg (Int.mul_nonneg (by decide : (0 : Int) ≤ 9) (int_sq_nonneg dx)) hdelB
    have p2 : 0 ≤ 9 * (dR * dR) * (16 * dx * dx - 25 * normSq3 x) :=
      Int.mul_nonneg (Int.mul_nonneg (by decide : (0 : Int) ≤ 9) (int_sq_nonneg dR)) hdelA
    have p3 : 0 ≤ (16 * dx * dx - 25 * normSq3 x) * (16 * dR * dR - 25 * normSq3 b) :=
      Int.mul_nonneg hdelA hdelB
    have hnnPlus : 0 ≤ 625 * cayleyCompDen (vecNeg3 x) dx b dR - 81 * (dx * dx) * (dR * dR) := by
      rw [hneg_den, hsosPlus]
      have p0 : 0 ≤ 625 * normSq3 (vecAdd3 (vecScale3 dR x) (vecScale3 dx b)) :=
        Int.mul_nonneg (by decide : (0 : Int) ≤ 625) (normSq3_nonneg _)
      exact Int.add_nonneg (Int.add_nonneg (Int.add_nonneg p0 p1) p2) p3
    have hnnMinus : 0 ≤ 625 * cayleyCompDen x dx b dR - 81 * (dx * dx) * (dR * dR) := by
      unfold cayleyCompDen
      rw [hsosMinus]
      have p0 : 0 ≤ 625 * normSq3 (vecSub3 (vecScale3 dR x) (vecScale3 dx b)) :=
        Int.mul_nonneg (by decide : (0 : Int) ≤ 625) (normSq3_nonneg _)
      exact Int.add_nonneg (Int.add_nonneg (Int.add_nonneg p0 p1) p2) p3
    have hd2pos : 0 < (dx * dx) * (dR * dR) := Int.mul_pos (int_sq_pos_of_pos hdx) (int_sq_pos_of_pos hdR)
    unfold QLe; dsimp only
    have ha_assoc : 81 * ((dx * dx) * (dR * dR)) = 81 * (dx * dx) * (dR * dR) :=
      (Int.mul_assoc 81 (dx * dx) (dR * dR)).symm
    have hc1 : 625 * cayleyCompDen (vecNeg3 x) dx b dR = cayleyCompDen (vecNeg3 x) dx b dR * 625 :=
      Int.mul_comm 625 _
    have hc2 : 625 * cayleyCompDen x dx b dR = cayleyCompDen x dx b dR * 625 :=
      Int.mul_comm 625 _
    refine ⟨by omega, by omega, by omega, by omega⟩
  rcases core a d1 hd1 ha with ⟨h1, h2, h3, h4⟩
  have hua := acute_ball_closed_under_unit_omega a d1 u v hd1 hu0 huv ha
  rcases core (vecScale3 u a) (v * d1) (Int.mul_pos hv hd1) hua with ⟨h5, h6, _, _⟩
  exact ⟨h1, h2, h3, h4, h5, h6⟩

/-- Tightness witness for Theorem 3: antiparallel boundary points `w_1 = (4/5,0,0), w_R = (-4/5,0,0)`
    lie inside the `4/5` acute ball (`25 * ||w||^2 ≤ 16 * 5^2`) and attain `9/25` and `81/625` exactly! -/
theorem thm3c_acute_ball_81_625_tight :
    let w1 : Vec3Z := (4, 0, 0)
    let wr : Vec3Z := (-4, 0, 0)
    25 * normSq3 w1 ≤ 16 * 5 * 5 ∧
    25 * normSq3 wr ≤ 16 * 5 * 5 ∧
    QEq ⟨5 * 5 + dot3 w1 wr, 5 * 5⟩ ⟨9, 25⟩ ∧
    QEq ⟨cayleyCompDen (vecNeg3 w1) 5 wr 5, (5 * 5) * (5 * 5)⟩ ⟨81, 625⟩ := by
  decide

/-! ## 4. Theorem 4: Commutative / Collinear Exact Collapse (All Orders!) -/

private def compCommPair : (PExpr × PExpr) × (Vec3E × Vec3E) :=
  let x : Vec3E := (.var 0, .var 1, .var 2); let dx := PExpr.var 3
  let y : Vec3E := (.var 4, .var 5, .var 6); let dy := PExpr.var 7
  ((cayleyCompDenE x dx y dy, cayleyCompDenE y dy x dx),
   (vsub3E (cayleyCompNumE x dx y dy) (cayleyCompNumE y dy x dx),
    vscale3E (4 * dx * dy) (cross3E x y)))

/-- Exact commutator formula for `cayleyCompNum`:
    `cayleyCompDen` is symmetric, and `cayleyCompNum x dx y dy - cayleyCompNum y dy x dx = 4 dx dy (x × y)`. -/
theorem cayley_comp_commutator_exact (x : Vec3Z) (dx : Int) (y : Vec3Z) (dy : Int) :
    cayleyCompDen x dx y dy = cayleyCompDen y dy x dx ∧
    vecSub3 (cayleyCompNum x dx y dy) (cayleyCompNum y dy x dx) =
      vecScale3 (4 * dx * dy) (cross3 x y) := by
  let env := [x.1, x.2.1, x.2.2, dx, y.1, y.2.1, y.2.2, dy]
  have hd : cayleyCompDen x dx y dy = cayleyCompDen y dy x dx :=
    poly_eq_of_norm_eq env compCommPair.1.1 compCommPair.1.2 rfl
  have h1 : (vecSub3 (cayleyCompNum x dx y dy) (cayleyCompNum y dy x dx)).1 = (vecScale3 (4 * dx * dy) (cross3 x y)).1 :=
    poly_eq_of_norm_eq env compCommPair.2.1.1 compCommPair.2.2.1 rfl
  have h2 : (vecSub3 (cayleyCompNum x dx y dy) (cayleyCompNum y dy x dx)).2.1 = (vecScale3 (4 * dx * dy) (cross3 x y)).2.1 :=
    poly_eq_of_norm_eq env compCommPair.2.1.2.1 compCommPair.2.2.2.1 rfl
  have h3 : (vecSub3 (cayleyCompNum x dx y dy) (cayleyCompNum y dy x dx)).2.2 = (vecScale3 (4 * dx * dy) (cross3 x y)).2.2 :=
    poly_eq_of_norm_eq env compCommPair.2.1.2.2 compCommPair.2.2.2.2 rfl
  refine ⟨hd, ?_⟩
  rcases hL : vecSub3 (cayleyCompNum x dx y dy) (cayleyCompNum y dy x dx) with ⟨l1, l2, l3⟩
  rcases hR : vecScale3 (4 * dx * dy) (cross3 x y) with ⟨r1, r2, r3⟩
  rw [hL, hR] at h1 h2 h3
  dsimp only at h1 h2 h3
  rw [h1, h2, h3]

private theorem vec_eq_of_sub_zero {u v : Vec3Z} (h : vecSub3 u v = (0, 0, 0)) : u = v := by
  rcases u with ⟨u1, u2, u3⟩; rcases v with ⟨v1, v2, v3⟩
  have h1 : u1 - v1 = 0 := congrArg Prod.fst h
  have h2 : u2 - v2 = 0 := congrArg (fun p => p.2.1) h
  have h3 : u3 - v3 = 0 := congrArg (fun p => p.2.2) h
  have e1 : u1 = v1 := by omega
  have e2 : u2 = v2 := by omega
  have e3 : u3 = v3 := by omega
  rw [e1, e2, e3]

private def quatCommPair : (PExpr × PExpr) × (Vec3E × Vec3E) :=
  let a : Vec3E := (.var 0, .var 1, .var 2); let d1 := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let dR := PExpr.var 7
  let ab := qmulE (cayleyQuatNumE a d1) (cayleyQuatNumE b dR)
  let ba := qmulE (cayleyQuatNumE b dR) (cayleyQuatNumE a d1)
  ((ab.1, ba.1),
   ((ab.2.1 - ba.2.1, ab.2.2.1 - ba.2.2.1, ab.2.2.2 - ba.2.2.2),
    vscale3E (8 * d1 * dR) (cross3E a b)))

/-- **THEOREM 4A (`SU(2, Q)` Quaternion Commutativity When `[w_1, w_R] = 0`)**:
    Whenever `cross3 w1 wR = (0, 0, 0)`, `C(w1 / d1) * C(wR / dR) = C(wR / dR) * C(w1 / d1)` identically! -/
theorem thm4a_commuting_cayley_quat_commute (w1 : Vec3Z) (d1 : Int) (wR : Vec3Z) (dR : Int)
    (hcomm : cross3 w1 wR = (0, 0, 0)) :
    quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR) =
      quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1) := by
  let env := [w1.1, w1.2.1, w1.2.2, d1, wR.1, wR.2.1, wR.2.2, dR]
  have h0 : (quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR)).1 =
      (quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1)).1 :=
    poly_eq_of_norm_eq env quatCommPair.1.1 quatCommPair.1.2 rfl
  have d1_eq : (quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR)).2.1 -
      (quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1)).2.1 = (vecScale3 (8 * d1 * dR) (cross3 w1 wR)).1 :=
    poly_eq_of_norm_eq env quatCommPair.2.1.1 quatCommPair.2.2.1 rfl
  have d2_eq : (quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR)).2.2.1 -
      (quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1)).2.2.1 = (vecScale3 (8 * d1 * dR) (cross3 w1 wR)).2.1 :=
    poly_eq_of_norm_eq env quatCommPair.2.1.2.1 quatCommPair.2.2.2.1 rfl
  have d3_eq : (quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR)).2.2.2 -
      (quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1)).2.2.2 = (vecScale3 (8 * d1 * dR) (cross3 w1 wR)).2.2 :=
    poly_eq_of_norm_eq env quatCommPair.2.1.2.2 quatCommPair.2.2.2.2 rfl
  have hrhs : vecScale3 (8 * d1 * dR) (cross3 w1 wR) = (0, 0, 0) := by
    rw [hcomm]; simp only [vecScale3, Int.mul_zero]
  rw [hrhs] at d1_eq d2_eq d3_eq
  dsimp only at d1_eq d2_eq d3_eq
  rcases hL : quatMul (cayleyQuatNum w1 d1) (cayleyQuatNum wR dR) with ⟨l0, l1, l2, l3⟩
  rcases hR : quatMul (cayleyQuatNum wR dR) (cayleyQuatNum w1 d1) with ⟨r0, r1, r2, r3⟩
  rw [hL, hR] at h0 d1_eq d2_eq d3_eq
  dsimp only at h0 d1_eq d2_eq d3_eq
  have e1 : l1 = r1 := by omega
  have e2 : l2 = r2 := by omega
  have e3 : l3 = r3 := by omega
  rw [h0, e1, e2, e3]

private def crossNegPair : Vec3E × Vec3E :=
  let a : Vec3E := (.var 0, .var 1, .var 2)
  let b : Vec3E := (.var 3, .var 4, .var 5)
  (cross3E (vneg3E a) b, vneg3E (cross3E a b))

private def commDeltaCrossPair : Vec3E × Vec3E :=
  let a : Vec3E := (.var 0, .var 1, .var 2); let d1 := PExpr.var 3
  let b : Vec3E := (.var 4, .var 5, .var 6); let dR := PExpr.var 7
  let u := PExpr.var 8
  let na := vneg3E a
  let c1 := d1 * (dR * dR - normSq3E b)
  let c2 := dR * (d1 * d1 - normSq3E na)
  let nd0 := vadd3E (vscale3E c1 na) (vscale3E c2 b)
  (cross3E b (vscale3E u nd0), vscale3E (u * c1) (cross3E a b))

/-- **THEOREM 4B (Commutative `[w_1, w_R] = 0` Exact Non-Perturbative Left == Right Collapse)**:
    Whenever `cross3 w1 wR = (0, 0, 0)`, right-ordered `w_R ⊕_C (ω w_Δ)` and left-ordered
    `(ω w_Δ) ⊕_C w_R` coincide **component-for-component to all orders** for ALL `ω = u / v`! -/
theorem thm4b_commutative_left_right_collapse (w1 : Vec3Z) (d1 : Int) (wR : Vec3Z) (dR : Int)
    (u v : Int) (hcomm : cross3 w1 wR = (0, 0, 0)) :
    let nd := cayleyCompNum (vecNeg3 w1) d1 wR dR
    let dd := cayleyCompDen (vecNeg3 w1) d1 wR dR
    cayleyCompNum wR dR (vecScale3 u nd) (v * dd) = cayleyCompNum (vecScale3 u nd) (v * dd) wR dR ∧
    cayleyCompDen wR dR (vecScale3 u nd) (v * dd) = cayleyCompDen (vecScale3 u nd) (v * dd) wR dR := by
  intro nd dd
  rcases cayley_comp_commutator_exact wR dR (vecScale3 u nd) (v * dd) with ⟨hden, hsub⟩
  refine ⟨?_, hden⟩
  have hcr : cross3 wR (vecScale3 u nd) = (0, 0, 0) := by
    let env := [w1.1, w1.2.1, w1.2.2, d1, wR.1, wR.2.1, wR.2.2, dR, u]
    have hcr_neg : cross3 (vecNeg3 w1) wR = (0, 0, 0) := by
      let env6 := [w1.1, w1.2.1, w1.2.2, wR.1, wR.2.1, wR.2.2]
      have n1 : (cross3 (vecNeg3 w1) wR).1 = (vecNeg3 (cross3 w1 wR)).1 :=
        poly_eq_of_norm_eq env6 crossNegPair.1.1 crossNegPair.2.1 rfl
      have n2 : (cross3 (vecNeg3 w1) wR).2.1 = (vecNeg3 (cross3 w1 wR)).2.1 :=
        poly_eq_of_norm_eq env6 crossNegPair.1.2.1 crossNegPair.2.2.1 rfl
      have n3 : (cross3 (vecNeg3 w1) wR).2.2 = (vecNeg3 (cross3 w1 wR)).2.2 :=
        poly_eq_of_norm_eq env6 crossNegPair.1.2.2 crossNegPair.2.2.2 rfl
      have hnr : vecNeg3 (cross3 w1 wR) = (0, 0, 0) := by rw [hcomm]; rfl
      rw [hnr] at n1 n2 n3
      dsimp only at n1 n2 n3
      rcases hcn : cross3 (vecNeg3 w1) wR with ⟨c1, c2, c3⟩
      rw [hcn] at n1 n2 n3; dsimp only at n1 n2 n3; rw [n1, n2, n3]
    have hnd_eq : nd = vecAdd3 (vecScale3 (d1 * (dR * dR - normSq3 wR)) (vecNeg3 w1))
                               (vecScale3 (dR * (d1 * d1 - normSq3 (vecNeg3 w1))) wR) := by
      unfold nd cayleyCompNum
      dsimp only
      rw [hcr_neg]
      have hz0 : vecScale3 (2 * d1 * dR) (0, 0, 0) = (0, 0, 0) := by simp only [vecScale3, Int.mul_zero]
      rw [hz0]
      rcases vecAdd3 (vecScale3 (d1 * (dR * dR - normSq3 wR)) (vecNeg3 w1))
                     (vecScale3 (dR * (d1 * d1 - normSq3 (vecNeg3 w1))) wR) with ⟨p1, p2, p3⟩
      simp only [vecAdd3, Int.add_zero]
    rw [hnd_eq]
    let nd0 := vecAdd3 (vecScale3 (d1 * (dR * dR - normSq3 wR)) (vecNeg3 w1))
                       (vecScale3 (dR * (d1 * d1 - normSq3 (vecNeg3 w1))) wR)
    have k1 : (cross3 wR (vecScale3 u nd0)).1 = (vecScale3 (u * (d1 * (dR * dR - normSq3 wR))) (cross3 w1 wR)).1 :=
      poly_eq_of_norm_eq env commDeltaCrossPair.1.1 commDeltaCrossPair.2.1 rfl
    have k2 : (cross3 wR (vecScale3 u nd0)).2.1 = (vecScale3 (u * (d1 * (dR * dR - normSq3 wR))) (cross3 w1 wR)).2.1 :=
      poly_eq_of_norm_eq env commDeltaCrossPair.1.2.1 commDeltaCrossPair.2.2.1 rfl
    have k3 : (cross3 wR (vecScale3 u nd0)).2.2 = (vecScale3 (u * (d1 * (dR * dR - normSq3 wR))) (cross3 w1 wR)).2.2 :=
      poly_eq_of_norm_eq env commDeltaCrossPair.1.2.2 commDeltaCrossPair.2.2.2 rfl
    have hrhs0 : vecScale3 (u * (d1 * (dR * dR - normSq3 wR))) (cross3 w1 wR) = (0, 0, 0) := by
      rw [hcomm]; simp only [vecScale3, Int.mul_zero]
    rw [hrhs0] at k1 k2 k3
    dsimp only at k1 k2 k3
    rcases hC : cross3 wR (vecScale3 u nd0) with ⟨c1v, c2v, c3v⟩
    rw [hC] at k1 k2 k3
    dsimp only at k1 k2 k3
    rw [k1, k2, k3]
  rw [hcr] at hsub
  have hz : vecScale3 (4 * dR * (v * dd)) (0, 0, 0) = (0, 0, 0) := by
    simp only [vecScale3, Int.mul_zero]
  rw [hz] at hsub
  exact vec_eq_of_sub_zero hsub

private def colCrossE : Vec3E :=
  let e : Vec3E := (.var 2, .var 3, .var 4)
  cross3E (vscale3E (.var 0) e) (vscale3E (.var 1) e)

private def colNegCrossE : Vec3E :=
  let e : Vec3E := (.var 2, .var 3, .var 4)
  cross3E (vneg3E (vscale3E (.var 0) e)) (vscale3E (.var 1) e)

/-- **THEOREM 4C (Collinear Vectors: Right-Ordered, Left-Ordered, & Cayley-Sum Collapse Identically)**:
    For ANY collinear vectors `w_1 = α • e`, `w_R = β • e` (`e ∈ Z^3`):
    1. `[w_1, w_R] = cross3 w_1 w_R = (0, 0, 0)`,
    2. Right-ordered `w_R ⊕_C (ω w_Δ)` == Left-ordered `(ω w_Δ) ⊕_C w_R` for ALL `ω = u/v`,
    3. At `ω = 1` (`u = 1, v = 1`), right-ordered `w_R ⊕_C ((-w_1) ⊕_C w_R)`, left-ordered
       `((-w_1) ⊕_C w_R) ⊕_C w_R`, AND iterated Cayley-sum `(w_R ⊕_C (-w_1)) ⊕_C w_R` are IDENTICAL! -/
theorem thm4c_collinear_left_right_cayley_sum_exact (α β d1 dR u v : Int) (e : Vec3Z) :
    let w1 := vecScale3 α e
    let wR := vecScale3 β e
    let ndR := cayleyCompNum (vecNeg3 w1) d1 wR dR
    let ddR := cayleyCompDen (vecNeg3 w1) d1 wR dR
    let ndL := cayleyCompNum wR dR (vecNeg3 w1) d1
    let ddL := cayleyCompDen wR dR (vecNeg3 w1) d1
    cross3 w1 wR = (0, 0, 0) ∧
    (cayleyCompNum wR dR (vecScale3 u ndR) (v * ddR) = cayleyCompNum (vecScale3 u ndR) (v * ddR) wR dR ∧
     cayleyCompDen wR dR (vecScale3 u ndR) (v * ddR) = cayleyCompDen (vecScale3 u ndR) (v * ddR) wR dR) ∧
    (cayleyCompNum wR dR ndR ddR = cayleyCompNum ndL ddL wR dR ∧
     cayleyCompDen wR dR ndR ddR = cayleyCompDen ndL ddL wR dR) := by
  intro w1 wR ndR ddR ndL ddL
  let env := [α, β, e.1, e.2.1, e.2.2]
  have hcr : cross3 w1 wR = (0, 0, 0) := by
    have c1 : (cross3 w1 wR).1 = 0 := poly_eq_of_norm_eq env colCrossE.1 0 rfl
    have c2 : (cross3 w1 wR).2.1 = 0 := poly_eq_of_norm_eq env colCrossE.2.1 0 rfl
    have c3 : (cross3 w1 wR).2.2 = 0 := poly_eq_of_norm_eq env colCrossE.2.2 0 rfl
    rcases h : cross3 w1 wR with ⟨x, y, z⟩
    rw [h] at c1 c2 c3; dsimp only at c1 c2 c3; rw [c1, c2, c3]
  have hlr := thm4b_commutative_left_right_collapse w1 d1 wR dR u v hcr
  have hcr_neg : cross3 (vecNeg3 w1) wR = (0, 0, 0) := by
    have c1 : (cross3 (vecNeg3 w1) wR).1 = 0 := poly_eq_of_norm_eq env colNegCrossE.1 0 rfl
    have c2 : (cross3 (vecNeg3 w1) wR).2.1 = 0 := poly_eq_of_norm_eq env colNegCrossE.2.1 0 rfl
    have c3 : (cross3 (vecNeg3 w1) wR).2.2 = 0 := poly_eq_of_norm_eq env colNegCrossE.2.2 0 rfl
    rcases h : cross3 (vecNeg3 w1) wR with ⟨x, y, z⟩
    rw [h] at c1 c2 c3; dsimp only at c1 c2 c3; rw [c1, c2, c3]
  rcases cayley_comp_commutator_exact (vecNeg3 w1) d1 wR dR with ⟨hdd, hnd_sub⟩
  rw [hcr_neg] at hnd_sub
  have hz : vecScale3 (4 * d1 * dR) (0, 0, 0) = (0, 0, 0) := by simp only [vecScale3, Int.mul_zero]
  rw [hz] at hnd_sub
  have hnd_eq : ndR = ndL := vec_eq_of_sub_zero hnd_sub
  have hdd_eq : ddR = ddL := hdd
  rcases thm4b_commutative_left_right_collapse w1 d1 wR dR 1 1 hcr with ⟨hw1_n, hw1_d⟩
  have h1u : vecScale3 1 ndR = ndR := by
    rcases ndR with ⟨x, y, z⟩; simp only [vecScale3, Int.one_mul]
  have h1v : 1 * ddR = ddR := Int.one_mul ddR
  rw [h1u, h1v] at hw1_n hw1_d
  refine ⟨hcr, hlr, ?_⟩
  rw [← hnd_eq, ← hdd_eq]
  exact ⟨hw1_n, hw1_d⟩

end LoopCDStrang
