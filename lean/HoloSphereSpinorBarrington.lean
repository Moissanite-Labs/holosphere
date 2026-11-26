-- Project HoloSphere / Track S119: Exact Spinor-Barrington 2I subset SU(2) Theorems (#46-#48)
-- Pure-Init Lean 4.36.0 Formalization (Taking Verified Library 46 -> 49/49)
-- Zero external imports, zero Mathlib, zero IEEE-754 floats, zero sorry,
-- zero Classical.choice, zero Lean.ofReduceBool (no native_decide), zero warnings.
--
-- Public Theorems proved:
--   #46 p46_spinor_barrington_c10_exact_closure          (Sign-free C_10^+ spinor AND/NOT closure)
--   #47 p47_su2_commutator_trace_and_order5_obstruction  (Universal Re([a,b]) & ord-5 obstruction)
--   #48 p48_spinor_barrington_acute_cayley_ball          (Ordered Z[sqrt(5)] ||omega||^2 < (4/5)^2)

namespace HoloSphereSpinorBarrington

-- ============================================================================
-- 0. Exact Ordered Ring Z[sqrt(5)] & Quaternions Over Int and Z[sqrt(5)]
-- ============================================================================

/-- Exact quadratic integer a + b * sqrt(5) in Z[sqrt(5)]. -/
structure ZSqrt5 where
  a : Int
  b : Int
  deriving DecidableEq, Repr

def z5Zero : ZSqrt5 := ⟨0, 0⟩
def z5One : ZSqrt5 := ⟨1, 0⟩

def z5Neg (x : ZSqrt5) : ZSqrt5 :=
  ⟨-x.a, -x.b⟩

def z5Add (x y : ZSqrt5) : ZSqrt5 :=
  ⟨x.a + y.a, x.b + y.b⟩

def z5Sub (x y : ZSqrt5) : ZSqrt5 :=
  ⟨x.a - y.a, x.b - y.b⟩

def z5Scale (s : Int) (x : ZSqrt5) : ZSqrt5 :=
  ⟨s * x.a, s * x.b⟩

/-- Exact ring product (x.a + x.b*sqrt(5))*(y.a + y.b*sqrt(5)) =
    (x.a*y.a + 5*x.b*y.b) + (x.a*y.b + x.b*y.a)*sqrt(5). -/
def z5Mul (x y : ZSqrt5) : ZSqrt5 :=
  ⟨x.a * y.a + 5 * x.b * y.b, x.a * y.b + x.b * y.a⟩

def z5Sq (x : ZSqrt5) : ZSqrt5 :=
  z5Mul x x

def z5Div4 (x : ZSqrt5) : ZSqrt5 :=
  ⟨x.a / 4, x.b / 4⟩

/-- Exact positivity a + b*sqrt(5) > 0 decided purely over Int via squaring comparisons:
      - both >= 0 and a + b > 0, OR
      - a > 0, b < 0, and 5*b^2 < a^2, OR
      - a < 0, b > 0, and a^2 < 5*b^2. -/
def z5Pos (z : ZSqrt5) : Prop :=
  (0 ≤ z.a ∧ 0 ≤ z.b ∧ 0 < z.a + z.b) ∨
  (0 < z.a ∧ z.b < 0 ∧ 5 * (z.b * z.b) < z.a * z.a) ∨
  (z.a < 0 ∧ 0 < z.b ∧ z.a * z.a < 5 * (z.b * z.b))

instance (z : ZSqrt5) : Decidable (z5Pos z) := by
  unfold z5Pos
  infer_instance

/-- Strict order x < y in Z[sqrt(5)] iff y - x > 0. -/
def z5Lt (x y : ZSqrt5) : Prop :=
  z5Pos (z5Sub y x)

instance (x y : ZSqrt5) : Decidable (z5Lt x y) := by
  unfold z5Lt
  infer_instance

private theorem z5_ext {x y : ZSqrt5} (ha : x.a = y.a) (hb : x.b = y.b) : x = y := by
  cases x
  cases y
  simp only at ha hb
  subst ha hb
  rfl

private theorem z5Add_comm (x y : ZSqrt5) : z5Add x y = z5Add y x := by
  apply z5_ext <;> simp only [z5Add, Int.add_comm]

private theorem z5Add_assoc (x y w : ZSqrt5) :
    z5Add (z5Add x y) w = z5Add x (z5Add y w) := by
  apply z5_ext <;> simp only [z5Add, Int.add_assoc]

private theorem z5Add_left_comm (x y w : ZSqrt5) :
    z5Add x (z5Add y w) = z5Add y (z5Add x w) := by
  apply z5_ext <;> simp only [z5Add] <;> omega

private theorem z5Add_zero (x : ZSqrt5) : z5Add x z5Zero = x := by
  apply z5_ext <;> simp only [z5Add, z5Zero, Int.add_zero]

private theorem z5Zero_add (x : ZSqrt5) : z5Add z5Zero x = x := by
  apply z5_ext <;> simp only [z5Add, z5Zero, Int.zero_add]

private theorem z5Sub_eq_add_neg (x y : ZSqrt5) : z5Sub x y = z5Add x (z5Neg y) := by
  apply z5_ext <;> simp only [z5Sub, z5Add, z5Neg, Int.sub_eq_add_neg]

private theorem z5Neg_add (x y : ZSqrt5) : z5Neg (z5Add x y) = z5Add (z5Neg x) (z5Neg y) := by
  apply z5_ext <;> simp only [z5Neg, z5Add, Int.neg_add]

private theorem z5Neg_neg (x : ZSqrt5) : z5Neg (z5Neg x) = x := by
  apply z5_ext <;> simp only [z5Neg, Int.neg_neg]

private theorem z5Add_neg_self (x : ZSqrt5) : z5Add x (z5Neg x) = z5Zero := by
  apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega

private theorem z5Neg_add_self (x : ZSqrt5) : z5Add (z5Neg x) x = z5Zero := by
  apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega

private theorem z5Mul_comm (x y : ZSqrt5) : z5Mul x y = z5Mul y x := by
  apply z5_ext
  · simp only [z5Mul, Int.mul_comm, Int.mul_left_comm]
  · simp only [z5Mul, Int.mul_comm, Int.add_comm]

private theorem z5Mul_assoc (x y w : ZSqrt5) :
    z5Mul (z5Mul x y) w = z5Mul x (z5Mul y w) := by
  apply z5_ext <;>
    simp only [z5Mul, Int.add_mul, Int.mul_add, Int.mul_assoc, Int.mul_left_comm] <;>
    omega

private theorem z5Mul_left_comm (x y w : ZSqrt5) :
    z5Mul x (z5Mul y w) = z5Mul y (z5Mul x w) := by
  rw [← z5Mul_assoc, z5Mul_comm x y, z5Mul_assoc]

private theorem z5Mul_add (x y w : ZSqrt5) :
    z5Mul x (z5Add y w) = z5Add (z5Mul x y) (z5Mul x w) := by
  apply z5_ext <;> simp only [z5Mul, z5Add, Int.mul_add] <;> omega

private theorem z5Add_mul (x y w : ZSqrt5) :
    z5Mul (z5Add x y) w = z5Add (z5Mul x w) (z5Mul y w) := by
  rw [z5Mul_comm, z5Mul_add, z5Mul_comm w x, z5Mul_comm w y]

private theorem z5Mul_neg_right (x y : ZSqrt5) :
    z5Mul x (z5Neg y) = z5Neg (z5Mul x y) := by
  apply z5_ext <;> simp only [z5Mul, z5Neg, Int.mul_neg] <;> omega

private theorem z5Mul_neg_left (x y : ZSqrt5) :
    z5Mul (z5Neg x) y = z5Neg (z5Mul x y) := by
  rw [z5Mul_comm, z5Mul_neg_right, z5Mul_comm]

private theorem z5Mul_zero (x : ZSqrt5) : z5Mul x z5Zero = z5Zero := by
  apply z5_ext <;> simp only [z5Mul, z5Zero, Int.mul_zero, Int.add_zero]

private theorem z5Zero_mul (x : ZSqrt5) : z5Mul z5Zero x = z5Zero := by
  apply z5_ext <;> simp only [z5Mul, z5Zero, Int.zero_mul, Int.mul_zero, Int.add_zero]

private theorem z5Scale_two (x : ZSqrt5) : z5Scale 2 x = z5Add x x := by
  apply z5_ext <;> simp only [z5Scale, z5Add] <;> omega

/-- Hamilton quaternion over Z[sqrt(5)] (scale 4 represents unit 2I elements with normSq = 16). -/
structure QuatZ5 where
  w : ZSqrt5
  x : ZSqrt5
  y : ZSqrt5
  z : ZSqrt5
  deriving DecidableEq, Repr

def q5ScaleInt (s : Int) (q : QuatZ5) : QuatZ5 :=
  ⟨z5Scale s q.w, z5Scale s q.x, z5Scale s q.y, z5Scale s q.z⟩

def q5ScaleZ5 (s : ZSqrt5) (q : QuatZ5) : QuatZ5 :=
  ⟨z5Mul s q.w, z5Mul s q.x, z5Mul s q.y, z5Mul s q.z⟩

def q5Div4 (q : QuatZ5) : QuatZ5 :=
  ⟨z5Div4 q.w, z5Div4 q.x, z5Div4 q.y, z5Div4 q.z⟩

def q5Neg (q : QuatZ5) : QuatZ5 :=
  ⟨z5Neg q.w, z5Neg q.x, z5Neg q.y, z5Neg q.z⟩

def q5Conj (q : QuatZ5) : QuatZ5 :=
  ⟨q.w, z5Neg q.x, z5Neg q.y, z5Neg q.z⟩

/-- Unscaled ring product in QuatZ5 (maps two scale-4 unit quaternions to scale 16). -/
def q5MulRaw (p q : QuatZ5) : QuatZ5 :=
  ⟨z5Sub (z5Sub (z5Sub (z5Mul p.w q.w) (z5Mul p.x q.x)) (z5Mul p.y q.y)) (z5Mul p.z q.z),
   z5Sub (z5Add (z5Add (z5Mul p.w q.x) (z5Mul p.x q.w)) (z5Mul p.y q.z)) (z5Mul p.z q.y),
   z5Add (z5Add (z5Sub (z5Mul p.w q.y) (z5Mul p.x q.z)) (z5Mul p.y q.w)) (z5Mul p.z q.x),
   z5Add (z5Sub (z5Add (z5Mul p.w q.z) (z5Mul p.x q.y)) (z5Mul p.y q.x)) (z5Mul p.z q.w)⟩

/-- Exact scale-4 group multiplication on 4 * (2I): (4*p * 4*q) / 4 = 4*(p*q). -/
def uMul (p q : QuatZ5) : QuatZ5 :=
  q5Div4 (q5MulRaw p q)

def q5Inner (p q : QuatZ5) : ZSqrt5 :=
  z5Add (z5Add (z5Add (z5Mul p.w q.w) (z5Mul p.x q.x)) (z5Mul p.y q.y)) (z5Mul p.z q.z)

def q5VecNormSq (q : QuatZ5) : ZSqrt5 :=
  z5Add (z5Add (z5Sq q.x) (z5Sq q.y)) (z5Sq q.z)

def q5NormSq (q : QuatZ5) : ZSqrt5 :=
  q5Inner q q

def q5CrossNormSq (a b : QuatZ5) : ZSqrt5 :=
  let cx := z5Sub (z5Mul a.y b.z) (z5Mul a.z b.y)
  let cy := z5Sub (z5Mul a.z b.x) (z5Mul a.x b.z)
  let cz := z5Sub (z5Mul a.x b.y) (z5Mul a.y b.x)
  z5Add (z5Add (z5Sq cx) (z5Sq cy)) (z5Sq cz)

def q5CommRaw (a b : QuatZ5) : QuatZ5 :=
  q5MulRaw (q5MulRaw a b) (q5MulRaw (q5Conj a) (q5Conj b))

/-- Scale-4 group power in 2I. -/
def uPow (q : QuatZ5) : Nat → QuatZ5
  | 0 => ⟨⟨4, 0⟩, z5Zero, z5Zero, z5Zero⟩
  | k + 1 => uMul (uPow q k) q

/-- Pure unscaled ring power (zero division!). -/
def rawPow (q : QuatZ5) : Nat → QuatZ5
  | 0 => ⟨z5One, z5Zero, z5Zero, z5Zero⟩
  | k + 1 => q5MulRaw (rawPow q k) q

/-- Scale-4 conjugation g * x * conj(g) in 2I. -/
def uConjBy (g x : QuatZ5) : QuatZ5 :=
  uMul (uMul g x) (q5Conj g)

/-- Scale-4 group commutator [a, b] = a * b * conj(a) * conj(b) in 2I. -/
def uComm (a b : QuatZ5) : QuatZ5 :=
  uMul (uMul a b) (uMul (q5Conj a) (q5Conj b))

private theorem q5_ext {p q : QuatZ5}
    (hw : p.w = q.w) (hx : p.x = q.x) (hy : p.y = q.y) (hz : p.z = q.z) : p = q := by
  cases p
  cases q
  simp only at hw hx hy hz
  subst hw hx hy hz
  rfl

-- ============================================================================
-- Canonical Scale-4 Witnesses from Track S119 (barrington_su2_dag_compiler.py)
-- ============================================================================

/-- +I and -I scaled by 4 (normSq = 16). -/
def I4 : QuatZ5 := ⟨⟨4, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩
def negI4 : QuatZ5 := ⟨⟨-4, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩

/-- Acute order-10 witness 4 * tau0 = (1 + sqrt(5), -1 + sqrt(5), 2, 0) in C_10^+ subset 4*(2I). -/
def tau0_4 : QuatZ5 := ⟨⟨1, 1⟩, ⟨-1, 1⟩, ⟨2, 0⟩, ⟨0, 0⟩⟩

/-- Conjugate generator 4 * g_alpha = (1 + sqrt(5), 1 - sqrt(5), 2, 0). -/
def gAlpha_4 : QuatZ5 := ⟨⟨1, 1⟩, ⟨1, -1⟩, ⟨2, 0⟩, ⟨0, 0⟩⟩

/-- Conjugate generator 4 * g_beta = (2, -2, 2, 2). -/
def gBeta_4 : QuatZ5 := ⟨⟨2, 0⟩, ⟨-2, 0⟩, ⟨2, 0⟩, ⟨2, 0⟩⟩

/-- Order-5 witness 4 * sigma0 = -4 * tau0 (sigma0^5 = +I, Re(sigma0) < 0). -/
def sigma0_4 : QuatZ5 := q5Neg tau0_4

/-- Spinor-Barrington AND gate on {I4, tau0_4}:
    AND(u, v) = [g_alpha * u * conj(g_alpha), g_beta * v * conj(g_beta)]. -/
def spinorAND (u v : QuatZ5) : QuatZ5 :=
  uComm (uConjBy gAlpha_4 u) (uConjBy gBeta_4 v)

/-- Spinor-Barrington NOT gate on {I4, tau0_4}:
    NOT(u) = tau0 * conj(u). -/
def spinorNOT (u : QuatZ5) : QuatZ5 :=
  uMul tau0_4 (q5Conj u)

-- ============================================================================
-- Theorem #46: Sign-Free Spinor-Barrington C_10^+ Exact Closure & Completeness
-- ============================================================================

/-- Theorem #46 (Track S119 — Sign-Free Spinor-Barrington C_10^+ Exact Closure):
    Over exact Z[sqrt(5)] quaternions at scale 4 (normSq = 16, zero floats):
      1. Exact unit norms: normSq(tau0_4) = normSq(gAlpha_4) = normSq(gBeta_4) = (16, 0),
      2. Exact spinor order 10 (both in lossless scale-4 group uPow AND pure division-free rawPow):
         tau0^5 = -I, tau0^10 = +I, and tau0^k != +I for all proper divisors k in {1, 2, 5},
      3. Exact lossless scale-4 integrality & sign-free commutator closure:
         for alpha = g_alpha * tau0 * conj(g_alpha) and beta = g_beta * tau0 * conj(g_beta),
         every intermediate scale-4 product satisfies 4 * uMul(X, Y) = q5MulRaw(X, Y) (zero remainder!),
         normSq(alpha) = normSq(beta) = (16, 0), AND [alpha, beta] = +tau0_4
         (zero sign canonicalization, zero -I leakage!),
      4. Complete Boolean gate truth table on {+I, +tau0}:
         AND(I, I) = I, AND(I, tau0) = I, AND(tau0, I) = I, AND(tau0, tau0) = tau0,
         NOT(I) = tau0, and NOT(tau0) = I. -/
theorem p46_spinor_barrington_c10_exact_closure :
    (q5NormSq tau0_4 = ⟨16, 0⟩ ∧
     q5NormSq gAlpha_4 = ⟨16, 0⟩ ∧
     q5NormSq gBeta_4 = ⟨16, 0⟩) ∧
    (uPow tau0_4 5 = negI4 ∧
     uPow tau0_4 10 = I4 ∧
     uPow tau0_4 1 ≠ I4 ∧
     uPow tau0_4 2 ≠ I4 ∧
     uPow tau0_4 5 ≠ I4 ∧
     rawPow tau0_4 5 = q5ScaleInt 256 negI4 ∧
     rawPow tau0_4 10 = q5ScaleInt 262144 I4) ∧
    (let alpha := uConjBy gAlpha_4 tau0_4
     let beta := uConjBy gBeta_4 tau0_4
     q5ScaleInt 4 (uMul gAlpha_4 tau0_4) = q5MulRaw gAlpha_4 tau0_4 ∧
     q5ScaleInt 4 alpha = q5MulRaw (uMul gAlpha_4 tau0_4) (q5Conj gAlpha_4) ∧
     q5ScaleInt 4 (uMul gBeta_4 tau0_4) = q5MulRaw gBeta_4 tau0_4 ∧
     q5ScaleInt 4 beta = q5MulRaw (uMul gBeta_4 tau0_4) (q5Conj gBeta_4) ∧
     q5NormSq alpha = ⟨16, 0⟩ ∧
     q5NormSq beta = ⟨16, 0⟩ ∧
     q5ScaleInt 4 (uMul alpha beta) = q5MulRaw alpha beta ∧
     q5ScaleInt 4 (uMul (q5Conj alpha) (q5Conj beta)) = q5MulRaw (q5Conj alpha) (q5Conj beta) ∧
     q5ScaleInt 4 (uComm alpha beta) =
       q5MulRaw (uMul alpha beta) (uMul (q5Conj alpha) (q5Conj beta)) ∧
     uComm alpha beta = tau0_4) ∧
    (spinorAND I4 I4 = I4 ∧
     spinorAND I4 tau0_4 = I4 ∧
     spinorAND tau0_4 I4 = I4 ∧
     spinorAND tau0_4 tau0_4 = tau0_4 ∧
     spinorNOT I4 = tau0_4 ∧
     spinorNOT tau0_4 = I4) := by
  decide

-- ============================================================================
-- Theorem #47: Universal Commutator Real-Part Law & Order-5 Z_2 Obstruction
-- ============================================================================

/-- Integer quaternion for the universal ring-level Re(a*b*conj(a)*conj(b)) theorem. -/
structure QuatZ where
  w : Int
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

def qzScale (s : Int) (q : QuatZ) : QuatZ :=
  ⟨s * q.w, s * q.x, s * q.y, s * q.z⟩

def qzConj (q : QuatZ) : QuatZ :=
  ⟨q.w, -q.x, -q.y, -q.z⟩

def qzMul (p q : QuatZ) : QuatZ :=
  ⟨p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
   p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
   p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
   p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w⟩

def qzInner (p q : QuatZ) : Int :=
  p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z

def qzNormSq (q : QuatZ) : Int :=
  qzInner q q

def qzCrossNormSq (a b : QuatZ) : Int :=
  let cx := a.y * b.z - a.z * b.y
  let cy := a.z * b.x - a.x * b.z
  let cz := a.x * b.y - a.y * b.x
  cx * cx + cy * cy + cz * cz

def qzCommRaw (a b : QuatZ) : QuatZ :=
  qzMul (qzMul a b) (qzMul (qzConj a) (qzConj b))

private theorem qz_ext {p q : QuatZ}
    (hw : p.w = q.w) (hx : p.x = q.x) (hy : p.y = q.y) (hz : p.z = q.z) : p = q := by
  cases p
  cases q
  simp only at hw hx hy hz
  subst hw hx hy hz
  rfl

-- Fast ZSqrt5 ring-level proof of Euler norm & commutator real-part (NEVER unfolds z5Mul to Int!)
private theorem q5_conj_ba (a b : QuatZ5) :
    q5MulRaw (q5Conj a) (q5Conj b) = q5Conj (q5MulRaw b a) := by
  apply q5_ext <;>
    dsimp only [q5MulRaw, q5Conj] <;>
    simp only [z5Sub_eq_add_neg, z5Neg_add, z5Neg_neg, z5Mul_neg_right, z5Mul_comm] <;>
    (apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega)

private theorem q5_re_mul_conj (p q : QuatZ5) :
    (q5MulRaw p (q5Conj q)).w = q5Inner p q := by
  dsimp only [q5MulRaw, q5Conj, q5Inner]
  simp only [z5Sub_eq_add_neg, z5Mul_neg_right, z5Neg_neg]

private theorem q5_left_adj (x y z : QuatZ5) :
    q5Inner x (q5MulRaw y z) = q5Inner (q5MulRaw (q5Conj y) x) z := by
  dsimp only [q5Inner, q5MulRaw, q5Conj]
  simp only [z5Sub_eq_add_neg, z5Mul_add, z5Mul_neg_right, z5Neg_neg,
             z5Mul_left_comm, z5Mul_comm]
  apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega

private theorem q5_mul_assoc (x y z : QuatZ5) :
    q5MulRaw (q5MulRaw x y) z = q5MulRaw x (q5MulRaw y z) := by
  apply q5_ext <;>
    dsimp only [q5MulRaw] <;>
    simp only [z5Sub_eq_add_neg, z5Mul_add, z5Add_mul,
               z5Mul_neg_left, z5Mul_neg_right, z5Neg_add, z5Neg_neg, z5Mul_assoc] <;>
    (apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega)

private theorem q5_conj_mul_self (x : QuatZ5) :
    q5MulRaw (q5Conj x) x = ⟨q5NormSq x, z5Zero, z5Zero, z5Zero⟩ := by
  apply q5_ext
  · dsimp only [q5MulRaw, q5Conj, q5NormSq, q5Inner]
    simp only [z5Sub_eq_add_neg, z5Mul_neg_left, z5Neg_neg]
  · dsimp only [q5MulRaw, q5Conj]
    simp only [z5Sub_eq_add_neg, z5Mul_neg_right, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega
  · dsimp only [q5MulRaw, q5Conj]
    simp only [z5Sub_eq_add_neg, z5Mul_neg_right, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega
  · dsimp only [q5MulRaw, q5Conj]
    simp only [z5Sub_eq_add_neg, z5Mul_neg_right, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega

private theorem q5_euler_norm (a b : QuatZ5) :
    q5NormSq (q5MulRaw a b) = z5Mul (q5NormSq a) (q5NormSq b) := by
  have hsc_left : ∀ s : ZSqrt5, ∀ v : QuatZ5,
      q5MulRaw ⟨s, z5Zero, z5Zero, z5Zero⟩ v = q5ScaleZ5 s v := by
    intro s v
    apply q5_ext <;>
      dsimp only [q5MulRaw, q5ScaleZ5] <;>
      simp only [z5Sub_eq_add_neg, z5Zero_mul] <;>
      (apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega)
  have hin_sc : ∀ s : ZSqrt5, ∀ u v : QuatZ5,
      q5Inner (q5ScaleZ5 s u) v = z5Mul s (q5Inner u v) := by
    intro s u v
    simp only [q5Inner, q5ScaleZ5, z5Mul_add, z5Mul_assoc]
  rw [q5NormSq, q5_left_adj (q5MulRaw a b) a b, ← q5_mul_assoc (q5Conj a) a b,
      q5_conj_mul_self a, hsc_left, hin_sc]
  rfl

private theorem z5_diff_mul_plus_minus (u c : ZSqrt5) :
    z5Sub (z5Sq (z5Add u c)) (z5Mul (z5Add u c) (z5Sub u c)) =
      z5Scale 2 (z5Add (z5Mul u c) (z5Sq c)) := by
  simp only [z5Sq, z5Sub_eq_add_neg, z5Mul_add, z5Add_mul,
             z5Mul_neg_right, z5Scale_two, z5Mul_comm c u]
  apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega

private theorem q5_comm_re_identity (a b : QuatZ5) :
    (q5CommRaw a b).w =
      z5Sub (z5Mul (q5NormSq a) (q5NormSq b)) (z5Scale 2 (q5CrossNormSq a b)) := by
  unfold q5CommRaw
  rw [q5_conj_ba, q5_re_mul_conj, ← q5_euler_norm]
  let ux := z5Add (z5Mul a.w b.x) (z5Mul b.w a.x)
  let uy := z5Add (z5Mul a.w b.y) (z5Mul b.w a.y)
  let uz := z5Add (z5Mul a.w b.z) (z5Mul b.w a.z)
  let cx := z5Sub (z5Mul a.y b.z) (z5Mul a.z b.y)
  let cy := z5Sub (z5Mul a.z b.x) (z5Mul a.x b.z)
  let cz := z5Sub (z5Mul a.x b.y) (z5Mul a.y b.x)
  have h_ortho : z5Add (z5Add (z5Mul ux cx) (z5Mul uy cy)) (z5Mul uz cz) = z5Zero := by
    dsimp only [ux, uy, uz, cx, cy, cz]
    simp only [z5Sub_eq_add_neg, z5Mul_add, z5Mul_neg_right, z5Mul_comm, z5Mul_left_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg, z5Zero] <;> omega
  have hab_x : (q5MulRaw a b).x = z5Add ux cx := by
    dsimp only [q5MulRaw, ux, cx]
    simp only [z5Sub_eq_add_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hab_y : (q5MulRaw a b).y = z5Add uy cy := by
    dsimp only [q5MulRaw, uy, cy]
    simp only [z5Sub_eq_add_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hab_z : (q5MulRaw a b).z = z5Add uz cz := by
    dsimp only [q5MulRaw, uz, cz]
    simp only [z5Sub_eq_add_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hba_w : (q5MulRaw b a).w = (q5MulRaw a b).w := by
    dsimp only [q5MulRaw]
    simp only [z5Mul_comm]
  have hba_x : (q5MulRaw b a).x = z5Sub ux cx := by
    dsimp only [q5MulRaw, ux, cx]
    simp only [z5Sub_eq_add_neg, z5Neg_add, z5Neg_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hba_y : (q5MulRaw b a).y = z5Sub uy cy := by
    dsimp only [q5MulRaw, uy, cy]
    simp only [z5Sub_eq_add_neg, z5Neg_add, z5Neg_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hba_z : (q5MulRaw b a).z = z5Sub uz cz := by
    dsimp only [q5MulRaw, uz, cz]
    simp only [z5Sub_eq_add_neg, z5Neg_add, z5Neg_neg, z5Mul_comm]
    apply z5_ext <;> simp only [z5Add, z5Neg] <;> omega
  have hdiff :
      z5Sub (q5NormSq (q5MulRaw a b)) (q5Inner (q5MulRaw a b) (q5MulRaw b a)) =
        z5Scale 2 (q5CrossNormSq a b) := by
    unfold q5NormSq q5Inner q5CrossNormSq
    rw [hba_w, hab_x, hab_y, hab_z, hba_x, hba_y, hba_z]
    have hx := z5_diff_mul_plus_minus ux cx
    have hy := z5_diff_mul_plus_minus uy cy
    have hz := z5_diff_mul_plus_minus uz cz
    dsimp only [cx, cy, cz] at hx hy hz h_ortho ⊢
    injection hx with hxa hxb
    injection hy with hya hyb
    injection hz with hza hzb
    injection h_ortho with hoa hob
    apply z5_ext <;>
      simp only [z5Sub, z5Add, z5Scale, z5Sq] at hxa hxb hya hyb hza hzb hoa hob ⊢ <;>
      omega
  injection hdiff with hda hdb
  apply z5_ext <;> simp only [z5Sub, z5Scale] at hda hdb ⊢ <;> omega

/-- Embed QuatZ into QuatZ5 via b = 0 (ring homomorphism z5OfInt), obtaining the
    integer commutator real-part theorem immediately from q5_comm_re_identity! -/
def qzToQ5 (q : QuatZ) : QuatZ5 :=
  ⟨⟨q.w, 0⟩, ⟨q.x, 0⟩, ⟨q.y, 0⟩, ⟨q.z, 0⟩⟩

private theorem qz_comm_re_identity (a b : QuatZ) :
    (qzCommRaw a b).w = qzNormSq a * qzNormSq b - 2 * qzCrossNormSq a b := by
  have h5 := q5_comm_re_identity (qzToQ5 a) (qzToQ5 b)
  have hlhs : (q5CommRaw (qzToQ5 a) (qzToQ5 b)).w.a = (qzCommRaw a b).w := by
    simp only [q5CommRaw, qzCommRaw, qzToQ5, q5MulRaw, qzMul, q5Conj, qzConj,
               z5Sub, z5Add, z5Mul, z5Neg, Int.mul_zero, Int.zero_mul,
               Int.add_zero, Int.sub_zero, Int.neg_zero]
  have hrhs :
      (z5Sub (z5Mul (q5NormSq (qzToQ5 a)) (q5NormSq (qzToQ5 b)))
             (z5Scale 2 (q5CrossNormSq (qzToQ5 a) (qzToQ5 b)))).a =
        qzNormSq a * qzNormSq b - 2 * qzCrossNormSq a b := by
    simp only [q5NormSq, q5Inner, q5CrossNormSq, qzNormSq, qzInner, qzCrossNormSq,
               qzToQ5, z5Sub, z5Add, z5Mul, z5Sq, z5Scale,
               Int.mul_zero, Int.zero_mul, Int.add_zero, Int.sub_zero]
  have ha := congrArg ZSqrt5.a h5
  rw [hlhs, hrhs] at ha
  exact ha

/-- Theorem #47 (Track S119 — Universal Commutator Real-Part Theorem & Order-5 Obstruction):
    1. Universal exact quaternion commutator real-part theorem over Int (QuatZ):
       for ALL a, b : QuatZ, Re(a * b * conj(a) * conj(b)) = normSq(a)*normSq(b) - 2*||vec(a) x vec(b)||^2,
    2. Universal exact quaternion commutator real-part theorem over Z[sqrt(5)] (QuatZ5):
       for ALL p, q : QuatZ5, Re(p * q * conj(p) * conj(q)) = normSq(p)*normSq(q) - 2*||vec(p) x vec(q)||^2,
    3. Exact order-5 spinor Z_2 obstruction on sigma0 = -tau0:
       - sigma0^5 = +I and sigma0^1 != +I (exact order 5),
       - [g_alpha * sigma0 * conj(g_alpha), g_beta * sigma0 * conj(g_beta)] = +tau0 = -sigma0 != +sigma0,
       - Re(sigma0) < 0 (¬ z5Pos sigma0.w and z5Pos (-sigma0.w)) whereas Re(+tau0) > 0 (z5Pos tau0.w). -/
theorem p47_su2_commutator_trace_and_order5_obstruction
    (a b : QuatZ) (p q : QuatZ5) :
    (qzCommRaw a b).w = qzNormSq a * qzNormSq b - 2 * qzCrossNormSq a b ∧
    (q5CommRaw p q).w =
      z5Sub (z5Mul (q5NormSq p) (q5NormSq q)) (z5Scale 2 (q5CrossNormSq p q)) ∧
    (uPow sigma0_4 5 = I4 ∧
     uPow sigma0_4 1 ≠ I4 ∧
     uComm (uConjBy gAlpha_4 sigma0_4) (uConjBy gBeta_4 sigma0_4) = tau0_4 ∧
     uComm (uConjBy gAlpha_4 sigma0_4) (uConjBy gBeta_4 sigma0_4) = q5Neg sigma0_4 ∧
     uComm (uConjBy gAlpha_4 sigma0_4) (uConjBy gBeta_4 sigma0_4) ≠ sigma0_4 ∧
     z5Pos tau0_4.w ∧
     ¬ z5Pos sigma0_4.w ∧
     z5Pos (z5Neg sigma0_4.w)) := by
  exact ⟨qz_comm_re_identity a b, q5_comm_re_identity p q, by decide⟩

-- ============================================================================
-- Theorem #48: Spinor-Barrington Acute Cayley Bounded-Ball Chart (t_max = 0.8)
-- ============================================================================

/-- Exact cleared-denominator Cayley ball predicate ||omega(Q/4)||^2 < (4/5)^2 = 16/25:
    for Q = 4*q with ||q|| = 1 and q_0 = Q.w / 4 > 0,
      ||omega||^2 = ||vec(q)||^2 / (1 + q_0)^2 = ||vec(Q)||^2 / (4 + Q.w)^2 < 16 / 25
    iff 25 * ||vec(Q)||^2 < 16 * (4 + Q.w)^2 in ordered Z[sqrt(5)]. -/
def inAcuteCayleyBall08 (Q : QuatZ5) : Prop :=
  z5Pos Q.w ∧
  z5Lt (z5Scale 25 (q5VecNormSq Q)) (z5Scale 16 (z5Sq (z5Add ⟨4, 0⟩ Q.w)))

instance (Q : QuatZ5) : Decidable (inAcuteCayleyBall08 Q) := by
  unfold inAcuteCayleyBall08
  infer_instance

private theorem z5Pos_not_neg (z : ZSqrt5) (hp : z5Pos z) : ¬ z5Pos (z5Neg z) := by
  unfold z5Pos at hp ⊢
  simp only [z5Neg, Int.neg_mul_neg]
  intro hn
  cases hp with
  | inl h1 =>
    cases hn with
    | inl n1 => omega
    | inr n23 =>
      cases n23 with
      | inl n2 => omega
      | inr n3 => omega
  | inr h23 =>
    cases h23 with
    | inl h2 =>
      cases hn with
      | inl n1 => omega
      | inr n23 =>
        cases n23 with
        | inl n2 => omega
        | inr n3 => omega
    | inr h3 =>
      cases hn with
      | inl n1 => omega
      | inr n23 =>
        cases n23 with
        | inl n2 => omega
        | inr n3 => omega

/-- Theorem #48 (Track S119 — Spinor-Barrington Acute Cayley Ball ||omega||^2 < (4/5)^2):
    In exact ordered Z[sqrt(5)] (zero IEEE-754 floats):
      1. Order soundness: z5Pos is strictly asymmetric (z5Pos z -> ¬ z5Pos (-z)) and
         brackets sqrt(5) tightly between 223/100 and 224/100 (223 < 100*sqrt(5) < 224),
      2. All five Spinor-Barrington 2I generators — tau0, g_alpha, g_beta,
         alpha = g_alpha*tau0*conj(g_alpha), and beta = g_beta*tau0*conj(g_beta) —
         satisfy BOTH strict acute real part q_0 > 0 AND strict Cayley bounded-ball
         25 * ||vec(Q)||^2 < 16 * (4 + Q_0)^2 (i.e. ||omega|| < 4/5 = 0.8),
      3. Exact algebraic radius of C_10^+: ||omega(tau0)||^2 = tan^2(pi/10) = (5 - 2*sqrt(5))/5,
         verified via (4 + tau0_4.w)^2 * (5 - 2*sqrt(5)) = 5 * ||vec(tau0_4)||^2
         and 0 < (5 - 2*sqrt(5))/5 < 16/25 (5 * (5 - 2*sqrt(5)) < 16),
      4. Exact algebraic radius of g_beta: ||omega(g_beta)||^2 = 1/3 < 16/25
         (3 * ||vec(gBeta_4)||^2 = (4 + gBeta_4.w)^2 = 36). -/
theorem p48_spinor_barrington_acute_cayley_ball (z : ZSqrt5) :
    (z5Pos z → ¬ z5Pos (z5Neg z)) ∧
    (z5Lt ⟨223, 0⟩ ⟨0, 100⟩ ∧ z5Lt ⟨0, 100⟩ ⟨224, 0⟩) ∧
    (inAcuteCayleyBall08 tau0_4 ∧
     inAcuteCayleyBall08 gAlpha_4 ∧
     inAcuteCayleyBall08 gBeta_4 ∧
     inAcuteCayleyBall08 (uConjBy gAlpha_4 tau0_4) ∧
     inAcuteCayleyBall08 (uConjBy gBeta_4 tau0_4)) ∧
    (z5Mul (z5Sq (z5Add ⟨4, 0⟩ tau0_4.w)) ⟨5, -2⟩ = z5Scale 5 (q5VecNormSq tau0_4) ∧
     z5Pos ⟨5, -2⟩ ∧
     z5Lt (z5Scale 5 ⟨5, -2⟩) ⟨16, 0⟩) ∧
    (z5Scale 3 (q5VecNormSq gBeta_4) = z5Sq (z5Add ⟨4, 0⟩ gBeta_4.w) ∧
     z5Sq (z5Add ⟨4, 0⟩ gBeta_4.w) = ⟨36, 0⟩) := by
  exact ⟨z5Pos_not_neg z, by decide, by decide, by decide, by decide⟩

-- ============================================================================
-- Machine-Checked Axiom Audit (#print axioms)
-- ============================================================================

#print axioms HoloSphereSpinorBarrington.p46_spinor_barrington_c10_exact_closure
#print axioms HoloSphereSpinorBarrington.p47_su2_commutator_trace_and_order5_obstruction
#print axioms HoloSphereSpinorBarrington.p48_spinor_barrington_acute_cayley_ball

end HoloSphereSpinorBarrington
