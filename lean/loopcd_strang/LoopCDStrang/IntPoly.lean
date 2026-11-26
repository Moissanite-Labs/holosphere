/-
  LoopCDStrang/IntPoly.lean

  Pure-Init (`Lean 4.36.0`) zero-`Classical.choice` constructive multivariate `Int`
  polynomial normalizer (`PExpr`, `norm`, `poly_eq_of_norm_eq`), exact order-2 Taylor
  with non-perturbative `t^3` remainder algebra (`Jet3`), and exact rational pairs (`QPair`).

  Axiom footprint: `[propext]` ONLY (`0` `Classical.choice`, `0` `Quot.sound`, `0` `bv_decide`,
  `0` `native_decide`, `0` `sorryAx`, `0` `partial def`).
-/

namespace LoopCDStrang

/-! ## 1. Constructive Multivariate `Int` Polynomial Normalizer (`[propext]` Only) -/

/-- Reified multivariate polynomial expression over `Int`. -/
inductive PExpr where
  | const (c : Int)
  | var   (i : Nat)
  | neg   (a : PExpr)
  | add   (a b : PExpr)
  | sub   (a b : PExpr)
  | mul   (a b : PExpr)
  deriving Repr, DecidableEq

instance : Add PExpr := ⟨PExpr.add⟩
instance : Sub PExpr := ⟨PExpr.sub⟩
instance : Mul PExpr := ⟨PExpr.mul⟩
instance : Neg PExpr := ⟨PExpr.neg⟩
instance (n : Nat) : OfNat PExpr n := ⟨PExpr.const (Int.ofNat n)⟩

/-- Total environment lookup (`0` out of bounds). -/
def lookupVar : List Int → Nat → Int
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, k + 1 => lookupVar xs k

/-- Evaluate a `PExpr` in `env : List Int`. -/
def PExpr.eval (env : List Int) : PExpr → Int
  | .const c => c
  | .var i   => lookupVar env i
  | .neg a   => - (a.eval env)
  | .add a b => a.eval env + b.eval env
  | .sub a b => a.eval env - b.eval env
  | .mul a b => a.eval env * b.eval env

abbrev Monom := List Nat
abbrev Poly  := List (Monom × Int)

def evalMonom (env : List Int) : Monom → Int
  | [] => 1
  | i :: is => lookupVar env i * evalMonom env is

def evalPoly (env : List Int) : Poly → Int
  | [] => 0
  | (m, c) :: rest => c * evalMonom env m + evalPoly env rest

def insertVar (i : Nat) : Monom → Monom
  | [] => [i]
  | j :: js => if Nat.ble i j then i :: j :: js else j :: insertVar i js

def mulMonom : Monom → Monom → Monom
  | [], m2 => m2
  | i :: is, m2 => insertVar i (mulMonom is m2)

def cmpMonom : Monom → Monom → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | x :: xs, y :: ys =>
    if Nat.blt x y then .lt
    else if Nat.beq x y then cmpMonom xs ys
    else .gt

/-- Constructive (`0`-axiom) zero test on `Int`. -/
def intIsZero : Int → Bool
  | .ofNat 0 => true
  | _ => false

def insertTerm (m : Monom) (c : Int) : Poly → Poly
  | [] => if intIsZero c then [] else [(m, c)]
  | (m', c') :: rest =>
    match cmpMonom m m' with
    | .lt => if intIsZero c then (m', c') :: rest else (m, c) :: (m', c') :: rest
    | .eq =>
      let s := c + c'
      if intIsZero s then rest else (m', s) :: rest
    | .gt => (m', c') :: insertTerm m c rest

def addPoly : Poly → Poly → Poly
  | [], q => q
  | (m, c) :: ps, q => insertTerm m c (addPoly ps q)

def negPoly : Poly → Poly
  | [] => []
  | (m, c) :: ps => if intIsZero (-c) then negPoly ps else (m, -c) :: negPoly ps

def mulTermPoly (m : Monom) (c : Int) : Poly → Poly
  | [] => []
  | (m', c') :: qs => insertTerm (mulMonom m m') (c * c') (mulTermPoly m c qs)

def mulPoly : Poly → Poly → Poly
  | [], _ => []
  | (m, c) :: ps, q => addPoly (mulTermPoly m c q) (mulPoly ps q)

/-- Sound sparse multivariate polynomial normalizer. -/
def norm : PExpr → Poly
  | .const c => if intIsZero c then [] else [([], c)]
  | .var i   => [([i], 1)]
  | .neg a   => negPoly (norm a)
  | .add a b => addPoly (norm a) (norm b)
  | .sub a b => addPoly (norm a) (negPoly (norm b))
  | .mul a b => mulPoly (norm a) (norm b)

private theorem intIsZero_spec : ∀ c : Int, intIsZero c = true → c = 0
  | .ofNat 0, _ => rfl
  | .ofNat (_ + 1), h => Bool.noConfusion h
  | .negSucc _, h => Bool.noConfusion h

private theorem evalMonom_insertVar (env : List Int) (i : Nat) (m : Monom) :
    evalMonom env (insertVar i m) = lookupVar env i * evalMonom env m := by
  induction m with
  | nil => simp [insertVar, evalMonom]
  | cons j js ih =>
    unfold insertVar
    cases Nat.ble i j
    · simp only [Bool.false_eq_true, ↓reduceIte, evalMonom, ih]
      rw [← Int.mul_assoc, Int.mul_comm (lookupVar env j) (lookupVar env i), Int.mul_assoc]
    · simp only [↓reduceIte, evalMonom]

private theorem evalMonom_mulMonom (env : List Int) (m1 m2 : Monom) :
    evalMonom env (mulMonom m1 m2) = evalMonom env m1 * evalMonom env m2 := by
  induction m1 with
  | nil => simp [mulMonom, evalMonom]
  | cons i is ih =>
    simp only [mulMonom, evalMonom, evalMonom_insertVar, ih, Int.mul_assoc]

private theorem cmpMonom_eq_iff : ∀ m1 m2 : Monom, cmpMonom m1 m2 = .eq → m1 = m2
  | [], [], _ => rfl
  | [], _ :: _, h => by cases h
  | _ :: _, [], h => by cases h
  | x :: xs, y :: ys, h => by
    unfold cmpMonom at h
    cases hlt : Nat.blt x y <;> rw [hlt] at h
    · simp only [Bool.false_eq_true, ↓reduceIte] at h
      cases hbeq : Nat.beq x y <;> rw [hbeq] at h
      · cases h
      · simp only [↓reduceIte] at h
        have hxy : x = y := Nat.eq_of_beq_eq_true hbeq
        rw [hxy, cmpMonom_eq_iff xs ys h]
    · cases h

private theorem evalPoly_insertTerm (env : List Int) (m : Monom) (c : Int) (p : Poly) :
    evalPoly env (insertTerm m c p) = c * evalMonom env m + evalPoly env p := by
  induction p with
  | nil =>
    unfold insertTerm
    cases hz : intIsZero c
    · simp [evalPoly]
    · rw [intIsZero_spec c hz]; simp [evalPoly]
  | cons hd rest ih =>
    rcases hd with ⟨m', c'⟩
    unfold insertTerm
    cases hcmp : cmpMonom m m'
    · cases hz : intIsZero c
      · simp [evalPoly]
      · rw [intIsZero_spec c hz]; simp [evalPoly]
    · dsimp only
      have hm_eq : m = m' := cmpMonom_eq_iff m m' hcmp
      cases hz : intIsZero (c + c')
      · simp only [Bool.false_eq_true, ↓reduceIte, evalPoly, hm_eq]
        rw [Int.add_mul, Int.add_assoc]
      · simp only [↓reduceIte, evalPoly, hm_eq]
        have hs0 : c + c' = 0 := intIsZero_spec (c + c') hz
        rw [← Int.add_assoc, ← Int.add_mul, hs0, Int.zero_mul, Int.zero_add]
    · simp only [evalPoly, ih]
      rw [← Int.add_assoc, Int.add_comm (c' * evalMonom env m') (c * evalMonom env m), Int.add_assoc]

private theorem evalPoly_addPoly (env : List Int) (p q : Poly) :
    evalPoly env (addPoly p q) = evalPoly env p + evalPoly env q := by
  induction p with
  | nil => simp [addPoly, evalPoly]
  | cons hd ps ih =>
    rcases hd with ⟨m, c⟩
    simp only [addPoly, evalPoly_insertTerm, ih, evalPoly, Int.add_assoc]

private theorem evalPoly_negPoly (env : List Int) (p : Poly) :
    evalPoly env (negPoly p) = - evalPoly env p := by
  induction p with
  | nil => simp [negPoly, evalPoly]
  | cons hd ps ih =>
    rcases hd with ⟨m, c⟩
    unfold negPoly
    cases hz : intIsZero (-c)
    · simp only [Bool.false_eq_true, ↓reduceIte, evalPoly, ih, Int.neg_mul, Int.neg_add]
    · simp only [↓reduceIte, evalPoly, ih]
      have hc0 : -c = 0 := intIsZero_spec (-c) hz
      rw [Int.neg_add, ← Int.neg_mul, hc0, Int.zero_mul, Int.zero_add]

private theorem evalPoly_mulTermPoly (env : List Int) (m : Monom) (c : Int) (q : Poly) :
    evalPoly env (mulTermPoly m c q) = (c * evalMonom env m) * evalPoly env q := by
  induction q with
  | nil => simp [mulTermPoly, evalPoly]
  | cons hd qs ih =>
    rcases hd with ⟨m', c'⟩
    simp only [mulTermPoly, evalPoly_insertTerm, evalMonom_mulMonom, ih, evalPoly, Int.mul_add]
    have hmul : c * c' * (evalMonom env m * evalMonom env m') = c * evalMonom env m * (c' * evalMonom env m') := by
      rw [Int.mul_assoc c c', ← Int.mul_assoc c' (evalMonom env m), Int.mul_comm c' (evalMonom env m),
          Int.mul_assoc (evalMonom env m), ← Int.mul_assoc]
    rw [hmul]

private theorem evalPoly_mulPoly (env : List Int) (p q : Poly) :
    evalPoly env (mulPoly p q) = evalPoly env p * evalPoly env q := by
  induction p with
  | nil => simp [mulPoly, evalPoly]
  | cons hd ps ih =>
    rcases hd with ⟨m, c⟩
    simp only [mulPoly, evalPoly_addPoly, evalPoly_mulTermPoly, ih, evalPoly, Int.add_mul]

theorem evalPoly_norm (env : List Int) (e : PExpr) :
    evalPoly env (norm e) = e.eval env := by
  induction e with
  | const c =>
    unfold norm
    cases hz : intIsZero c
    · simp [evalPoly, evalMonom, PExpr.eval]
    · rw [intIsZero_spec c hz]; simp [evalPoly, PExpr.eval]
  | var i => simp [norm, evalPoly, evalMonom, PExpr.eval]
  | neg a ih => simp [norm, evalPoly_negPoly, ih, PExpr.eval]
  | add a b iha ihb => simp [norm, evalPoly_addPoly, iha, ihb, PExpr.eval]
  | sub a b iha ihb => simp [norm, evalPoly_addPoly, evalPoly_negPoly, iha, ihb, PExpr.eval, Int.sub_eq_add_neg]
  | mul a b iha ihb => simp [norm, evalPoly_mulPoly, iha, ihb, PExpr.eval]

/-- **Reflective Commutative Ring Solver (`[propext]` Only)**:
    Whenever `norm lhs = norm rhs` holds by kernel `rfl`, `lhs.eval env = rhs.eval env` holds for all `env`! -/
theorem poly_eq_of_norm_eq (env : List Int) (lhs rhs : PExpr) (h : norm lhs = norm rhs) :
    lhs.eval env = rhs.eval env := by
  rw [← evalPoly_norm env lhs, h, evalPoly_norm env rhs]

/-! ## 2. Exact Order-2 Jet with Non-Perturbative `t^3` Remainder (`Jet3`) -/

/-- Exact decomposition `c0 + t * c1 + t^2 * c2 + t^3 * rem` over `Int` (zero truncation error!). -/
structure Jet3 where
  c0  : Int
  c1  : Int
  c2  : Int
  rem : Int
  deriving Repr, DecidableEq

def Jet3.eval (t : Int) (j : Jet3) : Int :=
  j.c0 + t * j.c1 + (t * t) * j.c2 + (t * t * t) * j.rem

def Jet3.const (c : Int) : Jet3 := ⟨c, 0, 0, 0⟩

def Jet3.varT : Jet3 := ⟨0, 1, 0, 0⟩

def Jet3.neg (j : Jet3) : Jet3 := ⟨-j.c0, -j.c1, -j.c2, -j.rem⟩

def Jet3.add (x y : Jet3) : Jet3 :=
  ⟨x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2, x.rem + y.rem⟩

def Jet3.sub (x y : Jet3) : Jet3 :=
  ⟨x.c0 - y.c0, x.c1 - y.c1, x.c2 - y.c2, x.rem - y.rem⟩

def Jet3.scale (c : Int) (x : Jet3) : Jet3 :=
  ⟨c * x.c0, c * x.c1, c * x.c2, c * x.rem⟩

def Jet3.mul (t : Int) (x y : Jet3) : Jet3 where
  c0  := x.c0 * y.c0
  c1  := x.c0 * y.c1 + x.c1 * y.c0
  c2  := x.c0 * y.c2 + x.c1 * y.c1 + x.c2 * y.c0
  rem := (x.c1 * y.c2 + x.c2 * y.c1) + t * (x.c2 * y.c2) +
         (x.c0 + t * x.c1 + (t * t) * x.c2) * y.rem +
         x.rem * (y.c0 + t * y.c1 + (t * t) * y.c2 + (t * t * t) * y.rem)

private def jConstPair : PExpr × PExpr :=
  let t := PExpr.var 0; let c := PExpr.var 1
  (c + t * 0 + (t * t) * 0 + (t * t * t) * 0, c)

private def jVarTPair : PExpr × PExpr :=
  let t := PExpr.var 0
  ((0 : PExpr) + t * 1 + (t * t) * 0 + (t * t * t) * 0, t)

private def jNegPair : PExpr × PExpr :=
  let t := PExpr.var 0
  let x0 := PExpr.var 1; let x1 := PExpr.var 2; let x2 := PExpr.var 3; let xr := PExpr.var 4
  ((-x0) + t * (-x1) + (t * t) * (-x2) + (t * t * t) * (-xr),
   -(x0 + t * x1 + (t * t) * x2 + (t * t * t) * xr))

private def jAddPair : PExpr × PExpr :=
  let t := PExpr.var 0
  let x0 := PExpr.var 1; let x1 := PExpr.var 2; let x2 := PExpr.var 3; let xr := PExpr.var 4
  let y0 := PExpr.var 5; let y1 := PExpr.var 6; let y2 := PExpr.var 7; let yr := PExpr.var 8
  ((x0 + y0) + t * (x1 + y1) + (t * t) * (x2 + y2) + (t * t * t) * (xr + yr),
   (x0 + t * x1 + (t * t) * x2 + (t * t * t) * xr) + (y0 + t * y1 + (t * t) * y2 + (t * t * t) * yr))

private def jSubPair : PExpr × PExpr :=
  let t := PExpr.var 0
  let x0 := PExpr.var 1; let x1 := PExpr.var 2; let x2 := PExpr.var 3; let xr := PExpr.var 4
  let y0 := PExpr.var 5; let y1 := PExpr.var 6; let y2 := PExpr.var 7; let yr := PExpr.var 8
  ((x0 - y0) + t * (x1 - y1) + (t * t) * (x2 - y2) + (t * t * t) * (xr - yr),
   (x0 + t * x1 + (t * t) * x2 + (t * t * t) * xr) - (y0 + t * y1 + (t * t) * y2 + (t * t * t) * yr))

private def jScalePair : PExpr × PExpr :=
  let t := PExpr.var 0; let c := PExpr.var 1
  let x0 := PExpr.var 2; let x1 := PExpr.var 3; let x2 := PExpr.var 4; let xr := PExpr.var 5
  ((c * x0) + t * (c * x1) + (t * t) * (c * x2) + (t * t * t) * (c * xr),
   c * (x0 + t * x1 + (t * t) * x2 + (t * t * t) * xr))

theorem Jet3.eval_const (t c : Int) : (Jet3.const c).eval t = c :=
  poly_eq_of_norm_eq [t, c] jConstPair.1 jConstPair.2 rfl

theorem Jet3.eval_varT (t : Int) : Jet3.varT.eval t = t :=
  poly_eq_of_norm_eq [t] jVarTPair.1 jVarTPair.2 rfl

theorem Jet3.eval_neg (t : Int) (x : Jet3) : (x.neg).eval t = - (x.eval t) :=
  poly_eq_of_norm_eq [t, x.c0, x.c1, x.c2, x.rem] jNegPair.1 jNegPair.2 rfl

theorem Jet3.eval_add (t : Int) (x y : Jet3) : (x.add y).eval t = x.eval t + y.eval t :=
  poly_eq_of_norm_eq [t, x.c0, x.c1, x.c2, x.rem, y.c0, y.c1, y.c2, y.rem] jAddPair.1 jAddPair.2 rfl

theorem Jet3.eval_sub (t : Int) (x y : Jet3) : (x.sub y).eval t = x.eval t - y.eval t :=
  poly_eq_of_norm_eq [t, x.c0, x.c1, x.c2, x.rem, y.c0, y.c1, y.c2, y.rem] jSubPair.1 jSubPair.2 rfl

theorem Jet3.eval_scale (t c : Int) (x : Jet3) : (x.scale c).eval t = c * x.eval t :=
  poly_eq_of_norm_eq [t, c, x.c0, x.c1, x.c2, x.rem] jScalePair.1 jScalePair.2 rfl

private def jetMulLHS : PExpr :=
  let tE := PExpr.var 0
  let x0 := PExpr.var 1; let x1 := PExpr.var 2; let x2 := PExpr.var 3; let xr := PExpr.var 4
  let y0 := PExpr.var 5; let y1 := PExpr.var 6; let y2 := PExpr.var 7; let yr := PExpr.var 8
  let t2 := PExpr.mul tE tE
  let t3 := PExpr.mul t2 tE
  let remE := PExpr.add
    (.add (.add (.add (.mul x1 y2) (.mul x2 y1)) (.mul tE (.mul x2 y2)))
          (.mul (.add (.add x0 (.mul tE x1)) (.mul t2 x2)) yr))
    (.mul xr (.add (.add (.add y0 (.mul tE y1)) (.mul t2 y2)) (.mul t3 yr)))
  .add (.add (.add (.mul x0 y0) (.mul tE (.add (.mul x0 y1) (.mul x1 y0))))
        (.mul t2 (.add (.add (.mul x0 y2) (.mul x1 y1)) (.mul x2 y0))))
        (.mul t3 remE)

private def jetMulRHS : PExpr :=
  let tE := PExpr.var 0
  let x0 := PExpr.var 1; let x1 := PExpr.var 2; let x2 := PExpr.var 3; let xr := PExpr.var 4
  let y0 := PExpr.var 5; let y1 := PExpr.var 6; let y2 := PExpr.var 7; let yr := PExpr.var 8
  let t2 := PExpr.mul tE tE
  let t3 := PExpr.mul t2 tE
  .mul (.add (.add (.add x0 (.mul tE x1)) (.mul t2 x2)) (.mul t3 xr))
       (.add (.add (.add y0 (.mul tE y1)) (.mul t2 y2)) (.mul t3 yr))

theorem Jet3.eval_mul (t : Int) (x y : Jet3) : (Jet3.mul t x y).eval t = x.eval t * y.eval t :=
  poly_eq_of_norm_eq [t, x.c0, x.c1, x.c2, x.rem, y.c0, y.c1, y.c2, y.rem] jetMulLHS jetMulRHS rfl

/-! ## 3. Constructive `Int` Square / Positivity Helpers & Exact Rational Pairs (`QPair`) -/

/-- Constructive square nonnegativity `0 ≤ x * x` for all `x : Int` (`[propext]` only). -/
theorem int_sq_nonneg (x : Int) : 0 ≤ x * x := by
  cases x with
  | ofNat n =>
    exact Int.ofNat_le.mpr (Nat.zero_le (n * n))
  | negSucc n =>
    have hpos : 0 < Int.negSucc n * Int.negSucc n := by
      change 0 < Int.ofNat ((n + 1) * (n + 1))
      exact Int.ofNat_lt.mpr (Nat.mul_pos (Nat.succ_pos n) (Nat.succ_pos n))
    exact Int.le_of_lt hpos

/-- Constructive strict square positivity `0 < x * x` for `0 < x`. -/
theorem int_sq_pos_of_pos {x : Int} (hx : 0 < x) : 0 < x * x :=
  Int.mul_pos hx hx

/-- Exact unreduced rational pair `num / den` (`Wf` asserts `0 < den`). -/
structure QPair where
  num : Int
  den : Int
  deriving Repr, DecidableEq

def QPair.Wf (q : QPair) : Prop := 0 < q.den

/-- Cross-multiplied exact rational equality `a.num * b.den = b.num * a.den`. -/
def QEq (a b : QPair) : Prop := a.num * b.den = b.num * a.den

/-- Cross-multiplied strict rational inequality `a.num * b.den < b.num * a.den` (valid when `0 < a.den, 0 < b.den`). -/
def QLt (a b : QPair) : Prop := a.num * b.den < b.num * a.den

/-- Cross-multiplied non-strict rational inequality `a.num * b.den ≤ b.num * a.den`. -/
def QLe (a b : QPair) : Prop := a.num * b.den ≤ b.num * a.den

instance (a b : QPair) : Decidable (QEq a b) :=
  inferInstanceAs (Decidable (a.num * b.den = b.num * a.den))

instance (a b : QPair) : Decidable (QLt a b) :=
  inferInstanceAs (Decidable (a.num * b.den < b.num * a.den))

instance (a b : QPair) : Decidable (QLe a b) :=
  inferInstanceAs (Decidable (a.num * b.den ≤ b.num * a.den))

def qadd (a b : QPair) : QPair := ⟨a.num * b.den + b.num * a.den, a.den * b.den⟩
def qsub (a b : QPair) : QPair := ⟨a.num * b.den - b.num * a.den, a.den * b.den⟩
def qmul (a b : QPair) : QPair := ⟨a.num * b.num, a.den * b.den⟩
def qdiv (a b : QPair) : QPair := ⟨a.num * b.den, a.den * b.num⟩

end LoopCDStrang
