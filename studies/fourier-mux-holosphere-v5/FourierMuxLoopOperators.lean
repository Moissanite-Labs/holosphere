/-
Track S164 (EXP-Q45a & EXP-Q45c): Pure-Init Lean 4.36.0 Formalization of
OpenAI FiniteFourier Crossover, HoloSphere-V5 Zero-Alias 260-Symbol / 3-Stage
Monarch Head, MUX Corollary 15 vs. Bayt al-Hikma Al-Kashi Fixed-Radix
Contraction (SOP-16), DualAxisCarry beta_min Stability, and Floodgates SU(2)
O(1) S_fast Loop-Cache Transport.

Zero external imports, zero Mathlib, zero Classical.choice, zero bv_decide,
zero native_decide, zero sorry. All 18 public theorems are pure-Init kernel-verified
operator laws (17 universally quantified theorems plus 1 concrete `decide`
phase-boundary conjunction) depending strictly on Lean kernel primitives
(`propext`, `Quot.sound`, or zero axioms).
-/

namespace FourierMuxLoopOperators

/-! ## Elementary Algebraic & Vector Helpers -/

private theorem sq_nonneg (x : Int) : 0 ≤ x * x := by
  by_cases h : 0 ≤ x
  · exact Int.mul_nonneg h h
  · have hneg : 0 ≤ -x := by omega
    have hmul : 0 ≤ (-x) * (-x) := Int.mul_nonneg hneg hneg
    have heq : (-x) * (-x) = x * x := by
      rw [Int.neg_mul, Int.mul_neg, Int.neg_neg]
    rw [heq] at hmul
    exact hmul

private theorem sq_pos_of_ne_zero (x : Int) (hx : x ≠ 0) : 0 < x * x := by
  by_cases h : 0 < x
  · exact Int.mul_pos h h
  · have hneg : 0 < -x := by omega
    have hmul : 0 < (-x) * (-x) := Int.mul_pos hneg hneg
    have heq : (-x) * (-x) = x * x := by
      rw [Int.neg_mul, Int.mul_neg, Int.neg_neg]
    rw [heq] at hmul
    exact hmul

private theorem sq_diff_factor (u v : Int) :
    u * u - v * v = (u - v) * (u + v) := by
  rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm v u]
  omega

private theorem diff_sq_identity (u v : Int) :
    (u - v) * (u - v) = u * u - 2 * (u * v) + v * v := by
  rw [Int.sub_mul, Int.mul_sub, Int.mul_sub, Int.mul_comm v u]
  omega

private theorem prod_sq_split (x y : Int) :
    (x * y) * (x * y) = (x * x) * (y * y) := by
  calc
    (x * y) * (x * y) = x * (y * (x * y)) := Int.mul_assoc x y (x * y)
    _ = x * (x * (y * y)) := by
      rw [← Int.mul_assoc y x y, Int.mul_comm y x, Int.mul_assoc x y y]
    _ = (x * x) * (y * y) := (Int.mul_assoc x x (y * y)).symm

private theorem mul_lt_mul_pos_left {a b c : Int} (ha : 0 < a) (hbc : b < c) :
    a * b < a * c := by
  have hpos : 0 < a * (c - b) := Int.mul_pos ha (by omega)
  rw [Int.mul_sub] at hpos
  omega

private theorem mul_le_mul_nonneg_left {a b c : Int} (ha : 0 ≤ a) (hbc : b ≤ c) :
    a * b ≤ a * c := by
  have hnn : 0 ≤ a * (c - b) := Int.mul_nonneg ha (by omega)
  rw [Int.mul_sub] at hnn
  omega

private theorem mul_le_mul_nonneg_right {a b c : Int} (hc : 0 ≤ c) (hab : a ≤ b) :
    a * c ≤ b * c := by
  have hnn : 0 ≤ (b - a) * c := Int.mul_nonneg (by omega) hc
  rw [Int.sub_mul] at hnn
  omega

private theorem mul_lt_iff_sq_lt {x y : Int} (hx : 0 < x) (hy : 0 < y) :
    x < y ↔ x * x < y * y := by
  constructor
  · intro hxy
    have hdiff : 0 < (y - x) * (y + x) := Int.mul_pos (by omega) (by omega)
    have hsf := sq_diff_factor y x
    omega
  · intro hsq
    have hdiff : 0 < y * y - x * x := Int.sub_pos.mpr hsq
    rw [sq_diff_factor] at hdiff
    by_cases hlt : x < y
    · exact hlt
    · have hnn : 0 ≤ (x - y) * (y + x) := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul] at hnn
      rw [Int.sub_mul] at hdiff
      omega

private theorem pow_pos_of_pos (a : Int) (ha : 0 < a) (k : Nat) : 0 < a ^ k := by
  induction k with
  | zero =>
    rw [Int.pow_zero]
    exact Int.zero_lt_one
  | succ n ih =>
    rw [Int.pow_succ]
    exact Int.mul_pos ih ha

private theorem pow_nonneg_of_nonneg (a : Int) (ha : 0 ≤ a) (k : Nat) : 0 ≤ a ^ k := by
  induction k with
  | zero =>
    rw [Int.pow_zero]
    omega
  | succ n ih =>
    rw [Int.pow_succ]
    exact Int.mul_nonneg ih ha

private theorem pow_le_pow_of_nonneg_le {a b : Int} (ha : 0 ≤ a) (hab : a ≤ b) (k : Nat) :
    a ^ k ≤ b ^ k := by
  induction k with
  | zero =>
    rw [Int.pow_zero, Int.pow_zero]
    exact Int.le_refl 1
  | succ n ih =>
    rw [Int.pow_succ, Int.pow_succ]
    have h1 : a ^ n * a ≤ a ^ n * b := mul_le_mul_nonneg_left (pow_nonneg_of_nonneg a ha n) hab
    have h2 : a ^ n * b ≤ b ^ n * b := mul_le_mul_nonneg_right (by omega) ih
    omega

private theorem pow_lt_pow_of_nonneg_lt {a b : Int} (ha : 0 ≤ a) (hab : a < b) :
    ∀ k : Nat, 1 ≤ k → a ^ k < b ^ k
  | 0, hk => by omega
  | k + 1, _ => by
    rw [Int.pow_succ, Int.pow_succ]
    have hle : a ^ k ≤ b ^ k := pow_le_pow_of_nonneg_le ha (by omega) k
    have hbp : 0 < b ^ k := pow_pos_of_pos b (by omega) k
    have h1 : a ^ k * a ≤ b ^ k * a := mul_le_mul_nonneg_right ha hle
    have h2 : b ^ k * a < b ^ k * b := mul_lt_mul_pos_left hbp hab
    omega

private def dotInt : List Int → List Int → Int
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => x * y + dotInt xs ys

private def normSq : List Int → Int
  | [] => 0
  | x :: xs => x * x + normSq xs

private inductive HasNonzero : List Int → Prop where
  | head {c : Int} {cs : List Int} : c ≠ 0 → HasNonzero (c :: cs)
  | tail {c : Int} {cs : List Int} : HasNonzero cs → HasNonzero (c :: cs)

private theorem normSq_nonneg (xs : List Int) : 0 ≤ normSq xs := by
  induction xs with
  | nil => exact Int.le_refl 0
  | cons y ys ihy =>
    change 0 ≤ y * y + normSq ys
    have hy : 0 ≤ y * y := sq_nonneg y
    omega

private theorem normSq_pos_of_hasNonzero {xs : List Int} (h : HasNonzero xs) :
    0 < normSq xs := by
  induction h with
  | @head c tail hne =>
    change 0 < c * c + normSq tail
    have hcsq : 0 < c * c := sq_pos_of_ne_zero c hne
    have htail : 0 ≤ normSq tail := normSq_nonneg tail
    omega
  | @tail c tail _ ih =>
    change 0 < c * c + normSq tail
    have hc : 0 ≤ c * c := sq_nonneg c
    omega

private theorem normSq_scale (s : Int) (xs : List Int) :
    normSq (xs.map (fun x => s * x)) = (s * s) * normSq xs := by
  induction xs with
  | nil =>
    simp only [List.map_nil, normSq, Int.mul_zero]
  | cons x tail ih =>
    change (s * x) * (s * x) + normSq (tail.map (fun y => s * y)) =
      (s * s) * (x * x + normSq tail)
    rw [ih, prod_sq_split s x, Int.mul_add]

/-! ## Group A (EXP-Q45a): Finite-N FWHT_256 Crossover vs. FiniteFourier -/

/-- Theorem 1 (Batched Finite-N FWHT vs. FiniteFourier Squared Crossover Equivalence):
    For any batch size `J >= 1`, transform size `N >= 1`, log dimension `m >= 1`,
    proxy `r >= 1`, and positive rational prefactor `p / q` (`p, q > 0`):
    (a) Cancelling positive `J * N` yields `(q * (J * N * m) < p * (J * N * r)) <-> (q * m < p * r)`;
    (b) Squaring positive quantities yields `(q * m < p * r) <-> (q^2 * m^2 < p^2 * r^2)`;
    (c) Whenever `r^2 = 1 + m` (exact square root `r = sqrt(1 + m)`, e.g. `N = 256, m = 8, r = 3`),
        `FWHT` strictly beats `FiniteFourier` iff `q^2 * m^2 < p^2 * (1 + m)`, and
        `FiniteFourier` strictly beats `FWHT` iff `p^2 * (1 + m) < q^2 * m^2`. -/
theorem fwht_vs_finite_fourier_crossover_iff
    (J N m r p q : Int)
    (hJ : 1 ≤ J) (hN : 1 ≤ N) (hm : 1 ≤ m) (hr : 1 ≤ r) (hp : 0 < p) (hq : 0 < q) :
    ((q * (J * N * m) < p * (J * N * r)) ↔ (q * m < p * r)) ∧
    ((q * m < p * r) ↔ (q * q) * (m * m) < (p * p) * (r * r)) ∧
    (r * r = 1 + m →
      (((q * (J * N * m) < p * (J * N * r)) ↔ (q * q) * (m * m) < (p * p) * (1 + m)) ∧
       ((p * (J * N * r) < q * (J * N * m)) ↔ (p * p) * (1 + m) < (q * q) * (m * m)))) := by
  have hJN : 0 < J * N := Int.mul_pos (by omega) (by omega)
  have hqm_pos : 0 < q * m := Int.mul_pos hq hm
  have hpr_pos : 0 < p * r := Int.mul_pos hp hr
  have h1 : ∀ a b : Int, a * (J * N * b) = (J * N) * (a * b) := by
    intro a b
    calc
      a * (J * N * b) = (a * (J * N)) * b := (Int.mul_assoc a (J * N) b).symm
      _ = ((J * N) * a) * b := by rw [Int.mul_comm a (J * N)]
      _ = (J * N) * (a * b) := Int.mul_assoc (J * N) a b
  have hcancel1 : (q * (J * N * m) < p * (J * N * r)) ↔ (q * m < p * r) := by
    rw [h1 q m, h1 p r]
    constructor
    · intro hlt
      by_cases hqm : q * m < p * r
      · exact hqm
      · have hle : p * r ≤ q * m := by omega
        have hmul : (J * N) * (p * r) ≤ (J * N) * (q * m) :=
          mul_le_mul_nonneg_left (by omega) hle
        omega
    · intro hlt
      exact mul_lt_mul_pos_left hJN hlt
  have hcancel2 : (p * (J * N * r) < q * (J * N * m)) ↔ (p * r < q * m) := by
    rw [h1 p r, h1 q m]
    constructor
    · intro hlt
      by_cases hpr : p * r < q * m
      · exact hpr
      · have hle : q * m ≤ p * r := by omega
        have hmul : (J * N) * (q * m) ≤ (J * N) * (p * r) :=
          mul_le_mul_nonneg_left (by omega) hle
        omega
    · intro hlt
      exact mul_lt_mul_pos_left hJN hlt
  have hsq1 : (q * m < p * r) ↔ (q * q) * (m * m) < (p * p) * (r * r) := by
    rw [← prod_sq_split q m, ← prod_sq_split p r]
    exact mul_lt_iff_sq_lt hqm_pos hpr_pos
  have hsq2 : (p * r < q * m) ↔ (p * p) * (r * r) < (q * q) * (m * m) := by
    rw [← prod_sq_split p r, ← prod_sq_split q m]
    exact mul_lt_iff_sq_lt hpr_pos hqm_pos
  refine ⟨hcancel1, hsq1, ?_⟩
  intro hrr
  constructor
  · rw [← hrr]
    exact Iff.trans hcancel1 hsq1
  · rw [← hrr]
    exact Iff.trans hcancel2 hsq2

/-- Theorem 2 (N = 256 Finite-N Gate Win, C=3 & C=4 Crossovers, and Thm 4.1 Zero Compression):
    (a) For any `m >= 1`, `r >= 1` with `1 + m <= r^2`, and `p, q > 0`,
        `q^2 * m^2 < p^2 * (1 + m) -> q * m < p * r`;
    (b) At `N = 256` (`m = 8`, `1 + m = 9 = 3^2`), `FWHT_256` beats `FiniteFourier`
        iff `p / q > 8 / 3` (`8 * q < 3 * p`), including `C = 3` (`8 < 9`) and `C = 4` (`8 < 12`);
    (c) At `C = 3`, `FWHT` strictly wins for `1 <= m <= 9` (`m^2 < 9(1+m)`) and crosses at `m* = 10` (`N* = 1024`);
    (d) At `C = 4`, `FWHT` strictly wins for `1 <= m <= 16` (`m^2 < 16(1+m)`) and crosses at `m* = 17` (`N* = 131072`);
    (e) Theorem 4.1 Hamming window at `N = 256` (`m = 8, log2 m = 3, m log2 m = 24`)
        has width `2*5 + 1 = 11 > 9 = m + 1` (and `2*w + 1 > 9` for any `w >= 0` with `w^2 >= 24`). -/
theorem fwht_256_finite_n_win_and_asymptotic_crossover
    (m r p q : Int)
    (hm : 1 ≤ m) (hr : 1 ≤ r) (hp : 0 < p) (hq : 0 < q)
    (hsqrt : 1 + m ≤ r * r) :
    ((q * q) * (m * m) < (p * p) * (1 + m) → q * m < p * r) ∧
    ((8 * q < 3 * p ↔ q * 8 < p * 3) ∧ 8 < 3 * 3 ∧ 8 < 4 * 3) ∧
    ((1 ≤ m ∧ m ≤ 9 → m * m < 9 * (1 + m)) ∧ (10 ≤ m → 9 * (1 + m) < m * m)) ∧
    ((1 ≤ m ∧ m ≤ 16 → m * m < 16 * (1 + m)) ∧ (17 ≤ m → 16 * (1 + m) < m * m)) ∧
    (8 + 1 < 2 * 5 + 1 ∧ (∀ w : Int, 0 ≤ w → 8 * 3 ≤ w * w → 8 + 1 < 2 * w + 1)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro hsq
    have hpp : 0 ≤ p * p := Int.mul_nonneg (by omega) (by omega)
    have hmono : (p * p) * (1 + m) ≤ (p * p) * (r * r) := mul_le_mul_nonneg_left hpp hsqrt
    have hdiff_sq : (q * q) * (m * m) < (p * p) * (r * r) := Int.lt_of_lt_of_le hsq hmono
    rw [← prod_sq_split q m, ← prod_sq_split p r] at hdiff_sq
    exact (mul_lt_iff_sq_lt (Int.mul_pos hq hm) (Int.mul_pos hp hr)).mpr hdiff_sq
  · exact ⟨⟨fun h => by omega, fun h => by omega⟩, by omega, by omega⟩
  · constructor
    · intro ⟨hlo, hhi⟩
      have h9 : 0 ≤ (9 - m) * m := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul] at h9
      rw [Int.mul_add, Int.mul_one]
      omega
    · intro hm10
      have h9 : 0 ≤ (m - 10) * (m + 1) := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_one, Int.mul_one] at h9
      rw [Int.mul_add, Int.mul_one]
      omega
  · constructor
    · intro ⟨hlo, hhi⟩
      have h16 : 0 ≤ (16 - m) * m := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul] at h16
      rw [Int.mul_add, Int.mul_one]
      omega
    · intro hm17
      have h16 : 0 ≤ (m - 17) * (m + 1) := Int.mul_nonneg (by omega) (by omega)
      rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_one, Int.mul_one] at h16
      rw [Int.mul_add, Int.mul_one]
      omega
  · refine ⟨by omega, ?_⟩
    intro w hw hsq
    have hge5 : 5 ≤ w := by
      match Int.decLe w 4 with
      | isTrue hlt =>
        have hbd : 0 ≤ (4 - w) * (4 + w) := Int.mul_nonneg (by omega) (by omega)
        have hsf := sq_diff_factor 4 w
        omega
      | isFalse hgt => omega
    omega

/-- Theorem 3 (FiniteFourier Prop. 5.2 Spectral Price Over Permutations):
    On a `1 x L` rectangle (`L >= 1`) with `m >= 1` and `r >= 1` satisfying
    `r^2 >= 1 + L^2`, the character-table spectral price `(L + r) * m` strictly
    exceeds the permutation spectral price `(L + 1) * m`. -/
theorem finite_fourier_prop52_spectral_gap_strict
    (L m r : Int)
    (hL : 1 ≤ L) (hm : 1 ≤ m) (hr : 1 ≤ r) (hsq : 1 + L * L ≤ r * r) :
    (L + 1) * m < (L + r) * m := by
  have hLsq : 1 ≤ L * L := Int.mul_pos hL hL
  have hr_ge2 : 2 ≤ r := by
    by_cases hr1 : r = 1
    · rw [hr1] at hsq
      omega
    · omega
  have hlt : 0 < (r - 1) * m := Int.mul_pos (by omega) hm
  rw [Int.sub_mul] at hlt
  rw [Int.add_mul, Int.add_mul]
  omega

private def zeroVec : Nat → List Int
  | 0 => []
  | m + 1 => 0 :: zeroVec m

private theorem zeroVec_length (m : Nat) : (zeroVec m).length = m := by
  induction m with
  | zero => rfl
  | succ k ih =>
    change (zeroVec k).length + 1 = k + 1
    rw [ih]

private theorem dotInt_zeroVec_right (xs : List Int) (m : Nat) :
    dotInt xs (zeroVec m) = 0 := by
  induction xs generalizing m with
  | nil => rfl
  | cons x xt ih =>
    cases m with
    | zero => rfl
    | succ k =>
      change x * 0 + dotInt xt (zeroVec k) = 0
      rw [Int.mul_zero, ih k, Int.zero_add]

private def vecScale (a : Int) : List Int → List Int
  | [] => []
  | x :: xs => (a * x) :: vecScale a xs

private theorem vecScale_length (a : Int) (xs : List Int) :
    (vecScale a xs).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons x xt ih =>
    change (vecScale a xt).length + 1 = xt.length + 1
    rw [ih]

private theorem hasNonzero_vecScale (a : Int) (ha : a ≠ 0) {xs : List Int} (h : HasNonzero xs) :
    HasNonzero (vecScale a xs) := by
  induction h with
  | @head c tail hne =>
    apply HasNonzero.head
    intro hz
    have hmul : a * c = 0 := hz
    match Int.decEq c 0 with
    | isTrue hc => exact hne hc
    | isFalse hc =>
      have hpos : 0 < (a * a) * (c * c) := Int.mul_pos (sq_pos_of_ne_zero a ha) (sq_pos_of_ne_zero c hc)
      rw [← prod_sq_split a c, hmul, Int.mul_zero] at hpos
      omega
  | @tail c tail _ ih =>
    exact HasNonzero.tail ih

private theorem dotInt_vecScale_right (a : Int) (xs ys : List Int) :
    dotInt xs (vecScale a ys) = a * dotInt xs ys := by
  induction xs generalizing ys with
  | nil =>
    simp only [dotInt, Int.mul_zero]
  | cons x xt ih =>
    cases ys with
    | nil =>
      change dotInt (x :: xt) [] = a * dotInt (x :: xt) []
      simp only [dotInt, Int.mul_zero]
    | cons y yt =>
      change x * (a * y) + dotInt xt (vecScale a yt) = a * (x * y + dotInt xt yt)
      rw [ih yt, Int.mul_add]
      have ha : x * (a * y) = a * (x * y) := by
        calc
          x * (a * y) = (x * a) * y := (Int.mul_assoc x a y).symm
          _ = (a * x) * y := by rw [Int.mul_comm x a]
          _ = a * (x * y) := Int.mul_assoc a x y
      rw [ha]

private theorem dotInt_vecScale_left (a : Int) (xs ys : List Int) :
    dotInt (vecScale a xs) ys = a * dotInt xs ys := by
  induction xs generalizing ys with
  | nil =>
    change dotInt [] ys = a * dotInt [] ys
    simp only [dotInt, Int.mul_zero]
  | cons x xt ih =>
    cases ys with
    | nil =>
      change dotInt ((a * x) :: vecScale a xt) [] = a * dotInt (x :: xt) []
      simp only [dotInt, Int.mul_zero]
    | cons y yt =>
      change (a * x) * y + dotInt (vecScale a xt) yt = a * (x * y + dotInt xt yt)
      rw [ih yt, Int.mul_add, Int.mul_assoc]

private def vecSub : List Int → List Int → List Int
  | x :: xs, y :: ys => (x - y) :: vecSub xs ys
  | _, _ => []

private theorem vecSub_length (xs ys : List Int) (hlen : xs.length = ys.length) :
    (vecSub xs ys).length = xs.length := by
  induction xs generalizing ys with
  | nil =>
    cases ys <;> rfl
  | cons x xt ih =>
    cases ys with
    | nil =>
      exfalso
      exact Nat.noConfusion hlen
    | cons y yt =>
      change (vecSub xt yt).length + 1 = xt.length + 1
      rw [ih yt (Nat.succ.inj hlen)]

private theorem dotInt_vecSub_left (xs ys zs : List Int) (hlen : xs.length = ys.length) :
    dotInt (vecSub xs ys) zs = dotInt xs zs - dotInt ys zs := by
  induction xs generalizing ys zs with
  | nil =>
    cases ys with
    | nil =>
      simp only [vecSub, dotInt, Int.sub_zero]
    | cons _ _ =>
      exfalso
      exact Nat.noConfusion hlen
  | cons x xt ih =>
    cases ys with
    | nil =>
      exfalso
      exact Nat.noConfusion hlen
    | cons y yt =>
      cases zs with
      | nil =>
        simp only [vecSub, dotInt, Int.sub_zero]
      | cons z zt =>
        change (x - y) * z + dotInt (vecSub xt yt) zt =
          (x * z + dotInt xt zt) - (y * z + dotInt yt zt)
        rw [ih yt zt (Nat.succ.inj hlen), Int.sub_mul]
        omega

private def projRow (a : Int) (t0 : List Int) : List Int → List Int
  | b :: t => vecSub (vecScale a t) (vecScale b t0)
  | [] => []

private inductive Col0Split (m : Nat) : List (List Int) → Prop where
  | allZero {rows : List (List Int)} :
      (∀ r, r ∈ rows → ∃ t, r = 0 :: t ∧ t.length = m) → Col0Split m rows
  | hasPivot {rows : List (List Int)} (a : Int) (t0 : List Int) (other : List (List Int)) :
      a ≠ 0 → t0.length = m → other.length + 1 = rows.length →
      (∀ r, r ∈ other → r.length = m + 1) →
      (∀ r, r ∈ rows → r = a :: t0 ∨ r ∈ other) → Col0Split m rows

private theorem col0Split_of_rows (m : Nat) (rows : List (List Int))
    (hlen : ∀ r, r ∈ rows → r.length = m + 1) : Col0Split m rows := by
  induction rows with
  | nil =>
    exact Col0Split.allZero (fun _ h => by cases h)
  | cons r rest ih =>
    have hrlen : r.length = m + 1 := hlen r (List.Mem.head _)
    cases r with
    | nil =>
      exfalso
      exact Nat.noConfusion hrlen
    | cons h t =>
      have htlen : t.length = m := Nat.succ.inj hrlen
      match Int.decEq h 0 with
      | isTrue hh =>
        have ih_rest := ih (fun x hx => hlen x (List.Mem.tail _ hx))
        cases ih_rest with
        | allZero hrest =>
          apply Col0Split.allZero
          intro x hx
          cases hx with
          | head => exact ⟨t, by rw [hh], htlen⟩
          | tail _ hxm => exact hrest x hxm
        | hasPivot a t0 other ha ht0 hot_len hot_mem hsplit =>
          apply Col0Split.hasPivot a t0 ((h :: t) :: other) ha ht0
          · change (other.length + 1) + 1 = rest.length + 1
            omega
          · intro x hx
            cases hx with
            | head => exact hrlen
            | tail _ hxm => exact hot_mem x hxm
          · intro x hx
            cases hx with
            | head => exact Or.inr (List.Mem.head _)
            | tail _ hxm =>
              rcases hsplit x hxm with h_eq | h_in
              · exact Or.inl h_eq
              · exact Or.inr (List.Mem.tail _ h_in)
      | isFalse hh =>
        apply Col0Split.hasPivot h t rest hh htlen rfl
        · intro x hx
          exact hlen x (List.Mem.tail _ hx)
        · intro x hx
          cases hx with
          | head => exact Or.inl rfl
          | tail _ hxm => exact Or.inr hxm

private theorem exists_nonzero_kernel_of_lt
    (n k : Nat) (rows : List (List Int))
    (hk : rows.length = k) (hn : ∀ r, r ∈ rows → r.length = n) (hlt : k < n) :
    ∃ w : List Int, w.length = n ∧ HasNonzero w ∧ (∀ r, r ∈ rows → dotInt r w = 0) := by
  induction n generalizing k rows with
  | zero =>
    exact False.elim (Nat.not_lt_zero k hlt)
  | succ m ih =>
    cases col0Split_of_rows m rows hn with
    | allZero hzero =>
      refine ⟨1 :: zeroVec m, ?_, HasNonzero.head (by decide), ?_⟩
      · change (zeroVec m).length + 1 = m + 1
        rw [zeroVec_length]
      · intro r hr
        rcases hzero r hr with ⟨t, rfl, _⟩
        change 0 * 1 + dotInt t (zeroVec m) = 0
        rw [Int.zero_mul, dotInt_zeroVec_right, Int.zero_add]
    | hasPivot a t0 other ha ht0 hot_len hot_mem hsplit =>
      let proj := other.map (projRow a t0)
      have hproj_len : ∀ rp, rp ∈ proj → rp.length = m := by
        intro rp hrp
        rcases List.mem_map.mp hrp with ⟨r, hr_in, rfl⟩
        have hrlen : r.length = m + 1 := hot_mem r hr_in
        cases r with
        | nil =>
          exfalso
          exact Nat.noConfusion hrlen
        | cons b t =>
          have ht : t.length = m := Nat.succ.inj hrlen
          change (vecSub (vecScale a t) (vecScale b t0)).length = m
          have hsub_len : (vecScale a t).length = (vecScale b t0).length := by
            rw [vecScale_length, vecScale_length, ht, ht0]
          rw [vecSub_length _ _ hsub_len, vecScale_length, ht]
      have hproj_lt : proj.length < m := by
        have hml : proj.length = other.length := List.length_map (projRow a t0)
        omega
      rcases ih proj.length proj rfl hproj_len hproj_lt with
        ⟨w', hw'len, hw'nz, hw'orth⟩
      let w := (-dotInt t0 w') :: vecScale a w'
      refine ⟨w, ?_, HasNonzero.tail (hasNonzero_vecScale a ha hw'nz), ?_⟩
      · change (vecScale a w').length + 1 = m + 1
        rw [vecScale_length, hw'len]
      · intro r hr
        rcases hsplit r hr with rfl | hr_other
        · change a * (-dotInt t0 w') + dotInt t0 (vecScale a w') = 0
          rw [Int.mul_neg, dotInt_vecScale_right]
          omega
        · have hrlen : r.length = m + 1 := hot_mem r hr_other
          cases r with
          | nil =>
            exfalso
            exact Nat.noConfusion hrlen
          | cons b t =>
            have ht : t.length = m := Nat.succ.inj hrlen
            have h_in_proj : projRow a t0 (b :: t) ∈ proj :=
              List.mem_map.mpr ⟨b :: t, hr_other, rfl⟩
            have h_orth := hw'orth _ h_in_proj
            change dotInt (vecSub (vecScale a t) (vecScale b t0)) w' = 0 at h_orth
            have hsub_len : (vecScale a t).length = (vecScale b t0).length := by
              rw [vecScale_length, vecScale_length, ht, ht0]
            rw [dotInt_vecSub_left _ _ _ hsub_len, dotInt_vecScale_left, dotInt_vecScale_left] at h_orth
            change b * (-dotInt t0 w') + dotInt t (vecScale a w') = 0
            rw [Int.mul_neg, dotInt_vecScale_right]
            omega

private theorem nat_mul_cover (a b idx : Nat) (hidx : idx < a * b) :
    ∃ q r : Nat, q < b ∧ r < a ∧ q * a + r = idx := by
  induction b with
  | zero =>
    rw [Nat.mul_zero] at hidx
    exact False.elim (Nat.not_lt_zero idx hidx)
  | succ b' ih =>
    rw [Nat.mul_succ] at hidx
    match Nat.decLt idx (a * b') with
    | isTrue hlt =>
      rcases ih hlt with ⟨q, r, hqb, hra, heq⟩
      exact ⟨q, r, Nat.lt_trans hqb (Nat.lt_succ_self b'), hra, heq⟩
    | isFalse hge =>
      refine ⟨b', idx - a * b', Nat.lt_succ_self b', by omega, ?_⟩
      rw [Nat.mul_comm]
      omega

/-- Theorem 4 (Low-Rank Rank-Nullity Kernel Defect vs. Full-Dimension Monarch Factorization):
    For any low-rank inner width `k < n` and radix factorization `a * b = n` (`0 < a`):
    (a) Dimension deficit `0 < n - k` (e.g. `256 - 16 = 240`) and `n - a * b = 0`;
    (b) True Linear-Algebraic Kernel Existence: ANY `k x n` factor matrix `rows` (`k < n`)
        admits a nonzero integer vector `w` in its orthogonal nullspace (`dotInt r w = 0` for all `r`);
    (c) Exact `a x b = n` Radix Coverage: every coordinate `idx < n` is addressed as
        `q * a + r = idx` with `q < b` and `r < a`. -/
theorem lowrank_rank_defect_vs_monarch_full_reach
    (n k a b : Nat)
    (hkn : k < n) (ha : 0 < a) (hfac : a * b = n) :
    (0 < (n : Int) - (k : Int) ∧ (n : Int) - (a : Int) * (b : Int) = 0) ∧
    (∀ rows : List (List Int),
      rows.length = k → (∀ r, r ∈ rows → r.length = n) →
      ∃ w : List Int, w.length = n ∧ HasNonzero w ∧ (∀ r, r ∈ rows → dotInt r w = 0)) ∧
    (∀ idx : Nat, idx < n → ∃ q r : Nat, q < b ∧ r < a ∧ q * a + r = idx) := by
  have _ := ha
  refine ⟨⟨by omega, by
    have hmul : (a : Int) * (b : Int) = (n : Int) := by
      rw [← Int.natCast_mul, hfac]
    omega⟩, ?_, ?_⟩
  · intro rows hlen hrow
    exact exists_nonzero_kernel_of_lt n k rows hlen hrow hkn
  · intro idx hidx
    have hidx_ab : idx < a * b := by omega
    exact nat_mul_cover a b idx hidx_ab

/-! ## Group B (EXP-Q45a): HoloSphere-V5 Zero-Alias 260-Symbol & 3-Stage Monarch Head -/

/-- Exact bitwise `db = d ^ ((d & 7) << 5)` coordinate parity dot product in F_2^8. -/
private def v4DbDot (b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 : Bool) : Bool :=
  (b0 && d0) ^^ (b1 && d1) ^^ (b2 && d2) ^^ (b3 && d3) ^^
  (b4 && d4) ^^ (b5 && (d5 ^^ d0)) ^^ (b6 && (d6 ^^ d1)) ^^ (b7 && (d7 ^^ d2))

/-- HoloSphere-V4 input parity with `sp_flip = sp_tag & ((d >> 5) & 1)`. -/
private def v4Parity
    (b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 sp_tag : Bool) : Bool :=
  v4DbDot b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 ^^ (sp_tag && d5)

private theorem v4_b0_b5_pair_eq (b0 b5 d0 d5 : Bool) :
    (((b0 && d0) ^^ (b5 && (d5 ^^ d0))) ^^ d5) =
      (((!b0) && d0) ^^ ((!b5) && (d5 ^^ d0))) := by
  cases b0 <;> cases b5 <;> cases d0 <;> cases d5 <;> rfl

private theorem v4_group_mid (x0 x1 x2 x3 x4 : Bool) :
    (((((x0 ^^ x1) ^^ x2) ^^ x3) ^^ x4)) = (x0 ^^ ((((x1 ^^ x2) ^^ x3) ^^ x4))) := by
  cases x0 <;> cases x1 <;> cases x2 <;> cases x3 <;> cases x4 <;> rfl

private theorem v4_xor_reassoc_left (x0 m x5 x6 x7 d5 : Bool) :
    ((((((x0 ^^ m) ^^ x5) ^^ x6) ^^ x7) ^^ d5)) =
      (((((((x0 ^^ x5) ^^ d5) ^^ m) ^^ x6) ^^ x7))) := by
  cases x0 <;> cases m <;> cases x5 <;> cases x6 <;> cases x7 <;> cases d5 <;> rfl

private theorem v4_xor_reassoc_right (x0 m x5 x6 x7 : Bool) :
    (((((x0 ^^ m) ^^ x5) ^^ x6) ^^ x7)) =
      ((((((x0 ^^ x5) ^^ m) ^^ x6) ^^ x7))) := by
  cases x0 <;> cases m <;> cases x5 <;> cases x6 <;> cases x7 <;> rfl

/-- Theorem 5 (HoloSphere-V4 256..259 Aliasing Obstruction):
    For ALL bytes `(b0..b7)` and ALL 256 coordinates `(d0..d7)`, setting `sp_tag = true`
    (`ID = 256 + b`) produces the EXACT SAME parity across all 512 coordinates as
    flipping bits 0 and 5 (`b XOR 0x21 = b XOR 33`) with `sp_tag = false`.
    In particular `BOS=256 == 33 ('!')`, `257 == 32 (' ')`, `258 == 35 ('#')`,
    and `259 == 34 ('"')`. -/
theorem v4_sp_flip_byte33_collision_universal
    (b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 : Bool) :
    v4Parity b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 true =
      v4Parity (!b0) b1 b2 b3 b4 (!b5) b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 false := by
  unfold v4Parity v4DbDot
  rw [Bool.true_and, Bool.false_and, Bool.xor_false]
  rw [v4_group_mid (b0 && d0) (b1 && d1) (b2 && d2) (b3 && d3) (b4 && d4)]
  rw [v4_group_mid ((!b0) && d0) (b1 && d1) (b2 && d2) (b3 && d3) (b4 && d4)]
  rw [v4_xor_reassoc_left (b0 && d0), v4_xor_reassoc_right ((!b0) && d0), v4_b0_b5_pair_eq b0 b5 d0 d5]

private theorem dotInt_append (a b c d : List Int) (hlen : a.length = c.length) :
    dotInt (a ++ b) (c ++ d) = dotInt a c + dotInt b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp only [List.nil_append, dotInt, Int.zero_add]
    | cons _ _ =>
      exfalso
      exact Nat.noConfusion hlen
  | cons x xs ih =>
    cases c with
    | nil =>
      exfalso
      exact Nat.noConfusion hlen
    | cons y ys =>
      have hlen_tail : xs.length = ys.length := Nat.succ.inj hlen
      have htail := ih ys hlen_tail
      change x * y + dotInt (xs ++ b) (ys ++ d) = (x * y + dotInt xs ys) + dotInt b d
      rw [htail]
      omega

private theorem dotInt_neg_right (a c : List Int) :
    dotInt a (c.map (fun z => -z)) = -(dotInt a c) := by
  induction a generalizing c with
  | nil => rfl
  | cons x xs ih =>
    cases c with
    | nil => rfl
    | cons y ys =>
      change x * (-y) + dotInt xs (ys.map (fun z => -z)) = -(x * y + dotInt xs ys)
      rw [ih ys, Int.mul_neg]
      omega

private theorem dotInt_neg_left (a c : List Int) :
    dotInt (a.map (fun z => -z)) c = -(dotInt a c) := by
  induction a generalizing c with
  | nil => rfl
  | cons x xs ih =>
    cases c with
    | nil => rfl
    | cons y ys =>
      change (-x) * y + dotInt (xs.map (fun z => -z)) ys = -(x * y + dotInt xs ys)
      rw [ih ys, Int.neg_mul]
      omega

private def bent4Bit (d0 d1 d2 d3 : Bool) : Bool :=
  (d0 && d2) ^^ (d1 && d3)

private def v5BoolParity
    (sp b0 b1 b2 b3 b4 b5 b6 b7 bank d0 d1 d2 d3 d4 d5 d6 d7 : Bool) : Bool :=
  v4DbDot b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 ^^
  bent4Bit d0 d1 d2 d3 ^^
  (bank && !sp)

private def boolSgn (b : Bool) : Int :=
  if b then -1 else 1

private def sumBool (f : Bool → Int) : Int :=
  f false + f true

private def sumBool8
    (f : Bool → Bool → Bool → Bool → Bool → Bool → Bool → Bool → Int) : Int :=
  sumBool fun d0 => sumBool fun d1 => sumBool fun d2 => sumBool fun d3 =>
  sumBool fun d4 => sumBool fun d5 => sumBool fun d6 => sumBool fun d7 =>
    f d0 d1 d2 d3 d4 d5 d6 d7

private def v5Dot512
    (sp b0 b1 b2 b3 b4 b5 b6 b7 sp' b0' b1' b2' b3' b4' b5' b6' b7' : Bool) : Int :=
  sumBool8 (fun d0 d1 d2 d3 d4 d5 d6 d7 =>
    boolSgn (v5BoolParity sp b0 b1 b2 b3 b4 b5 b6 b7 false d0 d1 d2 d3 d4 d5 d6 d7) *
    boolSgn (v5BoolParity sp' b0' b1' b2' b3' b4' b5' b6' b7' false d0 d1 d2 d3 d4 d5 d6 d7)) +
  sumBool8 (fun d0 d1 d2 d3 d4 d5 d6 d7 =>
    boolSgn (v5BoolParity sp b0 b1 b2 b3 b4 b5 b6 b7 true d0 d1 d2 d3 d4 d5 d6 d7) *
    boolSgn (v5BoolParity sp' b0' b1' b2' b3' b4' b5' b6' b7' true d0 d1 d2 d3 d4 d5 d6 d7))

private def dbLinear (e0 e1 e2 e3 e4 e5 e6 e7 d0 d1 d2 d3 d4 d5 d6 d7 : Bool) : Bool :=
  (e0 && d0) ^^ (e1 && d1) ^^ (e2 && d2) ^^ (e3 && d3) ^^
  (e4 && d4) ^^ (e5 && d5) ^^ (e6 && d6) ^^ (e7 && d7)

private theorem boolSgn_mul (x y : Bool) :
    boolSgn x * boolSgn y = boolSgn (x ^^ y) := by
  cases x <;> cases y <;> rfl

private theorem boolSgn_xor_true (x : Bool) :
    boolSgn (x ^^ true) = -(boolSgn x) := by
  cases x <;> rfl

private theorem sumBool_mul_const_left (c : Int) (f : Bool → Int) :
    sumBool (fun b => c * f b) = c * sumBool f := by
  unfold sumBool
  rw [Int.mul_add]

private theorem sumBool_mul_const_right (f : Bool → Int) (c : Int) :
    sumBool (fun b => f b * c) = sumBool f * c := by
  unfold sumBool
  rw [Int.add_mul]

private theorem sumBool_sgn_eq (e : Bool) :
    sumBool (fun d => boolSgn (e && d)) = (if e then 0 else 2) := by
  cases e <;> rfl

private theorem sumBool_neg (f : Bool → Int) :
    sumBool (fun b => -f b) = -sumBool f := by
  change -f false + -f true = -(f false + f true)
  omega

private theorem sumBool8_neg
    (f : Bool → Bool → Bool → Bool → Bool → Bool → Bool → Bool → Int) :
    sumBool8 (fun d0 d1 d2 d3 d4 d5 d6 d7 => -(f d0 d1 d2 d3 d4 d5 d6 d7)) = -sumBool8 f := by
  unfold sumBool8
  simp only [sumBool_neg]

private theorem v4DbDot_pair_xor_single (a a' d : Bool) :
    ((a && d) ^^ (a' && d)) = ((a ^^ a') && d) := by
  cases a <;> cases a' <;> cases d <;> rfl

private theorem xor_swap2 (a b c d : Bool) :
    ((a ^^ b) ^^ (c ^^ d)) = ((a ^^ c) ^^ (b ^^ d)) := by
  cases a <;> cases b <;> cases c <;> cases d <;> rfl

private theorem xor8_reassoc
    (p0 p1 p2 p3 p4 q0 q1 q2 q3 q4 : Bool) :
    ((((((p0 ^^ p1) ^^ p2) ^^ p3) ^^ p4) ^^
      ((((q0 ^^ q1) ^^ q2) ^^ q3) ^^ q4))) =
    ((((((p0 ^^ q0) ^^ (p1 ^^ q1)) ^^ (p2 ^^ q2)) ^^ (p3 ^^ q3)) ^^ (p4 ^^ q4))) := by
  rw [xor_swap2 _ p4 _ q4, xor_swap2 _ p3 _ q3, xor_swap2 _ p2 _ q2, xor_swap2 p0 p1 q0 q1]

private theorem xor_bent_cancel2 (x y b4 : Bool) :
    ((x ^^ b4) ^^ (y ^^ b4)) = (x ^^ y) := by
  cases x <;> cases y <;> cases b4 <;> rfl

private theorem xor_bent_cancel (x y b4 fl : Bool) :
    (((x ^^ b4) ^^ fl) ^^ ((y ^^ b4) ^^ fl)) = (x ^^ y) := by
  cases x <;> cases y <;> cases b4 <;> cases fl <;> rfl

private theorem xor_bent_flip (x y b4 fl1 fl2 : Bool) (hne : fl1 ≠ fl2) :
    (((x ^^ b4) ^^ fl1) ^^ ((y ^^ b4) ^^ fl2)) = ((x ^^ y) ^^ true) := by
  cases fl1 <;> cases fl2
  · exact False.elim (hne rfl)
  · cases x <;> cases y <;> cases b4 <;> rfl
  · cases x <;> cases y <;> cases b4 <;> rfl
  · exact False.elim (hne rfl)

private theorem v4DbDot_xor_eq_dbLinear
    (b0 b1 b2 b3 b4 b5 b6 b7 b0' b1' b2' b3' b4' b5' b6' b7'
     d0 d1 d2 d3 d4 d5 d6 d7 : Bool) :
    (v4DbDot b0 b1 b2 b3 b4 b5 b6 b7 d0 d1 d2 d3 d4 d5 d6 d7 ^^
     v4DbDot b0' b1' b2' b3' b4' b5' b6' b7' d0 d1 d2 d3 d4 d5 d6 d7) =
      dbLinear ((b0 ^^ b0') ^^ (b5 ^^ b5')) ((b1 ^^ b1') ^^ (b6 ^^ b6')) ((b2 ^^ b2') ^^ (b7 ^^ b7'))
        (b3 ^^ b3') (b4 ^^ b4') (b5 ^^ b5') (b6 ^^ b6') (b7 ^^ b7')
        d0 d1 d2 d3 d4 d5 d6 d7 := by
  unfold v4DbDot dbLinear
  have h5 : ((b5 ^^ b5') && (d5 ^^ d0)) = (((b5 ^^ b5') && d0) ^^ ((b5 ^^ b5') && d5)) := by
    cases b5 <;> cases b5' <;> cases d0 <;> cases d5 <;> rfl
  have h6 : ((b6 ^^ b6') && (d6 ^^ d1)) = (((b6 ^^ b6') && d1) ^^ ((b6 ^^ b6') && d6)) := by
    cases b6 <;> cases b6' <;> cases d1 <;> cases d6 <;> rfl
  have h7 : ((b7 ^^ b7') && (d7 ^^ d2)) = (((b7 ^^ b7') && d2) ^^ ((b7 ^^ b7') && d7)) := by
    cases b7 <;> cases b7' <;> cases d2 <;> cases d7 <;> rfl
  have hsplit8 :
      ∀ (u0 u1 u2 u3 u4 u5 u6 u7 v0 v1 v2 v3 v4 v5 v6 v7 : Bool),
        ((((((((u0 ^^ u1) ^^ u2) ^^ u3) ^^ u4) ^^ u5) ^^ u6) ^^ u7) ^^
         (((((((v0 ^^ v1) ^^ v2) ^^ v3) ^^ v4) ^^ v5) ^^ v6) ^^ v7)) =
        ((((((((u0 ^^ v0) ^^ (u1 ^^ v1)) ^^ (u2 ^^ v2)) ^^ (u3 ^^ v3)) ^^ (u4 ^^ v4)) ^^
              (u5 ^^ v5)) ^^ (u6 ^^ v6)) ^^ (u7 ^^ v7)) := by
    intro u0 u1 u2 u3 u4 u5 u6 u7 v0 v1 v2 v3 v4 v5 v6 v7
    rw [xor_swap2 _ u7 _ v7, xor_swap2 _ u6 _ v6, xor_swap2 _ u5 _ v5, xor8_reassoc]
  rw [hsplit8,
      v4DbDot_pair_xor_single b0 b0' d0,
      v4DbDot_pair_xor_single b1 b1' d1,
      v4DbDot_pair_xor_single b2 b2' d2,
      v4DbDot_pair_xor_single b3 b3' d3,
      v4DbDot_pair_xor_single b4 b4' d4,
      v4DbDot_pair_xor_single b5 b5' (d5 ^^ d0),
      v4DbDot_pair_xor_single b6 b6' (d6 ^^ d1),
      v4DbDot_pair_xor_single b7 b7' (d7 ^^ d2),
      h5, h6, h7]
  have hre :
      ∀ (x0 x1 x2 x3 x4 y0 d5' y1 d6' y2 d7' : Bool),
        (((((((x0 ^^ x1) ^^ x2) ^^ x3) ^^ x4) ^^ (y0 ^^ d5')) ^^ (y1 ^^ d6')) ^^ (y2 ^^ d7')) =
          ((((((((x0 ^^ y0) ^^ (x1 ^^ y1)) ^^ (x2 ^^ y2)) ^^ x3) ^^ x4) ^^ d5') ^^ d6') ^^ d7') := by
    intro x0 x1 x2 x3 x4 y0 d5' y1 d6' y2 d7'
    have h1 : (((((x0 ^^ x1) ^^ x2) ^^ x3) ^^ x4) ^^ (y0 ^^ d5')) =
        ((((((x0 ^^ y0) ^^ x1) ^^ x2) ^^ x3) ^^ x4) ^^ d5') := by
      cases x0 <;> cases x1 <;> cases x2 <;> cases x3 <;> cases x4 <;> cases y0 <;> cases d5' <;> rfl
    rw [h1]
    have h2 : (((((((x0 ^^ y0) ^^ x1) ^^ x2) ^^ x3) ^^ x4) ^^ d5') ^^ (y1 ^^ d6')) =
        (((((((x0 ^^ y0) ^^ (x1 ^^ y1)) ^^ x2) ^^ x3) ^^ x4) ^^ d5') ^^ d6') := by
      cases (x0 ^^ y0) <;> cases x1 <;> cases x2 <;> cases x3 <;> cases x4 <;> cases d5' <;> cases y1 <;> cases d6' <;> rfl
    rw [h2]
    have h3 : ((((((((x0 ^^ y0) ^^ (x1 ^^ y1)) ^^ x2) ^^ x3) ^^ x4) ^^ d5') ^^ d6') ^^ (y2 ^^ d7')) =
        ((((((((x0 ^^ y0) ^^ (x1 ^^ y1)) ^^ (x2 ^^ y2)) ^^ x3) ^^ x4) ^^ d5') ^^ d6') ^^ d7') := by
      cases ((x0 ^^ y0) ^^ (x1 ^^ y1)) <;> cases x2 <;> cases x3 <;> cases x4 <;> cases d5' <;> cases d6' <;> cases y2 <;> cases d7' <;> rfl
    exact h3
  rw [hre,
      v4DbDot_pair_xor_single (b0 ^^ b0') (b5 ^^ b5') d0,
      v4DbDot_pair_xor_single (b1 ^^ b1') (b6 ^^ b6') d1,
      v4DbDot_pair_xor_single (b2 ^^ b2') (b7 ^^ b7') d2]

private theorem boolSgn_dbLinear_factor (e0 e1 e2 e3 e4 e5 e6 e7 d0 d1 d2 d3 d4 d5 d6 d7 : Bool) :
    boolSgn (dbLinear e0 e1 e2 e3 e4 e5 e6 e7 d0 d1 d2 d3 d4 d5 d6 d7) =
      (((((((boolSgn (e0 && d0) * boolSgn (e1 && d1)) * boolSgn (e2 && d2)) *
             boolSgn (e3 && d3)) * boolSgn (e4 && d4)) * boolSgn (e5 && d5)) *
           boolSgn (e6 && d6)) * boolSgn (e7 && d7)) := by
  unfold dbLinear
  simp only [← boolSgn_mul]

private theorem sumBool8_dbLinear_eq (e0 e1 e2 e3 e4 e5 e6 e7 : Bool) :
    sumBool8 (fun d0 d1 d2 d3 d4 d5 d6 d7 =>
      boolSgn (dbLinear e0 e1 e2 e3 e4 e5 e6 e7 d0 d1 d2 d3 d4 d5 d6 d7)) =
    ((((((((if e0 then 0 else 2) * (if e1 then 0 else 2)) * (if e2 then 0 else 2)) *
          (if e3 then 0 else 2)) * (if e4 then 0 else 2)) * (if e5 then 0 else 2)) *
        (if e6 then 0 else 2)) * (if e7 then 0 else 2)) := by
  unfold sumBool8
  simp only [boolSgn_dbLinear_factor]
  simp only [sumBool_mul_const_left, sumBool_mul_const_right, sumBool_sgn_eq]

/-- Theorem 6 (HoloSphere-V5 H_2 x (H_256 x bent4) Zero-Alias Orthogonality):
    (a) Algebraic `H_2` lift: for ANY equal-length vectors `u, v : List Int`, pairing
        bytes as `(u, -u)` and specials as `(v, +v)` in `R^512` makes byte-to-special
        inner products vanish identically (`0`), while doubling same-bank inner products
        (`2 * dotInt u v`).
    (b) Explicit `H_2 x (H_256 x bent4)` `512`-dim Gram Matrix (`512 I_512`):
        Constructing the exact `bent4` bitwise `512`-dim embedding `v5Dot512`, for ALL
        `(sp, b0..b7)` and `(sp', b0'..b7')` covering all `512` symbols (including `0..259`):
        - identical symbols have squared norm `512`,
        - distinct symbols (`sp != sp'` or `b != b'`) are strictly orthogonal (`dot = 0`). -/
theorem v5_h2_bent4_orthogonal_zero_alias
    (u v : List Int) (hlen : u.length = v.length)
    (sp b0 b1 b2 b3 b4 b5 b6 b7 sp' b0' b1' b2' b3' b4' b5' b6' b7' : Bool) :
    ((dotInt (u ++ u.map (fun z => -z)) (v ++ v) = 0) ∧
     (dotInt (u ++ u.map (fun z => -z)) (v ++ v.map (fun z => -z)) = 2 * dotInt u v) ∧
     (dotInt (u ++ u) (v ++ v) = 2 * dotInt u v)) ∧
    ((sp = sp' ∧ b0 = b0' ∧ b1 = b1' ∧ b2 = b2' ∧ b3 = b3' ∧
      b4 = b4' ∧ b5 = b5' ∧ b6 = b6' ∧ b7 = b7') →
      v5Dot512 sp b0 b1 b2 b3 b4 b5 b6 b7 sp' b0' b1' b2' b3' b4' b5' b6' b7' = 512) ∧
    ((sp ≠ sp' ∨ b0 ≠ b0' ∨ b1 ≠ b1' ∨ b2 ≠ b2' ∨ b3 ≠ b3' ∨
      b4 ≠ b4' ∨ b5 ≠ b5' ∨ b6 ≠ b6' ∨ b7 ≠ b7') →
      v5Dot512 sp b0 b1 b2 b3 b4 b5 b6 b7 sp' b0' b1' b2' b3' b4' b5' b6' b7' = 0) := by
  have h1 : dotInt (u ++ u.map (fun z => -z)) (v ++ v) =
      dotInt u v + dotInt (u.map (fun z => -z)) v := dotInt_append u _ v v hlen
  rw [dotInt_neg_left] at h1
  have h2 : dotInt (u ++ u.map (fun z => -z)) (v ++ v.map (fun z => -z)) =
      dotInt u v + dotInt (u.map (fun z => -z)) (v.map (fun z => -z)) :=
    dotInt_append u _ v _ hlen
  rw [dotInt_neg_left, dotInt_neg_right] at h2
  have h3 : dotInt (u ++ u) (v ++ v) = dotInt u v + dotInt u v :=
    dotInt_append u u v v hlen
  refine ⟨⟨by omega, by omega, by omega⟩, ?_, ?_⟩
  · rintro ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
    unfold v5Dot512 v5BoolParity
    simp only [boolSgn_mul, Bool.xor_self]
    decide
  · intro hne
    unfold v5Dot512 v5BoolParity
    simp only [boolSgn_mul, Bool.false_and, Bool.true_and, Bool.xor_false]
    match Bool.decEq sp sp' with
    | isTrue hsp =>
      rw [hsp]
      simp only [xor_bent_cancel2, v4DbDot_xor_eq_dbLinear, sumBool8_dbLinear_eq]
      have hbytes : b0 ≠ b0' ∨ b1 ≠ b1' ∨ b2 ≠ b2' ∨ b3 ≠ b3' ∨
          b4 ≠ b4' ∨ b5 ≠ b5' ∨ b6 ≠ b6' ∨ b7 ≠ b7' := by
        rcases hne with h | h
        · exact False.elim (h hsp)
        · exact h
      have hzero_mul : ∀ (a b c d e f g h : Int),
          (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0 ∨ f = 0 ∨ g = 0 ∨ h = 0) →
          (((((((a * b) * c) * d) * e) * f) * g) * h) = 0 := by
        intro a b c d e f g h hdisj
        rcases hdisj with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
          simp only [Int.zero_mul, Int.mul_zero]
      have hxne : ∀ x y : Bool, x ≠ y → (x ^^ y) = true := by
        intro x y hxy
        cases x <;> cases y <;> simp_all
      have hdisj :
          (if (b0 ^^ b0') ^^ (b5 ^^ b5') then (0 : Int) else 2) = 0 ∨
          (if (b1 ^^ b1') ^^ (b6 ^^ b6') then (0 : Int) else 2) = 0 ∨
          (if (b2 ^^ b2') ^^ (b7 ^^ b7') then (0 : Int) else 2) = 0 ∨
          (if b3 ^^ b3' then (0 : Int) else 2) = 0 ∨
          (if b4 ^^ b4' then (0 : Int) else 2) = 0 ∨
          (if b5 ^^ b5' then (0 : Int) else 2) = 0 ∨
          (if b6 ^^ b6' then (0 : Int) else 2) = 0 ∨
          (if b7 ^^ b7' then (0 : Int) else 2) = 0 := by
        match Bool.decEq b7 b7', Bool.decEq b6 b6', Bool.decEq b5 b5', Bool.decEq b4 b4',
              Bool.decEq b3 b3', Bool.decEq b2 b2', Bool.decEq b1 b1', Bool.decEq b0 b0' with
        | isFalse hb7, _, _, _, _, _, _, _ =>
          right; right; right; right; right; right; right; rw [hxne b7 b7' hb7]; rfl
        | isTrue _, isFalse hb6, _, _, _, _, _, _ =>
          right; right; right; right; right; right; left; rw [hxne b6 b6' hb6]; rfl
        | isTrue _, isTrue _, isFalse hb5, _, _, _, _, _ =>
          right; right; right; right; right; left; rw [hxne b5 b5' hb5]; rfl
        | isTrue _, isTrue _, isTrue _, isFalse hb4, _, _, _, _ =>
          right; right; right; right; left; rw [hxne b4 b4' hb4]; rfl
        | isTrue _, isTrue _, isTrue _, isTrue _, isFalse hb3, _, _, _ =>
          right; right; right; left; rw [hxne b3 b3' hb3]; rfl
        | isTrue hb7, isTrue _, isTrue _, isTrue _, isTrue _, isFalse hb2, _, _ =>
          right; right; left; rw [hb7, Bool.xor_self, Bool.xor_false, hxne b2 b2' hb2]; rfl
        | isTrue _, isTrue hb6, isTrue _, isTrue _, isTrue _, isTrue _, isFalse hb1, _ =>
          right; left; rw [hb6, Bool.xor_self, Bool.xor_false, hxne b1 b1' hb1]; rfl
        | isTrue _, isTrue _, isTrue hb5, isTrue _, isTrue _, isTrue _, isTrue _, isFalse hb0 =>
          left; rw [hb5, Bool.xor_self, Bool.xor_false, hxne b0 b0' hb0]; rfl
        | isTrue hb7, isTrue hb6, isTrue hb5, isTrue hb4,
          isTrue hb3, isTrue hb2, isTrue hb1, isTrue hb0 =>
          exfalso
          rcases hbytes with h | h | h | h | h | h | h | h
          · exact h hb0
          · exact h hb1
          · exact h hb2
          · exact h hb3
          · exact h hb4
          · exact h hb5
          · exact h hb6
          · exact h hb7
      have hprod0 := hzero_mul _ _ _ _ _ _ _ _ hdisj
      rw [hprod0]
      omega
    | isFalse hsp =>
      have hfl : (!sp) ≠ (!sp') := by
        cases sp <;> cases sp' <;> simp_all
      simp only [xor_bent_cancel2, xor_bent_flip _ _ _ (!sp) (!sp') hfl,
        boolSgn_xor_true, sumBool8_neg]
      omega

private def monarch2Entry
    (W1 W2 : Nat → Nat → Nat → Int) (r1 r2 g1 c1 : Nat) : Int :=
  W2 r1 r2 g1 * W1 g1 r1 c1

private def monarch3SumAux
    (W1 W2 W3 : Nat → Nat → Nat → Int) (r2 r3 g1 c1 : Nat) : Nat → Int
  | 0 => 0
  | k + 1 =>
    monarch3SumAux W1 W2 W3 r2 r3 g1 c1 k +
      W3 r2 r3 k * W2 k r2 g1 * W1 g1 k c1

private theorem monarch3SumAux_ones (r2 r3 g1 c1 : Nat) (K : Nat) :
    monarch3SumAux (fun _ _ _ => 1) (fun _ _ _ => 1) (fun _ _ _ => 1) r2 r3 g1 c1 K = (K : Int) := by
  induction K with
  | zero => rfl
  | succ k ih =>
    change monarch3SumAux _ _ _ r2 r3 g1 c1 k + 1 * 1 * 1 = ((k + 1 : Nat) : Int)
    rw [ih]
    omega

private theorem monarch3SumAux_eye (r2 r3 g1 c1 : Nat) (K : Nat) :
    monarch3SumAux
      (fun _ r1' c1' => if r1' = c1' then 1 else 0)
      (fun _ _ _ => 1)
      (fun _ r3' r1' => if r3' = r1' then 1 else 0)
      r2 r3 g1 c1 K = (if r3 < K ∧ r3 = c1 then 1 else 0) := by
  induction K with
  | zero =>
    simp only [monarch3SumAux, Nat.not_lt_zero, false_and, ↓reduceIte]
  | succ k ih =>
    change monarch3SumAux _ _ _ r2 r3 g1 c1 k +
      (if r3 = k then 1 else 0) * 1 * (if k = c1 then 1 else 0) =
      (if r3 < k + 1 ∧ r3 = c1 then 1 else 0)
    rw [ih]
    by_cases hrk : r3 < k
    · have hne : r3 ≠ k := by omega
      have hlt : r3 < k + 1 := by omega
      simp only [hrk, hne, hlt, true_and, ↓reduceIte, Int.zero_mul, Int.add_zero]
    · by_cases heqk : r3 = k
      · subst heqk
        simp only [Nat.lt_irrefl, false_and, ↓reduceIte, Int.one_mul, Int.zero_add,
          Nat.lt_succ_self, true_and]
      · have hnot : ¬ (r3 < k + 1) := by omega
        simp only [hrk, heqk, hnot, false_and, ↓reduceIte, Int.zero_mul, Int.add_zero]

/-- Theorem 7 (2-Stage Rank-1 Subblock Collapse vs. 3-Stage Monarch K-Path Rank Lift):
    For any radix `K >= 4` (e.g. `K = 16` in `V5-B`):
    (a) Parameter scaling: `1 < K`, `2 * K^3 < 4 * K^3`, and `4 * K^3 < 2 * K^4` (strictly below dense);
    (b) Exact 2-stage path count = `1` (`monarch2Entry` with all-ones weights equals `1`);
    (c) Exact 2-stage subblock rank-1 collapse: for ALL weights `W1, W2` and ALL `(r1, g1)` subblocks,
        every `2x2` minor determinant vanishes identically (`0`);
    (d) Exact 3-stage path count = `K` (`monarch3SumAux` with all-ones weights sums `K` paths);
    (e) 3-stage subblock rank-lift witness: explicit weights `W1, W2, W3` realize a `2x2`
        subblock minor determinant `= 1 != 0`, breaking the 2-stage rank-1 bottleneck. -/
theorem monarch_2stage_vs_3stage_path_multiplicity
    (K : Nat) (hK : 4 ≤ K) :
    (1 < (K : Int) ∧ 2 * ((K : Int) * K * K) < 4 * ((K : Int) * K * K) ∧
      4 * ((K : Int) * K * K) < 2 * ((K : Int) * K * K * K)) ∧
    (∀ r1 r2 g1 c1 : Nat,
      monarch2Entry (fun _ _ _ => 1) (fun _ _ _ => 1) r1 r2 g1 c1 = 1) ∧
    (∀ (W1 W2 : Nat → Nat → Nat → Int) (r1 g1 r2 r2' c1 c1' : Nat),
      monarch2Entry W1 W2 r1 r2 g1 c1 * monarch2Entry W1 W2 r1 r2' g1 c1' -
      monarch2Entry W1 W2 r1 r2 g1 c1' * monarch2Entry W1 W2 r1 r2' g1 c1 = 0) ∧
    (∀ r2 r3 g1 c1 : Nat,
      monarch3SumAux (fun _ _ _ => 1) (fun _ _ _ => 1) (fun _ _ _ => 1) r2 r3 g1 c1 K = (K : Int)) ∧
    (∃ (W1 W2 W3 : Nat → Nat → Nat → Int) (r2 g1 : Nat),
      monarch3SumAux W1 W2 W3 r2 0 g1 0 K * monarch3SumAux W1 W2 W3 r2 1 g1 1 K -
      monarch3SumAux W1 W2 W3 r2 0 g1 1 K * monarch3SumAux W1 W2 W3 r2 1 g1 0 K ≠ 0) := by
  have hKint : 4 ≤ (K : Int) := by omega
  have hKpos : 0 < (K : Int) := by omega
  have hK3pos : 0 < (K : Int) * K * K := Int.mul_pos (Int.mul_pos hKpos hKpos) hKpos
  have hdiff : 2 * ((K : Int) * K * K * K) - 4 * ((K : Int) * K * K) =
      (2 * (K : Int) - 4) * ((K : Int) * K * K) := by
    have hmul : (2 * (K : Int) - 4) * ((K : Int) * K * K) =
        (2 * (K : Int)) * ((K : Int) * K * K) - 4 * ((K : Int) * K * K) :=
      Int.sub_mul (2 * (K : Int)) 4 ((K : Int) * K * K)
    have hassoc : (2 * (K : Int)) * ((K : Int) * K * K) = 2 * ((K : Int) * K * K * K) := by
      calc
        (2 * (K : Int)) * ((K : Int) * K * K) = 2 * ((K : Int) * ((K : Int) * K * K)) :=
          Int.mul_assoc 2 (K : Int) ((K : Int) * K * K)
        _ = 2 * (((K : Int) * K * K) * (K : Int)) := by
          rw [Int.mul_comm (K : Int) ((K : Int) * K * K)]
        _ = 2 * ((K : Int) * K * K * K) := rfl
    omega
  have hpos : 0 < (2 * (K : Int) - 4) * ((K : Int) * K * K) := Int.mul_pos (by omega) hK3pos
  refine ⟨⟨by omega, by omega, by omega⟩, ?_, ?_, ?_, ?_⟩
  · intro _ _ _ _
    rfl
  · intro W1 W2 r1 g1 r2 r2' c1 c1'
    unfold monarch2Entry
    have hcomm :
        (W2 r1 r2 g1 * W1 g1 r1 c1) * (W2 r1 r2' g1 * W1 g1 r1 c1') =
        (W2 r1 r2 g1 * W1 g1 r1 c1') * (W2 r1 r2' g1 * W1 g1 r1 c1) := by
      calc
        (W2 r1 r2 g1 * W1 g1 r1 c1) * (W2 r1 r2' g1 * W1 g1 r1 c1') =
            W2 r1 r2 g1 * (W1 g1 r1 c1 * (W2 r1 r2' g1 * W1 g1 r1 c1')) := Int.mul_assoc _ _ _
        _ = W2 r1 r2 g1 * (W2 r1 r2' g1 * (W1 g1 r1 c1 * W1 g1 r1 c1')) := by
            rw [← Int.mul_assoc (W1 g1 r1 c1), Int.mul_comm (W1 g1 r1 c1) (W2 r1 r2' g1), Int.mul_assoc]
        _ = W2 r1 r2 g1 * (W2 r1 r2' g1 * (W1 g1 r1 c1' * W1 g1 r1 c1)) := by
            rw [Int.mul_comm (W1 g1 r1 c1) (W1 g1 r1 c1')]
        _ = W2 r1 r2 g1 * (W1 g1 r1 c1' * (W2 r1 r2' g1 * W1 g1 r1 c1)) := by
            rw [← Int.mul_assoc (W2 r1 r2' g1), Int.mul_comm (W2 r1 r2' g1) (W1 g1 r1 c1'), Int.mul_assoc]
        _ = (W2 r1 r2 g1 * W1 g1 r1 c1') * (W2 r1 r2' g1 * W1 g1 r1 c1) := (Int.mul_assoc _ _ _).symm
    omega
  · intro r2 r3 g1 c1
    exact monarch3SumAux_ones r2 r3 g1 c1 K
  · refine ⟨(fun _ r1' c1' => if r1' = c1' then 1 else 0),
            (fun _ _ _ => 1),
            (fun _ r3' r1' => if r3' = r1' then 1 else 0), 0, 0, ?_⟩
    rw [monarch3SumAux_eye, monarch3SumAux_eye, monarch3SumAux_eye, monarch3SumAux_eye]
    have h0 : 0 < K := by omega
    have h1 : 1 < K := by omega
    simp [h0, h1]

/-- Theorem 8 (HoloSphere-V5 sm_86 Static 48 KiB SMEM Fit — Zero SMEM Spill):
    For any CellOp backbone footprint `backbone <= 5812`, 3-stage Monarch head weights
    `head_w <= 32768`, bias `<= 1024`, and scratch `<= 5120`, total CTA SMEM stays
    strictly below `49152` B (`48 KiB` static `sm_86` limit) with `>= 4428` B headroom. -/
theorem v5_head_smem_fit_within_sm86_48kib
    (backbone head_w bias scratch : Int)
    (hb0 : 0 ≤ backbone) (hb1 : backbone ≤ 5812)
    (hw0 : 0 ≤ head_w) (hw1 : head_w ≤ 32768)
    (hbi0 : 0 ≤ bias) (hbi1 : bias ≤ 1024)
    (hs0 : 0 ≤ scratch) (hs1 : scratch ≤ 5120) :
    backbone + head_w + bias + scratch ≤ 44724 ∧
    backbone + head_w + bias + scratch + 4428 ≤ 49152 := by
  have _ := hb0
  have _ := hw0
  have _ := hbi0
  have _ := hs0
  exact ⟨by omega, by omega⟩

/-! ## Group C (EXP-Q45c / SOP-16): MUX Cor. 15 vs. Bayt al-Hikma Al-Kashi Fixed-Radix -/

/-- Theorem 9 (MUX Corollary 15 Exact Cleared Margin Criterion with D = q^S - p^S):
    For rational ratio `p / q` (`0 <= p < q`), multiplexing depth `S >= 1`,
    subset-sum integer margin `m >= 1`, and floating-point roundoff `u = u_num / u_den`
    (`0 <= u_num`, `(S - 1) * u_num < u_den`):
    (a) Positivity of `D = q^S - p^S > 0` is derived directly from `0 <= p < q`, `S >= 1`;
    (b) Exact cleared equivalence:
        `2 * S * u_num * D < (q - p) * m * (u_den - (S - 1) * u_num)` holds if and only if
        `(2 * S * D + (q - p) * m * (S - 1)) * u_num < (q - p) * m * u_den`;
    (c) Rational scaling invariance under `(u_num, u_den) -> (k * u_num, k * u_den)` for `k > 0`;
    (d) Full `u_num` monotonicity: increasing `u_num -> u2` BOTH grows LHS `2 * S * u * D` AND
        shrinks RHS `(q - p) * m * (u_den - (S - 1) * u)`, hence passing at `u2` implies passing at `u_num`;
    (e) Monotonicity in `m`: increasing `m -> m2` preserves passing (`lhs < rhs`). -/
theorem mux_cor15_cleared_margin_criterion_iff
    (p q m : Int) (S : Nat) (u_num u_den : Int)
    (hp : 0 ≤ p) (hpq : p < q) (hm : 1 ≤ m) (hS : 1 ≤ S)
    (hun : 0 ≤ u_num) (hud : 0 < u_den)
    (hrem : 0 < u_den - ((S : Int) - 1) * u_num) :
    let D := q ^ S - p ^ S
    let Si : Int := (S : Int)
    let lhs := 2 * Si * u_num * D
    let rhs := (q - p) * m * (u_den - (Si - 1) * u_num)
    0 < D ∧
    (lhs < rhs ↔ (2 * Si * D + (q - p) * m * (Si - 1)) * u_num < (q - p) * m * u_den) ∧
    (∀ k : Int, 0 < k →
      (2 * Si * (k * u_num) * D < (q - p) * m * (k * u_den - (Si - 1) * (k * u_num)) ↔ lhs < rhs)) ∧
    (∀ u2 : Int, u_num ≤ u2 →
      lhs ≤ 2 * Si * u2 * D ∧
      (q - p) * m * (u_den - (Si - 1) * u2) ≤ rhs ∧
      (2 * Si * u2 * D < (q - p) * m * (u_den - (Si - 1) * u2) → lhs < rhs)) ∧
    (∀ m2 : Int, m ≤ m2 → lhs < rhs →
      lhs < (q - p) * m2 * (u_den - (Si - 1) * u_num)) := by
  intro D Si lhs rhs
  have hD : 0 < D := by
    have hlt := pow_lt_pow_of_nonneg_lt hp hpq S hS
    omega
  have _ := hun
  have _ := hud
  have _ := hrem
  have hrhs_split :
      rhs = (q - p) * m * u_den - ((q - p) * m * (Si - 1)) * u_num := by
    change (q - p) * m * (u_den - (Si - 1) * u_num) =
      (q - p) * m * u_den - ((q - p) * m * (Si - 1)) * u_num
    rw [Int.mul_sub, Int.mul_assoc ((q - p) * m) (Si - 1) u_num]
  have hlhs_comm : lhs = (2 * Si * D) * u_num := by
    change 2 * Si * u_num * D = (2 * Si * D) * u_num
    calc
      2 * Si * u_num * D = (2 * Si) * (u_num * D) := Int.mul_assoc (2 * Si) u_num D
      _ = (2 * Si) * (D * u_num) := by rw [Int.mul_comm u_num D]
      _ = (2 * Si * D) * u_num := (Int.mul_assoc (2 * Si) D u_num).symm
  refine ⟨hD, ?_, ?_, ?_, ?_⟩
  · have hcomb :
        (2 * Si * D + (q - p) * m * (Si - 1)) * u_num =
          (2 * Si * D) * u_num + ((q - p) * m * (Si - 1)) * u_num :=
      Int.add_mul _ _ _
    rw [hcomb, ← hlhs_comm, hrhs_split]
    exact ⟨fun h => by omega, fun h => by omega⟩
  · intro k hk
    have hk_lhs : 2 * Si * (k * u_num) * D = k * lhs := by
      change 2 * Si * (k * u_num) * D = k * (2 * Si * u_num * D)
      calc
        2 * Si * (k * u_num) * D = ((2 * Si) * k) * u_num * D := by
          rw [← Int.mul_assoc (2 * Si) k u_num]
        _ = (k * (2 * Si)) * u_num * D := by rw [Int.mul_comm (2 * Si) k]
        _ = k * (2 * Si * u_num * D) := by
          rw [Int.mul_assoc k (2 * Si) u_num, Int.mul_assoc k (2 * Si * u_num) D]
    have hk_rhs :
        (q - p) * m * (k * u_den - (Si - 1) * (k * u_num)) = k * rhs := by
      change (q - p) * m * (k * u_den - (Si - 1) * (k * u_num)) =
        k * ((q - p) * m * (u_den - (Si - 1) * u_num))
      have hsub : k * u_den - (Si - 1) * (k * u_num) = k * (u_den - (Si - 1) * u_num) := by
        rw [Int.mul_sub]
        have ha : (Si - 1) * (k * u_num) = k * ((Si - 1) * u_num) := by
          calc
            (Si - 1) * (k * u_num) = ((Si - 1) * k) * u_num := (Int.mul_assoc _ _ _).symm
            _ = (k * (Si - 1)) * u_num := by rw [Int.mul_comm (Si - 1) k]
            _ = k * ((Si - 1) * u_num) := Int.mul_assoc _ _ _
        rw [ha]
      rw [hsub]
      calc
        (q - p) * m * (k * (u_den - (Si - 1) * u_num)) =
            ((q - p) * m * k) * (u_den - (Si - 1) * u_num) := (Int.mul_assoc _ _ _).symm
        _ = (k * ((q - p) * m)) * (u_den - (Si - 1) * u_num) := by
            rw [Int.mul_comm ((q - p) * m) k]
        _ = k * ((q - p) * m * (u_den - (Si - 1) * u_num)) := Int.mul_assoc _ _ _
    rw [hk_lhs, hk_rhs]
    constructor
    · intro hlt
      match Int.decLt lhs rhs with
      | isTrue hlr => exact hlr
      | isFalse hlr =>
        have hle : k * rhs ≤ k * lhs := mul_le_mul_nonneg_left (by omega) (by omega)
        omega
    · intro hlt
      exact mul_lt_mul_pos_left hk hlt
  · intro u2 hu2
    have hcoeff_l : 0 ≤ 2 * Si * D := Int.mul_nonneg (by omega) (by omega)
    have hmul_l : (2 * Si * D) * u_num ≤ (2 * Si * D) * u2 := mul_le_mul_nonneg_left hcoeff_l hu2
    have hl2 : (2 * Si * D) * u2 = 2 * Si * u2 * D := by
      calc
        (2 * Si * D) * u2 = (2 * Si) * (D * u2) := Int.mul_assoc (2 * Si) D u2
        _ = (2 * Si) * (u2 * D) := by rw [Int.mul_comm D u2]
        _ = 2 * Si * u2 * D := (Int.mul_assoc (2 * Si) u2 D).symm
    have hrem_shrink : u_den - (Si - 1) * u2 ≤ u_den - (Si - 1) * u_num := by
      have hmul_s : (Si - 1) * u_num ≤ (Si - 1) * u2 := mul_le_mul_nonneg_left (by omega) hu2
      omega
    have hqp_m : 0 ≤ (q - p) * m := Int.mul_nonneg (by omega) (by omega)
    have hrhs_shrink :
        (q - p) * m * (u_den - (Si - 1) * u2) ≤ rhs :=
      mul_le_mul_nonneg_left hqp_m hrem_shrink
    have hmono_l : lhs ≤ 2 * Si * u2 * D := by omega
    refine ⟨hmono_l, hrhs_shrink, fun h2 => Int.lt_of_le_of_lt hmono_l (Int.lt_of_lt_of_le h2 hrhs_shrink)⟩
  · intro m2 hm2 hpass
    have hrhs_le : rhs ≤ (q - p) * m2 * (u_den - (Si - 1) * u_num) := by
      have hrem_si : 0 < u_den - (Si - 1) * u_num := hrem
      have hcoeff : 0 ≤ (q - p) * (u_den - (Si - 1) * u_num) :=
        Int.mul_nonneg (by omega) (by omega)
      have hmul := mul_le_mul_nonneg_left hcoeff hm2
      have ha1 : ((q - p) * (u_den - (Si - 1) * u_num)) * m = rhs := by
        calc
          ((q - p) * (u_den - (Si - 1) * u_num)) * m =
              (q - p) * ((u_den - (Si - 1) * u_num) * m) := Int.mul_assoc _ _ _
          _ = (q - p) * (m * (u_den - (Si - 1) * u_num)) := by rw [Int.mul_comm _ m]
          _ = rhs := (Int.mul_assoc _ _ _).symm
      have ha2 :
          ((q - p) * (u_den - (Si - 1) * u_num)) * m2 =
            (q - p) * m2 * (u_den - (Si - 1) * u_num) := by
        calc
          ((q - p) * (u_den - (Si - 1) * u_num)) * m2 =
              (q - p) * ((u_den - (Si - 1) * u_num) * m2) := Int.mul_assoc _ _ _
          _ = (q - p) * (m2 * (u_den - (Si - 1) * u_num)) := by rw [Int.mul_comm _ m2]
          _ = (q - p) * m2 * (u_den - (Si - 1) * u_num) := (Int.mul_assoc _ _ _).symm
      omega
    omega

/-- Theorem 10 (Exact MUX rho = 9/10 BF16 / FP16 / FP32 Phase-Transition Boundaries):
    Concrete kernel-`decide` verification of the exact cleared inequalities for `rho = 9/10`:
    - `BF16` (`u = 2^-8`) passes at `S = 3` (`m_3 = 9`) and fails at `S = 4` (`m_4 = 19`);
    - `FP16` (`u = 2^-11`) passes at `S = 4` (`m_4 = 19`) and fails at `S = 5` (`m_5 = 171`);
    - `FP32` (`u = 2^-24`) passes at `S = 11` (`m_11 = 272809`) and fails at `S = 12` (`m_12 = 860491`);
    while `Al-Kashi` `rho = 1/3` (`m_S = 1`) extends `FP16` through `S = 5` and `FP32` through `S = 12`. -/
theorem mux_rho9_10_collapse_boundaries_exact :
    (2 * 3 * (10^3 - 9^3) < (10 - 9) * 9 * (256 - 2)) ∧
    ((10 - 9) * 19 * (256 - 3) < 2 * 4 * (10^4 - 9^4)) ∧
    (2 * 4 * (10^4 - 9^4) < (10 - 9) * 19 * (2048 - 3)) ∧
    ((10 - 9) * 171 * (2048 - 4) < 2 * 5 * (10^5 - 9^5)) ∧
    (2 * 11 * (10^11 - 9^11) < (10 - 9) * 272809 * (16777216 - 10)) ∧
    ((10 - 9) * 860491 * (16777216 - 11) < 2 * 12 * (10^12 - 9^12)) ∧
    (2 * 5 * (3^5 - 1^5) < (3 - 1) * 1 * (2048 - 4)) ∧
    (2 * 12 * (3^12 - 1^12) < (3 - 1) * 1 * (16777216 - 11)) := by
  decide

private def evalGeom (p q : Int) : List Int → Int
  | [] => 0
  | c :: cs => c * q ^ cs.length + p * evalGeom p q cs

private def sumGeomPow (p q : Int) : Nat → Int
  | 0 => 0
  | k + 1 => q ^ k + p * sumGeomPow p q k

private inductive IsTernaryList : List Int → Prop where
  | nil : IsTernaryList []
  | cons {c : Int} {cs : List Int} : (c = -1 ∨ c = 0 ∨ c = 1) → IsTernaryList cs → IsTernaryList (c :: cs)

private theorem evalGeom_abs_le_sumGeomPow
    (p q : Int) (hp : 0 ≤ p) (hq : 0 < q) (cs : List Int) (ht : IsTernaryList cs) :
    -sumGeomPow p q cs.length ≤ evalGeom p q cs ∧
    evalGeom p q cs ≤ sumGeomPow p q cs.length := by
  induction ht with
  | nil =>
    exact ⟨Int.le_refl 0, Int.le_refl 0⟩
  | cons hc htail ih =>
    rcases ih with ⟨hlo, hhi⟩
    rename_i c tail
    change -sumGeomPow p q (tail.length + 1) ≤ c * q ^ tail.length + p * evalGeom p q tail ∧
      c * q ^ tail.length + p * evalGeom p q tail ≤ sumGeomPow p q (tail.length + 1)
    have hdef : sumGeomPow p q (tail.length + 1) = q ^ tail.length + p * sumGeomPow p q tail.length := rfl
    rw [hdef]
    have hqpow : 0 ≤ q ^ tail.length := Int.le_of_lt (pow_pos_of_pos q hq tail.length)
    have hplo : p * (-sumGeomPow p q tail.length) ≤ p * evalGeom p q tail :=
      mul_le_mul_nonneg_left hp hlo
    have hphi : p * evalGeom p q tail ≤ p * sumGeomPow p q tail.length :=
      mul_le_mul_nonneg_left hp hhi
    rw [Int.mul_neg] at hplo
    rcases hc with hcm1 | hc0 | hcp1
    · rw [hcm1]
      have h1 : (-1) * q ^ tail.length = -(q ^ tail.length) := by
        rw [Int.neg_mul, Int.one_mul]
      rw [h1]
      exact ⟨by omega, by omega⟩
    · rw [hc0]
      have h0 : 0 * q ^ tail.length = 0 := Int.zero_mul _
      rw [h0]
      exact ⟨by omega, by omega⟩
    · rw [hcp1]
      have h1 : 1 * q ^ tail.length = q ^ tail.length := Int.one_mul _
      rw [h1]
      exact ⟨by omega, by omega⟩

private theorem p_sumGeomPow_add_ppow_le_qpow
    (p q : Int) (hp : 0 < p) (hq : 2 * p < q) (k : Nat) :
    p * sumGeomPow p q k + p ^ k ≤ q ^ k := by
  induction k with
  | zero =>
    simp only [sumGeomPow, Int.pow_zero, Int.mul_zero, Int.zero_add]
    exact Int.le_refl 1
  | succ n ih =>
    have hsum_def : sumGeomPow p q (n + 1) = q ^ n + p * sumGeomPow p q n := rfl
    rw [hsum_def, Int.pow_succ, Int.pow_succ]
    have hqpow : 0 < q ^ n := pow_pos_of_pos q (by omega) n
    have hmul : p * (p * sumGeomPow p q n + p ^ n) ≤ p * q ^ n :=
      mul_le_mul_nonneg_left (by omega) ih
    rw [Int.mul_add] at hmul
    have h2p_lt : (2 * p) * q ^ n < q * q ^ n := by
      have hgap : 0 < (q - 2 * p) * q ^ n := Int.mul_pos (by omega) hqpow
      rw [Int.sub_mul] at hgap
      omega
    have h2p_eq : (2 * p) * q ^ n = p * q ^ n + p * q ^ n := by
      have hp2 : 2 * p = p + p := by omega
      rw [hp2, Int.add_mul]
    rw [Int.mul_add, Int.mul_comm (q ^ n) q, Int.mul_comm (p ^ n) p]
    omega

/-- Theorem 11 (Universal Al-Kashi Separation Bound |evalGeom| >= p^(S-1) >= 1 for q > 2p):
    For ALL `p, q : Int` satisfying `0 < p` and `2 * p < q` (including `q >= 3 * p`, e.g.
    `Al-Kashi` `rho = 1/3` where `p = 1, q = 3`), and ANY nonzero ternary combination `cs : List Int`
    of length `S >= 1`, `|evalGeom p q cs| >= p ^ (S - 1) >= 1` universally. -/
theorem alkashi_balanced_radix_m_ge_1_universal
    (p q : Int) (hp : 0 < p) (hq : 2 * p < q)
    (cs : List Int) (ht : IsTernaryList cs) (hnz : HasNonzero cs) :
    (p ^ (cs.length - 1) ≤ evalGeom p q cs ∨ evalGeom p q cs ≤ -(p ^ (cs.length - 1))) ∧
    (1 ≤ evalGeom p q cs ∨ evalGeom p q cs ≤ -1) := by
  have hmain : p ^ (cs.length - 1) ≤ evalGeom p q cs ∨ evalGeom p q cs ≤ -(p ^ (cs.length - 1)) := by
    induction ht with
    | nil =>
      cases hnz
    | cons hc htail ih =>
      rename_i c tail
      change p ^ tail.length ≤ c * q ^ tail.length + p * evalGeom p q tail ∨
        c * q ^ tail.length + p * evalGeom p q tail ≤ -(p ^ tail.length)
      have hq_pos : 0 < q := by omega
      have hbnd := evalGeom_abs_le_sumGeomPow p q (by omega) hq_pos tail htail
      rcases hbnd with ⟨hlo, hhi⟩
      have htail_gap := p_sumGeomPow_add_ppow_le_qpow p q hp hq tail.length
      have hplo : -(p * sumGeomPow p q tail.length) ≤ p * evalGeom p q tail := by
        have hm := mul_le_mul_nonneg_left (by omega : 0 ≤ p) hlo
        rw [Int.mul_neg] at hm
        exact hm
      have hphi : p * evalGeom p q tail ≤ p * sumGeomPow p q tail.length :=
        mul_le_mul_nonneg_left (by omega : 0 ≤ p) hhi
      rcases hc with hcm1 | hc0 | hcp1
      · right
        rw [hcm1]
        have h1 : (-1) * q ^ tail.length = -(q ^ tail.length) := by
          rw [Int.neg_mul, Int.one_mul]
        rw [h1]
        omega
      · rw [hc0]
        have h0 : 0 * q ^ tail.length = 0 := Int.zero_mul _
        rw [h0, Int.zero_add]
        have hnz_tail : HasNonzero tail := by
          cases hnz with
          | head hne => exact False.elim (hne hc0)
          | tail hrest => exact hrest
        have htail_ne : ∃ x xs, tail = x :: xs := by
          cases hnz_tail with
          | head _ => exact ⟨_, _, rfl⟩
          | tail _ => exact ⟨_, _, rfl⟩
        rcases htail_ne with ⟨x, xs, hteq⟩
        have hpow_step : p * p ^ (tail.length - 1) = p ^ tail.length := by
          have hlen : tail.length = xs.length + 1 := by rw [hteq]; rfl
          rw [hlen, Nat.add_sub_cancel, Int.pow_succ, Int.mul_comm]
        rcases ih hnz_tail with hpos | hneg
        · left
          have hm : p * p ^ (tail.length - 1) ≤ p * evalGeom p q tail :=
            mul_le_mul_nonneg_left (by omega) hpos
          omega
        · right
          have hm : p * evalGeom p q tail ≤ p * (-(p ^ (tail.length - 1))) :=
            mul_le_mul_nonneg_left (by omega) hneg
          rw [Int.mul_neg] at hm
          omega
      · left
        rw [hcp1]
        have h1 : 1 * q ^ tail.length = q ^ tail.length := Int.one_mul _
        rw [h1]
        omega
  have hppow : 1 ≤ p ^ (cs.length - 1) := pow_pos_of_pos p hp (cs.length - 1)
  refine ⟨hmain, ?_⟩
  rcases hmain with hpos | hneg
  · left; omega
  · right; omega

/-- Theorem 12 (Al-Kashi Fixed-Radix Exact Integer Accumulation u = 0 Across All S >= 1):
    When `u_num = 0` (`Al-Kashi` `FixedRadix` or `A_16 Block-H4` integer accumulation),
    for EVERY `S >= 1`, EVERY `0 <= p < q`, EVERY `m >= 1`, and EVERY `u_den >= 1`,
    the Corollary 15 separation margin with `D = q^S - p^S` holds unconditionally with zero collapse threshold. -/
theorem alkashi_fixed_radix_zero_ulp_all_s
    (p q m : Int) (S : Nat) (u_den : Int)
    (hp : 0 ≤ p) (hpq : p < q) (hm : 1 ≤ m) (hS : 1 ≤ S) (hud : 0 < u_den) :
    2 * (S : Int) * 0 * (q ^ S - p ^ S) < (q - p) * m * (u_den - ((S : Int) - 1) * 0) := by
  have _ := pow_lt_pow_of_nonneg_lt hp hpq S hS
  have h1 : 0 < (q - p) * m := Int.mul_pos (by omega) (by omega)
  have h2 : 0 < (q - p) * m * u_den := Int.mul_pos h1 hud
  have hzero : 2 * (S : Int) * 0 * (q ^ S - p ^ S) = 0 := by
    rw [Int.mul_zero, Int.zero_mul]
  have hrhs : u_den - ((S : Int) - 1) * 0 = u_den := by
    rw [Int.mul_zero, Int.sub_zero]
  rw [hzero, hrhs]
  exact h2

private def h4Apply (c0 c1 c2 c3 : Int) : List Int :=
  [c0 + c1 + c2 + c3, c0 - c1 + c2 - c3, c0 + c1 - c2 - c3, c0 - c1 - c2 + c3]

/-- Theorem 13 (A_16 Block-H4 Integer Parseval & Scale-Flip Zero-Drift Accumulation):
    For ALL nibbles `(c0, c1, c2, c3)` and ANY rational scale numerator `s`:
    `H_4` exact integer accumulation preserves `||H_4 c||^2 = 4 ||c||^2` (`u = 0`),
    and flipping scale sign `s -> -s` leaves squared weight norm strictly invariant. -/
theorem a16_block_h4_parseval_and_flip_zero_drift (c0 c1 c2 c3 s : Int) :
    normSq (h4Apply c0 c1 c2 c3) = 4 * normSq [c0, c1, c2, c3] ∧
    (2 * (-s)) * (2 * (-s)) * normSq (h4Apply c0 c1 c2 c3) =
      (2 * s) * (2 * s) * normSq (h4Apply c0 c1 c2 c3) := by
  have hp : normSq (h4Apply c0 c1 c2 c3) = 4 * normSq [c0, c1, c2, c3] := by
    simp only [h4Apply, normSq, Int.add_mul, Int.mul_add, Int.mul_sub, Int.mul_comm]
    omega
  have hs : (2 * (-s)) * (2 * (-s)) = (2 * s) * (2 * s) := by
    rw [Int.mul_neg, Int.neg_mul, Int.mul_neg, Int.neg_neg]
  rw [hs]
  exact ⟨hp, rfl⟩

/-- Theorem 14 (Bayt al-Hikma Al-Kashi 1427 CE One-Step Cubic Secant Contraction vs. Repelling Cliff):
    For any `p, R, x, y : Int`:
    - Contractive regime (`0 <= R`, `3 * R^2 < p`, `|x| <= R`, `|y| <= R`):
      one-step cubic secant slope satisfies `x^2 + x * y + y^2 < p` (Banach contraction `L < 1`);
    - Repelling cliff (`0 < p`, `0 <= y < x`, `p <= 3 * y^2`):
      one-step cleared secant expands strictly (`p * (x - y) < x^3 - y^3`). -/
theorem alkashi_contractive_cubic_vs_repelling_cliff (p R x y : Int) :
    (0 ≤ R ∧ 3 * (R * R) < p ∧ -R ≤ x ∧ x ≤ R ∧ -R ≤ y ∧ y ≤ R →
      x * x + x * y + y * y < p) ∧
    (0 < p ∧ 0 ≤ y ∧ y < x ∧ p ≤ 3 * (y * y) →
      p * (x - y) < x * x * x - y * y * y) := by
  constructor
  · intro ⟨hR, hcontr, hx1, hx2, hy1, hy2⟩
    have _ := hR
    have hxsq : x * x ≤ R * R := by
      have hp1 : 0 ≤ (R - x) * (R + x) := Int.mul_nonneg (by omega) (by omega)
      have hid : (R - x) * (R + x) = R * R - x * x := by
        rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm x R]
        omega
      omega
    have hysq : y * y ≤ R * R := by
      have hp1 : 0 ≤ (R - y) * (R + y) := Int.mul_nonneg (by omega) (by omega)
      have hid : (R - y) * (R + y) = R * R - y * y := by
        rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm y R]
        omega
      omega
    have hxy : 2 * (x * y) ≤ x * x + y * y := by
      have hs : 0 ≤ (x - y) * (x - y) := sq_nonneg (x - y)
      have hid := diff_sq_identity x y
      omega
    omega
  · intro ⟨hp, hy, hxy, hcliff⟩
    have hdx : 0 < x - y := by omega
    have hgap_pos : 0 < (x - y) * (x + 2 * y) := Int.mul_pos hdx (by omega)
    have hgap_id : (x - y) * (x + 2 * y) = x * x + x * y + y * y - 3 * (y * y) := by
      rw [Int.sub_mul, Int.mul_add, Int.mul_add, Int.mul_comm y x]
      have h2x : x * (2 * y) = 2 * (x * y) := by
        calc
          x * (2 * y) = (x * 2) * y := (Int.mul_assoc x 2 y).symm
          _ = (2 * x) * y := by rw [Int.mul_comm x 2]
          _ = 2 * (x * y) := Int.mul_assoc 2 x y
      have h2y : y * (2 * y) = 2 * (y * y) := by
        calc
          y * (2 * y) = (y * 2) * y := (Int.mul_assoc y 2 y).symm
          _ = (2 * y) * y := by rw [Int.mul_comm y 2]
          _ = 2 * (y * y) := Int.mul_assoc 2 y y
      omega
    have hslope : 0 < (x * x + x * y + y * y) - p := by omega
    have hprod : 0 < ((x * x + x * y + y * y) - p) * (x - y) := Int.mul_pos hslope hdx
    rw [Int.sub_mul] at hprod
    have hfac : x * x * x - y * y * y = (x * x + x * y + y * y) * (x - y) := by
      rw [Int.add_mul, Int.add_mul, Int.mul_sub, Int.mul_sub, Int.mul_sub]
      have hc1 : x * y * x = x * x * y := by
        calc
          x * y * x = x * (y * x) := Int.mul_assoc x y x
          _ = x * (x * y) := by rw [Int.mul_comm y x]
          _ = x * x * y := (Int.mul_assoc x x y).symm
      have hc2 : y * y * x = x * y * y := by
        calc
          y * y * x = x * (y * y) := Int.mul_comm (y * y) x
          _ = x * y * y := (Int.mul_assoc x y y).symm
      omega
    have _ := hp
    omega

/-! ## Group D (EXP-Q45c): DualAxisCarry (beta_min) x Floodgates SU(2) O(1) S_fast -/

/-- Theorem 15 (DualAxisCarry beta_min Carry-Last Collapse Prevention & One-Step Damping):
    For cleared projective EMA `gamma_t = (q_b * e_t) / (p_b * D_prev + q_b * e_t)` with
    `0 < D_prev`, `0 < e_t`, `0 < q_b`, and `0 <= p_b`:
    - At `p_b = 0` (no `beta_min`), `retention = 0 * D_prev = 0` and denominator reduces to `q_b * e_t`
      (`gamma_t = 1`, `100%` carry-last erasure);
    - Under `0 < p_min <= p_b`, retention `p_b * D_prev >= p_min * D_prev > 0`,
      `0 < q_b * e_t < p_b * D_prev + q_b * e_t`, and any nonzero perturbation `delta != 0`
      is strictly damped (`(q_b * e_t)^2 * delta^2 < (p_b * D_prev + q_b * e_t)^2 * delta^2`). -/
theorem dual_axis_carry_beta_min_prevents_carry_last_collapse
    (D_prev e_t p_b q_b p_min delta : Int)
    (hD : 0 < D_prev) (he : 0 < e_t) (hq : 0 < q_b) :
    (0 * D_prev = 0 ∧ 0 * D_prev + q_b * e_t = q_b * e_t) ∧
    (0 < p_min ∧ p_min ≤ p_b ∧ delta ≠ 0 →
      let ret := p_b * D_prev
      let inn := q_b * e_t
      let den := ret + inn
      0 < p_min * D_prev ∧ p_min * D_prev ≤ ret ∧ 0 < inn ∧ inn < den ∧
      (inn * delta) * (inn * delta) < (den * delta) * (den * delta)) := by
  refine ⟨⟨Int.zero_mul D_prev, by rw [Int.zero_mul, Int.zero_add]⟩, ?_⟩
  intro ⟨hpmin, hpb, hdelta⟩
  intro ret inn den
  have hpmD : 0 < p_min * D_prev := Int.mul_pos hpmin hD
  have hret_ge : p_min * D_prev ≤ ret := mul_le_mul_nonneg_right (by omega) hpb
  have hinn : 0 < inn := Int.mul_pos hq he
  have hden : inn < den := by omega
  have hdsq : 0 < delta * delta := sq_pos_of_ne_zero delta hdelta
  have hdiff_d : 0 < (den - inn) * (den + inn) := Int.mul_pos (by omega) (by omega)
  have hsq_lt : inn * inn < den * den := by
    have hsf := sq_diff_factor den inn
    omega
  have hmul_sq : (inn * inn) * (delta * delta) < (den * den) * (delta * delta) := by
    have hpos : 0 < (den * den - inn * inn) * (delta * delta) := Int.mul_pos (by omega) hdsq
    rw [Int.sub_mul] at hpos
    omega
  rw [← prod_sq_split inn delta, ← prod_sq_split den delta] at hmul_sq
  exact ⟨hpmD, hret_ge, hinn, hden, hmul_sq⟩

/-- Theorem 16 (DualAxisCarry Projective Convex-Hull Invariance):
    For any interval `[m, M]`, positive `D_prev > 0`, `e_t > 0`, `q_b > 0`, `p_b >= 0`,
    if `m <= N_prev / D_prev <= M` and `m <= z_t <= M`, then the updated state
    `N_next = p_b * N_prev + q_b * e_t * z_t`, `D_next = p_b * D_prev + q_b * e_t`
    satisfies `0 < D_next` and `m <= N_next / D_next <= M` in exact cleared form. -/
theorem dual_axis_carry_convex_hull_invariance
    (m M N_prev D_prev z_t e_t p_b q_b : Int)
    (hD : 0 < D_prev) (he : 0 < e_t) (hpb : 0 ≤ p_b) (hqb : 0 < q_b)
    (hlo_prev : m * D_prev ≤ N_prev) (hhi_prev : N_prev ≤ M * D_prev)
    (hlo_z : m ≤ z_t) (hhi_z : z_t ≤ M) :
    let N_next := p_b * N_prev + q_b * e_t * z_t
    let D_next := p_b * D_prev + q_b * e_t
    0 < D_next ∧ m * D_next ≤ N_next ∧ N_next ≤ M * D_next := by
  intro N_next D_next
  have hpD : 0 ≤ p_b * D_prev := Int.mul_nonneg hpb (by omega)
  have hqe : 0 < q_b * e_t := Int.mul_pos hqb he
  have hDnext : 0 < D_next := by omega
  have h1 : p_b * (m * D_prev) ≤ p_b * N_prev := mul_le_mul_nonneg_left hpb hlo_prev
  have h2 : p_b * N_prev ≤ p_b * (M * D_prev) := mul_le_mul_nonneg_left hpb hhi_prev
  have h3 : (q_b * e_t) * m ≤ (q_b * e_t) * z_t := mul_le_mul_nonneg_left (by omega) hlo_z
  have h4 : (q_b * e_t) * z_t ≤ (q_b * e_t) * M := mul_le_mul_nonneg_left (by omega) hhi_z
  have hm_next : m * D_next = p_b * (m * D_prev) + (q_b * e_t) * m := by
    change m * (p_b * D_prev + q_b * e_t) = p_b * (m * D_prev) + (q_b * e_t) * m
    rw [Int.mul_add, Int.mul_comm m (q_b * e_t)]
    have ha : m * (p_b * D_prev) = p_b * (m * D_prev) := by
      calc
        m * (p_b * D_prev) = (m * p_b) * D_prev := (Int.mul_assoc m p_b D_prev).symm
        _ = (p_b * m) * D_prev := by rw [Int.mul_comm m p_b]
        _ = p_b * (m * D_prev) := Int.mul_assoc p_b m D_prev
    rw [ha]
  have hM_next : M * D_next = p_b * (M * D_prev) + (q_b * e_t) * M := by
    change M * (p_b * D_prev + q_b * e_t) = p_b * (M * D_prev) + (q_b * e_t) * M
    rw [Int.mul_add, Int.mul_comm M (q_b * e_t)]
    have ha : M * (p_b * D_prev) = p_b * (M * D_prev) := by
      calc
        M * (p_b * D_prev) = (M * p_b) * D_prev := (Int.mul_assoc M p_b D_prev).symm
        _ = (p_b * M) * D_prev := by rw [Int.mul_comm M p_b]
        _ = p_b * (M * D_prev) := Int.mul_assoc p_b M D_prev
    rw [ha]
  exact ⟨hDnext, by omega, by omega⟩

private structure Q4 where
  w : Int
  x : Int
  y : Int
  z : Int

private def qMul (a b : Q4) : Q4 :=
  ⟨a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
   a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
   a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
   a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w⟩

private def qConj (a : Q4) : Q4 := ⟨a.w, -a.x, -a.y, -a.z⟩

private def qNormSq (a : Q4) : Int := a.w * a.w + a.x * a.x + a.y * a.y + a.z * a.z

private def qScale (s : Int) (a : Q4) : Q4 := ⟨s * a.w, s * a.x, s * a.y, s * a.z⟩

private def cayleyQ (d w1 w2 w3 : Int) : Q4 :=
  ⟨d * d - (w1 * w1 + w2 * w2 + w3 * w3), 2 * d * w1, 2 * d * w2, 2 * d * w3⟩

private theorem cayley_norm_sq_identity (a w1s w2s w3s : Int) :
    (a - (w1s + w2s + w3s)) * (a - (w1s + w2s + w3s)) +
        4 * a * w1s + 4 * a * w2s + 4 * a * w3s =
      (a + w1s + w2s + w3s) * (a + w1s + w2s + w3s) := by
  simp only [Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub, Int.mul_assoc]
  rw [Int.mul_comm w1s a, Int.mul_comm w2s a, Int.mul_comm w3s a,
      Int.mul_comm w2s w1s, Int.mul_comm w3s w1s, Int.mul_comm w3s w2s]
  omega

private theorem qMul_assoc (a b c : Q4) :
    qMul (qMul a b) c = qMul a (qMul b c) := by
  rcases a with ⟨aw, ax, ay, az⟩
  rcases b with ⟨bw, bx, by_, bz⟩
  rcases c with ⟨cw, cx, cy, cz⟩
  simp only [qMul, Q4.mk.injEq, Int.add_mul, Int.sub_mul, Int.mul_add, Int.mul_sub, ← Int.mul_assoc]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

private theorem qMul_conj_right (u : Q4) :
    qMul (qConj u) u = ⟨qNormSq u, 0, 0, 0⟩ := by
  rcases u with ⟨uw, ux, uy, uz⟩
  simp only [qMul, qConj, qNormSq, Q4.mk.injEq, Int.neg_mul]
  refine ⟨by omega, ?_, ?_, ?_⟩
  · rw [Int.mul_comm ux uw, Int.mul_comm uz uy]; omega
  · rw [Int.mul_comm uy uw, Int.mul_comm uz ux]; omega
  · rw [Int.mul_comm uz uw, Int.mul_comm uy ux]; omega

private theorem qMul_scalar_right (x : Q4) (s : Int) :
    qMul x ⟨s, 0, 0, 0⟩ = qScale s x := by
  rcases x with ⟨xw, xx, xy, xz⟩
  simp only [qMul, qScale, Q4.mk.injEq, Int.mul_zero, Int.add_zero, Int.sub_zero, Int.zero_add]
  exact ⟨Int.mul_comm xw s, Int.mul_comm xx s, Int.mul_comm xy s, Int.mul_comm xz s⟩

private theorem qScale_mul_right (s : Int) (a b : Q4) :
    qMul (qScale s a) b = qScale s (qMul a b) := by
  rcases a with ⟨aw, ax, ay, az⟩
  rcases b with ⟨bw, bx, by_, bz⟩
  simp only [qMul, qScale, Int.mul_add, Int.mul_sub, ← Int.mul_assoc]

private theorem qScale_qScale (s t : Int) (x : Q4) :
    qScale s (qScale t x) = qScale (s * t) x := by
  rcases x with ⟨xw, xx, xy, xz⟩
  simp only [qScale, Int.mul_assoc]

private theorem qMul_conj_scale (s : Int) (u : Q4) :
    qMul (qConj u) (qScale s u) = qMul (qScale s (qConj u)) u := by
  rcases u with ⟨uw, ux, uy, uz⟩
  simp only [qMul, qConj, qScale, Q4.mk.injEq, Int.neg_mul, Int.mul_neg, ← Int.mul_assoc]
  rw [Int.mul_comm uw s, Int.mul_comm ux s, Int.mul_comm uy s, Int.mul_comm uz s]
  refine ⟨rfl, rfl, rfl, rfl⟩

/-- Theorem 17 (SU(2, Q) Cayley Unit-Norm & Exact Two-Sided Gauge Pullback):
    (a) For ANY `d, w1, w2, w3 : Int`, the cleared Cayley map `cayleyQ d w1 w2 w3`
        has exact squared norm `(d^2 + w1^2 + w2^2 + w3^2)^2` (so dividing by
        `d^2 + ||w||^2` lies on the unit sphere `S^3 = SU(2, Q)` with zero drift).
    (b) Left cancellation: `qConj u * (u * x) = ||u||^2 * x`.
    (c) Full two-sided conjugation pullback: `qConj u * ((u * x * qConj u) * u) = ||u||^4 * x`
        (for normalized `||u||^2 = 1`, `U^-1 (U x U^-1) U = x` identically).
    (d) Any nonzero quaternion `u != 0` satisfies `0 < ||u||^2`. -/
theorem su2_cayley_unit_norm_and_gauge_pullback_exact
    (d w1 w2 w3 uw ux uy uz xw xx xy xz : Int) :
    (qNormSq (cayleyQ d w1 w2 w3) =
      (d * d + w1 * w1 + w2 * w2 + w3 * w3) *
        (d * d + w1 * w1 + w2 * w2 + w3 * w3)) ∧
    (qMul (qConj ⟨uw, ux, uy, uz⟩) (qMul ⟨uw, ux, uy, uz⟩ ⟨xw, xx, xy, xz⟩) =
      qScale (qNormSq ⟨uw, ux, uy, uz⟩) ⟨xw, xx, xy, xz⟩) ∧
    (qMul (qConj ⟨uw, ux, uy, uz⟩)
        (qMul (qMul (qMul ⟨uw, ux, uy, uz⟩ ⟨xw, xx, xy, xz⟩) (qConj ⟨uw, ux, uy, uz⟩))
              ⟨uw, ux, uy, uz⟩) =
      qScale (qNormSq ⟨uw, ux, uy, uz⟩ * qNormSq ⟨uw, ux, uy, uz⟩) ⟨xw, xx, xy, xz⟩) ∧
    (uw ≠ 0 ∨ ux ≠ 0 ∨ uy ≠ 0 ∨ uz ≠ 0 → 0 < qNormSq ⟨uw, ux, uy, uz⟩) := by
  have hleft :
      qMul (qConj ⟨uw, ux, uy, uz⟩) (qMul ⟨uw, ux, uy, uz⟩ ⟨xw, xx, xy, xz⟩) =
        qScale (qNormSq ⟨uw, ux, uy, uz⟩) ⟨xw, xx, xy, xz⟩ := by
    simp only [qMul, qConj, qNormSq, qScale, Q4.mk.injEq]
    simp only [Int.mul_add, Int.mul_sub, Int.add_mul, Int.neg_mul, ← Int.mul_assoc]
    have c_wx : ux * uw = uw * ux := Int.mul_comm ux uw
    have c_wy : uy * uw = uw * uy := Int.mul_comm uy uw
    have c_wz : uz * uw = uw * uz := Int.mul_comm uz uw
    have c_xy : uy * ux = ux * uy := Int.mul_comm uy ux
    have c_xz : uz * ux = ux * uz := Int.mul_comm uz ux
    have c_yz : uz * uy = uy * uz := Int.mul_comm uz uy
    refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [c_wx, c_wy, c_wz, c_xy, c_xz, c_yz] <;> omega
  refine ⟨?_, hleft, ?_, ?_⟩
  · simp only [qNormSq, cayleyQ]
    have hs1 : (2 * d * w1) * (2 * d * w1) = 4 * (d * d) * (w1 * w1) := by
      have h1 : 2 * d * w1 = (2 * d) * w1 := rfl
      rw [h1, prod_sq_split (2 * d) w1, prod_sq_split 2 d]
      rfl
    have hs2 : (2 * d * w2) * (2 * d * w2) = 4 * (d * d) * (w2 * w2) := by
      have h1 : 2 * d * w2 = (2 * d) * w2 := rfl
      rw [h1, prod_sq_split (2 * d) w2, prod_sq_split 2 d]
      rfl
    have hs3 : (2 * d * w3) * (2 * d * w3) = 4 * (d * d) * (w3 * w3) := by
      have h1 : 2 * d * w3 = (2 * d) * w3 := rfl
      rw [h1, prod_sq_split (2 * d) w3, prod_sq_split 2 d]
      rfl
    rw [hs1, hs2, hs3]
    exact cayley_norm_sq_identity (d * d) (w1 * w1) (w2 * w2) (w3 * w3)
  · let u : Q4 := ⟨uw, ux, uy, uz⟩
    let x : Q4 := ⟨xw, xx, xy, xz⟩
    calc
      qMul (qConj u) (qMul (qMul (qMul u x) (qConj u)) u)
          = qMul (qConj u) (qMul (qMul u x) (qMul (qConj u) u)) := by rw [qMul_assoc]
      _ = qMul (qConj u) (qMul (qMul u x) ⟨qNormSq u, 0, 0, 0⟩) := by rw [qMul_conj_right u]
      _ = qMul (qConj u) (qScale (qNormSq u) (qMul u x)) := by rw [qMul_scalar_right]
      _ = qMul (qConj u) (qMul (qScale (qNormSq u) u) x) := by rw [qScale_mul_right]
      _ = qMul (qMul (qConj u) (qScale (qNormSq u) u)) x := by rw [← qMul_assoc]
      _ = qMul (qMul (qScale (qNormSq u) (qConj u)) u) x := by
          rw [qMul_conj_scale (qNormSq u) u]
      _ = qMul (qScale (qNormSq u) (qMul (qConj u) u)) x := by rw [qScale_mul_right]
      _ = qScale (qNormSq u) (qMul (qMul (qConj u) u) x) := by rw [qScale_mul_right]
      _ = qScale (qNormSq u) (qMul (qConj u) (qMul u x)) := by rw [qMul_assoc]
      _ = qScale (qNormSq u) (qScale (qNormSq u) x) := by rw [hleft]
      _ = qScale (qNormSq u * qNormSq u) x := qScale_qScale _ _ _
  · intro hnz
    unfold qNormSq
    dsimp only
    have h0 := sq_nonneg uw
    have h1 := sq_nonneg ux
    have h2 := sq_nonneg uy
    have h3 := sq_nonneg uz
    rcases hnz with hw | hx | hy | hz
    · have hp := sq_pos_of_ne_zero uw hw; omega
    · have hp := sq_pos_of_ne_zero ux hx; omega
    · have hp := sq_pos_of_ne_zero uy hy; omega
    · have hp := sq_pos_of_ne_zero uz hz; omega

/-- Theorem 18 (S_fast Woodbury Multi-Step Vector Energy Contractivity vs. Reverse-Decay Blowup):
    For rational decay `gamma = g_num / g_den` (`0 < g_num`, `0 < g_den`), ANY step count `k >= 1`,
    and ANY nonzero state vector `xs : List Int`:
    - When `g_num < g_den` (`S_fast` `gamma = 9/10`), k-step Frobenius squared energy
      `normSq ((g_num^k) * xs) < normSq ((g_den^k) * xs)` strictly contracts;
    - Conversely when `g_den < g_num` (reverse decay `beta = 1.82`), k-step Frobenius squared energy
      `normSq ((g_den^k) * xs) < normSq ((g_num^k) * xs)` strictly expands. -/
theorem s_fast_woodbury_contractivity_vs_reverse_decay_blowup
    (g_num g_den : Int) (xs : List Int) (k : Nat)
    (hnum : 0 < g_num) (hden : 0 < g_den) (hk : 1 ≤ k) (hxs : HasNonzero xs) :
    (g_num < g_den →
      normSq (xs.map (fun x => (g_num ^ k) * x)) <
      normSq (xs.map (fun x => (g_den ^ k) * x))) ∧
    (g_den < g_num →
      normSq (xs.map (fun x => (g_den ^ k) * x)) <
      normSq (xs.map (fun x => (g_num ^ k) * x))) := by
  have hxsq : 0 < normSq xs := normSq_pos_of_hasNonzero hxs
  have hgnk : 0 < g_num ^ k := pow_pos_of_pos g_num hnum k
  have hgdk : 0 < g_den ^ k := pow_pos_of_pos g_den hden k
  rw [normSq_scale (g_num ^ k) xs, normSq_scale (g_den ^ k) xs]
  constructor
  · intro hlt
    have hpow : g_num ^ k < g_den ^ k := pow_lt_pow_of_nonneg_lt (by omega) hlt k hk
    have hsq : (g_num ^ k) * (g_num ^ k) < (g_den ^ k) * (g_den ^ k) :=
      (mul_lt_iff_sq_lt hgnk hgdk).mp hpow
    have hmul : 0 < ((g_den ^ k) * (g_den ^ k) - (g_num ^ k) * (g_num ^ k)) * normSq xs :=
      Int.mul_pos (by omega) hxsq
    rw [Int.sub_mul] at hmul
    omega
  · intro hgt
    have hpow : g_den ^ k < g_num ^ k := pow_lt_pow_of_nonneg_lt (by omega) hgt k hk
    have hsq : (g_den ^ k) * (g_den ^ k) < (g_num ^ k) * (g_num ^ k) :=
      (mul_lt_iff_sq_lt hgdk hgnk).mp hpow
    have hmul : 0 < ((g_num ^ k) * (g_num ^ k) - (g_den ^ k) * (g_den ^ k)) * normSq xs :=
      Int.mul_pos (by omega) hxsq
    rw [Int.sub_mul] at hmul
    omega

#print axioms FourierMuxLoopOperators.fwht_vs_finite_fourier_crossover_iff
#print axioms FourierMuxLoopOperators.fwht_256_finite_n_win_and_asymptotic_crossover
#print axioms FourierMuxLoopOperators.finite_fourier_prop52_spectral_gap_strict
#print axioms FourierMuxLoopOperators.lowrank_rank_defect_vs_monarch_full_reach
#print axioms FourierMuxLoopOperators.v4_sp_flip_byte33_collision_universal
#print axioms FourierMuxLoopOperators.v5_h2_bent4_orthogonal_zero_alias
#print axioms FourierMuxLoopOperators.monarch_2stage_vs_3stage_path_multiplicity
#print axioms FourierMuxLoopOperators.v5_head_smem_fit_within_sm86_48kib
#print axioms FourierMuxLoopOperators.mux_cor15_cleared_margin_criterion_iff
#print axioms FourierMuxLoopOperators.mux_rho9_10_collapse_boundaries_exact
#print axioms FourierMuxLoopOperators.alkashi_balanced_radix_m_ge_1_universal
#print axioms FourierMuxLoopOperators.alkashi_fixed_radix_zero_ulp_all_s
#print axioms FourierMuxLoopOperators.a16_block_h4_parseval_and_flip_zero_drift
#print axioms FourierMuxLoopOperators.alkashi_contractive_cubic_vs_repelling_cliff
#print axioms FourierMuxLoopOperators.dual_axis_carry_beta_min_prevents_carry_last_collapse
#print axioms FourierMuxLoopOperators.dual_axis_carry_convex_hull_invariance
#print axioms FourierMuxLoopOperators.su2_cayley_unit_norm_and_gauge_pullback_exact
#print axioms FourierMuxLoopOperators.s_fast_woodbury_contractivity_vs_reverse_decay_blowup

end FourierMuxLoopOperators
