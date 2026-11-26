-- Project HoloSphere (PR #2 Candidate / Track S117 Convergence Sprint)
-- Pure-Init Lean 4.36.0 Formalization of Theorems #41-#45 (Taking Library 41 -> 46/46)
-- Zero external imports, zero Mathlib, zero sorry, zero Classical.choice,
-- zero Lean.ofReduceBool (no native_decide), zero compiler warnings.
--
-- Public Theorems proved:
--   #41 p41_woodbury_zero_inv_chunked_adjoint               (W10-OBS-60)
--   #42 p42_relational_macro_holonomy_left_gauge_invariance (OBS-ORDA-26)
--   #43 p43_a16_hamming16_11_4_inline_secded                (OBS-ORDA-10)
--   #44 p44_dyadic_23bit_gumbel_strict_unit_interval        (OBS-ORDA-17 / W10-OBS-56)
--   #45 p45_su2_nonabelian_3line_adjoint_identity           (OBS-ORDA-18 / OBS-ORDA-19)

namespace HoloSphereConvergenceTheorems

-- ============================================================================
-- 0. Constructive Finite Sums, Matrix/Vector Algebra, and Quaternion Ring Q4
-- ============================================================================

/-- Finite sum f(0) + f(1) + ... + f(n - 1) over Int. -/
def sumN : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumN n f + f n

def dotN (n : Nat) (u v : Nat → Int) : Int :=
  sumN n (fun i => u i * v i)

/-- Matrix-transpose-vector product (S^T v)_c = sum_{r=0}^{n-1} S_{r,c} * v_r. -/
def matTransVec (n : Nat) (S : Nat → Nat → Int) (v : Nat → Int) (c : Nat) : Int :=
  sumN n (fun r => S r c * v r)

private theorem sumN_congr (n : Nat) (f g : Nat → Int)
    (hfg : ∀ i, i < n → f i = g i) : sumN n f = sumN n g := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumN]
    rw [ih (fun i hi => hfg i (by omega)), hfg m (by omega)]

private theorem sumN_add (n : Nat) (f g : Nat → Int) :
    sumN n (fun i => f i + g i) = sumN n f + sumN n g := by
  induction n with
  | zero => rfl
  | succ m ih =>
    simp only [sumN, ih]
    omega

private theorem sumN_mul_left (n : Nat) (c : Int) (f : Nat → Int) :
    sumN n (fun i => c * f i) = c * sumN n f := by
  induction n with
  | zero =>
    simp only [sumN, Int.mul_zero]
  | succ m ih =>
    simp only [sumN, ih, Int.mul_add]

private theorem sumN_mul_right (n : Nat) (c : Int) (f : Nat → Int) :
    sumN n (fun i => f i * c) = sumN n f * c := by
  induction n with
  | zero =>
    simp only [sumN, Int.zero_mul]
  | succ m ih =>
    simp only [sumN, ih, Int.add_mul]

/-- Hamilton quaternion ring Q4 over Int matching HoloSphereInvariants.QuatZ. -/
structure QuatZ where
  w : Int
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

def qone : QuatZ := ⟨1, 0, 0, 0⟩
def qi : QuatZ := ⟨0, 1, 0, 0⟩
def qj : QuatZ := ⟨0, 0, 1, 0⟩

def qRe (p : QuatZ) : Int := p.w

def qadd (p q : QuatZ) : QuatZ :=
  ⟨p.w + q.w, p.x + q.x, p.y + q.y, p.z + q.z⟩

def qscale (s : Int) (p : QuatZ) : QuatZ :=
  ⟨s * p.w, s * p.x, s * p.y, s * p.z⟩

def qconj (p : QuatZ) : QuatZ :=
  ⟨p.w, -p.x, -p.y, -p.z⟩

def qmul (p q : QuatZ) : QuatZ :=
  ⟨p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
   p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
   p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
   p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w⟩

def inner4 (a b : QuatZ) : Int :=
  a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z

def normSq (p : QuatZ) : Int :=
  inner4 p p

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

private theorem qconj_qmul (p q : QuatZ) :
    qconj (qmul p q) = qmul (qconj q) (qconj p) := by
  apply quat_ext <;>
    simp only [qmul, qconj, Int.sub_eq_add_neg, Int.neg_add,
               Int.mul_neg, Int.neg_neg, Int.mul_comm] <;>
    omega

private theorem qconj_qadd (p q : QuatZ) :
    qconj (qadd p q) = qadd (qconj p) (qconj q) := by
  apply quat_ext <;> simp only [qconj, qadd] <;> omega

private theorem qconj_involutive (p : QuatZ) : qconj (qconj p) = p := by
  apply quat_ext <;> simp only [qconj, Int.neg_neg]

private theorem qmul_qadd_left (p q r : QuatZ) :
    qmul (qadd p q) r = qadd (qmul p r) (qmul q r) := by
  apply quat_ext <;> simp only [qmul, qadd, Int.add_mul] <;> omega

private theorem qmul_conj_self (p : QuatZ) :
    qmul p (qconj p) = ⟨normSq p, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, inner4, Int.mul_neg, Int.mul_comm] <;>
    omega

private theorem qconj_mul_self (p : QuatZ) :
    qmul (qconj p) p = ⟨normSq p, 0, 0, 0⟩ := by
  apply quat_ext <;>
    simp only [qmul, qconj, normSq, inner4, Int.sub_eq_add_neg,
               Int.mul_neg, Int.neg_neg, Int.mul_comm] <;>
    omega

private theorem qmul_scalar_left (s : Int) (v : QuatZ) :
    qmul ⟨s, 0, 0, 0⟩ v = qscale s v := by
  apply quat_ext <;> simp only [qmul, qscale, Int.zero_mul] <;> omega

private theorem qmul_scalar_right (v : QuatZ) (s : Int) :
    qmul v ⟨s, 0, 0, 0⟩ = qscale s v := by
  apply quat_ext <;> simp only [qmul, qscale, Int.mul_zero, Int.mul_comm] <;> omega

private theorem qscale_qmul_right (s : Int) (p q : QuatZ) :
    qmul p (qscale s q) = qscale s (qmul p q) := by
  apply quat_ext <;>
    simp only [qmul, qscale, Int.sub_eq_add_neg, Int.mul_add, Int.mul_neg, Int.mul_left_comm]

private theorem qscale_one (p : QuatZ) : qscale 1 p = p := by
  apply quat_ext <;> simp only [qscale, Int.one_mul]

private theorem inner4_symm (a b : QuatZ) : inner4 a b = inner4 b a := by
  simp only [inner4, Int.mul_comm]

private theorem inner4_qadd_left (a b c : QuatZ) :
    inner4 (qadd a b) c = inner4 a c + inner4 b c := by
  simp only [inner4, qadd, Int.add_mul]
  omega

private theorem inner4_qadd_right (a b c : QuatZ) :
    inner4 a (qadd b c) = inner4 a b + inner4 a c := by
  simp only [inner4, qadd, Int.mul_add]
  omega

private theorem inner4_qconj_both (a b : QuatZ) :
    inner4 (qconj a) (qconj b) = inner4 a b := by
  simp only [inner4, qconj, Int.neg_mul_neg]

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
-- Theorem #41: Exact Zero-gamma^-1 Woodbury Chunked Adjoint (W10-OBS-60)
-- ============================================================================

/-- One-step fast-weight delta-rule update S_{t+1} = gamma * S_t + eta * e_t * x_tilde_t^T. -/
def sfastStep (gamma eta : Int) (S : Nat → Nat → Int) (ek xk : Nat → Int) :
    Nat → Nat → Int :=
  fun r c => gamma * S r c + eta * ek r * xk c

/-- Multi-step fast-weight state S_t starting from chunk checkpoint S_0 (zero division). -/
def sfastSeq (gamma eta : Int) (S0 : Nat → Nat → Int) (e x_tilde : Nat → Nat → Int) :
    Nat → (Nat → Nat → Int)
  | 0 => S0
  | t + 1 => sfastStep gamma eta (sfastSeq gamma eta S0 e x_tilde t) (e t) (x_tilde t)

/-- Closed-form Woodbury transpose action without gamma^-1:
    (S_t^T v)_c = gamma^t (S_0^T v)_c + sum_{k=0}^{t-1} gamma^{t-1-k} eta <e_k, v> x_tilde_{k,c}. -/
def woodburyTransClosedForm (n : Nat) (gamma eta : Int) (S0 : Nat → Nat → Int)
    (e x_tilde : Nat → Nat → Int) (v : Nat → Int) (t : Nat) (c : Nat) : Int :=
  (gamma ^ t) * matTransVec n S0 v c +
    sumN t (fun k => (gamma ^ (t - 1 - k)) * eta * dotN n (e k) v * x_tilde k c)

/-- Theorem #41 (W10-OBS-60 — Exact Zero-gamma^-1 Woodbury Chunked Adjoint):
    For every dimension n, step count t, ring scalars gamma and eta (including gamma = 0
    with zero division by gamma!), checkpoint S0, rank-1 update streams (e_k, x_tilde_k),
    query vector v, and column c:
      1. Exact closed-form transpose identity:
         (S_t^T v)_c = gamma^t (S_0^T v)_c + sum_{k=0}^{t-1} gamma^{t-1-k} eta <e_k, v> x_tilde_{k,c},
      2. Exact gamma = 0 boundary safety (where naive gamma^-1 backward divides by zero):
         at gamma = 0 and step t + 1, (S_{t+1}^T v)_c = eta * <e_t, v> * x_tilde_{t,c}. -/
theorem p41_woodbury_zero_inv_chunked_adjoint
    (n t : Nat) (gamma eta : Int) (S0 : Nat → Nat → Int)
    (e x_tilde : Nat → Nat → Int) (v : Nat → Int) (c : Nat) :
    matTransVec n (sfastSeq gamma eta S0 e x_tilde t) v c =
      woodburyTransClosedForm n gamma eta S0 e x_tilde v t c ∧
    matTransVec n (sfastSeq 0 eta S0 e x_tilde (t + 1)) v c =
      eta * dotN n (e t) v * x_tilde t c := by
  have hmain : ∀ m : Nat,
      matTransVec n (sfastSeq gamma eta S0 e x_tilde m) v c =
        woodburyTransClosedForm n gamma eta S0 e x_tilde v m c := by
    intro m
    induction m with
    | zero =>
      simp only [sfastSeq, woodburyTransClosedForm, Int.pow_zero, Int.one_mul, sumN]
      omega
    | succ s ih =>
      unfold sfastSeq matTransVec sfastStep
      have hsplit : ∀ r, r < n →
          (gamma * sfastSeq gamma eta S0 e x_tilde s r c + eta * e s r * x_tilde s c) * v r =
            gamma * (sfastSeq gamma eta S0 e x_tilde s r c * v r) +
              eta * ((e s r * v r) * x_tilde s c) := by
        intro r _
        simp only [Int.add_mul, Int.mul_assoc]
        rw [Int.mul_comm (x_tilde s c) (v r)]
      rw [sumN_congr n _ _ hsplit, sumN_add, sumN_mul_left, sumN_mul_left, sumN_mul_right]
      change gamma * matTransVec n (sfastSeq gamma eta S0 e x_tilde s) v c +
          eta * (dotN n (e s) v * x_tilde s c) =
        woodburyTransClosedForm n gamma eta S0 e x_tilde v (s + 1) c
      rw [ih]
      unfold woodburyTransClosedForm
      rw [Int.mul_add, ← Int.mul_assoc gamma (gamma ^ s),
          Int.mul_comm gamma (gamma ^ s), ← Int.pow_succ,
          ← sumN_mul_left s gamma]
      have hsum_step : ∀ k, k < s →
          gamma * ((gamma ^ (s - 1 - k)) * eta * dotN n (e k) v * x_tilde k c) =
            (gamma ^ (s - k)) * eta * dotN n (e k) v * x_tilde k c := by
        intro k hk
        have hexp : s - k = (s - 1 - k) + 1 := by omega
        rw [hexp, Int.pow_succ, Int.mul_comm (gamma ^ (s - 1 - k)) gamma]
        simp only [Int.mul_assoc]
      have hrhs_congr : ∀ k, k < s →
          (gamma ^ (s + 1 - 1 - k)) * eta * dotN n (e k) v * x_tilde k c =
            (gamma ^ (s - k)) * eta * dotN n (e k) v * x_tilde k c := by
        intro k _
        have hsub : s + 1 - 1 - k = s - k := by omega
        rw [hsub]
      simp only [sumN]
      rw [sumN_congr s _ _ hsum_step, sumN_congr s _ _ hrhs_congr]
      have hlast : s + 1 - 1 - s = 0 := by omega
      rw [hlast, Int.pow_zero, Int.one_mul]
      simp only [Int.mul_assoc, Int.add_assoc]
  have hzero :
      matTransVec n (sfastSeq 0 eta S0 e x_tilde (t + 1)) v c =
        eta * dotN n (e t) v * x_tilde t c := by
    unfold sfastSeq matTransVec sfastStep
    have h0 : ∀ r, r < n →
        (0 * sfastSeq 0 eta S0 e x_tilde t r c + eta * e t r * x_tilde t c) * v r =
          eta * ((e t r * v r) * x_tilde t c) := by
      intro r _
      simp only [Int.zero_mul, Int.zero_add, Int.mul_assoc]
      rw [Int.mul_comm (x_tilde t c) (v r)]
    rw [sumN_congr n _ _ h0, sumN_mul_left, sumN_mul_right]
    unfold dotN
    simp only [Int.mul_assoc]
  exact ⟨hmain t, hzero⟩

-- ============================================================================
-- Theorem #42: Relational Macro Holonomy Left-Gauge Invariance (OBS-ORDA-26)
-- ============================================================================

/-- Relational macro holonomy DeltaU = U_1 * conj(U_0) across subroutine [tau_0, tau_1]. -/
def relMacro (U1 U0 : QuatZ) : QuatZ :=
  qmul U1 (qconj U0)

/-- Theorem #42 (OBS-ORDA-26 — Relational Macro Holonomy Exact Gauge Invariance):
    Over the Hamilton quaternion ring Q4:
      1. Homogeneous right-gauge scaling: for ALL U1, U0, G,
         (U1 * G) * conj(U0 * G) = normSq(G) * (U1 * conj(U0)),
      2. Exact unit-gauge invariance: when normSq(G) = 1 and U_t' = U_t * G,
         DeltaU' = U1' * conj(U0') = U1 * conj(U0) = DeltaU identically,
      3. Exact endpoint recovery & arbitrary-caller transport: when normSq(U0) = 1,
         DeltaU * U0 = U1 AND DeltaU * (U0 * H) = U1 * H for any caller context H,
      4. Universal failure of raw endpoint storage: for EVERY unit U1 (normSq U1 = 1)
         and EVERY non-identity context G != 1, U1 * G != U1. -/
theorem p42_relational_macro_holonomy_left_gauge_invariance
    (U1 U0 G H : QuatZ) :
    relMacro (qmul U1 G) (qmul U0 G) = qscale (normSq G) (relMacro U1 U0) ∧
    (normSq G = 1 → relMacro (qmul U1 G) (qmul U0 G) = relMacro U1 U0) ∧
    (normSq U0 = 1 →
      qmul (relMacro U1 U0) U0 = U1 ∧
      qmul (relMacro U1 U0) (qmul U0 H) = qmul U1 H) ∧
    (normSq U1 = 1 → G ≠ qone → qmul U1 G ≠ U1) := by
  have hhom : relMacro (qmul U1 G) (qmul U0 G) = qscale (normSq G) (relMacro U1 U0) := by
    unfold relMacro
    rw [qconj_qmul, qmul_assoc U1 G (qmul (qconj G) (qconj U0)),
        ← qmul_assoc G (qconj G) (qconj U0), qmul_conj_self,
        qmul_scalar_left, qscale_qmul_right]
  have hg1 : normSq G = 1 → relMacro (qmul U1 G) (qmul U0 G) = relMacro U1 U0 := by
    intro hG
    rw [hhom, hG, qscale_one]
  have hrec : normSq U0 = 1 →
      qmul (relMacro U1 U0) U0 = U1 ∧
      qmul (relMacro U1 U0) (qmul U0 H) = qmul U1 H := by
    intro hU0
    have hstep1 : qmul (relMacro U1 U0) U0 = U1 := by
      unfold relMacro
      rw [qmul_assoc, qconj_mul_self, hU0, qmul_scalar_right, qscale_one]
    have hstep2 : qmul (relMacro U1 U0) (qmul U0 H) = qmul U1 H := by
      rw [← qmul_assoc, hstep1]
    exact ⟨hstep1, hstep2⟩
  have hraw_fail : normSq U1 = 1 → G ≠ qone → qmul U1 G ≠ U1 := by
    intro hU1 hGne heq
    have hleft := congrArg (qmul (qconj U1)) heq
    rw [← qmul_assoc, qconj_mul_self, hU1] at hleft
    rw [qmul_scalar_left, qscale_one] at hleft
    exact hGne hleft
  exact ⟨hhom, hg1, hrec, hraw_fail⟩

-- ============================================================================
-- Theorem #43: [16, 11, 4] Extended Hamming Inline LSB SECDED + 2-Bit DED
-- ============================================================================

/-- Recursive k-bit XOR parity over BitVec 16. -/
def parityPrefix : Nat → BitVec 16 → Bool
  | 0, _ => false
  | k + 1, w => parityPrefix k w ^^ w.getLsbD k

/-- Full 16-bit XOR parity (true iff odd number of set bits in w). -/
def parity16 (w : BitVec 16) : Bool :=
  parityPrefix 16 w

def bit01 (b : Bool) : BitVec 5 :=
  if b then 1#5 else 0#5

/-- Exact 5-bit Extended Hamming [16, 11, 4] syndrome matching CUDA a16_parity_check_16
    and s116_a16_hamming_ste.py:103-111:
      bit 0 (p0): overall parity over 0xFFFF
      bit 1 (s0): parity over 0xAAAA
      bit 2 (s1): parity over 0xCCCC
      bit 3 (s2): parity over 0xF0F0
      bit 4 (s3): parity over 0xFF00 -/
def a16Syndrome5 (m : BitVec 16) : BitVec 5 :=
  bit01 (parity16 (m &&& 0xFFFF#16)) ^^^
  (bit01 (parity16 (m &&& 0xAAAA#16)) <<< 1) ^^^
  (bit01 (parity16 (m &&& 0xCCCC#16)) <<< 2) ^^^
  (bit01 (parity16 (m &&& 0xF0F0#16)) <<< 3) ^^^
  (bit01 (parity16 (m &&& 0xFF00#16)) <<< 4)

/-- Identical OR-packed syndrome form matching Python p0 | (s0 << 1) | ... | (s3 << 4). -/
def a16Syndrome5Or (m : BitVec 16) : BitVec 5 :=
  bit01 (parity16 (m &&& 0xFFFF#16)) |||
  (bit01 (parity16 (m &&& 0xAAAA#16)) <<< 1) |||
  (bit01 (parity16 (m &&& 0xCCCC#16)) <<< 2) |||
  (bit01 (parity16 (m &&& 0xF0F0#16)) <<< 3) |||
  (bit01 (parity16 (m &&& 0xFF00#16)) <<< 4)

/-- Overall parity bit p = s &&& 1 (1 for single-bit error, 0 for double-bit error). -/
def a16OverallParity (m : BitVec 16) : BitVec 5 :=
  a16Syndrome5 m &&& 1#5

/-- 4-bit position syndrome (s >>> 1) &&& 0xF recovering flipped lane j in 0..15. -/
def a16PosLane (m : BitVec 16) : Nat :=
  ((a16Syndrome5 m >>> 1) &&& 0x0F#5).toNat

/-- Single-bit error mask 1 <<< j on BitVec 16. -/
def errBit16 (j : Fin 16) : BitVec 16 :=
  1#16 <<< j.val

private theorem bool_xor_4_interchange (p q x y : Bool) :
    ((p ^^ q) ^^ (x ^^ y)) = ((p ^^ x) ^^ (q ^^ y)) := by
  cases p <;> cases q <;> cases x <;> cases y <;> rfl

private theorem bool_and_xor_distrib (x y m : Bool) :
    ((x ^^ y) && m) = ((x && m) ^^ (y && m)) := by
  cases x <;> cases y <;> cases m <;> rfl

private theorem parityPrefix_xor (k : Nat) (a b : BitVec 16) :
    parityPrefix k (a ^^^ b) = (parityPrefix k a ^^ parityPrefix k b) := by
  induction k with
  | zero => rfl
  | succ s ih =>
    simp only [parityPrefix, BitVec.getLsbD_xor, ih]
    exact bool_xor_4_interchange (parityPrefix s a) (parityPrefix s b) (a.getLsbD s) (b.getLsbD s)

private theorem parityPrefix_mask_xor (k : Nat) (a b mask : BitVec 16) :
    parityPrefix k ((a ^^^ b) &&& mask) =
      (parityPrefix k (a &&& mask) ^^ parityPrefix k (b &&& mask)) := by
  induction k with
  | zero => rfl
  | succ s ih =>
    simp only [parityPrefix, BitVec.getLsbD_and, BitVec.getLsbD_xor, ih]
    rw [bool_and_xor_distrib]
    exact bool_xor_4_interchange
      (parityPrefix s (a &&& mask)) (parityPrefix s (b &&& mask))
      (a.getLsbD s && mask.getLsbD s) (b.getLsbD s && mask.getLsbD s)

private theorem bit01_xor (a b : Bool) :
    bit01 (a ^^ b) = (bit01 a ^^^ bit01 b) := by
  cases a <;> cases b <;> rfl

private theorem bit01_shl1_xor (a b : Bool) :
    (bit01 (a ^^ b) <<< 1) = ((bit01 a <<< 1) ^^^ (bit01 b <<< 1)) := by
  cases a <;> cases b <;> rfl

private theorem bit01_shl2_xor (a b : Bool) :
    (bit01 (a ^^ b) <<< 2) = ((bit01 a <<< 2) ^^^ (bit01 b <<< 2)) := by
  cases a <;> cases b <;> rfl

private theorem bit01_shl3_xor (a b : Bool) :
    (bit01 (a ^^ b) <<< 3) = ((bit01 a <<< 3) ^^^ (bit01 b <<< 3)) := by
  cases a <;> cases b <;> rfl

private theorem bit01_shl4_xor (a b : Bool) :
    (bit01 (a ^^ b) <<< 4) = ((bit01 a <<< 4) ^^^ (bit01 b <<< 4)) := by
  cases a <;> cases b <;> rfl

private theorem bv5_xor_interchange (a0 a1 a2 a3 a4 b0 b1 b2 b3 b4 : BitVec 5) :
    (((a0 ^^^ b0) ^^^ (a1 ^^^ b1) ^^^ (a2 ^^^ b2) ^^^ (a3 ^^^ b3)) ^^^ (a4 ^^^ b4)) =
      ((a0 ^^^ a1 ^^^ a2 ^^^ a3 ^^^ a4) ^^^ (b0 ^^^ b1 ^^^ b2 ^^^ b3 ^^^ b4)) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases a0.getLsbD i <;> cases b0.getLsbD i <;>
  cases a1.getLsbD i <;> cases b1.getLsbD i <;>
  cases a2.getLsbD i <;> cases b2.getLsbD i <;>
  cases a3.getLsbD i <;> cases b3.getLsbD i <;>
  cases a4.getLsbD i <;> cases b4.getLsbD i <;> rfl

private theorem a16Syndrome5_or_eq_xor (m : BitVec 16) :
    a16Syndrome5Or m = a16Syndrome5 m := by
  unfold a16Syndrome5Or a16Syndrome5
  cases parity16 (m &&& 0xFFFF#16) <;>
  cases parity16 (m &&& 0xAAAA#16) <;>
  cases parity16 (m &&& 0xCCCC#16) <;>
  cases parity16 (m &&& 0xF0F0#16) <;>
  cases parity16 (m &&& 0xFF00#16) <;> rfl

private theorem a16Syndrome5_xor (a b : BitVec 16) :
    a16Syndrome5 (a ^^^ b) = (a16Syndrome5 a ^^^ a16Syndrome5 b) := by
  unfold a16Syndrome5 parity16
  rw [parityPrefix_mask_xor 16 a b 0xFFFF#16,
      parityPrefix_mask_xor 16 a b 0xAAAA#16,
      parityPrefix_mask_xor 16 a b 0xCCCC#16,
      parityPrefix_mask_xor 16 a b 0xF0F0#16,
      parityPrefix_mask_xor 16 a b 0xFF00#16,
      bit01_xor,
      bit01_shl1_xor,
      bit01_shl2_xor,
      bit01_shl3_xor,
      bit01_shl4_xor]
  exact bv5_xor_interchange _ _ _ _ _ _ _ _ _ _

/-- Kernel-decidable 16-lane single-error check (16 cases, zero native_decide). -/
private theorem a16_basis_sec_all16 :
    ∀ j : Fin 16,
      a16Syndrome5 (errBit16 j) ≠ 0#5 ∧
      a16OverallParity (errBit16 j) = 1#5 ∧
      a16PosLane (errBit16 j) = j.val := by
  decide

/-- Kernel-decidable 16x16 double-error check (all pairs j < k, 120 pairs, zero native_decide). -/
private theorem a16_basis_ded_all_pairs :
    ∀ j k : Fin 16, j.val < k.val →
      (a16Syndrome5 (errBit16 j) ^^^ a16Syndrome5 (errBit16 k)) ≠ 0#5 ∧
      ((a16Syndrome5 (errBit16 j) ^^^ a16Syndrome5 (errBit16 k)) &&& 1#5) = 0#5 := by
  decide

private theorem bv16_xor_cancel_right (c e : BitVec 16) :
    (c ^^^ e) ^^^ e = c := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  simp only [BitVec.getLsbD_xor]
  cases c.getLsbD i <;> cases e.getLsbD i <;> rfl

/-- Theorem #43 (OBS-ORDA-10 — Exhaustive [16, 11, 4] Extended Hamming Inline SECDED):
    For EVERY valid 16-bit codeword c (a16Syndrome5 c = 0, all 2^11 = 2048 codewords):
      1. Equivalence of OR-packed and XOR-packed hardware syndrome formulas,
      2. 100% Single-Error Correction (SEC) across all 16 bit lanes j in 0..15:
         for corrupted word w1 = c ^^^ (1 <<< j), syndrome s != 0, overall parity p = 1,
         lane recovery ((s >>> 1) &&& 0xF) == j, AND inline correction w1 ^^^ (1 <<< lane) = c,
      3. 100% Double-Error Detection (DED) across all 120 bit pairs j < k:
         for corrupted word w2 = c ^^^ (1 <<< j) ^^^ (1 <<< k), syndrome s != 0 AND
         overall parity p = 0 (zero silent miscorrection -> triggers Weft 16 B rollback!). -/
theorem p43_a16_hamming16_11_4_inline_secded
    (c : BitVec 16) (hc : a16Syndrome5 c = 0#5)
    (j k : Fin 16) :
    a16Syndrome5Or c = a16Syndrome5 c ∧
    (a16Syndrome5 (c ^^^ errBit16 j) ≠ 0#5 ∧
     a16OverallParity (c ^^^ errBit16 j) = 1#5 ∧
     a16PosLane (c ^^^ errBit16 j) = j.val ∧
     ((c ^^^ errBit16 j) ^^^ (1#16 <<< a16PosLane (c ^^^ errBit16 j))) = c) ∧
    (j.val < k.val →
      a16Syndrome5 ((c ^^^ errBit16 j) ^^^ errBit16 k) ≠ 0#5 ∧
      a16OverallParity ((c ^^^ errBit16 j) ^^^ errBit16 k) = 0#5) := by
  have hzero_xor : ∀ x : BitVec 5, (0#5 ^^^ x) = x := by
    intro x
    apply BitVec.eq_of_getLsbD_eq
    intro i _
    simp only [BitVec.getLsbD_xor]
    have h0 : (0#5).getLsbD i = false := by
      simp only [BitVec.getLsbD_zero]
    rw [h0]
    cases x.getLsbD i <;> rfl
  have h1_syn : a16Syndrome5 (c ^^^ errBit16 j) = a16Syndrome5 (errBit16 j) := by
    rw [a16Syndrome5_xor, hc, hzero_xor]
  have hsec := a16_basis_sec_all16 j
  have hp1 : a16OverallParity (c ^^^ errBit16 j) = 1#5 := by
    unfold a16OverallParity
    rw [h1_syn]
    exact hsec.2.1
  have hlane : a16PosLane (c ^^^ errBit16 j) = j.val := by
    unfold a16PosLane
    rw [h1_syn]
    exact hsec.2.2
  have hrecover : ((c ^^^ errBit16 j) ^^^ (1#16 <<< a16PosLane (c ^^^ errBit16 j))) = c := by
    rw [hlane]
    exact bv16_xor_cancel_right c (errBit16 j)
  have hded : j.val < k.val →
      a16Syndrome5 ((c ^^^ errBit16 j) ^^^ errBit16 k) ≠ 0#5 ∧
      a16OverallParity ((c ^^^ errBit16 j) ^^^ errBit16 k) = 0#5 := by
    intro hjk
    have h2_syn :
        a16Syndrome5 ((c ^^^ errBit16 j) ^^^ errBit16 k) =
          (a16Syndrome5 (errBit16 j) ^^^ a16Syndrome5 (errBit16 k)) := by
      rw [a16Syndrome5_xor, h1_syn]
    have hpair := a16_basis_ded_all_pairs j k hjk
    refine ⟨?_, ?_⟩
    · rw [h2_syn]
      exact hpair.1
    · unfold a16OverallParity
      rw [h2_syn]
      exact hpair.2
  exact ⟨a16Syndrome5_or_eq_xor c,
         ⟨by rw [h1_syn]; exact hsec.1, hp1, hlane, hrecover⟩,
         hded⟩

-- ============================================================================
-- Theorem #44: Exact 23-Bit Dyadic Strictly-Inside-(0, 1) Gumbel Safety
-- ============================================================================

/-- Top-23-bit mantissa index m = floor(u / 2^9) in [0, 2^23 - 1]. -/
def gumbelMantissa23 (u : Nat) : Nat :=
  u / 512

/-- Odd 24-bit dyadic numerator 2*m + 1 in [1, 2^24 - 1] over exact denominator 2^24. -/
def gumbelDyadicNum24 (u : Nat) : Nat :=
  2 * gumbelMantissa23 u + 1

/-- Theorem #44 (OBS-ORDA-17 / W10-OBS-56 — 23-Bit Dyadic Gumbel Strict Unit-Interval Safety):
    For EVERY u in [0, 2^32 - 1] (and every u32 : UInt32):
      1. Exact top-23-bit range: gumbelMantissa23(u) <= 2^23 - 1 = 8388607,
      2. Strict interior 24-bit odd numerator bounds:
         1 <= 2*m + 1 <= 2^24 - 1 = 16777215 (so (2*m + 1) / 2^24 lies strictly inside
         [2^-24, 1 - 2^-24] with an exact 24-bit binary32 significand and zero rounding to 0 or 1!),
      3. Exact endpoint attainment: u = 0 -> 1, u = 2^32 - 1 -> 2^24 - 1,
      4. Exact naive 32-bit rounding-to-1.0f witness & sharp threshold:
         for u_bad = 2^32 - 128 (0xFFFFFF80) — and EVERY u in [0xFFFFFF80, 0xFFFFFFFF] —
         naive (u + 1/2)/2^32 = (2*u + 1)/2^33 satisfies 2^33 - (2*u + 1) < 2^8
         (i.e. 1 - (u + 1/2)/2^32 < 2^-25, strictly inside the half-ULP rounding-to-1.0f zone!),
         whereas u = 2^32 - 129 (0xFFFFFF7F) satisfies 2^33 - (2*u + 1) = 257 > 2^8. -/
theorem p44_dyadic_23bit_gumbel_strict_unit_interval
    (u : Nat) (hu : u < 2 ^ 32) (u32 : UInt32) :
    gumbelMantissa23 u ≤ 2 ^ 23 - 1 ∧
    (1 ≤ gumbelDyadicNum24 u ∧ gumbelDyadicNum24 u ≤ 2 ^ 24 - 1) ∧
    (1 ≤ gumbelDyadicNum24 u32.toNat ∧ gumbelDyadicNum24 u32.toNat ≤ 2 ^ 24 - 1) ∧
    gumbelDyadicNum24 0 = 1 ∧
    gumbelDyadicNum24 (2 ^ 32 - 1) = 2 ^ 24 - 1 ∧
    (2 ^ 33 - (2 * (2 ^ 32 - 128) + 1) = 255 ∧ 255 < 2 ^ 8) ∧
    (∀ u_hi : Nat, 2 ^ 32 - 128 ≤ u_hi → u_hi < 2 ^ 32 →
      2 ^ 33 - (2 * u_hi + 1) < 2 ^ 8) ∧
    2 ^ 33 - (2 * (2 ^ 32 - 129) + 1) = 257 := by
  have hu32 : u32.toNat < 2 ^ 32 := u32.toNat_lt
  unfold gumbelDyadicNum24 gumbelMantissa23
  refine ⟨by omega, ⟨by omega, by omega⟩, ⟨by omega, by omega⟩,
          by decide, by decide, ⟨by decide, by decide⟩, ?_, by decide⟩
  intro u_hi hlo hhi
  omega

-- ============================================================================
-- Theorem #45: Exact Non-Abelian SU(2) 3-Line Adjoint Identity (OBS-ORDA-18/19)
-- ============================================================================

/-- Theorem #45 (OBS-ORDA-18 / OBS-ORDA-19 — Non-Abelian SU(2) 3-Line Adjoint Identity):
    Over Hamilton quaternion ring Q4 with inner product <a, b> = Re(a * conj(b)):
      1. Exact inner-product real-part representation: inner4 a b = qRe (qmul a (qconj b)),
      2. Exact Track S117 3-line covariant un-transport adjoint (xt = conj(U_t) * x_t,
         U_t = q_t * U_{t-1}, dU_t = dq_t * U_{t-1} + q_t * dU_{t-1},
         dxt = conj(dU_t) * x_t + conj(U_t) * dx_t):
         with gU_cur = gU_next + x_t * conj(gxt), gq_t = gU_cur * conj(U_{t-1}),
         gU_prev = conj(q_t) * gU_cur, and gx_in = U_t * gxt,
         <gxt, dxt> + <gU_next, dU_t> = <gq_t, dq_t> + <gU_prev, dU_{t-1}> + <gx_in, dx_t>,
      3. Exact forward-rotation dual (x_rot = U_t * x_t, dx_rot = dU_t * x_t + U_t * dx_t):
         with gU_cur_f = gU_next + gxt * conj(x_t), gq_t_f = gU_cur_f * conj(U_{t-1}),
         gU_prev_f = conj(q_t) * gU_cur_f, and gx_in_f = conj(U_t) * gxt,
         <gxt, dx_rot> + <gU_next, dU_t> = <gq_t_f, dq_t> + <gU_prev_f, dU_{t-1}> + <gx_in_f, dx_t>,
      4. Non-Abelian order-reversal refutation (OBS-ORDA-18):
         conj(i) * (conj(j) * 1) != conj(i * j) * 1. -/
theorem p45_su2_nonabelian_3line_adjoint_identity
    (a b q_t U_prev x_t gxt gU_next dq_t dU_prev dx_t : QuatZ) :
    let U_t := qmul q_t U_prev
    let dU_t := qadd (qmul dq_t U_prev) (qmul q_t dU_prev)
    let dxt := qadd (qmul (qconj dU_t) x_t) (qmul (qconj U_t) dx_t)
    let gU_cur := qadd gU_next (qmul x_t (qconj gxt))
    let gq_t := qmul gU_cur (qconj U_prev)
    let gU_prev := qmul (qconj q_t) gU_cur
    let gx_in := qmul U_t gxt
    let dx_rot := qadd (qmul dU_t x_t) (qmul U_t dx_t)
    let gU_cur_f := qadd gU_next (qmul gxt (qconj x_t))
    let gq_t_f := qmul gU_cur_f (qconj U_prev)
    let gU_prev_f := qmul (qconj q_t) gU_cur_f
    let gx_in_f := qmul (qconj U_t) gxt
    inner4 a b = qRe (qmul a (qconj b)) ∧
    inner4 gxt dxt + inner4 gU_next dU_t =
      inner4 gq_t dq_t + inner4 gU_prev dU_prev + inner4 gx_in dx_t ∧
    inner4 gxt dx_rot + inner4 gU_next dU_t =
      inner4 gq_t_f dq_t + inner4 gU_prev_f dU_prev + inner4 gx_in_f dx_t ∧
    qmul (qconj qi) (qmul (qconj qj) qone) ≠ qmul (qconj (qmul qi qj)) qone := by
  have hre : inner4 a b = qRe (qmul a (qconj b)) := by
    simp only [inner4, qRe, qmul, qconj, Int.sub_eq_add_neg, Int.mul_neg, Int.neg_neg]
  have hstep_split : ∀ gU : QuatZ,
      inner4 gU (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) =
        inner4 (qmul gU (qconj U_prev)) dq_t + inner4 (qmul (qconj q_t) gU) dU_prev := by
    intro gU
    rw [inner4_qadd_right, quat_right_adj gU dq_t U_prev, quat_left_adj gU q_t dU_prev]
  have hcov :
      inner4 gxt (qadd (qmul (qconj (qadd (qmul dq_t U_prev) (qmul q_t dU_prev))) x_t)
                       (qmul (qconj (qmul q_t U_prev)) dx_t)) +
        inner4 gU_next (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) =
      inner4 (qmul (qadd gU_next (qmul x_t (qconj gxt))) (qconj U_prev)) dq_t +
        inner4 (qmul (qconj q_t) (qadd gU_next (qmul x_t (qconj gxt)))) dU_prev +
        inner4 (qmul (qmul q_t U_prev) gxt) dx_t := by
    rw [inner4_qadd_right gxt]
    have hdx :
        inner4 gxt (qmul (qconj (qmul q_t U_prev)) dx_t) =
          inner4 (qmul (qmul q_t U_prev) gxt) dx_t := by
      rw [quat_left_adj, qconj_involutive]
    have hdU :
        inner4 gxt (qmul (qconj (qadd (qmul dq_t U_prev) (qmul q_t dU_prev))) x_t) =
          inner4 (qmul x_t (qconj gxt)) (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) := by
      rw [← inner4_qconj_both gxt, qconj_qmul, qconj_involutive, quat_left_adj, qconj_involutive]
    rw [hdx, hdU]
    have hmerge :
        inner4 (qmul x_t (qconj gxt)) (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) +
          inner4 (qmul (qmul q_t U_prev) gxt) dx_t +
          inner4 gU_next (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) =
        inner4 (qadd gU_next (qmul x_t (qconj gxt)))
               (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) +
          inner4 (qmul (qmul q_t U_prev) gxt) dx_t := by
      rw [inner4_qadd_left]
      omega
    rw [hmerge, hstep_split (qadd gU_next (qmul x_t (qconj gxt)))]
  have hfwd :
      inner4 gxt (qadd (qmul (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) x_t)
                       (qmul (qmul q_t U_prev) dx_t)) +
        inner4 gU_next (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) =
      inner4 (qmul (qadd gU_next (qmul gxt (qconj x_t))) (qconj U_prev)) dq_t +
        inner4 (qmul (qconj q_t) (qadd gU_next (qmul gxt (qconj x_t)))) dU_prev +
        inner4 (qmul (qconj (qmul q_t U_prev)) gxt) dx_t := by
    rw [inner4_qadd_right gxt,
        quat_right_adj gxt (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) x_t,
        quat_left_adj gxt (qmul q_t U_prev) dx_t]
    have hmerge :
        inner4 (qmul gxt (qconj x_t)) (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) +
          inner4 (qmul (qconj (qmul q_t U_prev)) gxt) dx_t +
          inner4 gU_next (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) =
        inner4 (qadd gU_next (qmul gxt (qconj x_t)))
               (qadd (qmul dq_t U_prev) (qmul q_t dU_prev)) +
          inner4 (qmul (qconj (qmul q_t U_prev)) gxt) dx_t := by
      rw [inner4_qadd_left]
      omega
    rw [hmerge, hstep_split (qadd gU_next (qmul gxt (qconj x_t)))]
  exact ⟨hre, hcov, hfwd, by decide⟩

-- ============================================================================
-- Machine-Checked Axiom Audit (#print axioms)
-- ============================================================================

#print axioms HoloSphereConvergenceTheorems.p41_woodbury_zero_inv_chunked_adjoint
#print axioms HoloSphereConvergenceTheorems.p42_relational_macro_holonomy_left_gauge_invariance
#print axioms HoloSphereConvergenceTheorems.p43_a16_hamming16_11_4_inline_secded
#print axioms HoloSphereConvergenceTheorems.p44_dyadic_23bit_gumbel_strict_unit_interval
#print axioms HoloSphereConvergenceTheorems.p45_su2_nonabelian_3line_adjoint_identity

end HoloSphereConvergenceTheorems
