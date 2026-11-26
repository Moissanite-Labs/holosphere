/-
  LoopCDStrang/SphereChord.lean

  Theorem 1 & Supporting `S^{d-1}` Rational Geometry (Spherical Chord Self-Damping):
  - Stereographic rational unit-circle parameterization `stereoCos`, `stereoSin`
  - Euclidean `LoopCD-Hidden` (`h_raw = (1 + ω) h_R - ω h_1`) + `RMSNorm` angular law
  - **Theorem 1 (`thm1_loopcd_chord_saturation_gap_exact`)**:
      `tan((π - θ)/2) - tan(α) = (1 + s^2) / (s * (1 + (1 + 2ω) s^2)) > 0`
      universally for all `s = p/q > 0` and `ω = u/v ≥ 0` (`0` antipodal wraparound!)
  - Strict `ω`-monotonicity (`loopcd_tan_alpha_strict_mono_omega`)
  - Constructive SLERP antipodal-wraparound contrast witness at `θ = π/2, ω = 1`

  Pure `Init` (`Lean 4.36.0`), `0` `Classical.choice`, `0` `bv_decide`, `0` `native_decide`, `0` `sorryAx`.
-/

import LoopCDStrang.IntPoly

namespace LoopCDStrang

/-- Rational stereographic `cos θ = (q^2 - p^2) / (q^2 + p^2)` for `s = tan(θ/2) = p / q`. -/
def stereoCos (p q : Int) : QPair :=
  ⟨q * q - p * p, q * q + p * p⟩

/-- Rational stereographic `sin θ = (2 p q) / (q^2 + p^2)` for `s = tan(θ/2) = p / q`. -/
def stereoSin (p q : Int) : QPair :=
  ⟨2 * p * q, q * q + p * p⟩

/-- Rational `tan((π - θ)/2) = cot(θ/2) = 1 / s = q / p`. -/
def cotHalfTheta (p q : Int) : QPair :=
  ⟨q, p⟩

/-- Closed-form rational `LoopCD` angular tangent `tan α = 2 ω s / (1 + (1 + 2ω) s^2)`
    for `s = p / q` and `ω = u / v`. -/
def loopcdTanAlpha (p q u v : Int) : QPair :=
  ⟨2 * u * p * q, v * q * q + (v + 2 * u) * p * p⟩

/-- Closed-form rational `LoopCD` chord-normal saturation margin
    `tan((π - θ)/2) - tan α = (1 + s^2) / (s * (1 + (1 + 2ω) s^2))`. -/
def loopcdSaturationGap (p q u v : Int) : QPair :=
  ⟨v * q * (q * q + p * p), p * (v * q * q + (v + 2 * u) * p * p)⟩

/-- Trigonometric form `(ω sin θ) / (1 + ω (1 - cos θ))` of `LoopCD`'s extrapolation angle
    for `s = p / q` and `ω = u / v`. -/
def loopcdTanAlphaTrig (p q u v : Int) : QPair :=
  let c := stereoCos p q
  let sn := stereoSin p q
  let num := qmul ⟨u, v⟩ sn
  let den := qadd ⟨1, 1⟩ (qmul ⟨u, v⟩ (qsub ⟨1, 1⟩ c))
  qdiv num den

/-! ### Top-Level Reified `PExpr` Certificates (1-to-1 Identical Operator Associativity) -/

private def stereoCircleLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1
  (q * q - p * p) * (q * q - p * p) + (2 * p * q) * (2 * p * q)

private def stereoCircleRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1
  (q * q + p * p) * (q * q + p * p)

private def trigDenomLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let qp := q * q + p * p
  let qm := q * q - p * p
  v * qp * (1 * (v * (1 * qp)) + u * (1 * qp - qm * 1) * 1)

private def trigDenomRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let qp := q * q + p * p
  (v * qp) * (v * q * q + (v + 2 * u) * p * p)

private def trigEqRatLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let qp := q * q + p * p
  let tNum := (u * (2 * p * q)) * (1 * (v * (1 * qp)))
  let rDen := v * q * q + (v + 2 * u) * p * p
  tNum * rDen

private def trigEqRatRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let qp := q * q + p * p
  let qm := q * q - p * p
  let tDen := (v * qp) * (1 * (v * (1 * qp)) + u * (1 * qp - qm * 1) * 1)
  let rNum := 2 * u * p * q
  rNum * tDen

private def rawNormLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let cosN := q * q - p * p
  let sinN := 2 * p * q
  let scD  := q * q + p * p
  let rx   := (v + u) * scD - u * cosN
  let ry   := u * sinN
  rx * rx + ry * ry

private def rawNormRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let scD  := q * q + p * p
  (v * scD) * (v * scD) + 2 * u * (v + u) * (2 * p * p) * scD

private def satGapEqLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let d := v * q * q + (v + 2 * u) * p * p
  (q * d - (2 * u * p * q) * p) * (p * d)

private def satGapEqRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let d := v * q * q + (v + 2 * u) * p * p
  (v * q * (q * q + p * p)) * (p * d)

private def satGapDiffLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let u := PExpr.var 2; let v := PExpr.var 3
  let d := v * q * q + (v + 2 * u) * p * p
  q * d - (2 * u * p * q) * p

private def satGapDiffRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1; let v := PExpr.var 3
  v * q * (q * q + p * p)

private def monoDiffLHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1
  let u1 := PExpr.var 2; let v1 := PExpr.var 3
  let u2 := PExpr.var 4; let v2 := PExpr.var 5
  (2 * u2 * p * q) * (v1 * q * q + (v1 + 2 * u1) * p * p) -
  (2 * u1 * p * q) * (v2 * q * q + (v2 + 2 * u2) * p * p)

private def monoDiffRHS : PExpr :=
  let p := PExpr.var 0; let q := PExpr.var 1
  let u1 := PExpr.var 2; let v1 := PExpr.var 3
  let u2 := PExpr.var 4; let v2 := PExpr.var 5
  2 * p * q * (q * q + p * p) * (u2 * v1 - u1 * v2)

/-- Exact stereographic unit-circle identity `cos^2 θ + sin^2 θ = 1` with positive denominator. -/
theorem stereo_unit_circle_exact (p q : Int) (hp : 0 < p) (hq : 0 < q) :
    0 < q * q + p * p ∧
    (q * q - p * p) * (q * q - p * p) + (2 * p * q) * (2 * p * q) =
      (q * q + p * p) * (q * q + p * p) :=
  ⟨Int.add_pos (int_sq_pos_of_pos hq) (int_sq_pos_of_pos hp),
   poly_eq_of_norm_eq [p, q] stereoCircleLHS stereoCircleRHS rfl⟩

/-- Strict positivity of `v * q^2 + (v + 2u) * p^2` for all `p > 0, q > 0, u ≥ 0, v > 0`. -/
theorem loopcd_denom_pos (p q u v : Int) (hp : 0 < p) (hq : 0 < q) (hu : 0 ≤ u) (hv : 0 < v) :
    0 < v * q * q + (v + 2 * u) * p * p := by
  have hqq : 0 < q * q := int_sq_pos_of_pos hq
  have hpp : 0 < p * p := int_sq_pos_of_pos hp
  have hvqq : 0 < v * (q * q) := Int.mul_pos hv hqq
  have hvqq' : v * q * q = v * (q * q) := Int.mul_assoc v q q
  have h2u : 0 ≤ 2 * u := Int.mul_nonneg (by decide : (0 : Int) ≤ 2) hu
  have hv2u : 0 < v + 2 * u := Int.add_pos_of_pos_of_nonneg hv h2u
  have hterm2 : 0 < (v + 2 * u) * (p * p) := Int.mul_pos hv2u hpp
  have hterm2' : (v + 2 * u) * p * p = (v + 2 * u) * (p * p) := Int.mul_assoc (v + 2 * u) p p
  rw [hvqq', hterm2']
  exact Int.add_pos hvqq hterm2

/-- Exact equivalence between the geometric `(ω sin θ) / (1 + ω (1 - cos θ))` tangent ratio
    and the rational `s = tan(θ/2)` law `2 ω s / (1 + (1 + 2ω) s^2)`. -/
theorem loopcd_tan_alpha_trig_eq_rational (p q u v : Int) (hp : 0 < p) (hq : 0 < q) (hu : 0 ≤ u) (hv : 0 < v) :
    (loopcdTanAlphaTrig p q u v).Wf ∧
    (loopcdTanAlpha p q u v).Wf ∧
    QEq (loopcdTanAlphaTrig p q u v) (loopcdTanAlpha p q u v) := by
  have hqp_pos : 0 < q * q + p * p := Int.add_pos (int_sq_pos_of_pos hq) (int_sq_pos_of_pos hp)
  have hden_pos := loopcd_denom_pos p q u v hp hq hu hv
  have htrig_wf : 0 < (loopcdTanAlphaTrig p q u v).den := by
    unfold loopcdTanAlphaTrig qdiv qadd qmul qsub stereoCos stereoSin
    dsimp only
    have heq : v * (q * q + p * p) * (1 * (v * (1 * (q * q + p * p))) + u * (1 * (q * q + p * p) - (q * q - p * p) * 1) * 1) =
        (v * (q * q + p * p)) * (v * q * q + (v + 2 * u) * p * p) :=
      poly_eq_of_norm_eq [p, q, u, v] trigDenomLHS trigDenomRHS rfl
    rw [heq]
    exact Int.mul_pos (Int.mul_pos hv hqp_pos) hden_pos
  refine ⟨htrig_wf, hden_pos, ?_⟩
  exact poly_eq_of_norm_eq [p, q, u, v] trigEqRatLHS trigEqRatRHS rfl

/-- Exact squared Euclidean norm `||h_raw||^2 = 1 + 2 ω (1 + ω) (1 - cos θ)` of `h_raw = (1+ω)h_R - ω h_1`. -/
theorem loopcd_raw_norm_sq_exact (p q u v : Int) :
    let cosNum := q * q - p * p
    let sinNum := 2 * p * q
    let scDen  := q * q + p * p
    let rawX   := (v + u) * scDen - u * cosNum
    let rawY   := u * sinNum
    rawX * rawX + rawY * rawY =
      (v * scDen) * (v * scDen) + 2 * u * (v + u) * (2 * p * p) * scDen :=
  poly_eq_of_norm_eq [p, q, u, v] rawNormLHS rawNormRHS rfl

/-- **THEOREM 1 (`LoopCD` Spherical Chord Self-Damping Saturation Gap)**:
    For ALL rational half-angle tangents `s = p / q > 0` (`p > 0, q > 0`) and ALL guidance
    strengths `ω = u / v ≥ 0` (`u ≥ 0, v > 0`):
    1. `cot(θ/2) - tan(α) == (1 + s^2) / (s * (1 + (1 + 2ω) s^2))` (`QEq`),
    2. Both numerator and denominator of the gap are strictly positive (`> 0`),
    3. `tan(α) < cot(θ/2)` (`QLt`), proving algebraically in rational half-angle tangents
       that spherical chord guidance maintains a strictly positive margin below the chord normal `cot(θ/2)`. -/
theorem thm1_loopcd_chord_saturation_gap_exact (p q u v : Int)
    (hp : 0 < p) (hq : 0 < q) (hu : 0 ≤ u) (hv : 0 < v) :
    (cotHalfTheta p q).Wf ∧
    (loopcdTanAlpha p q u v).Wf ∧
    (loopcdSaturationGap p q u v).Wf ∧
    QEq (qsub (cotHalfTheta p q) (loopcdTanAlpha p q u v)) (loopcdSaturationGap p q u v) ∧
    0 < (loopcdSaturationGap p q u v).num ∧
    QLt (loopcdTanAlpha p q u v) (cotHalfTheta p q) := by
  have hden := loopcd_denom_pos p q u v hp hq hu hv
  have hqp : 0 < q * q + p * p := Int.add_pos (int_sq_pos_of_pos hq) (int_sq_pos_of_pos hp)
  have hgap_den : 0 < p * (v * q * q + (v + 2 * u) * p * p) := Int.mul_pos hp hden
  have hgap_num : 0 < v * q * (q * q + p * p) := Int.mul_pos (Int.mul_pos hv hq) hqp
  refine ⟨hp, hden, hgap_den, ?_, hgap_num, ?_⟩
  · exact poly_eq_of_norm_eq [p, q, u, v] satGapEqLHS satGapEqRHS rfl
  · unfold QLt loopcdTanAlpha cotHalfTheta
    dsimp only
    have hdiff : q * (v * q * q + (v + 2 * u) * p * p) - (2 * u * p * q) * p =
        v * q * (q * q + p * p) :=
      poly_eq_of_norm_eq [p, q, u, v] satGapDiffLHS satGapDiffRHS rfl
    have hsub_pos : 0 < q * (v * q * q + (v + 2 * u) * p * p) - (2 * u * p * q) * p := by
      rw [hdiff]; exact hgap_num
    exact Int.lt_of_sub_pos hsub_pos

/-- Strict monotonicity of `tan α(s, ω)` in `ω`: if `0 ≤ ω_1 < ω_2` and `s > 0`, then `tan α(s, ω_1) < tan α(s, ω_2)`. -/
theorem loopcd_tan_alpha_strict_mono_omega (p q u1 v1 u2 v2 : Int)
    (hp : 0 < p) (hq : 0 < q) (hu1 : 0 ≤ u1) (hv1 : 0 < v1) (hu2 : 0 ≤ u2) (hv2 : 0 < v2)
    (hω : QLt ⟨u1, v1⟩ ⟨u2, v2⟩) :
    QLt (loopcdTanAlpha p q u1 v1) (loopcdTanAlpha p q u2 v2) := by
  unfold QLt at hω ⊢
  unfold loopcdTanAlpha
  dsimp only
  have hdiff :
      (2 * u2 * p * q) * (v1 * q * q + (v1 + 2 * u1) * p * p) -
      (2 * u1 * p * q) * (v2 * q * q + (v2 + 2 * u2) * p * p) =
      2 * p * q * (q * q + p * p) * (u2 * v1 - u1 * v2) :=
    poly_eq_of_norm_eq [p, q, u1, v1, u2, v2] monoDiffLHS monoDiffRHS rfl
  have _ := hu1; have _ := hv1; have _ := hu2; have _ := hv2
  have hω_sub : 0 < u2 * v1 - u1 * v2 := Int.sub_pos_of_lt hω
  have hqp : 0 < q * q + p * p := Int.add_pos (int_sq_pos_of_pos hq) (int_sq_pos_of_pos hp)
  have h2pq : 0 < 2 * p * q := Int.mul_pos (Int.mul_pos (by decide : (0 : Int) < 2) hp) hq
  have hprod : 0 < 2 * p * q * (q * q + p * p) * (u2 * v1 - u1 * v2) :=
    Int.mul_pos (Int.mul_pos h2pq hqp) hω_sub
  rw [← hdiff] at hprod
  exact Int.lt_of_sub_pos hprod

/-- Geodesic doubling operator (`SLERP` at `ω = 1`, reflecting `h_1` across `h_R` on `S^1`
    cleared by `||h_R||^2 = scDen^2`): `2 * ⟨h_R, h_1⟩ * h_R - scDen^2 * h_1`. -/
def slerpDoubledScaled (hR h1 : Int × Int) (scDen : Int) : Int × Int :=
  let dot := hR.1 * h1.1 + hR.2 * h1.2
  (2 * dot * hR.1 - (scDen * scDen) * h1.1, 2 * dot * hR.2 - (scDen * scDen) * h1.2)

/-- Derived geodesic-doubling antipodal wraparound at `s = 1` (`cos θ = 0, sin θ = 1`) and `ω = 1`:
    applying `slerpDoubledScaled` to `h_R = (scDen, 0)` and `h_1 = (stereoCos(1,1).num, -stereoSin(1,1).num)`
    yields `scDen^2 * (-h_1)` (strictly negative inner product `-scDen^4` with `h_1`), whereas
    `LoopCD` yields `tan α = 1/2 < 1 = cotHalfTheta(1, 1)`. -/
theorem slerp_wraps_antipode_at_90deg_vs_loopcd :
    let scDen := (stereoCos 1 1).den
    let hR := (scDen, (0 : Int))
    let h1 := ((stereoCos 1 1).num, -(stereoSin 1 1).num)
    let hSlerp := slerpDoubledScaled hR h1 scDen
    hSlerp = (-(scDen * scDen) * h1.1, -(scDen * scDen) * h1.2) ∧
    (hSlerp.1 * h1.1 + hSlerp.2 * h1.2 = -(scDen * scDen * scDen * scDen)) ∧
    QEq (loopcdTanAlpha 1 1 1 1) ⟨1, 2⟩ ∧
    QLt (loopcdTanAlpha 1 1 1 1) (cotHalfTheta 1 1) := by
  decide

end LoopCDStrang
