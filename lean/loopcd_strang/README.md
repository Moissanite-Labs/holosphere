# `LoopCDStrang` — Pure-`Init` `Lean 4.36.0` Spherical Chord Self-Damping & Palindromic Strang `SU(2, Q)` Holonomy Suite

`LoopCDStrang` (`lean/loopcd_strang/`) is a self-contained, zero-dependency (`"packages": []` in `lake-manifest.json`) **`Lean 4.36.0`** formalization proving **`38 / 38` constructive theorems** (`3` zero-axiom `[]`, `29` `[propext]`, `6` `[propext, Quot.sound]`, `0` `Classical.choice`, `0` `bv_decide`, `0` `native_decide`, `0` `sorryAx`, `0` `partial def`) governing latent spherical chord guidance and non-Abelian `SU(2, Q)` spinor holonomy extrapolation in `HoloSphere`.

Together with the `51` pure-`Init` `Lean 4.36.0` theorems in `lean/*.lean`, `LoopCDStrang` brings `HoloSphere`'s machine-checked formal suite to **`89` pure-`Init` `Lean 4.36.0` theorems**.

---

## Summary of Main Constructive Theorems

1. **Spherical Chord Self-Damping Saturation Gap (`thm1_loopcd_chord_saturation_gap_exact`, `[propext]`):**
   - For any unit vectors $h_1, h_R \in S^{d-1}$ with half-angle tangent $s = \tan(\theta/2) = p/q > 0$ ($p, q \in \mathbb{Z}_{>0}$) and any guidance weight $\omega = u/v \ge 0$ ($u \ge 0, v > 0$), Euclidean chord extrapolation $h_{\mathrm{raw}} = (1 + \omega) h_R - \omega h_1$ followed by radial projection satisfies:
     $$\tan\alpha = \frac{\omega \sin\theta}{1 + \omega(1 - \cos\theta)} = \frac{2 \omega s}{1 + (1 + 2\omega) s^2}, \qquad \tan\!\left(\frac{\pi - \theta}{2}\right) - \tan\alpha = \frac{1 + s^2}{s\bigl(1 + (1 + 2\omega) s^2\bigr)} > 0.$$
   - Consequently `tan α < cot(θ/2)` strictly for all `ω ≥ 0` (maintaining a strictly positive rational tangent margin below the chord normal, whereas geodesic doubling `slerpDoubledScaled` at `s = 1, ω = 1` reflects `h_1` to `-h_1`: `slerp_wraps_antipode_at_90deg_vs_loopcd`, `[]`).
2. **Right-Ordered Palindromic Strang vs. Left-Ordered `SU(2, Q)` BCH Quadratic Cancellation (`thm2a_right_strang_bch_quadratic_zero` & `thm2b_left_ordered_bch_quadratic_defect_nonzero`, `[propext]`):**
   - Constructs a multivariate commutative-ring normalizer (`poly_eq_of_norm_eq`) and non-perturbative cubic-remainder jet algebra (`Jet3`: $c_0 + t c_1 + t^2 c_2 + t^3 \mathrm{rem}$ over $\mathbb{Z}$) with zero Taylor truncation error.
   - Proves universally over $(a, b, S, u, v)$ that under Cayley-parameter scaling $\omega \cdot w_\Delta$ ($w_\Delta = (-w_1) \oplus_C w_R$), right-ordered palindromic composition $C(w_R) \cdot C(\omega w_\Delta)$ (`w_R ⊕_C (ω ((-w_1) ⊕_C w_R))`) against linear Cayley guidance $w_{\mathrm{guide}} = (1 + \omega) w_R - \omega w_1$ cancels degree-0, degree-1, **and** degree-2 BCH commutator defects identically with exact $O(t^3)$ remainder ($t^3 \cdot \mathrm{rem}$), whereas left ordering $C(\omega w_\Delta) \cdot C(w_R)$ retains an exact nonzero quadratic commutator defect $-4 v^2 u S^9 (a \times b)$ (`(0, 0, -4)` on the orthogonal witness; `thm2c_dyadic_su2_chordal_hierarchy_witness`, `[]`).
3. **Acute-Ball Denominator Safety Floors `9/25` & `81/625` (`thm3a_acute_ball_spinor_denom_ge_9_25`, `thm3b_acute_ball_cayley_denom_ge_81_625`, `thm3c_acute_ball_81_625_tight`):**
   - Via an exact sum-of-nonnegatives identity (`acute_ball_cayley_denom_sos_identity`), proves that for all $\|w_1\|, \|w_R\| \le 4/5$ and $\omega \in [0, 1]$, spinor half-tangent denominators satisfy $\ge 9/25 > 0$ and Cayley composition denominators satisfy $\ge 81/625 > 0$ tightly.
4. **Commutative / Collinear All-Orders Collapse (`thm4a_commuting_cayley_quat_commute`, `thm4b_commutative_left_right_collapse`, `thm4c_collinear_left_right_cayley_sum_exact`):**
   - Whenever $w_1 \times w_R = (0, 0, 0)$, quaternion Cayley rotors commute identically, right-ordered and left-ordered extrapolators coincide to all orders for every $\omega = u/v$, and at $\omega = 1$ both equal the iterated Cayley sum $(w_R \oplus_C (-w_1)) \oplus_C w_R$.

---

## Complete 38-Theorem Catalog

| # | Module | Public Theorem | Axiom Footprint |
| :---: | :--- | :--- | :--- |
| 1 | `IntPoly` | `evalPoly_norm` | `[propext]` |
| 2 | `IntPoly` | `poly_eq_of_norm_eq` | `[propext]` |
| 3 | `IntPoly` | `Jet3.eval_const` | `[propext]` |
| 4 | `IntPoly` | `Jet3.eval_varT` | `[propext]` |
| 5 | `IntPoly` | `Jet3.eval_neg` | `[propext]` |
| 6 | `IntPoly` | `Jet3.eval_add` | `[propext]` |
| 7 | `IntPoly` | `Jet3.eval_sub` | `[propext]` |
| 8 | `IntPoly` | `Jet3.eval_scale` | `[propext]` |
| 9 | `IntPoly` | `Jet3.eval_mul` | `[propext]` |
| 10 | `IntPoly` | `int_sq_nonneg` | `[propext]` |
| 11 | `IntPoly` | `int_sq_pos_of_pos` | `[propext]` |
| 12 | `SphereChord` | `stereo_unit_circle_exact` | `[propext]` |
| 13 | `SphereChord` | `loopcd_denom_pos` | `[propext]` |
| 14 | `SphereChord` | `loopcd_tan_alpha_trig_eq_rational` | `[propext]` |
| 15 | `SphereChord` | `loopcd_raw_norm_sq_exact` | `[propext]` |
| 16 | `SphereChord` | `thm1_loopcd_chord_saturation_gap_exact` | `[propext]` |
| 17 | `SphereChord` | `loopcd_tan_alpha_strict_mono_omega` | `[propext]` |
| 18 | `SphereChord` | `slerp_wraps_antipode_at_90deg_vs_loopcd` | `[]` |
| 19 | `QuatStrang` | `quat_mul_assoc` | `[propext]` |
| 20 | `QuatStrang` | `palindromic_conjugation_identity` | `[propext]` |
| 21 | `QuatStrang` | `quat_norm_sq_mul` | `[propext]` |
| 22 | `QuatStrang` | `normSq3_nonneg` | `[propext]` |
| 23 | `QuatStrang` | `cayley_su2_unit_norm_exact` | `[propext]` |
| 24 | `QuatStrang` | `cayley_compose_su2_exact` | `[propext, Quot.sound]` |
| 25 | `QuatStrang` | `rightStrangDefectJet_eval_eq` | `[propext]` |
| 26 | `QuatStrang` | `leftOrderedDefectJet_eval_eq` | `[propext]` |
| 27 | `QuatStrang` | `thm2a_right_strang_bch_quadratic_zero` | `[propext]` |
| 28 | `QuatStrang` | `thm2b_left_ordered_bch_quadratic_defect_nonzero` | `[propext]` |
| 29 | `QuatStrang` | `thm2c_dyadic_su2_chordal_hierarchy_witness` | `[]` |
| 30 | `QuatStrang` | `acute_ball_cayley_denom_sos_identity` | `[propext]` |
| 31 | `QuatStrang` | `acute_ball_closed_under_unit_omega` | `[propext]` |
| 32 | `QuatStrang` | `thm3a_acute_ball_spinor_denom_ge_9_25` | `[propext, Quot.sound]` |
| 33 | `QuatStrang` | `thm3b_acute_ball_cayley_denom_ge_81_625` | `[propext, Quot.sound]` |
| 34 | `QuatStrang` | `thm3c_acute_ball_81_625_tight` | `[]` |
| 35 | `QuatStrang` | `cayley_comp_commutator_exact` | `[propext]` |
| 36 | `QuatStrang` | `thm4a_commuting_cayley_quat_commute` | `[propext, Quot.sound]` |
| 37 | `QuatStrang` | `thm4b_commutative_left_right_collapse` | `[propext, Quot.sound]` |
| 38 | `QuatStrang` | `thm4c_collinear_left_right_cayley_sum_exact` | `[propext, Quot.sound]` |

---

## Verification Commands

```bash
# 1. Verify self-bound on-disk certificate & pure-Q witnesses (0 floats, 0 ast.Div)
python3 -I -B lean/loopcd_strang/verify_loopcd_strang.py --check

# 2. Run negative-control & tamper-rejection unit suite
python3 -I -B tests/test_loopcd_strang_r5.py

# 3. Full cold Lean 4.36.0 build + leanchecker --fresh replay + live 38-theorem axiom check
python3 -I -B lean/loopcd_strang/verify_loopcd_strang.py --check --live-lean
```
