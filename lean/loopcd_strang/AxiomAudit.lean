import LoopCDStrang

-- Module 1: IntPoly & Jet3 (11 public theorems)
#print axioms LoopCDStrang.evalPoly_norm
#print axioms LoopCDStrang.poly_eq_of_norm_eq
#print axioms LoopCDStrang.Jet3.eval_const
#print axioms LoopCDStrang.Jet3.eval_varT
#print axioms LoopCDStrang.Jet3.eval_neg
#print axioms LoopCDStrang.Jet3.eval_add
#print axioms LoopCDStrang.Jet3.eval_sub
#print axioms LoopCDStrang.Jet3.eval_scale
#print axioms LoopCDStrang.Jet3.eval_mul
#print axioms LoopCDStrang.int_sq_nonneg
#print axioms LoopCDStrang.int_sq_pos_of_pos

-- Module 2: SphereChord (Theorem 1 & S^{d-1} Geometry — 7 public theorems)
#print axioms LoopCDStrang.stereo_unit_circle_exact
#print axioms LoopCDStrang.loopcd_denom_pos
#print axioms LoopCDStrang.loopcd_tan_alpha_trig_eq_rational
#print axioms LoopCDStrang.loopcd_raw_norm_sq_exact
#print axioms LoopCDStrang.thm1_loopcd_chord_saturation_gap_exact
#print axioms LoopCDStrang.loopcd_tan_alpha_strict_mono_omega
#print axioms LoopCDStrang.slerp_wraps_antipode_at_90deg_vs_loopcd

-- Module 3: QuatStrang (Theorems 2, 3, & 4 — 20 public theorems)
#print axioms LoopCDStrang.quat_mul_assoc
#print axioms LoopCDStrang.palindromic_conjugation_identity
#print axioms LoopCDStrang.quat_norm_sq_mul
#print axioms LoopCDStrang.normSq3_nonneg
#print axioms LoopCDStrang.cayley_su2_unit_norm_exact
#print axioms LoopCDStrang.cayley_compose_su2_exact
#print axioms LoopCDStrang.rightStrangDefectJet_eval_eq
#print axioms LoopCDStrang.leftOrderedDefectJet_eval_eq
#print axioms LoopCDStrang.thm2a_right_strang_bch_quadratic_zero
#print axioms LoopCDStrang.thm2b_left_ordered_bch_quadratic_defect_nonzero
#print axioms LoopCDStrang.thm2c_dyadic_su2_chordal_hierarchy_witness
#print axioms LoopCDStrang.acute_ball_cayley_denom_sos_identity
#print axioms LoopCDStrang.acute_ball_closed_under_unit_omega
#print axioms LoopCDStrang.thm3a_acute_ball_spinor_denom_ge_9_25
#print axioms LoopCDStrang.thm3b_acute_ball_cayley_denom_ge_81_625
#print axioms LoopCDStrang.thm3c_acute_ball_81_625_tight
#print axioms LoopCDStrang.cayley_comp_commutator_exact
#print axioms LoopCDStrang.thm4a_commuting_cayley_quat_commute
#print axioms LoopCDStrang.thm4b_commutative_left_right_collapse
#print axioms LoopCDStrang.thm4c_collinear_left_right_cayley_sum_exact
