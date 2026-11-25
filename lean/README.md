# HoloSphere Pure-`Init` Lean 4.36.0 Algebraic Invariants

Machine-checked formal proofs for **Project HoloSphere** (`HoloSphereInvariants.lean`), compiled strictly in **Pure `Init` `Lean 4.36.0`** (`0` external imports, `0` Mathlib, `0` `Classical.choice`, `0` `sorryAx`, `0` `native_decide`, `0` compiler warnings; every public theorem depends solely on subsets of `{propext, Quot.sound}`).

---

## Silicon–Geometry Isomorphism & 36-Theorem Catalog

### Pillar 1 — $SU(2) \cong S^3$ Associative Prefix Scan, Euler Norm, Rational `cayley_su2`, Local-$S^3$ & Suffix-Sum Adjoint (`holosphere/geometry/su2.py`, `holosphere/kernels/csrc/su2_holonomy_scan_sm86.cu`)

| # | Lean 4 Theorem | Algebraic Guarantee & Hardware Consequence |
|---|---|---|
| 1 | `p1_quat_mul_assoc` | Universal Hamilton quaternion associativity $(pq)r = p(qr)$ |
| 2 | `p1_quat_identity_and_conj_antihom` | Identity $(1,0,0,0)$, anti-homomorphic $\overline{pq} = \bar{q}\bar{p}$, and $\|\bar{p}\|^2 = \|p\|^2$ |
| 3 | `p1_quat_noncommutative_su2_holonomy_witness` | Non-Abelian witness $\mathbf{i}\mathbf{j} = \mathbf{k} \ne -\mathbf{k} = \mathbf{j}\mathbf{i}$ separating $SU(2)$ holonomy from Abelian `RoPE`/`LieRE` |
| 4 | `p1_quat_euler_four_square_norm` | Multiplicative norm conservation $\|pq\|^2 = \|p\|^2 \|q\|^2$ |
| 5 | `p1_quat_prefix_scan_unit_norm` | Arbitrary-length $S^3$ prefix scan preserves $\|{\Pi}_t\|^2 = 1$ identically (zero norm drift at arbitrary $T$) |
| 6 | `p1_quat_chunked_scan_eq_sequential` | Chunked/segmented prefix scan equals token-by-token $O(1)$-state sequential recurrence |
| 7 | `p1_quat_hillis_steele_8lane_eq_sequential` | Explicit 3-round 8-lane Hillis–Steele warp-shuffle scan equals sequential recurrence across all 8 lanes |
| 8 | `p1_quat_bilinear_two_sided_adjoint` | Universal reverse-mode adjoint $\langle a, (bx)c \rangle = \langle (\bar{b}a)\bar{c}, x \rangle$ (zero unit restriction) |
| 9 | `p1_quat_scan_step_backward_factorization` | Eliminates intermediate suffix $r_{t\to j}$ via $\|\pi_t\|^2 \langle g, (\pi_{t-1}\delta u)r_{t\to j}\rangle = \langle (\overline{\pi_{t-1}}(g\overline{\pi_j}))\pi_t, \delta u\rangle$ |
| 10 | `p1_quat_scan_suffix_sum_adjoint_collapse` | Collapses non-Abelian backward scan into a **commutative 4-lane additive suffix sum** $\Sigma_t = \sum_{j \ge t} g_j \overline{\Pi_j}$ (`0` quaternion multiplies inside the warp scan) |
| 11 | `p1_cayley_su2_unit_norm_and_acute_positivity` | Homogenized SFU-free rational $SU(2)$ Cayley map $(d^2 - \|\omega\|^2, 2d\omega)$ satisfies exact Euler norm $(d^2 + \|\omega\|^2)^2$ and strict $w > 0$ (`-I` pole immunity) for $\|\omega\|^2 < d^2$ |
| 12 | `p1_local_4d_s3_zero_cross_bank_bleed` | Modifying bank $B$ leaves local block-4 $S^3$ normalized bank $A$ identically invariant (`0` bleed), whereas a global coupled denominator $S_A + S_B$ strictly perturbs bank $A$ |

### Pillar 2 — Covariant $(Q, K, V)$ Parallel Transport & Un-Transport (`holosphere/geometry/su2.py`)

| # | Lean 4 Theorem | Algebraic Guarantee & Hardware Consequence |
|---|---|---|
| 13 | `p2_covariant_left_right_inverse_cancel` | Exact conjugation inverse $\bar{\pi}(\pi v) = \|\pi\|^2 v$ and $(v\pi)\bar{\pi} = \|\pi\|^2 v$ (`= v` on $S^3$) |
| 14 | `p2_covariant_left_transport_isometry` | Euclidean inner-product preservation $\langle \pi u, \pi v \rangle = \|\pi\|^2 \langle u, v \rangle$ |
| 15 | `p2_covariant_qk_score_relative_holonomy` | Standard FlashAttention dot product $\langle \Pi_t Q_t, \Pi_s K_s \rangle = \langle Q_t, (\overline{\Pi_t}\Pi_s) K_s \rangle$ |
| 16 | `p2_covariant_interval_path_cancellation` | Prefix history $\Pi_s$ cancels identically: $\overline{\Pi_s}(\Pi_s \Delta_{s\to t}) = \Delta_{s\to t}$ and $\overline{\Pi_s \Delta_{s\to t}}\Pi_s = \overline{\Delta_{s\to t}}$ |
| 17 | `p2_covariant_value_untransport_single` | Single-value covariant un-transport $\overline{\Pi_t}(\Pi_s V_s) = (\overline{\Pi_t}\Pi_s) V_s$ and self-identity $\overline{\Pi_t}(\Pi_t V_t) = V_t$ |
| 18 | `p2_covariant_value_untransport_weighted_sum` | Attention output un-transport $\overline{\Pi_t} \sum_s \alpha_{ts}(\Pi_s V_s) = \sum_s \alpha_{ts}(\overline{\Pi_t}\Pi_s) V_s$ over arbitrary key-value lists |

### Pillar 3 — SFU-Free Rational Cayley Safe-Hopf Clock & Hopf $S^3 \to S^2$ Fiber (`holosphere/geometry/hopf.py`)

| # | Lean 4 Theorem | Algebraic Guarantee & Hardware Consequence |
|---|---|---|
| 19 | `p3_cayley_rational_unit_circle_identity` | Exact rational stereographic unit-circle law $(d^2 - n^2)^2 + (2dn)^2 = (d^2 + n^2)^2$ and $(1 - t^2)^2 + (2t)^2 = (1 + t^2)^2$ (zero SFU `sin`/`cos`) |
| 20 | `p3_cayley_acute_cone_strict_pos_and_floor` | Acute-cone Safe-Hopf schedule $0 \le n \le n_{\max} < d$ strictly avoids $-I$ with $c(t) \ge c(t_{\max}) > 0$ |
| 21 | `p3_cayley_acute_strict_monotone_distinct` | Strict monotonicity $c(t_2) < c(t_1)$ for $0 \le t_1 < t_2 < 1$ (zero depth-recurrence aliasing) |
| 22 | `p3_w10_obs31_obtuse_antipodal_sign_flip` | Proves `W10-OBS-31` antipodal sign flip $\langle v, R(d,n)v \rangle < 0$ for all $v \ne 0$ when $0 < d < n$ ($|t| > 1$) |
| 23 | `p3_hopf_s3_to_s2_norm_and_right_fiber_invariance` | Proves $q \mathbf{i} \bar{q} = (0, h_x, h_y, h_z)$, $\|h(q)\|_{S^2}^2 = \|q\|^4$, and exact right-$S^1$ fiber gauge invariance $h(q(a + b\mathbf{i})) = (a^2 + b^2) h(q)$ |

### Pillar 4 — Beltrami `[C, K] = 0` Helicity, Closed-Form `Cayley2x2` Adjoint, Poloidal Bingham & Darboux (`holosphere/geometry/{beltrami,bingham,darboux}.py`)

| # | Lean 4 Theorem | Algebraic Guarantee & Hardware Consequence |
|---|---|---|
| 24 | `p4_beltrami_cayley_2x2_orthogonality_and_det` | $2\times 2$ skew-Cayley norm isometry and positive $SO(2)$ determinant $(d^2 + n^2)^2$ |
| 25 | `p4_beltrami_paired_commutator_and_helicity_conservation` | Paired chiral $[R_{\text{Bel}}, K_{\text{Bel}}] = 0$, exact energy & helicity conservation, and cross-sector swap helicity breach |
| 26 | `p4_beltrami_symmetric_residual_rayleigh_drift` | Symmetric commuting residual strictly shifts the Rayleigh helicity-to-energy ratio |
| 27 | `p4_poloidal_bingham_hodge_stagnant_lid` | Below-yield poloidal zeroing ($\|u_{\text{pol}}\|^2 \le \tau^2$) with exact unattenuated toroidal preservation |
| 28 | `p4_darboux_isospectral_intertwining_and_dark_kernel` | Rectangular Darboux intertwining $A(A^T A + c I_n) = (A A^T + c I_m)A$ and kernel collision $A(x + k) = Ax$ |
| 29 | `p4_cayley2x2_closed_form_adjoint` | Exact closed-form reverse-mode gradient $(1 + w^2)\langle g, \partial y/\partial w\rangle = 2(g_0 y_1 - g_1 y_0)$ (`1` rcp + `7` FMAs, `0` trig) |

### Pillar 5 — Weft Composed-$\Delta W$ `C_row` Gate, `16 B` Holonomy Rollback, `gateAsymmetric`, & Left-Factor Quantization (`holosphere/models/weft_shell.py`)

| # | Lean 4 Theorem | Algebraic Guarantee & Hardware Consequence |
|---|---|---|
| 30 | `p5_weft_w10_obs20_cancellation_attack_blocked` | Universal `W10-OBS-20` proof: $\Delta W_1, \Delta W_2$ each pass $C_{\text{row}} = 1/2 \le 3/5$, yet $\Delta W_1 + \Delta W_2$ hits $C_{\text{row}} = 1 > 3/5$ and is rejected |
| 31 | `p5_weft_tri_conjunctive_commit_soundness` | Tri-conjunctive gate accepts iff positive Frobenius norm, composed $C_{\text{row}} \le c_{\max}$, deterministic receipt, and holdout safety all hold |
| 32 | `p5_weft_isotropic_sleep_preserves_crow_gate` | Isotropic sleep scaling $M \mapsto \lambda M$ ($\lambda \ne 0$) preserves $C_{\text{row}} \le c_{\max}$ (`iff`) |
| 33 | `p5_weft_atomic_rollback_exact_involution` | Exact entrywise single-patch and multi-patch LIFO atomic rollback restoration |
| 34 | `p5_su2_holonomy_repoisoning_and_16b_rollback` | Infinite-half-life $SU(2)$ holonomy re-poisoning law ($t_{1/2} = \infty$ constant error across all clean steps if $\Pi_{t_0}$ is un-restored) + exact `16 B` $\Pi_{t_0}$ checkpoint restoration cure |
| 35 | `p5_gate_asymmetric_recovery_vs_symmetric_deadlock` | `gateAsymmetric` ($\sum (u_i - u_{0,i})_+^2 \le R^2$) admits every strict safety descent at $R \le 0$, whereas symmetric trust-region gates deadlock |
| 36 | `p5_asymmetric_left_factor_zero_nullspace_drift` | Quantizing the left factor $Q(B)Ax = 0$ preserves $\ker(A)$ identically (`0` drift), whereas right-factor $Q(A)x \ne 0$ shatters $\ker(A)$ |

---

## Reproducible CPU-Only Verification (3 Isolated `-I` Modes)

```bash
# 1. Verify frozen rational witness certificate + live replay
python3 -I -B lean/verify_holosphere_lean.py --check

# 2. Run R13b 18-rule static gate + scrubbed live Lean 4.36.0 kernel & #print axioms check
python3 -I -B lean/verify_holosphere_lean.py --verify-lean

# 3. Run full positive + adversarial R5 + live Lean test suite across all 3 isolated -I modes
python3 -I -B -m unittest discover -s lean -p "test_*.py" -v
python3 -I -O -B -m unittest discover -s lean -p "test_*.py" -v
python3 -I -B -W error -m unittest discover -s lean -p "test_*.py" -v
```
