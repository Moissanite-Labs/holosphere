# Track S164 (`EXP-Q45a` + `EXP-Q45c`): Finite-`N` `FWHT_256` / `Monarch-3` (`HoloSphere-V5`) vs. `OpenAI FiniteFourier` & `MUX` Cor. 15 / `Bayt al-Ḥikma` `Al-Kāshī` Fixed-Radix × `DualAxisCarry` × `Floodgates` `SU(2)` Looped Transport

**Track:** `S164` (`EXP-Q45a` + `EXP-Q45c`, Wave-22 Discovery Lane)  
**Environment & Policy:** CPU-only verification (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, `ZERO` git pushes, `ZERO` IEEE-754 floats in production arithmetic/logic); `energy_power_status = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"`; `gpu_ablation_status = "CPU_ONLY_NOT_RUN_ON_GPU_NO_FABRICATED_BPB"`.  
**Oracle & Certificate:** Pure-$\mathbb{Z}/\mathbb{Q}$ (`fractions.Fraction`) schema `s164-fourier-mux-loop-cert-v1` (`payload_sha256 = ad791c90f9b3658fe60279527b6256dbcc037ff22cc311b3cd0a315b09e086cb`, envelope `certificate_sha256 = d3caaea5903cbea9b50f90405e2f0e5cb24eceaf072386cd5e58e98d9563c1ec`).  
**Formal Verification:** Pure-`Init` `Lean 4.36.0` (`FourierMuxLoopOperators.lean`, `18/18` pure-`Init` theorems: `17` universally quantified + `1` concrete kernel-`decide` phase-boundary conjunction, `0` `sorry`, `0` `Classical.choice`, `[]` or `[propext, Quot.sound]` axioms only).

---

## 1. Prior Art Credit & Problem Framing (Law-4 & Law-5)

Track S164 connects four independent Wave-22 lines with verified HoloSphere / Bayt al-Ḥikma primitives in exact pure-$\mathbb{Z}/\mathbb{Q}$ arithmetic and Lean 4.36.0:

1. **OpenAI `FiniteFourier` (Theorems `1.1`, `1.2`, `4.1`, and `Prop 5.2`):** Established optimal batched asymptotic circuit complexity $\Theta(J N \sqrt{\log N})$ (upper bound $C_\varepsilon J N (1+\log N)^{1/2+\varepsilon}$, lower bound $c J N \sqrt{\log N}$), Zariski-closure linear+low-rank+sparse resistance lower bounds $s \ge c h \sqrt{\log N}$, Theorem `4.1` shift-concentration ($O(\sqrt{m\log m})$ Hamming window), and `Prop 5.2` $1 \times L$ rectangle character vs. permutation price $(L + \sqrt{1+L^2})\log N$ vs. $(L+1)\log N$. Crucially, `FiniteFourier` itself observes that the character transform on $(\mathbb{Z}/2\mathbb{Z})^m$ (`FWHT`) uses $m N$ butterfly gates (`2,048` gates at `N = 256, m = 8`).
2. **`Track S156b`, `Track S157`, and `Track S159` (`HoloSphere-V4`):** Replaced byte vocab HBM embeddings with on-the-fly `FWHT_256 × bent4` input encoding (`0 B` HBM vocab) and a `2`-stage `16×16` Monarch output head (`8,448` params = `16,896 B`, closing `92.63%` of the S156b dense head gap at `26,292 B` total E2E SMEM, `0 B` spill). However, `Track S159` (`15,110,144` params, `54` `A_16` layers, `T=8192`) left a `+0.1581 BPB` (`42871423033/271186500000`) gap between `V4` (`1486778/542373 = 2.7412 BPB`) and dense `Row 3` (`1291579/500000 = 2.5832 BPB`).
3. **Multiplexed Token Transport `MUX` (`arXiv:2607.18264`, Corollaries `14` & `15`) & `Bayt al-Ḥikma` `Al-Kāshī` Fixed-Radix (`SOP-16`, `Track S91`):** `MUX` derives the geometric minimum demultiplexing margin $\mathcal{E}(\boldsymbol{\alpha}) = \frac{q-p}{q^S - p^S} m_S$ and proves unique recovery requires accumulation drift $\eta_S(u) = \frac{S u}{1 - (S-1)u} < \mathcal{E}(\boldsymbol{\alpha})/2$. We contrast floating-point `MUX` superpositions against `Jamshīd al-Kāshī` (`1427 CE`, *Miftāḥ al-Ḥisāb*) exact fixed-radix arithmetic (`u = 0`) and cubic fixed-point contraction ($x_{n+1} = (q + x_n^3)/p$), alongside exact integer `A_16 Block-H4` residual quantization (`Track S163`).
4. **Looped Multi-Token Transport (`DualAxisCarry` / `LOOM` / `InfiLoop`, `AgenticPPL`, `Floodgates` `arXiv:2611.33896v1`, `MoLT` `arXiv:2611.34923` Prop. `1`):** Addresses carried-state erasure at $\beta \to 0$ vs. $\beta_{\min} = 0.55 = 11/20$ projective EMA damping, `AgenticPPL` `256`-byte vocabulary regret invariance (`0.0003 bpb` vs. `128K` BPE `0.081 bpb` / `0.32` code), `Floodgates`'s `Ouro-1.4B` `r=1→4` agreement collapse (`88% → 33%`) and `[0.2, 1.5]` scalar clamping / `MERGE_LORA=0`, and `MoLT`'s $O(TRL)$ KV memory vs. `HoloSphere`'s `SU(2)`-aligned $O(1)$ `S_fast` Woodbury cache.

---

## 2. Deliverable 1 (`EXP-Q45a`): Finite-`N` `FWHT_256` Crossover & `HoloSphere-V5` Head

### 2.1 Exact Finite-`N` `FWHT_256` vs. `FiniteFourier` Crossover (`N = 256`, `m = 8`)

Because $\log_2 256 = 8$, $1 + m = 9 = 3^2$ is an **exact integer square**: $\sqrt{1 + \log_2 256} = 3$ with zero approximation error. Comparing `FWHT_256`'s `m = 8` butterfly gates/coord (`2,048` gates/vector, `4,096` per dual-bank token) against `FiniteFourier`'s asymptotic curve $C_{\mathrm{FF}} N (1 + \log_2 N)^{1/2+\varepsilon}$ yields the exact rational crossover boundary (even at $\varepsilon = 0$, the most favorable possible exponent for `FiniteFourier`):

| Prefactor $C_{\mathrm{FF}}$ | `FWHT_256` Gates/Coord | `FiniteFourier` ($\varepsilon=0$) Gates/Coord | `FWHT_256` Wins Strictly? |
|---|---|---|---|
| `1` | `8` | `3` | `False` |
| `2` | `8` | `6` | `False` |
| `8/3` (exact tie) | `8` | `8` | `False` (exact parity) |
| `3` | `8` | `9` | **`True`** (`8 < 9`) |
| `4` | `8` | `12` | **`True`** (`8 < 12`) |
| `8` | `8` | `24` | **`True`** (`8 < 24`) |
| `16` | `8` | `48` | **`True`** (`8 < 48`) |

Asymptotic crossover dimension $N^* = 2^{m^*}$ (where `m > C * sqrt(1 + m)`):
- At **$C_{\mathrm{FF}} = 3$**: `FWHT` still strictly wins at $m=8$ (`N=256`: `8 < 9`, `8^2=64 < 81`) and $m=9$ (`N=512`: `9 < 12`, `81 < 90`), still beats the ceiling-sqrt proxy at $m=10$ (`10 < 3 * ceil(sqrt(11)) = 12`, `fwht_wins_strict = true`) while crossing the exact squared boundary at **$m^* = 10$ ($N^* = 1{,}024$: `m^2 = 100 > 99 = 9*(1+10)`, `fwht_wins_sq_bound = false`)**, and ties the ceiling-sqrt proxy at $m = 12$ (`12 == 3 * ceil(sqrt(13)) = 12`).
- At **$C_{\mathrm{FF}} = 4$**: `FWHT` strictly beats $4\sqrt{1+m}$ through $m = 16$ (`N = 65,536`: `16 < 20`, `256 < 272`), crossing at **$m^* = 17$ ($N^* = 131{,}072$: `289 > 288`)**.
- **Theorem `4.1` Zero Compression at `N = 256`:** For $m = 8$ ($\log_2 m = 3$), Theorem `4.1`'s concentration width is $2 \lceil \sqrt{8 \cdot 3} \rceil + 1 = 2(5) + 1 = 11 > 9 = m + 1$, proving `0` Hamming-level pruning compression exists at `N = 256`.
- **`Prop 5.2` Spectral Price Gap:** At $m = 8$, character-basis cost $(L + \lceil\sqrt{1+L^2}\rceil) m$ strictly exceeds permutation cost $(L+1) m$ by `8, 16, 24, 32` gates/coord across $L = 1, 2, 3, 4$, confirming why `FWHT_256 × bent4` (`96/256` negated coordinates across `12/32` lanes; exact involution $H_{256}^2 = 256 I$ and Parseval energy `1,408 -> 360,448 = 256 × 1,408`) requires no asymptotic `FiniteFourier` substitution at byte width `N = 256`.

### 2.2 Diagnosis & Structural Closure of `Track S159`'s `+0.1581 BPB` Output-Head Gap (`HoloSphere-V5`)

Pure-$\mathbb{Z}$ inspection of `V4` (`v4_input_embedding_512_int`, `monarch_stage_allones`) and `Track S159` isolates two concrete structural bottlenecks behind the `+42871423033/271186500000 BPB` (`+0.1581 BPB`) deficit vs. dense `Row 3`:

1. **Special-Token `sp_flip` Aliasing (`4/4` exact byte collisions):** In `V4`, dual-bank `bank 1 = -bank 0` constrains embeddings to $\text{span}(\text{bank 0}) \cong \mathbb{R}^{256}$. On bytes, XORing coordinate index by `sp_flip = 0x21 = 33` multiplies the Hadamard character $\langle x, i \rangle$ by $(-1)^{\langle x, \text{0x21} \rangle} = d_5(x)$, forcing exact 1-to-1 identity collisions for all four control symbols (`IDs 256..259`):
   - `256 (BOS) == 33 ('!')`
   - `257 (EOS) == 32 (' ')`
   - `258 (PAD) == 35 ('#')`
   - `259 (UNK) == 34 ('"')`
2. **Subblock Rank-`1` & `256`-dim Truncation:** `V4`'s `2`-stage `16×16` Monarch head discards `bank 1` (`head_input_dim = 256`) and provides only `1` path per `(i, j)` pair across all `256 × 256` pairs (`min_paths_2stage = max_paths_2stage = 1`), capping `(r_1, g_1)` `16×16` subblock rank at `1`, analogous to `Track S156b`'s Zariski-defective rank-`16` head (`240`-dimensional nullity; `arm8a` `4.5241` vs. `arm9` `3.6617`, `92.63%` gap closed).

**`HoloSphere-V5` Zero-Alias + `3`-Stage Monarch Derivation (`0 B` HBM vocab, `0 B` SMEM spill):**
- **Input Orthogonality ($H_2 \otimes (H_{256} \times \text{bent4})$):** Encoding bytes `0..255` as `(u, -u)` (`H_2` sign `(-1, +1)`) and special symbols `256..511` as `(v, +v)` makes all `260` (and universally all `512`) input symbols pairwise orthogonal in $\mathbb{Z}^{512}$ (`512 I_260`, `v5_unique_vectors_of_260 = 260`, `v5_max_offdiag_dot = 0`, `input_aliases = 0`), preserving `AgenticPPL`'s `256`-byte vocabulary advantage (`3/10000` regret vs. `128K` BPE `81/1000 = 270×` and `8/25` code `3200/3×`).
- **Two Candidate `3`-Stage Monarch Heads Within `sm_86` `48 KiB` (`49,152 B`) Static SMEM:**

| Architecture | Input $\to$ Output Dim | Stages | Paths / Pair (All `(i,j)`) | Subblock Rank Cap | Head Params (`FP16 B`) | HBM Vocab | E2E SMEM (`sm_86` Headroom) | Input Aliases |
|---|---|---|---|---|---|---|---|---|
| `V4_Monarch2_16x16` | `512` (`256` used) $\to$ `256` | `2` | `1` | `1` | `8,448` (`16,896 B`) | `0 B` | `26,292 B` (`+22,860 B`) | `4` (`256..259`) |
| `V5A_Monarch3_8x8x8_512` | `512` $\to$ `512` | `3` | `1` | `1` | `12,548` (`25,096 B`) | `0 B` | `35,508 B` (`+13,644 B`) | **`0`** |
| **`V5B_Monarch3_16x32_16x16_16x16`** | **`512` $\to$ `256`** | **`3`** | **`16`** | **`16`** | **`16,644` (`33,288 B`)** | **`0 B`** | **`43,700 B` (`+5,452 B`)** | **`0`** |

`V5B_Monarch3_16x32_16x16_16x16` (`16,384` Monarch weights across `256` core stage-3 channels + `260`-entry bias budget = `16,644` params = `33,288 B` FP16, ratio `33280/4161 ≈ 7.998×` vs. `512 × 260` dense FP16 weight matrix `266,240 B`, or `11115/1387 ≈ 8.014×` vs. `266,760 B` weights+bias) achieves full `512`-dim bank integration (`16×32` stage-1 contraction), `16`-path `rank = 16` subblock capacity verified exhaustively across all `512 × 256` `(i, j)` pairs (`min_paths_per_pair = max_paths_per_pair = 16`), `0` special-token collisions, `0 B` HBM vocab, and `0 B` SMEM spill (`43,700 B <= 49,152 B`). **Scope note (`Law-5`):** end-to-end GPU BPB retraining for `V5-A`/`V5-B` was not executed in this CPU-only verification track (`gpu_ablation_status = "CPU_ONLY_NOT_RUN_ON_GPU_NO_FABRICATED_BPB"`).

---

## 3. Deliverable 2 (`EXP-Q45c` / `SOP-16`): `MUX` Cor. 15 vs. `Al-Kāshī` Fixed-Radix & `DualAxisCarry` × `Floodgates` `SU(2)`

### 3.1 `MUX` Cor. 15 Precision Collapse vs. `Bayt al-Ḥikma` `Al-Kāshī` Fixed-Radix & `A_16 Block-H4`

Using exact meet-in-the-middle rational subset separation $m_S = \min_{c \in \{-1,0,1\}^S \setminus \{0\}} |\sum_{j=1}^S c_j p^{j-1} q^{S-j}|$ and `MUX` Cor. 15 criterion $\eta_S(u) = \frac{Su}{1-(S-1)u} < \frac{(q-p)m_S}{2(q^S - p^S)}$:

| Multiplex Depth $S$ | $m_S$ ($\rho=9/10$) | Half-Margin $\mathcal{E}/2$ | `BF16` ($u=2^{-8}$) | `FP16` ($u=2^{-11}$) | `FP32` ($u=2^{-24}$) | `Al-Kāshī` Fixed-Radix ($u=0$) | $\rho=1/3$ ($m_S=1$) `FP16` / `FP32` |
|---|---|---|---|---|---|---|---|
| `1..3` (`m_3=9`) | `1, 1, 9` | `>= 9/542` | `PASS` (`<= 3/254`) | `PASS` | `PASS` | **`PASS` (`eta = 0`)** | `PASS` / `PASS` |
| **`4`** | **`19`** | **`1/362`** | **`FAIL` (`4/253`)** | `PASS` (`4/2045`) | `PASS` | **`PASS` (`eta = 0`)** | `PASS` / `PASS` |
| **`5`** | **`171`** | **`171/81902`** | `FAIL` | **`FAIL` (`5/2044`)** | `PASS` | **`PASS` (`eta = 0`)** | **`PASS`** (`FAILS at S=6`) / `PASS` |
| `6..11` | `1539..272809` | `272809/137237880782` | `FAIL` | `FAIL` | `PASS` | **`PASS` (`eta = 0`)** | `FAIL` / `PASS` |
| **`12`** | **`860491`** | **`45289/75533733002`** | `FAIL` | `FAIL` | **`FAIL` (`12/16777205`)** | **`PASS` (`eta = 0`)** | `FAIL` / **`PASS`** (`FAILS at S=13`) |
| `13` | `1110709` | `1110709/14916268343342` | `FAIL` | `FAIL` | `FAIL` | **`PASS` (`eta = 0`)** | `FAIL` / `FAIL` |

- **Exact `A_16 Block-H4` Integer Accumulation (`u = 0`):** Exhaustively across all **`2,048` / `2,048`** zero-syndrome codewords of `A_16` — testing both raw `{0,1}^16` codewords and LSB-encoded 4-bit signed nibble blocks (`sum_c_sq = 704,512`, `sum_h4_sq = 2,818,048 = 4 × sum_c_sq`) — integer `H_4` accumulation has `parseval_mismatch_count = 0` and exact `0` sign-flip norm drift under symmetric scaling `±3/8`.
- **Al-Kāshī (`1427 CE`) Cubic Fixed-Point Contraction vs. Cliff:** Narrow-range fixed-point iteration ($p=45, x^*=1/10, L=1/1500$) contracts error strictly from `1/10` (`step 0`) to `214853054519110138741156187758986794107 / 3102863559971923828125000000000000000000000000000` (`step 3`), whereas outside the Lipschitz basin ($p=3, x^*=6/5$, slope `36/25 > 1`), error diverges from `1/20` to `38925189153799259561190888891321037235401 / 213986410758144000000000000000000000000000`.

### 3.2 `DualAxisCarry` (`beta_min`) × `Floodgates` `SU(2)` `O(1)` `S_fast` Loop Stability

1. **Projective `DualAxisCarry` EMA (16-step jitter around `1` + step-4 `-1/2` shock with `e_4 = 1`):**
   - **Carry-last ($\beta = 0$):** Full erasure shock `max_drift = 3/2`, `final_drift = 1/100`.
   - **`beta_min = 11/20 = 0.55` guarded:** Bounds shock to `max_drift = 72051579/68181865 ≈ 1.0568 < 3/2` and recovers to `final_drift = 41727253063229644769 / 7291086444631243669515 ≈ 0.00572` (also outperforming uniform `beta = 1` final drift `451/2100 ≈ 0.2148`).
2. **`Floodgates` `Ouro` (`88% → 33%`) Alignment Collapse vs. `SU(2)` Gauge Pullback & `O(1)` `S_fast`:**
   - Using exact rational Cayley generator $w = (0, 0, 1/12)$, the iterated `SU(2)` transport $U_r$ maintains exact unit norm $\|U_r\|^2 = 1$ across $r = 1..4$. Unaligned residual cosine decays steadily (`19873/21025`, `347821633/442050625`, `5039646554593/9294114390625`, `46551021702682753/195408755062890625`), whereas exact two-sided gauge pullback $U_r^{-1} (U_r x_{\mathrm{canon}} U_r^{-1}) U_r$ preserves cosine **`1`** identically across all $r = 1..4$!
   - `O(1)` `S_fast` Woodbury cache (`65,536 B` invariant of $T$) vs. $T = 8192, L = 6, R = 4$ FP16 KV caches gives **`1,536×`** reduction vs. `Ouro`/`Floodgates` (`100,663,296 B`) and **`6,144×`** reduction vs. `MoLT` (`402,653,184 B`). `S_fast` scalar contraction `gamma = 9/10` shrinks the 4-step Frobenius amplitude/norm factor to `(9/10)^4 = 6561/10000` (squared Frobenius energy `(9/10)^8 = 43046721/100000000`) vs. reverse decay amplitude `(91/50)^4 = 68574961/6250000` (squared energy `(91/50)^8`), consistent with `Floodgates` `[1/5, 3/2]` scalar clamping (`MERGE_LORA=0` required for `A_16` integer weight parity).

---

## 4. Formal `Lean 4.36.0` Theorem Inventory (`FourierMuxLoopOperators.lean`)

All `18` theorems are pure-`Init` (zero `Mathlib`), zero `sorry`/`admit`/`axiom`/`native_decide`, zero `Classical.choice` (`17` universally quantified `∀` theorems plus `1` concrete kernel-`decide` phase-boundary conjunction `#10`):

| # | Theorem Name | Axioms Used | What It Proves (Exact Lean Statement Scope) |
|---|---|---|---|
| 1 | `fwht_vs_finite_fourier_crossover_iff` | `[propext, Quot.sound]` | Positive `J*N` batch cancellation, exact two-way squared equivalence `(q m < p r) <-> (q^2 m^2 < p^2 r^2)`, and exact-sqrt `FWHT`/`FiniteFourier` crossover equivalence whenever `r^2 = 1 + m` |
| 2 | `fwht_256_finite_n_win_and_asymptotic_crossover` | `[propext, Quot.sound]` | `q^2 m^2 < p^2(1+m) -> q m < p r` (`1+m <= r^2`); `N=256` (`m=8, r=3`) win iff `8q < 3p` (`C=3: 8<9`, `C=4: 8<12`); `C=3` crossing at `m=10`; `C=4` crossing at `m=17`; `Thm 4.1` `2*5+1 > 9` (and `2w+1 > 9` for all `w>=0, w^2>=24`) |
| 3 | `finite_fourier_prop52_spectral_gap_strict` | `[propext, Quot.sound]` | `(L+1)m < (L+r)m` for all `L, m, r >= 1` satisfying `1 + L^2 <= r^2` |
| 4 | `lowrank_rank_defect_vs_monarch_full_reach` | `[propext, Quot.sound]` | Dimension deficit `0 < n - k` & `n - ab = 0`; linear-algebraic kernel existence (`∀` `k x n` matrix `rows`, `k < n => ∃ w != 0` with `dot(r, w) = 0`); exact `a x b = n` grid coverage `q*a + r = idx` |
| 5 | `v4_sp_flip_byte33_collision_universal` | `[]` (0 axioms) | Exact byte XOR `0x21` parity collision `[256..259] -> [33, 32, 35, 34]` universally across all bytes and all 256 coordinates |
| 6 | `v5_h2_bent4_orthogonal_zero_alias` | `[propext, Quot.sound]` | Universal `H_2` lift `(u, -u)·(v, v) = 0` & `2(u·v)`, plus explicit `H_2 x (H_256 x bent4)` `512`-dim Gram matrix `512 I_512` across all `512` symbols (`0..511`, including `0..259`) |
| 7 | `monarch_2stage_vs_3stage_path_multiplicity` | `[propext, Quot.sound]` | Parameter scaling (`4K^3 < 2K^4` for `K >= 4`), exact 2-stage path count `= 1` with universal `2×2` subblock minor `= 0` (rank `<= 1`), and exact 3-stage path count `= K` with nonzero `2×2` minor rank-lift witness |
| 8 | `v5_head_smem_fit_within_sm86_48kib` | `[propext, Quot.sound]` | Both `V5-A` (`35,508 B`) & `V5-B` (`43,700 B`) fit strictly below `49,152 B` static `sm_86` SMEM (`>= 4,428 B` headroom under upper budget) |
| 9 | `mux_cor15_cleared_margin_criterion_iff` | `[propext, Quot.sound]` | Derives `D = q^S - p^S > 0` (`0 <= p < q, S >= 1`); exact cleared equivalence `2SuD < (q-p)m(u_den-(S-1)u) <-> (2SD+(q-p)m(S-1))u < (q-p)mu_den`; `k`-scaling invariance; full `u` monotonicity (LHS grows AND RHS `(u_den-(S-1)u)` shrinks) & `m` monotonicity |
| 10 | `mux_rho9_10_collapse_boundaries_exact` | `[]` (0 axioms) | Concrete kernel-`decide` conjunction for `rho = 9/10` (`BF16 S=3/4`, `FP16 S=4/5`, `FP32 S=11/12`) and `rho = 1/3` (`FP16 S=5`, `FP32 S=12`) |
| 11 | `alkashi_balanced_radix_m_ge_1_universal` | `[propext, Quot.sound]` | For all `0 < p` and `q > 2p` (including `q >= 3p` at `rho = 1/3`), nonzero ternary combinations satisfy `|evalGeom| >= p^(S-1) >= 1` (`m_S = 1` at `p = 1`) universally |
| 12 | `alkashi_fixed_radix_zero_ulp_all_s` | `[propext, Quot.sound]` | Fixed-radix (`u_num = 0`) satisfies cleared Cor. 15 margin (`D = q^S - p^S`) for all `S >= 1, 0 <= p < q, m >= 1, u_den >= 1` |
| 13 | `a16_block_h4_parseval_and_flip_zero_drift` | `[propext, Quot.sound]` | Exact `H_4` `‖H_4 c‖^2 = 4‖c‖^2` Parseval norm & zero sign-flip squared-norm drift (`s -> -s`) |
| 14 | `alkashi_contractive_cubic_vs_repelling_cliff` | `[propext, Quot.sound]` | One-step cubic secant contraction (`x^2 + x y + y^2 < p` on `|x|, |y| <= R`, `3 R^2 < p`) vs. one-step repelling secant (`p(x - y) < x^3 - y^3`) |
| 15 | `dual_axis_carry_beta_min_prevents_carry_last_collapse` | `[propext, Quot.sound]` | `p_b = 0` yields `0 * D_prev = 0` (`gamma_t = 1`), while `0 < p_min <= p_b` preserves positive retention `p_min * D_prev` and strict one-step perturbation damping |
| 16 | `dual_axis_carry_convex_hull_invariance` | `[propext, Quot.sound]` | Projective weighted EMA preserves `[m, M]` invariant hull whenever `D_prev > 0, e_t > 0, q_b > 0, p_b >= 0` |
| 17 | `su2_cayley_unit_norm_and_gauge_pullback_exact` | `[propext, Quot.sound]` | Cleared Cayley squared norm `(d^2 + ||w||^2)^2`; left cancellation `conj(u)*(u*x) = ||u||^2 x`; two-sided conjugation pullback `conj(u)*((u*x*conj(u))*u) = ||u||^4 x`; `0 < ||u||^2` whenever `u != 0` |
| 18 | `s_fast_woodbury_contractivity_vs_reverse_decay_blowup` | `[propext, Quot.sound]` | Multi-step (`k >= 1`) vector squared-norm (`normSq`) contraction `(g_num^k * xs)` when `g_num < g_den` vs. expansion when `g_den < g_num` for any nonzero vector `xs` |

---

## 5. Verification Gates & Artifact Manifest

All verification commands below were executed on CPU only (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`), leaving `0` `.lake/`, `0` `__pycache__/`, `0` `.pyc`, and `0` symlinks in the track directory (`default_live_lean_runner` stages Lean files via `read_regular_bytes_capped` inside `tempfile.TemporaryDirectory()`, passing on `chmod -R a-w` read-only staging as well as RW):

- **`3× -I` Bit-Identical Parity:** `python3 -I -B`, `python3 -I -O -B`, and `python3 -I -B -W error` produce bit-identical `verify_s164_fourier_mux_loop.py --check --json` (`status="PASS"`), default text output, and `test_s164_fourier_mux_loop_r5.py` (`7/7 OK`, `0` skips, `0` bare asserts). *(Scope note: `T06` in `R5` exercises `default_live_lean_runner` subprocess staging/args via mocked `subprocess.run` for fast mutation sweeps, while cold real `Lean 4.36.0` builds are run via `--check --json --live-lean` on both RW and `chmod -R a-w` RO copies.)*
- **Live `Lean 4.36.0` Execution:** `verify_s164_fourier_mux_loop.py --check --json --live-lean` -> `PASS` on both RW and `chmod -R a-w` RO staging (`18/18` live axioms verified, `0` warnings, `0` `.lake/` residue).
- **Dual `aletheia mutate` (`--max-mutants 0`, `--workers 12`, `null_mutant=passed`, `truncated=false`):**
  - `s164_fourier_mux_loop_oracle.py`: **`30/30 (100.0%, 0 survivors)`** in `default` AND **`30/30 (100.0%, 0 survivors)`** in `--guards require_ok`.
  - `verify_s164_fourier_mux_loop.py`: **`138/138 (100.0%, 0 survivors)`** in `default` AND **`138/138 (100.0%, 0 survivors)`** in `--guards require_ok` (`336/336 = 100.0%` combined).
- **Full AST `BoolOp` (`and <-> or`) Sweep:** **`29/29 (100.0%, 0 survivors)`** (`oracle`: `9/9` + `verify`: `20/20`, `2` killed by timeout).
- **`aletheia audit-check`:** `python3 -B -m aletheia audit-check . --lean-audit --strict-ancestors --safety-scan --strict-unseeded --scripts verify_s164_fourier_mux_loop.py test_s164_fourier_mux_loop_r5.py --json` -> **`PASS`** (`0` Lean findings/notices, `0` unsafe scans, `0` unseeded RNG).

### Track File SHA-256 Manifest

| File | Bytes | SHA-256 Digest |
|---|---|---|
| `FourierMuxLoopOperators.lean` | `84447` | `8236cba7eabda4df47b560646f5670b7e3b3e29d995f7770c5a7c9028164b0f9` |
| `lake-manifest.json` | `90` | `79ff7de19171b1eade041516df1cd0ac49262d36237a09376beca08243e8e449` |
| `lakefile.toml` | `146` | `acf4019f43647a7799b741738425b0d73123541d5374f9643508cc1e7860bbdf` |
| `lean-toolchain` | `25` | `582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a` |
| `s164_fourier_mux_loop_certificate.json` | `22474` | `c17ed72e517dfc00b77397c52ed091e2a6a7e4201e3aca7ad6c3939af8b0a6d7` |
| `s164_fourier_mux_loop_oracle.py` | `39822` | `cd50c2fe2f2ea8d054d306d9e3a8e2b79c473861a258f43d96531911e9bee58c` |
| `test_s164_fourier_mux_loop_r5.py` | `44621` | `5257b2c7bece225eb98a03d9f3224bf04206504242141c4da7e95ad57f923f5c` |
| `verify_s164_fourier_mux_loop.py` | `34209` | `c419b685697d98bee76ddca1a4e90020ccf718f8768d4a5538681b5b1e0abfd9` |
