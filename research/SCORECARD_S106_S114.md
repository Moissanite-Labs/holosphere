# Honest Law-4 Research Scorecard: Tracks `S106–S114`

Every empirical number below comes from preregistered, strictly held-out, `mlos` Zone-B hardware-stamped RTX 3080 runs (`WEIGHTS=0`, `HF_HUB_OFFLINE=1`, zero HARKing) or pure-`Init` `Lean 4.36.0` kernel-checked proofs. Falsified hypotheses are preserved alongside confirmed wins so HoloSphere builds exclusively on verified mechanisms.

---

## 1. Track `S106` Phase 3 (`Foundation-0/aletheia#102`): `H5a–H6c`

| Hypothesis | Verdict | Verified Quantitative Evidence | Architectural Takeaway for HoloSphere |
| :--- | :---: | :--- | :--- |
| **`H6c`** (Native Hopf-$S^3$ Residual Micro-LM) | **`CONFIRMED`** | Exact `10,818,432`-param parity (`0` extra params): Hopf seeds **`2.6612` & `2.6197`** vs flat **`2.9348` & `3.0295`** Val BPB (**`-0.342 BPB` / `-11.5%`**); 3-bit PTQ Hopf (`2.663`) still beats FP32 flat (`2.982`). Unfused PyTorch ops ran `2.57×` slower (`55.4s` vs `21.6s`). | **Adopted as flagship `hopf_s3` residual mode** (`fused_spherical_step`), fused into `float4` registers to erase the `2.57×` eager penalty. |
| **`H5a`** (Zero-Bit Flow-SphereMap) | **`FALSIFIED`** | Algebraic Law-5 obstruction: Sinkhorn column normalization gauge-annihilates right-diagonal flow scaling identically. | Eliminated from design. |
| **`H5b` / `H6a`** ($S^3$ Polar-Curvature Depth Foliation) | **`FALSIFIED`** *(scale-emergent)* | Polar beats Equatorial on `3/3` models; slashes `Qwen2.5-3B` WikiKL `-28.8%` (`8.8 SE`) & Python KL `-60.8%` (`18.6 SE`), but only `1/3` models cleared `>= 10%` locked gate. | Kept as optional `width_profile` (`polar` / `equatorial` / `cylinder`) for controlled scaling tests. |
| **`H6b`** (Factored Hopf `EQ(S^2) × S^1` Quantizer) | **`FALSIFIED`** | Exact $U(1)$ fiber equivariance (`0.999998–1.000000`), but directional SNR (`10.778 dB`) trailed direct `EQ(S^3)` (`13.487 dB`). | Use full $S^3$ / $E_8$ / $A_{16}$ lattice quantization (`siliquant_bridge.py`), not factored `S^2 × S^1`. |

---

## 2. Track `S107` (`Foundation-0/aletheia#105`): `H7a`, `H7b`, `H7c` (`111` Live Runs)

| Hypothesis | Verdict | Verified Quantitative Evidence (`3` Seeds `[42, 137, 2026]`) | Architectural Takeaway for HoloSphere |
| :--- | :---: | :--- | :--- |
| **`H7a`** ($S^3$ Width Foliation on `Std` vs `nGPT`) | **`FALSIFIED`** | `0/4` BPB/KL gates passed. Uncovered **`W10-OBS-30`** geometry polarity flip: `S3-Equatorial-Std` hurts Euclidean (`+3.40% z=+2.61`), while `S3-Equatorial-nGPT` reached lowest overall BPB (`2.4652` vs `2.4907`, `-1.02% z=-1.53`). | Hyperspherical normalization (`nGPT`) outperforms Pre-Norm across all `5/5` schedules (`-1.7%` to `-5.9%`). |
| **`H7b`** (Data-Dependent Path-Ordered $SU(2)$ Holonomy Scan) | **`PARTIAL`** | **`T=2048` (`4×` zero-shot extrapolation):** `SU2-Holonomy-Scan-nGPT` (`2.7462 BPB`) beats `RoPE-2D-nGPT` by **`-7.37% (z=-13.32)`**, `Block4-RandOrtho` by **`-8.13% (z=-18.17)`**, `RoPE-2D-Std` by **`-15.38% (z=-12.30)`**, and `LieRE-4D` (`N0`) by **`-4.93% (z=-3.91)`**. Caveats: `+1.22%` at `T=512`, chance-level QK-only $S_3/A_5$, **`3.913×`** eager wall-clock slowdown. | **Core of Project HoloSphere (`H14a` + `H14c`):** fuse scan into zero-spill `sm_86` warp shuffles (`<= 1.15×` wall-clock) AND extend to covariant `(Q, K, V)` + `Pi_t^{-1} O_t` to test $S_3/A_5$ rescue! |
| **`H7c`** (Clifford-Torus / Hopf Looped Core) | **`PARTIAL`** | Fires `R=3` BPB strongly (`-5.29%` vs `FlatLoop-Std`, **`-9.44% z=-42.46`** vs `NoClock`, **`-12.53%`** vs `Unlooped-16L`), but `SphereLoop-RandRotClock` matches (`2.6378` vs `2.6621`) showing **general orthogonal pass-symmetry breaking** with Hopf winning on `O(d)` cost (`+0.0156%` vs `+1.33%` FLOPs). Uncovered **`W10-OBS-31`**: $\phi_5 = 10\pi/11 \approx \pi$ antipodal collapse at `R_test=6` (`8.07 BPB`). | **Fixed in HoloSphere (`safe_hopf_acute` & `cayley_acute`):** constrain $\phi_r \in [0, \phi_{\max})$ with $\phi_{\max} < \pi/2$ (or rational $|t_r| < 1 \implies \cos\phi_r > 0$) so unseen `R_test` NEVER sign-flips! |

---

## 3. Track `S108` (`Foundation-0/aletheia#106`): `H8a`, `H8b`, `H8c` (`51` Live Runs)

| Sub-Track | Verdict | Verified Quantitative Evidence | Architectural Takeaway for HoloSphere |
| :--- | :---: | :--- | :--- |
| **`H8a`** (Hybrid Kronecker-Spherical Embeddings) | **`FALSIFIED`** | `28.98×` embedding compression (`1.57M -> 54.3k`), but loses `+5.28%` to `+8.55%` BPB vs `flat_bpe`. | Keep full untied/tied spherical token embeddings in flagship models. |
| **`H8b`** (2nd-Order Geodesic Turning Byte Patching) | **`FALSIFIED`** | Boundary collapse (`avg_plen = 1.020`, `4.9569 BPB`) because consecutive byte tangents on $S^{95}$ are nearly orthogonal. | Use standard BPE/byte windows or entropy-based boundaries (`BLT` `4.2223` beat `H-Net` `4.4584`). |
| **`H8c` Part 1** (Adaptive Loop Pacing `r_bar=2.0`) | **`FALSIFIED`** | Exact `sum(r)=2N` and contractive `rho=0.88–0.97` hold, but zero BPB gain vs uniform `r=2` (`-0.026% z=-0.18`). | Prefer clean uniform `r` loops with acute-cone Safe-Hopf clocks unless task-conditioned depth proves useful. |
| **`H8c` Part 2** (`Markov-MTP8` Coupled Draft Head) | **`CONFIRMED`** *(vs Parallel)* | Beats `Parallel-MTP8` by **`-19.34% (z=-63.31)`** (`3.7008` vs `4.5879`) and cuts unrolled exposure-bias drift **`-12.37%`** vs `LocalAR` (`5.2840` vs `6.0296`). | **Integrated as flagship optional `MarkovMTP8Head` (`holosphere/models/heads.py`).** |
| **`H8c` Part 3** (`nullmodem` Block-8 $E_8$ Cell Bus) | **`PARTIAL`** | **`99.74%`** field acc / **`98.96%`** exact acc in `32` payload bits (`100%` SHA-256), but single-state `320`-bit header+hash totaled `352 > 144` (`ABIOS/3`). | **Upgraded to `BatchedNullModemCell` (`nullmodem_e8.py`):** $M \ge 4$ amortizes total wire to `112` bits/state (`< 144` `ABIOS/3`). |

---

## 4. Tracks `S109–S112`: Geodynamic Shells, Weft Governance, & Lean 4 Mechanization

| Track | Verdict | Verified Quantitative / Formal Evidence | Architectural Takeaway for HoloSphere |
| :--- | :---: | :--- | :--- |
| **`S109` (`H9a–H9d`)** | `H9a CONFIRMED`, `H9b–d FALSIFIED` | `13.95×` neural unpredictability jump on raw-deflate (`0.00%` R2 exact inversion); `MHPlate` `100.00%` English/Arabic IR accuracy; `0/18` `CS-21` leaks; Seed 42 zero-oracle modifier recovery `93.98%` at `Y=0.25`. | Keep `VectorProximalPoloidalBingham` (no post-renorm) & Darboux dark-mode diagnostics cleanly available. |
| **`S110` (`H10a`)** | **Live (`H10a` done; `H10b` running)** | `Cayley2x2` cuts `T=512` BPB `-3.94%` (`2.3033` vs `2.3976`) and `T=1024` (`2.5610` vs `2.6592`); `BinghamProx` (`M1-H1=1.874`) and `DarcyEMA` (`1.621`) push loop quantization drift `< 2.0`. | Included as optional `use_cayley`, `use_bingham`, `use_darcy` switches in `HoloSphereBlock`. |
| **`S111` (`Weft-Geosphere`)** | **CPU Preflight `PASS`** | Frozen Core SHA-256 + narrow Shell-3 plates + Composed Triple Gate (`StaticProof ∧ Receipt ∧ Holdout`) + `~20 KiB` Merkle ledger + `O(1)` atomic rollback. | Integrated as optional `WeftAuditShell` (`holosphere/models/weft_shell.py`). |
| **`S112`** (`aletheia#103`) | **`VERIFIED` (`34/34` Lean thms)** | Pure-`Init` `Lean 4.36.0` (`0` Mathlib, `0` `Classical.choice`, `0` `sorryAx`) across all 5 Natural-Systems Portals; extended in `holosphere/lean/HoloSphereInvariants.lean` for $SU(2)$ scan & acute-cone clocks. | Machine-checked algebraic guarantees embedded directly in repo (`lean/`). |

---

## 5. Track `S114` (`PREREG-S114`): Upcoming Live RTX 3080 Benchmarks

- **`H14a` (Fused `sm_86` CUDA/PTX + Triton Parity & Speedup):** `<= 1e-4` FP32 / `<= 5e-3` BF16 parity, `0 B` register spills (`ptxas -v`), collapsing eager `H7b`'s `3.913×` slowdown down to `<= 1.15×` of `RoPE-2D-nGPT` (`>= 3.4×` speedup over eager `H7b`).
- **`H14b` (Unified Kernel-Isomorphic Entity):** Equal-wall-clock (`CS-08`) & `mJ/tok` evaluation on `T=512` and `4×` `T=2048` zero-shot extrapolation vs `RoPE-2D-Std` and `RoPE-2D-nGPT`.
- **`H14c` (Full Covariant $(Q,K,V)$ Transport + $\Pi_t^{-1} O_t$):** Testing whether zero-extra-scan covariant value transport breaks decisively above `H7b`'s QK-only chance floor on non-Abelian $S_3$ and $A_5$ OOD state tracking.
