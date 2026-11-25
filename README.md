# HoloSphere (`Moissanite-Labs/holosphere`)

**A Kernel-Isomorphic $SU(2) \cong S^3$ Hypersphere Transformer Unifying Bare-Metal Ampere `sm_86` Quaternion Kernels, Pure-`Init` Lean 4 Invariants, and Verified Natural-Systems Architecture Research (`S106–S114`).**

> **Founding Thesis ([`docs/MANIFESTO.md`](docs/MANIFESTO.md)) — Ending the False Dualism Between Architecture and Kernel:**  
> One unit quaternion $q = (w, x, y, z) \in SU(2) \cong S^3$ **is** one 128-bit `float4` GPU register quad (`LDG.E.128`). Instead of bolting geometric layers onto eager PyTorch graphs (which paid a `3.913×` HBM round-trip slowdown in `Track S107 H7b`), **HoloSphere** fuses the kernel and the transformer into **one silicon-isomorphic mathematical entity** — lowering every layer onto zero-spill `float4` warp-shuffle primitives with machine-checked pure-`Init` `Lean 4.36.0` proofs (`lean/`), `Moissanite-Labs/silikernels` export compatibility, and full 3-mode isolated CPU parity.

---

## The Silicon–Geometry Isomorphism (`NVIDIA Ampere sm_86`)

| Mathematical Primitive ($SU(2) \cong S^3$) | Bare-Metal `sm_86` Silicon Primitive | Algebraic Identity / Hardware Guarantee (`FP32 <= 5.87e-6`) |
| :--- | :--- | :--- |
| **1 Unit Quaternion** $q=(w,x,y,z) \in SU(2)$ | **1 `float4` (`LDG.E.128`) Register Quad** | 128-bit load -> 4 FP32 regs; Hamilton product = **16 FMAs in registers** (`0` HBM, `0` SMEM); Euler 4-square norm conservation + SFU-free `cayley_su2` |
| **Associative Prefix Holonomy** $\Pi_t = \prod_{s \le t} q_s$ | **1 Warp (`32` lanes) × 5 `__shfl_up_sync`** | Mathematically associative 5-step Hillis–Steele scan (**`27/27` kernels `0 B` spills**), algebraically equivalent to $O(1)$-state streaming recurrence via Lean-checked $(pq)r = p(qr)$ |
| **Non-Abelian Reverse-Mode Adjoint** $\partial\mathcal{L}/\partial u_t$ | **5 `__shfl_down_sync` FP32 Additions** | Exact algebraic pullback $S_t = \sum_{j \ge t} \operatorname{Im}(g_j \otimes \overline{\Pi_j})$, $\partial\mathcal{L}/\partial u_t = \overline{\Pi_{t-1}} \otimes (0, S_t) \otimes \Pi_t$ (**0 quaternion multiplies in backward scan!**) |
| **Covariant $(Q,K,V)$ Transport (`H14c`)** | **Same Register Quad $\Pi_t$ + $\overline{\Pi_t}$** | $\tilde{Q}_t,\tilde{K}_t,\tilde{V}_t = L(\Pi_t)(Q_t,K_t,V_t)$ and $O_t^{\text{cov}} = L(\Pi_t^{-1}) O_t = \sum_s w_{ts} L(\Pi_t^{-1}\Pi_s) V_s$ (**0 extra scans**, block-4 tied `sqk`) |
| **Safe-Hopf Loop Clocks (`W10-OBS-31`)** | **SFU-Free Rational Cayley FMAs** | `cayley_acute` & `cayley_decay` ($\phi_r \to 0$): $c = \frac{1-t^2}{1+t^2} > 0,\; s = \frac{2t}{1+t^2}$ ($|t| < 1$): zero `sin`/`cos` SFU ops, permanently forbids `R_test = 6` antipodal flip |
| **Quantized Weight & Wire Bridge** | **`siliquant` $A_{16}$ / $E_8$ Register Bridge** | `1 LOP3 + 5 POPC` `[16,11,4]` syndrome + batched Block-8 $E_8$ `RAM==WIRE==HASH` cells (`112` bits/state at $M=4$) |

---

## Architecture Overview

```text
Input x_t in S^{d-1}
  │
  ├─► W_omega(x_t) ──► omega_t in su(2) (R^3 per 4D block)
  │                      │
  │                      ▼
  │         [Kernel 1] su2_holonomy_transport (float4 = 1 SU(2) quad)
  │         • du_t = exp(alpha * tanh(omega_t)) in SU(2)
  │         • Forward:  5-step warp __shfl_up_sync Hillis–Steele Pi_t = du_0 * ... * du_t
  │         • Backward: pure 4-float additive suffix sum S_t = sum_{j>=t} g_j * conj(Pi_j)
  │         • In-register transport: Q' = L(Pi) Q,  K' = L(Pi) K,  V' = L(Pi) V  (H14c)
  │                      │
  ├─► Causal Flash / SDPA(Q', K', V') ──► [Kernel 2] su2_inverse_untransport: O = L(Pi^{-1}) O'
  │   (Exact interval holonomy sum_s w_{ts} L(Pi_t^{-1} Pi_s) V_s at ZERO extra scan cost!)
  │                      │
  ├─► [Kernel 3] fused_spherical_step
  │   • H6c Native Hopf-S^3 residual: g * L(u) h  (CONFIRMED -11.5% BPB, 0 extra params)
  │   • Safe SLERP (nGPT) with smooth sqrt(||x||^2 + eps^2) antipodal floor
  │   • Optional S107/W10-OBS-31 non-antipodal acute-cone / rational Cayley Safe-Hopf loop
  │   • Optional S110 Beltrami [C,K]=0 2x2 Cayley, Poloidal Bingham, & Darcy EMA
  │                      │
  └─► Output Heads & Governance
      • Optional S108 H8c2 Markov-MTP8 coupled multi-token draft head (CONFIRMED -19.34% vs Parallel)
      • Optional S111 Weft-Geosphere Shell-3 Merkle audited live updates (StaticProof ∧ Receipt ∧ Holdout)
      • Optional S108 H8c3 Batched nullmodem Block-8 E_8 RAM==WIRE==HASH cell bus
```

---

## Verified Empirical & Formal Foundation (`S106–S114`, Honest Law-4 Scorecard)

HoloSphere retains **only** mechanisms that survived preregistered RTX 3080 (`mlos` Zone-B hardware-stamped) experiments and explicit falsification audits — and transparently documents every falsified hypothesis and structural obstruction in [`research/SCORECARD_S106_S114.md`](research/SCORECARD_S106_S114.md).

| Track | Mechanism / Experiment | Literal Locked Verdict | Headline Verified Result (`WEIGHTS=0`, Zone-B Stamped) | Provenance |
| :--- | :--- | :---: | :--- | :--- |
| **`S106 H6c`** | Native Hopf-$S^3$ coupled quaternion residual (`0` extra params) | **`CONFIRMED`** | **`-11.5%` Val BPB** (`2.6612` & `2.6197` vs flat `2.9348` & `3.0295`) at exact `10,818,432`-param parity; even 3-bit PTQ Hopf (`2.663`) beats FP32 flat (`2.982`). *(Caveat: unfused eager ops `2.57×` slower — motivating HoloSphere fusion.)* | `Foundation-0/aletheia#102` |
| **`S107 H7b`** | Path-ordered $SU(2)$ holonomy scan (`T=512` train $\to$ `T=2048` zero-shot) | **`PARTIAL`** | **`-7.37%` (`z=-13.32`)** `4×` `T=2048` BPB vs `RoPE-2D-nGPT`, **`-8.13%`** vs `Block4-RandOrtho`, **`-15.38%`** vs `RoPE-2D-Std`, **`-4.93%`** vs `LieRE-4D` (`N0`). *(Eager `3.913×` slowdown & QK-only $S_3/A_5$ floor motivated `H14a`/`H14c`!)* | `Foundation-0/aletheia#105` |
| **`S107 H7c`** | Orthogonal per-pass looped core (`2+4×3+2 = 16` eff. layers) | **`PARTIAL`** | **`-9.44%` (`z=-42.46`)** vs unclocked `SphereLoop-NoClock`, **`-12.53%`** vs `Unlooped-16L` (`2×` params). Uncovered **`W10-OBS-31`** (`R_test=6` antipodal flip $\phi_5 = 10\pi/11 \approx \pi$) — fixed here by **`acute_cone` / `cayley_acute`**! | `Foundation-0/aletheia#105` |
| **`S108 H8c2`** | `Markov-MTP8` coupled multi-token draft head | **`CONFIRMED`** *(vs Parallel)* | **`-19.34%` (`z=-63.31`)** TF BPB vs `Parallel-MTP8` (`3.7008` vs `4.5879`) and **`-12.37%`** less unrolled exposure-bias drift vs `LocalAR`. | `Foundation-0/aletheia#106` |
| **`S108 H8c3`** | Block-8 $E_8$ `nullmodem` `RAM==WIRE==HASH` cell bus | **`PARTIAL`** | **`99.74%`** field acc / **`98.96%`** exact acc in **`32` payload bits** (`100%` SHA-256). Batched framing (`BatchedNullModemCell`, $M \ge 4$) amortizes total wire to `112` bits/state `< 144` (`ABIOS/3`). | `Foundation-0/aletheia#106` |
| **`S109 H9a–d`** | Raw-deflate learnability obstruction & TerraCore Bingham plates | **`H9a CONFIRMED`**, `H9b–d FALSIFIED` | `13.95×` neural unpredictability jump on raw-deflate (`0.00%` R2 exact decompression); `MHPlate` `100.00%` English & Arabic IR recovery; `0/18` `CS-21` leaks. | `Foundation-0/aletheia#107` |
| **`S110 H10a/b`** | Pruned 4-operator Geosphere-Net (`42/42` live runs) | `H10a PARTIAL`, `H10b FALSIFIED` | **`H10a_Shell1_Cayley2x2` `#1/14`** (`2.3033` vs `2.3976` `-3.94%`, `z=+3.82`, `[C,K]=0`); `H10b_Geosphere_Full` falsified (`2.6976`, `0/4` LOSO). | `Foundation-0/aletheia#109` |
| **`lean/`** *(+ external `S112`)* | Pure-`Init` `Lean 4.36.0` Algebraic Invariants over `QuatZ` ($\mathbb{Z}$) | **`36/36` In-Repo `VERIFIED`** | **`36/36`** in-repo public theorems in `lean/HoloSphereInvariants.lean` (`0` Mathlib, `0` `Classical.choice`, `0` `sorryAx`, `0` warnings; exact algebraic identities over $\mathbb{Z}$), extending `34` external `S112` theorems (`aletheia#103`). | `lean/` · `Foundation-0/aletheia#103` |
| **`S114` (`H14a` + `V2`)** | Fused `sm_86` `SU(2)` Holonomy + `Cayley2x2` + `cayley_su2` | **`H14a CONFIRMED` + `V2` Live** | **`27/27` `0 B`-spill `sm_86` kernels**: **`0.6313×` `RoPE-2D-nGPT`** (`71.57 ms`, **`-13.4%` VRAM**, **`-40.1%` `mJ/tok` at `1.431 mJ/tok`**, `6.35×` vs eager `H7b`); `Cayley2x2` **`9.77×`**, `Hopf-S^3+Cayley2x2` **`28.58×`**; measured FP32 parity `<= 5.87e-6` (V1 fiber-major `348.37 GB/s` = `97.69%` Triad; V2 zero-permute `329.40 GB/s` = `92.37%` Triad). | `Moissanite-Labs/silikernels#3` |
| **`S114` (`H14b`)** | Equal-Wall-Clock (`CS-08`) + Equal-Step `15M` Showdown (`7` arms × `3` seeds) | **`CONFIRMED`** | **`3/3` unlooped `HoloSphere-V2` arms pass ALL gates (`C1/C2/C3`)**: **`#1` SFU-free `CayleySU2`** **`-11.31%`** Eq-WC `T=2048` (`t=-15.43, p_t=0.00418`; **`-20.16%`** vs `RoPE-2D-Std`), **`-4.80%`** Eq-WC `T=512` (`p_t=0.00228`), **`0.80×`** `mJ/tok`; **`#2` `Stage1a-Cayley2x2`** **`-10.81%`** (`p_t=0.00155`, **`#1` Eq-Step `-7.40%` `p_t=0.000079`**); **`#3` `SU2-Unlooped`** **`-10.00%`**, **`0.76×`** `mJ/tok`. `RoPE+MarkovMTP8` control `+0.66%` (zero MTP confound). | `research/SCORECARD_S106_S114.md` |
| **`S114` (`H14c`)** | Full Covariant $(Q,K,V)$ + $\Pi_t^{-1} O_t$ on Non-Abelian $S_3$ / $A_5$ (`5` arms × `3` seeds) | **`FALSIFIED`** | `Full-Covariant-QKV` ranked **`#1/5` on $S_3$ (`35.30%` OOD, `+15.81 pp` vs `RoPE` `p_t=0.00943`)**, below locked `>= 50%` gate (`+4.24 pp` vs `QK-Only`); $A_5$ remained near chance (`2.58%` vs `2.42%` `RoPE`). Directly confirms the **softmax convex-hull group-closure obstruction**. | `research/SCORECARD_S106_S114.md` |

---

## Repository Layout

- **`holosphere/geometry/`** — Exact $SU(2) \cong S^3$ quaternion algebra (`su2.py`, incl. SFU-free pure-rational `cayley_su2`), Hopf fibration / `H6c` residual / `acute_cone` & rational `cayley_acute` Safe-Hopf clocks / $S^3$ width schedules (`hopf.py`), Beltrami `[C,K]=0` 2×2 Cayley blocks with exact closed-form reverse-mode adjoint (`beltrami.py`), $SO(k)$-equivariant Poloidal Bingham (`bingham.py`), and Darboux dark-mode diagnostics (`darboux.py`).
- **`holosphere/telemetry.py`** — Pure-stdlib one-way `SiliSense -> HoloSphere` step-boundary telemetry & tuning contract (`HoloSphereStepTelemetry`, `HoloSphereTuningProfile`; zero `torch`, zero sensor I/O).
- **`holosphere/kernels/`** — Unified typed dispatch (`dispatch.py`), custom reverse-mode additive suffix-sum PyTorch adjoint (`torch_ops.py`), Ampere `sm_86` CUDA/PTX bridge (`cuda_bridge.py`, `csrc/`), Triton bridge (`triton_bridge.py`), and `siliquant` $A_{16}$/$E_8$ register quantization bridge (`siliquant_bridge.py`).
- **`holosphere/models/`** — Flagship `HoloSphereTransformer` (`transformer.py`, with opt-in batch-uniform variable-depth `r ∈ {1..4}` & step telemetry hooks), kernel-isomorphic `HoloSphereBlock` & `CovariantSU2Attention` (`blocks.py`), named presets across `S106–S114` plus Phase-2 surgical `s114_stage1a_cayley` / `s114_stage1b_cayley_vardepth` (`config.py`), matched baselines (`controls.py`: `RoPE-2D`, `LieRE-4D`, `Block4-RandOrtho`, `DiagonalDarcyEMA`), `MarkovMTP8Head` / `ParallelMTP8Head` / `LocalARByteDecoder` (`heads.py`), `WeftAuditShell` Composed Triple Gate + Shell-3 Merkle ledger (`weft_shell.py`), and batched `nullmodem` $E_8$ cell bus (`nullmodem_e8.py`).
- **`holosphere/benchmarks/` & `benchmarks/`** — Deterministic $S_3$ / $A_5$ non-Abelian state tracking & Dyck bracket probes (`tasks.py`), `PREREG-S114` CPU preflight + RTX 3080 harness (`prereg_s114_suite.py`), and `PREREG-S114` Amendment-1 `S107`/`S108` convergence harness (`benchmarks/s114_s107_s108_convergence.py`).
- **`docs/MANIFESTO.md`** — Founding Silicon–Geometry Isomorphism Manifesto (`float4` = $SU(2)$ quaternion, pure-additive backward adjoint, rational Cayley Safe-Hopf law, and `PREREG-S114` falsifiable kill gates).
- **`lean/`** — Pure-`Init` `Lean 4.36.0` mechanized algebraic invariants (`HoloSphereInvariants.lean` — `36` public theorems across all 5 pillars, `0` Mathlib, `0` `Classical.choice`, `0` `sorryAx`, `0` warnings; verify via `make lean-verify`).
- **`research/`** — Complete verified Law-4 scorecard ([`SCORECARD_S106_S114.md`](research/SCORECARD_S106_S114.md)), deep kernel-transformer isomorphism specification ([`ARCHITECTURE_KERNEL_ISOMORPHISM.md`](research/ARCHITECTURE_KERNEL_ISOMORPHISM.md)), and Phase-2 `S110`/`S111`/`S115` surgical integration spec ([`S110_S111_SURGICAL_INTEGRATION.md`](research/S110_S111_SURGICAL_INTEGRATION.md)).
- **`tests/`** — Assertion-free 3-mode isolated `-I` CPU test suite (`-I -B`, `-I -O -B`, `-I -B -W error`), FP64 `gradcheck`, exact algebraic checks, and repo hygiene gate.

---

## Quickstart (CPU-Safe, Zero GPU Required)

```bash
# Editable install (no isolated build network fetch required)
python3 -m pip install --no-build-isolation --no-deps -e .

# Run full 3-mode isolated CPU verification suite (-I -B, -I -O -B, -I -B -W error)
make test-3mode

# Run PREREG-S114 CPU preflight (FP64 gradcheck + covariant QKV + acute-cone + CS-08 parity)
make bench-cpu

# Verify all 36 pure-Init Lean 4.36.0 invariants (requires lean 4.36.0)
make lean-verify
```

```python
import torch
from holosphere import HoloSphereTransformer, build_preset_config

cfg = build_preset_config("s114_entity", d_model=128, n_heads=4, base_d_ff=352, n_pre=1, n_core=2, n_suffix=1)
model = HoloSphereTransformer(cfg)
tokens = torch.randint(0, cfg.vocab_size, (2, 32))
logits, trace = model(tokens, r_loops=6, return_trace=True)
```

---

## License

Licensed under the **Apache License 2.0** (matching [Moissanite-Labs/silikernels](https://github.com/Moissanite-Labs/silikernels)). See [`LICENSE`](LICENSE).
