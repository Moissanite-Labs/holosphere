# Phase-2 Surgical Integration Spec: `S110` (`Cayley2x2` + Variable-Depth ACT), `S111` (`Weft`), & `S115` (`SiliSense`) Into Project HoloSphere

> **Scope:** Additive Phase-2 architectural & bare-metal `sm_86` kernel design specification for `Moissanite-Labs/holosphere` (and companion `Moissanite-Labs/silikernels` / `Moissanite-Labs/silisense` boundaries).  
> **Empirical Anchors:** `Track S110` (`42/42` live RTX 3080 `15M`-parity runs + `2` `sm_86` microbenches, `3/3` clean Zone-B receipts, `nan_steps = 0`, `WEIGHTS = 0`), `Track S111` (`Weft-Geosphere` Composed Triple Gate + Shell-3 Merkle ledger), `Track S114` (`9/9` `0 B`-spill `sm_86` kernels, `PREREG-S114`), and `Track S115` (`SiliSense` chip-level feedback).

---

## 1. Surgical Composition Blueprint (Anti-Kitchen-Sink Law 5)

### 1.1 Exact Root-Cause Diagnosis of `Track S110`: Why `Shell1_Cayley2x2` Won (`#1/14`) While `Geosphere_Full` Falsified (`0/4` LOSO)

Live `Track S110` (`15M ± 0.27%` parity, `3` seeds `[42, 137, 2026]`, bit-identical offline multidomain corpus) produced a decisive separation between **surgical orthogonal composition** and **kitchen-sink stacking**:

| Arm (`15M` Parity) | `T=512` BPB | `T=1024` BPB (`Δ`) | `R=6` BPB | `M1-H1` `Δ3/Δ1` | `3b` PTQ KL | Eager Step Overhead | Literal Role / Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`H10a_Shell1_Cayley2x2`** | **`2.3033 ± 0.0266`** | **`2.5610 (+0.258)`** | `6.6218` | `2.035` | `0.03685` | `+15.55%` | **#1 of 14 arms** (`-3.94%`, `z=+3.82`, `p_Holm=0.000262` vs `Baseline_Loop`; beats all `4/4` `N0` baselines `-3.91%..-11.00%`); `||[C,K]||_F = 0.0`, `m0..m4` drift `<= 1.2e-7` |
| `H10a_Shell2_BinghamProx` | `2.3782 ± 0.0363` | `2.6079 (+0.230)` | `2.9677` | `1.874` | `0.03284` | `+1.41%` | `INCONCLUSIVE` BPB (`z=+1.03`), cuts `M1-H1` `-14.28% (z=+2.25)` |
| `H10a_Baseline_Loop` | `2.3976 ± 0.0250` | `2.6592 (+0.262)` | `2.9124` | `2.186` | `0.02873` | `0.00%` | Matched looped baseline |
| `H10b_LOSO_NoACTDraft` | `2.4542 ± 0.0605` | `2.7005 (+0.246)` | `5.6297` | **`1.012`** | `0.03682` | — | `Cayley + Bingham + Darcy`: near-unity `M1-H1`, but `R=6` still blows up without variable-depth ACT |
| `H10b_LOSO_NoDarcyEMA` | `2.4721 ± 0.0423` | `2.6505 (+0.178)` | **`2.9389`** | `1.801` | `0.04533` | — | Removing `DarcyEMA` recovers **`-0.2255 BPB` (`z=-6.82`)** AND variable-depth ACT cures `Cayley` `R=6` blowup (`6.6218 -> 2.9389`)! |
| `H10a_Shell4_ACTPlume_MTP8` | `2.5342 ± 0.0280` | `2.7527 (+0.219)` | **`2.8646`** | `2.066` | `0.02955` | `+10.04%` | `FALSIFIED` BPB (`z=-6.74`), lowest `R=6` & spike ASR (`0.2375`) |
| `H10a_Shell3_DarcyEMA` | `2.5414 ± 0.0118` | `2.6988 (+0.157)` | `3.5298` | `1.621` | `0.02615` | `+20.97%` | `FALSIFIED` BPB (`z=-5.63`), cuts `Δ1024-512` `-39.85%` & `M1-H1` `-25.84%` |
| `H10b_Geosphere_Full` | `2.6976 ± 0.0229` | `2.8080` (**`+0.110`**) | `3.0678` | `1.367` | `0.03558` | `+62.31%` | **`FALSIFIED`** (`+0.3943 BPB` worse than `Cayley2x2`, loses to all `4/4` `N0` baselines `z=-6.02..-23.46`, `0/4` LOSO positive) |

Three concrete algebraic conflicts explain `H10b`'s `+0.3943 BPB` regression:

1. **Isometry vs. Dissipation Conflict (`Cayley2x2` $\perp$ `DarcyEMA`):** `Cayley2x2` is an exact norm- and Beltrami-helicity-preserving rotation ($Q^T Q = I_d$, $\|Q x\|_2 = \|x\|_2$) that sharpens channel-pair feature mixing (`-0.0943 BPB`). `DarcyEMA` ($s_t = (1-\alpha) s_{t-1} + \alpha x_t$) is an **irreversible dissipative low-pass filter** ($\|s_t\| < \|x_t\|$) that smears high-frequency token-sharp states — explaining why removing `DarcyEMA` recovered `-0.2255 BPB` (`z=-6.82`) while `DarcyEMA` alone hurt BPB by `+0.1438` (`z=-5.63`).
2. **Double-Damping (`DarcyEMA` + `BinghamProx`):** Stacking Poloidal Bingham deadzone shrinkage on top of `DarcyEMA` further removes small-amplitude signal energy (`NoBingham` improved `-0.0479 BPB, z=-2.39`), pushing `M1-H1` down to `1.012–1.367` at an unacceptable primary modeling-capacity cost.
3. **Rank-Quota Starvation & Auxiliary Gradient Drag (`Shell4` ACT + `MTP-8`):** `GeothermalACTController` forced `37.5%` of tokens down to `r=1` pass to hold exact `r_bar = 2.0` (mirroring `S108 H8c1` where adaptive `r_bar = 2` failed to beat uniform `r=2`), while an un-decoupled parallel `MTP-8` auxiliary head competed for hidden-state capacity (`NoACTDraft` improved `-0.2434 BPB, z=-3.35`). **Crucially**, however, exposing the looped weights to variable loop counts `r ∈ {1..4}` during training **completely cured `Cayley2x2`'s `R_test = 6` phase over-winding (`6.6218 -> 2.9389 BPB`)**!

### 1.2 Pairwise Operator Compatibility Matrix for HoloSphere

| Operator Pair | Algebraic Relationship | Empirical Evidence | Composition Rule in HoloSphere |
| :--- | :--- | :--- | :--- |
| **`SU(2)` Holonomy (`H7b`/`H14a`) + `Hopf-S^3` (`H6c`)** | Sequence-axis $SO(4)$ transport + layer-axis $SU(2)$ update; both exact norm-controlled isometries | `-7.37%..-15.38%` `4×` `T=2048` + `-11.5%` BPB (`0` extra params) | **Tier A Core:** Always fused in `float4` registers |
| **`SU(2)` Holonomy + `Cayley2x2` (`S110 #1`)** | Orthogonal axes: `SU(2)` acts along **sequence path $t$** inside attention; `Cayley2x2` acts across **channel pairs $(2j, 2j+1)$** post-MLP (`[C,K]=0`, `||Q x|| = ||x||`) | `Cayley2x2` `#1/14` (`-3.94%`, `z=+3.82`, `+0.25%` params) | **Tier A+1 Surgical Candidate:** Epilogue-fused in `sm_86` registers (zero extra HBM round-trip!) |
| **`Cayley2x2` + `cayley_acute` + Stochastic Variable-Depth `r ∈ {1..4}`** | Rational acute-cone loop clock ($c_r = \frac{1-t_r^2}{1+t_r^2} > 0$) + uniform batch-level `r ~ Uniform{1..4}` (NO rank-quota `37.5%` `r=1` token starvation!) | Cures BOTH `W10-OBS-31` (`8.07` `R=6`) AND `Cayley` over-winding (`6.62 -> 2.94`) | **Tier B Loop Cure:** Test batch-uniform `r ∈ {1..4}` + optional skew scaling $\omega^{(r)} = \omega / \sqrt{r_{\max}}$ |
| **`HoloSphere Core` + `Markov-MTP8` (`S108 H8c2`)** | Coupled low-rank Markov draft head (replaces independent `Parallel-MTP8`) | `-19.34%` (`z=-63.31`) vs `Parallel-MTP8`, `-12.37%` unrolled drift vs `LocalAR` | **Tier C Decoupled Head:** Attach with `stop_grad` or $\lambda_{\text{mtp}} \le 0.1$ so primary `t+1` BPB suffers `0` drag |
| **`HoloSphere Core` + `DarcyEMA` / `BinghamProx`** | Dissipative low-pass / deadzone damping conflicts with isometric sharpness | `H10a`/`H10b`: `+0.144..+0.225 BPB` penalty, but `M1-H1 -> 1.012–1.621` | **EXCLUDED from primary BPB flagship**; opt-in ONLY for a dedicated 3-bit PTQ-hardened (`M1-H1 <= 1.6`) profile |

### 1.3 Staged Earn-Your-Keep Ablation Ladder (`PREREG-S114` Phase-2 Gate)

Every surgical addition MUST clear its own equal-wall-clock (`CS-08`) + equal-param gate before promoting into default `HoloSphere`:
1. **`Stage 0` (`HoloSphere-Base`):** Fused `sm_86` `SU(2)` Holonomy (`H14a`/`H14c`) + `Hopf-S^3` residual (`H6c`).
2. **`Stage +1a` (`+ Fused Cayley2x2`):** Add register-fused `Cayley2x2` post-MLP epilogue. **Pass gate:** beats `Stage 0` on `T=512` AND `T=2048` at equal wall-clock (`<= +2%` step latency vs `Stage 0`) while preserving `||[C,K]||_F = 0` (`m0..m4` drift `< 1e-6`).
3. **`Stage +1b` (`+ Variable-Depth r∈{1..4} + cayley_acute`):** Replace fixed `r_train` with batch-uniform variable-depth training `r ~ Uniform{1,2,3,4}` + pure-rational `cayley_acute` clock (zero rank-quota starvation). **Pass gate:** holds `T=512` BPB within `<= +0.5%` of `Stage +1a` AND keeps `R_test = 6` BPB `<= 3.10` (curing `Cayley2x2`'s `6.62` blowup).
4. **`Stage +2` (`+ Decoupled Markov-MTP8`):** Add low-weight / detached `Markov-MTP8` head. **Pass gate:** `<= +0.2%` primary `t+1` BPB change AND beats `Parallel-MTP8` by `>= 15%` on 8-step draft BPB.

---

## 2. Bare-Metal Ampere `sm_86` Register-Fused `Cayley2x2` Kernel Specification

### 2.1 Why Eager `Cayley2x2` Paid `+15.55%` — And How Fusion Erases It to `< +1.5%`

In `Track S110`'s RTX 3080 microbench (`B=8, T=512, d=384, BF16`), `op1_cayley_2x2_so` cost **`0.5667 ms`/block (`+15.55%` full step, `85.43 ms` vs `73.94 ms`)** despite performing only `4` FMAs per 2D channel pair. Almost the entire cost came from 7+ unfused PyTorch HBM round-trips (`float()`, `reshape`, `w*w`, `1/(1+w2)`, `stack`, `reshape_as`, `.to(bf16)`).

Because `2` Cayley channel pairs `((x0, x1), (x2, x3))` occupy **exactly `1` `float4` (`LDG.E.128`) register quad** (matching HoloSphere's quaternion layout 1-to-1!), `Cayley2x2` fuses cleanly into the **`sm_86` `0 B`-spill kernel foundry** (`holosphere/kernels/csrc/su2_holonomy_scan_sm86.cu` and `Moissanite-Labs/silikernels`) in two forms:
1. **Standalone `float4` kernel** (`holosphere_cayley2x2_{fwd,bwd}_f32`): single `LDG.E.128(x)` + `LDG.E.64(omega)` + `STG.E.128(y)` pass (`~0.04 ms`/block).
2. **Zero-HBM-Epilogue Fusion** (`holosphere_hopf_s3_cayley2x2_{fwd,bwd}_f32` and `fused_spherical_slerp_hopf` epilogue): applies the `2×2` Cayley rotation directly to the `float4` output register **before** `STG.E.128` — adding **`0` extra global loads/stores** for `x`!

### 2.2 Exact Forward & Reverse-Mode Adjoint Mathematics (`0` Trig, `0` Matrix Inverses, `0 B` Spills)

For each 2D pair $(x_0, x_1)$ with bounded skew parameter $w \in \mathbb{R}$ (`w = base_omega + 0.75 * tanh(w_up(silu(w_down(x))))`):

**Forward (per pair: `1` reciprocal + `6` FMAs, `0` SMEM, `0` barriers):**
$$d = \frac{1}{1 + w^2}, \qquad c = (1 - w^2)\,d, \qquad s = (2 w)\,d,$$
$$y_0 = c\,x_0 + s\,x_1, \qquad y_1 = -s\,x_0 + c\,x_1.$$

**Exact Reverse-Mode Adjoint (per pair: `1` reciprocal + `7` FMAs, exact `O(1)` skew derivative!):**  
Let $g = (\partial\mathcal{L}/\partial y_0, \partial\mathcal{L}/\partial y_1)^T$. Since $Q(w) \in SO(2)$ satisfies $Q^T = Q(-w)$ and
$$\frac{dQ}{dw} = \frac{2}{1 + w^2}\begin{pmatrix} -s & c \\ -c & -s \end{pmatrix} = \frac{2}{1 + w^2} \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} Q(w),$$
recomputing $y = Q(w) x$ in registers yields the exact closed-form gradients with **zero extra HBM reads of $y$**:
$$\frac{\partial\mathcal{L}}{\partial x_0} = c\,g_0 - s\,g_1, \qquad \frac{\partial\mathcal{L}}{\partial x_1} = s\,g_0 + c\,g_1,$$
$$\frac{\partial\mathcal{L}}{\partial w} = g^T \frac{dQ}{dw} x = \frac{2}{1 + w^2}\big(g_0\,y_1 - g_1\,y_0\big).$$

### 2.3 Exact `sm_86` Register & Launch Contract

```c
// Packed float4 (2 Cayley 2x2 pairs per thread): n_quads = (B * T * d_model) / 4
extern "C" {
int holosphere_cayley2x2_fwd_f32(
    const float* x_in,       // [n_quads * 4], 16-byte aligned (LDG.E.128)
    const float* omega_in,   // [n_quads * 2], 8-byte aligned  (LDG.E.64)
    float*       y_out,      // [n_quads * 4], 16-byte aligned (STG.E.128)
    int64_t      n_quads,
    void*        stream_ptr
);

int holosphere_cayley2x2_bwd_f32(
    const float* x_in,       // [n_quads * 4]
    const float* omega_in,   // [n_quads * 2]
    const float* grad_y,     // [n_quads * 4]
    float*       grad_x,     // [n_quads * 4]
    float*       grad_omega, // [n_quads * 2]
    int64_t      n_quads,
    void*        stream_ptr
);

// Zero-extra-HBM epilogue fusion with H6c Native Hopf-S^3 residual:
int holosphere_hopf_s3_cayley2x2_fwd_f32(
    const float* h_in, const float* delta_in, const float* omega_in,
    float* h_out, int64_t n_quads, void* stream_ptr
);
int holosphere_hopf_s3_cayley2x2_bwd_f32(
    const float* h_in, const float* delta_in, const float* omega_in,
    const float* grad_out, float* grad_h, float* grad_delta, float* grad_omega,
    int64_t n_quads, void* stream_ptr
);
}
```

- **Resource budget (`nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v`):** `__launch_bounds__(256, 4)`, `<= 24` regs (standalone) / `<= 32` regs (Hopf-$S^3$ epilogue-fused), `0 B` smem, `0` barriers, **`0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**.
- **Low-rank generator cost:** For `d=384, rank=16`, `w_down/w_up` is two tiny GEMMs (`384 -> 16 -> 192`, `0.008M` FLOPs/tok — `< 0.5%` of total block FLOPs), which stays tensor-core bound in `cuBLASLt` while `Cayley2x2` runs register-fused in the residual epilogue.
- **Verification gate before GPU deployment:** Exact FP64 `torch.autograd.gradcheck` on `holosphere_cayley2x2_{fwd,bwd}` (`< 1e-10`), `||[C,K]||_F = 0.0` preserved, and `ptxas -v` `0 B` spill receipt logged alongside `su2_holonomy_scan_sm86.cu`.

---

## 3. Clean Architectural Boundaries: `S111` (`Weft`) & `S115` (`SiliSense`)

### 3.1 `S111` (`WeftAuditShell`) — Surgical Live-Adaptation Wrapper Over Verified Backbones

The `Track S111` `WeftGeosphereModel` design (mirrored in `holosphere/models/weft_shell.py`) confirms a critical architectural advantage: it attaches its Shell-3 `NarrowPlateBank` (`d_plate = 128`, decoupled `W_base + A @ B` avoiding the `bf16 < 0.5-ULP` destructive merge cliff) and non-generative `FactorizedWeftSmithRewriter` (`0.99M`, `has_lm_head = False`, `can_call_tools = False`, `can_mutate_core = False`) **strictly downstream of a frozen SHA-256-pinned Core (`core.forward_hidden`)**. It does **not** hard-wire `H10b_Geosphere_Full`!

To keep `HoloSphere` lean and free of governance bloat:
1. **Strict Backbone Decoupling:** `HoloSphereTransformer` stays a pure, minimal inference/training core exporting only `core_weights_sha256()` and hidden-state outputs. `WeftAuditShell` (`holosphere/models/weft_shell.py`) wraps *any* verified frozen Core — specifically prioritizing **`H10a_Shell1_Cayley2x2` (`#1` verified `S110` backbone)** and **`HoloSphere` (`S114` fused core)** rather than the falsified `H10b_Geosphere_Full`.
2. **Zero Hot-Path Cost:** During normal inference/training, `WeftAuditShell` imposes **zero** per-layer hooks inside `HoloSphereBlock`. The Composed Triple Gate (`StaticProof ∧ Receipt ∧ Holdout` — FP32 nullspace projection on factor $A$, cumulative composed `C_row(sum dW) <= C_max` & `||sum dW||_F <= F_cum` blocking `W10-OBS-20` cancellation attacks, cryptographic binding `H(evaluated) == H(committed)`, and randomized `A_gate` holdout check) plus `~20 KiB` Shell-3 Merkle receipts and `O(1)` atomic rollback (slot restore + KV-cache epoch invalidation + offending trace quarantine) run **exclusively** on live-adaptation commit boundaries.

### 3.2 `S115` (`SiliSense`) — Zero-Overhead Out-of-Band Chip-Feedback Interface

`Track S115` (`Moissanite-Labs/silisense`) uses physical silicon telemetry (Mode 1 default: `intel-rapl` + NVML GPU power/clock/throttle; Mode 2 fail-closed opt-in: authorized audio-only DMIC `A1/A2/A3/A5` observables) to optimize real `EDP = mJ/tok × ms/tok` and predictive throttling.

To guarantee **zero privacy/hardware coupling** and **zero hot-path latency** inside `holosphere`:
1. **Hard Dependency Boundary:** `holosphere` NEVER imports `silisense`, NEVER touches `/dev/video*` or `/dev/snd/*`, and NEVER opens RAPL/NVML sysfs handles. Dependency direction is strictly one-way: **`silisense -> holosphere`** (via optional JSON/dataclass boundary).
2. **Epoch/Step-Boundary Control Contract (`0` Per-Token Python Overhead):**

```python
@dataclass(frozen=True)
class HoloSphereStepTelemetry:
    """Read-only per-step summary emitted by training/bench drivers to external governors (e.g. silisense)."""
    step_idx: int
    seq_len: int
    batch_size: int
    r_loops: int
    backend: str          # "cuda_sm86" | "triton" | "torch_cpu"
    block_t: int          # Triton/CUDA tile size (64 | 128 | 256)
    step_ms: float
    tok_per_sec: float
    norm_drift_max: float

@dataclass(frozen=True)
class HoloSphereTuningProfile:
    """Semantics-preserving hardware execution knobs accepted ONLY between steps/epochs."""
    backend: str = "auto"          # "auto" | "cuda_sm86" | "triton"
    block_t: int = 128             # tile size for fused scan/epilogue kernels
    microbatch_size: int | None = None
    inter_step_cooldown_ms: float = 0.0
```

- **Fail-Closed Safety Invariant:** External chip-feedback governors may adjust **only** execution-schedule knobs (`backend`, `block_t`, `microbatch_size`, `inter_step_cooldown_ms`) between steps — **never** mutating model weights, **never** modifying `WeftAuditShell` thresholds, and **never** altering mathematical equivalence across `CS-08` controlled runs (any knob change is stamped into the run's provenance receipt so `mJ/tok`, `ms/tok`, and `EDP` remain 100% auditable).
