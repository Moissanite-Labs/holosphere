# Finite-N `FWHT_256` / `Monarch-3` (`HoloSphere-V5`), `MUX` Cor. 15 vs. Fixed-Radix, & `SU(2)` Loop-Cache Stability

This study provides a self-contained, pure-Python stdlib (`int`/`Fraction`) and pure-`Init` Lean `4.36.0` verification bundle analyzing four exact algebraic and operator regimes for `HoloSphere-V5` byte-level transforms, fixed-radix quantized accumulation, and looped `SU(2)` state transport.

## Epistemic Scope & Policy Disclosures

- **Arithmetic & Execution Policy:** CPU-only verification (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`), zero IEEE-754 floats in production arithmetic/logic, zero `ast.Div` or `ast.Pow`, zero bare `assert` statements, zero unseeded RNG calls, and zero model weights or `.lake/` build caches committed.
- **Energy & Power Scope:** `energy_power_status = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"` (no controlled power or energy measurements are claimed).
- **GPU Ablation Scope:** `gpu_ablation_status = "CPU_ONLY_NOT_RUN_ON_GPU_NO_FABRICATED_BPB"` (end-to-end GPU BPB retraining for `HoloSphere-V5` candidate heads was not executed in this CPU-only formal and pure-$\mathbb{Q}$ study).

## Mathematical & Architectural Summary

1. **Finite-$N$ `FWHT_256` Crossover vs. Asymptotic `FiniteFourier`:**
   - At byte width $N = 256$ ($m = \log_2 N = 8$), $\sqrt{1 + m} = \sqrt{9} = 3$ is an exact integer. Comparing `FWHT_256`'s `m = 8` butterfly gates/coordinate (`2,048` gates/vector; `4,096` per dual-bank token) against $C_{\mathrm{FF}} N (1 + \log_2 N)^{1/2 + \varepsilon}$ yields exact parity at $C_{\mathrm{FF}} = 8/3$ and strict `FWHT_256` advantage for all $C_{\mathrm{FF}} \ge 3$ (crossing the squared bound at $m^* = 10$, $N^* = 1{,}024$ for $C_{\mathrm{FF}} = 3$, and $m^* = 17$, $N^* = 131{,}072$ for $C_{\mathrm{FF}} = 4$).
   - Hamming concentration width at $m = 8$ is `11 > 9` (`0` Hamming-level pruning compression), `Prop 5.2` character price strictly exceeds permutation price by `8..32` gates/coord for $L = 1..4$, and `FWHT_256 x bent4` satisfies exact involution ($H_{256}^2 = 256 I$) and Parseval energy (`1,408 -> 360,448 = 256 x 1,408`, `96/256` negated coordinates across `12/32` lanes).
2. **`HoloSphere-V5` Zero-Alias Input & `3`-Stage Monarch Head (`sm_86` `48 KiB` SMEM Fit):**
   - Isolates the `4/4` special-token byte collisions (`256..259 -> 33, 32, 35, 34` under `0x21` XOR flip) and the `1`-path/pair subblock rank-`1` bottleneck of `V4`'s `2`-stage `16x16` (`256`-dim input) head.
   - Replaces input encoding with $H_2 \otimes (H_{256} \times \text{bent4})$ (`260` / `512` pairwise orthogonal symbols in $\mathbb{Z}^{512}$, `v5_max_offdiag_dot = 0`, `0` aliases) and evaluates `V5B_Monarch3_16x32_16x16_16x16` (`512 -> 256`, `16` paths/pair verified exhaustively across all `512 x 256` pairs, subblock rank `16`, `16,644` params = `33,288 B` FP16, `0 B` HBM vocab, `43,700 B` E2E SMEM leaving `5,452 B` headroom under `49,152 B` static `sm_86` SMEM) alongside `V5A_Monarch3_8x8x8_512` (`35,508 B` E2E SMEM, `13,644 B` headroom).
3. **`MUX` Cor. 15 Precision Collapse vs. Fixed-Radix (`u = 0`) & `A_16 Block-H4`:**
   - Exact meet-in-the-middle rational separation $m_S$ shows floating-point `MUX` demultiplexing at $\rho = 9/10$ collapses at $S = 4$ (`BF16`), $S = 5$ (`FP16`), and $S = 12$ (`FP32`), whereas exact fixed-radix (`u = 0`) holds for all $S \ge 1$ ($\eta = 0$).
   - Exhaustive audit over all `2,048` zero-syndrome `A_16` codewords confirms `0` `Block-H4` Parseval mismatches (`sum_c_sq = 704,512`, `sum_h4_sq = 2,818,048`) and `0` symmetric sign-flip drift, alongside exact rational cubic fixed-point contraction vs. Lipschitz cliff divergence.
4. **`DualAxisCarry` ($\beta_{\min}$) & `SU(2)` `O(1)` `S_fast` Loop Transport:**
   - Guarded EMA damping $\beta_{\min} = 11/20$ bounds adversarial step-`4` shock drift to `72051579/68181865` vs. `3/2` carry-last ($\beta = 0$) and recovers to `41727253063229644769/7291086444631243669515`.
   - Exact Cayley `SU(2)` transport maintains unit norm $\|U_r\|^2 = 1$ and exact two-sided gauge-pullback alignment cosine `1` across loops $r = 1..4$, while the `65,536 B` `O(1)` `S_fast` Woodbury cache reduces memory by `1,536x` vs. `O(TL)` and `6,144x` vs. `O(TRL)` FP16 KV caches at `T = 8192, L = 6, R = 4` with `gamma = 9/10` contraction (`(9/10)^4 = 6561/10000`) and `MERGE_LORA=0` integer parity.

## Bundle Contents

- `FourierMuxLoopOperators.lean` — `18` pure-`Init` `Lean 4.36.0` theorems (`17` universally quantified + `1` concrete kernel-`decide` phase-boundary conjunction; axioms `[]` or `[propext, Quot.sound]` only; `0` `sorry`, `0` `Classical.choice`, `0` `native_decide`).
- `lakefile.toml`, `lake-manifest.json`, `lean-toolchain` — Standalone `Lean 4.36.0` package config (`"packages": []`).
- `s164_fourier_mux_loop_oracle.py` — Pure-stdlib exact integer/rational (`int`/`Fraction`) oracle.
- `verify_s164_fourier_mux_loop.py` — Zone-B hardened verifier, AST pure-$\mathbb{Q}$ auditor, secondary mathematical cross-auditor, static R12 Lean scanner, and tempdir-staged `--live-lean` runner (`0` `.lake/` residue).
- `s164_fourier_mux_loop_certificate.json` — Canonical certificate (`schema_version = "s164-fourier-mux-loop-cert-v1"`, `payload_sha256 = ad791c90f9b3658fe60279527b6256dbcc037ff22cc311b3cd0a315b09e086cb`, `certificate_sha256 = d3caaea5903cbea9b50f90405e2f0e5cb24eceaf072386cd5e58e98d9563c1ec`).
- `test_s164_fourier_mux_loop_r5.py` — `7` assertion-free R5 and adversarial forgery/filesystem/scanner tests (`0` bare `assert`s, `0` skips).
- `REPORT.md` — Detailed technical report, tables, theorem inventory, and SHA-256 manifest.

## Quick Verification

```bash
# 1. Verify certificate + oracle replay across all 3 isolated -I modes
for flags in "-I -B" "-I -O -B" "-I -B -W error"; do
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 ${flags} \
    studies/fourier-mux-holosphere-v5/verify_s164_fourier_mux_loop.py --check --json
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 ${flags} \
    studies/fourier-mux-holosphere-v5/test_s164_fourier_mux_loop_r5.py
done

# 2. Cold live Lean 4.36.0 build & axiom audit (tempdir-staged, read-only safe, zero .lake/ residue)
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python3 -I -B \
  studies/fourier-mux-holosphere-v5/verify_s164_fourier_mux_loop.py --check --json --live-lean
```
