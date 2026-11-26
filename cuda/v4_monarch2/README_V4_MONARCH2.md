# Fused `HoloSphere-V4` Zero-HBM-Input & Zero-HBM-Output-Vocab `sm_86` Megakernel

Additive Ampere `sm_86` (uncontrolled `RTX 3080 Laptop GPU`, `GA104M`, `nvcc 13.4.92`) fused end-to-end `HoloSphere-V4` decode megakernel combining zero-HBM-table `H_256 x bent4` byte injection, multi-layer `HoloSphere CellOp` (`SU(2)` Cayley spinor + INT8 `__dp4a` + SWAR `[16,11,4]` SECDED + per-lane saturating `.sys` CAS BER canary + 8-slot KV + `S^31` Cayley step), and the `16,896 B` CTA-SMEM-resident 2-stage Monarch-`16x16` + `+512 B` byte-bias + `FWHT_256 x bent4` output head + Gumbel-23bit argmax inside a single CTA (`Refs #2`).

- **Telemetry Policy:** `energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED` — zero uncontrolled laptop power or energy claims are published. All latency (`us/tok`) and throughput (`tok/s`) figures below are operational readings recorded on an **uncontrolled `RTX 3080 Laptop GPU` (`GA104M`)** alongside exact `ptxas`/SASS register and spill counts, NaN-safe FP64 parity, live `16/16` SEC heal, and held-out multi-context UTF-8 BPB.
- **Self-Bound On-Disk Verification:** Every SHA-256 digest in [`receipts/v4_monarch2/holosphere_v4_monarch2_ptxas_receipt.json`](../../receipts/v4_monarch2/holosphere_v4_monarch2_ptxas_receipt.json) and [`receipts/v4_monarch2/holosphere_v4_monarch2_results.json`](../../receipts/v4_monarch2/holosphere_v4_monarch2_results.json) binds directly to committed files on disk ([`holosphere_v4_monarch2_e2e_sm86.cu`](holosphere_v4_monarch2_e2e_sm86.cu), [`holosphere_v4_monarch2_live_bench.cu`](holosphere_v4_monarch2_live_bench.cu), and [`holosphere_v4_monarch2_bpb_study.py`](holosphere_v4_monarch2_bpb_study.py)) plus the SHA-256-pinned external UTF-8 corpus (`holosphere-v3-flagship-corpus`), verified with zero-TOCTOU `dir_fd` `O_NOFOLLOW | O_NONBLOCK` symlink/FIFO refusal, `1 MiB` read caps, `1024`-digit integer bounds, duplicate-key refusal, closed nested key allowlists, `--require-verdict CONFIRMED`, comment-stripped contextual scale-token + study AST checks, pure-`Z` algebraic + 6-impulse KAT oracles, and canonical raw-byte checking.

---

## 1. Zero-Spill `sm_86` `ptxas`, SASS, FP64 Parity, SEC Heal & Uncontrolled Laptop Latency (`2 / 2` Kernels)

Compiled cold from [`cuda/v4_monarch2/holosphere_v4_monarch2_e2e_sm86.cu`](holosphere_v4_monarch2_e2e_sm86.cu) under `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror` (zero unions, zero `reinterpret_cast`, raw `alignas(16)` byte-backed `memcpy` SMEM overlay), and replayed on an **uncontrolled `RTX 3080 Laptop GPU` (`GA104M`, `sm_86`)** via [`cuda/v4_monarch2/holosphere_v4_monarch2_live_bench.cu`](holosphere_v4_monarch2_live_bench.cu) with NaN-poisoned buffers and P0-warmed `3`-trial interleaved CUDA-Graph replay (`8,192` launches/trial):

| Kernel (`holosphere_v4_monarch2::*`) | HBM In/Out Vocab (B) | Regs | Barriers | SMEM (B) | Stack / Spill (B) | `LDL` / `STL` | `.sys` `CAS` / `OR` / `ATOMS` | Uncontrolled `GA104M` `1L` (`us/tok` / `tok/s`) | Uncontrolled `GA104M` `4L` (`us/tok` / `tok/s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `holosphere_v3_e2e_scalar_fwht_sm86` | `0 / 0` | `55` | `1` | `14,004` | `0 / 0 / 0` | `0 / 0` | `1 / 1 / 3` | `4.5096` (`221,749 tok/s`) | `9.1649` (`109,112 tok/s`) |
| **`holosphere_v4_e2e_monarch2_bias_fwht_sm86`** | **`0 / 0`** | **`55` (`+0`)** | **`1`** | **`26,292`** | **`0 / 0 / 0`** | **`0 / 0`** | **`1 / 1 / 3`** | **`5.7706` (`173,291 tok/s`)** | **`10.4209` (`95,961 tok/s`)** |

- **NaN-Safe FP32-vs-FP64 Parity (FP16-Rounded Head Weights):** `V3 1L = 2.8890e-6`, `V4 1L = 5.1548e-7`, `V4 4L = 3.5004e-7` (all `< 1e-5`), comparing the fused FP32-accumulated CUDA kernel against the exact FP64 host reference evaluated on identical `__half`-rounded head weights (`__half2float`).
- **Paired Hadamard Scaling Convention & BOS-to-`uint8_t` Equivalence:** `fwht256_plain_8warp_raw` applies orthonormal scale `0.0625f` (`1/16`, `||H u||_2^2 = ||u||_2^2`), whereas `fwht256_bent4_8warp_raw` applies `0.17677669529663688f` (`1/sqrt(32) = sqrt(8) * orthonormal`, so `||y||_2^2 = 8 * ||u||_2^2`), matching `fwht256_bent4_fp64` (`1.0 / std::sqrt(32.0)`) in [`holosphere_v4_monarch2_live_bench.cu`](holosphere_v4_monarch2_live_bench.cu) and `fwht_mat` in [`holosphere_v4_monarch2_bpb_study.py`](holosphere_v4_monarch2_bpb_study.py). In [`verify_holosphere_v4_monarch2.py`](verify_holosphere_v4_monarch2.py), the exact integer oracle `fwht256_bent4_int` operates unnormalized over `Z` (`sum y^2 = 256 * sum u^2` before `(1/sqrt(32))^2 = 1/32` scaling) and verifies 6 impulse delta vectors against closed-form signed-Hadamard rows. For input injection, [`holosphere_v4_monarch2_bpb_study.py`](holosphere_v4_monarch2_bpb_study.py) builds `257` rows (`0..255` UTF-8 bytes plus `BOS = 256` via `sp_flip = (d256 >> 5) & 1`), whereas [`holosphere_v4_monarch2_e2e_sm86.cu`](holosphere_v4_monarch2_e2e_sm86.cu) accepts `uint8_t` (`0..255`) directly without a separate `sp_flip` bit; algebraically, `H_256 x bent4` defines `256` distinct byte-alphabet vectors and `BOS = 256` is bit-identical to UTF-8 byte `33` (`0x21`, `'!'`, verified by `bos_256_matches_byte_33 = true` in the oracle), so a host runtime starting from `BOS = 256` passes `0x21` directly to the `uint8_t` CUDA kernel.
- **Live `[16,11,4]` SECDED SEC Heal inside Fused V4 `4L`:** `16/16` single-bit corruptions across all 16 byte positions healed in-flight at **`0.0` hidden error** and **`0.0` logit error**, reporting `v4_sec16_packed_u32 = 16` via exactly `1` CTA `.sys` CAS flush (`0` `.sys` atomics on clean steps).
- **Zero Extra Register Pressure:** Overlaying the `20,480 B` Monarch-2 workspace directly on top of the post-loop KV value tile in CTA SMEM keeps `holosphere_v4_e2e_monarch2_bias_fwht_sm86` at **`55` registers** (`0 B` spills) with `+1.26 us/tok` incremental head latency on the uncontrolled `RTX 3080 Laptop GPU` (`GA104M`).

---

## 2. Controlled 4-Arm x 3-Seed Multi-Context UTF-8 BPB Study (`T = 512, 2048, 8192`)

Reproduced via [`cuda/v4_monarch2/holosphere_v4_monarch2_bpb_study.py`](holosphere_v4_monarch2_bpb_study.py): iso-budget 4-layer `d_model = 512`, `d_ff = 800`, `n_heads = 16`, `rope_base = 500,000` backbone with `0 B` HBM input embedding (`H_256 x bent4`) trained for `300` steps (`1,228,800` tokens/seed) across `3` seeds `{42, 137, 2026}` and evaluated on `65,536` held-out UTF-8 bytes at each context length (`T = 512, 2048, 8192`) under FP16-rounded head weights (`eval_bpb_q`, with BF16 vs FP16 quantization delta recorded in `fp16_quant_bpb_diff_q`; external UTF-8 corpus binary `holosphere-v3-flagship-corpus` is SHA-256-pinned in `source_bindings` and not checked into this repository):

| Arm | Head Params | Head FP16 (B) | HBM Out Vocab (B) | Mean BPB (`T=512`) | Mean BPB (`T=2048`) | Mean BPB (`T=8192`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `V3_ScalarFWHT` | `1` | `2` | `0` | `5.6967` | `5.6935` | `5.6957` |
| `V3_LowRank16Bias` | `8,448` | `16,896` | `16,896` | `4.8619` | `4.8569` | `4.8611` |
| **`V4_Monarch2BiasFWHT`** | **`8,448`** | **`16,896`** | **`0`** | **`4.0175`** | **`4.0130`** | **`4.0176`** |
| `V3_DenseHead` | `131,329` | `262,658` | `262,658` | `3.8290` | `3.8244` | `3.8274` |

| Context | V4 Lift vs Scalar (`3/3` seeds) | V4 Lift vs Matched-Byte Rank-16 (`3/3` seeds) | V4 Minus Dense (`262 KiB`) | Dense Gap Closed |
| :--- | ---: | ---: | ---: | ---: |
| `T = 512` | **`+1.6792 BPB`** | **`+0.8444 BPB`** | `0.1885 BPB` | **`89.91%`** |
| `T = 2048` | **`+1.6805 BPB`** | **`+0.8440 BPB`** | `0.1886 BPB` | **`89.91%`** |
| `T = 8192` | **`+1.6781 BPB`** | **`+0.8434 BPB`** | `0.1902 BPB` | **`89.82%`** |

### Gate Grading & Scope
- **`G1_e2e_multicontext_quality`:** Requires `v4_lift_over_scalar >= 0.50` BPB and `3/3` seed wins over `V3_LowRank16Bias` across all `T in {512, 2048, 8192}`, plus long-context dense-ceiling proximity `v4_minus_dense <= 0.20` BPB at `T in {2048, 8192}` (`T = 512` also satisfies `0.1885 <= 0.20` BPB).
- **`G2_fused_sm86_zero_spill_and_latency`:** Requires `2/2` zero-spill `sm_86` kernels, `0` clean syndromes/atomics, `16/16` live SEC heals at `0.0` hidden/logit drift via `1` `.sys` CAS flush, and `1L` latency `<= 25.0 us/tok` (`5.7706 us/tok` measured on uncontrolled `GA104M`).
- **Scope:** Measurements are at `d_model = 512`, `V = 256` UTF-8 bytes, `n_layers = 4`, and `300` steps (`1.23M` tokens/seed); scaling to deeper backbones, larger BPE vocabularies via factored radix-`64`/`128` Monarch-FWHT, and longer horizons remains open follow-on work.

---

## 3. Build & Offline Verification

```bash
# Pure-stdlib 3-mode isolated -I receipt + pure-Z oracle + tamper-refusal verification (no GPU required)
make -f cuda/v4_monarch2/holosphere_v4_monarch2.mk verify-v4-monarch2

# On NVIDIA Ampere sm_86 with nvcc: compile (--Werror all-warnings) and verify 2/2 zero-spill kernels
make -f cuda/v4_monarch2/holosphere_v4_monarch2.mk check-v4-monarch2-zero-spill
```
