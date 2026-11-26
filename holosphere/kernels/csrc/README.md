# Bare-Metal Ampere `sm_86` CUDA/PTX Kernel ABI (`csrc/`)

This directory hosts **40 bare-metal zero-spill NVIDIA Ampere `sm_86` CUDA/PTX kernels**: `27` core `SU(2)` / `Cayley2x2` kernels (`su2_holonomy_scan_sm86.cu`, `Moissanite-Labs/silikernels#3` companion mirror), `11` Chunk32 Woodbury / `H_256 x bent4` / single-CTA `16`-warp `d_model=512` CellOp `.sys` BER-canary kernels (`holosphere_h256_chunk32_sm86.cu`, verified in `ptxas_s117_zero_spill_receipt.json`), and `2` V3 Frontier kernels (`holosphere_v3_frontier_sm86.cu`, verified in `ptxas_v3_frontier_zero_spill_receipt.json`).

## Track S117 V3 Frontier (`holosphere_v3_frontier_sm86.cu` — `2/2` `0 B` Spills)

- **`s117_chunk32_bwd_wmma_sm86` (`W10-OBS-61` Double-Woodbury Symmetric Gram Tensor-Core Backward):** `128` regs (`100%` RF @ `16` warps/SM), `6,336 B` SMEM (`99%` SMEM), `0 B` spills, `28` `HMMA.16816`, `30` `SHFL` (`-93.35%` vs Choice B's `451`); live RTX 3080 (`B=4, T=8192, D=512, L=12`): **`148,382.09 tok/s` (`220.84 ms / 12L`, `2.409×` over Choice B, `0.4105 mJ/tok` NVML)** with `<= 4.36e-5` 3-chunk FP64 parity.
- **`s117_uniform_cell_16warp_d512_v3_sm86` (`0`-`POPC` Dual-SWAR SECDED + Cross-Layer `LDG.128` Prefetch):** `52` regs, `13,992 B` SMEM, `0 B` spills, `24` `IDP.4A`, `7` `LDG.128`, `1` `RED.E.ADD.STRONG.SYS`; live RTX 3080: **`20.51 us/tok`** CellOp / **`21.52 us/tok` (`46,468 tok/s`)** E2E w/ `0 B` tied `FWHT_256 x bent4` (`1.96e-7` dense FP64 parity) and **`16/16` SEC `0.0` bit-exact** `.sys` BER canary (`clean=0, sec16=16, ded=1`).

## Track S117 Full-Stack Collapse & Razor `.sys` Canary (`holosphere_h256_chunk32_sm86.cu`)

- **Zero-Spill Gate (`ptxas_s117_zero_spill_receipt.json`):** **`11/11` kernels `0 B` stack / `0 B` spills** (`LDL: 0, STL: 0, ATOMG_GPU: 0, MEMBAR_GL: 0, IDP4A: 16, IMMA_16832: 6, RED_SYS: 2, MEMBAR_SYS: 2`).
- **Single-CTA `16`-Warp `d_model = 512` V2 CellOp (`s117_uniform_cell_16warp_d512_v2_sm86`):** `48` regs, `16,040 B` SMEM, `0` cross-CTA barriers; live RTX 3080 (`GA104M`) **`26.23 us/tok`** CellOp-only and **`27.43 us/tok` / `0.989 mJ/tok`** full E2E with `0 B`-VRAM tied `FWHT_256 x bent4` LM head (`1.83e-7` parity vs FP64 dense) + double-buffered inline `[16, 11, 4]` LSB SECDED `.sys` BER canary (**`0` clean PCIe writes, `16/16` SEC `0.0` bit-exact across all `512` hidden floats & `256` logits, `1/1` DED flagged**).
- **Razor `64 B` `.sys` Telemetry ABI (`include/razor_abi_4byte_sys.h`):** Cacheline-aligned `HoloSphereRazorCanary64` (`sizeof=64, alignof=64`) + `16 B` `HoloSpherePiRollback16` (`Apache-2.0` SPDX) bridging `30 kHz` BER detection to [`Moissanite-Labs/silisense`](https://github.com/Moissanite-Labs/silisense). *Scope boundary:* `[16, 11, 4]` LSB SECDED covers `16/16` LSB flips and `120/120` 2-bit LSB pairs (`0/48` on signed-nibble upper bits `1..3` alone); `include/razor_abi_4byte_sys.h` closes `48/48` upper-bit single faults via folded `bit1^bit2^bit3` parity (`0` extra DRAM B) and epoch-rotated bit-plane checks (`b = epoch & 3`).

## Core Isomorphism: `1` Unit Quaternion `q ∈ SU(2)` = `1` `float4` (`LDG.E.128`)

- **Register layout:** `float4 q = {w, x, y, z}` (`16` bytes aligned, `4` FP32 registers per thread).
- **Forward associative scan (V1 fiber-major + V2 zero-permute strided BHTD):** 5-step warp-level `__shfl_up_sync(0xFFFFFFFFu, ...)` Hillis–Steele right-accumulating Hamilton product `Pi[t] = du[0] * ... * du[t]` + hierarchical block-prefix propagation across `T = 512..2048+`, supporting both `su2_exp` and SFU-free pure-rational `cayley_su2` in FP32 and direct BF16 I/O (`0` `.permute().contiguous()` copies!). Both `sh_q[4][34]` (`2,176 B` fwd) and `sh_m{x,y,z}[4][40]` (`1,920 B` bwd) are bank-conflict-free across 8-thread `128 B` SMEM bus beats.
- **Reverse-mode adjoint backward:** Exploits the exact algebraic unit-quaternion SU(2) identity (proved over exact integer `QuatZ` ($\mathbb{Z}$) in `lean/HoloSphereInvariants.lean`; empirical RTX 3080 FP32 parity `<= 5.87e-6`, BF16 `<= 1.95e-3`):
  1. Per-thread `float4` pullback `m[t] = Im(grad_Pi[t] * conj(Pi[t]))` in $\mathfrak{su}(2) \cong \mathbb{R}^3$
  2. Pure additive 5-step warp `__shfl_down_sync` suffix sum `S[t] = sum_{j>=t} m[j]` (`0` quaternion multiplies inside the scan!)
  3. Per-thread `grad_du[t] = conj(Pi[t-1]) * ((0, S[t]) * Pi[t])`
- **V2 Beltrami `Cayley2x2` & Zero-HBM-Epilogue `Hopf-S^3 + Cayley2x2` (`+8` kernels, FP32 & BF16):** Packs `2` channel pairs per `float4` quad (`LDG.E.128`) with closed-form reverse-mode adjoint $\partial\mathcal{L}/\partial w = \frac{2}{1+w^2}(g_0 y_1 - g_1 y_0)$ (`9.77×` / `28.58×` vs eager).
- **Safety Hardening:** `m2 < 1e-3f` SLERP antipodal radial-term guard (eliminating `O(eps^-3)` gradient spikes just above `eps^2`), `q_norm(0) -> (1, 0, 0, 0)` non-sticky identity fallback, host `_valid_eps(eps)` + `int64_t` grid checks, and explicit `stride % 4 == 0` / `% 3 == 0` divisibility guards in `cuda_bridge.py`.
- **Zero-Spill Gate (`ptxas -v`):** All **`27/27`** exported kernels (`17` `__global__` templates) compiled with `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` report **`0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**.
