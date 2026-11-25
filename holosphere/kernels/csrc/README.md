# Bare-Metal Ampere `sm_86` CUDA/PTX Kernel ABI (`csrc/`)

This directory hosts all **27 bare-metal zero-spill NVIDIA Ampere `sm_86` CUDA/PTX kernels** integrated from Track S114 V2 (`Moissanite-Labs/silikernels` companion drop).

## Core Isomorphism: `1` Unit Quaternion `q ∈ SU(2)` = `1` `float4` (`LDG.E.128`)

- **Register layout:** `float4 q = {w, x, y, z}` (`16` bytes aligned, `4` FP32 registers per thread).
- **Forward associative scan (V1 fiber-major + V2 zero-permute strided BHTD):** 5-step warp-level `__shfl_up_sync(0xFFFFFFFFu, ...)` Hillis–Steele right-accumulating Hamilton product `Pi[t] = du[0] * ... * du[t]` + hierarchical block-prefix propagation across `T = 512..2048+`, supporting both `su2_exp` and SFU-free pure-rational `cayley_su2` in FP32 and direct BF16 I/O (`0` `.permute().contiguous()` copies!).
- **Reverse-mode adjoint backward:** Exploits the exact algebraic unit-quaternion SU(2) identity (independently verified in `Lean 4.36.0` and measured `<= 5.87e-6` FP32 parity on RTX 3080):
  1. Per-thread `float4` pullback `m[t] = Im(grad_Pi[t] * conj(Pi[t]))` in $\mathfrak{su}(2) \cong \mathbb{R}^3$
  2. Pure additive 5-step warp `__shfl_down_sync` suffix sum `S[t] = sum_{j>=t} m[j]` (`0` quaternion multiplies inside the scan!)
  3. Per-thread `grad_du[t] = conj(Pi[t-1]) * ((0, S[t]) * Pi[t])`
- **V2 Beltrami `Cayley2x2` & Zero-HBM-Epilogue `Hopf-S^3 + Cayley2x2` (`+8` kernels, FP32 & BF16):** Packs `2` channel pairs per `float4` quad (`LDG.E.128`) with closed-form reverse-mode adjoint $\partial\mathcal{L}/\partial w = \frac{2}{1+w^2}(g_0 y_1 - g_1 y_0)$ (`9.77×` / `28.58×` vs eager).
- **Grok 4.7 P0/P1 Safety Hardening:** `m2 < 1e-3f` SLERP antipodal radial-term guard (eliminating `O(eps^-3)` gradient spikes just above `eps^2`), `q_norm(0) -> (1, 0, 0, 0)` non-sticky identity fallback, and explicit `stride % 4 == 0` / `% 3 == 0` divisibility guards in `cuda_bridge.py`. *(V2.1 optimization roadmap: split forward `sh_q[4][34]` into conflict-free scalar arrays `sh_qw/x/y/z[4][33]`.)*
- **Zero-Spill Gate (`ptxas -v`):** All **`27/27`** exported kernels compiled with `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v` report **`0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`**.
