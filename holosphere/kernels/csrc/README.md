# Bare-Metal Ampere `sm_86` CUDA/PTX Kernel ABI (`csrc/`)

This directory hosts bare-metal zero-spill NVIDIA Ampere `sm_86` CUDA/PTX kernels integrated from Track S114 (`Moissanite-Labs/silikernels` companion drop).

## Core Isomorphism: `1` Unit Quaternion `q ∈ SU(2)` = `1` `float4` (`LDG.E.128`)

- **Register layout:** `float4 q = {w, x, y, z}` (`16` bytes aligned, `4` FP32 registers per thread).
- **Forward associative scan:** 5-step warp-level `__shfl_up_sync(0xFFFFFFFFu, ...)` Hillis–Steele right-accumulating Hamilton product `Pi[t] = du[0] * ... * du[t]` + hierarchical block-prefix propagation across `T = 512..2048+`, fused directly with in-register left-transport `Q' = L(Pi) Q`, `K' = L(Pi) K`, and optional covariant `V' = L(Pi) V` (`H14c`).
- **Reverse-mode adjoint backward:** Exploits the exact unit-quaternion SU(2) identity:
  1. Per-thread `float4` pullback `term[t] = grad_Pi[t] * conj(Pi[t])`
  2. Pure additive 5-step warp `__shfl_down_sync` suffix sum `S[t] = sum_{j>=t} term[j]` (`0` quaternion multiplies inside the scan!)
  3. Per-thread `grad_u[t] = conj(Pi[t-1]) * S[t] * Pi[t]`
- **Zero-Spill Gate (`ptxas -v`):** Every exported kernel compiled with `nvcc -O3 -arch=sm_86 --ptxas-options=-v` MUST report `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.
