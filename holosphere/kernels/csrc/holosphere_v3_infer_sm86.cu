// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Track S117 E2E Engine (Part B): Bare-Metal Ampere (GA104M sm_86) Real-Weight
// HoloSphere-V3 Autoregressive Inference Runtime (0 B Spills, 0 LDL/STL)
//
// Executes REAL trained HoloSphere-V3 (up to D=512, H=16; e.g., 15M: D=512,
// H=16, L=6, d_ff=800, V=260) checkpoints end-to-end on sm_86:
//   - Zero-VRAM H_256 x bent4 byte injection + CausalDepthwiseByteConv(k=4)
//   - A_16 Block-H4 [16, 11, 4] uint64_t GEMV with 0-POPC dual-SWAR hot-path
//     detector, cold-path POPC syndrome decode & inline 1-bit SEC bit-exact
//     repair, 2-bit DED detection, and
//     cacheline-aligned 64 B HoloSphereRazorCanary64 (.sys atomic D2H, 0 PCIe
//     writes on clean cycles, ZERO cudaMemcpy!)
//   - Unified CellOp step: feature-space V_ring Hard-FMNMX + bounded CayleySU2
//     (||w|| <= 8.0) + exact FP32 S_fast (32x32/head) + S^{31}/S^{D-1} geodesic
//   - Fused SwiGLU + hyperspherical residual + 23-bit finite Gumbel FMNMX
//     (supporting BOTH trained A_16 lm_head[260, D] AND 0 B tied FWHT_256!)
//   - O(1) Weft 16 B Pi_{t_0} holonomy + O(1) state checkpoint & exact rollback
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stddef.h>

#include "../../../include/razor_abi_4byte_sys.h"

namespace holosphere_v3_infer {

__device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float4 cayley_su2(float3 w) {
    float w2  = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv = 1.0f / (1.0f + w2);
    return make_float4((1.0f - w2) * inv, (2.0f * w.x) * inv, (2.0f * w.y) * inv, (2.0f * w.z) * inv);
}

__device__ __forceinline__ float4 quat_conj(float4 q) {
    return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float4 quat_mul(float4 a, float4 b) {
    return make_float4(
        fmaf(a.x, b.x, -fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w))),
        fmaf(a.x, b.y,  fmaf(a.y, b.x, fmaf(a.z, b.w, -a.w * b.z))),
        fmaf(a.x, b.z,  fmaf(-a.y, b.w, fmaf(a.z, b.x, a.w * b.y))),
        fmaf(a.x, b.w,  fmaf(a.y, b.z, fmaf(-a.z, b.y, a.w * b.x)))
    );
}

__device__ __forceinline__ float4 quat_unit_norm(float4 q) {
    float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    float s  = rsqrtf(n2 + 1e-24f);
    return make_float4(q.x * s, q.y * s, q.z * s, q.w * s);
}

__device__ __forceinline__ float4 gather_spinor4(float x_i, int base) {
    return make_float4(
        __shfl_sync(0xFFFFFFFFu, x_i, base + 0),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 1),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 2),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 3)
    );
}

__device__ __forceinline__ float select_lane4(float4 q, int lmod4) {
    float v01 = (lmod4 & 1) ? q.y : q.x;
    float v23 = (lmod4 & 1) ? q.w : q.z;
    return (lmod4 & 2) ? v23 : v01;
}

__device__ __forceinline__ float gumbel_23bit_finite(uint32_t u32, float temperature) {
    float top23       = __uint2float_rn(u32 >> 9);
    float u           = fmaf(top23, 1.1920928955078125e-7f, 5.9604644775390625e-8f);
    float neg_lg2_u   = -__log2f(u);
    float lg2_neg_lg2 = __log2f(neg_lg2_u);
    float g           = fmaf(-0.69314718f, lg2_neg_lg2, 0.36651292f);
    return temperature * g;
}

// ============================================================================
// 0-POPC Dual-SWAR INT32 [16, 11, 4] SECDED + Razor 64 B .sys Telemetry
// ============================================================================

__device__ __forceinline__ uint32_t extract_nibble_lsbs8(uint32_t w32) {
    uint32_t m = w32 & 0x11111111u;
    m = (m | (m >> 3)) & 0x03030303u;
    m = (m | (m >> 6)) & 0x000F000Fu;
    return (m | (m >> 12)) & 0xFFu;
}

__device__ __forceinline__ uint32_t extract_a16_odd16(uint64_t w64) {
    uint32_t lo = uint32_t(w64);
    uint32_t hi = uint32_t(w64 >> 32);
    return extract_nibble_lsbs8(lo) | (extract_nibble_lsbs8(hi) << 8);
}

__device__ __forceinline__ uint32_t dual_a16_syndrome_nonzero_swar(uint32_t w32) {
    uint32_t a  = w32 ^ (w32 >> 1);
    uint32_t b  = a   ^ (a   >> 2);
    uint32_t c  = b   ^ (b   >> 4);
    uint32_t d  = c   ^ (c   >> 8);
    uint32_t x1 = w32 & 0xAAAAAAAAu;
    x1 ^= (x1 >> 2);
    x1 ^= (x1 >> 4);
    x1 ^= (x1 >> 8);
    uint32_t x2 = a   & 0x44444444u;
    x2 ^= (x2 >> 4);
    x2 ^= (x2 >> 8);
    uint32_t p3 = (b ^ (b >> 8)) & 0x00100010u;
    uint32_t p4 = c & 0x01000100u;
    return (d & 0x00010001u) | (x1 & 0x00020002u) | (x2 & 0x00040004u) | p3 | p4;
}

__device__ __forceinline__ void infer_canary_emit_sys(
    HoloSphereRazorCanary64* host_canary,
    uint32_t lane_sec,
    uint32_t lane_ded,
    uint32_t diag_word,
    int lane
) {
    uint32_t sec_mask = __ballot_sync(0xFFFFFFFFu, lane_sec != 0u);
    uint32_t ded_mask = __ballot_sync(0xFFFFFFFFu, lane_ded != 0u);
    uint32_t any_mask = sec_mask | ded_mask;
    if (any_mask != 0u) {
        int      src_lane = __ffs(int(any_mask)) - 1;
        uint32_t diag_out = __shfl_sync(0xFFFFFFFFu, diag_word, src_lane);
        if (lane == 0 && host_canary != nullptr) {
            if (sec_mask != 0u) {
                ::razor_atom_release_sys_add_u32(&host_canary->sec_count_u32, uint32_t(__popc(sec_mask)));
            }
            if (ded_mask != 0u) {
                ::razor_atom_release_sys_add_u32(&host_canary->ded_count_u32, uint32_t(__popc(ded_mask)));
            }
            ::razor_st_release_sys_u32(&host_canary->last_syndrome_u32, diag_out);
        }
    }
}

__device__ __forceinline__ float a16_blockh4_dot16(uint64_t w64, const float4* __restrict__ x4_ptr) {
    float acc = 0.0f;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint32_t nibs = uint32_t((w64 >> (g * 16)) & 0xFFFFULL);
        int c0 = int((nibs << 28)) >> 28;
        int c1 = int((nibs << 24)) >> 28;
        int c2 = int((nibs << 20)) >> 28;
        int c3 = int((nibs << 16)) >> 28;
        int s01 = c0 + c1;
        int d01 = c0 - c1;
        int s23 = c2 + c3;
        int d23 = c2 - c3;
        float4 xv = x4_ptr[g];
        acc = fmaf(float(s01 + s23), xv.x, acc);
        acc = fmaf(float(d01 + d23), xv.y, acc);
        acc = fmaf(float(s01 - s23), xv.z, acc);
        acc = fmaf(float(d01 - d23), xv.w, acc);
    }
    return acc;
}

// ============================================================================
// Kernel B1: Zero-VRAM H_256 x bent4 Byte Injection + CausalDepthwiseByteConv4
//   Matches holosphere_v3_ablation.py::h256_bent4_inject_zero_vram + ByteConv4!
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_inject_byteconv4_sm86(
    const int32_t* __restrict__ tok_id_ptr,    // [1] token id in 0..259
    const float* __restrict__ conv_w,          // [D, 4] CausalDepthwiseByteConv weights [t-3,t-2,t-1,t]
    float* __restrict__ conv_ring,             // [4, D] circular 4-tap history buffer
    int32_t* __restrict__ conv_pos_io,         // [1] monotonic token position t
    float* __restrict__ x_out,                 // [D] unit-normalized S^{D-1} state
    int d_model
) {
    __shared__ float smem_sq[32];
    int tid = threadIdx.x;
    if (tid >= d_model) return;

    int tok     = tok_id_ptr[0];
    int t_pos   = conv_pos_io[0];
    int bank    = tid >> 8;        // 0, 1, 2
    int d_in256 = tid & 255;
    int lane    = d_in256 & 31;
    int db      = d_in256 ^ ((lane & 7) << 5);
    int b4_raw  = (lane & 3) & ((lane >> 2) & 3);
    int bent4   = (b4_raw ^ (b4_raw >> 1)) & 1;

    int b_u8        = tok & 0xFF;
    int special_tag = (tok >> 8) & 3;
    int b_eff       = (bank >= 2) ? (b_u8 ^ ((0xA5 * (bank - 1)) & 0xFF)) : b_u8;
    float sign_mult = (bank == 1) ? -1.0f : 1.0f;

    int sp_flip = special_tag & ((d_in256 >> 5) & 1);
    int parity  = (__popc(b_eff & db) ^ bent4 ^ sp_flip) & 1;
    const float inv_sqrt32 = 0.17677669529663688f;
    float emb = (parity ? -1.0f : 1.0f) * (inv_sqrt32 * sign_mult);

    // Push into 4-tap ring buffer at slot (t_pos & 3)
    int cur_slot = t_pos & 3;
    conv_ring[cur_slot * d_model + tid] = emb;

    // Causal taps: tap 0 = t-3, tap 1 = t-2, tap 2 = t-1, tap 3 = t (strict 0.0 padding for t < 3!)
    float c_out = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        int age = 3 - tap;
        float v = (t_pos >= age) ? conv_ring[((t_pos - age) & 3) * d_model + tid] : 0.0f;
        c_out = fmaf(__ldg(&conv_w[tid * 4 + tap]), v, c_out);
    }

    // Exact S^{D-1} unit normalization across block (eps = 1e-6)
    float wsum = warp_sum_32(c_out * c_out);
    int warp_id = tid >> 5;
    int n_warps = (d_model + 31) >> 5;
    if ((tid & 31) == 0) smem_sq[warp_id] = wsum;
    __syncthreads();

    float tot_sq = 0.0f;
    if (warp_id == 0) {
        float v = (tid < n_warps) ? smem_sq[tid] : 0.0f;
        tot_sq = warp_sum_32(v);
        if (tid == 0) {
            smem_sq[0] = rsqrtf(tot_sq + 1e-12f);
            conv_pos_io[0] = t_pos + 1;
        }
    }
    __syncthreads();
    x_out[tid] = c_out * smem_sq[0];
}

// ============================================================================
// Kernel B2: Inline [16, 11, 4] A_16 Block-H4 GEMV + SECDED + Razor 64 B .sys
//   - Exact hardware mirror of SiliquantA16BlockH4STE (0.5f * fp16_scale * H4)
//   - Half-warp (16 lanes/row, 8 rows/CTA @ 128 threads) -> 100% lane occupancy
//     at D=512 (n_grp32=16) & saturates all 48 SMs!
//   - 0-POPC dual-SWAR on hot path; cold-path heals 1-bit SEC bit-exact & emits
//     HoloSphereRazorCanary64 .sys release ops (0 PCIe traffic on clean cycles!)
// ============================================================================

__global__ __launch_bounds__(128, 4)
void v3_infer_a16_blockh4_gemv_secded_sm86(
    const float* __restrict__ x_in,             // [in_dim] (multiple of 32, <= 1024)
    const uint64_t* __restrict__ w_packed_u64,  // [out_dim, in_dim / 16]
    const __half* __restrict__ w_scales_f16,    // [out_dim, in_dim / 32]
    float* __restrict__ y_out,                  // [out_dim]
    HoloSphereRazorCanary64* host_canary,       // 64 B mapped D2H telemetry (NULL-safe)
    float out_scale,                            // optional output multiplier (e.g. logit_scale or 1.0f)
    int out_dim,
    int in_dim
) {
    __shared__ float4 smem_x4[256]; // 4,096 B: caches full x_in (up to in_dim=1024) ONCE per CTA!
    int tid         = threadIdx.x;
    int lane        = tid & 31;
    int hw_in_cta   = tid >> 4;
    int hlane       = tid & 15;
    int row         = blockIdx.x * 8 + hw_in_cta;
    bool active_row = (row < out_dim);

    int n_f4 = in_dim >> 2;
    const float4* x_g4 = reinterpret_cast<const float4*>(x_in);
    for (int i = tid; i < n_f4; i += 128) {
        smem_x4[i] = __ldg(&x_g4[i]);
    }
    __syncthreads();

    int n_grp32 = in_dim >> 5;
    const uint64_t* row_u64 = active_row ? (w_packed_u64 + int64_t(row) * int64_t(n_grp32 * 2)) : nullptr;
    const __half*   row_sc  = active_row ? (w_scales_f16 + int64_t(row) * int64_t(n_grp32)) : nullptr;

    float acc = 0.0f;
    int n_iters = (n_grp32 + 15) >> 4;
    #pragma unroll 1
    for (int it = 0; it < n_iters; ++it) {
        int g = it * 16 + hlane;
        bool valid_g = active_row && (g < n_grp32);
        uint64_t w0 = 0ULL, w1 = 0ULL;
        float sc = 0.0f;
        uint32_t odd0 = 0u, odd1 = 0u, nz = 0u;
        if (valid_g) {
            w0   = __ldg(&row_u64[2 * g + 0]);
            w1   = __ldg(&row_u64[2 * g + 1]);
            sc   = 0.5f * __half2float(__ldg(&row_sc[g]));
            odd0 = extract_a16_odd16(w0);
            odd1 = extract_a16_odd16(w1);
            nz   = dual_a16_syndrome_nonzero_swar(odd0 | (odd1 << 16));
        }
        uint32_t any = __ballot_sync(0xFFFFFFFFu, nz != 0u);
        if (__builtin_expect(any != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (nz != 0u) {
                uint32_t s0 = ::razor_a16_secded_syndrome(odd0);
                if (s0 != 0u) {
                    if (s0 & 1u) {
                        uint32_t j = (s0 >> 1) & 15u;
                        w0 ^= (1ULL << (j * 4u));
                        l_sec = 1u;
                        diag = s0 & 0x1Fu;
                    } else {
                        l_ded = 1u;
                        diag = (1u << 5) | (s0 & 0x1Fu);
                    }
                }
                uint32_t s1 = ::razor_a16_secded_syndrome(odd1);
                if (s1 != 0u) {
                    if (s1 & 1u) {
                        uint32_t j = (s1 >> 1) & 15u;
                        w1 ^= (1ULL << (j * 4u));
                        l_sec = 1u;
                        if (diag == 0u) diag = (1u << 8) | (s1 & 0x1Fu);
                    } else {
                        l_ded = 1u;
                        diag = (1u << 8) | (1u << 5) | (s1 & 0x1Fu);
                    }
                }
            }
            infer_canary_emit_sys(host_canary, l_sec, l_ded, diag, lane);
        }
        if (valid_g) {
            float dot32 = a16_blockh4_dot16(w0, smem_x4 + g * 8 + 0)
                        + a16_blockh4_dot16(w1, smem_x4 + g * 8 + 4);
            acc = fmaf(dot32, sc, acc);
        }
    }

    #pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, mask);
    }
    if (active_row && hlane == 0) {
        y_out[row] = acc * out_scale;
    }
}

// ============================================================================
// Kernel B2B: Fused A_16 Block-H4 Gate+Up GEMV + SwiGLU Activation (1 Launch!)
//   - Cuts 3 serial kernels (w_gate + w_up + swiglu_act) -> 1 CTA-shared kernel!
// ============================================================================

__global__ __launch_bounds__(128, 4)
void v3_infer_a16_swiglu_gemv_secded_sm86(
    const float* __restrict__ x_in,              // [in_dim]
    const uint64_t* __restrict__ wg_u64,         // [d_ff, in_dim / 16]
    const __half* __restrict__ wg_sc,            // [d_ff, in_dim / 32]
    const uint64_t* __restrict__ wu_u64,         // [d_ff, in_dim / 16]
    const __half* __restrict__ wu_sc,            // [d_ff, in_dim / 32]
    float* __restrict__ act_out,                 // [d_ff] = silu(wg @ x) * (wu @ x)
    HoloSphereRazorCanary64* host_canary,
    int d_ff,
    int in_dim
) {
    __shared__ float4 smem_x4[256];
    int tid         = threadIdx.x;
    int lane        = tid & 31;
    int hw_in_cta   = tid >> 4;
    int hlane       = tid & 15;
    int row         = blockIdx.x * 8 + hw_in_cta;
    bool active_row = (row < d_ff);

    int n_f4 = in_dim >> 2;
    const float4* x_g4 = reinterpret_cast<const float4*>(x_in);
    for (int i = tid; i < n_f4; i += 128) smem_x4[i] = __ldg(&x_g4[i]);
    __syncthreads();

    int n_grp32 = in_dim >> 5;
    int64_t row_off_u64 = int64_t(row) * int64_t(n_grp32 * 2);
    int64_t row_off_sc  = int64_t(row) * int64_t(n_grp32);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    int n_iters = (n_grp32 + 15) >> 4;
    #pragma unroll 1
    for (int it = 0; it < n_iters; ++it) {
        int g = it * 16 + hlane;
        bool valid_g = active_row && (g < n_grp32);

        // Gate weights
        uint64_t g0 = 0ULL, g1 = 0ULL, u0 = 0ULL, u1 = 0ULL;
        float scg = 0.0f, scu = 0.0f;
        uint32_t og0 = 0u, og1 = 0u, ou0 = 0u, ou1 = 0u, nz = 0u;
        if (valid_g) {
            g0  = __ldg(&wg_u64[row_off_u64 + 2 * g + 0]);
            g1  = __ldg(&wg_u64[row_off_u64 + 2 * g + 1]);
            scg = 0.5f * __half2float(__ldg(&wg_sc[row_off_sc + g]));
            u0  = __ldg(&wu_u64[row_off_u64 + 2 * g + 0]);
            u1  = __ldg(&wu_u64[row_off_u64 + 2 * g + 1]);
            scu = 0.5f * __half2float(__ldg(&wu_sc[row_off_sc + g]));
            og0 = extract_a16_odd16(g0); og1 = extract_a16_odd16(g1);
            ou0 = extract_a16_odd16(u0); ou1 = extract_a16_odd16(u1);
            nz  = dual_a16_syndrome_nonzero_swar(og0 | (og1 << 16))
                | dual_a16_syndrome_nonzero_swar(ou0 | (ou1 << 16));
        }
        uint32_t any = __ballot_sync(0xFFFFFFFFu, nz != 0u);
        if (__builtin_expect(any != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (nz != 0u) {
                auto fix_one = [&](uint64_t& wv, uint32_t oddv, uint32_t tag) {
                    uint32_t s = ::razor_a16_secded_syndrome(oddv);
                    if (s != 0u) {
                        if (s & 1u) { wv ^= (1ULL << (((s >> 1) & 15u) * 4u)); l_sec = 1u; if (!diag) diag = tag | (s & 0x1Fu); }
                        else        { l_ded = 1u; diag = tag | (1u << 5) | (s & 0x1Fu); }
                    }
                };
                fix_one(g0, og0, 0u); fix_one(g1, og1, 1u << 8);
                fix_one(u0, ou0, 2u << 8); fix_one(u1, ou1, 3u << 8);
            }
            infer_canary_emit_sys(host_canary, l_sec, l_ded, diag, lane);
        }
        if (valid_g) {
            acc_g = fmaf(a16_blockh4_dot16(g0, smem_x4 + g * 8 + 0) + a16_blockh4_dot16(g1, smem_x4 + g * 8 + 4), scg, acc_g);
            acc_u = fmaf(a16_blockh4_dot16(u0, smem_x4 + g * 8 + 0) + a16_blockh4_dot16(u1, smem_x4 + g * 8 + 4), scu, acc_u);
        }
    }

    #pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) {
        acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, mask);
        acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, mask);
    }
    if (active_row && hlane == 0) {
        act_out[row] = (acc_g / (1.0f + expf(-acc_g))) * acc_u;
    }
}

// ============================================================================
// Kernel B2C: Fused 3-Output Q/K/Ctrl A_16 Block-H4 GEMV (1 Launch vs 3!)
// ============================================================================

__global__ __launch_bounds__(128, 4)
void v3_infer_a16_qkc_gemv_secded_sm86(
    const float* __restrict__ x_in,
    const uint64_t* __restrict__ wq_u64, const __half* __restrict__ wq_sc,
    const uint64_t* __restrict__ wk_u64, const __half* __restrict__ wk_sc,
    const uint64_t* __restrict__ wc_u64, const __half* __restrict__ wc_sc,
    float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ c_out,
    HoloSphereRazorCanary64* host_canary,
    int d_model, int c_dim
) {
    __shared__ float4 smem_x4[256];
    int tid         = threadIdx.x;
    int lane        = tid & 31;
    int hw_in_cta   = tid >> 4;
    int hlane       = tid & 15;
    int total_rows  = 2 * d_model + c_dim;
    int g_row       = blockIdx.x * 8 + hw_in_cta;
    bool active_row = (g_row < total_rows);

    int n_f4 = d_model >> 2;
    const float4* x_g4 = reinterpret_cast<const float4*>(x_in);
    for (int i = tid; i < n_f4; i += 128) smem_x4[i] = __ldg(&x_g4[i]);
    __syncthreads();

    int n_grp32 = d_model >> 5;
    const uint64_t* row_u64 = nullptr;
    const __half*   row_sc  = nullptr;
    float*          out_ptr = nullptr;
    if (active_row) {
        if (g_row < d_model) {
            row_u64 = wq_u64 + int64_t(g_row) * int64_t(n_grp32 * 2);
            row_sc  = wq_sc  + int64_t(g_row) * int64_t(n_grp32);
            out_ptr = q_out  + g_row;
        } else if (g_row < 2 * d_model) {
            int lr  = g_row - d_model;
            row_u64 = wk_u64 + int64_t(lr) * int64_t(n_grp32 * 2);
            row_sc  = wk_sc  + int64_t(lr) * int64_t(n_grp32);
            out_ptr = k_out  + lr;
        } else {
            int lr  = g_row - 2 * d_model;
            row_u64 = wc_u64 + int64_t(lr) * int64_t(n_grp32 * 2);
            row_sc  = wc_sc  + int64_t(lr) * int64_t(n_grp32);
            out_ptr = c_out  + lr;
        }
    }

    float acc = 0.0f;
    int n_iters = (n_grp32 + 15) >> 4;
    #pragma unroll 1
    for (int it = 0; it < n_iters; ++it) {
        int g = it * 16 + hlane;
        bool valid_g = active_row && (g < n_grp32);
        uint64_t w0 = 0ULL, w1 = 0ULL;
        float sc = 0.0f;
        uint32_t odd0 = 0u, odd1 = 0u, nz = 0u;
        if (valid_g) {
            w0   = __ldg(&row_u64[2 * g + 0]);
            w1   = __ldg(&row_u64[2 * g + 1]);
            sc   = 0.5f * __half2float(__ldg(&row_sc[g]));
            odd0 = extract_a16_odd16(w0);
            odd1 = extract_a16_odd16(w1);
            nz   = dual_a16_syndrome_nonzero_swar(odd0 | (odd1 << 16));
        }
        uint32_t any = __ballot_sync(0xFFFFFFFFu, nz != 0u);
        if (__builtin_expect(any != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (nz != 0u) {
                uint32_t s0 = ::razor_a16_secded_syndrome(odd0);
                if (s0 != 0u) {
                    if (s0 & 1u) { w0 ^= (1ULL << (((s0 >> 1) & 15u) * 4u)); l_sec = 1u; diag = s0 & 0x1Fu; }
                    else         { l_ded = 1u; diag = (1u << 5) | (s0 & 0x1Fu); }
                }
                uint32_t s1 = ::razor_a16_secded_syndrome(odd1);
                if (s1 != 0u) {
                    if (s1 & 1u) { w1 ^= (1ULL << (((s1 >> 1) & 15u) * 4u)); l_sec = 1u; if (!diag) diag = (1u << 8) | (s1 & 0x1Fu); }
                    else         { l_ded = 1u; diag = (1u << 8) | (1u << 5) | (s1 & 0x1Fu); }
                }
            }
            infer_canary_emit_sys(host_canary, l_sec, l_ded, diag, lane);
        }
        if (valid_g) {
            acc = fmaf(a16_blockh4_dot16(w0, smem_x4 + g * 8 + 0) + a16_blockh4_dot16(w1, smem_x4 + g * 8 + 4), sc, acc);
        }
    }

    #pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, mask);
    }
    if (active_row && hlane == 0) {
        *out_ptr = acc;
    }
}

// ============================================================================
// Kernel B3: Single-Token Unified CellOp V_ring Hard-FMNMX Retrieval (1 CTA)
//   - Reads device-side conv_pos_io[0] - 1 so FULL 6-layer step is Graph-capturable!
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_vring_step_sm86(
    const float* __restrict__ x_in,         // [D]
    const float* __restrict__ q_ring_raw,   // [D]
    const float* __restrict__ k_ring_raw,   // [D]
    const float* __restrict__ ctrl_raw,     // [3 * H]
    float* __restrict__ k_ring_buf,         // [H, 8, 32]
    float* __restrict__ v_ring_buf,         // [H, 8, 32]
    float* __restrict__ y_mem_out,          // [D]
    float* __restrict__ recall_gate_out,    // [H]
    const int32_t* __restrict__ conv_pos_next_ptr, // [1] equals t_pos + 1 after inject_byteconv4
    int n_heads
) {
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    if (warp_id >= n_heads) return;

    int t_pos = conv_pos_next_ptr[0] - 1;
    if (t_pos < 0) t_pos = 0;

    float x_i = x_in[tid];
    float q_i = q_ring_raw[tid];
    float k_i = k_ring_raw[tid];
    q_i *= rsqrtf(warp_sum_32(q_i * q_i) + 1e-12f);
    k_i *= rsqrtf(warp_sum_32(k_i * k_i) + 1e-12f);

    float w_gate = 1.0f / (1.0f + expf(-ctrl_raw[warp_id * 3 + 0]));
    float v_gate = 1.0f / (1.0f + expf(-ctrl_raw[warp_id * 3 + 1]));
    float r_gate = 1.0f / (1.0f + expf(-ctrl_raw[warp_id * 3 + 2]));
    if (lane == 0) recall_gate_out[warp_id] = r_gate;

    // Causal ring slot update every C=32 chunk boundary (or slot 0 at t=0)
    int slot = (t_pos >> 5) & 7;
    if (t_pos == 0 || (t_pos & 31) == 0) {
        int off = (warp_id * 8 + slot) * 32 + lane;
        k_ring_buf[off] = k_i;
        float prev_v = (t_pos == 0) ? x_i : v_ring_buf[off];
        float g_w = w_gate * v_gate;
        float v_new = fmaf(1.0f - g_w, prev_v, g_w * x_i);
        v_new *= rsqrtf(warp_sum_32(v_new * v_new) + 1e-12f);
        v_ring_buf[off] = v_new;
    }

    int n_valid = (t_pos >> 5) + 1;
    if (n_valid > 8) n_valid = 8;

    float best_sc = -3.402823466e38f;
    int   best_m  = 0;
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m < n_valid) {
            float km = k_ring_buf[(warp_id * 8 + m) * 32 + lane];
            float sc = warp_sum_32(q_i * km) * 0.17677669529663688f;
            if (sc > best_sc) {
                best_sc = sc;
                best_m  = m;
            }
        }
    }
    y_mem_out[tid] = v_ring_buf[(warp_id * 8 + best_m) * 32 + lane];
}

// ============================================================================
// Kernel B4: Hyperspherical V_ring Mix -> h_gen (Exact S^{D-1} Unit Norm)
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_vring_mix_sm86(
    const float* __restrict__ x_in,         // [D]
    const float* __restrict__ mem_proj,     // [D] = w_mem_out(y_mem)
    const float* __restrict__ recall_gate,  // [H]
    const float* __restrict__ alpha_ring,   // [D]
    float* __restrict__ h_gen_out,          // [D]
    int d_model
) {
    __shared__ float smem_sq[32];
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int n_warps = (d_model + 31) >> 5;
    if (tid >= d_model) return;

    float mf = mem_proj[tid] * recall_gate[warp_id];
    float mf_w = warp_sum_32(mf * mf);
    if ((tid & 31) == 0) smem_sq[warp_id] = mf_w;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < n_warps) ? smem_sq[tid] : 0.0f;
        float tot = warp_sum_32(v);
        if (tid == 0) smem_sq[0] = rsqrtf(tot + 1e-12f);
    }
    __syncthreads();

    float mf_u = mf * smem_sq[0];
    float ga = fminf(fabsf(alpha_ring[tid]), 0.95f);
    float hg = fmaf(1.0f - ga, x_in[tid], ga * mf_u);

    float hg_w = warp_sum_32(hg * hg);
    __syncthreads();
    if ((tid & 31) == 0) smem_sq[warp_id] = hg_w;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < n_warps) ? smem_sq[tid] : 0.0f;
        float tot = warp_sum_32(v);
        if (tid == 0) smem_sq[0] = rsqrtf(tot + 1e-12f);
    }
    __syncthreads();
    h_gen_out[tid] = hg * smem_sq[0];
}

// ============================================================================
// Kernel B5: Single-Token CayleySU2 (||w||<=8.0) + W_q_fast + S_fast Step!
//   - 1 CTA (up to 16 warps for D<=512, H<=16), 0 B spills!
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_cayley_sfast_step_sm86(
    const float* __restrict__ h_gen,        // [D]
    const float3* __restrict__ raw_w_su2,   // [D/4] = w_omega(h_gen)
    const float* __restrict__ omega_scale,  // [D/4]
    const float* __restrict__ W_q_fast,     // [H, 32, 32]
    float* __restrict__ S_fast_io,          // [H, 32, 32] FP32
    float4* __restrict__ U_state_io,        // [H, 8] FP32
    float* __restrict__ y_cell_out,         // [D]
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int n_heads
) {
    __shared__ float smem_xt[16][33]; // 2,112 B (D=512 -> 16 warps)

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    if (warp_id >= n_heads) return;

    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;
    int sp_glob = warp_id * 8 + spin_id;

    // Cure F5 / Gate G4: smooth rational clamp ||w|| <= 8.0
    float3 rw = raw_w_su2[sp_glob];
    float  sc = omega_scale[sp_glob];
    float3 ws = make_float3(rw.x * sc, rw.y * sc, rw.z * sc);
    float  n2 = fmaf(ws.x, ws.x, fmaf(ws.y, ws.y, ws.z * ws.z));
    float  cl = 8.0f * rsqrtf(64.0f + n2);
    ws = make_float3(ws.x * cl, ws.y * cl, ws.z * cl);

    float4 U_cur = U_state_io[sp_glob];
    U_cur = quat_unit_norm(quat_mul(cayley_su2(ws), U_cur));
    if (lmod4 == 0) U_state_io[sp_glob] = U_cur;

    float  x_i = h_gen[tid];
    float4 xs  = gather_spinor4(x_i, base);
    float  xt  = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);
    smem_xt[warp_id][lane] = xt;
    __syncwarp();

    // Row-owned W_q_fast @ xt and S_fast @ xt (0 SHFLs!)
    const float* Wh = W_q_fast  + warp_id * 1024 + lane * 32;
    float*       Sh = S_fast_io + warp_id * 1024 + lane * 32;
    float z_slow = 0.0f;
    float s_pred = 0.0f;
    #pragma unroll
    for (int c = 0; c < 32; ++c) {
        float xtc = smem_xt[warp_id][c];
        z_slow = fmaf(__ldg(&Wh[c]), xtc, z_slow);
        s_pred = fmaf(Sh[c],         xtc, s_pred);
    }

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    float z_i   = z_slow + s_pred;
    float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
    float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
    float y_i   = fmaf(c_st, x_i, s_st * (v_i * v_inv));
    y_cell_out[tid] = y_i;

    float er = eta * (y_i - s_pred);
    #pragma unroll
    for (int c = 0; c < 32; ++c) {
        Sh[c] = fmaf(gamma, Sh[c], er * smem_xt[warp_id][c]);
    }
}

// ============================================================================
// Kernel B6: Hyperspherical Residual Step x <- norm(x + |alpha|*(norm(h) - x))
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_sphere_resid_sm86(
    const float* __restrict__ x_in,
    const float* __restrict__ h_raw,
    const float* __restrict__ alpha,
    float* __restrict__ x_out,
    int d_model
) {
    __shared__ float smem_sq[32];
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int n_warps = (d_model + 31) >> 5;
    if (tid >= d_model) return;

    float h_i = h_raw[tid];
    float hw  = warp_sum_32(h_i * h_i);
    if ((tid & 31) == 0) smem_sq[warp_id] = hw;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < n_warps) ? smem_sq[tid] : 0.0f;
        float tot = warp_sum_32(v);
        if (tid == 0) smem_sq[0] = rsqrtf(tot + 1e-12f);
    }
    __syncthreads();

    float h_u = h_i * smem_sq[0];
    float x_v = x_in[tid];
    float y_v = fmaf(fabsf(alpha[tid]), h_u - x_v, x_v);

    float yw = warp_sum_32(y_v * y_v);
    __syncthreads();
    if ((tid & 31) == 0) smem_sq[warp_id] = yw;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < n_warps) ? smem_sq[tid] : 0.0f;
        float tot = warp_sum_32(v);
        if (tid == 0) smem_sq[0] = rsqrtf(tot + 1e-12f);
    }
    __syncthreads();
    x_out[tid] = y_v * smem_sq[0];
}

// ============================================================================
// Kernel B7: Elementwise SwiGLU Activation out = silu(gate) * up
// ============================================================================

__global__ __launch_bounds__(256, 4)
void v3_infer_swiglu_act_sm86(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int d_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_ff) {
        float g = gate[idx];
        float s = g / (1.0f + expf(-g));
        out[idx] = s * up[idx];
    }
}

// ============================================================================
// Kernel B8: 23-Bit Finite Gumbel + Deterministic FMNMX Argmax (V = 260)
// ============================================================================

__global__ __launch_bounds__(288, 1)
void v3_infer_sample_gumbel23_sm86(
    const float* __restrict__ logits,     // [vocab_size] (<= 288)
    const uint32_t* __restrict__ rng_u32, // [vocab_size] (NULL when temperature == 0)
    int32_t* __restrict__ out_tok_id,     // [1]
    float* __restrict__ out_score,        // [1]
    float temperature,
    int vocab_size
) {
    __shared__ float    smem_v[9];
    __shared__ uint32_t smem_i[9];

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    float    val = -3.402823466e38f;
    uint32_t idx = uint32_t(vocab_size);
    if (tid < vocab_size) {
        float g = (temperature > 0.0f && rng_u32 != nullptr)
            ? gumbel_23bit_finite(rng_u32[tid], temperature)
            : 0.0f;
        val = logits[tid] + g;
        idx = uint32_t(tid);
    }
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float    ov = __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, idx, mask);
        bool     take = (ov > val) || ((ov == val) && (oi < idx));
        val = fmaxf(val, ov);
        idx = take ? oi : idx;
    }
    if (lane == 0) {
        smem_v[warp_id] = val;
        smem_i[warp_id] = idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float    wv = (lane < 9) ? smem_v[lane] : -3.402823466e38f;
        uint32_t wi = (lane < 9) ? smem_i[lane] : uint32_t(vocab_size);
        #pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) {
            float    ov = __shfl_xor_sync(0xFFFFFFFFu, wv, mask);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, wi, mask);
            bool     take = (ov > wv) || ((ov == wv) && (oi < wi));
            wv = fmaxf(wv, ov);
            wi = take ? oi : wi;
        }
        if (lane == 0) {
            out_tok_id[0] = int32_t(wi);
            out_score[0]  = wv;
        }
    }
}

// ============================================================================
// Kernel B9: O(1) Weft 16 B Pi_{t_0} Holonomy & State Snapshot / Rollback
//   - Copies exact 16 B HoloSpherePiRollback16 per spinor + O(1) S_fast/conv/ring
//     in ONE parallel device-side pass (0.0 bit-exact restoration on DED!)
// ============================================================================

__global__ __launch_bounds__(256, 4)
void v3_infer_weft_state_copy_sm86(
    float* __restrict__ dst_state,
    const float* __restrict__ src_state,
    int n_floats
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_floats) {
        dst_state[idx] = src_state[idx];
    }
}

// ============================================================================
// Kernel B10: 0 B VRAM Tied Orthonormal FWHT_256 x bent4 LM Head (D = 512)
// ============================================================================

__global__ __launch_bounds__(512, 1)
void v3_infer_tied_fwht256_lm_head_sm86(
    const float* __restrict__ x_in,       // [512]
    float* __restrict__ logits_out,       // [256]
    float logit_scale
) {
    __shared__ float smem[32][32]; // 4,096 B: [0..15]=input, [16..23]=ping, [24..31]=pong
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    smem[warp_id][lane] = x_in[tid];
    __syncthreads();

    float v = 0.0f;
    if (warp_id < 8) {
        int src_w = warp_id ^ (lane & 7);
        float diff = smem[src_w][lane] - smem[src_w + 8][lane];
        int b4 = __popc((lane & 3) & ((lane >> 2) & 3)) & 1;
        v = b4 ? -diff : diff;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem[16 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[16 + (warp_id ^ 1)][lane];
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem[24 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[24 + (warp_id ^ 2)][lane];
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem[16 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[16 + (warp_id ^ 4)][lane];
        v = (warp_id & 4) ? (o - v) : (v + o);
        logits_out[tid] = v * 0.17677669529663688f * logit_scale;
    }
}

} // namespace holosphere_v3_infer

extern "C" {

int launch_v3_infer_inject_byteconv4(
    const int32_t* tok_id_ptr, const float* conv_w, float* conv_ring,
    int32_t* conv_pos_io, float* x_out, int d_model, cudaStream_t stream
) {
    if (d_model <= 0 || d_model > 512 || (d_model & 31) != 0) return -1;
    holosphere_v3_infer::v3_infer_inject_byteconv4_sm86<<<1, d_model, 0, stream>>>(
        tok_id_ptr, conv_w, conv_ring, conv_pos_io, x_out, d_model);
    return int(cudaGetLastError());
}

int launch_v3_infer_a16_blockh4_gemv_secded(
    const float* x_in, const uint64_t* w_packed_u64, const void* w_scales_f16,
    float* y_out, void* host_canary, float out_scale, int out_dim, int in_dim,
    cudaStream_t stream
) {
    if (out_dim <= 0 || in_dim <= 0 || in_dim > 1024 || (in_dim & 31) != 0) return -1;
    int blocks = (out_dim + 7) / 8;
    holosphere_v3_infer::v3_infer_a16_blockh4_gemv_secded_sm86<<<blocks, 128, 0, stream>>>(
        x_in, w_packed_u64, reinterpret_cast<const __half*>(w_scales_f16),
        y_out, reinterpret_cast<HoloSphereRazorCanary64*>(host_canary),
        out_scale, out_dim, in_dim);
    return int(cudaGetLastError());
}

int launch_v3_infer_a16_swiglu_gemv_secded(
    const float* x_in, const uint64_t* wg_u64, const void* wg_sc,
    const uint64_t* wu_u64, const void* wu_sc, float* act_out,
    void* host_canary, int d_ff, int in_dim, cudaStream_t stream
) {
    if (d_ff <= 0 || in_dim <= 0 || in_dim > 1024 || (in_dim & 31) != 0) return -1;
    int blocks = (d_ff + 7) / 8;
    holosphere_v3_infer::v3_infer_a16_swiglu_gemv_secded_sm86<<<blocks, 128, 0, stream>>>(
        x_in, wg_u64, reinterpret_cast<const __half*>(wg_sc),
        wu_u64, reinterpret_cast<const __half*>(wu_sc), act_out,
        reinterpret_cast<HoloSphereRazorCanary64*>(host_canary), d_ff, in_dim);
    return int(cudaGetLastError());
}

int launch_v3_infer_a16_qkc_gemv_secded(
    const float* x_in,
    const uint64_t* wq_u64, const void* wq_sc,
    const uint64_t* wk_u64, const void* wk_sc,
    const uint64_t* wc_u64, const void* wc_sc,
    float* q_out, float* k_out, float* c_out,
    void* host_canary, int d_model, int c_dim, cudaStream_t stream
) {
    if (d_model <= 0 || d_model > 512 || (d_model & 31) != 0 || c_dim <= 0) return -1;
    int blocks = (2 * d_model + c_dim + 7) / 8;
    holosphere_v3_infer::v3_infer_a16_qkc_gemv_secded_sm86<<<blocks, 128, 0, stream>>>(
        x_in, wq_u64, reinterpret_cast<const __half*>(wq_sc),
        wk_u64, reinterpret_cast<const __half*>(wk_sc),
        wc_u64, reinterpret_cast<const __half*>(wc_sc),
        q_out, k_out, c_out, reinterpret_cast<HoloSphereRazorCanary64*>(host_canary),
        d_model, c_dim);
    return int(cudaGetLastError());
}

int launch_v3_infer_vring_step(
    const float* x_in, const float* q_ring_raw, const float* k_ring_raw,
    const float* ctrl_raw, float* k_ring_buf, float* v_ring_buf,
    float* y_mem_out, float* recall_gate_out, const int32_t* conv_pos_next_ptr, int n_heads,
    cudaStream_t stream
) {
    if (n_heads <= 0 || n_heads > 16) return -1;
    holosphere_v3_infer::v3_infer_vring_step_sm86<<<1, n_heads * 32, 0, stream>>>(
        x_in, q_ring_raw, k_ring_raw, ctrl_raw, k_ring_buf, v_ring_buf,
        y_mem_out, recall_gate_out, conv_pos_next_ptr, n_heads);
    return int(cudaGetLastError());
}

int launch_v3_infer_tied_fwht256_lm_head(
    const float* x_in, float* logits_out, float logit_scale, int d_model, int vocab_size, cudaStream_t stream
) {
    if (d_model != 512 || vocab_size < 256) return -1;
    holosphere_v3_infer::v3_infer_tied_fwht256_lm_head_sm86<<<1, 512, 0, stream>>>(
        x_in, logits_out, logit_scale);
    return int(cudaGetLastError());
}

int launch_v3_infer_vring_mix(
    const float* x_in, const float* mem_proj, const float* recall_gate,
    const float* alpha_ring, float* h_gen_out, int d_model, cudaStream_t stream
) {
    if (d_model <= 0 || d_model > 512 || (d_model & 31) != 0) return -1;
    holosphere_v3_infer::v3_infer_vring_mix_sm86<<<1, d_model, 0, stream>>>(
        x_in, mem_proj, recall_gate, alpha_ring, h_gen_out, d_model);
    return int(cudaGetLastError());
}

int launch_v3_infer_cayley_sfast_step(
    const float* h_gen, const float* raw_w_su2, const float* omega_scale,
    const float* W_q_fast, float* S_fast_io, float* U_state_io, float* y_cell_out,
    float cayley_t, float gamma, float eta, float eps, int n_heads, cudaStream_t stream
) {
    if (n_heads <= 0 || n_heads > 16) return -1;
    holosphere_v3_infer::v3_infer_cayley_sfast_step_sm86<<<1, n_heads * 32, 0, stream>>>(
        h_gen, reinterpret_cast<const float3*>(raw_w_su2), omega_scale,
        W_q_fast, S_fast_io, reinterpret_cast<float4*>(U_state_io), y_cell_out,
        cayley_t, gamma, eta, eps, n_heads);
    return int(cudaGetLastError());
}

int launch_v3_infer_sphere_resid(
    const float* x_in, const float* h_raw, const float* alpha,
    float* x_out, int d_model, cudaStream_t stream
) {
    if (d_model <= 0 || d_model > 512 || (d_model & 31) != 0) return -1;
    holosphere_v3_infer::v3_infer_sphere_resid_sm86<<<1, d_model, 0, stream>>>(
        x_in, h_raw, alpha, x_out, d_model);
    return int(cudaGetLastError());
}

int launch_v3_infer_swiglu_act(
    const float* gate, const float* up, float* out, int d_ff, cudaStream_t stream
) {
    if (d_ff <= 0) return int(cudaErrorInvalidValue);
    int blocks = (d_ff + 255) / 256;
    holosphere_v3_infer::v3_infer_swiglu_act_sm86<<<blocks, 256, 0, stream>>>(gate, up, out, d_ff);
    return int(cudaGetLastError());
}

int launch_v3_infer_sample_gumbel23(
    const float* logits, const uint32_t* rng_u32, int32_t* out_tok_id,
    float* out_score, float temperature, int vocab_size, cudaStream_t stream
) {
    if (vocab_size <= 0 || vocab_size > 288) return int(cudaErrorInvalidValue);
    holosphere_v3_infer::v3_infer_sample_gumbel23_sm86<<<1, 288, 0, stream>>>(
        logits, rng_u32, out_tok_id, out_score, temperature, vocab_size);
    return int(cudaGetLastError());
}

int launch_v3_infer_weft_state_copy(
    float* dst_state, const float* src_state, int n_floats, cudaStream_t stream
) {
    if (n_floats <= 0) return int(cudaErrorInvalidValue);
    int blocks = (n_floats + 255) / 256;
    holosphere_v3_infer::v3_infer_weft_state_copy_sm86<<<blocks, 256, 0, stream>>>(
        dst_state, src_state, n_floats);
    return int(cudaGetLastError());
}

// Host-mapped 64 B Razor canary allocator/poller (ZERO cudaMemcpy!)
int v3_infer_alloc_razor_canary64(void** host_ptr_out, void** dev_ptr_out) {
    if (!host_ptr_out || !dev_ptr_out) return int(cudaErrorInvalidValue);
    cudaError_t e = cudaHostAlloc(host_ptr_out, sizeof(HoloSphereRazorCanary64), cudaHostAllocMapped);
    if (e != cudaSuccess) return int(e);
    e = cudaHostGetDevicePointer(dev_ptr_out, *host_ptr_out, 0);
    if (e != cudaSuccess) {
        cudaFreeHost(*host_ptr_out);
        *host_ptr_out = nullptr;
        return int(e);
    }
    uint32_t* p = reinterpret_cast<uint32_t*>(*host_ptr_out);
    for (int i = 0; i < 16; ++i) p[i] = 0u;
    return 0;
}

int v3_infer_free_razor_canary64(void* host_ptr) {
    if (!host_ptr) return 0;
    return int(cudaFreeHost(host_ptr));
}

void v3_infer_poll_razor_canary64(const void* host_ptr, uint32_t out_fields[10]) {
    if (!host_ptr || !out_fields) return;
    const HoloSphereRazorCanary64* line = reinterpret_cast<const HoloSphereRazorCanary64*>(host_ptr);
    HoloSphereRazorSnapshot snap = holosphere_razor_poll_acquire(line);
    out_fields[0] = snap.sec_count_u32;
    out_fields[1] = snap.ded_count_u32;
    out_fields[2] = snap.epoch_u32;
    out_fields[3] = snap.rollback_token_u32;
    out_fields[4] = snap.non_lsb_canary_u32;
    out_fields[5] = snap.last_syndrome_u32;
    out_fields[6] = snap.ewma_ber_q16_u32;
    out_fields[7] = snap.clean_streak_u32;
    out_fields[8] = snap.hold_remaining_u32;
    out_fields[9] = snap.ded_latch_remaining_u32;
}

void v3_infer_reset_razor_canary64(void* host_ptr) {
    if (!host_ptr) return;
    uint32_t* p = reinterpret_cast<uint32_t*>(host_ptr);
    for (int i = 0; i < 16; ++i) __atomic_store_n(&p[i], 0u, __ATOMIC_RELEASE);
}

} // extern "C"
