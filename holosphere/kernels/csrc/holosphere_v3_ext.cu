// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Track S117 E2E Engine (Part A): Persistent Multi-Chunk Fused HoloSphere-V3
// Training Kernels for Bare-Metal Ampere (GA104M sm_86, 0 B Spills, 0 LDL/STL)
//
// Eliminates the 1,536 Python->CUDA ctypes forward launches + 1,536 backward
// SU(2) launches + 13-step PyTorch su2_prefix_left_action scan (W10-OBS-62):
//   1. holosphere_v3_chunk32_fwd_persistent_sm86:
//      - 1 single launch/layer for ALL n_chunks (e.g. 256 chunks @ T=8192)!
//      - Fuses CayleySU2 prefix holonomy + W_q @ x_tilde (4,224 B smem_W) +
//        S_fast rank-1 recurrence (row-owned FP32 S_row[32], 0 SHFLs!) +
//        Cayley S^31 geodesic + strict FP32 S_ckpts_f32 (Gate G2!) + U_ckpts_end!
//   2. holosphere_v3_chunk32_bwd_wmma_sm86 (W10-OBS-61):
//      - 1 single launch/layer for ALL n_chunks: 128 regs, 6,336 B SMEM,
//        28x HMMA.16816, 30 SHFLs, 0 B spills!
//   3. holosphere_v3_su2_3line_gu_bwd_persistent_sm86:
//      - 1 single launch/layer for ALL n_chunks: exact 3-line non-Abelian SU(2)
//        adjoint recurrence with optional Gate G4 chunk-boundary T_U S^3 projection!
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace holosphere_v3_ext {

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

__device__ __forceinline__ float quat_dot(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
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

__device__ __forceinline__ float3 cayley_pullback_w(float3 w, float4 gq) {
    float w2      = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv     = 1.0f / (1.0f + w2);
    float dot_wv  = fmaf(w.x, gq.y, fmaf(w.y, gq.z, w.z * gq.w));
    float coeff_w = -4.0f * inv * inv * (gq.x + dot_wv);
    return make_float3(
        fmaf(2.0f * inv, gq.y, coeff_w * w.x),
        fmaf(2.0f * inv, gq.z, coeff_w * w.y),
        fmaf(2.0f * inv, gq.w, coeff_w * w.z)
    );
}

__device__ __forceinline__ float pow_gamma_u5(float gamma, int t) {
    float res  = 1.0f;
    float base = gamma;
    if (t & 1)  res *= base;
    base *= base;
    if (t & 2)  res *= base;
    base *= base;
    if (t & 4)  res *= base;
    base *= base;
    if (t & 8)  res *= base;
    base *= base;
    if (t & 16) res *= base;
    return res;
}

__device__ __forceinline__ uint32_t pack_h2_u32(__half h0, __half h1) {
    __half2 h2 = __halves2half2(h0, h1);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ void hmma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ============================================================================
// Kernel A1: Persistent Multi-Chunk Fused Forward (1 launch for ALL n_chunks!)
//   - Fuses SU(2) prefix + W_q @ x_tilde + S_fast recurrence + S^31 geodesic
//   - Row-owned FP32 S_row[32] stays 100% in registers across T=8192 (0 SHFLs!)
//   - Stores strict FP32 S_ckpts_f32 (Gate G2) & U_ckpts_end at every C=32 chunk
// ============================================================================

__global__ __launch_bounds__(32, 16)
void holosphere_v3_chunk32_fwd_persistent_sm86(
    const float* __restrict__ x_seq,           // [B, T, D]
    const float3* __restrict__ w_su2_seq,      // [B, T, H, 8]
    const float* __restrict__ W_q,             // [H, 32, 32]
    const float* __restrict__ S_0_opt,         // [BH, 32, 32] FP32 (NULL -> zeros)
    const float4* __restrict__ U_0_opt,        // [BH, 8] FP32 (NULL -> identity)
    float* __restrict__ y_out,                 // [B, T, D]
    float* __restrict__ x_tilde_out,           // [B, T, D]
    float* __restrict__ z_slow_out,            // [B, T, D]
    float* __restrict__ S_ckpts_f32_out,       // [n_chunks, BH, 32, 32] STRICT FP32!
    float4* __restrict__ U_ckpts_end_out,      // [n_chunks, BH, 8]
    float* __restrict__ S_end_out,             // [BH, 32, 32] FP32
    float4* __restrict__ U_end_out,            // [BH, 8]
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int num_heads,
    int stride_b,
    int stride_t,
    int n_chunks,
    int total_bh
) {
    // 4,356 B SMEM/warp -> 16 warps/SM @ 69,696 B <= 102,400 B!
    __shared__ float smem_W[32][33];   // 4,224 B: W_q[h] row-major, bank-conflict-free
    __shared__ float smem_xt[33];      //   132 B: current token x_tilde[0..31]

    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    // Load W_q[h] ONCE into SMEM (coalesced over columns!)
    const float* Wh = W_q + int64_t(h_idx) * 1024;
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        smem_W[r][lane] = __ldg(&Wh[r * 32 + lane]);
    }
    __syncwarp();

    float S_row[32];
    if (S_0_opt != nullptr) {
        const float* S0_h = S_0_opt + int64_t(bh) * 1024;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            S_row[c] = __ldg(&S0_h[lane * 32 + c]);
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            S_row[c] = 0.0f;
        }
    }

    float4 U_cur = (U_0_opt != nullptr) ? U_0_opt[int64_t(bh) * 8 + spin_id] : make_float4(1.0f, 0.0f, 0.0f, 0.0f);

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    #pragma unroll 1
    for (int ch = 0; ch < n_chunks; ++ch) {
        // Snapshot STRICT FP32 S_ckpt_0 for this chunk (Gate G2!)
        float* S_ckpt_dst = S_ckpts_f32_out + (int64_t(ch) * int64_t(total_bh) + int64_t(bh)) * 1024LL;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            S_ckpt_dst[lane * 32 + c] = S_row[c];
        }

        int64_t ch_base = bh_base + int64_t(ch) * 32LL * int64_t(stride_t);
        #pragma unroll 1
        for (int t = 0; t < 32; ++t) {
            int64_t t_base = ch_base + int64_t(t) * int64_t(stride_t);
            int64_t off    = t_base + lane;
            int64_t w_off  = (t_base >> 2) + spin_id;

            float x_i = x_seq[off];
            const float3* wp = &w_su2_seq[w_off];
            float3 w_s = make_float3(__ldg(&wp->x), __ldg(&wp->y), __ldg(&wp->z));
            U_cur = quat_unit_norm(quat_mul(cayley_su2(w_s), U_cur));

            float4 xs = gather_spinor4(x_i, base);
            float  xt = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);
            smem_xt[lane]    = xt;
            x_tilde_out[off] = xt;
            __syncwarp();

            // Fused row-owned W_q @ xt AND S_t @ xt (0 SHFLs!)
            float z_slow = 0.0f;
            float s_pred = 0.0f;
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                float xtc = smem_xt[c];
                z_slow = fmaf(smem_W[lane][c], xtc, z_slow);
                s_pred = fmaf(S_row[c],        xtc, s_pred);
            }
            z_slow_out[off] = z_slow;

            float z_i   = z_slow + s_pred;
            float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
            float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
            float y_i   = fmaf(c_st, x_i, s_st * (v_i * v_inv));
            y_out[off]  = y_i;

            float er = eta * (y_i - s_pred);
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                S_row[c] = fmaf(gamma, S_row[c], er * smem_xt[c]);
            }
        }

        if (lmod4 == 0) {
            U_ckpts_end_out[(int64_t(ch) * int64_t(total_bh) + int64_t(bh)) * 8LL + spin_id] = U_cur;
        }
    }

    if (S_end_out != nullptr) {
        float* S_end_h = S_end_out + int64_t(bh) * 1024;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            S_end_h[lane * 32 + c] = S_row[c];
        }
    }
    if (U_end_out != nullptr && lmod4 == 0) {
        U_end_out[int64_t(bh) * 8 + spin_id] = U_cur;
    }
}

// ============================================================================
// Kernel A2: W10-OBS-61 Double-Woodbury Symmetric Gram HMMA.16816 Backward
//   - 128 regs, 6,336 B SMEM, 0 B spills
//   - 28x HMMA.16816, 30 SHFLs, 1 single launch for ALL n_chunks!
// ============================================================================

__global__ __launch_bounds__(32, 16)
void holosphere_v3_chunk32_bwd_wmma_sm86(
    const float* __restrict__ x_seq,
    const float* __restrict__ x_tilde_seq,
    const float* __restrict__ z_slow_seq,
    const float* __restrict__ S_ckpts_f32,
    const float* __restrict__ grad_y_seq,
    const float* __restrict__ grad_S_in,
    float* __restrict__ grad_z_slow_out,
    float* __restrict__ gx_tilde_sfast_out,
    float* __restrict__ gx_geo_out,
    float* __restrict__ grad_S_out,
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int num_heads,
    int stride_b,
    int stride_t,
    int n_chunks,
    int total_bh
) {
    alignas(16) __shared__ __half smem_xt[32][33];
    alignas(16) __shared__ __half smem_err[32][33];
    alignas(16) __shared__ __half smem_v[32][33];

    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    int gid = lane >> 2;
    int tid = lane & 3;

    const float* gS_src_ptr = grad_S_in + int64_t(bh) * 1024;
    float*       gS_dst_ptr = grad_S_out + int64_t(bh) * 1024;

    #pragma unroll 1
    for (int ch = n_chunks - 1; ch >= 0; --ch) {
        int64_t ch_base = bh_base + int64_t(ch) * 32LL * int64_t(stride_t);
        const float* S0_head = S_ckpts_f32 + (int64_t(ch) * int64_t(total_bh) + int64_t(bh)) * 1024LL;

        // Phase 1 (t = 0 -> 31)
        {
            float S_row[32];
            #pragma unroll
            for (int c = 0; c < 32; ++c) S_row[c] = __ldg(&S0_head[lane * 32 + c]);

            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                int64_t off = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                float   x_i = x_seq[off];
                float   xt  = x_tilde_seq[off];
                smem_xt[t][lane] = __float2half(xt);
                __syncwarp();

                float s_pred = 0.0f;
                #pragma unroll
                for (int c = 0; c < 32; ++c) s_pred = fmaf(S_row[c], __half2float(smem_xt[t][c]), s_pred);

                float z_i   = z_slow_seq[off] + s_pred;
                float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
                float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
                float err_i = (c_st * x_i + s_st * (v_i * v_inv)) - s_pred;

                smem_err[t][lane] = __float2half(err_i);
                smem_v[t][lane]   = __float2half(s_pred);

                float er = eta * err_i;
                #pragma unroll
                for (int c = 0; c < 32; ++c) S_row[c] = fmaf(gamma, S_row[c], er * __half2float(smem_xt[t][c]));
            }
        }
        __syncwarp();

        // Phase 1B: eta * gamma^{31-t} * (gS_end^T e_t)
        {
            float gSe_col[32];
            #pragma unroll
            for (int r = 0; r < 32; ++r) gSe_col[r] = gS_src_ptr[r * 32 + lane];
            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                float acc = 0.0f;
                #pragma unroll
                for (int r = 0; r < 32; ++r) acc = fmaf(gSe_col[r], __half2float(smem_err[t][r]), acc);
                gx_tilde_sfast_out[ch_base + int64_t(t) * int64_t(stride_t) + lane] = (eta * pow_gamma_u5(gamma, 31 - t)) * acc;
            }
        }

        // Phase 2 (t = 31 -> 0)
        {
            float gS_row[32];
            #pragma unroll
            for (int c = 0; c < 32; ++c) gS_row[c] = gS_src_ptr[lane * 32 + c];

            #pragma unroll 1
            for (int t = 31; t >= 0; --t) {
                int64_t off    = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                float   x_i    = x_seq[off];
                float   s_pred = __half2float(smem_v[t][lane]);
                float   zs_i   = z_slow_seq[off];
                float   gy_i   = grad_y_seq[off];

                float ge_i = 0.0f;
                #pragma unroll
                for (int c = 0; c < 32; ++c) ge_i = fmaf(gS_row[c], __half2float(smem_xt[t][c]), ge_i);
                ge_i *= eta;

                float z_i      = zs_i + s_pred;
                float proj     = warp_sum_32(z_i * x_i);
                float v_i      = z_i - proj * x_i;
                float v_inv    = rsqrtf(warp_sum_32(v_i * v_i) + eps);
                float v_inv3   = v_inv * v_inv * v_inv;
                float gx_next  = gy_i + ge_i;
                float gu_i     = s_st * gx_next;
                float gv_i     = fmaf(-v_inv3 * warp_sum_32(gu_i * v_i), v_i, v_inv * gu_i);
                float gv_dot_x = warp_sum_32(gv_i * x_i);
                float gz_i     = gv_i - gv_dot_x * x_i;
                float gx_geo_i = fmaf(c_st, gx_next, -fmaf(proj, gv_i, gv_dot_x * z_i));

                grad_z_slow_out[off] = gz_i;
                gx_geo_out[off]      = gx_geo_i;

                float v_t = gz_i - ge_i;
                smem_v[t][lane] = __float2half(v_t);

                #pragma unroll
                for (int c = 0; c < 32; ++c) gS_row[c] = fmaf(gamma, gS_row[c], v_t * __half2float(smem_xt[t][c]));
            }

            #pragma unroll
            for (int c = 0; c < 32; ++c) gS_dst_ptr[lane * 32 + c] = gS_row[c];
        }
        __syncwarp();

        // Phase 2B: strict FP32 gamma^t * (S_0^T v_t)
        {
            float S0_col[32];
            #pragma unroll
            for (int r = 0; r < 32; ++r) S0_col[r] = __ldg(&S0_head[r * 32 + lane]);
            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                float acc = 0.0f;
                #pragma unroll
                for (int r = 0; r < 32; ++r) acc = fmaf(S0_col[r], __half2float(smem_v[t][r]), acc);
                int64_t off = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                gx_tilde_sfast_out[off] = fmaf(pow_gamma_u5(gamma, t), acc, gx_tilde_sfast_out[off]);
            }
        }
        __syncwarp();

        // Phase 3A: 12x HMMA.16816 lower-triangle G = V E^T -> symmetric M in smem_v
        {
            float g_d0[6], g_d1[6], g_d2[6], g_d3[6];
            #pragma unroll
            for (int tile = 0; tile < 6; ++tile) {
                int m0 = (tile < 2) ? 0 : 16;
                int n0 = (tile < 2) ? (tile * 8) : ((tile - 2) * 8);
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                #pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 16) {
                    uint32_t a0 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 0], smem_v[m0 + gid + 0][k0 + 2 * tid + 1]);
                    uint32_t a1 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 0], smem_v[m0 + gid + 8][k0 + 2 * tid + 1]);
                    uint32_t a2 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 8], smem_v[m0 + gid + 0][k0 + 2 * tid + 9]);
                    uint32_t a3 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 8], smem_v[m0 + gid + 8][k0 + 2 * tid + 9]);
                    uint32_t b0 = pack_h2_u32(smem_err[n0 + gid][k0 + 2 * tid + 0], smem_err[n0 + gid][k0 + 2 * tid + 1]);
                    uint32_t b1 = pack_h2_u32(smem_err[n0 + gid][k0 + 2 * tid + 8], smem_err[n0 + gid][k0 + 2 * tid + 9]);
                    hmma_m16n8k16_f32_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
                }
                g_d0[tile] = d0; g_d1[tile] = d1; g_d2[tile] = d2; g_d3[tile] = d3;
            }
            __syncwarp();

            #pragma unroll
            for (int tile = 0; tile < 6; ++tile) {
                int m0 = (tile < 2) ? 0 : 16;
                int n0 = (tile < 2) ? (tile * 8) : ((tile - 2) * 8);
                int r0 = m0 + gid, r1 = m0 + gid + 8;
                int c0 = n0 + 2 * tid, c1 = c0 + 1;
                if (r0 > c0) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r0 - 1 - c0) * eta * g_d0[tile]);
                    smem_v[r0][c0] = mv; smem_v[c0][r0] = mv;
                } else if (r0 == c0) { smem_v[r0][c0] = __float2half(0.0f); }
                if (r0 > c1) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r0 - 1 - c1) * eta * g_d1[tile]);
                    smem_v[r0][c1] = mv; smem_v[c1][r0] = mv;
                } else if (r0 == c1) { smem_v[r0][c1] = __float2half(0.0f); }
                if (r1 > c0) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r1 - 1 - c0) * eta * g_d2[tile]);
                    smem_v[r1][c0] = mv; smem_v[c0][r1] = mv;
                } else if (r1 == c0) { smem_v[r1][c0] = __float2half(0.0f); }
                if (r1 > c1) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r1 - 1 - c1) * eta * g_d3[tile]);
                    smem_v[r1][c1] = mv; smem_v[c1][r1] = mv;
                } else if (r1 == c1) { smem_v[r1][c1] = __float2half(0.0f); }
            }
        }
        __syncwarp();

        // Phase 3B: 16x HMMA.16816 W = M @ X_tilde -> FP32 smem_w16 overlaid on dead smem_err
        {
            float (*smem_w16)[33] = reinterpret_cast<float (*)[33]>(&smem_err[0][0]);
            #pragma unroll
            for (int m0 = 0; m0 < 32; m0 += 16) {
                #pragma unroll
                for (int n0 = 0; n0 < 32; n0 += 8) {
                    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    #pragma unroll
                    for (int k0 = 0; k0 < 32; k0 += 16) {
                        uint32_t a0 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 0], smem_v[m0 + gid + 0][k0 + 2 * tid + 1]);
                        uint32_t a1 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 0], smem_v[m0 + gid + 8][k0 + 2 * tid + 1]);
                        uint32_t a2 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 8], smem_v[m0 + gid + 0][k0 + 2 * tid + 9]);
                        uint32_t a3 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 8], smem_v[m0 + gid + 8][k0 + 2 * tid + 9]);
                        uint32_t b0 = pack_h2_u32(smem_xt[k0 + 2 * tid + 0][n0 + gid], smem_xt[k0 + 2 * tid + 1][n0 + gid]);
                        uint32_t b1 = pack_h2_u32(smem_xt[k0 + 2 * tid + 8][n0 + gid], smem_xt[k0 + 2 * tid + 9][n0 + gid]);
                        hmma_m16n8k16_f32_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
                    }
                    int c0 = n0 + 2 * tid;
                    smem_w16[gid + 0][c0 + 0] = d0;
                    smem_w16[gid + 0][c0 + 1] = d1;
                    smem_w16[gid + 8][c0 + 0] = d2;
                    smem_w16[gid + 8][c0 + 1] = d3;
                }
                __syncwarp();

                #pragma unroll 1
                for (int dt = 0; dt < 16; ++dt) {
                    int64_t off = ch_base + int64_t(m0 + dt) * int64_t(stride_t) + lane;
                    gx_tilde_sfast_out[off] += smem_w16[dt][lane];
                }
                __syncwarp();
            }
        }
        gS_src_ptr = gS_dst_ptr;
    }
}

// ============================================================================
// Kernel A3: Persistent Multi-Chunk Exact 3-Line Non-Abelian SU(2) Adjoint
//   - 1 single launch/layer for ALL n_chunks (replaces 256 Python launches!)
//   - 0 B SMEM, 0 B spills, gU stays 100% in registers across T=8192!
// ============================================================================

__global__ __launch_bounds__(32, 16)
void holosphere_v3_su2_3line_gu_bwd_persistent_sm86(
    const float* __restrict__ x_seq,            // [B, T, D]
    const float* __restrict__ gx_tilde_total,   // [B, T, D]
    const float* __restrict__ gx_geo_in,        // [B, T, D]
    const float3* __restrict__ w_su2_seq,       // [B, T, H, 8]
    const float4* __restrict__ U_ckpts_end,     // [n_chunks, BH, 8]
    const float4* __restrict__ grad_U_in_opt,   // [BH, 8] (NULL -> zeros)
    float* __restrict__ grad_x_out,             // [B, T, D]
    float3* __restrict__ grad_w_su2_out,        // [B, T, H, 8]
    float4* __restrict__ grad_U_out,            // [BH, 8]
    int project_boundary_tangent,
    int num_heads,
    int stride_b,
    int stride_t,
    int n_chunks,
    int total_bh
) {
    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    float4 gU = (grad_U_in_opt != nullptr)
        ? grad_U_in_opt[int64_t(bh) * 8 + spin_id]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 1
    for (int ch = n_chunks - 1; ch >= 0; --ch) {
        int64_t ch_base = bh_base + int64_t(ch) * 32LL * int64_t(stride_t);
        float4  U_post  = U_ckpts_end[(int64_t(ch) * int64_t(total_bh) + int64_t(bh)) * 8LL + spin_id];

        #pragma unroll 1
        for (int t = 31; t >= 0; --t) {
            int64_t t_base = ch_base + int64_t(t) * int64_t(stride_t);
            int64_t off    = t_base + lane;
            int64_t w_off  = (t_base >> 2) + spin_id;

            float x_i   = x_seq[off];
            float gxt_i = gx_tilde_total[off];
            float gxg_i = gx_geo_in[off];
            const float3* wp = &w_su2_seq[w_off];
            float3 w_s  = make_float3(__ldg(&wp->x), __ldg(&wp->y), __ldg(&wp->z));
            float3 neg_w = make_float3(-w_s.x, -w_s.y, -w_s.z);
            float4 q_inv = cayley_su2(neg_w);
            float4 U_prev = quat_unit_norm(quat_mul(q_inv, U_post));

            float4 xs   = gather_spinor4(x_i, base);
            float4 gxt4 = gather_spinor4(gxt_i, base);

            float4 gx_rot4  = quat_mul(U_post, gxt4);
            grad_x_out[off] = gxg_i + select_lane4(gx_rot4, lmod4);

            float4 dU_local = quat_mul(xs, quat_conj(gxt4));
            gU              = make_float4(gU.x + dU_local.x, gU.y + dU_local.y, gU.z + dU_local.z, gU.w + dU_local.w);
            float4 gq       = quat_mul(gU, quat_conj(U_prev));
            gU              = quat_mul(q_inv, gU);

            float3 gw = cayley_pullback_w(w_s, gq);
            if (lmod4 == 0) {
                grad_w_su2_out[w_off] = gw;
            }
            U_post = U_prev;
        }

        if (project_boundary_tangent) {
            float radial = quat_dot(gU, U_post);
            gU = make_float4(
                fmaf(-radial, U_post.x, gU.x),
                fmaf(-radial, U_post.y, gU.y),
                fmaf(-radial, U_post.z, gU.z),
                fmaf(-radial, U_post.w, gU.w)
            );
        }
    }

    if (grad_U_out != nullptr && lmod4 == 0) {
        grad_U_out[int64_t(bh) * 8 + spin_id] = gU;
    }
}

// ============================================================================
// Kernel A4: Fused Single-Launch A_16 Block-H4 [16, 11, 4] STE Quantizer
//   - Replaces ~45 tiny PyTorch elementwise CUDA ops in SiliquantA16BlockH4STE
//     with ONE 0 B-SMEM, 0 B-spill warp-shuffle kernel (~40x fewer launches!)
//   - Exact hardware mirror: 0.5*(w @ H_4) -> grp32 FP16 scale -> [16,11,4]
//     nearest-codeword LSB snap -> 0.5 * (c_valid @ H_4) * scale (0.0 drift!)
// ============================================================================

__device__ __forceinline__ float warp_max_abs_32(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, mask));
    }
    return a;
}

__device__ __forceinline__ int snap_nibble_lsb_dev(float u_p, int c_p, int req_lsb) {
    if ((c_p & 1) == req_lsb) return c_p;
    int step = (u_p >= float(c_p)) ? 1 : -1;
    if (c_p == 7 && step > 0) step = -1;
    else if (c_p == -8 && step < 0) step = 1;
    return c_p + step;
}

__global__ __launch_bounds__(128, 4)
void holosphere_v3_a16_blockh4_ste_quant_sm86(
    const float* __restrict__ w_in,   // [n_grp32 * 32] contiguous FP32
    float* __restrict__ w_out,        // [n_grp32 * 32] contiguous FP32 A_16 quantized
    int n_grp32
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    if (warp_id >= n_grp32) return;

    int64_t idx = int64_t(warp_id) * 32LL + lane;
    float w_i = w_in[idx];

    // 1. 4-point Sylvester H_4 transform in 2 XOR shuffles: u_raw = 0.5 * (w @ H_4)
    float s1 = __shfl_xor_sync(0xFFFFFFFFu, w_i, 1);
    float t1 = (lane & 1) ? (s1 - w_i) : (w_i + s1);
    float s2 = __shfl_xor_sync(0xFFFFFFFFu, t1, 2);
    float u_raw = 0.5f * ((lane & 2) ? (s2 - t1) : (t1 + s2));

    // 2. Group-32 FP16 scale matching s116_a16_hamming_ste.py (clamp [1e-6, 65504*7.5] / 7.5)
    float amax = fminf(fmaxf(warp_max_abs_32(u_raw), 1e-6f), 491280.0f);
    float sc   = __half2float(__float2half(amax * (1.0f / 7.5f)));

    // 3. Nearest signed int4 nibble in [-8, +7] (round-to-nearest-even matching torch.round)
    float u_sc = fminf(fmaxf(u_raw / sc, -8.0f), 7.0f);
    int   c_i  = __float2int_rn(u_sc);
    c_i = (c_i > 7) ? 7 : ((c_i < -8) ? -8 : c_i);

    // 4. Extended Hamming [16, 11, 4] LSB parity snapping across both 16-nibble halves
    uint32_t odd32 = __ballot_sync(0xFFFFFFFFu, (c_i & 1) != 0);
    uint32_t odd16 = (lane < 16) ? (odd32 & 0xFFFFu) : (odd32 >> 16);
    int l16 = lane & 15;
    if (l16 == 1)      c_i = snap_nibble_lsb_dev(u_sc, c_i, __popc(odd16 & 0xAAA8u) & 1);
    else if (l16 == 2) c_i = snap_nibble_lsb_dev(u_sc, c_i, __popc(odd16 & 0xCCC8u) & 1);
    else if (l16 == 4) c_i = snap_nibble_lsb_dev(u_sc, c_i, __popc(odd16 & 0xF0E0u) & 1);
    else if (l16 == 8) c_i = snap_nibble_lsb_dev(u_sc, c_i, __popc(odd16 & 0xFE00u) & 1);

    odd32 = __ballot_sync(0xFFFFFFFFu, (c_i & 1) != 0);
    odd16 = (lane < 16) ? (odd32 & 0xFFFFu) : (odd32 >> 16);
    if (l16 == 0) c_i = snap_nibble_lsb_dev(u_sc, c_i, __popc(odd16 & 0xFFFEu) & 1);

    // 5. Inverse H_4 butterfly in 2 XOR shuffles -> w_quant = 0.5 * r_int8 * sc
    float cf = float(c_i);
    float r1 = __shfl_xor_sync(0xFFFFFFFFu, cf, 1);
    float v1 = (lane & 1) ? (r1 - cf) : (cf + r1);
    float r2 = __shfl_xor_sync(0xFFFFFFFFu, v1, 2);
    float r_int8 = (lane & 2) ? (r2 - v1) : (v1 + r2);
    w_out[idx] = 0.5f * r_int8 * sc;
}

} // namespace holosphere_v3_ext

extern "C" {

int launch_holosphere_v3_chunk32_fwd_persistent(
    const float* x_seq, const float* w_su2_seq, const float* W_q,
    const float* S_0_opt, const float* U_0_opt,
    float* y_out, float* x_tilde_out, float* z_slow_out,
    float* S_ckpts_f32_out, float* U_ckpts_end_out,
    float* S_end_out, float* U_end_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int n_chunks, int total_bh,
    cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0 || n_chunks <= 0) return 0;
    holosphere_v3_ext::holosphere_v3_chunk32_fwd_persistent_sm86<<<total_bh, 32, 0, stream>>>(
        x_seq, reinterpret_cast<const float3*>(w_su2_seq), W_q,
        S_0_opt, reinterpret_cast<const float4*>(U_0_opt),
        y_out, x_tilde_out, z_slow_out,
        S_ckpts_f32_out, reinterpret_cast<float4*>(U_ckpts_end_out),
        S_end_out, reinterpret_cast<float4*>(U_end_out),
        cayley_t, gamma, eta, eps, num_heads, stride_b, stride_t, n_chunks, total_bh
    );
    return int(cudaGetLastError());
}

int launch_holosphere_v3_chunk32_bwd_wmma(
    const float* x_seq, const float* x_tilde_seq, const float* z_slow_seq,
    const float* S_ckpts_f32, const float* grad_y_seq, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int n_chunks, int total_bh,
    cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0 || n_chunks <= 0) return 0;
    holosphere_v3_ext::holosphere_v3_chunk32_bwd_wmma_sm86<<<total_bh, 32, 0, stream>>>(
        x_seq, x_tilde_seq, z_slow_seq, S_ckpts_f32, grad_y_seq, grad_S_in,
        grad_z_slow_out, gx_tilde_sfast_out, gx_geo_out, grad_S_out,
        cayley_t, gamma, eta, eps, num_heads, stride_b, stride_t, n_chunks, total_bh
    );
    return int(cudaGetLastError());
}

int launch_holosphere_v3_su2_3line_gu_bwd_persistent(
    const float* x_seq, const float* gx_tilde_total, const float* gx_geo_in,
    const float* w_su2_seq, const float* U_ckpts_end, const float* grad_U_in_opt,
    float* grad_x_out, float* grad_w_su2_out, float* grad_U_out,
    int project_boundary_tangent, int num_heads, int stride_b, int stride_t,
    int n_chunks, int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0 || n_chunks <= 0) return 0;
    holosphere_v3_ext::holosphere_v3_su2_3line_gu_bwd_persistent_sm86<<<total_bh, 32, 0, stream>>>(
        x_seq, gx_tilde_total, gx_geo_in,
        reinterpret_cast<const float3*>(w_su2_seq),
        reinterpret_cast<const float4*>(U_ckpts_end),
        reinterpret_cast<const float4*>(grad_U_in_opt),
        grad_x_out, reinterpret_cast<float3*>(grad_w_su2_out),
        reinterpret_cast<float4*>(grad_U_out),
        project_boundary_tangent, num_heads, stride_b, stride_t, n_chunks, total_bh
    );
    return int(cudaGetLastError());
}

int launch_holosphere_v3_a16_blockh4_ste_quant(
    const float* w_in, float* w_out, int numel, cudaStream_t stream
) {
    if (numel <= 0 || (numel & 31) != 0) return -1;
    int n_grp32 = numel >> 5;
    int blocks  = (n_grp32 + 3) >> 2;
    holosphere_v3_ext::holosphere_v3_a16_blockh4_ste_quant_sm86<<<blocks, 128, 0, stream>>>(
        w_in, w_out, n_grp32);
    return int(cudaGetLastError());
}

} // extern "C"
