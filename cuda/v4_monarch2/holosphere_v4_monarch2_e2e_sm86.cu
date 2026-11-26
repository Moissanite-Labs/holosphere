// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// Bare-Metal Ampere sm_86 Fused HoloSphere-V4 E2E Megakernel (Refs #2)
//   Zero-HBM-Input (H_256 x bent4 byte injection) + n_layers HoloSphere CellOp
//   (SU(2) Cayley spinor + INT8 __dp4a + SWAR [16,11,4] SECDED + CTA-SMEM
//   per-lane saturating .sys CAS canary + 8-slot KV + S^31 Cayley step)
//   + Fused 16.5 KiB CTA-SMEM 2-Stage Monarch-16x16 + 512 B Byte-Bias
//   + FWHT_256 x bent4 Output Head + Gumbel-23bit Argmax.
//   Zero unions, zero reinterpret_cast, 0 B stack/register spills under
//   nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace holosphere_v4_monarch2 {

__device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float warp_max_abs_32(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, mask));
    }
    return a;
}

__device__ __forceinline__ int8_t quant_f32_to_s8(float scaled) {
    int q = __float2int_rn(scaled);
    q = (q > 127) ? 127 : ((q < -128) ? -128 : q);
    return static_cast<int8_t>(q);
}

__device__ __forceinline__ float4 cayley_su2(float3 w) {
    float w2  = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv = __frcp_rn(1.0f + w2);
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

__device__ __forceinline__ float4 quat_nr_polish(float4 q) {
    float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    float s  = fmaf(-0.5f, n2, 1.5f);
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

__device__ __forceinline__ int8_t h256_bent4_byte_inject_s8(uint8_t utf8_byte, int warp_id, int lane_id) {
    int d      = ((warp_id & 7) << 5) | (lane_id & 31);
    int db     = d ^ ((lane_id & 7) << 5);
    int bent4  = __popc((lane_id & 3) & ((lane_id >> 2) & 3)) & 1;
    int parity = (__popc(int(utf8_byte) & db) ^ bent4) & 1;
    return parity ? int8_t(-127) : int8_t(127);
}

__device__ __forceinline__ float gumbel_23bit_finite(uint32_t u32, float temperature) {
    float top23       = __uint2float_rn(u32 >> 9);
    float u           = fmaf(top23, 1.1920928955078125e-7f, 5.9604644775390625e-8f);
    float neg_lg2_u   = -__log2f(u);
    float lg2_neg_lg2 = __log2f(neg_lg2_u);
    float g           = fmaf(-0.69314718f, lg2_neg_lg2, 0.36651292f);
    return temperature * g;
}

__device__ __forceinline__ void warp_fmnmx_argmax_32(float& val, uint32_t& idx) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float    ov = __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, idx, mask);
        float    bv = fmaxf(val, ov);
        bool     take_other = (ov > val) || ((ov == val) && (oi < idx));
        val = bv;
        idx = take_other ? oi : idx;
    }
}

__device__ __forceinline__ uint32_t extract_lsb4_u32(uint32_t w4) {
    return ((w4 & 0x01010101u) * 0x01020408u) >> 24;
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

__device__ __forceinline__ uint32_t a16_secded_syndrome_cold(uint32_t w16) {
    uint32_t p0 = uint32_t(__popc(w16)) & 1u;
    uint32_t p1 = uint32_t(__popc(w16 & 0xAAAAu)) & 1u;
    uint32_t p2 = uint32_t(__popc(w16 & 0xCCCCu)) & 1u;
    uint32_t p3 = uint32_t(__popc(w16 & 0xF0F0u)) & 1u;
    uint32_t p4 = uint32_t(__popc(w16 & 0xFF00u)) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

__device__ __forceinline__ void a16_secded_classify_fix_u4_cold(
    uint4& wv, uint32_t w16, uint32_t& lane_sec, uint32_t& lane_ded, uint32_t& diag
) {
    uint32_t synd = a16_secded_syndrome_cold(w16);
    if (synd != 0u) {
        if ((synd & 1u) != 0u) {
            uint32_t j    = (synd >> 1) & 0xFu;
            uint32_t mask = 1u << ((j & 3u) << 3);
            uint32_t widx = j >> 2;
            if (widx == 0u)      wv.x ^= mask;
            else if (widx == 1u) wv.y ^= mask;
            else if (widx == 2u) wv.z ^= mask;
            else                 wv.w ^= mask;
            lane_sec += 1u;
            if (diag == 0u) diag = synd & 0x1Fu;
        } else {
            lane_ded += 1u;
            diag = (1u << 5) | (synd & 0x1Fu);
        }
    }
}

__device__ __forceinline__ void silisense_ber_canary_sat_add_sys(
    uint32_t* host_mapped_ptr,
    uint32_t sec_delta,
    uint32_t ded_delta,
    uint32_t& diag_io
) {
    uint32_t s_add = (sec_delta > 0xFFFFu) ? 0x10000u : sec_delta;
    uint32_t d_add = (ded_delta > 0xFFFFu) ? 0x10000u : ded_delta;
    uint32_t assumed = 0u;
    for (;;) {
        uint32_t old_sec = assumed & 0xFFFFu;
        uint32_t old_ded = assumed >> 16;
        uint32_t sec_sum = old_sec + s_add;
        uint32_t ded_sum = old_ded + d_add;
        bool ovf_sec = sec_sum > 0xFFFFu;
        bool ovf_ded = ded_sum > 0xFFFFu;
        uint32_t new_sec = ovf_sec ? 0xFFFFu : sec_sum;
        uint32_t new_ded = ovf_ded ? 0xFFFFu : ded_sum;
        bool ovf = ovf_sec || ovf_ded;
        if (ovf && new_ded == 0u) {
            new_ded = 1u;
        }
        uint32_t desired = (new_ded << 16) | new_sec;
        uint32_t prev = 0u;
        asm volatile(
            "atom.release.sys.global.cas.b32 %0, [%1], %2, %3;"
            : "=r"(prev)
            : "l"(host_mapped_ptr), "r"(assumed), "r"(desired)
            : "memory"
        );
        if (prev == assumed) {
            if (ovf) diag_io |= 0x80000000u;
            return;
        }
        assumed = prev;
    }
}

__device__ __forceinline__ void silisense_or_release_sys_u32(uint32_t* host_mapped_ptr, uint32_t bits) {
    asm volatile(
        "atom.release.sys.global.or.b32 _, [%0], %1;"
        :
        : "l"(host_mapped_ptr), "r"(bits)
        : "memory"
    );
}

// Raw byte-backed SMEM helpers (zero unions, zero reinterpret_cast)
__device__ __forceinline__ void smem_put_f32(unsigned char* base, int idx, float val) {
    std::memcpy(base + idx * 4, &val, sizeof(float));
}

__device__ __forceinline__ float smem_get_f32(const unsigned char* base, int idx) {
    float val;
    std::memcpy(&val, base + idx * 4, sizeof(float));
    return val;
}

__device__ __forceinline__ float fwht256_plain_8warp_raw(
    float v, int warp_id, int lane, unsigned char* buf_raw
) {
    if (warp_id < 8) {
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem_put_f32(buf_raw, warp_id * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (warp_id ^ 1) * 32 + lane);
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem_put_f32(buf_raw, (8 + warp_id) * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (8 + (warp_id ^ 2)) * 32 + lane);
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem_put_f32(buf_raw, warp_id * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (warp_id ^ 4) * 32 + lane);
        v = (warp_id & 4) ? (o - v) : (v + o);
        v *= 0.0625f;
    }
    return v;
}

__device__ __forceinline__ float fwht256_bent4_8warp_raw(
    const unsigned char* in_raw, int warp_id, int lane, unsigned char* buf_raw
) {
    float v = 0.0f;
    if (warp_id < 8) {
        int src_w   = warp_id ^ (lane & 7);
        int src_idx = (src_w << 5) | lane;
        float u_val = smem_get_f32(in_raw, src_idx);
        int b4      = __popc((lane & 3) & ((lane >> 2) & 3)) & 1;
        v = b4 ? -u_val : u_val;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem_put_f32(buf_raw, warp_id * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (warp_id ^ 1) * 32 + lane);
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem_put_f32(buf_raw, (8 + warp_id) * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (8 + (warp_id ^ 2)) * 32 + lane);
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem_put_f32(buf_raw, warp_id * 32 + lane, v);
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_get_f32(buf_raw, (warp_id ^ 4) * 32 + lane);
        v = (warp_id & 4) ? (o - v) : (v + o);
        v *= 0.17677669529663688f;
    }
    return v;
}

__device__ __forceinline__ void monarch16_transpose_matvec_16warp_raw(
    const unsigned char* B_raw,
    const unsigned char* z_in_raw,
    unsigned char* z_out_raw,
    int warp_id,
    int lane
) {
    int g      = warp_id;
    int row    = lane & 15;
    int half_k = lane >> 4;
    int c0     = half_k << 3;
    float acc  = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int h2_idx = ((g * 16 + row) * 8) + ((half_k << 2) + j);
        __half2 wh2;
        std::memcpy(&wh2, B_raw + (h2_idx * 4), sizeof(__half2));
        float2 w2 = __half22float2(wh2);
        float  x0 = smem_get_f32(z_in_raw, ((c0 + (j << 1) + 0) << 4) + g);
        float  x1 = smem_get_f32(z_in_raw, ((c0 + (j << 1) + 1) << 4) + g);
        acc = fmaf(w2.x, x0, fmaf(w2.y, x1, acc));
    }
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 16);
    if (half_k == 0) {
        smem_put_f32(z_out_raw, (g << 4) + row, acc);
    }
}

// ============================================================================
// Templated Fused HoloSphere-V3 (Scalar FWHT) vs HoloSphere-V4 (Monarch-2 + Bias + FWHT)
// ============================================================================

template <bool USE_MONARCH2_V4>
__device__ __forceinline__ void holosphere_v4_e2e_impl(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const __half* __restrict__ scalar_scale_f16,  // [1] for V3 (NULL for V4)
    const uint4* __restrict__ B1_u4,              // [512] uint4 = 8,192 B for V4 (NULL for V3)
    const uint4* __restrict__ B2_u4,              // [512] uint4 = 8,192 B for V4 (NULL for V3)
    const __half* __restrict__ bias_f16,          // [256] __half = 512 B for V4 (NULL for V3)
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out,
    uint32_t* __restrict__ dev_sys_atomic_cnt,
    float cayley_t,
    float temperature,
    int n_layers,
    int batch_size
) {
    __shared__ int8_t  smem_x_s8[2][16][32];
    __shared__ alignas(16) unsigned char smem_xmix_raw[16][32];
    __shared__ float   smem_x_scale[2][16];
    __shared__ uint4   smem_kv_key[16][8][2];
    // Overlay: 20,480 B for V4 (B1[8192] + B2[8192] + z1[1024] + z3[1024] + buf[2048])
    //          8,192 B for V3 (kv_val[8192] -> x_tmp[2048] + u[1024] + buf[2048])
    __shared__ alignas(16) unsigned char smem_overlay_raw[USE_MONARCH2_V4 ? 20480 : 8192];
    __shared__ float   smem_ep_val[8];
    __shared__ uint8_t smem_ep_idx[8];
    __shared__ uint32_t smem_cta_sec;
    __shared__ uint32_t smem_cta_ded;
    __shared__ uint32_t smem_cta_diag;

    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    if (tid == 0) {
        smem_cta_sec  = 0u;
        smem_cta_ded  = 0u;
        smem_cta_diag = 0u;
    }

    // Zero-HBM-table H_256 x bent4 byte injection
    uint8_t tok_b = byte_in[b_idx];
    int8_t  inj_q = h256_bent4_byte_inject_s8(tok_b, warp_id & 7, lane);
    if (warp_id >= 8) inj_q = int8_t(-inj_q);
    const float inv_sqrt32 = 0.17677669529663688f;
    float x_i = (inj_q > 0) ? inv_sqrt32 : -inv_sqrt32;

    smem_x_s8[0][warp_id][lane] = inj_q;
    if (lane == 0) smem_x_scale[0][warp_id] = inv_sqrt32 * (1.0f / 127.0f);

    if (lane < 16) {
        smem_kv_key[warp_id][lane >> 1][lane & 1] = __ldg(&kv_keys_u128[(b_idx * 16 + warp_id) * 16 + lane]);
    }
    uint4 kv_val_in = __ldg(&kv_vals_u128[(b_idx * 16 + warp_id) * 32 + lane]);
    std::memcpy(&smem_overlay_raw[(warp_id * 32 + lane) * 16], &kv_val_in, sizeof(uint4));

    uint4  w_lo  = make_uint4(0u, 0u, 0u, 0u);
    uint4  w_hi  = make_uint4(0u, 0u, 0u, 0u);
    float  w_sc  = 0.0f;
    float3 w_su2 = make_float3(0.0f, 0.0f, 0.0f);
    if (n_layers > 0) {
        const uint4* w0_base = W_packed_u128 + warp_id * 64;
        w_lo  = __ldg(&w0_base[0 * 32 + lane]);
        w_hi  = __ldg(&w0_base[1 * 32 + lane]);
        w_sc  = __ldg(&W_layer_scale[warp_id * 32 + lane]);
        const float3* wp0 = &w_su2_layer[warp_id * 8 + spin_id];
        w_su2 = make_float3(__ldg(&wp0->x), __ldg(&wp0->y), __ldg(&wp0->z));
    }

    float4 U_cur = U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id];
    float  t2    = cayley_t * cayley_t;
    float  inv_t = __frcp_rn(1.0f + t2);
    float  c_st  = (1.0f - t2) * inv_t;
    float  s_st  = (2.0f * cayley_t) * inv_t;

    __syncthreads();

    #pragma unroll 1
    for (int layer = 0; layer < n_layers; ++layer) {
        int cur_buf = layer & 1;
        int nxt_buf = cur_buf ^ 1;

        U_cur     = quat_nr_polish(quat_mul(cayley_su2(w_su2), U_cur));
        float4 xs = gather_spinor4(x_i, base);
        float  xt = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);

        int   partner   = warp_id ^ (1 << (layer & 3));
        float x_partner = float(smem_x_s8[cur_buf][partner][lane]) * smem_x_scale[cur_buf][partner];
        float x_mix     = fmaf(0.75f, xt, 0.25f * x_partner);

        float x_amax   = warp_max_abs_32(x_mix);
        float x_scale  = x_amax * (1.0f / 127.0f);
        float x_inv_sc = (x_amax > 1e-12f) ? (127.0f * __frcp_rn(x_amax)) : 0.0f;
        uint32_t xq_b  = uint32_t(uint8_t(quant_f32_to_s8(x_mix * x_inv_sc)));
        uint32_t xq_p0 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 0);
        uint32_t xq_p1 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 1);
        uint32_t xq_p2 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 2);
        uint32_t xq_p3 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 3);
        if (lmod4 == 0) {
            uint32_t packed_x = xq_p0 | (xq_p1 << 8) | (xq_p2 << 16) | (xq_p3 << 24);
            std::memcpy(&smem_xmix_raw[warp_id][(lane >> 2) * 4], &packed_x, sizeof(uint32_t));
        }
        __syncwarp();
        uint4 x_lo, x_hi;
        std::memcpy(&x_lo, &smem_xmix_raw[warp_id][0], sizeof(uint4));
        std::memcpy(&x_hi, &smem_xmix_raw[warp_id][16], sizeof(uint4));

        int acc = 0;
        acc = __dp4a(int(w_lo.x), int(x_lo.x), acc);
        acc = __dp4a(int(w_lo.y), int(x_lo.y), acc);
        acc = __dp4a(int(w_lo.z), int(x_lo.z), acc);
        acc = __dp4a(int(w_lo.w), int(x_lo.w), acc);
        acc = __dp4a(int(w_hi.x), int(x_hi.x), acc);
        acc = __dp4a(int(w_hi.y), int(x_hi.y), acc);
        acc = __dp4a(int(w_hi.z), int(x_hi.z), acc);
        acc = __dp4a(int(w_hi.w), int(x_hi.w), acc);

        uint32_t w16_lo = extract_lsb4_u32(w_lo.x) | (extract_lsb4_u32(w_lo.y) << 4)
                        | (extract_lsb4_u32(w_lo.z) << 8) | (extract_lsb4_u32(w_lo.w) << 12);
        uint32_t w16_hi = extract_lsb4_u32(w_hi.x) | (extract_lsb4_u32(w_hi.y) << 4)
                        | (extract_lsb4_u32(w_hi.z) << 8) | (extract_lsb4_u32(w_hi.w) << 12);
        uint32_t nz_swar  = dual_a16_syndrome_nonzero_swar(w16_lo | (w16_hi << 16));
        uint32_t err_mask = __ballot_sync(0xFFFFFFFFu, nz_swar != 0u);
        if (__builtin_expect(err_mask != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (nz_swar != 0u) {
                a16_secded_classify_fix_u4_cold(w_lo, w16_lo, l_sec, l_ded, diag);
                a16_secded_classify_fix_u4_cold(w_hi, w16_hi, l_sec, l_ded, diag);
                if (l_sec != 0u) {
                    acc = 0;
                    acc = __dp4a(int(w_lo.x), int(x_lo.x), acc);
                    acc = __dp4a(int(w_lo.y), int(x_lo.y), acc);
                    acc = __dp4a(int(w_lo.z), int(x_lo.z), acc);
                    acc = __dp4a(int(w_lo.w), int(x_lo.w), acc);
                    acc = __dp4a(int(w_hi.x), int(x_hi.x), acc);
                    acc = __dp4a(int(w_hi.y), int(x_hi.y), acc);
                    acc = __dp4a(int(w_hi.z), int(x_hi.z), acc);
                    acc = __dp4a(int(w_hi.w), int(x_hi.w), acc);
                }
            }
            uint32_t w_sec = warp_sum_u32(l_sec);
            uint32_t w_ded = warp_sum_u32(l_ded);
            int src_lane = __ffs(int(err_mask)) - 1;
            uint32_t diag_out = __shfl_sync(0xFFFFFFFFu, diag, src_lane);
            if (lane == 0 && (w_sec | w_ded) != 0u) {
                if (w_sec != 0u) atomicAdd(&smem_cta_sec, w_sec);
                if (w_ded != 0u) atomicAdd(&smem_cta_ded, w_ded);
                if (diag_out != 0u) atomicCAS(&smem_cta_diag, 0u, diag_out);
            }
        }
        float z_slow = float(acc) * w_sc * x_scale;

        int next_l = layer + 1;
        if (next_l < n_layers) {
            const uint4* wn_base = W_packed_u128 + (next_l * 16 + warp_id) * 64;
            w_lo  = __ldg(&wn_base[0 * 32 + lane]);
            w_hi  = __ldg(&wn_base[1 * 32 + lane]);
            w_sc  = __ldg(&W_layer_scale[(next_l * 16 + warp_id) * 32 + lane]);
            const float3* wpn = &w_su2_layer[(next_l * 16 + warp_id) * 8 + spin_id];
            w_su2 = make_float3(__ldg(&wpn->x), __ldg(&wpn->y), __ldg(&wpn->z));
        }

        float    kv_sc  = -3.402823466e38f;
        uint32_t kv_idx = uint32_t(lane);
        if (lane < 8) {
            uint4 k_lo = smem_kv_key[warp_id][lane][0];
            uint4 k_hi = smem_kv_key[warp_id][lane][1];
            int kacc = 0;
            kacc = __dp4a(int(k_lo.x), int(x_lo.x), kacc);
            kacc = __dp4a(int(k_lo.y), int(x_lo.y), kacc);
            kacc = __dp4a(int(k_lo.z), int(x_lo.z), kacc);
            kacc = __dp4a(int(k_lo.w), int(x_lo.w), kacc);
            kacc = __dp4a(int(k_hi.x), int(x_hi.x), kacc);
            kacc = __dp4a(int(k_hi.y), int(x_hi.y), kacc);
            kacc = __dp4a(int(k_hi.z), int(x_hi.z), kacc);
            kacc = __dp4a(int(k_hi.w), int(x_hi.w), kacc);
            kv_sc = float(kacc);
        }
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov   = __shfl_xor_sync(0xFFFFFFFFu, kv_sc, mask);
            uint32_t oi   = __shfl_xor_sync(0xFFFFFFFFu, kv_idx, mask);
            float    bv   = fmaxf(kv_sc, ov);
            bool     take = (ov > kv_sc) || ((ov == kv_sc) && (oi < kv_idx));
            kv_sc  = bv;
            kv_idx = take ? oi : kv_idx;
        }
        int   best_slot = int(__shfl_sync(0xFFFFFFFFu, kv_idx, 0)) & 7;
        __half kv_h;
        std::memcpy(&kv_h, &smem_overlay_raw[((warp_id * 8 + best_slot) * 32 + lane) * 2], sizeof(__half));
        float kv_v      = __half2float(kv_h);

        float z_tot = z_slow + kv_v;
        float v_i   = z_tot - warp_sum_32(z_tot * x_i) * x_i;
        float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + 1e-12f);
        x_i         = fmaf(c_st, x_i, s_st * (v_i * v_inv));

        float amax   = warp_max_abs_32(x_i);
        float inv_sc = (amax > 1e-12f) ? (127.0f * __frcp_rn(amax)) : 0.0f;
        smem_x_s8[nxt_buf][warp_id][lane] = quant_f32_to_s8(x_i * inv_sc);
        if (lane == 0) smem_x_scale[nxt_buf][warp_id] = amax * (1.0f / 127.0f);
        __syncthreads();
    }

    // CTA-SMEM saturating .sys canary flush (0 writes on clean steps)
    if (tid == 0 && host_packed_canary != nullptr) {
        uint32_t sec_tot = smem_cta_sec;
        uint32_t ded_tot = smem_cta_ded;
        if ((sec_tot | ded_tot) != 0u) {
            uint32_t diag_w = smem_cta_diag;
            silisense_ber_canary_sat_add_sys(host_packed_canary, sec_tot, ded_tot, diag_w);
            if (host_diag_out != nullptr) silisense_or_release_sys_u32(host_diag_out, diag_w);
            if (dev_sys_atomic_cnt != nullptr) atomicAdd(dev_sys_atomic_cnt, 1u);
        }
    }

    if (hidden_out != nullptr) hidden_out[b_idx * 512 + tid] = x_i;
    if (lmod4 == 0) U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id] = U_cur;

    // ------------------------------------------------------------------------
    // Fused Zero-HBM-Table LM Head + Gumbel-23bit Argmax (same CTA)
    // ------------------------------------------------------------------------
    float logit = 0.0f;
    if (USE_MONARCH2_V4) {
        // Coalesced 2 x LDG.128 load of 16 KiB B1+B2 + stage x_i in buf_raw[18432..20479]
        unsigned char* b1_raw  = &smem_overlay_raw[0];
        unsigned char* b2_raw  = &smem_overlay_raw[8192];
        unsigned char* z1_raw  = &smem_overlay_raw[16384];
        unsigned char* z3_raw  = &smem_overlay_raw[17408];
        unsigned char* buf_raw = &smem_overlay_raw[18432];

        uint4 b1_v = __ldg(&B1_u4[tid]);
        uint4 b2_v = __ldg(&B2_u4[tid]);
        std::memcpy(b1_raw + tid * 16, &b1_v, sizeof(uint4));
        std::memcpy(b2_raw + tid * 16, &b2_v, sizeof(uint4));
        smem_put_f32(buf_raw, tid, x_i);
        __syncthreads();

        float u = 0.0f;
        float b = 0.0f;
        if (warp_id < 8) {
            u = smem_get_f32(buf_raw, tid) - smem_get_f32(buf_raw, 256 + tid);
            b = __half2float(__ldg(&bias_f16[tid]));
        }
        __syncthreads();

        float z1 = fwht256_plain_8warp_raw(u, warp_id, lane, buf_raw);
        if (warp_id < 8) smem_put_f32(z1_raw, tid, z1);
        __syncthreads();

        monarch16_transpose_matvec_16warp_raw(b1_raw, z1_raw, z3_raw, warp_id, lane);
        __syncthreads();

        monarch16_transpose_matvec_16warp_raw(b2_raw, z3_raw, z1_raw, warp_id, lane);
        __syncthreads();

        logit = fwht256_bent4_8warp_raw(z1_raw, warp_id, lane, buf_raw) + b;
    } else {
        unsigned char* x_raw   = &smem_overlay_raw[0];
        unsigned char* u_raw   = &smem_overlay_raw[2048];
        unsigned char* buf_raw = &smem_overlay_raw[4096];
        smem_put_f32(x_raw, tid, x_i);
        __syncthreads();
        if (warp_id < 8) {
            smem_put_f32(u_raw, tid, smem_get_f32(x_raw, tid) - smem_get_f32(x_raw, 256 + tid));
        }
        __syncthreads();
        float s = (warp_id < 8) ? __half2float(__ldg(&scalar_scale_f16[0])) : 0.0f;
        logit = fwht256_bent4_8warp_raw(u_raw, warp_id, lane, buf_raw) * s;
    }

    if (warp_id < 8) {
        if (logits_out != nullptr) logits_out[b_idx * 256 + tid] = logit;
        float    score = logit + gumbel_23bit_finite(rng_u32[b_idx * 256 + tid], temperature);
        uint32_t idx   = uint32_t(tid);
        warp_fmnmx_argmax_32(score, idx);
        if (lane == 0) {
            smem_ep_val[warp_id] = score;
            smem_ep_idx[warp_id] = uint8_t(idx);
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float    w_val = (lane < 8) ? smem_ep_val[lane] : -3.402823466e38f;
        uint32_t w_idx = (lane < 8) ? uint32_t(smem_ep_idx[lane]) : 256u;
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov = __shfl_xor_sync(0xFFFFFFFFu, w_val, mask);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, w_idx, mask);
            float    bv = fmaxf(w_val, ov);
            bool     take_other = (ov > w_val) || ((ov == w_val) && (oi < w_idx));
            w_val = bv;
            w_idx = take_other ? oi : w_idx;
        }
        if (lane == 0) {
            out_byte[b_idx]  = uint8_t(w_idx);
            out_score[b_idx] = w_val;
        }
    }
}

__global__ __launch_bounds__(512, 1)
void holosphere_v3_e2e_scalar_fwht_sm86(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const __half* __restrict__ scalar_scale_f16,
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out,
    uint32_t* __restrict__ dev_sys_atomic_cnt,
    float cayley_t,
    float temperature,
    int n_layers,
    int batch_size
) {
    holosphere_v4_e2e_impl<false>(
        byte_in, W_packed_u128, W_layer_scale, w_su2_layer, kv_keys_u128, kv_vals_u128,
        scalar_scale_f16, nullptr, nullptr, nullptr, rng_u32,
        U_state_io, hidden_out, logits_out, out_byte, out_score,
        host_packed_canary, host_diag_out, dev_sys_atomic_cnt,
        cayley_t, temperature, n_layers, batch_size
    );
}

__global__ __launch_bounds__(512, 1)
void holosphere_v4_e2e_monarch2_bias_fwht_sm86(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const uint4* __restrict__ B1_u4,
    const uint4* __restrict__ B2_u4,
    const __half* __restrict__ bias_f16,
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out,
    uint32_t* __restrict__ dev_sys_atomic_cnt,
    float cayley_t,
    float temperature,
    int n_layers,
    int batch_size
) {
    holosphere_v4_e2e_impl<true>(
        byte_in, W_packed_u128, W_layer_scale, w_su2_layer, kv_keys_u128, kv_vals_u128,
        nullptr, B1_u4, B2_u4, bias_f16, rng_u32,
        U_state_io, hidden_out, logits_out, out_byte, out_score,
        host_packed_canary, host_diag_out, dev_sys_atomic_cnt,
        cayley_t, temperature, n_layers, batch_size
    );
}

} // namespace holosphere_v4_monarch2
