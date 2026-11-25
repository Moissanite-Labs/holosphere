// Bare-Metal NVIDIA Ampere (sm_86) Fused SU(2) Holonomy-Scan, Covariant (Q,K,V)
// Parallel Transport, Inverse Un-Transport, & Spherical SLERP + Safe-Hopf Kernels.
//
// Track S114 (PREREG-S114-SILISPHERE-FUSED-ENTITY-V1) / silikernels sm_86 Foundry.
//
// Core Silicon-Geometry Isomorphism:
//   1 unit quaternion q = (w, x, y, z) in SU(2) ~= S^3  <==>  1 CUDA float4 (LDG.E.128)
//   - Forward associative right-accumulating prefix product:
//       U(t) = du_0 * du_1 * ... * du_t
//     executes inside 32-lane warps via 5 __shfl_up_sync steps + hierarchical 256-step
//     block tiles with 144 B shared memory and 0 B register/stack spills, fusing
//     su2_exp(angle_scale * tanh(raw)) and covariant L(U(t)) transport of (Q, K, V)
//     in a single HBM pass.
//   - Reverse-mode adjoint reduces algebraically on SU(2) (since U_t^-1 = conj(U_t))
//     to a 3-float Lie-algebraic su(2) additive suffix sum via 5 __shfl_down_sync steps:
//       m_s = Im(g_s * conj(U_s)) in R^3,
//       S_t = sum_{s=t}^{T-1} m_s,
//       dL/d(du_t) = conj(U_{t-1}) * (0, S_t) * U_t  (t > 0),  (0, S_0) * U_0  (t == 0).
//   - Fused in-register spherical SLERP + non-antipodal acute-cone Safe-Hopf rotation
//     uses single-pass warp-reduced rsqrtf (MUFU.RSQ) with smooth eps floor.
//
// Compile & zero-spill gate:
//   nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --shared -Xcompiler -fPIC \
//        su2_holonomy_scan_sm86.cu -o libsu2_holonomy_scan_sm86.so

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace silisphere_sm86 {

constexpr int SCAN_BLOCK_THREADS = 256;
constexpr int SCAN_WARPS_PER_BLOCK = SCAN_BLOCK_THREADS / 32;

// ---------------------------------------------------------------------------
// Register-Native Unit Quaternion (float4: .x=w, .y=x, .z=y, .w=z) Primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 q_identity() {
    return make_float4(1.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float4 q_conj(float4 q) {
    return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float4 q_mul(float4 p, float4 q) {
    return make_float4(
        fmaf(p.x, q.x, fmaf(-p.y, q.y, fmaf(-p.z, q.z, -p.w * q.w))),
        fmaf(p.x, q.y, fmaf( p.y, q.x, fmaf( p.z, q.w, -p.w * q.z))),
        fmaf(p.x, q.z, fmaf(-p.y, q.w, fmaf( p.z, q.x,  p.w * q.y))),
        fmaf(p.x, q.w, fmaf( p.y, q.z, fmaf(-p.z, q.y,  p.w * q.x)))
    );
}

__device__ __forceinline__ float4 q_add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 q_norm(float4 q, float eps = 1e-6f) {
    float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    float inv = rsqrtf(fmaxf(n2, eps * eps));
    return make_float4(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
}

// Pure-imaginary left product: (0, sx, sy, sz) * u
__device__ __forceinline__ float4 imag_q_mul(float3 s, float4 u) {
    return make_float4(
        fmaf(-s.x, u.y, fmaf(-s.y, u.z, -s.z * u.w)),
        fmaf( s.x, u.x, fmaf( s.y, u.w, -s.z * u.z)),
        fmaf(-s.x, u.w, fmaf( s.y, u.x,  s.z * u.y)),
        fmaf( s.x, u.z, fmaf(-s.y, u.y,  s.z * u.x))
    );
}

// Imaginary part of (g * conj(u)) = su(2) tangent element in R^3
__device__ __forceinline__ float3 imag_mul_conj(float4 g, float4 u) {
    return make_float3(
        fmaf(-g.x, u.y, fmaf( g.y, u.x, fmaf(-g.z, u.w,  g.w * u.z))),
        fmaf(-g.x, u.z, fmaf( g.y, u.w, fmaf( g.z, u.x, -g.w * u.y))),
        fmaf(-g.x, u.w, fmaf(-g.y, u.z, fmaf( g.z, u.y,  g.w * u.x)))
    );
}

// Register-only Cody-Waite + Cephes minimax sincosf (< 1e-7 err, 0 B stack frame!)
__device__ __forceinline__ void sincosf_nospill(float x, float& s_out, float& c_out) {
    float qf = rintf(x * 0.63661977236758134308f);
    int q = __float2int_rn(qf);
    float r = fmaf(-qf, 1.57079637050628662109f, x);
    r = fmaf(-qf, -4.37113900018624283e-8f, r);
    float r2 = r * r;
    float p_s = fmaf(fmaf(-1.9515295891e-4f, r2, 8.3321608736e-3f), r2, -1.6666654611e-1f);
    float sr = fmaf(r * r2, p_s, r);
    float p_c = fmaf(fmaf(fmaf(2.443315711809948e-5f, r2, -1.388731625493765e-3f), r2, 4.166664568298827e-2f), r2, -0.5f);
    float cr = fmaf(r2, p_c, 1.0f);
    int quad = q & 3;
    float s0 = (quad & 1) ? cr : sr;
    float c0 = (quad & 1) ? sr : cr;
    s_out = (quad & 2) ? -s0 : s0;
    c_out = ((quad + 1) & 2) ? -c0 : c0;
}

__device__ __forceinline__ float4 su2_exp_from_raw(
    float r1, float r2, float r3, float angle_scale, int use_tanh, float eps = 1e-6f
) {
    float w1 = use_tanh ? (angle_scale * tanhf(r1)) : r1;
    float w2 = use_tanh ? (angle_scale * tanhf(r2)) : r2;
    float w3 = use_tanh ? (angle_scale * tanhf(r3)) : r3;
    float th2 = fmaf(w1, w1, fmaf(w2, w2, w3 * w3));
    float th = sqrtf(fmaf(eps, eps, th2));
    float s = 0.0f;
    float c = 1.0f;
    sincosf_nospill(th, s, c);
    float sc = s / th;
    return q_norm(make_float4(c, sc * w1, sc * w2, sc * w3), eps);
}

__device__ __forceinline__ float4 warp_shfl_up_q(float4 q, int delta) {
    return make_float4(
        __shfl_up_sync(0xffffffffu, q.x, delta),
        __shfl_up_sync(0xffffffffu, q.y, delta),
        __shfl_up_sync(0xffffffffu, q.z, delta),
        __shfl_up_sync(0xffffffffu, q.w, delta)
    );
}

__device__ __forceinline__ float3 warp_shfl_down_f3(float3 v, int delta) {
    return make_float3(
        __shfl_down_sync(0xffffffffu, v.x, delta),
        __shfl_down_sync(0xffffffffu, v.y, delta),
        __shfl_down_sync(0xffffffffu, v.z, delta)
    );
}

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: Fused Forward SU(2) Holonomy Prefix Scan + Covariant (Q,K,V) Transport
// Layout: fiber-major [N_fibers, T, 3] for raw_omega; [N_fibers, T, 4] float4 for U,Q,K,V.
// Grid: dim3(N_fibers), Block: dim3(256).
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(SCAN_BLOCK_THREADS, 4)
void su2_holonomy_scan_qkv_fwd_sm86(
    const float* __restrict__ raw_omega,  // [F, T, 3]
    const float4* __restrict__ q_in,      // optional [F, T] float4 (or nullptr)
    const float4* __restrict__ k_in,      // optional [F, T] float4 (or nullptr)
    const float4* __restrict__ v_in,      // optional [F, T] float4 (or nullptr)
    float4* __restrict__ u_out,           // [F, T] float4
    float4* __restrict__ q_out,           // optional [F, T] float4
    float4* __restrict__ k_out,           // optional [F, T] float4
    float4* __restrict__ v_out,           // optional [F, T] float4
    int seq_len,
    float angle_scale,
    int use_tanh,
    float eps
) {
    const int fiber = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int64_t base_q = static_cast<int64_t>(fiber) * static_cast<int64_t>(seq_len);
    const int64_t base_r = base_q * 3;

    __shared__ float4 sh_warp_totals[SCAN_WARPS_PER_BLOCK];
    __shared__ float4 sh_carry;
    if (tid == 0) {
        sh_carry = q_identity();
    }
    __syncthreads();

    for (int tile_base = 0; tile_base < seq_len; tile_base += SCAN_BLOCK_THREADS) {
        const int t = tile_base + tid;
        float4 q = q_identity();
        if (t < seq_len) {
            const int64_t r_off = base_r + static_cast<int64_t>(t) * 3;
            q = su2_exp_from_raw(raw_omega[r_off], raw_omega[r_off + 1], raw_omega[r_off + 2], angle_scale, use_tanh, eps);
        }

        // 5-step warp-native Hillis-Steele associative unit-quaternion prefix product
        #pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            float4 prev = warp_shfl_up_q(q, delta);
            if (lane >= delta) {
                q = q_norm(q_mul(prev, q), eps);
            }
        }

        if (lane == 31) {
            sh_warp_totals[wid] = q;
        }
        __syncthreads();

        // Warp 0 scans all 8 warp totals in 3 shuffles and updates inter-tile carry
        if (wid == 0) {
            float4 wval = (lane < SCAN_WARPS_PER_BLOCK) ? sh_warp_totals[lane] : q_identity();
            #pragma unroll
            for (int delta = 1; delta < SCAN_WARPS_PER_BLOCK; delta <<= 1) {
                float4 prev = warp_shfl_up_q(wval, delta);
                if (lane >= delta && lane < SCAN_WARPS_PER_BLOCK) {
                    wval = q_norm(q_mul(prev, wval), eps);
                }
            }
            float4 carry = sh_carry;
            float4 excl = warp_shfl_up_q(wval, 1);
            float4 warp_prefix = (lane == 0) ? carry : q_norm(q_mul(carry, excl), eps);
            if (lane < SCAN_WARPS_PER_BLOCK) {
                sh_warp_totals[lane] = warp_prefix;
            }
            if (lane == SCAN_WARPS_PER_BLOCK - 1) {
                sh_carry = q_norm(q_mul(carry, wval), eps);
            }
        }
        __syncthreads();

        q = q_norm(q_mul(sh_warp_totals[wid], q), eps);

        if (t < seq_len) {
            const int64_t q_off = base_q + static_cast<int64_t>(t);
            u_out[q_off] = q;
            if (q_in != nullptr && q_out != nullptr) {
                q_out[q_off] = q_mul(q, q_in[q_off]);
            }
            if (k_in != nullptr && k_out != nullptr) {
                k_out[q_off] = q_mul(q, k_in[q_off]);
            }
            if (v_in != nullptr && v_out != nullptr) {
                v_out[q_off] = q_mul(q, v_in[q_off]);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Fused Reverse-Mode Adjoint Backward Scan via su(2) float3 Suffix Sum
// Grid: dim3(N_fibers), Block: dim3(256).
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(SCAN_BLOCK_THREADS, 4)
void su2_holonomy_scan_qkv_bwd_sm86(
    const float* __restrict__ raw_omega,  // [F, T, 3]
    const float4* __restrict__ u_path,    // [F, T] float4
    const float4* __restrict__ q_in,      // optional [F, T]
    const float4* __restrict__ k_in,      // optional [F, T]
    const float4* __restrict__ v_in,      // optional [F, T]
    const float4* __restrict__ du_ext,    // optional [F, T] (e.g. from untransport_bwd)
    const float4* __restrict__ dq_out,    // optional [F, T]
    const float4* __restrict__ dk_out,    // optional [F, T]
    const float4* __restrict__ dv_out,    // optional [F, T]
    float* __restrict__ d_raw_omega,      // [F, T, 3]
    float4* __restrict__ dq_in,           // optional [F, T]
    float4* __restrict__ dk_in,           // optional [F, T]
    float4* __restrict__ dv_in,           // optional [F, T]
    int seq_len,
    float angle_scale,
    int use_tanh,
    float eps
) {
    const int fiber = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int64_t base_q = static_cast<int64_t>(fiber) * static_cast<int64_t>(seq_len);
    const int64_t base_r = base_q * 3;

    __shared__ float3 sh_warp_sums[SCAN_WARPS_PER_BLOCK];
    __shared__ float3 sh_suffix_carry;
    if (tid == 0) {
        sh_suffix_carry = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    const int n_tiles = (seq_len + SCAN_BLOCK_THREADS - 1) / SCAN_BLOCK_THREADS;
    for (int tile_idx = n_tiles - 1; tile_idx >= 0; --tile_idx) {
        const int t = tile_idx * SCAN_BLOCK_THREADS + tid;
        float4 u = q_identity();
        float3 m = make_float3(0.0f, 0.0f, 0.0f);

        if (t < seq_len) {
            const int64_t q_off = base_q + static_cast<int64_t>(t);
            u = u_path[q_off];
            float4 u_c = q_conj(u);
            float4 g_u = (du_ext != nullptr) ? du_ext[q_off] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (q_in != nullptr && dq_out != nullptr && dq_in != nullptr) {
                float4 x = q_in[q_off];
                float4 dx = dq_out[q_off];
                dq_in[q_off] = q_mul(u_c, dx);
                g_u = q_add(g_u, q_mul(dx, q_conj(x)));
            }
            if (k_in != nullptr && dk_out != nullptr && dk_in != nullptr) {
                float4 x = k_in[q_off];
                float4 dx = dk_out[q_off];
                dk_in[q_off] = q_mul(u_c, dx);
                g_u = q_add(g_u, q_mul(dx, q_conj(x)));
            }
            if (v_in != nullptr && dv_out != nullptr && dv_in != nullptr) {
                float4 x = v_in[q_off];
                float4 dx = dv_out[q_off];
                dv_in[q_off] = q_mul(u_c, dx);
                g_u = q_add(g_u, q_mul(dx, q_conj(x)));
            }
            m = imag_mul_conj(g_u, u);
        }

        // 5-step warp-native additive suffix sum in su(2) (3 FP32 registers)
        #pragma unroll
        for (int delta = 1; delta < 32; delta <<= 1) {
            float3 nxt = warp_shfl_down_f3(m, delta);
            if (lane + delta < 32) {
                m.x += nxt.x;
                m.y += nxt.y;
                m.z += nxt.z;
            }
        }

        if (lane == 0) {
            sh_warp_sums[wid] = m;
        }
        __syncthreads();

        if (wid == 0) {
            float3 wsum = (lane < SCAN_WARPS_PER_BLOCK) ? sh_warp_sums[lane] : make_float3(0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int delta = 1; delta < SCAN_WARPS_PER_BLOCK; delta <<= 1) {
                float3 nxt = warp_shfl_down_f3(wsum, delta);
                if (lane + delta < SCAN_WARPS_PER_BLOCK) {
                    wsum.x += nxt.x;
                    wsum.y += nxt.y;
                    wsum.z += nxt.z;
                }
            }
            float3 carry = sh_suffix_carry;
            float3 excl = warp_shfl_down_f3(wsum, 1);
            float3 warp_suffix = (lane == SCAN_WARPS_PER_BLOCK - 1)
                ? carry
                : make_float3(carry.x + excl.x, carry.y + excl.y, carry.z + excl.z);
            if (lane < SCAN_WARPS_PER_BLOCK) {
                sh_warp_sums[lane] = warp_suffix;
            }
            if (lane == 0) {
                sh_suffix_carry = make_float3(carry.x + wsum.x, carry.y + wsum.y, carry.z + wsum.z);
            }
        }
        __syncthreads();

        if (t < seq_len) {
            float3 ws = sh_warp_sums[wid];
            float3 s_vec = make_float3(m.x + ws.x, m.y + ws.y, m.z + ws.z);
            float4 u_bar = imag_q_mul(s_vec, u);
            float4 g_du = u_bar;
            if (t > 0) {
                float4 u_prev = u_path[base_q + static_cast<int64_t>(t - 1)];
                g_du = q_mul(q_conj(u_prev), u_bar);
            }

            // Chain exact g_du through safe_unit_norm(su2_exp(omega))
            const int64_t r_off = base_r + static_cast<int64_t>(t) * 3;
            float r1 = raw_omega[r_off];
            float r2 = raw_omega[r_off + 1];
            float r3 = raw_omega[r_off + 2];
            float th1 = use_tanh ? tanhf(r1) : 0.0f;
            float th2 = use_tanh ? tanhf(r2) : 0.0f;
            float th3 = use_tanh ? tanhf(r3) : 0.0f;
            float w1 = use_tanh ? (angle_scale * th1) : r1;
            float w2 = use_tanh ? (angle_scale * th2) : r2;
            float w3 = use_tanh ? (angle_scale * th3) : r3;
            float norm_w2 = fmaf(w1, w1, fmaf(w2, w2, w3 * w3));
            float theta = sqrtf(fmaf(eps, eps, norm_w2));
            float s_th = 0.0f;
            float c_th = 1.0f;
            sincosf_nospill(theta, s_th, c_th);
            float sinc = s_th / theta;

            // Project g_du through final safe_unit_norm(q0, eps)
            float4 q0 = make_float4(c_th, sinc * w1, sinc * w2, sinc * w3);
            float q0_n2 = fmaf(q0.x, q0.x, fmaf(q0.y, q0.y, fmaf(q0.z, q0.z, q0.w * q0.w)));
            float inv_q0 = rsqrtf(fmaxf(q0_n2, eps * eps));
            float4 du_u = make_float4(q0.x * inv_q0, q0.y * inv_q0, q0.z * inv_q0, q0.w * inv_q0);
            float dot_g_du = fmaf(g_du.x, du_u.x, fmaf(g_du.y, du_u.y, fmaf(g_du.z, du_u.z, g_du.w * du_u.w)));
            float4 g_q0 = make_float4(
                (g_du.x - dot_g_du * du_u.x) * inv_q0,
                (g_du.y - dot_g_du * du_u.y) * inv_q0,
                (g_du.z - dot_g_du * du_u.z) * inv_q0,
                (g_du.w - dot_g_du * du_u.w) * inv_q0
            );

            // Analytic d(sinc)/d(theta) / theta = (theta*cos(theta) - sin(theta)) / theta^3
            float dsinc_div_th;
            if (theta < 1e-2f) {
                float t2 = theta * theta;
                dsinc_div_th = fmaf(t2, fmaf(t2, -1.0f / 840.0f, 1.0f / 30.0f), -1.0f / 3.0f);
            } else {
                dsinc_div_th = (fmaf(theta, c_th, -s_th)) / (theta * theta * theta);
            }
            float dot_gv_w = fmaf(g_q0.y, w1, fmaf(g_q0.z, w2, g_q0.w * w3));
            float radial_coeff = fmaf(-sinc, g_q0.x, dsinc_div_th * dot_gv_w);

            float dw1 = fmaf(sinc, g_q0.y, radial_coeff * w1);
            float dw2 = fmaf(sinc, g_q0.z, radial_coeff * w2);
            float dw3 = fmaf(sinc, g_q0.w, radial_coeff * w3);

            d_raw_omega[r_off]     = use_tanh ? (dw1 * angle_scale * fmaf(-th1, th1, 1.0f)) : dw1;
            d_raw_omega[r_off + 1] = use_tanh ? (dw2 * angle_scale * fmaf(-th2, th2, 1.0f)) : dw2;
            d_raw_omega[r_off + 2] = use_tanh ? (dw3 * angle_scale * fmaf(-th3, th3, 1.0f)) : dw3;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernels 3 & 4: Covariant Inverse Un-Transport O_cov = conj(U_t) * O_t (H14c)
// Grid: dim3((N_total + 255) / 256), Block: dim3(256).
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4)
void su2_untransport_fwd_sm86(
    const float4* __restrict__ u_path,
    const float4* __restrict__ o_in,
    float4* __restrict__ o_out,
    int64_t n_quads
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_quads) {
        o_out[idx] = q_mul(q_conj(u_path[idx]), o_in[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void su2_untransport_bwd_sm86(
    const float4* __restrict__ u_path,
    const float4* __restrict__ o_in,
    const float4* __restrict__ do_out,
    float4* __restrict__ do_in,
    float4* __restrict__ du_out,
    int64_t n_quads
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_quads) {
        float4 u = u_path[idx];
        float4 o = o_in[idx];
        float4 d_o = do_out[idx];
        do_in[idx] = q_mul(u, d_o);
        du_out[idx] = q_mul(o, q_conj(d_o));
    }
}

// ---------------------------------------------------------------------------
// Kernels 5 & 6: Fused In-Register Spherical SLERP + Acute-Cone Safe-Hopf Clock
// Each 32-lane warp processes 1 token row of D floats (D % 4 == 0) in float4 quads.
// Grid: dim3((N_rows + 3) / 4), Block: dim3(128) (4 warps/CTA, 0 B spills!).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 hopf_rot_quad(float4 q, float c, float s) {
    return make_float4(
        fmaf(q.x, c, -q.y * s),
        fmaf(q.x, s,  q.y * c),
        fmaf(q.z, c,  q.w * s),
        fmaf(-q.z, s, q.w * c)
    );
}

__device__ __forceinline__ float4 hopf_rot_quad_t(float4 g, float c, float s) {
    return make_float4(
        fmaf( g.x, c,  g.y * s),
        fmaf(-g.x, s,  g.y * c),
        fmaf( g.z, c, -g.w * s),
        fmaf( g.z, s,  g.w * c)
    );
}

__global__ __launch_bounds__(128, 4)
void fused_spherical_slerp_hopf_fwd_sm86(
    const float4* __restrict__ h_in,          // [N, D/4]
    const float4* __restrict__ delta_in,      // [N, D/4]
    const float4* __restrict__ alpha_logit,   // [D/4]
    float4* __restrict__ out,                 // [N, D/4]
    int n_rows,
    int n_quads,
    float alpha_scale,                        // e.g. 0.45f * pi
    float cos_phi,
    float sin_phi,
    float eps
) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_global >= n_rows) {
        return;
    }
    const int64_t row_off = static_cast<int64_t>(warp_global) * static_cast<int64_t>(n_quads);

    // Pass 1: warp-reduced L2 norms of h and delta (Hopf rotation is orthogonal -> preserves ||delta||)
    float h2 = 0.0f;
    float d2 = 0.0f;
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = delta_in[row_off + q];
        h2 = fmaf(hv.x, hv.x, fmaf(hv.y, hv.y, fmaf(hv.z, hv.z, fmaf(hv.w, hv.w, h2))));
        d2 = fmaf(dv.x, dv.x, fmaf(dv.y, dv.y, fmaf(dv.z, dv.z, fmaf(dv.w, dv.w, d2))));
    }
    h2 = warp_reduce_sum_f32(h2);
    d2 = warp_reduce_sum_f32(d2);
    const float inv_h = rsqrtf(fmaxf(h2, eps * eps));
    const float inv_d = rsqrtf(fmaxf(d2, eps * eps));

    // Pass 2: warp-reduced L2 norm of mixed = cos(alpha)*h_u + sin(alpha)*R_phi(d_u)
    float m2 = 0.0f;
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = hopf_rot_quad(delta_in[row_off + q], cos_phi, sin_phi);
        float4 al = alpha_logit[q];
        float s0, c0, s1, c1, s2, c2, s3, c3;
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.x)), s0, c0);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.y)), s1, c1);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.z)), s2, c2);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.w)), s3, c3);
        float m0 = fmaf(c0, hv.x * inv_h, s0 * dv.x * inv_d);
        float m1 = fmaf(c1, hv.y * inv_h, s1 * dv.y * inv_d);
        float m2c = fmaf(c2, hv.z * inv_h, s2 * dv.z * inv_d);
        float m3 = fmaf(c3, hv.w * inv_h, s3 * dv.w * inv_d);
        m2 = fmaf(m0, m0, fmaf(m1, m1, fmaf(m2c, m2c, fmaf(m3, m3, m2))));
    }
    m2 = warp_reduce_sum_f32(m2);
    const float inv_m = rsqrtf(fmaf(eps, eps, m2));

    // Pass 3: emit normalized spherical output (LDG.E.128 L1 hit -> STG.E.128)
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = hopf_rot_quad(delta_in[row_off + q], cos_phi, sin_phi);
        float4 al = alpha_logit[q];
        float s0, c0, s1, c1, s2, c2, s3, c3;
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.x)), s0, c0);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.y)), s1, c1);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.z)), s2, c2);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.w)), s3, c3);
        out[row_off + q] = make_float4(
            fmaf(c0, hv.x * inv_h, s0 * dv.x * inv_d) * inv_m,
            fmaf(c1, hv.y * inv_h, s1 * dv.y * inv_d) * inv_m,
            fmaf(c2, hv.z * inv_h, s2 * dv.z * inv_d) * inv_m,
            fmaf(c3, hv.w * inv_h, s3 * dv.w * inv_d) * inv_m
        );
    }
}

__global__ __launch_bounds__(128, 4)
void fused_spherical_slerp_hopf_bwd_sm86(
    const float4* __restrict__ h_in,          // [N, D/4]
    const float4* __restrict__ delta_in,      // [N, D/4]
    const float4* __restrict__ alpha_logit,   // [D/4]
    const float4* __restrict__ grad_out,      // [N, D/4]
    float4* __restrict__ grad_h,              // [N, D/4]
    float4* __restrict__ grad_delta,          // [N, D/4]
    float* __restrict__ grad_alpha_logit,     // [D] atomicAdd (must be zero-initialized)
    int n_rows,
    int n_quads,
    float alpha_scale,
    float cos_phi,
    float sin_phi,
    float eps
) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_global >= n_rows) {
        return;
    }
    const int64_t row_off = static_cast<int64_t>(warp_global) * static_cast<int64_t>(n_quads);

    // Pass 1: ||h||^2, ||delta||^2
    float h2 = 0.0f;
    float d2 = 0.0f;
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = delta_in[row_off + q];
        h2 = fmaf(hv.x, hv.x, fmaf(hv.y, hv.y, fmaf(hv.z, hv.z, fmaf(hv.w, hv.w, h2))));
        d2 = fmaf(dv.x, dv.x, fmaf(dv.y, dv.y, fmaf(dv.z, dv.z, fmaf(dv.w, dv.w, d2))));
    }
    h2 = warp_reduce_sum_f32(h2);
    d2 = warp_reduce_sum_f32(d2);
    const bool h_active = h2 > (eps * eps);
    const bool d_active = d2 > (eps * eps);
    const float inv_h = rsqrtf(fmaxf(h2, eps * eps));
    const float inv_d = rsqrtf(fmaxf(d2, eps * eps));

    // Pass 2: ||mixed||^2 and dot(grad_out, mixed)
    float m2 = 0.0f;
    float dot_go_m = 0.0f;
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = hopf_rot_quad(delta_in[row_off + q], cos_phi, sin_phi);
        float4 go = grad_out[row_off + q];
        float4 al = alpha_logit[q];
        float s0, c0, s1, c1, s2, c2, s3, c3;
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.x)), s0, c0);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.y)), s1, c1);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.z)), s2, c2);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.w)), s3, c3);
        float m0 = fmaf(c0, hv.x * inv_h, s0 * dv.x * inv_d);
        float m1 = fmaf(c1, hv.y * inv_h, s1 * dv.y * inv_d);
        float m2c = fmaf(c2, hv.z * inv_h, s2 * dv.z * inv_d);
        float m3 = fmaf(c3, hv.w * inv_h, s3 * dv.w * inv_d);
        m2 = fmaf(m0, m0, fmaf(m1, m1, fmaf(m2c, m2c, fmaf(m3, m3, m2))));
        dot_go_m = fmaf(go.x, m0, fmaf(go.y, m1, fmaf(go.z, m2c, fmaf(go.w, m3, dot_go_m))));
    }
    m2 = warp_reduce_sum_f32(m2);
    dot_go_m = warp_reduce_sum_f32(dot_go_m);
    const float inv_m = rsqrtf(fmaf(eps, eps, m2));
    const float inv_m3 = inv_m * inv_m * inv_m;

    // Pass 3: dot(g_hu, h_u) and dot(g_du, d_u) for exact unit-norm backward
    float dot_ghu_hu = 0.0f;
    float dot_gdu_du = 0.0f;
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = hopf_rot_quad(delta_in[row_off + q], cos_phi, sin_phi);
        float4 go = grad_out[row_off + q];
        float4 al = alpha_logit[q];
        float s0, c0, s1, c1, s2, c2, s3, c3;
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.x)), s0, c0);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.y)), s1, c1);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.z)), s2, c2);
        sincosf_nospill(alpha_scale / (1.0f + expf(-al.w)), s3, c3);
        float hu0 = hv.x * inv_h, hu1 = hv.y * inv_h, hu2 = hv.z * inv_h, hu3 = hv.w * inv_h;
        float du0 = dv.x * inv_d, du1 = dv.y * inv_d, du2 = dv.z * inv_d, du3 = dv.w * inv_d;
        float gm0 = fmaf(go.x, inv_m, -dot_go_m * inv_m3 * fmaf(c0, hu0, s0 * du0));
        float gm1 = fmaf(go.y, inv_m, -dot_go_m * inv_m3 * fmaf(c1, hu1, s1 * du1));
        float gm2 = fmaf(go.z, inv_m, -dot_go_m * inv_m3 * fmaf(c2, hu2, s2 * du2));
        float gm3 = fmaf(go.w, inv_m, -dot_go_m * inv_m3 * fmaf(c3, hu3, s3 * du3));
        dot_ghu_hu = fmaf(gm0 * c0, hu0, fmaf(gm1 * c1, hu1, fmaf(gm2 * c2, hu2, fmaf(gm3 * c3, hu3, dot_ghu_hu))));
        dot_gdu_du = fmaf(gm0 * s0, du0, fmaf(gm1 * s1, du1, fmaf(gm2 * s2, du2, fmaf(gm3 * s3, du3, dot_gdu_du))));
    }
    dot_ghu_hu = warp_reduce_sum_f32(dot_ghu_hu);
    dot_gdu_du = warp_reduce_sum_f32(dot_gdu_du);

    // Pass 4: emit grad_h, grad_delta, and accumulate grad_alpha_logit
    for (int q = lane; q < n_quads; q += 32) {
        float4 hv = h_in[row_off + q];
        float4 dv = hopf_rot_quad(delta_in[row_off + q], cos_phi, sin_phi);
        float4 go = grad_out[row_off + q];
        float4 al = alpha_logit[q];
        float sg0 = 1.0f / (1.0f + expf(-al.x));
        float sg1 = 1.0f / (1.0f + expf(-al.y));
        float sg2 = 1.0f / (1.0f + expf(-al.z));
        float sg3 = 1.0f / (1.0f + expf(-al.w));
        float s0, c0, s1, c1, s2, c2, s3, c3;
        sincosf_nospill(alpha_scale * sg0, s0, c0);
        sincosf_nospill(alpha_scale * sg1, s1, c1);
        sincosf_nospill(alpha_scale * sg2, s2, c2);
        sincosf_nospill(alpha_scale * sg3, s3, c3);
        float hu0 = hv.x * inv_h, hu1 = hv.y * inv_h, hu2 = hv.z * inv_h, hu3 = hv.w * inv_h;
        float du0 = dv.x * inv_d, du1 = dv.y * inv_d, du2 = dv.z * inv_d, du3 = dv.w * inv_d;
        float gm0 = fmaf(go.x, inv_m, -dot_go_m * inv_m3 * fmaf(c0, hu0, s0 * du0));
        float gm1 = fmaf(go.y, inv_m, -dot_go_m * inv_m3 * fmaf(c1, hu1, s1 * du1));
        float gm2 = fmaf(go.z, inv_m, -dot_go_m * inv_m3 * fmaf(c2, hu2, s2 * du2));
        float gm3 = fmaf(go.w, inv_m, -dot_go_m * inv_m3 * fmaf(c3, hu3, s3 * du3));

        float ch = h_active ? dot_ghu_hu : 0.0f;
        float cd = d_active ? dot_gdu_du : 0.0f;
        grad_h[row_off + q] = make_float4(
            fmaf(-ch, hu0, gm0 * c0) * inv_h,
            fmaf(-ch, hu1, gm1 * c1) * inv_h,
            fmaf(-ch, hu2, gm2 * c2) * inv_h,
            fmaf(-ch, hu3, gm3 * c3) * inv_h
        );
        float4 gd_rot = make_float4(
            fmaf(-cd, du0, gm0 * s0) * inv_d,
            fmaf(-cd, du1, gm1 * s1) * inv_d,
            fmaf(-cd, du2, gm2 * s2) * inv_d,
            fmaf(-cd, du3, gm3 * s3) * inv_d
        );
        grad_delta[row_off + q] = hopf_rot_quad_t(gd_rot, cos_phi, sin_phi);

        atomicAdd(grad_alpha_logit + (q * 4 + 0), gm0 * fmaf(-s0, hu0, c0 * du0) * alpha_scale * sg0 * (1.0f - sg0));
        atomicAdd(grad_alpha_logit + (q * 4 + 1), gm1 * fmaf(-s1, hu1, c1 * du1) * alpha_scale * sg1 * (1.0f - sg1));
        atomicAdd(grad_alpha_logit + (q * 4 + 2), gm2 * fmaf(-s2, hu2, c2 * du2) * alpha_scale * sg2 * (1.0f - sg2));
        atomicAdd(grad_alpha_logit + (q * 4 + 3), gm3 * fmaf(-s3, hu3, c3 * du3) * alpha_scale * sg3 * (1.0f - sg3));
    }
}

// ---------------------------------------------------------------------------
// Kernels 7 & 8: Native Hopf-S^3 Quaternionic Residual Update (H6c Confirmed!)
// Per 4-block: h_next = exp(tanh(d0)) * (normalize(1, d1, d2, d3) (x) h)
// Single-pass float4 LDG.E.128 -> STG.E.128, 0 B spills, exact analytic backward!
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4)
void hopf_s3_residual_fwd_sm86(
    const float4* __restrict__ h_in,
    const float4* __restrict__ delta_in,
    float4* __restrict__ h_out,
    int64_t n_quads
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_quads) {
        float4 h = h_in[idx];
        float4 d = delta_in[idx];
        float inv_u = rsqrtf(fmaf(d.y, d.y, fmaf(d.z, d.z, fmaf(d.w, d.w, 1.0f))));
        float4 u = make_float4(inv_u, d.y * inv_u, d.z * inv_u, d.w * inv_u);
        float g = expf(tanhf(d.x));
        float4 prod = q_mul(u, h);
        h_out[idx] = make_float4(g * prod.x, g * prod.y, g * prod.z, g * prod.w);
    }
}

__global__ __launch_bounds__(256, 4)
void hopf_s3_residual_bwd_sm86(
    const float4* __restrict__ h_in,
    const float4* __restrict__ delta_in,
    const float4* __restrict__ grad_out,
    float4* __restrict__ grad_h,
    float4* __restrict__ grad_delta,
    int64_t n_quads
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_quads) {
        float4 h = h_in[idx];
        float4 d = delta_in[idx];
        float4 go = grad_out[idx];
        float inv_u = rsqrtf(fmaf(d.y, d.y, fmaf(d.z, d.z, fmaf(d.w, d.w, 1.0f))));
        float4 u = make_float4(inv_u, d.y * inv_u, d.z * inv_u, d.w * inv_u);
        float th0 = tanhf(d.x);
        float g = expf(th0);
        float4 prod = q_mul(u, h);
        float dg = fmaf(go.x, prod.x, fmaf(go.y, prod.y, fmaf(go.z, prod.z, go.w * prod.w)));
        float dd0 = dg * g * fmaf(-th0, th0, 1.0f);

        float4 g_prod = make_float4(g * go.x, g * go.y, g * go.z, g * go.w);
        grad_h[idx] = q_mul(q_conj(u), g_prod);
        float4 gu = q_mul(g_prod, q_conj(h));
        float dot_gu_u = fmaf(gu.x, u.x, fmaf(gu.y, u.y, fmaf(gu.z, u.z, gu.w * u.w)));
        grad_delta[idx] = make_float4(
            dd0,
            (gu.y - dot_gu_u * u.y) * inv_u,
            (gu.z - dot_gu_u * u.z) * inv_u,
            (gu.w - dot_gu_u * u.w) * inv_u
        );
    }
}

// ---------------------------------------------------------------------------
// Kernel 9: Register-Resident siliquant A_16 [16,11,4] Syndrome-Verified Dequant
//           Fused Into float4 Quaternion Quads + Optional Hopf S^1 Rotation!
//
// Each 16-bit codeword carries 16 binary signs (+/-1) satisfying the 5 Extended
// Hamming parity masks (0xFFFF, 0xAAAA, 0xCCCC, 0xF0F0, 0xFF00) checked in 5
// hardware __popc instructions with 0 B spills, emitting 4 float4 quaternion
// quads (16 FP32 weights) scaled by per-block scale and rotated in registers!
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t a16_syndrome_u16(uint32_t cw) {
    uint32_t s0 = static_cast<uint32_t>(__popc(cw & 0xFFFFu) & 1);
    uint32_t s1 = static_cast<uint32_t>(__popc(cw & 0xAAAAu) & 1);
    uint32_t s2 = static_cast<uint32_t>(__popc(cw & 0xCCCCu) & 1);
    uint32_t s3 = static_cast<uint32_t>(__popc(cw & 0xF0F0u) & 1);
    uint32_t s4 = static_cast<uint32_t>(__popc(cw & 0xFF00u) & 1);
    return s0 | (s1 << 1) | (s2 << 2) | (s3 << 3) | (s4 << 4);
}

__global__ __launch_bounds__(256, 4)
void siliquant_a16_dequant_hopf_sm86(
    const uint16_t* __restrict__ codewords,  // [N_blocks]
    const float* __restrict__ scales,        // [N_blocks]
    float4* __restrict__ out_quads,          // [N_blocks * 4] float4 (= 16 weights/block)
    uint32_t* __restrict__ syndrome_errors,  // [1] atomic counter (or nullptr)
    int64_t n_blocks,
    float cos_phi,
    float sin_phi
) {
    const int64_t blk = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) {
        return;
    }
    const uint32_t cw = static_cast<uint32_t>(codewords[blk]);
    const uint32_t syn = a16_syndrome_u16(cw);
    if (syn != 0u && syndrome_errors != nullptr) {
        atomicAdd(syndrome_errors, 1u);
    }
    const float sc = scales[blk];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int base_bit = q * 4;
        float4 v = make_float4(
            ((cw >> (base_bit + 0)) & 1u) ? sc : -sc,
            ((cw >> (base_bit + 1)) & 1u) ? sc : -sc,
            ((cw >> (base_bit + 2)) & 1u) ? sc : -sc,
            ((cw >> (base_bit + 3)) & 1u) ? sc : -sc
        );
        out_quads[blk * 4 + q] = hopf_rot_quad(v, cos_phi, sin_phi);
    }
}

}  // namespace silisphere_sm86

// ---------------------------------------------------------------------------
// C ABI Launch Entrypoints (for PyTorch ctypes / C++ extension / silikernels)
// ---------------------------------------------------------------------------

extern "C" {

int silisphere_su2_holonomy_fwd_f32(
    const float* raw_omega,
    const float* q_in,
    const float* k_in,
    const float* v_in,
    float* u_out,
    float* q_out,
    float* k_out,
    float* v_out,
    int n_fibers,
    int seq_len,
    float angle_scale,
    int use_tanh,
    float eps,
    void* stream_ptr
) {
    if (n_fibers <= 0 || seq_len <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    silisphere_sm86::su2_holonomy_scan_qkv_fwd_sm86<<<
        dim3(static_cast<unsigned>(n_fibers)),
        dim3(silisphere_sm86::SCAN_BLOCK_THREADS),
        0,
        stream
    >>>(
        raw_omega,
        reinterpret_cast<const float4*>(q_in),
        reinterpret_cast<const float4*>(k_in),
        reinterpret_cast<const float4*>(v_in),
        reinterpret_cast<float4*>(u_out),
        reinterpret_cast<float4*>(q_out),
        reinterpret_cast<float4*>(k_out),
        reinterpret_cast<float4*>(v_out),
        seq_len,
        angle_scale,
        use_tanh,
        eps
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_su2_holonomy_bwd_f32(
    const float* raw_omega,
    const float* u_path,
    const float* q_in,
    const float* k_in,
    const float* v_in,
    const float* du_ext,
    const float* dq_out,
    const float* dk_out,
    const float* dv_out,
    float* d_raw_omega,
    float* dq_in,
    float* dk_in,
    float* dv_in,
    int n_fibers,
    int seq_len,
    float angle_scale,
    int use_tanh,
    float eps,
    void* stream_ptr
) {
    if (n_fibers <= 0 || seq_len <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    silisphere_sm86::su2_holonomy_scan_qkv_bwd_sm86<<<
        dim3(static_cast<unsigned>(n_fibers)),
        dim3(silisphere_sm86::SCAN_BLOCK_THREADS),
        0,
        stream
    >>>(
        raw_omega,
        reinterpret_cast<const float4*>(u_path),
        reinterpret_cast<const float4*>(q_in),
        reinterpret_cast<const float4*>(k_in),
        reinterpret_cast<const float4*>(v_in),
        reinterpret_cast<const float4*>(du_ext),
        reinterpret_cast<const float4*>(dq_out),
        reinterpret_cast<const float4*>(dk_out),
        reinterpret_cast<const float4*>(dv_out),
        d_raw_omega,
        reinterpret_cast<float4*>(dq_in),
        reinterpret_cast<float4*>(dk_in),
        reinterpret_cast<float4*>(dv_in),
        seq_len,
        angle_scale,
        use_tanh,
        eps
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_hopf_s3_residual_fwd_f32(
    const float* h_in,
    const float* delta_in,
    float* h_out,
    int64_t n_quads,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    silisphere_sm86::hopf_s3_residual_fwd_sm86<<<dim3(grid), dim3(256), 0, stream>>>(
        reinterpret_cast<const float4*>(h_in),
        reinterpret_cast<const float4*>(delta_in),
        reinterpret_cast<float4*>(h_out),
        n_quads
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_hopf_s3_residual_bwd_f32(
    const float* h_in,
    const float* delta_in,
    const float* grad_out,
    float* grad_h,
    float* grad_delta,
    int64_t n_quads,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    silisphere_sm86::hopf_s3_residual_bwd_sm86<<<dim3(grid), dim3(256), 0, stream>>>(
        reinterpret_cast<const float4*>(h_in),
        reinterpret_cast<const float4*>(delta_in),
        reinterpret_cast<const float4*>(grad_out),
        reinterpret_cast<float4*>(grad_h),
        reinterpret_cast<float4*>(grad_delta),
        n_quads
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_su2_untransport_fwd_f32(
    const float* u_path,
    const float* o_in,
    float* o_out,
    int64_t n_quads,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    silisphere_sm86::su2_untransport_fwd_sm86<<<dim3(grid), dim3(256), 0, stream>>>(
        reinterpret_cast<const float4*>(u_path),
        reinterpret_cast<const float4*>(o_in),
        reinterpret_cast<float4*>(o_out),
        n_quads
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_su2_untransport_bwd_f32(
    const float* u_path,
    const float* o_in,
    const float* do_out,
    float* do_in,
    float* du_out,
    int64_t n_quads,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    silisphere_sm86::su2_untransport_bwd_sm86<<<dim3(grid), dim3(256), 0, stream>>>(
        reinterpret_cast<const float4*>(u_path),
        reinterpret_cast<const float4*>(o_in),
        reinterpret_cast<const float4*>(do_out),
        reinterpret_cast<float4*>(do_in),
        reinterpret_cast<float4*>(du_out),
        n_quads
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_spherical_slerp_hopf_fwd_f32(
    const float* h_in,
    const float* delta_in,
    const float* alpha_logit,
    float* out,
    int n_rows,
    int d_model,
    float alpha_scale,
    float cos_phi,
    float sin_phi,
    float eps,
    void* stream_ptr
) {
    if (n_rows <= 0 || d_model <= 0 || (d_model & 3) != 0) return -1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_rows + 3) / 4);
    silisphere_sm86::fused_spherical_slerp_hopf_fwd_sm86<<<dim3(grid), dim3(128), 0, stream>>>(
        reinterpret_cast<const float4*>(h_in),
        reinterpret_cast<const float4*>(delta_in),
        reinterpret_cast<const float4*>(alpha_logit),
        reinterpret_cast<float4*>(out),
        n_rows,
        d_model >> 2,
        alpha_scale,
        cos_phi,
        sin_phi,
        eps
    );
    return static_cast<int>(cudaGetLastError());
}

int silisphere_spherical_slerp_hopf_bwd_f32(
    const float* h_in,
    const float* delta_in,
    const float* alpha_logit,
    const float* grad_out,
    float* grad_h,
    float* grad_delta,
    float* grad_alpha_logit,
    int n_rows,
    int d_model,
    float alpha_scale,
    float cos_phi,
    float sin_phi,
    float eps,
    void* stream_ptr
) {
    if (n_rows <= 0 || d_model <= 0 || (d_model & 3) != 0) return -1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_rows + 3) / 4);
    silisphere_sm86::fused_spherical_slerp_hopf_bwd_sm86<<<dim3(grid), dim3(128), 0, stream>>>(
        reinterpret_cast<const float4*>(h_in),
        reinterpret_cast<const float4*>(delta_in),
        reinterpret_cast<const float4*>(alpha_logit),
        reinterpret_cast<const float4*>(grad_out),
        reinterpret_cast<float4*>(grad_h),
        reinterpret_cast<float4*>(grad_delta),
        grad_alpha_logit,
        n_rows,
        d_model >> 2,
        alpha_scale,
        cos_phi,
        sin_phi,
        eps
    );
    return static_cast<int>(cudaGetLastError());
}

int holosphere_siliquant_a16_dequant_hopf_f32(
    const uint16_t* codewords,
    const float* scales,
    float* out_weights,
    uint32_t* syndrome_errors,
    int64_t n_blocks,
    float cos_phi,
    float sin_phi,
    void* stream_ptr
) {
    if (n_blocks <= 0) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    unsigned grid = static_cast<unsigned>((n_blocks + 255) / 256);
    silisphere_sm86::siliquant_a16_dequant_hopf_sm86<<<dim3(grid), dim3(256), 0, stream>>>(
        codewords,
        scales,
        reinterpret_cast<float4*>(out_weights),
        syndrome_errors,
        n_blocks,
        cos_phi,
        sin_phi
    );
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
