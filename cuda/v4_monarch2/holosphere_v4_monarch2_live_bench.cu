// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// Live Ampere sm_86 Fused HoloSphere-V4 E2E Benchmark & NaN-Safe Parity Harness
//   Measures (Latency, Parity & SEC Heal Only; Zero Power/Energy Telemetry):
//     1. NaN-safe FP64 parity (< 1e-5, NaN-poisoned buffers) for fused V3 (1L) and
//        fused V4 Monarch-2+Bias (1L & 4L)
//     2. Exact 16/16 single-bit SEC heal inside fused V4 E2E kernel (0.0 hidden/logit err)
//     3. P0-warmed 3-trial interleaved CUDA-Graph decode latency (us/tok & tok/s)
//        for V3 (1L, 4L) and V4 Monarch-2+Bias (1L, 4L)
// ============================================================================

#include "holosphere_v4_monarch2_e2e_sm86.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA_ERR %s:%d %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(err__));                                     \
            std::exit(2);                                                                \
        }                                                                                \
    } while (0)

namespace {

static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static float rand_u(uint64_t& s, float lo, float hi) {
    double u = double(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
    return float(lo + (hi - lo) * u);
}

static uint32_t host_a16_encode(uint32_t data11) {
    static const int dpos[11] = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    uint32_t w = 0u;
    for (int i = 0; i < 11; ++i) {
        if ((data11 >> i) & 1u) w |= (1u << dpos[i]);
    }
    uint32_t p1 = __builtin_popcount(w & 0xAAAAu) & 1u;
    uint32_t p2 = __builtin_popcount(w & 0xCCCCu) & 1u;
    uint32_t p3 = __builtin_popcount(w & 0xF0F0u) & 1u;
    uint32_t p4 = __builtin_popcount(w & 0xFF00u) & 1u;
    w |= (p1 << 1) | (p2 << 2) | (p3 << 4) | (p4 << 8);
    uint32_t p0 = __builtin_popcount(w) & 1u;
    return w | p0;
}

static uint4 pack_16_s8_with_secded(const int8_t vals[16], uint32_t cw16) {
    uint8_t b[16];
    for (int i = 0; i < 16; ++i) {
        b[i] = uint8_t((uint8_t(vals[i]) & 0xFEu) | ((cw16 >> i) & 1u));
    }
    uint4 out;
    std::memcpy(&out, b, 16);
    return out;
}

static void flip_lsb_in_uint4(uint4& v, int idx16) {
    uint8_t b[16];
    std::memcpy(b, &v, 16);
    b[idx16] ^= 1u;
    std::memcpy(&v, b, 16);
}

static std::vector<double> fwht256_plain_fp64(const std::vector<double>& u) {
    std::vector<double> out(256, 0.0);
    for (int k = 0; k < 256; ++k) {
        double acc = 0.0;
        for (int j = 0; j < 256; ++j) {
            int par = __builtin_popcount(k & j) & 1;
            acc += par ? -u[j] : u[j];
        }
        out[k] = acc * 0.0625;
    }
    return out;
}

static std::vector<double> fwht256_bent4_fp64(const std::vector<double>& u) {
    const double inv_sqrt32 = 1.0 / std::sqrt(32.0);
    std::vector<double> out(256, 0.0);
    for (int tok = 0; tok < 256; ++tok) {
        double acc = 0.0;
        for (int idx = 0; idx < 256; ++idx) {
            int lane = idx & 31;
            int db   = idx ^ ((lane & 7) << 5);
            int b4   = __builtin_popcount((lane & 3) & ((lane >> 2) & 3)) & 1;
            int par  = (__builtin_popcount(tok & db) ^ b4) & 1;
            acc += (par ? -1.0 : 1.0) * inv_sqrt32 * u[idx];
        }
        out[tok] = acc;
    }
    return out;
}

static std::vector<double> monarch16_stage_fp64(
    const std::vector<__half>& B, const std::vector<double>& z_in
) {
    std::vector<double> z_out(256, 0.0);
    for (int g = 0; g < 16; ++g) {
        for (int r = 0; r < 16; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 16; ++c) {
                acc += double(__half2float(B[(g * 16 + r) * 16 + c])) * z_in[c * 16 + g];
            }
            z_out[g * 16 + r] = acc;
        }
    }
    return z_out;
}

static double max_abs_diff_nan_safe(const std::vector<double>& ref, const std::vector<float>& got) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i], g = double(got[i]);
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(m)) {
            return std::numeric_limits<double>::infinity();
        }
        double d = std::fabs(r - g);
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        if (d > m) m = d;
    }
    return m;
}

static double max_abs_diff_f32_nan_safe(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = double(a[i]), y = double(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(m)) {
            return std::numeric_limits<double>::infinity();
        }
        double d = std::fabs(x - y);
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        if (d > m) m = d;
    }
    return m;
}

static double med3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

} // namespace

int main() {
    uint64_t rng = 0x484F4C4F56342026ULL;
    const int L_MAX = 4;
    const int B = 1;
    const float cayley_t = 0.25f;
    const float temperature = 0.0f;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<uint8_t> h_byte(B, 73);
    std::vector<uint4>   h_W_clean(L_MAX * 16 * 2 * 32);
    std::vector<float>   h_Wsc(L_MAX * 16 * 32);
    std::vector<float3>  h_wsu2(L_MAX * 16 * 8);
    std::vector<uint4>   h_kvk(B * 16 * 16);
    std::vector<uint4>   h_kvv(B * 16 * 32);
    std::vector<uint32_t> h_rng(B * 256);
    std::vector<float4>  h_U(B * 16 * 8);

    __half h_scale = __float2half(1.75f);
    std::vector<__half> h_bias(256), h_B1(16 * 16 * 16), h_B2(16 * 16 * 16);
    for (int i = 0; i < 256; ++i) h_bias[i] = __float2half(rand_u(rng, -3.0f, 3.0f));
    for (size_t i = 0; i < h_B1.size(); ++i) {
        h_B1[i] = __float2half(rand_u(rng, -0.45f, 0.45f));
        h_B2[i] = __float2half(rand_u(rng, -0.45f, 0.45f));
    }

    for (int l = 0; l < L_MAX; ++l) {
        for (int w = 0; w < 16; ++w) {
            for (int lane = 0; lane < 32; ++lane) {
                int8_t lo_v[16], hi_v[16];
                for (int k = 0; k < 16; ++k) {
                    lo_v[k] = int8_t((int(splitmix64(rng) % 121) - 60) * 2);
                    hi_v[k] = int8_t((int(splitmix64(rng) % 121) - 60) * 2);
                }
                uint32_t cw_lo = host_a16_encode(uint32_t(splitmix64(rng) & 0x7FFu));
                uint32_t cw_hi = host_a16_encode(uint32_t(splitmix64(rng) & 0x7FFu));
                int base_idx = (l * 16 + w) * 64;
                h_W_clean[base_idx + 0 * 32 + lane] = pack_16_s8_with_secded(lo_v, cw_lo);
                h_W_clean[base_idx + 1 * 32 + lane] = pack_16_s8_with_secded(hi_v, cw_hi);
                h_Wsc[(l * 16 + w) * 32 + lane] = rand_u(rng, 0.002f, 0.015f);
            }
            for (int sp = 0; sp < 8; ++sp) {
                h_wsu2[(l * 16 + w) * 8 + sp] = make_float3(
                    rand_u(rng, -0.15f, 0.15f), rand_u(rng, -0.15f, 0.15f), rand_u(rng, -0.15f, 0.15f)
                );
            }
        }
    }
    for (size_t i = 0; i < h_kvk.size(); ++i) {
        h_kvk[i] = make_uint4(uint32_t(splitmix64(rng)), uint32_t(splitmix64(rng)),
                              uint32_t(splitmix64(rng)), uint32_t(splitmix64(rng)));
    }
    for (size_t i = 0; i < h_kvv.size(); ++i) {
        __half h8[8];
        for (int j = 0; j < 8; ++j) h8[j] = __float2half(rand_u(rng, -0.15f, 0.15f));
        std::memcpy(&h_kvv[i], h8, 16);
    }
    for (size_t i = 0; i < h_rng.size(); ++i) h_rng[i] = uint32_t(splitmix64(rng));
    for (size_t i = 0; i < h_U.size(); ++i) h_U[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

    uint8_t*  d_byte = nullptr;
    uint4*    d_W = nullptr;
    float*    d_Wsc = nullptr;
    float3*   d_wsu2 = nullptr;
    uint4*    d_kvk = nullptr;
    uint4*    d_kvv = nullptr;
    __half*   d_sc = nullptr;
    uint4*    d_B1 = nullptr;
    uint4*    d_B2 = nullptr;
    __half*   d_bias = nullptr;
    uint32_t* d_rng = nullptr;
    float4*   d_U = nullptr;
    float*    d_hid = nullptr;
    float*    d_logits = nullptr;
    uint8_t*  d_outb = nullptr;
    float*    d_outs = nullptr;
    uint32_t* d_sys_cnt = nullptr;
    uint32_t* h_packed_canary = nullptr;
    uint32_t* d_packed_canary = nullptr;
    uint32_t* h_diag_canary = nullptr;
    uint32_t* d_diag_canary = nullptr;

    CUDA_CHECK(cudaMalloc(&d_byte, B));
    CUDA_CHECK(cudaMalloc(&d_W, h_W_clean.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_Wsc, h_Wsc.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wsu2, h_wsu2.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_kvk, h_kvk.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_kvv, h_kvv.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_sc, sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B1, h_B1.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B2, h_B2.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, h_bias.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_rng, h_rng.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_U, h_U.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_hid, B * 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, B * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outb, B));
    CUDA_CHECK(cudaMalloc(&d_outs, B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sys_cnt, sizeof(uint32_t)));
    CUDA_CHECK(cudaHostAlloc(&h_packed_canary, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_packed_canary, h_packed_canary, 0));
    CUDA_CHECK(cudaHostAlloc(&h_diag_canary, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_diag_canary, h_diag_canary, 0));

    CUDA_CHECK(cudaMemcpy(d_byte, h_byte.data(), B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W_clean.data(), h_W_clean.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wsc, h_Wsc.data(), h_Wsc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wsu2, h_wsu2.data(), h_wsu2.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvk, h_kvk.data(), h_kvk.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvv, h_kvv.data(), h_kvv.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc, &h_scale, sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B1.data(), h_B1.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B2.data(), h_B2.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rng, h_rng.data(), h_rng.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    auto poison = [&]() {
        CUDA_CHECK(cudaMemset(d_hid, 0xFF, B * 512 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_logits, 0xFF, B * 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_outs, 0xFF, B * sizeof(float)));
    };

    auto ref_v4_logits = [&](const std::vector<float>& hid) -> std::vector<double> {
        std::vector<double> u(256);
        for (int i = 0; i < 256; ++i) u[i] = double(hid[i]) - double(hid[256 + i]);
        auto z1 = fwht256_plain_fp64(u);
        auto z3 = monarch16_stage_fp64(h_B1, z1);
        auto z5 = monarch16_stage_fp64(h_B2, z3);
        auto lg = fwht256_bent4_fp64(z5);
        for (int i = 0; i < 256; ++i) lg[i] += double(__half2float(h_bias[i]));
        return lg;
    };

    std::vector<float> h_hid(512), h_lg(256);

    // 1. V3 1L fused parity
    *h_packed_canary = 0u; *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison();
    holosphere_v4_monarch2::holosphere_v3_e2e_scalar_fwht_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_sc, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs, d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, 1, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hid.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lg.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<double> u_v3(256);
    for (int i = 0; i < 256; ++i) u_v3[i] = double(h_hid[i]) - double(h_hid[256 + i]);
    auto ref_v3 = fwht256_bent4_fp64(u_v3);
    double s_ref = double(__half2float(h_scale));
    for (int i = 0; i < 256; ++i) ref_v3[i] *= s_ref;
    double err_v3_1l = max_abs_diff_nan_safe(ref_v3, h_lg);

    // 2. V4 1L fused parity
    *h_packed_canary = 0u; *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison();
    holosphere_v4_monarch2::holosphere_v4_e2e_monarch2_bias_fwht_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_B1, d_B2, d_bias, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs, d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, 1, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hid.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lg.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    double err_v4_1l = max_abs_diff_nan_safe(ref_v4_logits(h_hid), h_lg);

    // 3. V4 4L fused parity + clean canary
    *h_packed_canary = 0u; *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison();
    holosphere_v4_monarch2::holosphere_v4_e2e_monarch2_bias_fwht_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_B1, d_B2, d_bias, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs, d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, 4, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_hid_4l_clean(512), h_lg_4l_clean(256);
    CUDA_CHECK(cudaMemcpy(h_hid_4l_clean.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lg_4l_clean.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    double err_v4_4l = max_abs_diff_nan_safe(ref_v4_logits(h_hid_4l_clean), h_lg_4l_clean);
    uint32_t clean_packed = *h_packed_canary;
    uint32_t clean_sys = 0u;
    CUDA_CHECK(cudaMemcpy(&clean_sys, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 4. Live 16/16 Single-Bit SEC Heal inside Fused V4 4L Kernel
    std::vector<uint4> h_W_sec16 = h_W_clean;
    for (int bit = 0; bit < 16; ++bit) flip_lsb_in_uint4(h_W_sec16[bit * 64 + 0], bit);
    CUDA_CHECK(cudaMemcpy(d_W, h_W_sec16.data(), h_W_sec16.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    *h_packed_canary = 0u; *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison();
    holosphere_v4_monarch2::holosphere_v4_e2e_monarch2_bias_fwht_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_B1, d_B2, d_bias, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs, d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, 4, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_hid_sec16(512), h_lg_sec16(256);
    CUDA_CHECK(cudaMemcpy(h_hid_sec16.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lg_sec16.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    double sec16_hid_err = max_abs_diff_f32_nan_safe(h_hid_4l_clean, h_hid_sec16);
    double sec16_lg_err  = max_abs_diff_f32_nan_safe(h_lg_4l_clean, h_lg_sec16);
    uint32_t sec16_packed = *h_packed_canary;
    uint32_t sec16_sys = 0u;
    CUDA_CHECK(cudaMemcpy(&sec16_sys, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Restore clean weights for latency benchmark
    CUDA_CHECK(cudaMemcpy(d_W, h_W_clean.data(), h_W_clean.size() * sizeof(uint4), cudaMemcpyHostToDevice));

    // 5. P0 Warmup + 3-Trial Interleaved CUDA-Graph Latency (V3 1L, V4 1L, V3 4L, V4 4L)
    cudaStream_t st;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaStreamCreate(&st));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    for (int w = 0; w < 1024; ++w) {
        holosphere_v4_monarch2::holosphere_v4_e2e_monarch2_bias_fwht_sm86<<<B, 512, 0, st>>>(
            d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_B1, d_B2, d_bias, d_rng,
            d_U, nullptr, nullptr, d_outb, d_outs, d_packed_canary, nullptr, nullptr,
            cayley_t, temperature, 4, B
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));

    const int total_iters = 8192;
    const int graph_unroll = 64;
    const int n_launches = total_iters / graph_unroll;

    auto bench_cfg = [&](bool is_v4, int layers) -> double {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < graph_unroll; ++k) {
            if (is_v4) {
                holosphere_v4_monarch2::holosphere_v4_e2e_monarch2_bias_fwht_sm86<<<B, 512, 0, st>>>(
                    d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_B1, d_B2, d_bias, d_rng,
                    d_U, nullptr, nullptr, d_outb, d_outs, d_packed_canary, nullptr, nullptr,
                    cayley_t, temperature, layers, B
                );
            } else {
                holosphere_v4_monarch2::holosphere_v3_e2e_scalar_fwht_sm86<<<B, 512, 0, st>>>(
                    d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_sc, d_rng,
                    d_U, nullptr, nullptr, d_outb, d_outs, d_packed_canary, nullptr, nullptr,
                    cayley_t, temperature, layers, B
                );
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 8; ++w) CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaEventRecord(ev0, st));
        for (int it = 0; it < n_launches; ++it) CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaEventRecord(ev1, st));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        CUDA_CHECK(cudaGraphExecDestroy(ge));
        CUDA_CHECK(cudaGraphDestroy(g));
        return (double(ms) * 1000.0) / double(total_iters);
    };

    double tv3_1[3], tv4_1[3], tv3_4[3], tv4_4[3];
    for (int tr = 0; tr < 3; ++tr) {
        tv3_1[tr] = bench_cfg(false, 1);
        tv4_1[tr] = bench_cfg(true, 1);
        tv3_4[tr] = bench_cfg(false, 4);
        tv4_4[tr] = bench_cfg(true, 4);
    }

    double v3_1_us = med3(tv3_1[0], tv3_1[1], tv3_1[2]);
    double v4_1_us = med3(tv4_1[0], tv4_1[1], tv4_1[2]);
    double v3_4_us = med3(tv3_4[0], tv3_4[1], tv3_4[2]);
    double v4_4_us = med3(tv4_4[0], tv4_4[1], tv4_4[2]);

    std::printf("{\n");
    std::printf("  \"device\": \"%s\",\n", prop.name);
    std::printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    std::printf("  \"clean_packed_u32\": %u,\n", clean_packed);
    std::printf("  \"clean_sys_atomics\": %u,\n", clean_sys);
    std::printf("  \"v3_1l_fp64_max_err\": %.9e,\n", err_v3_1l);
    std::printf("  \"v4_1l_fp64_max_err\": %.9e,\n", err_v4_1l);
    std::printf("  \"v4_4l_fp64_max_err\": %.9e,\n", err_v4_4l);
    std::printf("  \"v4_sec16_packed_u32\": %u,\n", sec16_packed);
    std::printf("  \"v4_sec16_sys_atomics\": %u,\n", sec16_sys);
    std::printf("  \"v4_sec16_hidden_max_abs_err\": %.9e,\n", sec16_hid_err);
    std::printf("  \"v4_sec16_logits_max_abs_err\": %.9e,\n", sec16_lg_err);
    std::printf("  \"v3_scalar_1l_us_per_tok\": %.6f,\n", v3_1_us);
    std::printf("  \"v4_monarch2_1l_us_per_tok\": %.6f,\n", v4_1_us);
    std::printf("  \"v3_scalar_4l_us_per_tok\": %.6f,\n", v3_4_us);
    std::printf("  \"v4_monarch2_4l_us_per_tok\": %.6f\n", v4_4_us);
    std::printf("}\n");

    bool ok = (clean_packed == 0u) && (clean_sys == 0u)
           && std::isfinite(err_v3_1l) && (err_v3_1l < 1e-5)
           && std::isfinite(err_v4_1l) && (err_v4_1l < 1e-5)
           && std::isfinite(err_v4_4l) && (err_v4_4l < 1e-5)
           && (sec16_packed == 16u) && (sec16_sys == 1u)
           && std::isfinite(sec16_hid_err) && (sec16_hid_err == 0.0)
           && std::isfinite(sec16_lg_err)  && (sec16_lg_err == 0.0);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaStreamDestroy(st));
    CUDA_CHECK(cudaFreeHost(h_packed_canary));
    CUDA_CHECK(cudaFreeHost(h_diag_canary));
    CUDA_CHECK(cudaFree(d_byte));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Wsc));
    CUDA_CHECK(cudaFree(d_wsu2));
    CUDA_CHECK(cudaFree(d_kvk));
    CUDA_CHECK(cudaFree(d_kvv));
    CUDA_CHECK(cudaFree(d_sc));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_hid));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_outb));
    CUDA_CHECK(cudaFree(d_outs));
    CUDA_CHECK(cudaFree(d_sys_cnt));
    return ok ? 0 : 3;
}
