#!/usr/bin/env python3
"""Triton 3.9 Fused SU(2) Holonomy-Scan, Covariant Transport, & Spherical-Norm Kernels.

Track S114 (PREREG-S114-SILISPHERE-FUSED-ENTITY-V1).
Cleanly importable on CPU-only environments (CUDA_VISIBLE_DEVICES="") without touching GPU.
All kernels mirror the exact algebra of su2_holonomy_scan_sm86.cu:
  - Right-accumulating associative unit-quaternion scan U(t) = du_0 * ... * du_t
    via tl.associative_scan (custom Hamilton combine on 4 scalar components).
  - Exact reverse-mode adjoint via 3-component su(2) additive suffix sum
    S_t = sum_{s=t}^{T-1} Im(g_s * conj(U_s)) in R^3.
  - Fused in-register spherical SLERP + non-antipodal acute-cone Safe-Hopf rotation.
  - Fused native Hopf-S^3 residual update (H6c).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _quat_combine_normalized(
        pw, px, py, pz, qw, qx, qy, qz
    ):
        """Associative Hamilton combine with unit-sphere renormalization for tl.associative_scan."""
        rw = pw * qw - px * qx - py * qy - pz * qz
        rx = pw * qx + px * qw + py * qz - pz * qy
        ry = pw * qy - px * qz + py * qw + pz * qx
        rz = pw * qz + px * qy - py * qx + pz * qw
        n2 = rw * rw + rx * rx + ry * ry + rz * rz
        inv = tl.rsqrt(tl.maximum(n2, 1e-12))
        return rw * inv, rx * inv, ry * inv, rz * inv

    @triton.jit
    def _su2_holonomy_fwd_triton_kernel(
        raw_ptr,
        q_in_ptr,
        k_in_ptr,
        v_in_ptr,
        u_out_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        seq_len,
        angle_scale,
        eps,
        USE_TANH: tl.constexpr,
        HAS_Q: tl.constexpr,
        HAS_K: tl.constexpr,
        HAS_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Single-fiber Triton associative SU(2) holonomy scan + fused (Q,K,V) transport.

        Handles arbitrary seq_len via sequential carry across BLOCK_T tiles.
        Layout: fiber-major [N_fibers, T, 3] for raw_omega; [N_fibers, T, 4] for U,Q,K,V.
        """
        fiber = tl.program_id(0)
        base_q = fiber * seq_len * 4
        base_r = fiber * seq_len * 3

        cw = tl.full((), 1.0, dtype=tl.float32)
        cx = tl.full((), 0.0, dtype=tl.float32)
        cy = tl.full((), 0.0, dtype=tl.float32)
        cz = tl.full((), 0.0, dtype=tl.float32)

        offs = tl.arange(0, BLOCK_T)
        n_tiles = tl.cdiv(seq_len, BLOCK_T)
        for tile_idx in range(0, n_tiles):
            t = tile_idx * BLOCK_T + offs
            mask = t < seq_len
            r_off = base_r + t * 3
            r1 = tl.load(raw_ptr + r_off + 0, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(raw_ptr + r_off + 1, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(raw_ptr + r_off + 2, mask=mask, other=0.0).to(tl.float32)

            if USE_TANH:
                # Triton-portable tanh(x) = (exp(2x)-1)/(exp(2x)+1) clamped for stability
                e1 = tl.exp(2.0 * tl.clamp(r1, -10.0, 10.0))
                e2 = tl.exp(2.0 * tl.clamp(r2, -10.0, 10.0))
                e3 = tl.exp(2.0 * tl.clamp(r3, -10.0, 10.0))
                w1 = angle_scale * ((e1 - 1.0) / (e1 + 1.0))
                w2 = angle_scale * ((e2 - 1.0) / (e2 + 1.0))
                w3 = angle_scale * ((e3 - 1.0) / (e3 + 1.0))
            else:
                w1, w2, w3 = r1, r2, r3

            th = tl.sqrt(w1 * w1 + w2 * w2 + w3 * w3 + eps * eps)
            s_th = tl.sin(th)
            c_th = tl.cos(th)
            sc = s_th / th
            du_w = tl.where(mask, c_th, 1.0)
            du_x = tl.where(mask, sc * w1, 0.0)
            du_y = tl.where(mask, sc * w2, 0.0)
            du_z = tl.where(mask, sc * w3, 0.0)
            dn = tl.rsqrt(tl.maximum(du_w * du_w + du_x * du_x + du_y * du_y + du_z * du_z, eps * eps))
            du_w, du_x, du_y, du_z = du_w * dn, du_x * dn, du_y * dn, du_z * dn

            # Associative Hillis-Steele prefix product inside tile
            pw, px, py, pz = tl.associative_scan(
                (du_w, du_x, du_y, du_z), axis=0, combine_fn=_quat_combine_normalized
            )
            # Multiply tile prefix on left by carry
            uw = cw * pw - cx * px - cy * py - cz * pz
            ux = cw * px + cx * pw + cy * pz - cz * py
            uy = cw * py - cx * pz + cy * pw + cz * px
            uz = cw * pz + cx * py - cy * px + cz * pw
            un = tl.rsqrt(tl.maximum(uw * uw + ux * ux + uy * uy + uz * uz, eps * eps))
            uw, ux, uy, uz = uw * un, ux * un, uy * un, uz * un

            # Extract last element of tile as carry for next tile
            last_mask = offs == (BLOCK_T - 1)
            cw = tl.sum(tl.where(last_mask, uw, 0.0), axis=0)
            cx = tl.sum(tl.where(last_mask, ux, 0.0), axis=0)
            cy = tl.sum(tl.where(last_mask, uy, 0.0), axis=0)
            cz = tl.sum(tl.where(last_mask, uz, 0.0), axis=0)

            q_off = base_q + t * 4
            tl.store(u_out_ptr + q_off + 0, uw, mask=mask)
            tl.store(u_out_ptr + q_off + 1, ux, mask=mask)
            tl.store(u_out_ptr + q_off + 2, uy, mask=mask)
            tl.store(u_out_ptr + q_off + 3, uz, mask=mask)

            if HAS_Q:
                q0 = tl.load(q_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                q1 = tl.load(q_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                q2 = tl.load(q_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                q3 = tl.load(q_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                tl.store(q_out_ptr + q_off + 0, uw * q0 - ux * q1 - uy * q2 - uz * q3, mask=mask)
                tl.store(q_out_ptr + q_off + 1, uw * q1 + ux * q0 + uy * q3 - uz * q2, mask=mask)
                tl.store(q_out_ptr + q_off + 2, uw * q2 - ux * q3 + uy * q0 + uz * q1, mask=mask)
                tl.store(q_out_ptr + q_off + 3, uw * q3 + ux * q2 - uy * q1 + uz * q0, mask=mask)

            if HAS_K:
                k0 = tl.load(k_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                k1 = tl.load(k_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                k2 = tl.load(k_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                k3 = tl.load(k_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                tl.store(k_out_ptr + q_off + 0, uw * k0 - ux * k1 - uy * k2 - uz * k3, mask=mask)
                tl.store(k_out_ptr + q_off + 1, uw * k1 + ux * k0 + uy * k3 - uz * k2, mask=mask)
                tl.store(k_out_ptr + q_off + 2, uw * k2 - ux * k3 + uy * k0 + uz * k1, mask=mask)
                tl.store(k_out_ptr + q_off + 3, uw * k3 + ux * k2 - uy * k1 + uz * k0, mask=mask)

            if HAS_V:
                v0 = tl.load(v_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                v1 = tl.load(v_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                v2 = tl.load(v_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                v3 = tl.load(v_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                tl.store(v_out_ptr + q_off + 0, uw * v0 - ux * v1 - uy * v2 - uz * v3, mask=mask)
                tl.store(v_out_ptr + q_off + 1, uw * v1 + ux * v0 + uy * v3 - uz * v2, mask=mask)
                tl.store(v_out_ptr + q_off + 2, uw * v2 - ux * v3 + uy * v0 + uz * v1, mask=mask)
                tl.store(v_out_ptr + q_off + 3, uw * v3 + ux * v2 - uy * v1 + uz * v0, mask=mask)

    @triton.jit
    def _su2_holonomy_bwd_triton_kernel(
        raw_ptr,
        u_ptr,
        q_in_ptr,
        k_in_ptr,
        v_in_ptr,
        du_ext_ptr,
        dq_out_ptr,
        dk_out_ptr,
        dv_out_ptr,
        d_raw_ptr,
        dq_in_ptr,
        dk_in_ptr,
        dv_in_ptr,
        seq_len,
        angle_scale,
        eps,
        USE_TANH: tl.constexpr,
        HAS_DU_EXT: tl.constexpr,
        HAS_Q: tl.constexpr,
        HAS_K: tl.constexpr,
        HAS_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Exact Lie-algebraic su(2) reverse additive suffix sum backward Triton kernel."""
        fiber = tl.program_id(0)
        base_q = fiber * seq_len * 4
        base_r = fiber * seq_len * 3

        sx_c = tl.full((), 0.0, dtype=tl.float32)
        sy_c = tl.full((), 0.0, dtype=tl.float32)
        sz_c = tl.full((), 0.0, dtype=tl.float32)

        offs = tl.arange(0, BLOCK_T)
        n_tiles = tl.cdiv(seq_len, BLOCK_T)
        for rev_idx in range(0, n_tiles):
            tile_idx = n_tiles - 1 - rev_idx
            t = tile_idx * BLOCK_T + offs
            mask = t < seq_len
            q_off = base_q + t * 4

            uw = tl.load(u_ptr + q_off + 0, mask=mask, other=1.0).to(tl.float32)
            ux = tl.load(u_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
            uy = tl.load(u_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
            uz = tl.load(u_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)

            if HAS_DU_EXT:
                gw = tl.load(du_ext_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                gx = tl.load(du_ext_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                gy = tl.load(du_ext_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                gz = tl.load(du_ext_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
            else:
                gw = tl.zeros([BLOCK_T], dtype=tl.float32)
                gx = tl.zeros([BLOCK_T], dtype=tl.float32)
                gy = tl.zeros([BLOCK_T], dtype=tl.float32)
                gz = tl.zeros([BLOCK_T], dtype=tl.float32)

            if HAS_Q:
                x0 = tl.load(q_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                x1 = tl.load(q_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                x2 = tl.load(q_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                x3 = tl.load(q_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                d0 = tl.load(dq_out_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                d1 = tl.load(dq_out_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                d2 = tl.load(dq_out_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                d3 = tl.load(dq_out_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                # dq_in = conj(u) * d
                tl.store(dq_in_ptr + q_off + 0, uw * d0 + ux * d1 + uy * d2 + uz * d3, mask=mask)
                tl.store(dq_in_ptr + q_off + 1, uw * d1 - ux * d0 - uy * d3 + uz * d2, mask=mask)
                tl.store(dq_in_ptr + q_off + 2, uw * d2 + ux * d3 - uy * d0 - uz * d1, mask=mask)
                tl.store(dq_in_ptr + q_off + 3, uw * d3 - ux * d2 + uy * d1 - uz * d0, mask=mask)
                # g_u += d * conj(x)
                gw += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3
                gx += -d0 * x1 + d1 * x0 - d2 * x3 + d3 * x2
                gy += -d0 * x2 + d1 * x3 + d2 * x0 - d3 * x1
                gz += -d0 * x3 - d1 * x2 + d2 * x1 + d3 * x0

            if HAS_K:
                x0 = tl.load(k_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                x1 = tl.load(k_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                x2 = tl.load(k_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                x3 = tl.load(k_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                d0 = tl.load(dk_out_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                d1 = tl.load(dk_out_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                d2 = tl.load(dk_out_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                d3 = tl.load(dk_out_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                tl.store(dk_in_ptr + q_off + 0, uw * d0 + ux * d1 + uy * d2 + uz * d3, mask=mask)
                tl.store(dk_in_ptr + q_off + 1, uw * d1 - ux * d0 - uy * d3 + uz * d2, mask=mask)
                tl.store(dk_in_ptr + q_off + 2, uw * d2 + ux * d3 - uy * d0 - uz * d1, mask=mask)
                tl.store(dk_in_ptr + q_off + 3, uw * d3 - ux * d2 + uy * d1 - uz * d0, mask=mask)
                gw += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3
                gx += -d0 * x1 + d1 * x0 - d2 * x3 + d3 * x2
                gy += -d0 * x2 + d1 * x3 + d2 * x0 - d3 * x1
                gz += -d0 * x3 - d1 * x2 + d2 * x1 + d3 * x0

            if HAS_V:
                x0 = tl.load(v_in_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                x1 = tl.load(v_in_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                x2 = tl.load(v_in_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                x3 = tl.load(v_in_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                d0 = tl.load(dv_out_ptr + q_off + 0, mask=mask, other=0.0).to(tl.float32)
                d1 = tl.load(dv_out_ptr + q_off + 1, mask=mask, other=0.0).to(tl.float32)
                d2 = tl.load(dv_out_ptr + q_off + 2, mask=mask, other=0.0).to(tl.float32)
                d3 = tl.load(dv_out_ptr + q_off + 3, mask=mask, other=0.0).to(tl.float32)
                tl.store(dv_in_ptr + q_off + 0, uw * d0 + ux * d1 + uy * d2 + uz * d3, mask=mask)
                tl.store(dv_in_ptr + q_off + 1, uw * d1 - ux * d0 - uy * d3 + uz * d2, mask=mask)
                tl.store(dv_in_ptr + q_off + 2, uw * d2 + ux * d3 - uy * d0 - uz * d1, mask=mask)
                tl.store(dv_in_ptr + q_off + 3, uw * d3 - ux * d2 + uy * d1 - uz * d0, mask=mask)
                gw += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3
                gx += -d0 * x1 + d1 * x0 - d2 * x3 + d3 * x2
                gy += -d0 * x2 + d1 * x3 + d2 * x0 - d3 * x1
                gz += -d0 * x3 - d1 * x2 + d2 * x1 + d3 * x0

            # m = Im(g_u * conj(u)) in su(2)
            mx = tl.where(mask, -gw * ux + gx * uw - gy * uz + gz * uy, 0.0)
            my = tl.where(mask, -gw * uy + gx * uz + gy * uw - gz * ux, 0.0)
            mz = tl.where(mask, -gw * uz - gx * uy + gy * ux + gz * uw, 0.0)

            # Suffix sum inside tile = total - inclusive_prefix + self
            tot_x = tl.sum(mx, axis=0)
            tot_y = tl.sum(my, axis=0)
            tot_z = tl.sum(mz, axis=0)
            px_s = tl.cumsum(mx, axis=0)
            py_s = tl.cumsum(my, axis=0)
            pz_s = tl.cumsum(mz, axis=0)
            sv_x = sx_c + (tot_x - px_s + mx)
            sv_y = sy_c + (tot_y - py_s + my)
            sv_z = sz_c + (tot_z - pz_s + mz)
            sx_c = sx_c + tot_x
            sy_c = sy_c + tot_y
            sz_c = sz_c + tot_z

            # u_bar = (0, sv) * u
            ub_w = -sv_x * ux - sv_y * uy - sv_z * uz
            ub_x =  sv_x * uw + sv_y * uz - sv_z * uy
            ub_y = -sv_x * uz + sv_y * uw + sv_z * ux
            ub_z =  sv_x * uy - sv_y * ux + sv_z * uw

            has_prev = mask & (t > 0)
            prev_off = base_q + tl.maximum(t - 1, 0) * 4
            pw = tl.where(has_prev, tl.load(u_ptr + prev_off + 0, mask=has_prev, other=1.0).to(tl.float32), 1.0)
            px_u = tl.where(has_prev, tl.load(u_ptr + prev_off + 1, mask=has_prev, other=0.0).to(tl.float32), 0.0)
            py_u = tl.where(has_prev, tl.load(u_ptr + prev_off + 2, mask=has_prev, other=0.0).to(tl.float32), 0.0)
            pz_u = tl.where(has_prev, tl.load(u_ptr + prev_off + 3, mask=has_prev, other=0.0).to(tl.float32), 0.0)
            # g_du = conj(u_prev) * u_bar
            gdu_w = pw * ub_w + px_u * ub_x + py_u * ub_y + pz_u * ub_z
            gdu_x = pw * ub_x - px_u * ub_w - py_u * ub_z + pz_u * ub_y
            gdu_y = pw * ub_y + px_u * ub_z - py_u * ub_w - pz_u * ub_x
            gdu_z = pw * ub_z - px_u * ub_y + py_u * ub_x - pz_u * ub_w

            r_off = base_r + t * 3
            r1 = tl.load(raw_ptr + r_off + 0, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(raw_ptr + r_off + 1, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(raw_ptr + r_off + 2, mask=mask, other=0.0).to(tl.float32)
            if USE_TANH:
                e1 = tl.exp(2.0 * tl.clamp(r1, -10.0, 10.0))
                e2 = tl.exp(2.0 * tl.clamp(r2, -10.0, 10.0))
                e3 = tl.exp(2.0 * tl.clamp(r3, -10.0, 10.0))
                th1 = (e1 - 1.0) / (e1 + 1.0)
                th2 = (e2 - 1.0) / (e2 + 1.0)
                th3 = (e3 - 1.0) / (e3 + 1.0)
                w1, w2, w3 = angle_scale * th1, angle_scale * th2, angle_scale * th3
            else:
                th1 = tl.zeros([BLOCK_T], dtype=tl.float32)
                th2 = tl.zeros([BLOCK_T], dtype=tl.float32)
                th3 = tl.zeros([BLOCK_T], dtype=tl.float32)
                w1, w2, w3 = r1, r2, r3

            theta = tl.sqrt(w1 * w1 + w2 * w2 + w3 * w3 + eps * eps)
            s_th = tl.sin(theta)
            c_th = tl.cos(theta)
            sinc = s_th / theta
            q0_w, q0_x, q0_y, q0_z = c_th, sinc * w1, sinc * w2, sinc * w3
            inv_q0 = tl.rsqrt(tl.maximum(q0_w * q0_w + q0_x * q0_x + q0_y * q0_y + q0_z * q0_z, eps * eps))
            du_w, du_x, du_y, du_z = q0_w * inv_q0, q0_x * inv_q0, q0_y * inv_q0, q0_z * inv_q0
            dot_g = gdu_w * du_w + gdu_x * du_x + gdu_y * du_y + gdu_z * du_z
            gq_w = (gdu_w - dot_g * du_w) * inv_q0
            gq_x = (gdu_x - dot_g * du_x) * inv_q0
            gq_y = (gdu_y - dot_g * du_y) * inv_q0
            gq_z = (gdu_z - dot_g * du_z) * inv_q0

            t2 = theta * theta
            taylor = -1.0 / 3.0 + t2 * (1.0 / 30.0 - t2 / 840.0)
            exact = (theta * c_th - s_th) / (theta * t2)
            dsinc_div_th = tl.where(theta < 1e-2, taylor, exact)
            rc = -sinc * gq_w + dsinc_div_th * (gq_x * w1 + gq_y * w2 + gq_z * w3)
            dw1 = sinc * gq_x + rc * w1
            dw2 = sinc * gq_y + rc * w2
            dw3 = sinc * gq_z + rc * w3
            if USE_TANH:
                dw1 = dw1 * angle_scale * (1.0 - th1 * th1)
                dw2 = dw2 * angle_scale * (1.0 - th2 * th2)
                dw3 = dw3 * angle_scale * (1.0 - th3 * th3)
            tl.store(d_raw_ptr + r_off + 0, dw1, mask=mask)
            tl.store(d_raw_ptr + r_off + 1, dw2, mask=mask)
            tl.store(d_raw_ptr + r_off + 2, dw3, mask=mask)


def triton_su2_holonomy_fwd(
    raw_omega: torch.Tensor,
    q_in: Optional[torch.Tensor],
    k_in: Optional[torch.Tensor],
    v_in: Optional[torch.Tensor],
    *,
    angle_scale: float = 0.75,
    use_tanh: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Launch Triton forward SU(2) holonomy scan on CUDA fiber-major tensors [F, T, ...]."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in this environment")
    if not raw_omega.is_cuda:
        raise RuntimeError("triton_su2_holonomy_fwd requires CUDA tensors")
    n_fibers, seq_len, _ = raw_omega.shape
    raw_c = raw_omega.contiguous()
    u_out = torch.empty((n_fibers, seq_len, 4), device=raw_c.device, dtype=torch.float32)
    q_out = torch.empty_like(u_out) if q_in is not None else None
    k_out = torch.empty_like(u_out) if k_in is not None else None
    v_out = torch.empty_like(u_out) if v_in is not None else None
    dummy = u_out
    block_t = 256 if seq_len >= 256 else max(32, 1 << (seq_len - 1).bit_length())
    _su2_holonomy_fwd_triton_kernel[(n_fibers,)](
        raw_c,
        q_in.contiguous() if q_in is not None else dummy,
        k_in.contiguous() if k_in is not None else dummy,
        v_in.contiguous() if v_in is not None else dummy,
        u_out,
        q_out if q_out is not None else dummy,
        k_out if k_out is not None else dummy,
        v_out if v_out is not None else dummy,
        seq_len,
        float(angle_scale),
        float(eps),
        USE_TANH=use_tanh,
        HAS_Q=q_in is not None,
        HAS_K=k_in is not None,
        HAS_V=v_in is not None,
        BLOCK_T=block_t,
    )
    return u_out, q_out, k_out, v_out


def triton_su2_holonomy_bwd(
    raw_omega: torch.Tensor,
    u_path: torch.Tensor,
    q_in: Optional[torch.Tensor],
    k_in: Optional[torch.Tensor],
    v_in: Optional[torch.Tensor],
    du_ext: Optional[torch.Tensor],
    dq_out: Optional[torch.Tensor],
    dk_out: Optional[torch.Tensor],
    dv_out: Optional[torch.Tensor],
    *,
    angle_scale: float = 0.75,
    use_tanh: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Launch Triton reverse-mode su(2) additive-suffix-sum backward kernel on CUDA."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in this environment")
    if not raw_omega.is_cuda:
        raise RuntimeError("triton_su2_holonomy_bwd requires CUDA tensors")
    n_fibers, seq_len, _ = raw_omega.shape
    raw_c = raw_omega.contiguous()
    u_c = u_path.contiguous()
    d_raw = torch.empty((n_fibers, seq_len, 3), device=raw_c.device, dtype=torch.float32)
    dq_in = torch.empty_like(u_c) if (q_in is not None and dq_out is not None) else None
    dk_in = torch.empty_like(u_c) if (k_in is not None and dk_out is not None) else None
    dv_in = torch.empty_like(u_c) if (v_in is not None and dv_out is not None) else None
    dummy = u_c
    block_t = 256 if seq_len >= 256 else max(32, 1 << (seq_len - 1).bit_length())
    _su2_holonomy_bwd_triton_kernel[(n_fibers,)](
        raw_c,
        u_c,
        q_in.contiguous() if q_in is not None else dummy,
        k_in.contiguous() if k_in is not None else dummy,
        v_in.contiguous() if v_in is not None else dummy,
        du_ext.contiguous() if du_ext is not None else dummy,
        dq_out.contiguous() if dq_out is not None else dummy,
        dk_out.contiguous() if dk_out is not None else dummy,
        dv_out.contiguous() if dv_out is not None else dummy,
        d_raw,
        dq_in if dq_in is not None else dummy,
        dk_in if dk_in is not None else dummy,
        dv_in if dv_in is not None else dummy,
        seq_len,
        float(angle_scale),
        float(eps),
        USE_TANH=use_tanh,
        HAS_DU_EXT=du_ext is not None,
        HAS_Q=dq_in is not None,
        HAS_K=dk_in is not None,
        HAS_V=dv_in is not None,
        BLOCK_T=block_t,
    )
    return d_raw, dq_in, dk_in, dv_in


__all__ = [
    "TRITON_AVAILABLE",
    "triton_su2_holonomy_fwd",
    "triton_su2_holonomy_bwd",
]
