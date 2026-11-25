#!/usr/bin/env python3
"""Kernel dispatch, custom-adjoint backward, FP64 gradcheck, & siliquant bridge tests (zero bare asserts)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from holosphere.errors import HoloSphereError
from holosphere.geometry.su2 import quat_conj, quat_mul
from holosphere.kernels import (
    SU2UnitPrefixScanFunction,
    a16_encode_data11,
    a16_hamming_syndrome,
    available_backends,
    fused_spherical_step,
    quantize_e8_block8,
    safe_hopf_clock,
    su2_holonomy_transport,
    su2_inverse_untransport,
)
from holosphere.kernels.torch_ops import torch_su2_holonomy_transport


class TestKernelsAndGradcheck(unittest.TestCase):
    def test_01_custom_adjoint_matches_autograd_and_fp64_gradcheck(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(201)
        bsz, heads, seq_len, d_head = 2, 3, 11, 12
        n_blocks = d_head // 4

        q = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)
        k = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)
        v = torch.randn((bsz, heads, seq_len, d_head), generator=gen, dtype=torch.float64)
        om0 = torch.randn((bsz, heads, seq_len, n_blocks, 3), generator=gen, dtype=torch.float64)

        om_custom = om0.clone().requires_grad_(True)
        om_auto = om0.clone().requires_grad_(True)

        qc, kc, vc, pic = torch_su2_holonomy_transport(q, k, v, om_custom, transport_v=True, eps=1e-12, use_custom_adjoint=True)
        qa, ka, va, pia = torch_su2_holonomy_transport(q, k, v, om_auto, transport_v=True, eps=1e-12, use_custom_adjoint=False)
        self.assertLess(float(torch.max(torch.abs(qc - qa)).item()), 1e-13)
        self.assertLess(float(torch.max(torch.abs(vc - va)).item()), 1e-13)

        oc = su2_inverse_untransport(vc, pic, backend="torch_cpu")
        oa = su2_inverse_untransport(va, pia, backend="torch_cpu")
        ((qc * kc).sum() + (oc * q).sum() + (pic * pic).sum()).backward()
        ((qa * ka).sum() + (oa * q).sum() + (pia * pia).sum()).backward()

        self.assertIsNotNone(om_custom.grad)
        self.assertIsNotNone(om_auto.grad)
        self.assertLess(float(torch.max(torch.abs(om_custom.grad - om_auto.grad)).item()), 1e-11)

        # Full finite-difference FP64 gradcheck on SU2UnitPrefixScanFunction
        du = torch.randn((1, 2, 6, 2, 4), generator=gen, dtype=torch.float64, requires_grad=True)
        ok = torch.autograd.gradcheck(
            lambda x: SU2UnitPrefixScanFunction.apply(x, 1e-10),
            (du,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        self.assertTrue(ok)

    def test_02_h14c_covariant_qkv_matches_explicit_interval_transport(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(202)
        seq_len, d_head = 7, 8
        q = torch.randn((1, 2, seq_len, d_head), generator=gen, dtype=torch.float64)
        k = torch.randn((1, 2, seq_len, d_head), generator=gen, dtype=torch.float64)
        v = torch.randn((1, 2, seq_len, d_head), generator=gen, dtype=torch.float64)
        om = torch.randn((1, 2, seq_len, 2, 3), generator=gen, dtype=torch.float64)

        qr, kr, vr, pi = su2_holonomy_transport(q, k, v, om, transport_v=True, eps=1e-12, backend="torch_cpu")
        out_fast = su2_inverse_untransport(
            F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=1.0 / math.sqrt(float(d_head))),
            pi,
            backend="torch_cpu",
        )

        # Explicit O(T^2) path-ordered transport sum_{s<=t} w_{ts} L(Pi_t^{-1} Pi_s) V_s
        qb = q.reshape(1, 2, seq_len, 2, 4)
        kb = k.reshape(1, 2, seq_len, 2, 4)
        vb = v.reshape(1, 2, seq_len, 2, 4)
        out_ref = torch.zeros_like(out_fast)
        for t in range(seq_len):
            logits_t = []
            vs_t = []
            for s in range(t + 1):
                rel = quat_mul(quat_conj(pi[:, :, t]), pi[:, :, s])
                k_rel = quat_mul(rel, kb[:, :, s]).reshape(1, 2, d_head)
                v_rel = quat_mul(rel, vb[:, :, s]).reshape(1, 2, d_head)
                logits_t.append((qb[:, :, t].reshape(1, 2, d_head) * k_rel).sum(dim=-1) / math.sqrt(float(d_head)))
                vs_t.append(v_rel)
            w = F.softmax(torch.stack(logits_t, dim=-1), dim=-1)
            out_ref[:, :, t] = sum(w[:, :, s : s + 1] * vs_t[s] for s in range(t + 1))

        self.assertLess(float(torch.max(torch.abs(out_fast - out_ref)).item()), 1e-12)

    def test_03_dispatch_modes_safe_hopf_and_siliquant_bridge(self) -> None:
        backends = available_backends()
        self.assertTrue(backends["torch_cpu"]["available"])
        self.assertTrue(backends["torch_cpu"]["custom_adjoint"])

        gen = torch.Generator(device="cpu").manual_seed(203)
        h = torch.randn((2, 8, 16), generator=gen)
        d = torch.randn((2, 8, 16), generator=gen)
        y_hopf = fused_spherical_step(h, d, mode="hopf_s3", backend="auto")
        y_slerp = fused_spherical_step(h, d, 0.3, mode="slerp", backend="auto")
        y_loop = fused_spherical_step(h, d, 0.3, mode="safe_hopf_loop", phi=0.5, backend="auto")
        self.assertEqual(y_hopf.shape, h.shape)
        self.assertLess(float(torch.max(torch.abs(torch.linalg.vector_norm(y_slerp, dim=-1) - 1.0)).item()), 1e-3)
        self.assertLess(float(torch.max(torch.abs(torch.linalg.vector_norm(y_loop, dim=-1) - 1.0)).item()), 1e-3)

        # Safe-Hopf cayley_acute & acute_cone norm preservation
        for law in ("acute_cone", "cayley_acute"):
            rot = safe_hopf_clock(h, pass_idx=7, law=law)
            self.assertLess(
                float(torch.max(torch.abs(torch.linalg.vector_norm(rot, dim=-1) - torch.linalg.vector_norm(h, dim=-1))).item()),
                1e-5,
            )

        # Fail-closed on unavailable CUDA/Triton backends on CPU
        with self.assertRaises(HoloSphereError):
            fused_spherical_step(h, d, mode="hopf_s3", backend="cuda_sm86")
        with self.assertRaises(HoloSphereError):
            fused_spherical_step(h, d, mode="hopf_s3", backend="triton")

        # siliquant Extended Hamming [16, 11, 4] exact syndrome (0 for all valid codewords, != 0 for 1-bit flip)
        for data11 in (0, 1, 42, 1023, 2047):
            cw = a16_encode_data11(data11)
            self.assertEqual(a16_hamming_syndrome(cw), 0)
            for bit in range(16):
                self.assertNotEqual(a16_hamming_syndrome(cw ^ (1 << bit)), 0)

        # Block-8 E_8 quantization bridge
        q_x, idx, stats = quantize_e8_block8(h)
        self.assertEqual(q_x.shape, h.shape)
        self.assertEqual(idx.shape, (2, 8, 2))
        self.assertGreater(stats["mean_cosine"], 0.8)


if __name__ == "__main__":
    unittest.main()
