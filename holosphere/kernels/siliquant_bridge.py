"""Register-resident siliquant A_16 / E_8 & group-128 PTQ quantization bridge for HoloSphere.

Provides:
  - Exact 240-root E_8 sphere lattice codebook & Block-8 spherical quantization (S108 / siliquant).
  - Exact Extended Hamming [16, 11, 4] (A_16) 5-bit parity-check syndrome verifier (matching
    Moissanite-Labs/silikernels 1-LOP3 + 5-POPC zero-spill sm_86 decoder contract).
  - Symmetric RTN group-128 3-bit/4-bit PTQ helper for S106/S107/S114 quantization-drift probes.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from holosphere.errors import require_ok
from holosphere.geometry.su2 import safe_unit_norm

# Standard Extended Hamming [16, 11, 4] parity-check bitmasks over 16-bit codewords
# (bit 0 = overall parity, bits 1..15 = standard [15, 11, 3] Hamming positions):
A16_PARITY_MASKS: tuple[int, ...] = (
    0xFFFF,  # overall even parity (all 16 bits)
    0xAAAA,  # positions with bit 0 set: 1,3,5,7,9,11,13,15
    0xCCCC,  # positions with bit 1 set: 2,3,6,7,10,11,14,15
    0xF0F0,  # positions with bit 2 set: 4,5,6,7,12,13,14,15
    0xFF00,  # positions with bit 3 set: 8..15
)


def a16_hamming_syndrome(codeword_u16: int) -> int:
    """Compute exact 5-bit Extended Hamming [16, 11, 4] syndrome (0 iff valid codeword)."""
    require_ok(type(codeword_u16) is int and 0 <= codeword_u16 <= 0xFFFF, "codeword must be u16", code="A16_U16")
    syn = 0
    for bit_idx, mask in enumerate(A16_PARITY_MASKS):
        if (codeword_u16 & mask).bit_count() & 1:
            syn |= 1 << bit_idx
    return syn


def a16_encode_data11(data11: int) -> int:
    """Encode an 11-bit integer into a valid 16-bit Extended Hamming [16, 11, 4] codeword."""
    require_ok(type(data11) is int and 0 <= data11 < 2048, "data11 must be in 0..2047", code="A16_DATA11")
    data_positions = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    cw = 0
    for i, pos in enumerate(data_positions):
        if (data11 >> i) & 1:
            cw |= 1 << pos
    for p_bit, mask in zip((1, 2, 4, 8), A16_PARITY_MASKS[1:]):
        if (cw & mask).bit_count() & 1:
            cw |= 1 << p_bit
    if cw.bit_count() & 1:
        cw |= 1
    return cw


def build_e8_root_codebook() -> torch.Tensor:
    """Construct exact 240 unit-normalized E_8 root vectors on S^7 (raw norm^2 == 2)."""
    roots: list[list[float]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    vec = [0.0] * 8
                    vec[i] = si
                    vec[j] = sj
                    roots.append(vec)
    for mask in range(256):
        if mask.bit_count() % 2 == 0:
            roots.append([-0.5 if ((mask >> bit) & 1) else 0.5 for bit in range(8)])
    require_ok(len(roots) == 240, f"expected 240 E_8 roots, built {len(roots)}", code="E8_COUNT")
    return torch.tensor(roots, dtype=torch.float32) / math.sqrt(2.0)


@torch.no_grad()
def quantize_e8_block8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Quantize vectors [..., 8K] blockwise onto unit E_8 root codebook (scale-preserving)."""
    d = x.shape[-1]
    require_ok(d > 0 and d % 8 == 0, f"quantize_e8_block8 requires D % 8 == 0, got {d}", code="E8_DIM")
    n_blocks = d // 8
    cb = build_e8_root_codebook().to(device=x.device, dtype=torch.float32)
    blocks = x.float().reshape(*x.shape[:-1], n_blocks, 8)
    scales = torch.linalg.vector_norm(blocks, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
    units = blocks / scales
    idx = torch.matmul(units, cb.T).argmax(dim=-1)
    q_blocks = cb[idx] * scales
    q_full = q_blocks.reshape_as(x).to(x.dtype)
    cos_sim = float(torch.sum(safe_unit_norm(x.float(), dim=-1) * safe_unit_norm(q_full.float(), dim=-1), dim=-1).mean().item())
    return q_full, idx, {"n_blocks": n_blocks, "mean_cosine": round(cos_sim, 6)}
