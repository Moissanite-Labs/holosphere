"""Batched nullmodem Block-8 E_8 RAM == WIRE == HASH sphere-lattice cell bus (Track S108 H8c3).

Extends Track S108's verified single-state 240-root E_8 sphere cell (99.74% field accuracy,
98.96% exact 4-field accuracy in 32 payload bits, 100% SHA-256 verified) with batched
multi-state framing (M >= 1 states per authenticated envelope) so one 320-bit header+SHA-256
wrapper amortizes below unauthenticated ABIOS/3 (144 bits/state) for M >= 4 states:
  total_bits_per_state(M, n_blocks=4) = 32 + 320 / M  (112 bits/state at M=4, 72 at M=8!).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import struct
from typing import Sequence

import torch
import torch.nn as nn

from holosphere.errors import require_ok
from holosphere.geometry.su2 import safe_unit_norm
from holosphere.kernels.siliquant_bridge import build_e8_root_codebook


@dataclass(frozen=True)
class NullModemCell:
    """Single-state self-verifying RAM == WIRE == HASH binary cell for Block-8 E_8 states."""

    magic: bytes
    version: int
    n_blocks: int
    indices: bytes
    sha256_digest: bytes

    HEADER_FMT = "!4sBBH"  # 8 bytes header + 32 bytes SHA-256 = 40 bytes (320 bits)
    HASH_BYTES = 32

    @classmethod
    def pack(cls, indices: Sequence[int]) -> bytes:
        n_blocks = len(indices)
        require_ok(1 <= n_blocks <= 255, f"n_blocks must be in 1..255, got {n_blocks}", code="NM_BLOCKS")
        for idx in indices:
            require_ok(0 <= int(idx) < 240, f"E_8 root index out of range 0..239: {idx}", code="NM_INDEX")
        header = struct.pack(cls.HEADER_FMT, b"NM8S", 1, n_blocks, 0)
        payload = bytes(int(i) for i in indices)
        digest = hashlib.sha256(header + payload).digest()
        return header + payload + digest

    @classmethod
    def verify_and_unpack(cls, wire: bytes) -> "NullModemCell":
        hdr_len = struct.calcsize(cls.HEADER_FMT)
        min_len = hdr_len + 1 + cls.HASH_BYTES
        require_ok(len(wire) >= min_len, f"cell too short: {len(wire)} < {min_len}", code="NM_SHORT")
        magic, version, n_blocks, reserved = struct.unpack(cls.HEADER_FMT, wire[:hdr_len])
        require_ok(magic == b"NM8S" and version == 1 and reserved == 0, "invalid single cell header", code="NM_HDR")
        expected_len = hdr_len + n_blocks + cls.HASH_BYTES
        require_ok(len(wire) == expected_len, f"wire length {len(wire)} != {expected_len}", code="NM_LEN")
        payload = wire[hdr_len : hdr_len + n_blocks]
        digest = wire[hdr_len + n_blocks :]
        recomputed = hashlib.sha256(wire[: hdr_len + n_blocks]).digest()
        require_ok(hmac.compare_digest(digest, recomputed), "SHA-256 verification failed", code="NM_HASH")
        for b in payload:
            require_ok(b < 240, f"invalid E_8 index byte {b} >= 240", code="NM_INDEX")
        return cls(magic=magic, version=version, n_blocks=n_blocks, indices=payload, sha256_digest=digest)


@dataclass(frozen=True)
class BatchedNullModemCell:
    """Amortized multi-state RAM == WIRE == HASH binary cell (M states sharing one SHA-256).

    Header layout ``!4sBBH`` (8 bytes): magic ``NM8B``, version ``1``, ``n_blocks`` per state,
    ``batch_states`` (M in 1..65535), followed by ``M * n_blocks`` E_8 root bytes and 32-byte SHA-256.
    """

    magic: bytes
    version: int
    n_blocks: int
    batch_states: int
    states: tuple[bytes, ...]
    sha256_digest: bytes

    HEADER_FMT = "!4sBBH"
    HASH_BYTES = 32

    @classmethod
    def pack_batch(cls, batch_indices: Sequence[Sequence[int]]) -> bytes:
        m_states = len(batch_indices)
        require_ok(1 <= m_states <= 65535, f"batch_states must be in 1..65535, got {m_states}", code="NMB_COUNT")
        n_blocks = len(batch_indices[0])
        require_ok(1 <= n_blocks <= 255, f"n_blocks must be in 1..255, got {n_blocks}", code="NMB_BLOCKS")
        payload = bytearray()
        for row in batch_indices:
            require_ok(len(row) == n_blocks, "all states in batch must share identical n_blocks", code="NMB_RAGGED")
            for idx in row:
                require_ok(0 <= int(idx) < 240, f"E_8 root index out of range 0..239: {idx}", code="NMB_INDEX")
                payload.append(int(idx))
        header = struct.pack(cls.HEADER_FMT, b"NM8B", 1, n_blocks, m_states)
        body = header + bytes(payload)
        return body + hashlib.sha256(body).digest()

    @classmethod
    def verify_and_unpack(cls, wire: bytes) -> "BatchedNullModemCell":
        hdr_len = struct.calcsize(cls.HEADER_FMT)
        require_ok(len(wire) >= hdr_len + 1 + cls.HASH_BYTES, "batched cell too short", code="NMB_SHORT")
        magic, version, n_blocks, m_states = struct.unpack(cls.HEADER_FMT, wire[:hdr_len])
        require_ok(magic == b"NM8B" and version == 1 and n_blocks >= 1 and m_states >= 1, "invalid batched header", code="NMB_HDR")
        expected_len = hdr_len + n_blocks * m_states + cls.HASH_BYTES
        require_ok(len(wire) == expected_len, f"wire length {len(wire)} != {expected_len}", code="NMB_LEN")
        payload = wire[hdr_len : hdr_len + n_blocks * m_states]
        digest = wire[hdr_len + n_blocks * m_states :]
        recomputed = hashlib.sha256(wire[: hdr_len + n_blocks * m_states]).digest()
        require_ok(hmac.compare_digest(digest, recomputed), "batched SHA-256 verification failed", code="NMB_HASH")
        for b in payload:
            require_ok(b < 240, f"invalid E_8 index byte {b} >= 240", code="NMB_INDEX")
        states = tuple(payload[i * n_blocks : (i + 1) * n_blocks] for i in range(m_states))
        return cls(
            magic=magic,
            version=version,
            n_blocks=n_blocks,
            batch_states=m_states,
            states=states,
            sha256_digest=digest,
        )

    @staticmethod
    def wire_bits_per_state(n_blocks: int, batch_states: int) -> float:
        """Exact amortized total wire bits per state including 320-bit header + SHA-256."""
        require_ok(n_blocks >= 1 and batch_states >= 1, "n_blocks and batch_states must be >= 1", code="NMB_BITS")
        return 8.0 * float(n_blocks) + 320.0 / float(batch_states)


class BlockE8SphereCodec(nn.Module):
    """Block-8 E_8 sphere-lattice encoder/decoder with single and amortized batched wire packing."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        require_ok(d_latent > 0 and d_latent % 8 == 0, f"d_latent must be positive multiple of 8, got {d_latent}", code="E8_DIM")
        self.d_latent = d_latent
        self.n_blocks = d_latent // 8
        self.register_buffer("codebook", build_e8_root_codebook())

    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        blocks = safe_unit_norm(z.float().reshape(*z.shape[:-1], self.n_blocks, 8), dim=-1, eps=1e-6)
        return torch.matmul(blocks, self.codebook.T).argmax(dim=-1)

    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        vecs = self.codebook[idx].reshape(*idx.shape[:-1], self.d_latent)
        return safe_unit_norm(vecs, dim=-1, eps=1e-6)

    def pack_cell(self, z_single: torch.Tensor) -> bytes:
        require_ok(z_single.ndim == 1 and z_single.shape[0] == self.d_latent, "expected 1D [d_latent]", code="E8_SHAPE")
        return NullModemCell.pack(self.encode_indices(z_single).tolist())

    def unpack_cell(self, wire: bytes) -> torch.Tensor:
        cell = NullModemCell.verify_and_unpack(wire)
        require_ok(cell.n_blocks == self.n_blocks, "cell n_blocks mismatch", code="E8_BLOCKS")
        return self.decode_indices(torch.tensor(list(cell.indices), dtype=torch.long, device=self.codebook.device))

    def pack_batch_cells(self, z_batch: torch.Tensor) -> bytes:
        require_ok(z_batch.ndim == 2 and z_batch.shape[1] == self.d_latent, "expected 2D [M, d_latent]", code="E8_SHAPE")
        return BatchedNullModemCell.pack_batch(self.encode_indices(z_batch).tolist())

    def unpack_batch_cells(self, wire: bytes) -> torch.Tensor:
        cell = BatchedNullModemCell.verify_and_unpack(wire)
        require_ok(cell.n_blocks == self.n_blocks, "batched cell n_blocks mismatch", code="E8_BLOCKS")
        idx = torch.tensor([list(s) for s in cell.states], dtype=torch.long, device=self.codebook.device)
        return self.decode_indices(idx)
