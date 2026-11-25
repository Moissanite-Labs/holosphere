"""Optional Track S111 Weft-Geosphere audited live-update shell (Composed Triple Gate + Merkle rollback).

Wraps a frozen SHA-256-pinned HoloSphereTransformer Core with narrow Shell-3 adapter slots
governed by ``StaticProof AND Receipt AND Holdout``, incremental binary Merkle receipts,
and exact O(1) atomic rollback on any gate violation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import struct
from typing import Any, Sequence

from holosphere.errors import require_ok

DEFAULT_MERKLE_MAX_BYTES = 24_576


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def composed_c_row(dw: Sequence[Sequence[float]]) -> float:
    """Exact DPI row-concentration C_row(dW) = max_i ||dW[i,:]||_2^2 / ||dW||_F^2."""
    require_ok(len(dw) > 0 and len(dw[0]) > 0, "dw must be non-empty 2D matrix", code="CROW_EMPTY")
    cols = len(dw[0])
    row_sqs: list[float] = []
    for row in dw:
        require_ok(len(row) == cols, "ragged matrix", code="CROW_RAGGED")
        acc = 0.0
        for val in row:
            require_ok(math.isfinite(val), "non-finite value in weight patch", code="CROW_NONFINITE")
            acc += val * val
        row_sqs.append(acc)
    total = sum(row_sqs)
    return 0.0 if total <= 0.0 else max(row_sqs) / total


def _hash_leaf(slot_idx: int, payload: bytes) -> str:
    return _sha256_hex(f"holosphere:shell3:leaf:{slot_idx}:".encode("ascii") + payload)


def _hash_node(left_hex: str, right_hex: str) -> str:
    return _sha256_hex(f"holosphere:shell3:node:{left_hex}:{right_hex}".encode("ascii"))


class Shell3MerkleLedger:
    """Incremental binary Merkle ledger over narrow Shell-3 adapter slots."""

    def __init__(self, core_sha256: str, n_leaves: int = 16) -> None:
        require_ok(len(core_sha256) == 64, "core_sha256 must be 64-char hex", code="MERKLE_CORE_SHA")
        require_ok(n_leaves >= 2, "n_leaves must be >= 2", code="MERKLE_LEAVES")
        self.core_sha256 = core_sha256
        self.n_leaves = n_leaves
        self.padded_n = 1
        while self.padded_n < n_leaves:
            self.padded_n <<= 1
        self.leaf_hashes = [_hash_leaf(i, b"INIT_ZERO_PLATE") for i in range(self.padded_n)]
        self._tree = [""] * (2 * self.padded_n)
        for i in range(self.padded_n):
            self._tree[self.padded_n + i] = self.leaf_hashes[i]
        for idx in range(self.padded_n - 1, 0, -1):
            self._tree[idx] = _hash_node(self._tree[2 * idx], self._tree[2 * idx + 1])

    @property
    def root(self) -> str:
        return self._tree[1]

    def update_leaf(self, slot_idx: int, payload: bytes) -> str:
        require_ok(0 <= slot_idx < self.n_leaves, f"slot_idx {slot_idx} out of range", code="MERKLE_SLOT")
        lh = _hash_leaf(slot_idx, payload)
        self.leaf_hashes[slot_idx] = lh
        pos = self.padded_n + slot_idx
        self._tree[pos] = lh
        pos >>= 1
        while pos >= 1:
            self._tree[pos] = _hash_node(self._tree[2 * pos], self._tree[2 * pos + 1])
            pos >>= 1
        return self.root

    def inclusion_proof(self, slot_idx: int) -> list[dict[str, str]]:
        require_ok(0 <= slot_idx < self.n_leaves, f"slot_idx {slot_idx} out of range", code="MERKLE_SLOT")
        proof: list[dict[str, str]] = []
        pos = self.padded_n + slot_idx
        while pos > 1:
            sibling = pos ^ 1
            side = "right" if (pos % 2 == 0) else "left"
            proof.append({"side": side, "hash": self._tree[sibling]})
            pos >>= 1
        return proof

    def build_receipt(
        self,
        *,
        cycle_id: int,
        slot_idx: int,
        pre_root: str,
        evaluated_patch_sha256: str,
        committed_patch_sha256: str,
        status: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "schema": "holosphere.shell3_merkle_receipt.v1",
            "cycle_id": cycle_id,
            "core_sha256": self.core_sha256,
            "slot_idx": slot_idx,
            "n_leaves": self.n_leaves,
            "pre_root": pre_root,
            "post_root": self.root,
            "leaf_hash": self.leaf_hashes[slot_idx],
            "inclusion_proof": self.inclusion_proof(slot_idx),
            "evaluated_patch_sha256": evaluated_patch_sha256,
            "committed_patch_sha256": committed_patch_sha256,
            "status": status,
            "metrics": metrics,
        }
        receipt["receipt_sha256"] = _sha256_hex(_canonical_bytes(receipt))
        return receipt


def verify_shell3_merkle_receipt(receipt: dict[str, Any], expected_core_sha256: str) -> dict[str, Any]:
    """Fail-closed verifier for compact Shell-3 Merkle receipts."""
    raw = _canonical_bytes(receipt)
    require_ok(len(raw) <= DEFAULT_MERKLE_MAX_BYTES, "receipt exceeds byte cap", code="MERKLE_SIZE")
    require_ok(hmac.compare_digest(receipt["core_sha256"], expected_core_sha256), "Core SHA drift", code="MERKLE_CORE_DRIFT")
    if receipt["status"] == "COMMITTED":
        require_ok(
            hmac.compare_digest(receipt["evaluated_patch_sha256"], receipt["committed_patch_sha256"]),
            "H(evaluated) != H(committed)",
            code="MERKLE_BINDING",
        )
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    require_ok(hmac.compare_digest(_sha256_hex(_canonical_bytes(body)), receipt["receipt_sha256"]), "receipt tampered", code="MERKLE_DIGEST")
    cur = receipt["leaf_hash"]
    for step in receipt["inclusion_proof"]:
        if step["side"] == "right":
            cur = _hash_node(cur, step["hash"])
        elif step["side"] == "left":
            cur = _hash_node(step["hash"], cur)
        else:
            require_ok(False, f"invalid proof side {step['side']!r}", code="MERKLE_SIDE")
    require_ok(hmac.compare_digest(cur, receipt["post_root"]), "inclusion proof mismatch", code="MERKLE_ROOT")
    return {"verified": True, "receipt_bytes": len(raw), "post_root": receipt["post_root"]}


@dataclass(frozen=True)
class PlatePatchProposal:
    """Proposed Shell-3 narrow-plate weight patch awaiting Composed Triple Gate review."""

    slot_idx: int
    dw: list[list[float]]
    holdout_delta: float
    evaluated_patch_sha256: str


class WeftAuditShell:
    """Composed Triple Gate (StaticProof AND Receipt AND Holdout) with exact O(1) rollback."""

    def __init__(
        self,
        core_sha256: str,
        *,
        n_slots: int = 16,
        c_max: float = 0.60,
        f_cum: float = 4.0,
        min_holdout_delta: float = 0.0,
    ) -> None:
        self.core_sha256 = core_sha256
        self.c_max = float(c_max)
        self.f_cum = float(f_cum)
        self.min_holdout_delta = float(min_holdout_delta)
        self.ledger = Shell3MerkleLedger(core_sha256, n_leaves=n_slots)
        self.slots: dict[int, list[list[float]]] = {}
        self.cycle_id = 0

    @staticmethod
    def _pack_matrix(dw: Sequence[Sequence[float]]) -> bytes:
        buf = bytearray()
        for row in dw:
            for v in row:
                buf.extend(struct.pack(">f", float(v)))
        return bytes(buf)

    @classmethod
    def patch_sha256(cls, dw: Sequence[Sequence[float]]) -> str:
        return _sha256_hex(cls._pack_matrix(dw))

    def propose_and_commit(self, prop: PlatePatchProposal, current_core_sha256: str) -> dict[str, Any]:
        """Evaluate StaticProof AND Receipt AND Holdout; commit atomically or roll back in O(1)."""
        self.cycle_id += 1
        pre_root = self.ledger.root
        committed_sha = self.patch_sha256(prop.dw)
        c_row = composed_c_row(prop.dw)
        fro = math.sqrt(sum(v * v for row in prop.dw for v in row))

        static_ok = (c_row <= self.c_max) and (fro <= self.f_cum) and (fro > 1e-6)
        receipt_ok = hmac.compare_digest(current_core_sha256, self.core_sha256) and hmac.compare_digest(
            prop.evaluated_patch_sha256, committed_sha
        )
        holdout_ok = prop.holdout_delta >= self.min_holdout_delta

        if static_ok and receipt_ok and holdout_ok:
            self.slots[prop.slot_idx] = [list(r) for r in prop.dw]
            self.ledger.update_leaf(prop.slot_idx, self._pack_matrix(prop.dw))
            status = "COMMITTED"
        else:
            status = "ROLLED_BACK"

        return self.ledger.build_receipt(
            cycle_id=self.cycle_id,
            slot_idx=prop.slot_idx,
            pre_root=pre_root,
            evaluated_patch_sha256=prop.evaluated_patch_sha256,
            committed_patch_sha256=committed_sha,
            status=status,
            metrics={
                "c_row": round(c_row, 6),
                "fro_norm": round(fro, 6),
                "holdout_delta": round(prop.holdout_delta, 6),
                "static_ok": static_ok,
                "receipt_ok": receipt_ok,
                "holdout_ok": holdout_ok,
            },
        )
