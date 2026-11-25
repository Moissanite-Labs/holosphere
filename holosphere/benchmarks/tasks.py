"""Deterministic finite-group (S_3 / A_5) non-Abelian state-tracking & bracket-depth tasks.

Used by Track S107 H7b/H7c and Track S114 H14c to test whether register-fused covariant
(Q, K, V) SU(2) parallel transport breaks above QK-only chance floors on non-Abelian composition.
Zero external data dependencies; 100% deterministic from explicit seeds.
"""

from __future__ import annotations

from itertools import permutations
import random

import torch

from holosphere.errors import require_ok

S3_ORDER = 6
A5_ORDER = 60


def _perm_compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """Composition (p o q)(i) = p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(p)))


def _is_even_perm(p: tuple[int, ...]) -> bool:
    inv = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inv += 1
    return (inv % 2) == 0


def build_s3_cayley_table() -> list[list[int]]:
    """Exact 6x6 Cayley table for the non-Abelian symmetric group S_3."""
    elems = sorted(permutations(range(3)))
    idx = {e: i for i, e in enumerate(elems)}
    return [[idx[_perm_compose(a, b)] for b in elems] for a in elems]


def build_a5_cayley_table() -> list[list[int]]:
    """Exact 60x60 Cayley table for the non-solvable alternating group A_5."""
    elems = sorted(p for p in permutations(range(5)) if _is_even_perm(p))
    require_ok(len(elems) == A5_ORDER, f"expected 60 A_5 elements, got {len(elems)}", code="A5_ORDER")
    idx = {e: i for i, e in enumerate(elems)}
    return [[idx[_perm_compose(a, b)] for b in elems] for a in elems]


def generate_group_tracking_batch(
    group: str,
    batch_size: int,
    seq_len: int,
    *,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic (inputs [B, T], prefix_products [B, T]) over S_3 or A_5.

    At every step t, target[b, t] = g_0 * g_1 * ... * g_t in the group Cayley table.
    """
    require_ok(batch_size >= 1 and seq_len >= 1, "batch_size and seq_len must be >= 1", code="TASK_SHAPE")
    if group == "S3":
        table = build_s3_cayley_table()
    elif group == "A5":
        table = build_a5_cayley_table()
    else:
        raise ValueError(f"Unknown group {group!r}; expected 'S3' or 'A5'")

    order = len(table)
    rng = random.Random(seed)
    xs = torch.empty((batch_size, seq_len), dtype=torch.long)
    ys = torch.empty((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        state = 0  # identity permutation (0, 1, ...)
        for t in range(seq_len):
            g = rng.randrange(order)
            state = table[state][g]
            xs[b, t] = g
            ys[b, t] = state
    return xs, ys


def generate_bracket_depth_batch(
    batch_size: int,
    seq_len: int,
    *,
    max_depth: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic Dyck-1 bracket strings (0='(', 1=')') and running depth targets."""
    require_ok(batch_size >= 1 and seq_len >= 2 and max_depth >= 1, "invalid bracket batch args", code="TASK_SHAPE")
    rng = random.Random(seed)
    xs = torch.empty((batch_size, seq_len), dtype=torch.long)
    ys = torch.empty((batch_size, seq_len), dtype=torch.long)
    for b in range(batch_size):
        depth = 0
        for t in range(seq_len):
            if depth == 0:
                tok = 0
            elif depth >= max_depth:
                tok = 1
            else:
                tok = 0 if rng.random() < 0.5 else 1
            depth = depth + 1 if tok == 0 else depth - 1
            xs[b, t] = tok
            ys[b, t] = depth
    return xs, ys
