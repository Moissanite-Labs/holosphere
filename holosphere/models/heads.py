"""Multi-token prediction heads: Markov-MTP8 (S108 H8c2 confirmed -19.34%), Parallel-MTP8, LocalAR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from holosphere.errors import require_ok
from holosphere.models.controls import RMSNorm


class MarkovMTP8Head(nn.Module):
    """Low-rank Markov-coupled multi-token prediction head (Track S108 H8c2 CONFIRMED).

    Cut teacher-forced MTP-8 BPB by -19.34% (z=-63.31) vs independent Parallel-MTP8
    and resisted unrolled exposure-bias drift by -12.37% vs LocalAR.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 260,
        block_size: int = 8,
        markov_rank: int = 64,
    ) -> None:
        super().__init__()
        require_ok(block_size >= 1 and markov_rank >= 4, "invalid MarkovMTP8 dims", code="MTP_DIM")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.markov_rank = markov_rank
        self.init_proj = nn.Linear(d_model, markov_rank)
        self.step_cond = nn.ModuleList([nn.Linear(d_model, markov_rank, bias=False) for _ in range(block_size)])
        self.tok_emb = nn.Embedding(vocab_size, markov_rank)
        self.state_step = nn.Linear(markov_rank, markov_rank)
        self.out_head = nn.Linear(markov_rank, vocab_size)

    def forward(
        self,
        h_exit: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        unroll: bool = False,
    ) -> torch.Tensor:
        """h_exit: [..., d_model]; returns logits [..., block_size, vocab_size]."""
        lead_shape = h_exit.shape[:-1]
        flat_h = h_exit.reshape(-1, self.d_model)
        flat_t = target_tokens.reshape(-1, self.block_size) if target_tokens is not None else None
        state = torch.tanh(self.init_proj(flat_h))
        logits_steps: list[torch.Tensor] = []
        prev_tok: torch.Tensor | None = None
        for j in range(self.block_size):
            if j == 0:
                s_in = state + self.step_cond[j](flat_h)
            else:
                if unroll or flat_t is None:
                    b_in = prev_tok if prev_tok is not None else torch.zeros(flat_h.shape[0], dtype=torch.long, device=flat_h.device)
                else:
                    b_in = flat_t[:, j - 1]
                s_in = self.state_step(state) + self.tok_emb(b_in) + self.step_cond[j](flat_h)
            state = torch.tanh(s_in)
            step_logits = self.out_head(state)
            logits_steps.append(step_logits)
            prev_tok = step_logits.argmax(dim=-1)
        return torch.stack(logits_steps, dim=1).reshape(*lead_shape, self.block_size, self.vocab_size)


class ParallelMTP8Head(nn.Module):
    """Independent parallel K-token head (intra-patch coupling obstruction control)."""

    def __init__(self, d_model: int, vocab_size: int = 260, block_size: int = 8) -> None:
        super().__init__()
        self.block_size = block_size
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(block_size)])

    def forward(
        self,
        h_exit: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        unroll: bool = False,
    ) -> torch.Tensor:
        _ = (target_tokens, unroll)
        return torch.stack([head(h_exit) for head in self.heads], dim=-2)


class LocalARByteDecoder(nn.Module):
    """Causal 1-layer local autoregressive decoder control conditioned on exit hidden state."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 260,
        d_dec: int = 64,
        n_heads: int = 4,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        require_ok(d_dec % n_heads == 0, "d_dec must be divisible by n_heads", code="LOCALAR_DIM")
        self.d_dec = d_dec
        self.n_heads = n_heads
        self.block_size = block_size
        self.cond_proj = nn.Linear(d_model, d_dec)
        self.tok_emb = nn.Embedding(vocab_size, d_dec)
        self.pos_emb = nn.Embedding(block_size, d_dec)
        self.norm1 = RMSNorm(d_dec)
        self.qkv = nn.Linear(d_dec, 3 * d_dec, bias=False)
        self.o_proj = nn.Linear(d_dec, d_dec, bias=False)
        self.norm2 = RMSNorm(d_dec)
        self.fc1 = nn.Linear(d_dec, 2 * d_dec, bias=False)
        self.fc2 = nn.Linear(2 * d_dec, d_dec, bias=False)
        self.out_head = nn.Linear(d_dec, vocab_size)

    def _run_block(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        dh = d // self.n_heads
        q = q.view(bsz, t, self.n_heads, dh).transpose(1, 2)
        k = k.view(bsz, t, self.n_heads, dh).transpose(1, 2)
        v = v.view(bsz, t, self.n_heads, dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(bsz, t, d)
        x = x + self.o_proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))

    def forward(
        self,
        h_exit: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        unroll: bool = False,
    ) -> torch.Tensor:
        lead_shape = h_exit.shape[:-1]
        flat_h = h_exit.reshape(-1, h_exit.shape[-1])
        n = flat_h.shape[0]
        cond = self.cond_proj(flat_h).unsqueeze(1)
        pos = torch.arange(self.block_size, device=flat_h.device)
        if not unroll and target_tokens is not None:
            flat_t = target_tokens.reshape(-1, self.block_size)
            shifted = torch.zeros_like(flat_t)
            if self.block_size > 1:
                shifted[:, 1:] = flat_t[:, :-1]
            x = self.tok_emb(shifted) + self.pos_emb(pos).unsqueeze(0) + cond
            return self.out_head(self._run_block(x)).reshape(*lead_shape, self.block_size, -1)
        gen = torch.zeros((n, self.block_size), dtype=torch.long, device=flat_h.device)
        steps: list[torch.Tensor] = []
        for j in range(self.block_size):
            x = self.tok_emb(gen[:, : j + 1]) + self.pos_emb(pos[: j + 1]).unsqueeze(0) + cond
            logits = self.out_head(self._run_block(x))[:, -1, :]
            steps.append(logits)
            if j + 1 < self.block_size:
                gen[:, j + 1] = logits.argmax(dim=-1)
        return torch.stack(steps, dim=1).reshape(*lead_shape, self.block_size, -1)
