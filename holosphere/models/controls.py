"""Matched baseline positional operators, RMSNorm, and S110 Diagonal Darcy EMA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from holosphere.errors import require_ok


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


class RMSNorm(nn.Module):
    """Exact FP32-stabilized RMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        require_ok(d_model >= 1, "d_model must be >= 1", code="RMS_DIM")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        return (xf * rms * self.weight.float()).to(x.dtype)


def apply_rope_2d(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Standard 2D Abelian S^1 RoPE over [B, H, T, D]."""
    _, _, seq_len, d = x.shape
    require_ok(d % 2 == 0, f"RoPE requires even head dim, got {d}", code="ROPE_DIM")
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half)))
    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)
    cos = torch.cos(angles).to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).to(x.dtype).unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).reshape_as(x)


class LieRE4DAbelianPositional(nn.Module):
    """LieRE 4D Abelian RoPE (N0 anchor): R(t) = exp(t * A_b) in SO(4) per 4D block."""

    def __init__(self, n_heads: int, d_head: int, seed: int = 42) -> None:
        super().__init__()
        require_ok(d_head % 4 == 0, f"LieRE 4D requires d_head % 4 == 0, got {d_head}", code="LIERE_DIM")
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_blocks = d_head // 4
        init = torch.randn((n_heads, self.n_blocks, 6), generator=_seeded_gen(seed)) * 0.05
        self.skew_params = nn.Parameter(init)

    def _build_skew(self) -> torch.Tensor:
        a = torch.zeros((self.n_heads, self.n_blocks, 4, 4), device=self.skew_params.device, dtype=torch.float32)
        p = self.skew_params.float()
        for k, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
            a[:, :, i, j] = p[:, :, k]
            a[:, :, j, i] = -p[:, :, k]
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, seq_len, d = x.shape
        skew = self._build_skew()
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        rot = torch.matrix_exp(pos.view(seq_len, 1, 1, 1, 1) * skew.unsqueeze(0)).to(x.dtype)
        xb = x.reshape(bsz, h, seq_len, self.n_blocks, 4)
        return torch.einsum("thkij,bhtkj->bhtki", rot, xb).reshape(bsz, h, seq_len, d)


class Block4RandOrthoPositional(nn.Module):
    """Adversarial null: frozen Haar-random SO(4) mixing per 4D block + standard 2D RoPE."""

    def __init__(self, n_heads: int, d_head: int, seed: int = 137) -> None:
        super().__init__()
        require_ok(d_head % 4 == 0, f"Block4-RandOrtho requires d_head % 4 == 0, got {d_head}", code="ORTHO_DIM")
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_blocks = d_head // 4
        gen = _seeded_gen(seed)
        mats: list[torch.Tensor] = []
        for _ in range(n_heads * self.n_blocks):
            q_mat, r_mat = torch.linalg.qr(torch.randn((4, 4), generator=gen))
            signs = torch.sign(torch.diag(r_mat))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            q_mat = q_mat * signs.unsqueeze(0)
            if float(torch.linalg.det(q_mat).item()) < 0.0:
                q_mat[:, 0] = -q_mat[:, 0]
            mats.append(q_mat)
        self.register_buffer("ortho", torch.stack(mats, dim=0).reshape(n_heads, self.n_blocks, 4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, seq_len, d = x.shape
        xb = x.reshape(bsz, h, seq_len, self.n_blocks, 4)
        mixed = torch.einsum("hkij,bhtkj->bhtki", self.ortho.to(x.dtype), xb).reshape(bsz, h, seq_len, d)
        return apply_rope_2d(mixed)


class DiagonalDarcyEMA(nn.Module):
    """Multi-timescale causal porous-diffusion EMA via exact zero-wraparound FFT convolution (S110)."""

    def __init__(
        self,
        d_model: int,
        tau_min: float = 2.0,
        tau_max: float = 128.0,
        alpha_eps: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        require_ok(d_model > 0, "d_model must be > 0", code="DARCY_DIM")
        require_ok(1.0 < tau_min < tau_max, "require 1.0 < tau_min < tau_max", code="DARCY_TAU")
        require_ok(0.0 < alpha_eps < 0.05, "alpha_eps must lie in (0, 0.05)", code="DARCY_EPS")
        self.d_model = d_model
        self.alpha_eps = float(alpha_eps)
        taus = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), d_model))
        init_alpha = (1.0 / taus).clamp(alpha_eps * 2.0, 1.0 - alpha_eps * 2.0)
        scaled = (init_alpha - alpha_eps) / (1.0 - 2.0 * alpha_eps)
        self.alpha_logits = nn.Parameter(torch.log(scaled / (1.0 - scaled)))
        with torch.no_grad():
            mix_init = torch.empty(d_model).uniform_(0.08, 0.16, generator=_seeded_gen(seed))
        self.mix_gate = nn.Parameter(mix_init)

    def alpha(self) -> torch.Tensor:
        return self.alpha_eps + (1.0 - 2.0 * self.alpha_eps) * torch.sigmoid(self.alpha_logits.float())

    @staticmethod
    def parallel_ema_scan(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        xf = x.float().transpose(1, 2)
        af = alpha.float()
        t_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        kernel = af.unsqueeze(-1) * torch.exp(torch.log1p(-af).unsqueeze(-1) * t_idx.unsqueeze(0))
        n_fft = 1 << max(1, (2 * seq_len - 1).bit_length())
        y = torch.fft.irfft(
            torch.fft.rfft(xf, n=n_fft, dim=-1) * torch.fft.rfft(kernel, n=n_fft, dim=-1).unsqueeze(0),
            n=n_fft,
            dim=-1,
        )[..., :seq_len]
        return y.transpose(1, 2).contiguous().to(x.dtype)

    @staticmethod
    def sequential_ema_reference(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_dim = x.shape
        xf = x.float()
        af = alpha.float().view(1, d_dim)
        lam = 1.0 - af
        s = torch.zeros((bsz, d_dim), dtype=torch.float32, device=x.device)
        outs: list[torch.Tensor] = []
        for t in range(seq_len):
            s = lam * s + af * xf[:, t, :]
            outs.append(s)
        return torch.stack(outs, dim=1).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.parallel_ema_scan(x, self.alpha())
        gate = torch.tanh(self.mix_gate.float()).view(1, 1, self.d_model)
        return (x.float() + gate * (s.float() - x.float())).to(x.dtype)
