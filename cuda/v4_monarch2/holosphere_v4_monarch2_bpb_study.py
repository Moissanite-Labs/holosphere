#!/usr/bin/env python3
"""Standalone 4-Arm x 3-Seed Multi-Context UTF-8 Byte-LM Training & FP16 Head Evaluation Script.

Reproduces the controlled iso-budget 4-layer d_model=512 UTF-8 bits-per-byte (BPB) study across
held-out context lengths T in {512, 2048, 8192} (65,536 held-out bytes per context) comparing:
  1. V3_ScalarFWHT       : 2 B FP16 logit scalar + FWHT_256 x bent4 (0 B HBM input/output vocab)
  2. V4_Monarch2BiasFWHT : 16,896 B 2-stage Monarch-16x16 + 512 B FP16 byte bias + FWHT_256 x bent4
                           (CTA-SMEM-resident, 0 B HBM input/output vocab)
  3. V3_LowRank16Bias    : 16,896 B matched-byte rank-16 low-rank + 512 B FP16 byte bias (16,896 B HBM)
  4. V3_DenseHead        : 262,658 B dense 256x512 linear + scalar + 512 B FP16 byte bias (262,658 B HBM)

Mathematical Scale Alignment (Paired with CUDA Sources):
  - Plain 256-point Hadamard (`h256_mat`) uses orthonormal scale `1/16 = 0.0625` (`||H u||_2^2 = ||u||_2^2`),
    matching `fwht256_plain_8warp_raw` (`0.0625f`) in `holosphere_v4_monarch2_e2e_sm86.cu` and
    `fwht256_plain_fp64` (`0.0625`) in `holosphere_v4_monarch2_live_bench.cu`.
  - Bent-4 256-point signed-Hadamard transform (`fwht_mat` and `inj_table`) uses scale
    `1/sqrt(32) = 0.17677669529663688` (`sqrt(8) * orthonormal`, so `||y||_2^2 = 8 * ||u||_2^2`),
    matching `fwht256_bent4_8warp_raw` (`0.17677669529663688f`) in `holosphere_v4_monarch2_e2e_sm86.cu`
    and `fwht256_bent4_fp64` (`1.0 / std::sqrt(32.0)`) in `holosphere_v4_monarch2_live_bench.cu`.
  - Input Alphabet & BOS Aliasing Disclosure: `inj_table` indexes `257` rows (`0..255` UTF-8 bytes
    plus `BOS = 256` with `sp_tag = 1`, `sp_flip = (d256 >> 5) & 1`). Algebraically, `H_256 x bent4`
    defines `256` distinct byte-alphabet vectors and `BOS = 256` matches UTF-8 byte `33` (`0x21`, `'!'`)
    bit-for-bit (`popcount(33 & db) & 1 == (d256 >> 5) & 1` for all `d256 in 0..255`). The fused
    single-CTA CUDA megakernel (`holosphere_v4_monarch2_e2e_sm86.cu`) takes `uint8_t` (`0..255`) bytes
    directly without a separate `sp_flip` bit; a host runtime starting from `BOS = 256` passes `0x21`.

Telemetry Policy (`UNCONTROLLED_LAPTOP_NOT_CLAIMED`):
  - Zero device power or energy telemetry is read, logged, or emitted by this script.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
CORPUS_ID = "holosphere-v3-flagship-corpus"
EXPECTED_CORPUS_MANIFEST_SHA256 = "a05e415f7d33ef32376ff3d9da1eede0e3f561b64094b5b91147b2696a99e9d4"
EXPECTED_TRAIN_BIN_SHA256 = "528c6c792d2a563d82d21ce213b26209441b39fa5e76b8595607bab2ca853b7b"
EXPECTED_EVAL_BIN_SHA256 = "31fd132d888a21ecebe9e907b6c4b82e3e68e42b5a29e2839584447dcca631a0"
EVAL_CONTEXTS: Tuple[Tuple[int, int], ...] = ((512, 128), (2048, 32), (8192, 8))


class V4Monarch2BpbStudyError(RuntimeError):
    """Fail-closed typed error for the HoloSphere-V4 Monarch-2 UTF-8 BPB study runner."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise V4Monarch2BpbStudyError(code, message)


def sha256_file(path: Path) -> str:
    require_ok(not path.is_symlink(), "SYMLINK_FORBIDDEN", path.name)
    require_ok(path.is_file(), "MISSING_FILE", path.name)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(1 << 20)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def to_q_str(val: float, max_den: int = 1_000_000) -> str:
    require_ok(math.isfinite(val), "NON_FINITE", f"val={val!r}")
    f = Fraction(val).limit_denominator(max_den)
    return f"{f.numerator}/{f.denominator}"


def parse_q(raw: str) -> Fraction:
    num_s, den_s = raw.split("/", 1)
    return Fraction(int(num_s), int(den_s))


def format_q(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def run_e2e_bpb_study(
    corpus_dir: Path,
    *,
    steps: int = 300,
    batch_size: int = 2,
    train_seq_len: int = 2048,
    seeds: Optional[List[int]] = None,
    verify_pinned_corpus: bool = True,
) -> Dict[str, Any]:
    """Run the 4-arm x 3-seed multi-context UTF-8 byte-LM study with FP16-rounded head eval."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_list = [42, 137, 2026] if seeds is None else list(seeds)
    require_ok(torch.cuda.is_available(), "NO_CUDA", "CUDA device required for GPU training")
    device = torch.device("cuda:0")
    d_model, n_heads, d_ff, n_layers = 512, 16, 800, 4
    rope_base = 500_000.0
    max_ctx = max(t for t, _ in EVAL_CONTEXTS)

    require_ok(not corpus_dir.is_symlink(), "SYMLINK_FORBIDDEN", corpus_dir.name)
    man_path = corpus_dir.joinpath("flagship_corpus_manifest.json")
    tr_path = corpus_dir.joinpath("train.bin")
    ev_path = corpus_dir.joinpath("eval.bin")
    for p in (man_path, tr_path, ev_path):
        require_ok(not p.is_symlink(), "SYMLINK_FORBIDDEN", p.name)
        require_ok(p.is_file(), "CORPUS_MISSING", p.name)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man_sha = sha256_file(man_path)
    tr_raw = tr_path.read_bytes()
    ev_raw = ev_path.read_bytes()
    tr_sha = hashlib.sha256(tr_raw).hexdigest()
    ev_sha = hashlib.sha256(ev_raw).hexdigest()
    require_ok(tr_sha == man["train"]["sha256"], "TRAIN_SHA_MISMATCH", tr_sha)
    require_ok(ev_sha == man["eval"]["sha256"], "EVAL_SHA_MISMATCH", ev_sha)
    require_ok(man["strict_sha256_doc_overlap_count"] == 0, "CORPUS_OVERLAP", "overlap != 0")
    if verify_pinned_corpus:
        require_ok(man_sha == EXPECTED_CORPUS_MANIFEST_SHA256, "CORPUS_MANIFEST_PIN_MISMATCH", man_sha)
        require_ok(tr_sha == EXPECTED_TRAIN_BIN_SHA256, "TRAIN_PIN_MISMATCH", tr_sha)
        require_ok(ev_sha == EXPECTED_EVAL_BIN_SHA256, "EVAL_PIN_MISMATCH", ev_sha)

    # Orthonormal 256-point Hadamard (scale 1/16 = 0.0625)
    h256_fp64 = torch.tensor(
        [[(-0.0625 if ((k & j).bit_count() & 1) else 0.0625) for j in range(256)] for k in range(256)],
        dtype=torch.float64,
    )
    h256_cpu = h256_fp64.float()
    h256_ortho_err = float((h256_fp64 @ h256_fp64 - torch.eye(256, dtype=torch.float64)).abs().max().item())
    require_ok(h256_ortho_err < 1e-14, "H256_ORTHO_FAIL", f"err={h256_ortho_err}")

    # Bent-4 signed-Hadamard transform scaled by 1/sqrt(32) = sqrt(8) * orthonormal
    inv_sqrt32 = 1.0 / math.sqrt(32.0)
    m_rows = []
    for tok in range(256):
        row = []
        for d in range(256):
            lane = d & 31
            db = d ^ ((lane & 7) << 5)
            b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
            par = ((tok & db).bit_count() ^ b4) & 1
            row.append((-1.0 if par else 1.0) * inv_sqrt32)
        m_rows.append(row)
    fwht_bent_cpu = torch.tensor(m_rows, dtype=torch.float64).float()

    inj_rows = []
    for tok in range(257):
        b_u8 = tok & 0xFF
        sp_tag = (tok >> 8) & 3
        row512 = []
        for d in range(512):
            bank = d >> 8
            d256 = d & 255
            lane = d256 & 31
            db = d256 ^ ((lane & 7) << 5)
            b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
            sp_flip = sp_tag & ((d256 >> 5) & 1)
            par = ((b_u8 & db).bit_count() ^ b4 ^ sp_flip) & 1
            sgn = -1.0 if bank == 1 else 1.0
            row512.append((-1.0 if par else 1.0) * inv_sqrt32 * sgn)
        inj_rows.append(row512)
    inj_mat_cpu = torch.tensor(inj_rows, dtype=torch.float32)

    def _unit_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return t * torch.rsqrt((t * t).sum(dim=-1, keepdim=True) + eps * eps)

    class CausalDepthwiseByteConv4(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(dim, 1, 4))
            with torch.no_grad():
                self.weight[:, 0, 3] = 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xt = x.transpose(1, 2)
            out = F.conv1d(F.pad(xt, (3, 0)), self.weight, groups=xt.shape[1]).transpose(1, 2)
            return _unit_norm(out)

    class HoloSphereBlock(nn.Module):
        def __init__(self, dim: int, heads: int, ff: int) -> None:
            super().__init__()
            self.heads = heads
            self.hdim = dim // heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
            self.alpha_attn = nn.Parameter(torch.full((dim,), 0.1))
            self.w_gate = nn.Linear(dim, ff, bias=False)
            self.w_up = nn.Linear(dim, ff, bias=False)
            self.w_down = nn.Linear(ff, dim, bias=False)
            self.alpha_ff = nn.Parameter(torch.full((dim,), 0.1))

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            bsz, seq, dim = x.shape
            qkv = self.qkv(x).view(bsz, seq, 3, self.heads, self.hdim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            q1, q2 = q[..., 0::2], q[..., 1::2]
            k1, k2 = k[..., 0::2], k[..., 1::2]
            q = _unit_norm(torch.stack((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1).flatten(-2))
            k = _unit_norm(torch.stack((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1).flatten(-2))
            y = _unit_norm(
                self.out_proj(
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    .transpose(1, 2)
                    .contiguous()
                    .view(bsz, seq, dim)
                )
            )
            x = _unit_norm(x + self.alpha_attn.abs().clamp(max=0.95) * (y - x))
            ff = _unit_norm(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
            return _unit_norm(x + self.alpha_ff.abs().clamp(max=0.95) * (ff - x))

    class HoloSphereV4LM(nn.Module):
        def __init__(self, arm_name: str) -> None:
            super().__init__()
            self.arm_name = arm_name
            self.register_buffer("inj_table", inj_mat_cpu.clone(), persistent=False)
            self.register_buffer("h256_mat", h256_cpu.clone(), persistent=False)
            self.register_buffer("fwht_mat", fwht_bent_cpu.clone(), persistent=False)
            self.byte_conv = CausalDepthwiseByteConv4(d_model)
            self.blocks = nn.ModuleList([HoloSphereBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

            hdim = d_model // n_heads
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, hdim, 2, dtype=torch.float32) / float(hdim)))
            freqs = torch.outer(torch.arange(max_ctx, dtype=torch.float32), inv_freq)
            self.register_buffer("rot_cos", freqs.cos().view(1, 1, max_ctx, hdim // 2), persistent=False)
            self.register_buffer("rot_sin", freqs.sin().view(1, 1, max_ctx, hdim // 2), persistent=False)

            if arm_name == "V3_ScalarFWHT":
                self.logit_scale = nn.Parameter(torch.tensor(1.0))
            elif arm_name == "V4_Monarch2BiasFWHT":
                self.monarch_b1 = nn.Parameter(torch.eye(16).unsqueeze(0).repeat(16, 1, 1).clone())
                self.monarch_b2 = nn.Parameter(torch.eye(16).unsqueeze(0).repeat(16, 1, 1).clone())
                self.byte_bias = nn.Parameter(torch.zeros(256))
            elif arm_name == "V3_LowRank16Bias":
                self.lr_down = nn.Linear(256, 16, bias=False)
                self.lr_up = nn.Linear(16, 256, bias=False)
                self.byte_bias = nn.Parameter(torch.zeros(256))
                with torch.no_grad():
                    self.lr_up.weight.mul_(8.0)
            elif arm_name == "V3_DenseHead":
                self.lm_head = nn.Linear(d_model, 256, bias=False)
                self.logit_scale = nn.Parameter(torch.tensor(8.0))
                self.byte_bias = nn.Parameter(torch.zeros(256))
            else:
                raise V4Monarch2BpbStudyError("BAD_ARM", arm_name)

        def head_spec(self) -> Tuple[int, int, int, int]:
            specs = {
                "V3_ScalarFWHT": (1, 2, 0, 0),
                "V4_Monarch2BiasFWHT": (8448, 16896, 0, 0),
                "V3_LowRank16Bias": (8448, 16896, 0, 16896),
                "V3_DenseHead": (131329, 262658, 0, 262658),
            }
            return specs[self.arm_name]

        def forward(self, tok_ids: torch.Tensor, *, fp16_head_eval: bool = False) -> torch.Tensor:
            x = self.byte_conv(F.embedding(tok_ids, self.inj_table))
            seq = tok_ids.shape[1]
            cos = self.rot_cos[:, :, :seq, :].to(x.dtype)
            sin = self.rot_sin[:, :, :seq, :].to(x.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = _unit_norm(x)

            def _p(t: torch.Tensor) -> torch.Tensor:
                return t.half().to(x.dtype) if fp16_head_eval else t.to(x.dtype)

            if self.arm_name == "V3_ScalarFWHT":
                u = x[..., :256] - x[..., 256:]
                return F.linear(u, self.fwht_mat.to(u.dtype)) * _p(self.logit_scale)

            bias = _p(self.byte_bias)
            if self.arm_name == "V4_Monarch2BiasFWHT":
                bsz, sq, _ = x.shape
                u = x[..., :256] - x[..., 256:]
                z1 = F.linear(u, self.h256_mat.to(u.dtype))
                z2 = z1.view(bsz, sq, 16, 16).transpose(-1, -2).contiguous()
                z3 = torch.einsum("grc,bsgc->bsgr", _p(self.monarch_b1), z2)
                z4 = z3.transpose(-1, -2).contiguous()
                z5 = torch.einsum("grc,bsgc->bsgr", _p(self.monarch_b2), z4).reshape(bsz, sq, 256)
                return F.linear(z5, self.fwht_mat.to(z5.dtype)) + bias

            if self.arm_name == "V3_LowRank16Bias":
                u = x[..., :256] - x[..., 256:]
                return F.linear(F.linear(u, _p(self.lr_down.weight)), _p(self.lr_up.weight)) + bias

            return F.linear(x, _p(self.lm_head.weight)) * _p(self.logit_scale) + bias

    eval_tensors: Dict[int, torch.Tensor] = {}
    for ctx_t, n_chunks in EVAL_CONTEXTS:
        eval_tensors[ctx_t] = torch.stack(
            [
                torch.tensor([256] + list(ev_raw[ci * ctx_t : (ci + 1) * ctx_t]), dtype=torch.long)
                for ci in range(n_chunks)
            ],
            dim=0,
        )

    def _eval_ctx(mod: HoloSphereV4LM, ctx_t: int, *, fp16_head_eval: bool) -> Tuple[float, int]:
        mod.eval()
        ev_all = eval_tensors[ctx_t]
        eb = 8 if ctx_t == 512 else (2 if ctx_t == 2048 else 1)
        tot_nll, tot_toks = 0.0, 0
        with torch.no_grad():
            for i in range(0, ev_all.shape[0], eb):
                b = ev_all[i : i + eb].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = mod(b[:, :-1], fp16_head_eval=fp16_head_eval)
                tot_nll += float(
                    F.cross_entropy(lg.float().reshape(-1, 256), b[:, 1:].reshape(-1), reduction="sum").item()
                )
                tot_toks += int(b[:, 1:].numel())
        return ((tot_nll / float(tot_toks)) / math.log(2.0), tot_toks)

    arm_names = ["V3_ScalarFWHT", "V4_Monarch2BiasFWHT", "V3_LowRank16Bias", "V3_DenseHead"]
    max_start = len(tr_raw) - train_seq_len - 1
    backbone_init_by_seed: Dict[int, Dict[str, torch.Tensor]] = {}
    train_starts_by_seed: Dict[int, List[List[int]]] = {}
    for sd in seed_list:
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        ref_m = HoloSphereV4LM("V3_ScalarFWHT")
        backbone_init_by_seed[sd] = {
            k: v.detach().cpu().clone()
            for k, v in ref_m.state_dict().items()
            if k.startswith("byte_conv.") or k.startswith("blocks.")
        }
        g_dat = torch.Generator(device="cpu").manual_seed(sd * 7919 + 40)
        train_starts_by_seed[sd] = torch.randint(0, max_start, (steps, batch_size), generator=g_dat).tolist()

    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm in arm_names:
        per_seed: List[Dict[str, Any]] = []
        head_params, head_bytes, hbm_in, hbm_out = (0, 0, 0, 0)
        total_params = 0
        for sd in seed_list:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            model = HoloSphereV4LM(arm).to(device)
            cur_sd = model.state_dict()
            for k, v in backbone_init_by_seed[sd].items():
                cur_sd[k].copy_(v)
            model.load_state_dict(cur_sd, strict=True)
            head_params, head_bytes, hbm_in, hbm_out = model.head_spec()
            total_params = int(sum(p.numel() for p in model.parameters()))
            require_ok(total_params == 9_115_648 + head_params, "PARAM_COUNT_MISMATCH", f"{arm}: {total_params}")

            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.01)
            t0 = time.perf_counter()
            model.train()
            starts_sd = train_starts_by_seed[sd]
            for st in range(steps):
                w_steps = max(1, steps // 10)
                if st < w_steps:
                    lr_now = 1.5e-3 * float(st + 1) / float(w_steps)
                else:
                    prog = float(st - w_steps) / float(max(1, steps - w_steps))
                    lr_now = 1.5e-3 * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
                for pg in opt.param_groups:
                    pg["lr"] = lr_now

                rows = [[256] + list(tr_raw[p : p + train_seq_len]) for p in starts_sd[st]]
                batch = torch.tensor(rows, dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(batch[:, :-1], fp16_head_eval=False)
                    loss = F.cross_entropy(lg.float().reshape(-1, 256), batch[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - t0

            ctx_evals: Dict[str, Dict[str, Any]] = {}
            for ctx_t, _ in EVAL_CONTEXTS:
                bpb_bf16, n_toks = _eval_ctx(model, ctx_t, fp16_head_eval=False)
                bpb_fp16, _ = _eval_ctx(model, ctx_t, fp16_head_eval=True)
                q_fp16 = parse_q(to_q_str(bpb_fp16, 100_000))
                q_bf16 = parse_q(to_q_str(bpb_bf16, 100_000))
                ctx_evals[f"T{ctx_t}"] = {
                    "eval_bpb_q": format_q(q_fp16),
                    "eval_bpb_bf16_q": format_q(q_bf16),
                    "fp16_quant_bpb_diff_q": format_q(abs(q_fp16 - q_bf16)),
                    "eval_tokens": int(n_toks),
                }

            per_seed.append(
                {
                    "seed": int(sd),
                    "train_wall_s_q": to_q_str(wall_s, 1000),
                    "contexts": ctx_evals,
                }
            )
            del model, opt
            torch.cuda.empty_cache()

        mean_by_ctx: Dict[str, str] = {}
        for ctx_t, _ in EVAL_CONTEXTS:
            ck = f"T{ctx_t}"
            mq = sum((parse_q(r["contexts"][ck]["eval_bpb_q"]) for r in per_seed), Fraction(0, 1)) / Fraction(
                len(per_seed), 1
            )
            mean_by_ctx[ck] = format_q(mq)

        arm_results[arm] = {
            "head_params": head_params,
            "head_fp16_bytes": head_bytes,
            "hbm_input_vocab_bytes": hbm_in,
            "hbm_output_vocab_bytes": hbm_out,
            "total_params": total_params,
            "mean_eval_bpb_by_ctx_q": mean_by_ctx,
            "seeds": per_seed,
        }

    return {
        "energy_power_status": ENERGY_POLICY_STATUS,
        "corpus_id": CORPUS_ID,
        "corpus_manifest_sha256": man_sha,
        "corpus_train_bin_sha256": tr_sha,
        "corpus_eval_bin_sha256": ev_sha,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "rope_base": int(rope_base),
        "batch_size": batch_size,
        "train_seq_len": train_seq_len,
        "steps_per_seed": steps,
        "tokens_per_seed": steps * batch_size * train_seq_len,
        "eval_contexts": [{"seq_len": t, "chunks": c, "eval_bytes": t * c} for t, c in EVAL_CONTEXTS],
        "seeds": seed_list,
        "h256_ortho_fp64_max_err_q": to_q_str(h256_ortho_err, 10**12),
        "arms": arm_results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="4-Arm x 3-Seed Multi-Context UTF-8 Byte-LM Study Runner for HoloSphere-V4 Monarch-2"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing flagship_corpus_manifest.json, train.bin, and eval.bin",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-seq-len", type=int, default=2048)
    parser.add_argument("--skip-corpus-pin", action="store_true", help="Skip pinned corpus SHA-256 check")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    try:
        study = run_e2e_bpb_study(
            args.corpus_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            train_seq_len=args.train_seq_len,
            seeds=[42, 137, 2026],
            verify_pinned_corpus=not args.skip_corpus_pin,
        )
    except V4Monarch2BpbStudyError as exc:
        sys.stderr.write(
            json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True) + "\n"
        )
        return 2

    payload = json.dumps(study, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
