#!/usr/bin/env python3
"""Bare-Metal Real-Weight HoloSphere-V3 Autoregressive Inference Runtime (Ampere sm_86).

Loads any trained HoloSphere-V3 checkpoint (e.g., 15M D=512 H=16 L=6 d_ff=800 V=260),
packs all linear projections into Extended Hamming ``[16, 11, 4]`` ``A_16 Block-H4`` ``uint64_t``
codewords (asserting exact 0 syndrome errors), and executes bare-metal ``sm_86`` autoregressive
generation via ``holosphere/kernels/csrc/holosphere_v3_infer_sm86.cu`` (linked against
``include/razor_abi_4byte_sys.h``):
  - Zero-VRAM ``H_256 x bent4`` byte injection + ``CausalDepthwiseByteConv(k=4)``
  - 0-POPC dual-SWAR hot-path ``A_16 Block-H4`` GEMV + inline 1-bit SEC bit-exact heal +
    2-bit DED detection + zero-``cudaMemcpy`` 64-byte ``HoloSphereRazorCanary64`` ``.sys`` D2H
  - Unified CellOp step (``V_ring`` Hard-FMNMX + bounded ``CayleySU2`` + exact FP32 ``S_fast``)
  - Fused SwiGLU + hyperspherical residual + 23-bit finite Gumbel sampling (supporting both
    trained ``A_16`` ``lm_head`` and 0 B VRAM tied ``FWHT_256 x bent4`` LM head)
  - O(1) ``Weft`` 16-byte ``HoloSpherePiRollback16`` holonomy + state checkpoint/rollback

Pure-stdlib-importable (``--self-test --json`` runs without ``torch`` or CUDA).
Zero bare ``assert`` statements. Licensed under Apache-2.0.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from holosphere import e2e_engine as v3_eng
from holosphere.errors import HoloSphereError, require_ok

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - pure stdlib mode
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _configure_infer_lib(lib: ctypes.CDLL) -> None:
    lib.launch_v3_infer_inject_byteconv4.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_inject_byteconv4.restype = ctypes.c_int

    lib.launch_v3_infer_a16_blockh4_gemv_secded.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_a16_blockh4_gemv_secded.restype = ctypes.c_int

    lib.launch_v3_infer_a16_swiglu_gemv_secded.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_a16_swiglu_gemv_secded.restype = ctypes.c_int

    lib.launch_v3_infer_a16_qkc_gemv_secded.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_a16_qkc_gemv_secded.restype = ctypes.c_int

    lib.launch_v3_infer_vring_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_vring_step.restype = ctypes.c_int

    lib.launch_v3_infer_vring_mix.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_vring_mix.restype = ctypes.c_int

    lib.launch_v3_infer_cayley_sfast_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_cayley_sfast_step.restype = ctypes.c_int

    lib.launch_v3_infer_sphere_resid.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_sphere_resid.restype = ctypes.c_int

    lib.launch_v3_infer_swiglu_act.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_swiglu_act.restype = ctypes.c_int

    lib.launch_v3_infer_sample_gumbel23.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_sample_gumbel23.restype = ctypes.c_int

    lib.launch_v3_infer_tied_fwht256_lm_head.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_tied_fwht256_lm_head.restype = ctypes.c_int

    lib.launch_v3_infer_weft_state_copy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_v3_infer_weft_state_copy.restype = ctypes.c_int

    lib.v3_infer_alloc_razor_canary64.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    lib.v3_infer_alloc_razor_canary64.restype = ctypes.c_int
    lib.v3_infer_free_razor_canary64.argtypes = [ctypes.c_void_p]
    lib.v3_infer_free_razor_canary64.restype = ctypes.c_int
    lib.v3_infer_poll_razor_canary64.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    lib.v3_infer_poll_razor_canary64.restype = None
    lib.v3_infer_reset_razor_canary64.argtypes = [ctypes.c_void_p]
    lib.v3_infer_reset_razor_canary64.restype = None


def check_a16_zero_syndrome(syn_errors: int) -> None:
    """Fail-closed exact zero-syndrome gate for A_16 Block-H4 packing."""
    require_ok(
        v3_eng._is_strict_int(syn_errors) and syn_errors == 0,
        f"A_16 packing produced {syn_errors} syndrome errors",
        code="A16_TORCH_PACK_SYN_ERR",
    )


def check_tied_fwht256_dims(d_model: int, vocab_size: int) -> None:
    """Fail-closed bounds guard matching ``launch_v3_infer_tied_fwht256_lm_head`` (D == 512, V >= 256)."""
    require_ok(
        v3_eng._is_strict_int(d_model)
        and d_model == 512
        and v3_eng._is_strict_int(vocab_size)
        and vocab_size >= 256,
        f"Tied FWHT_256 LM head requires d_model==512 and vocab_size>=256 (got D={d_model}, V={vocab_size})",
        code="FWHT_HEAD_DIMS",
    )


def pack_linear_a16_blockh4_torch(w_f32: Any) -> Tuple[Any, Any, Any, int]:
    """Pack a 2D weight ``[out, in]`` into ``A_16 Block-H4`` ``uint64_t`` codewords + FP16 scales.

    Works on both CPU and CUDA PyTorch tensors and verifies exact ``0`` syndrome errors across
    all packed 16-weight codewords.
    Returns ``(packed_i64_tensor, scales_f16_tensor, w_dequant_f32_tensor, syndrome_errors)``.
    """
    require_ok(_HAS_TORCH, "PyTorch is required for vectorized A_16 packing", code="NO_TORCH")
    require_ok(
        isinstance(w_f32, torch.Tensor) and w_f32.ndim == 2 and w_f32.shape[0] > 0 and w_f32.shape[1] > 0,
        "w_f32 must be a non-empty 2D Tensor",
        code="A16_BAD_SHAPE",
    )
    require_ok(w_f32.is_floating_point(), "w_f32 must be a floating-point Tensor (not bool/int)", code="A16_BAD_DTYPE")
    out_dim, in_dim = int(w_f32.shape[0]), int(w_f32.shape[1])
    require_ok(in_dim % 32 == 0, f"in_dim={in_dim} must be divisible by 32", code="A16_IN_DIV32")
    require_ok(bool(torch.isfinite(w_f32).all().item()), "w_f32 must contain only finite values", code="A16_NONFINITE")
    dev = w_f32.device

    h4 = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
        dtype=torch.float32,
        device=dev,
    )
    w_blk = w_f32.float().view(out_dim, in_dim // 32, 8, 4)
    u_raw = 0.5 * torch.matmul(w_blk, h4)
    amax_raw = u_raw.abs().amax(dim=(-2, -1), keepdim=True)
    require_ok(
        bool(torch.isfinite(amax_raw).all().item()) and float(amax_raw.max().item()) <= v3_eng.A16_MAX_URAW_ABS,
        f"A_16 Block-H4 scale overflow: max|u_raw| exceeds FP16 scale bound {v3_eng.A16_MAX_URAW_ABS}",
        code="A16_OVERFLOW",
    )
    amax = amax_raw.clamp(min=1e-6)
    sc_f16 = (amax / 7.5).to(torch.float16)
    sc_f32 = sc_f16.float()

    u_scaled = (u_raw / sc_f32).clamp(-8.0, 7.0)
    c_int = torch.round(u_scaled).clamp(-8.0, 7.0).to(torch.int32).view(out_dim, in_dim // 16, 16)
    u_s16 = u_scaled.view(out_dim, in_dim // 16, 16)

    masks_p = (
        (1, (3, 5, 7, 9, 11, 13, 15)),
        (2, (3, 6, 7, 10, 11, 14, 15)),
        (4, (5, 6, 7, 12, 13, 14, 15)),
        (8, (9, 10, 11, 12, 13, 14, 15)),
    )

    def _snap_pos(pos: int, req_lsb: torch.Tensor) -> None:
        cur = c_int[..., pos]
        need = (cur & 1) != req_lsb
        u_p = u_s16[..., pos]
        step = torch.where(u_p >= cur.float(), torch.ones_like(cur), -torch.ones_like(cur))
        step = torch.where((cur == 7) & (step > 0), -torch.ones_like(cur), step)
        step = torch.where((cur == -8) & (step < 0), torch.ones_like(cur), step)
        c_int[..., pos] = torch.where(need, cur + step, cur)

    for p_pos, idxs in masks_p:
        req = torch.zeros_like(c_int[..., 0])
        for j in idxs:
            req = req ^ (c_int[..., j] & 1)
        _snap_pos(p_pos, req)

    req0 = torch.zeros_like(c_int[..., 0])
    for j in range(1, 16):
        req0 = req0 ^ (c_int[..., j] & 1)
    _snap_pos(0, req0)

    odd_bits = c_int & 1
    odd_mask = torch.zeros(c_int.shape[:-1], dtype=torch.int32, device=dev)
    for j in range(16):
        odd_mask = odd_mask | (odd_bits[..., j] << j)

    def _pop_odd(m: torch.Tensor) -> torch.Tensor:
        v = m ^ (m >> 8)
        v = v ^ (v >> 4)
        v = v ^ (v >> 2)
        v = v ^ (v >> 1)
        return v & 1

    syn = (
        _pop_odd(odd_mask)
        | (_pop_odd(odd_mask & 0xAAAA) << 1)
        | (_pop_odd(odd_mask & 0xCCCC) << 2)
        | (_pop_odd(odd_mask & 0xF0F0) << 3)
        | (_pop_odd(odd_mask & 0xFF00) << 4)
    )
    syn_errors = int((syn != 0).sum().item())
    check_a16_zero_syndrome(syn_errors)

    nib = (c_int & 0xF).to(torch.int64)
    packed = torch.zeros(c_int.shape[:-1], dtype=torch.int64, device=dev)
    for j in range(16):
        packed = packed | (nib[..., j] << (j * 4))

    r_int8 = torch.matmul(c_int.float().view(out_dim, in_dim // 32, 8, 4), h4)
    w_deq = (0.5 * r_int8 * sc_f32).reshape(out_dim, in_dim)
    return packed.contiguous(), sc_f16.squeeze(-1).squeeze(-1).contiguous(), w_deq.contiguous(), syn_errors


class HoloSphereV3BareMetalInferEngine:
    """Executes real trained HoloSphere-V3 checkpoints end-to-end on Ampere ``sm_86`` CUDA."""

    def __init__(self, ckpt_path: Any, device: Any, csrc_dir: Any = v3_eng.DEFAULT_CSRC_DIR) -> None:
        ckpt_p = v3_eng._coerce_path(ckpt_path, "CKPT_MISSING")
        require_ok(ckpt_p.is_file(), f"Missing checkpoint {ckpt_p}", code="CKPT_MISSING")
        require_ok(_HAS_TORCH and torch.cuda.is_available(), "CUDA GPU required for live inference", code="CUDA_REQ")
        csrc_p = v3_eng._coerce_path(csrc_dir, "CU_NOT_FOUND")
        self.device = device
        self.lib = v3_eng.compile_and_load_cuda_lib(csrc_p / v3_eng.INFER_CU_FILENAME)
        _configure_infer_lib(self.lib)

        ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.d_model = int(sd["byte_conv.conv.weight"].shape[0])
        self.n_heads = self.d_model // 32
        self.vocab_size = int(sd["lm_head.weight"].shape[0])
        self.n_layers = sum(1 for k in sd if k.endswith(".alpha_cell"))
        self.d_ff = int(sd["blocks.0.ffn.w_gate.weight"].shape[0])
        self.logit_scale = float(sd["logit_scale"].item())
        self.conv_w = sd["byte_conv.conv.weight"].squeeze(1).float().contiguous().to(device)

        self.layers: List[Dict[str, Any]] = []
        self.total_a16_blocks = 0
        self.total_a16_syndrome_errors = 0
        self.total_packed_bytes = 0

        def _pack(t_cpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            p_u64, sc_f16, w_deq, serr = pack_linear_a16_blockh4_torch(t_cpu.to(device))
            self.total_a16_blocks += int(p_u64.numel())
            self.total_a16_syndrome_errors += int(serr)
            self.total_packed_bytes += int(p_u64.numel() * 8 + sc_f16.numel() * 2)
            return p_u64, sc_f16, w_deq

        for layer_idx in range(self.n_layers):
            pfx = f"blocks.{layer_idx}."
            ld: Dict[str, Any] = {
                "omega_scale": sd[pfx + "omega_scale"].float().contiguous().to(device),
                "w_q_fast": sd[pfx + "w_q_fast"].float().contiguous().to(device),
                "alpha_cell": sd[pfx + "alpha_cell"].float().contiguous().to(device),
                "alpha_ring": sd[pfx + "alpha_ring"].float().contiguous().to(device),
                "alpha_mlp": sd[pfx + "alpha_mlp"].float().contiguous().to(device),
            }
            for lname in (
                "w_q_ring", "w_k_ring", "w_ring_ctrl", "w_mem_out",
                "w_omega", "w_cell_out", "ffn.w_gate", "ffn.w_up", "ffn.w_down",
            ):
                p_u64, sc_f16, w_deq = _pack(sd[pfx + lname + ".weight"])
                ld[lname + "_u64"] = p_u64
                ld[lname + "_sc"] = sc_f16
                ld[lname + "_deq"] = w_deq
            self.layers.append(ld)

        self.lm_u64, self.lm_sc, self.lm_deq = _pack(sd["lm_head.weight"])
        self._alloc_state_and_canary()

    def _alloc_state_and_canary(self) -> None:
        dev = self.device
        D, H, L, FF, V = self.d_model, self.n_heads, self.n_layers, self.d_ff, self.vocab_size
        n_conv = 4 * D
        n_sfast = L * H * 32 * 32
        n_u = L * H * 8 * 4
        n_kr = L * H * 8 * 32
        n_vr = L * H * 8 * 32
        self.state_floats = n_conv + n_sfast + n_u + n_kr + n_vr
        self.live_state = torch.zeros((self.state_floats,), dtype=torch.float32, device=dev)
        self.ckpt_state = torch.zeros((self.state_floats,), dtype=torch.float32, device=dev)

        o1 = n_conv
        o2 = o1 + n_sfast
        o3 = o2 + n_u
        o4 = o3 + n_kr
        self.conv_ring = self.live_state[0:o1].view(4, D)
        self.S_fast = self.live_state[o1:o2].view(L, H, 32, 32)
        self.U_state = self.live_state[o2:o3].view(L, H, 8, 4)
        self.k_ring = self.live_state[o3:o4].view(L, H, 8, 32)
        self.v_ring = self.live_state[o4:self.state_floats].view(L, H, 8, 32)
        self.conv_pos = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.ckpt_conv_pos = 0

        self.d_tok_in = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.d_tok_out = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.d_score_out = torch.zeros((1,), dtype=torch.float32, device=dev)
        self.d_rng_u32 = torch.zeros((max(V, 256),), dtype=torch.int32, device=dev)
        self.x_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.q_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.k_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.ctrl_buf = torch.empty((3 * H,), dtype=torch.float32, device=dev)
        self.ymem_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.rgate_buf = torch.empty((H,), dtype=torch.float32, device=dev)
        self.mproj_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.hgen_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.wraw_buf = torch.empty((3 * (D // 4),), dtype=torch.float32, device=dev)
        self.ycell_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.cout_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.xmid_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.act_buf = torch.empty((FF,), dtype=torch.float32, device=dev)
        self.down_buf = torch.empty((D,), dtype=torch.float32, device=dev)
        self.logits_buf = torch.empty((V,), dtype=torch.float32, device=dev)

        self.canary_host = ctypes.c_void_p(0)
        self.canary_dev = ctypes.c_void_p(0)
        rc = self.lib.v3_infer_alloc_razor_canary64(ctypes.byref(self.canary_host), ctypes.byref(self.canary_dev))
        v3_eng.check_cuda_rc(rc, "CANARY_ALLOC_FAIL", "cudaHostAlloc")
        self.reset_state()

    def close(self) -> None:
        if getattr(self, "canary_host", None) and self.canary_host.value:
            self.lib.v3_infer_free_razor_canary64(self.canary_host)
            self.canary_host = ctypes.c_void_p(0)
            self.canary_dev = ctypes.c_void_p(0)

    def reset_state(self) -> None:
        self.live_state.zero_()
        self.U_state[..., 0] = 1.0
        self.conv_pos.zero_()
        self.ckpt_conv_pos = 0
        self.lib.v3_infer_reset_razor_canary64(self.canary_host)

    def poll_canary_sys(self) -> Dict[str, int]:
        arr = (ctypes.c_uint32 * 10)()
        self.lib.v3_infer_poll_razor_canary64(self.canary_host, arr)
        return {name: int(arr[idx]) for idx, name in enumerate(v3_eng._RAZOR_FIELDS)}

    def save_weft_checkpoint(self, stream: Any = None) -> bytes:
        s_ptr = ctypes.c_void_p(int((stream or torch.cuda.current_stream(self.device)).cuda_stream))
        rc = self.lib.launch_v3_infer_weft_state_copy(
            ctypes.c_void_p(self.ckpt_state.data_ptr()),
            ctypes.c_void_p(self.live_state.data_ptr()),
            int(self.state_floats),
            s_ptr,
        )
        v3_eng.check_cuda_rc(rc, "WEFT_SAVE_ERR", "weft_state_save")
        self.ckpt_conv_pos = int(self.conv_pos.item())
        u0 = self.U_state[0, 0, 0].detach().cpu().tolist()
        return struct.pack("<4f", float(u0[0]), float(u0[1]), float(u0[2]), float(u0[3]))

    def restore_weft_checkpoint(self, stream: Any = None) -> None:
        s_ptr = ctypes.c_void_p(int((stream or torch.cuda.current_stream(self.device)).cuda_stream))
        rc = self.lib.launch_v3_infer_weft_state_copy(
            ctypes.c_void_p(self.live_state.data_ptr()),
            ctypes.c_void_p(self.ckpt_state.data_ptr()),
            int(self.state_floats),
            s_ptr,
        )
        v3_eng.check_cuda_rc(rc, "WEFT_RESTORE_ERR", "weft_state_restore")
        self.conv_pos.fill_(self.ckpt_conv_pos)

    def enqueue_token_step(self, *, use_tied_fwht_head: bool = False, temperature: float = 0.0, stream: Any = None) -> None:
        cur_s = stream or torch.cuda.current_stream(self.device)
        s_ptr = ctypes.c_void_p(int(cur_s.cuda_stream))
        D, H, FF, V = self.d_model, self.n_heads, self.d_ff, self.vocab_size
        lib = self.lib
        can = self.canary_dev

        def _check(rc: int, code: str) -> None:
            v3_eng.check_cuda_rc(rc, code, "v3_infer")

        def _gemv(x_t: torch.Tensor, u64_t: torch.Tensor, sc_t: torch.Tensor, y_t: torch.Tensor, scale: float = 1.0) -> None:
            _check(
                lib.launch_v3_infer_a16_blockh4_gemv_secded(
                    ctypes.c_void_p(x_t.data_ptr()),
                    ctypes.c_void_p(u64_t.data_ptr()),
                    ctypes.c_void_p(sc_t.data_ptr()),
                    ctypes.c_void_p(y_t.data_ptr()),
                    can,
                    float(scale),
                    int(u64_t.shape[0]),
                    int(x_t.numel()),
                    s_ptr,
                ),
                "GEMV_ERR",
            )

        _check(
            lib.launch_v3_infer_inject_byteconv4(
                ctypes.c_void_p(self.d_tok_in.data_ptr()),
                ctypes.c_void_p(self.conv_w.data_ptr()),
                ctypes.c_void_p(self.conv_ring.data_ptr()),
                ctypes.c_void_p(self.conv_pos.data_ptr()),
                ctypes.c_void_p(self.x_buf.data_ptr()),
                int(D),
                s_ptr,
            ),
            "INJECT_ERR",
        )

        for layer_idx, ld in enumerate(self.layers):
            _check(
                lib.launch_v3_infer_a16_qkc_gemv_secded(
                    ctypes.c_void_p(self.x_buf.data_ptr()),
                    ctypes.c_void_p(ld["w_q_ring_u64"].data_ptr()), ctypes.c_void_p(ld["w_q_ring_sc"].data_ptr()),
                    ctypes.c_void_p(ld["w_k_ring_u64"].data_ptr()), ctypes.c_void_p(ld["w_k_ring_sc"].data_ptr()),
                    ctypes.c_void_p(ld["w_ring_ctrl_u64"].data_ptr()), ctypes.c_void_p(ld["w_ring_ctrl_sc"].data_ptr()),
                    ctypes.c_void_p(self.q_buf.data_ptr()), ctypes.c_void_p(self.k_buf.data_ptr()),
                    ctypes.c_void_p(self.ctrl_buf.data_ptr()), can, int(D), int(3 * H), s_ptr,
                ),
                "QKC_ERR",
            )
            _check(
                lib.launch_v3_infer_vring_step(
                    ctypes.c_void_p(self.x_buf.data_ptr()),
                    ctypes.c_void_p(self.q_buf.data_ptr()),
                    ctypes.c_void_p(self.k_buf.data_ptr()),
                    ctypes.c_void_p(self.ctrl_buf.data_ptr()),
                    ctypes.c_void_p(self.k_ring[layer_idx].data_ptr()),
                    ctypes.c_void_p(self.v_ring[layer_idx].data_ptr()),
                    ctypes.c_void_p(self.ymem_buf.data_ptr()),
                    ctypes.c_void_p(self.rgate_buf.data_ptr()),
                    ctypes.c_void_p(self.conv_pos.data_ptr()),
                    int(H),
                    s_ptr,
                ),
                "VRING_ERR",
            )
            _gemv(self.ymem_buf, ld["w_mem_out_u64"], ld["w_mem_out_sc"], self.mproj_buf)
            _check(
                lib.launch_v3_infer_vring_mix(
                    ctypes.c_void_p(self.x_buf.data_ptr()),
                    ctypes.c_void_p(self.mproj_buf.data_ptr()),
                    ctypes.c_void_p(self.rgate_buf.data_ptr()),
                    ctypes.c_void_p(ld["alpha_ring"].data_ptr()),
                    ctypes.c_void_p(self.hgen_buf.data_ptr()),
                    int(D),
                    s_ptr,
                ),
                "VMIX_ERR",
            )
            _gemv(self.hgen_buf, ld["w_omega_u64"], ld["w_omega_sc"], self.wraw_buf)
            _check(
                lib.launch_v3_infer_cayley_sfast_step(
                    ctypes.c_void_p(self.hgen_buf.data_ptr()),
                    ctypes.c_void_p(self.wraw_buf.data_ptr()),
                    ctypes.c_void_p(ld["omega_scale"].data_ptr()),
                    ctypes.c_void_p(ld["w_q_fast"].data_ptr()),
                    ctypes.c_void_p(self.S_fast[layer_idx].data_ptr()),
                    ctypes.c_void_p(self.U_state[layer_idx].data_ptr()),
                    ctypes.c_void_p(self.ycell_buf.data_ptr()),
                    0.25, 0.90, 0.15, 1e-6, int(H), s_ptr,
                ),
                "CAYLEY_ERR",
            )
            _gemv(self.ycell_buf, ld["w_cell_out_u64"], ld["w_cell_out_sc"], self.cout_buf)
            _check(
                lib.launch_v3_infer_sphere_resid(
                    ctypes.c_void_p(self.hgen_buf.data_ptr()),
                    ctypes.c_void_p(self.cout_buf.data_ptr()),
                    ctypes.c_void_p(ld["alpha_cell"].data_ptr()),
                    ctypes.c_void_p(self.xmid_buf.data_ptr()),
                    int(D),
                    s_ptr,
                ),
                "RESID1_ERR",
            )
            _check(
                lib.launch_v3_infer_a16_swiglu_gemv_secded(
                    ctypes.c_void_p(self.xmid_buf.data_ptr()),
                    ctypes.c_void_p(ld["ffn.w_gate_u64"].data_ptr()), ctypes.c_void_p(ld["ffn.w_gate_sc"].data_ptr()),
                    ctypes.c_void_p(ld["ffn.w_up_u64"].data_ptr()), ctypes.c_void_p(ld["ffn.w_up_sc"].data_ptr()),
                    ctypes.c_void_p(self.act_buf.data_ptr()), can, int(FF), int(D), s_ptr,
                ),
                "SWIGLU_ERR",
            )
            _gemv(self.act_buf, ld["ffn.w_down_u64"], ld["ffn.w_down_sc"], self.down_buf)
            _check(
                lib.launch_v3_infer_sphere_resid(
                    ctypes.c_void_p(self.xmid_buf.data_ptr()),
                    ctypes.c_void_p(self.down_buf.data_ptr()),
                    ctypes.c_void_p(ld["alpha_mlp"].data_ptr()),
                    ctypes.c_void_p(self.x_buf.data_ptr()),
                    int(D),
                    s_ptr,
                ),
                "RESID2_ERR",
            )

        if use_tied_fwht_head:
            check_tied_fwht256_dims(D, V)
            _check(
                lib.launch_v3_infer_tied_fwht256_lm_head(
                    ctypes.c_void_p(self.x_buf.data_ptr()),
                    ctypes.c_void_p(self.logits_buf.data_ptr()),
                    float(self.logit_scale),
                    int(D),
                    int(V),
                    s_ptr,
                ),
                "FWHT_HEAD_ERR",
            )
            eff_v = 256
        else:
            _gemv(self.x_buf, self.lm_u64, self.lm_sc, self.logits_buf, scale=self.logit_scale)
            eff_v = V

        _check(
            lib.launch_v3_infer_sample_gumbel23(
                ctypes.c_void_p(self.logits_buf.data_ptr()),
                ctypes.c_void_p(self.d_rng_u32.data_ptr()),
                ctypes.c_void_p(self.d_tok_out.data_ptr()),
                ctypes.c_void_p(self.d_score_out.data_ptr()),
                float(temperature),
                int(eff_v),
                s_ptr,
            ),
            "SAMPLE_ERR",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HoloSphere-V3 Bare-Metal Inference Runtime & Self-Test.")
    parser.add_argument("--check", action="store_true", help="Verify inference ptxas & live receipts.")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-safe stdlib A_16 / Weft / Razor self-test.")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    args = parser.parse_args(argv)

    run_all = not (args.check or args.self_test)
    out: Dict[str, Any] = {"schema": "holosphere.track_s117.e2e_inference_cli.v1"}
    if args.check or run_all:
        ptx = v3_eng.verify_e2e_ptxas_receipt()
        live = v3_eng.verify_e2e_live_receipts()
        out["inference_ptxas_unit"] = ptx["units"]["inference_runtime"]
        out["inference_live_summary"] = {
            "zone_b_receipt_sha256": live["zone_b_receipt_sha256"],
            "a16_u64_blocks": live["a16_u64_blocks"],
            "infer_max_logit_abs_err": live["infer_max_logit_abs_err"],
            "sec16_bit_exact": live["sec16_bit_exact"],
            "ded1_weft_rollback_bit_exact": live["ded1_weft_rollback_bit_exact"],
        }
    if args.self_test or run_all:
        out["stdlib_a16_secded"] = v3_eng.run_e2e_stdlib_self_test()

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        sections = ",".join(k for k in sorted(out) if k != "schema")
        print(f"PASS holosphere.e2e_inference sections={sections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
