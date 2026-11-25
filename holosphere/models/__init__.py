"""Unified HoloSphere transformer models, matched controls, MTP heads, & audited Weft shell."""

from __future__ import annotations

from holosphere.models.blocks import CovariantSU2Attention, HoloSphereBlock, SwiGLUMLP
from holosphere.models.config import HoloSphereConfig, build_preset_config
from holosphere.models.controls import (
    Block4RandOrthoPositional,
    DiagonalDarcyEMA,
    LieRE4DAbelianPositional,
    RMSNorm,
    apply_rope_2d,
)
from holosphere.models.heads import LocalARByteDecoder, MarkovMTP8Head, ParallelMTP8Head
from holosphere.models.nullmodem_e8 import (
    BatchedNullModemCell,
    BlockE8SphereCodec,
    NullModemCell,
)
from holosphere.models.transformer import (
    HoloSphereTransformer,
    analytical_forward_flops_per_token,
    apply_ptq_copy,
    count_parameters,
    measure_m1_h1_loop_drift,
    quantize_tensor_rtn_group,
)
from holosphere.models.weft_shell import (
    PlatePatchProposal,
    Shell3MerkleLedger,
    WeftAuditShell,
    composed_c_row,
    verify_shell3_merkle_receipt,
)

__all__ = [
    "BatchedNullModemCell",
    "Block4RandOrthoPositional",
    "BlockE8SphereCodec",
    "CovariantSU2Attention",
    "DiagonalDarcyEMA",
    "LieRE4DAbelianPositional",
    "LocalARByteDecoder",
    "MarkovMTP8Head",
    "NullModemCell",
    "ParallelMTP8Head",
    "PlatePatchProposal",
    "RMSNorm",
    "Shell3MerkleLedger",
    "HoloSphereBlock",
    "HoloSphereConfig",
    "HoloSphereTransformer",
    "SwiGLUMLP",
    "WeftAuditShell",
    "analytical_forward_flops_per_token",
    "apply_ptq_copy",
    "apply_rope_2d",
    "build_preset_config",
    "composed_c_row",
    "count_parameters",
    "measure_m1_h1_loop_drift",
    "quantize_tensor_rtn_group",
    "verify_shell3_merkle_receipt",
]
