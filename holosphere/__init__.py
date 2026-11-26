"""HoloSphere: Kernel-Isomorphic SU(2) Hypersphere Transformer & Bare-Metal Operator Suite.

Unifies bare-metal Ampere ``sm_86`` ``float4`` quaternion holonomy kernels, mechanized
pure-``Init`` ``Lean 4.36.0`` algebraic invariants, and the verified ``S106-S114``
natural-systems hypersphere transformer research architecture into one entity
(``Moissanite-Labs/holosphere``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

PROJECT_NAME = "holosphere"
REPO_SLUG = "Moissanite-Labs/holosphere"
__version__ = "0.1.0"

from holosphere.errors import HoloSphereError, HoloSphereValueError, require_ok
from holosphere.telemetry import HoloSphereStepTelemetry, HoloSphereTuningProfile, build_step_telemetry

if TYPE_CHECKING:
    from holosphere.chunk32_ops import (
        chunk32_holosphere_layer,
        estimate_t8192_vram_budget,
        h256_bent4_byte_inject_s8,
        h256_bent4_embed_torch,
    )
    from holosphere.e2e_engine import (
        holosphere_v3_chunk32_fused,
        run_e2e_stdlib_self_test,
        verify_e2e_live_receipts,
        verify_e2e_ptxas_receipt,
    )
    from holosphere.e2e_inference import (
        HoloSphereV3BareMetalInferEngine,
        pack_linear_a16_blockh4_torch,
    )
    from holosphere.geometry.beltrami import Cayley2x2AutogradFn, cayley_2x2_rotate
    from holosphere.geometry.su2 import cayley_su2
    from holosphere.kernels.dispatch import (
        available_backends,
        cayley_2x2_step,
        fused_spherical_step,
        hopf_s3_cayley_2x2_step,
        safe_hopf_clock,
        su2_holonomy_transport,
        su2_inverse_untransport,
    )
    from holosphere.models.config import HoloSphereConfig, build_preset_config
    from holosphere.models.nullmodem_e8 import BatchedNullModemCell, BlockE8SphereCodec, NullModemCell
    from holosphere.models.transformer import HoloSphereTransformer

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BatchedNullModemCell": ("holosphere.models.nullmodem_e8", "BatchedNullModemCell"),
    "BlockE8SphereCodec": ("holosphere.models.nullmodem_e8", "BlockE8SphereCodec"),
    "Cayley2x2AutogradFn": ("holosphere.geometry.beltrami", "Cayley2x2AutogradFn"),
    "HoloSphereConfig": ("holosphere.models.config", "HoloSphereConfig"),
    "HoloSphereTransformer": ("holosphere.models.transformer", "HoloSphereTransformer"),
    "HoloSphereV3BareMetalInferEngine": ("holosphere.e2e_inference", "HoloSphereV3BareMetalInferEngine"),
    "NullModemCell": ("holosphere.models.nullmodem_e8", "NullModemCell"),
    "available_backends": ("holosphere.kernels.dispatch", "available_backends"),
    "build_preset_config": ("holosphere.models.config", "build_preset_config"),
    "cayley_2x2_rotate": ("holosphere.geometry.beltrami", "cayley_2x2_rotate"),
    "cayley_2x2_step": ("holosphere.kernels.dispatch", "cayley_2x2_step"),
    "cayley_su2": ("holosphere.geometry.su2", "cayley_su2"),
    "chunk32_holosphere_layer": ("holosphere.chunk32_ops", "chunk32_holosphere_layer"),
    "estimate_t8192_vram_budget": ("holosphere.chunk32_ops", "estimate_t8192_vram_budget"),
    "fused_spherical_step": ("holosphere.kernels.dispatch", "fused_spherical_step"),
    "h256_bent4_byte_inject_s8": ("holosphere.chunk32_ops", "h256_bent4_byte_inject_s8"),
    "h256_bent4_embed_torch": ("holosphere.chunk32_ops", "h256_bent4_embed_torch"),
    "holosphere_v3_chunk32_fused": ("holosphere.e2e_engine", "holosphere_v3_chunk32_fused"),
    "hopf_s3_cayley_2x2_step": ("holosphere.kernels.dispatch", "hopf_s3_cayley_2x2_step"),
    "pack_linear_a16_blockh4_torch": ("holosphere.e2e_inference", "pack_linear_a16_blockh4_torch"),
    "run_e2e_stdlib_self_test": ("holosphere.e2e_engine", "run_e2e_stdlib_self_test"),
    "safe_hopf_clock": ("holosphere.kernels.dispatch", "safe_hopf_clock"),
    "su2_holonomy_transport": ("holosphere.kernels.dispatch", "su2_holonomy_transport"),
    "su2_inverse_untransport": ("holosphere.kernels.dispatch", "su2_inverse_untransport"),
    "verify_e2e_live_receipts": ("holosphere.e2e_engine", "verify_e2e_live_receipts"),
    "verify_e2e_ptxas_receipt": ("holosphere.e2e_engine", "verify_e2e_ptxas_receipt"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    value = getattr(importlib.import_module(mod_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BatchedNullModemCell",
    "BlockE8SphereCodec",
    "Cayley2x2AutogradFn",
    "HoloSphereConfig",
    "HoloSphereError",
    "HoloSphereStepTelemetry",
    "HoloSphereTransformer",
    "HoloSphereTuningProfile",
    "HoloSphereV3BareMetalInferEngine",
    "HoloSphereValueError",
    "NullModemCell",
    "PROJECT_NAME",
    "REPO_SLUG",
    "__version__",
    "available_backends",
    "build_preset_config",
    "build_step_telemetry",
    "cayley_2x2_rotate",
    "cayley_2x2_step",
    "cayley_su2",
    "chunk32_holosphere_layer",
    "estimate_t8192_vram_budget",
    "fused_spherical_step",
    "h256_bent4_byte_inject_s8",
    "h256_bent4_embed_torch",
    "holosphere_v3_chunk32_fused",
    "hopf_s3_cayley_2x2_step",
    "pack_linear_a16_blockh4_torch",
    "require_ok",
    "run_e2e_stdlib_self_test",
    "safe_hopf_clock",
    "su2_holonomy_transport",
    "su2_inverse_untransport",
    "verify_e2e_live_receipts",
    "verify_e2e_ptxas_receipt",
]
