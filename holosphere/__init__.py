"""HoloSphere: Kernel-Isomorphic SU(2) Hypersphere Transformer & Bare-Metal Operator Suite.

Unifies bare-metal Ampere ``sm_86`` ``float4`` quaternion holonomy kernels, mechanized
pure-``Init`` ``Lean 4.36.0`` algebraic invariants, and the verified ``S106-S114``
natural-systems hypersphere transformer research architecture into one entity
(``Moissanite-Labs/holosphere``).
"""

from __future__ import annotations

PROJECT_NAME = "holosphere"
REPO_SLUG = "Moissanite-Labs/holosphere"
__version__ = "0.1.0"

from holosphere.errors import HoloSphereError, require_ok
from holosphere.kernels.dispatch import (
    available_backends,
    fused_spherical_step,
    safe_hopf_clock,
    su2_holonomy_transport,
    su2_inverse_untransport,
)
from holosphere.models.config import HoloSphereConfig, build_preset_config
from holosphere.models.nullmodem_e8 import BatchedNullModemCell, BlockE8SphereCodec, NullModemCell
from holosphere.models.transformer import HoloSphereTransformer

__all__ = [
    "BatchedNullModemCell",
    "BlockE8SphereCodec",
    "HoloSphereConfig",
    "HoloSphereError",
    "HoloSphereTransformer",
    "NullModemCell",
    "PROJECT_NAME",
    "REPO_SLUG",
    "__version__",
    "available_backends",
    "build_preset_config",
    "fused_spherical_step",
    "require_ok",
    "safe_hopf_clock",
    "su2_holonomy_transport",
    "su2_inverse_untransport",
]
