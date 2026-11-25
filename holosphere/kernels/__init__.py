"""Unified kernel-isomorphic operator interface for HoloSphere."""

from __future__ import annotations

from holosphere.kernels.dispatch import (
    available_backends,
    cayley_2x2_step,
    fused_spherical_step,
    hopf_s3_cayley_2x2_step,
    safe_hopf_clock,
    su2_holonomy_transport,
    su2_inverse_untransport,
)
from holosphere.kernels.siliquant_bridge import (
    A16_PARITY_MASKS,
    a16_encode_data11,
    a16_hamming_syndrome,
    build_e8_root_codebook,
    quantize_e8_block8,
)
from holosphere.kernels.torch_ops import (
    SU2UnitPrefixScanFunction,
    su2_unit_prefix_scan_fast,
)

__all__ = [
    "A16_PARITY_MASKS",
    "SU2UnitPrefixScanFunction",
    "a16_encode_data11",
    "a16_hamming_syndrome",
    "available_backends",
    "build_e8_root_codebook",
    "cayley_2x2_step",
    "fused_spherical_step",
    "hopf_s3_cayley_2x2_step",
    "quantize_e8_block8",
    "safe_hopf_clock",
    "su2_holonomy_transport",
    "su2_inverse_untransport",
    "su2_unit_prefix_scan_fast",
]
