"""Exact natural-systems geometry primitives unified across S106-S114."""

from __future__ import annotations

from holosphere.geometry.beltrami import (
    Cayley2x2AutogradFn,
    CayleyBlock2x2SO,
    build_paired_beltrami_operator,
    cayley_2x2_rotate,
    measure_trace_beltrami_rayleigh,
    run_beltrami_helicity_diagnostic,
)
from holosphere.geometry.bingham import VectorProximalPoloidalBingham
from holosphere.geometry.darboux import (
    darboux_intertwining_diagnostic,
    measure_dark_mode_energy_fraction,
)
from holosphere.geometry.hopf import (
    build_s3_width_schedule,
    cayley_acute_cos_sin,
    hopf_clock_angle,
    hopf_fiber_rotate,
    hopf_fiber_rotate_cs,
    hopf_s3_residual_step,
    hopf_s3_to_s2,
    spearman_rank_corr,
)
from holosphere.geometry.su2 import (
    cayley_su2,
    quat_conj,
    quat_inverse_untransport,
    quat_left_transport,
    quat_mul,
    safe_slerp_step,
    safe_unit_norm,
    su2_exp,
    su2_parallel_prefix_scan,
    su2_sequential_prefix_scan,
    verify_su2_interval_holonomy,
)

__all__ = [
    "Cayley2x2AutogradFn",
    "CayleyBlock2x2SO",
    "VectorProximalPoloidalBingham",
    "build_paired_beltrami_operator",
    "build_s3_width_schedule",
    "cayley_2x2_rotate",
    "cayley_acute_cos_sin",
    "cayley_su2",
    "darboux_intertwining_diagnostic",
    "hopf_clock_angle",
    "hopf_fiber_rotate",
    "hopf_fiber_rotate_cs",
    "hopf_s3_residual_step",
    "hopf_s3_to_s2",
    "measure_dark_mode_energy_fraction",
    "measure_trace_beltrami_rayleigh",
    "quat_conj",
    "quat_inverse_untransport",
    "quat_left_transport",
    "quat_mul",
    "run_beltrami_helicity_diagnostic",
    "safe_slerp_step",
    "safe_unit_norm",
    "spearman_rank_corr",
    "su2_exp",
    "su2_parallel_prefix_scan",
    "su2_sequential_prefix_scan",
    "verify_su2_interval_holonomy",
]
