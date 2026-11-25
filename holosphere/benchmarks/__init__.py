"""Deterministic benchmark tasks and PREREG-S114 verification harness."""

from __future__ import annotations

from holosphere.benchmarks.tasks import (
    A5_ORDER,
    S3_ORDER,
    build_a5_cayley_table,
    build_s3_cayley_table,
    generate_bracket_depth_batch,
    generate_group_tracking_batch,
)

__all__ = [
    "A5_ORDER",
    "S3_ORDER",
    "build_a5_cayley_table",
    "build_s3_cayley_table",
    "generate_bracket_depth_batch",
    "generate_group_tracking_batch",
]
