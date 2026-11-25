"""One-way SiliSense -> HoloSphere telemetry & step-boundary tuning contract.

Strict Architectural Boundary (Phase-2 S115 / SiliSense Integration):
  - Pure Python stdlib ONLY (zero torch, zero silisense imports).
  - Zero hardware/sensor/privacy coupling: NEVER opens camera, audio/microphone,
    RAPL sysfs, NVML, or any OS hardware sensor handle.
  - Strictly one-way dependency: external governors (e.g., Moissanite-Labs/silisense)
    read ``HoloSphereStepTelemetry`` emitted between steps and supply a validated,
    semantics-preserving ``HoloSphereTuningProfile`` whose SHA-256 stamps every run.
  - Zero bare assert statements; fail-closed validation via ``require_ok``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from holosphere.errors import require_ok

TELEMETRY_SCHEMA = "holosphere.step_telemetry.v1"
TUNING_PROFILE_SCHEMA = "holosphere.tuning_profile.v1"

VALID_EXECUTION_BACKENDS = frozenset({"cuda_sm86", "triton", "torch_cpu"})
VALID_TUNING_BACKENDS = frozenset({"auto", "cuda_sm86", "triton", "torch_cpu"})
VALID_BLOCK_TS = frozenset({32, 64, 128, 256})
MAX_COOLDOWN_MS = 60000.0


def _require_int_not_bool(val: Any, name: str, code: str) -> int:
    require_ok(
        isinstance(val, int) and not isinstance(val, bool),
        f"{name} must be an int (not bool), got {type(val).__name__}",
        code=code,
    )
    return int(val)


def _require_finite_float(val: Any, name: str, code: str) -> float:
    require_ok(
        isinstance(val, (int, float)) and not isinstance(val, bool),
        f"{name} must be numeric, got {type(val).__name__}",
        code=code,
    )
    fval = float(val)
    require_ok(math.isfinite(fval), f"{name} must be finite, got {val!r}", code=code)
    return 0.0 if fval == 0.0 else fval


def _require_str(val: Any, name: str, code: str) -> str:
    require_ok(isinstance(val, str), f"{name} must be a str, got {type(val).__name__}", code=code)
    return val


@dataclass(frozen=True)
class HoloSphereStepTelemetry:
    """Read-only per-step summary emitted between steps to external governors (e.g. silisense)."""

    step_idx: int
    seq_len: int
    batch_size: int
    r_loops: int
    backend: str
    block_t: int
    step_ms: float
    tok_per_sec: float
    norm_drift_max: float
    schema: str = TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _require_str(self.schema, "schema", "TEL_SCHEMA"))
        object.__setattr__(self, "step_idx", _require_int_not_bool(self.step_idx, "step_idx", "TEL_STEP"))
        object.__setattr__(self, "seq_len", _require_int_not_bool(self.seq_len, "seq_len", "TEL_SEQ"))
        object.__setattr__(self, "batch_size", _require_int_not_bool(self.batch_size, "batch_size", "TEL_BATCH"))
        object.__setattr__(self, "r_loops", _require_int_not_bool(self.r_loops, "r_loops", "TEL_RLOOPS"))
        object.__setattr__(self, "backend", _require_str(self.backend, "backend", "TEL_BACKEND"))
        object.__setattr__(self, "block_t", _require_int_not_bool(self.block_t, "block_t", "TEL_BLOCKT"))
        object.__setattr__(self, "step_ms", _require_finite_float(self.step_ms, "step_ms", "TEL_STEP_MS"))
        object.__setattr__(self, "tok_per_sec", _require_finite_float(self.tok_per_sec, "tok_per_sec", "TEL_TPS"))
        object.__setattr__(self, "norm_drift_max", _require_finite_float(self.norm_drift_max, "norm_drift_max", "TEL_DRIFT"))
        self.validate()

    def validate(self) -> None:
        schema = _require_str(self.schema, "schema", "TEL_SCHEMA")
        require_ok(schema == TELEMETRY_SCHEMA, f"unexpected telemetry schema {schema!r}", code="TEL_SCHEMA")
        step_idx = _require_int_not_bool(self.step_idx, "step_idx", "TEL_STEP")
        seq_len = _require_int_not_bool(self.seq_len, "seq_len", "TEL_SEQ")
        batch_size = _require_int_not_bool(self.batch_size, "batch_size", "TEL_BATCH")
        r_loops = _require_int_not_bool(self.r_loops, "r_loops", "TEL_RLOOPS")
        backend = _require_str(self.backend, "backend", "TEL_BACKEND")
        block_t = _require_int_not_bool(self.block_t, "block_t", "TEL_BLOCKT")
        step_ms = _require_finite_float(self.step_ms, "step_ms", "TEL_STEP_MS")
        tok_per_sec = _require_finite_float(self.tok_per_sec, "tok_per_sec", "TEL_TPS")
        norm_drift = _require_finite_float(self.norm_drift_max, "norm_drift_max", "TEL_DRIFT")

        require_ok(step_idx >= 0, f"step_idx must be >= 0, got {step_idx}", code="TEL_STEP")
        require_ok(seq_len >= 1, f"seq_len must be >= 1, got {seq_len}", code="TEL_SEQ")
        require_ok(batch_size >= 1, f"batch_size must be >= 1, got {batch_size}", code="TEL_BATCH")
        require_ok(r_loops >= 1, f"r_loops must be >= 1, got {r_loops}", code="TEL_RLOOPS")
        require_ok(
            backend in VALID_EXECUTION_BACKENDS,
            f"backend must be one of {sorted(VALID_EXECUTION_BACKENDS)}, got {backend!r}",
            code="TEL_BACKEND",
        )
        require_ok(
            block_t in VALID_BLOCK_TS,
            f"block_t must be one of {sorted(VALID_BLOCK_TS)}, got {block_t}",
            code="TEL_BLOCKT",
        )
        require_ok(step_ms >= 0.0, f"step_ms must be >= 0.0, got {step_ms}", code="TEL_STEP_MS")
        require_ok(tok_per_sec >= 0.0, f"tok_per_sec must be >= 0.0, got {tok_per_sec}", code="TEL_TPS")
        require_ok(norm_drift >= 0.0, f"norm_drift_max must be >= 0.0, got {norm_drift}", code="TEL_DRIFT")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HoloSphereStepTelemetry:
        require_ok(isinstance(data, Mapping), "telemetry payload must be a mapping", code="TEL_PAYLOAD")
        allowed = {
            "schema",
            "step_idx",
            "seq_len",
            "batch_size",
            "r_loops",
            "backend",
            "block_t",
            "step_ms",
            "tok_per_sec",
            "norm_drift_max",
        }
        extra = set(data.keys()) - allowed
        require_ok(not extra, f"unexpected telemetry fields: {sorted(extra)}", code="TEL_FIELDS")
        required = allowed - {"schema"}
        missing = required - set(data.keys())
        require_ok(not missing, f"missing telemetry fields: {sorted(missing)}", code="TEL_FIELDS")
        obj = cls(
            step_idx=data["step_idx"],
            seq_len=data["seq_len"],
            batch_size=data["batch_size"],
            r_loops=data["r_loops"],
            backend=data["backend"],
            block_t=data["block_t"],
            step_ms=data["step_ms"],
            tok_per_sec=data["tok_per_sec"],
            norm_drift_max=data["norm_drift_max"],
            schema=data.get("schema", TELEMETRY_SCHEMA),
        )
        return obj


@dataclass(frozen=True)
class HoloSphereTuningProfile:
    """Semantics-preserving execution knobs accepted ONLY at step/epoch boundaries.

    Fail-closed contract:
      - Never mutates model weights or WeftAuditShell thresholds.
      - Numeric fields are canonicalized in ``__post_init__`` so ``0`` and ``0.0`` always
        produce identical ``profile_sha256()`` provenance digests.
    """

    backend: str = "auto"
    block_t: int = 128
    microbatch_size: int | None = None
    inter_step_cooldown_ms: float = 0.0
    r_loops_override: int | None = None
    schema: str = TUNING_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _require_str(self.schema, "schema", "TUNE_SCHEMA"))
        object.__setattr__(self, "backend", _require_str(self.backend, "backend", "TUNE_BACKEND"))
        object.__setattr__(self, "block_t", _require_int_not_bool(self.block_t, "block_t", "TUNE_BLOCKT"))
        if self.microbatch_size is not None:
            object.__setattr__(self, "microbatch_size", _require_int_not_bool(self.microbatch_size, "microbatch_size", "TUNE_MBS"))
        object.__setattr__(
            self,
            "inter_step_cooldown_ms",
            _require_finite_float(self.inter_step_cooldown_ms, "inter_step_cooldown_ms", "TUNE_COOLDOWN"),
        )
        if self.r_loops_override is not None:
            object.__setattr__(self, "r_loops_override", _require_int_not_bool(self.r_loops_override, "r_loops_override", "TUNE_RLOOPS"))
        self.validate()

    def validate(self) -> None:
        schema = _require_str(self.schema, "schema", "TUNE_SCHEMA")
        require_ok(
            schema == TUNING_PROFILE_SCHEMA,
            f"unexpected tuning profile schema {schema!r}",
            code="TUNE_SCHEMA",
        )
        backend = _require_str(self.backend, "backend", "TUNE_BACKEND")
        require_ok(
            backend in VALID_TUNING_BACKENDS,
            f"backend must be one of {sorted(VALID_TUNING_BACKENDS)}, got {backend!r}",
            code="TUNE_BACKEND",
        )
        block_t = _require_int_not_bool(self.block_t, "block_t", "TUNE_BLOCKT")
        require_ok(
            block_t in VALID_BLOCK_TS,
            f"block_t must be one of {sorted(VALID_BLOCK_TS)}, got {block_t}",
            code="TUNE_BLOCKT",
        )
        if self.microbatch_size is not None:
            mbs = _require_int_not_bool(self.microbatch_size, "microbatch_size", "TUNE_MBS")
            require_ok(mbs >= 1, f"microbatch_size must be >= 1, got {mbs}", code="TUNE_MBS")
        cooldown = _require_finite_float(self.inter_step_cooldown_ms, "inter_step_cooldown_ms", "TUNE_COOLDOWN")
        require_ok(
            0.0 <= cooldown <= MAX_COOLDOWN_MS,
            f"inter_step_cooldown_ms must lie in [0, {MAX_COOLDOWN_MS}], got {cooldown}",
            code="TUNE_COOLDOWN",
        )
        if self.r_loops_override is not None:
            rl = _require_int_not_bool(self.r_loops_override, "r_loops_override", "TUNE_RLOOPS")
            require_ok(rl >= 1, f"r_loops_override must be >= 1, got {rl}", code="TUNE_RLOOPS")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def profile_sha256(self) -> str:
        """Canonical SHA-256 digest for stamping execution profiles into CS-08 provenance receipts."""
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HoloSphereTuningProfile:
        require_ok(isinstance(data, Mapping), "tuning profile payload must be a mapping", code="TUNE_PAYLOAD")
        allowed = {
            "schema",
            "backend",
            "block_t",
            "microbatch_size",
            "inter_step_cooldown_ms",
            "r_loops_override",
        }
        extra = set(data.keys()) - allowed
        require_ok(not extra, f"unexpected tuning profile fields: {sorted(extra)}", code="TUNE_FIELDS")
        obj = cls(
            backend=data.get("backend", "auto"),
            block_t=data.get("block_t", 128),
            microbatch_size=data.get("microbatch_size", None),
            inter_step_cooldown_ms=data.get("inter_step_cooldown_ms", 0.0),
            r_loops_override=data.get("r_loops_override", None),
            schema=data.get("schema", TUNING_PROFILE_SCHEMA),
        )
        return obj


def build_step_telemetry(
    *,
    step_idx: int,
    seq_len: int,
    batch_size: int,
    r_loops: int,
    backend: str,
    block_t: int,
    step_ms: float,
    norm_drift_max: float = 0.0,
) -> HoloSphereStepTelemetry:
    """Build and validate a ``HoloSphereStepTelemetry`` record with exact tokens/sec computation."""
    ms = _require_finite_float(step_ms, "step_ms", "TEL_STEP_MS")
    require_ok(ms >= 0.0, f"step_ms must be >= 0.0, got {ms}", code="TEL_STEP_MS")
    bsz = _require_int_not_bool(batch_size, "batch_size", "TEL_BATCH")
    slen = _require_int_not_bool(seq_len, "seq_len", "TEL_SEQ")
    tokens = float(bsz * slen)
    tps = (tokens * 1000.0 / ms) if ms > 0.0 else 0.0
    rec = HoloSphereStepTelemetry(
        step_idx=step_idx,
        seq_len=slen,
        batch_size=bsz,
        r_loops=r_loops,
        backend=backend,
        block_t=block_t,
        step_ms=round(ms, 6),
        tok_per_sec=round(tps, 3),
        norm_drift_max=round(_require_finite_float(norm_drift_max, "norm_drift_max", "TEL_DRIFT"), 9),
    )
    rec.validate()
    return rec


__all__ = [
    "HoloSphereStepTelemetry",
    "HoloSphereTuningProfile",
    "MAX_COOLDOWN_MS",
    "TELEMETRY_SCHEMA",
    "TUNING_PROFILE_SCHEMA",
    "VALID_BLOCK_TS",
    "VALID_EXECUTION_BACKENDS",
    "VALID_TUNING_BACKENDS",
    "build_step_telemetry",
]
