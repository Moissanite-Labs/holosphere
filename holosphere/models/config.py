"""Configuration schema & verified architecture presets across S106-S114."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from holosphere.errors import HoloSphereError, require_ok

VALID_WIDTH_PROFILES = frozenset({"cylinder", "polar", "equatorial", "permuted_pol", "permuted_eq"})
VALID_POS_MODES = frozenset({"su2_holonomy", "rope2d", "liere4d", "block4_rand_ortho"})
VALID_RESIDUAL_MODES = frozenset({"hopf_s3", "slerp", "flat"})
VALID_LOOP_MODES = frozenset({"none", "safe_hopf_acute", "clifford_coprime", "sphere_noclock", "sphere_randrot", "flat"})


@dataclass(frozen=True)
class HoloSphereConfig:
    """Unified configuration for HoloSphereTransformer and all S106-S114 control arms."""

    arm_name: str = "s114_entity"
    vocab_size: int = 260
    d_model: int = 320
    n_heads: int = 8
    n_layers: int = 12
    base_d_ff: int = 864
    width_profile: str = "cylinder"
    pos_mode: str = "su2_holonomy"
    transport_v: bool = True
    angle_scale: float = 0.75
    residual_mode: str = "hopf_s3"
    ngpt_qk_norm: bool = True
    # Looped core options (H7c + W10-OBS-31 acute-cone Safe-Hopf fix)
    loop_mode: str = "none"
    n_pre: int = 2
    n_core: int = 4
    n_suffix: int = 2
    r_train: int = 3
    acute_phi_max: float = 0.42 * math.pi
    acute_tau: float = 2.0
    # Optional S109/S110 Geosphere operators
    use_cayley: bool = False
    cayley_rank: int = 16
    use_bingham: bool = False
    bingham_yield: float = 0.5
    bingham_tau: float = 0.0
    use_darcy: bool = False
    # Optional S108 H8c2 Markov-MTP8 draft head
    use_markov_mtp: bool = False
    mtp_block_size: int = 8
    mtp_rank: int = 64
    # Exact CS-08 d_ff offset for W_omega when comparing iso-param against RoPE
    compensate_su2_dff: bool = True
    use_grad_checkpoint: bool = False
    backend: str = "auto"
    seed: int = 42

    def validate(self) -> None:
        require_ok(self.vocab_size >= 4, f"vocab_size must be >= 4, got {self.vocab_size}", code="CFG_VOCAB")
        require_ok(self.d_model >= 8 and self.d_model % 4 == 0, f"d_model must be >= 8 and % 4 == 0, got {self.d_model}", code="CFG_DMODEL")
        require_ok(self.n_heads >= 1 and self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads", code="CFG_HEADS")
        d_head = self.d_model // self.n_heads
        if self.pos_mode in ("su2_holonomy", "liere4d", "block4_rand_ortho"):
            require_ok(d_head % 4 == 0, f"pos_mode={self.pos_mode!r} requires d_head % 4 == 0, got {d_head}", code="CFG_DHEAD")
        else:
            require_ok(d_head % 2 == 0, f"rope2d requires even d_head, got {d_head}", code="CFG_DHEAD")
        require_ok(self.n_layers >= 1, f"n_layers must be >= 1, got {self.n_layers}", code="CFG_LAYERS")
        require_ok(self.base_d_ff >= 16 and self.base_d_ff % 16 == 0, "base_d_ff must be >= 16 and multiple of 16", code="CFG_DFF")
        require_ok(self.width_profile in VALID_WIDTH_PROFILES, f"unknown width_profile {self.width_profile!r}", code="CFG_PROFILE")
        require_ok(self.pos_mode in VALID_POS_MODES, f"unknown pos_mode {self.pos_mode!r}", code="CFG_POS")
        require_ok(self.residual_mode in VALID_RESIDUAL_MODES, f"unknown residual_mode {self.residual_mode!r}", code="CFG_RESID")
        require_ok(self.loop_mode in VALID_LOOP_MODES, f"unknown loop_mode {self.loop_mode!r}", code="CFG_LOOP")
        if self.loop_mode != "none":
            require_ok(self.n_pre >= 0 and self.n_core >= 1 and self.n_suffix >= 0, "loop requires n_core >= 1", code="CFG_LOOP_LAYERS")
            require_ok(self.r_train >= 1, "r_train must be >= 1", code="CFG_RTRAIN")
        require_ok(0.0 < self.acute_phi_max < 0.5 * math.pi, "acute_phi_max must lie in (0, pi/2)", code="CFG_PHI")


def build_preset_config(preset_name: str, **overrides: Any) -> HoloSphereConfig:
    """Return a validated named preset spanning Tracks S106-S114 plus matched controls."""
    presets: dict[str, HoloSphereConfig] = {
        # Track S114 unified kernel-isomorphic flagship entities
        "s114_entity": HoloSphereConfig(
            arm_name="s114_entity",
            d_model=256,
            n_heads=8,
            n_layers=8,
            base_d_ff=688,
            pos_mode="su2_holonomy",
            transport_v=True,
            residual_mode="hopf_s3",
            ngpt_qk_norm=True,
            loop_mode="safe_hopf_acute",
            n_pre=2,
            n_core=4,
            n_suffix=2,
            r_train=3,
            use_markov_mtp=True,
        ),
        "s114_unlooped_covariant": HoloSphereConfig(
            arm_name="s114_unlooped_covariant",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="su2_holonomy",
            transport_v=True,
            residual_mode="hopf_s3",
            ngpt_qk_norm=True,
            loop_mode="none",
        ),
        # Track S106 H6c confirmed Native Hopf-S^3 residual micro-LM
        "s106_h6c_hopf": HoloSphereConfig(
            arm_name="s106_h6c_hopf",
            vocab_size=256,
            d_model=384,
            n_heads=6,
            n_layers=6,
            base_d_ff=1024,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="hopf_s3",
            ngpt_qk_norm=False,
            compensate_su2_dff=False,
        ),
        # Track S107 H7a / H7b / H7c presets
        "s107_h7a_equatorial_ngpt": HoloSphereConfig(
            arm_name="s107_h7a_equatorial_ngpt",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            width_profile="equatorial",
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
        ),
        "s107_h7b_su2_qk": HoloSphereConfig(
            arm_name="s107_h7b_su2_qk",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="su2_holonomy",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
        ),
        "s107_h7c_safe_hopf_loop": HoloSphereConfig(
            arm_name="s107_h7c_safe_hopf_loop",
            d_model=256,
            n_heads=8,
            n_layers=8,
            base_d_ff=688,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
            loop_mode="safe_hopf_acute",
            n_pre=2,
            n_core=4,
            n_suffix=2,
            r_train=3,
        ),
        # Track S108 H8c2 Markov-MTP8 preset
        "s108_h8c_markov_mtp8": HoloSphereConfig(
            arm_name="s108_h8c_markov_mtp8",
            d_model=256,
            n_heads=8,
            n_layers=6,
            base_d_ff=512,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
            use_markov_mtp=True,
        ),
        # Track S109 TerraCore Poloidal Bingham preset
        "s109_terracore_bingham": HoloSphereConfig(
            arm_name="s109_terracore_bingham",
            d_model=256,
            n_heads=8,
            n_layers=6,
            base_d_ff=688,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
            use_bingham=True,
            bingham_yield=0.25,
        ),
        # Track S110 Geosphere-Net Beltrami Cayley + Bingham + Darcy preset
        "s110_geosphere_beltrami": HoloSphereConfig(
            arm_name="s110_geosphere_beltrami",
            d_model=256,
            n_heads=8,
            n_layers=8,
            base_d_ff=688,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
            loop_mode="safe_hopf_acute",
            use_cayley=True,
            use_bingham=True,
            use_darcy=True,
            use_markov_mtp=True,
        ),
        # Matched baselines & adversarial nulls
        "rope2d_std": HoloSphereConfig(
            arm_name="rope2d_std",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="flat",
            ngpt_qk_norm=False,
        ),
        "rope2d_ngpt": HoloSphereConfig(
            arm_name="rope2d_ngpt",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="rope2d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
        ),
        "liere4d_ngpt": HoloSphereConfig(
            arm_name="liere4d_ngpt",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="liere4d",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
        ),
        "block4_rand_ortho_ngpt": HoloSphereConfig(
            arm_name="block4_rand_ortho_ngpt",
            d_model=320,
            n_heads=8,
            n_layers=12,
            base_d_ff=864,
            pos_mode="block4_rand_ortho",
            transport_v=False,
            residual_mode="slerp",
            ngpt_qk_norm=True,
        ),
    }
    if preset_name not in presets:
        raise HoloSphereError("UNKNOWN_PRESET", f"unknown preset {preset_name!r}; available={sorted(presets)}")
    cfg = replace(presets[preset_name], **overrides)
    cfg.validate()
    return cfg
