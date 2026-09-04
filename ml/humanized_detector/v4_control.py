"""Shared safe inputs and capacity presets for V4 control-model runs."""

from __future__ import annotations

from pathlib import Path

from .model import ModelConfig
from .v4_prepare import load_v4_partition


CONTROL_ROLES = ("train", "development", "calibration")


def load_v4_control_partitions(data_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Load every non-sealed V4 role and never touch the final partition."""
    return {role: load_v4_partition(data_dir, role) for role in CONTROL_ROLES}


def model_config_for_capacity(vocab_size: int, capacity: str) -> ModelConfig:
    """Return the legacy control configuration or a roughly 12M-parameter ablation."""
    if capacity == "5m":
        return ModelConfig(vocab_size=vocab_size, hidden_size=192, heads=6, layers=4)
    if capacity == "12m":
        return ModelConfig(vocab_size=vocab_size, hidden_size=384, heads=8, layers=6)
    raise ValueError("capacity must be '5m' or '12m'")
