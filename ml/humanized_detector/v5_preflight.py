"""Fail-fast validation for an H1 curriculum continuation run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .model import ModelConfig
from .v3_train import _create_model
from .v4_prepare import load_v4_partition
from .v5_sampling import curriculum_mix, hierarchical_weights


_ROLES = ("train", "development", "calibration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_h1_inputs(data_dir: Path, source_artifacts: Path, output_dir: Path, allow_existing: bool = False) -> dict[str, object]:
    """Validate data roles and a source checkpoint without reading sealed data."""
    for role in _ROLES:
        path = data_dir / f"{role}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"required H1 input is missing: {path}")
    partitions = {role: load_v4_partition(data_dir, role) for role in _ROLES}
    lineage_roles: dict[str, str] = {}
    for role, rows in partitions.items():
        for row in rows:
            lineage = str(row["lineage_id"])
            previous = lineage_roles.setdefault(lineage, role)
            if previous != role:
                raise ValueError(f"lineage overlap across {previous} and {role}: {lineage}")
    # Resolve the first curriculum weights during preflight so a missing or
    # unsupported training stratum fails before tokenization or GPU setup.
    hierarchical_weights(partitions["train"], curriculum_mix(1))
    development_provenance = {
        str(row.get("provenance")) for row in partitions["development"] if int(row["label"]) == 1
    }
    required_development = {"expert_edited_ai", "llm_edited_ai", "raw_ai"}
    missing_development = required_development - development_provenance
    if missing_development:
        raise ValueError(f"missing development subtype: {', '.join(sorted(missing_development))}")
    for role in ("development", "calibration"):
        labels = {int(row["label"]) for row in partitions[role]}
        if labels != {0, 1}:
            raise ValueError(f"{role} partition must contain both binary classes")

    required_artifacts = (
        source_artifacts / "model.pt",
        source_artifacts / "tokenizer" / "vocab.json",
        source_artifacts / "tokenizer" / "merges.txt",
        source_artifacts / "feature_normalizer.json",
    )
    missing = [path for path in required_artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required H1 source artifact is missing: {missing[0]}")
    if not allow_existing and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"H1 output directory is not empty: {output_dir}")

    payload = torch.load(source_artifacts / "model.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    variant = str(payload["variant"])
    if variant != "fusion_concat":
        raise ValueError(f"H1 requires fusion_concat source checkpoint, got {variant}")
    if config.token_pooling != "masked_mean":
        raise ValueError(f"H1 requires masked_mean token pooling, got {config.token_pooling}")
    model = _create_model(config, variant)
    model.load_state_dict(payload["state_dict"])
    return {
        "partition_counts": {role: len(partitions[role]) for role in sorted(_ROLES)},
        "partition_sha256": {role: _sha256(data_dir / f"{role}.jsonl") for role in sorted(_ROLES)},
        "source_checkpoint_sha256": _sha256(source_artifacts / "model.pt"),
        "model": {
            "variant": variant,
            "token_pooling": config.token_pooling,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "config": config.__dict__,
        },
    }
