"""Sealed-safe V4 Transformer control training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import ModelConfig
from .v3_features import FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _predict, train_v3_model
from .v4_control import load_v4_control_partitions, model_config_for_capacity


_VARIANT = "fusion_concat"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _normalizer_from_artifact(path: Path) -> FeatureNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FeatureNormalizer(mean=np.asarray(payload["mean"]), scale=np.asarray(payload["scale"]))


def _write_calibration_predictions(artifact_dir: Path, rows: list[dict[str, object]], batch_size: int) -> Path:
    payload = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = _create_model(config, _VARIANT)
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = V3EncodedDataset(rows, artifact_dir / "tokenizer", config.max_tokens, _normalizer_from_artifact(artifact_dir / "feature_normalizer.json"))
    labels, probabilities = _predict(model, _VARIANT, DataLoader(dataset, batch_size=batch_size), device)
    path = artifact_dir / "calibration_predictions.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, label, probability in zip(rows, labels, probabilities, strict=True):
            handle.write(json.dumps({"id": str(row["id"]), "label": int(label), "probability": float(probability)}, separators=(",", ":")) + "\n")
    return path


def train_v4_transformer(
    data_dir: Path,
    artifact_dir: Path,
    capacity: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    token_pooling: str = "first",
    label_smoothing: float = 0.1,
    warmup_steps: int = 0,
    grad_clip_norm: float | None = None,
) -> dict[str, object]:
    """Train a V4 fusion-concat control model without loading sealed data."""
    config = replace(model_config_for_capacity(4_000, capacity), token_pooling=token_pooling)
    partitions = load_v4_control_partitions(data_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="v4-control-") as temporary:
        temporary_dir = Path(temporary)
        train_file = temporary_dir / "train.jsonl"
        development_file = temporary_dir / "development.jsonl"
        _write_jsonl(train_file, partitions["train"])
        _write_jsonl(development_file, partitions["development"])
        result = train_v3_model(
            train_file,
            development_file,
            artifact_dir,
            config,
            _VARIANT,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            warmup_steps=warmup_steps,
            grad_clip_norm=grad_clip_norm,
        )
    (artifact_dir / "tokenizer" / "training_corpus.txt").unlink(missing_ok=True)
    calibration_predictions = _write_calibration_predictions(artifact_dir, partitions["calibration"], batch_size)
    return {
        "capacity": capacity,
        "variant": _VARIANT,
        "checkpoint": str(result.checkpoint),
        "development_metrics": str(result.metrics_path),
        "calibration_predictions": str(calibration_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--capacity", choices=("5m", "12m"), default="5m")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--token-pooling", choices=("first", "masked_mean"), default="first")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float)
    args = parser.parse_args()
    result = train_v4_transformer(
        args.data_dir, args.artifacts_dir, args.capacity, args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.token_pooling, args.label_smoothing, args.warmup_steps, args.grad_clip_norm,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
