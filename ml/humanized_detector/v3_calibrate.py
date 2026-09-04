"""Deployment-threshold calibration for a selected V3 checkpoint."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import ModelConfig
from .v3_evaluate import evaluate_binary
from .v3_features import FEATURE_NAMES, FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _predict, load_v3_jsonl


def select_operating_threshold(
    labels: Sequence[int], probabilities: Sequence[float], max_human_fpr: float = 0.05
) -> float:
    """Choose the lowest threshold with the highest recall under a human-FPR ceiling."""
    actual = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) == 0 or len(actual) != len(scores):
        raise ValueError("labels and probabilities must be non-empty one-dimensional arrays of equal length")
    if not 0.0 <= max_human_fpr <= 1.0:
        raise ValueError("max_human_fpr must be between 0 and 1")
    if not (actual == 0).any() or not (actual == 1).any():
        raise ValueError("calibration requires both human and AI-labelled records")

    best: tuple[float, float, float] | None = None
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        human_fpr = float(predicted[actual == 0].mean())
        recall = float(predicted[actual == 1].mean())
        candidate = (recall, -human_fpr, -float(threshold))
        if human_fpr <= max_human_fpr and (best is None or candidate > best):
            best = candidate
    if best is None:
        raise RuntimeError("no threshold met the requested human-FPR ceiling")
    return -best[2]


def calibration_record(
    labels: Sequence[int], probabilities: Sequence[float], max_human_fpr: float = 0.05
) -> dict[str, object]:
    """Summarise an operating point selected solely on the calibration partition."""
    threshold = select_operating_threshold(labels, probabilities, max_human_fpr)
    return {
        "max_human_fpr": max_human_fpr,
        "threshold": threshold,
        "metrics": evaluate_binary(labels, probabilities, threshold),
    }


def calibrate_checkpoint(
    calibration_file: Path, artifact_dir: Path, max_human_fpr: float = 0.05
) -> dict[str, object]:
    """Score the isolated calibration split and persist a deployable operating threshold."""
    checkpoint_path = artifact_dir / "model.pt"
    normalizer_path = artifact_dir / "feature_normalizer.json"
    tokenizer_dir = artifact_dir / "tokenizer"
    for path in (checkpoint_path, normalizer_path, tokenizer_dir):
        if not path.exists():
            raise FileNotFoundError(f"required selected-model artifact is missing: {path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    variant = str(payload["variant"])
    normalizer_data = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if normalizer_data["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("feature normalizer does not match the current V3 feature schema")
    normalizer = FeatureNormalizer(
        mean=np.asarray(normalizer_data["mean"], dtype=np.float32),
        scale=np.asarray(normalizer_data["scale"], dtype=np.float32),
    )
    rows = load_v3_jsonl(calibration_file)
    dataset = V3EncodedDataset(rows, tokenizer_dir, config.max_tokens, normalizer)
    loader = DataLoader(dataset, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(config, variant).to(device)
    model.load_state_dict(payload["state_dict"])
    labels, probabilities = _predict(model, variant, loader, device)
    record = calibration_record(labels, probabilities, max_human_fpr)
    record.update({
        "partition": calibration_file.name,
        "variant": variant,
        "checkpoint": checkpoint_path.name,
    })
    (artifact_dir / "calibration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--max-human-fpr", type=float, default=0.05)
    args = parser.parse_args()
    record = calibrate_checkpoint(args.data_dir / "calibration.jsonl", args.artifacts_dir, args.max_human_fpr)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
