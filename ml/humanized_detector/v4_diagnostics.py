"""Read-only fit diagnostics for V4.8 experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .model import ModelConfig
from .v3_evaluate import evaluate_binary
from .v3_features import FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _predict
from .v4_prepare import load_v4_partition


def bootstrap_roc_auc(
    labels: Sequence[int], probabilities: Sequence[float], iterations: int = 1000, seed: int = 20260904
) -> dict[str, float]:
    """Return a deterministic stratified bootstrap ROC-AUC interval."""
    actual = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or len(actual) != len(scores) or len(actual) == 0:
        raise ValueError("labels and probabilities must be non-empty one-dimensional arrays of equal length")
    negative = np.flatnonzero(actual == 0)
    positive = np.flatnonzero(actual == 1)
    if not len(negative) or not len(positive):
        raise ValueError("bootstrap ROC-AUC requires both binary classes")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indexes = np.concatenate((generator.choice(negative, size=len(negative), replace=True), generator.choice(positive, size=len(positive), replace=True)))
        values.append(float(roc_auc_score(actual[indexes], scores[indexes])))
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "point": float(roc_auc_score(actual, scores)),
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
    }


def development_subtype_metrics(
    rows: Sequence[dict[str, object]], probabilities: Sequence[float]
) -> dict[str, dict[str, object]]:
    """Compare all human rows with one AI provenance subtype at a time."""
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length")
    human = [(row, float(probability)) for row, probability in zip(rows, probabilities, strict=True) if int(row["label"]) == 0]
    by_provenance: dict[str, list[tuple[dict[str, object], float]]] = {}
    for row, probability in zip(rows, probabilities, strict=True):
        if int(row["label"]) == 1:
            by_provenance.setdefault(str(row["provenance"]), []).append((row, float(probability)))
    reports: dict[str, dict[str, object]] = {}
    for provenance, positive in sorted(by_provenance.items()):
        subset = human + positive
        labels = [int(row["label"]) for row, _ in subset]
        scores = [probability for _, probability in subset]
        reports[provenance] = evaluate_binary(labels, scores)
    return reports


def _normalizer(path: Path) -> FeatureNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FeatureNormalizer(mean=np.asarray(payload["mean"], dtype=np.float32), scale=np.asarray(payload["scale"], dtype=np.float32))


def _predict_checkpoint(rows: list[dict[str, object]], artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    variant = str(payload["variant"])
    model = _create_model(config, variant)
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = V3EncodedDataset(rows, artifact_dir / "tokenizer", config.max_tokens, _normalizer(artifact_dir / "feature_normalizer.json"))
    return _predict(model, variant, DataLoader(dataset, batch_size=64), device)


def write_v4_fit_diagnostics(data_dir: Path, artifact_dir: Path, bootstrap_iterations: int = 1000) -> dict[str, object]:
    """Write train/development fit diagnostics without loading calibration or sealed data."""
    train_rows = load_v4_partition(data_dir, "train")
    development_rows = load_v4_partition(data_dir, "development")
    train_labels, train_probabilities = _predict_checkpoint(train_rows, artifact_dir)
    development_labels, development_probabilities = _predict_checkpoint(development_rows, artifact_dir)
    train_metrics = evaluate_binary(train_labels, train_probabilities)
    development_metrics = evaluate_binary(development_labels, development_probabilities)
    negative_count = int((development_labels == 0).sum())
    report: dict[str, object] = {
        "train": train_metrics,
        "development": {
            **development_metrics,
            "roc_auc_confidence_interval": bootstrap_roc_auc(development_labels, development_probabilities, bootstrap_iterations),
            "subtypes": development_subtype_metrics(development_rows, development_probabilities),
            "human_negative_count": negative_count,
            "human_fpr_resolution": None if not negative_count else 1.0 / negative_count,
        },
        "train_development_roc_auc_gap": None if train_metrics["roc_auc"] is None or development_metrics["roc_auc"] is None else float(train_metrics["roc_auc"]) - float(development_metrics["roc_auc"]),
    }
    (artifact_dir / "fit_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score_padben_diagnostic(diagnostic_dir: Path, artifact_dir: Path) -> dict[str, object]:
    """Score the explicitly unused PADBen diagnostic without touching control calibration or final data."""
    path = diagnostic_dir / "padben_diagnostic.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    labels, probabilities = _predict_checkpoint(rows, artifact_dir)
    report = evaluate_binary(labels, probabilities)
    report["cohort"] = "padben_unused_diagnostic"
    (artifact_dir / "padben_diagnostic_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
