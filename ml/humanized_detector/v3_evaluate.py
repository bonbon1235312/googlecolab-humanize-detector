"""Metrics and immutable-manifest utilities for V3 evaluation."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(error)


def evaluate_binary(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float | int | None]:
    """Return pre-registered thresholded, ranking, and calibration metrics."""
    actual = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or len(actual) == 0 or len(actual) != len(scores):
        raise ValueError("labels and probabilities must be non-empty one-dimensional arrays of equal length")
    predicted = (scores >= threshold).astype(int)
    negatives = actual == 0
    positives = actual == 1
    human_fpr = float((predicted[negatives] == 1).mean()) if negatives.any() else None
    if negatives.any() and positives.any():
        fpr, tpr, _ = roc_curve(actual, scores)
        tpr_at_1pct_fpr: float | None = float(tpr[fpr <= 0.01].max())
        roc_auc: float | None = float(roc_auc_score(actual, scores))
        pr_auc: float | None = float(average_precision_score(actual, scores))
    else:
        tpr_at_1pct_fpr = roc_auc = pr_auc = None
    return {"n": int(len(actual)), "accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)), "f1": float(f1_score(actual, predicted, zero_division=0)), "roc_auc": roc_auc, "pr_auc": pr_auc, "human_fpr": human_fpr, "tpr_at_1pct_fpr": tpr_at_1pct_fpr, "brier_score": float(brier_score_loss(actual, scores)), "expected_calibration_error": _expected_calibration_error(actual, scores)}


def metrics_by_field(rows: Sequence[dict[str, object]], probabilities: Sequence[float], field: str) -> dict[str, dict[str, float | int | None]]:
    """Evaluate every source, provenance, or external scenario independently."""
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length")
    grouped: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for row, probability in zip(rows, probabilities, strict=True):
        labels, scores = grouped[str(row[field])]
        labels.append(int(row["label"]))
        scores.append(float(probability))
    return {name: evaluate_binary(labels, scores) for name, (labels, scores) in grouped.items()}


def freeze_external_benchmark(benchmark_dir: Path, manifest_path: Path, dataset: str, revision: str, split: str) -> dict[str, object]:
    """Hash a downloaded benchmark without parsing its text or labels."""
    if not benchmark_dir.is_dir():
        raise ValueError(f"benchmark directory does not exist: {benchmark_dir}")
    files = []
    for path in sorted(candidate for candidate in benchmark_dir.rglob("*") if candidate.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": path.relative_to(benchmark_dir).as_posix(), "sha256": digest, "bytes": path.stat().st_size})
    manifest = {"dataset": dataset, "revision": revision, "split": split, "files": files, "frozen_without_label_parsing": True}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
