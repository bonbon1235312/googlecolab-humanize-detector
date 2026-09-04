"""Shared fixed-human-FPR calibration for V4 model artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .v3_calibrate import calibration_record
from .v3_evaluate import evaluate_binary
from .v4_prepare import load_v4_partition


_TARGETS = (("1pct", 0.01), ("2pct", 0.02), ("5pct", 0.05))


def _load_predictions(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_predictions(rows: Sequence[dict[str, object]], predictions: Sequence[dict[str, object]]) -> np.ndarray:
    expected = {str(row["id"]): int(row["label"]) for row in rows}
    actual = {str(row["id"]): int(row["label"]) for row in predictions}
    if set(expected) != set(actual):
        raise ValueError("calibration prediction IDs do not match the calibration partition")
    if expected != actual:
        raise ValueError("calibration prediction labels do not match the calibration partition")
    probabilities = {str(row["id"]): float(row["probability"]) for row in predictions}
    return np.asarray([probabilities[str(row["id"])] for row in rows], dtype=float)


def _source_human_fpr(rows: Sequence[dict[str, object]], probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        if int(row["label"]) == 0:
            groups[str(row.get("source", "unknown"))].append(float(probability))
    return {source: float((np.asarray(scores) >= threshold).mean()) for source, scores in sorted(groups.items()) if scores}


def calibrate_v4_artifact(data_dir: Path, artifact_dir: Path) -> dict[str, object]:
    """Freeze 1%, 2%, and 5% human-FPR thresholds from calibration only."""
    rows = load_v4_partition(data_dir, "calibration")
    prediction_file = artifact_dir / "calibration_predictions.jsonl"
    predictions = _load_predictions(prediction_file)
    probabilities = _validate_predictions(rows, predictions)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    raw_metrics = evaluate_binary(labels, probabilities)
    ranking_metrics = {key: raw_metrics[key] for key in ("n", "roc_auc", "pr_auc", "brier_score", "expected_calibration_error")}
    operating_points: dict[str, dict[str, object]] = {}
    for name, maximum_fpr in _TARGETS:
        record = calibration_record(labels, probabilities, maximum_fpr)
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        source_fpr = _source_human_fpr(rows, probabilities, float(record["threshold"]))
        operating_points[name] = {
            "max_human_fpr": maximum_fpr,
            "threshold": record["threshold"],
            "tpr": metrics["recall"],
            "human_fpr": metrics["human_fpr"],
            "source_human_fpr": source_fpr,
            "worst_source_human_fpr": max(source_fpr.values()) if source_fpr else None,
        }
    model_candidates = (artifact_dir / "model.pt", artifact_dir / "model.joblib")
    model_path = next((path for path in model_candidates if path.exists()), None)
    report: dict[str, object] = {
        "prediction_file": prediction_file.name,
        "ranking_metrics": ranking_metrics,
        "operating_points": operating_points,
        "model_artifact": None if model_path is None else model_path.name,
        "model_size_bytes": None if model_path is None else model_path.stat().st_size,
    }
    (artifact_dir / "calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate_v4_artifact(args.data_dir, args.artifacts_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
