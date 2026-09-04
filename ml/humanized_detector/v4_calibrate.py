"""Shared fixed-human-FPR calibration for V4 model artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

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


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _fit_platt(labels: np.ndarray, probabilities: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=20260904)
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return model


def crossfit_platt_calibration(
    rows: Sequence[dict[str, object]], probabilities: Sequence[float], folds: int = 5
) -> dict[str, object]:
    """Assess Platt calibration out-of-fold with Beemo prompt lineages kept atomic."""
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("rows and probabilities must be non-empty and have the same length")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    groups = np.asarray([str(row["lineage_id"]) for row in rows], dtype=object)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("cross-fitted Platt calibration requires at least two lineages")
    actual_folds = min(folds, len(unique_groups))
    if actual_folds < 2:
        raise ValueError("folds must be at least two")
    raw = np.asarray(probabilities, dtype=float)
    calibrated = np.empty_like(raw)
    splitter = GroupKFold(n_splits=actual_folds)
    for train_indexes, validation_indexes in splitter.split(raw, labels, groups):
        train_labels = labels[train_indexes]
        if len(np.unique(train_labels)) != 2:
            raise ValueError("each cross-fit training fold must contain both binary classes")
        model = _fit_platt(train_labels, raw[train_indexes])
        calibrated[validation_indexes] = model.predict_proba(_logit(raw[validation_indexes]).reshape(-1, 1))[:, 1]
    metrics = evaluate_binary(labels, calibrated)
    return {
        "folds": actual_folds,
        "group_field": "lineage_id",
        "metrics": {key: metrics[key] for key in ("n", "roc_auc", "pr_auc", "brier_score", "expected_calibration_error")},
    }


def calibrate_v4_artifact(data_dir: Path, artifact_dir: Path) -> dict[str, object]:
    """Freeze 1%, 2%, and 5% human-FPR thresholds from calibration only."""
    rows = load_v4_partition(data_dir, "calibration")
    prediction_file = artifact_dir / "calibration_predictions.jsonl"
    predictions = _load_predictions(prediction_file)
    probabilities = _validate_predictions(rows, predictions)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    raw_metrics = evaluate_binary(labels, probabilities)
    ranking_metrics = {key: raw_metrics[key] for key in ("n", "roc_auc", "pr_auc", "brier_score", "expected_calibration_error")}
    crossfit_platt = crossfit_platt_calibration(rows, probabilities)
    final_platt = _fit_platt(labels, probabilities)
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
        "crossfit_platt": crossfit_platt,
        "full_platt_scaler": {
            "input": "logit(raw_probability)",
            "coefficient": float(final_platt.coef_[0, 0]),
            "intercept": float(final_platt.intercept_[0]),
            "fit_partition": "calibration.jsonl",
            "note": "Cross-fitted calibration metrics are for diagnostics; fit this final scaler only after model selection is frozen.",
        },
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
