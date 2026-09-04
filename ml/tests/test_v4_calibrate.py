import json
from pathlib import Path

from humanized_detector.v4_calibrate import calibrate_v4_artifact, crossfit_platt_calibration


def test_calibration_freezes_fixed_fpr_thresholds_without_reading_sealed_data(tmp_path: Path) -> None:
    calibration_rows = [
        {"id": "h1", "lineage_id": "prompt-a", "text": "human", "label": 0, "source": "source-a"},
        {"id": "h2", "lineage_id": "prompt-b", "text": "human", "label": 0, "source": "source-b"},
        {"id": "a1", "lineage_id": "prompt-a", "text": "ai", "label": 1, "source": "source-a"},
        {"id": "a2", "lineage_id": "prompt-b", "text": "ai", "label": 1, "source": "source-b"},
    ]
    (tmp_path / "calibration.jsonl").write_text("\n".join(json.dumps(row) for row in calibration_rows) + "\n", encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text("this must never be read\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    predictions = [
        {"id": "h1", "label": 0, "probability": 0.80},
        {"id": "h2", "label": 0, "probability": 0.20},
        {"id": "a1", "label": 1, "probability": 0.95},
        {"id": "a2", "label": 1, "probability": 0.75},
    ]
    (artifacts / "calibration_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8")

    report = calibrate_v4_artifact(tmp_path, artifacts)

    assert set(report["operating_points"]) == {"1pct", "2pct", "5pct"}
    assert report["operating_points"]["5pct"]["human_fpr"] <= 0.05
    assert report["ranking_metrics"]["roc_auc"] is not None
    assert report["crossfit_platt"]["folds"] == 2
    assert (artifacts / "calibration.json").exists()


def test_crossfit_platt_keeps_lineages_together_and_reports_out_of_fold_metrics() -> None:
    rows = [
        {"id": "h1", "lineage_id": "p1", "label": 0},
        {"id": "a1", "lineage_id": "p1", "label": 1},
        {"id": "h2", "lineage_id": "p2", "label": 0},
        {"id": "a2", "lineage_id": "p2", "label": 1},
        {"id": "h3", "lineage_id": "p3", "label": 0},
        {"id": "a3", "lineage_id": "p3", "label": 1},
    ]

    report = crossfit_platt_calibration(rows, [0.10, 0.85, 0.20, 0.80, 0.30, 0.75], folds=3)

    assert report["folds"] == 3
    assert report["metrics"]["n"] == 6
    assert report["metrics"]["brier_score"] is not None
