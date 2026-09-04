import json
from pathlib import Path

from humanized_detector.v4_calibrate import calibrate_v4_artifact


def test_calibration_freezes_fixed_fpr_thresholds_without_reading_sealed_data(tmp_path: Path) -> None:
    calibration_rows = [
        {"id": "h1", "text": "human", "label": 0, "source": "source-a"},
        {"id": "h2", "text": "human", "label": 0, "source": "source-b"},
        {"id": "a1", "text": "ai", "label": 1, "source": "source-a"},
        {"id": "a2", "text": "ai", "label": 1, "source": "source-b"},
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
    assert (artifacts / "calibration.json").exists()
