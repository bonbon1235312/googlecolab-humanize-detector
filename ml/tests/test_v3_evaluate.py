import json
from pathlib import Path

from humanized_detector.v3_evaluate import evaluate_binary, freeze_external_benchmark, metrics_by_field
from humanized_detector.v3_calibrate import calibration_record, select_operating_threshold
from humanized_detector.v3_external import gradtex_test_c_rows


def test_evaluate_binary_reports_operating_and_calibration_metrics() -> None:
    metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9])

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["human_fpr"] == 0.0
    assert "brier_score" in metrics
    assert "tpr_at_1pct_fpr" in metrics


def test_metrics_by_field_keeps_unseen_scenarios_separate() -> None:
    rows = [{"scenario": "polish_sentence", "label": 0}, {"scenario": "polish_sentence", "label": 1}, {"scenario": "rewrite_skeptical", "label": 0}, {"scenario": "rewrite_skeptical", "label": 1}]

    result = metrics_by_field(rows, [0.1, 0.9, 0.4, 0.6], "scenario")

    assert set(result) == {"polish_sentence", "rewrite_skeptical"}
    assert result["polish_sentence"]["roc_auc"] == 1.0


def test_metrics_by_field_uses_the_supplied_calibrated_threshold() -> None:
    rows = [{"scenario": "polish_sentence", "label": 0}, {"scenario": "polish_sentence", "label": 1}]

    result = metrics_by_field(rows, [0.6, 0.7], "scenario", threshold=0.7)

    assert result["polish_sentence"]["human_fpr"] == 0.0
    assert result["polish_sentence"]["recall"] == 1.0


def test_freeze_external_benchmark_hashes_files_without_reading_labels(tmp_path: Path) -> None:
    benchmark = tmp_path / "gradtex-test-c"
    benchmark.mkdir()
    (benchmark / "records.jsonl").write_text(json.dumps({"label": 1, "text": "must stay unseen"}) + "\n", encoding="utf-8")

    manifest = freeze_external_benchmark(benchmark, tmp_path / "manifest.json", dataset="elisabeth-pl-pl/GRADTEX", revision="fixed-revision", split="test_c")

    assert manifest["dataset"] == "elisabeth-pl-pl/GRADTEX"
    assert manifest["files"][0]["sha256"]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["split"] == "test_c"


def test_select_operating_threshold_maximises_recall_within_human_fpr_ceiling() -> None:
    threshold = select_operating_threshold([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], max_human_fpr=0.0)

    assert threshold == 0.7


def test_calibration_record_reports_the_selected_operating_point() -> None:
    record = calibration_record([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], max_human_fpr=0.0)

    assert record["threshold"] == 0.7
    assert record["metrics"]["human_fpr"] == 0.0
    assert record["metrics"]["recall"] == 1.0


def test_gradtex_test_c_rows_maps_human_label_to_internal_negative_class() -> None:
    rows = gradtex_test_c_rows([
        {"text": "human text", "binary_label": 1, "test_split": "C", "scenario": "human", "scenario_family": "human"},
        {"text": "edited text", "binary_label": 0, "test_split": "C", "scenario": "polish_sentence", "scenario_family": "polish"},
        {"text": "other split", "binary_label": 0, "test_split": "B", "scenario": "polish_token", "scenario_family": "polish"},
    ])

    assert [row["label"] for row in rows] == [0, 1]
    assert [row["scenario"] for row in rows] == ["human", "polish_sentence"]
