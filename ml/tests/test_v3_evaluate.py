import json
from pathlib import Path

from humanized_detector.v3_evaluate import evaluate_binary, freeze_external_benchmark, metrics_by_field


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


def test_freeze_external_benchmark_hashes_files_without_reading_labels(tmp_path: Path) -> None:
    benchmark = tmp_path / "gradtex-test-c"
    benchmark.mkdir()
    (benchmark / "records.jsonl").write_text(json.dumps({"label": 1, "text": "must stay unseen"}) + "\n", encoding="utf-8")

    manifest = freeze_external_benchmark(benchmark, tmp_path / "manifest.json", dataset="elisabeth-pl-pl/GRADTEX", revision="fixed-revision", split="test_c")

    assert manifest["dataset"] == "elisabeth-pl-pl/GRADTEX"
    assert manifest["files"][0]["sha256"]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["split"] == "test_c"
