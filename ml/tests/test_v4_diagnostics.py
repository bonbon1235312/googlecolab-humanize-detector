import json
from pathlib import Path

from humanized_detector.model import ModelConfig
from humanized_detector.v3_train import train_v3_model
from humanized_detector.v4_diagnostics import bootstrap_roc_auc, development_subtype_metrics
from humanized_detector.v4_diagnostics import score_padben_diagnostic, write_v4_fit_diagnostics


def test_bootstrap_auc_is_deterministic_and_contains_the_observed_auc() -> None:
    first = bootstrap_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], iterations=100, seed=7)
    second = bootstrap_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], iterations=100, seed=7)

    assert first["lower"] <= first["point"] <= first["upper"]
    assert first == second


def test_subtype_metrics_compare_each_positive_provenance_with_humans() -> None:
    rows = [
        {"label": 0, "provenance": "human"},
        {"label": 1, "provenance": "raw_ai"},
        {"label": 1, "provenance": "expert_edited_ai"},
    ]

    report = development_subtype_metrics(rows, [0.1, 0.9, 0.8])

    assert set(report) == {"raw_ai", "expert_edited_ai"}
    assert report["raw_ai"]["roc_auc"] == 1.0


def test_fit_diagnostics_score_train_and_development_without_reading_other_roles(tmp_path: Path) -> None:
    rows = {
        "train": [
            {"id": "h1", "text": "human writing alpha", "label": 0, "source": "beemo", "provenance": "human"},
            {"id": "a1", "text": "generated writing alpha", "label": 1, "source": "beemo", "provenance": "raw_ai"},
        ],
        "development": [
            {"id": "h2", "text": "human writing beta", "label": 0, "source": "beemo", "provenance": "human"},
            {"id": "a2", "text": "edited generated writing beta", "label": 1, "source": "beemo", "provenance": "expert_edited_ai"},
        ],
    }
    for split, split_rows in rows.items():
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in split_rows) + "\n", encoding="utf-8")
    (tmp_path / "calibration.jsonl").write_text("not valid JSON\n", encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text("not valid JSON\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    train_v3_model(tmp_path / "train.jsonl", tmp_path / "development.jsonl", artifacts, ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), "fusion_concat", epochs=1, batch_size=2)

    report = write_v4_fit_diagnostics(tmp_path, artifacts, bootstrap_iterations=10)

    assert report["train"]["n"] == 2
    assert report["development"]["subtypes"]
    assert (artifacts / "fit_diagnostics.json").exists()


def test_padben_diagnostic_is_scored_separately_from_control_partitions(tmp_path: Path) -> None:
    train_rows = [
        {"id": "h1", "text": "human writing alpha", "label": 0, "source": "beemo", "provenance": "human"},
        {"id": "a1", "text": "generated writing alpha", "label": 1, "source": "beemo", "provenance": "raw_ai"},
    ]
    development_rows = [
        {"id": "h2", "text": "human writing beta", "label": 0, "source": "beemo", "provenance": "human"},
        {"id": "a2", "text": "edited generated writing beta", "label": 1, "source": "beemo", "provenance": "expert_edited_ai"},
    ]
    for split, rows in (("train", train_rows), ("development", development_rows)):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    diagnostic_dir = tmp_path / "padben-diagnostic"
    diagnostic_dir.mkdir()
    diagnostic_rows = [
        {"id": "p1", "text": "unused human padben sentence", "label": 0, "source": "padben", "provenance": "human"},
        {"id": "p2", "text": "unused ai padben paraphrase", "label": 1, "source": "padben", "provenance": "ai_humanized"},
    ]
    (diagnostic_dir / "padben_diagnostic.jsonl").write_text("\n".join(json.dumps(row) for row in diagnostic_rows) + "\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    train_v3_model(tmp_path / "train.jsonl", tmp_path / "development.jsonl", artifacts, ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), "fusion_concat", epochs=1, batch_size=2)

    report = score_padben_diagnostic(diagnostic_dir, artifacts)

    assert report["n"] == 2
    assert (artifacts / "padben_diagnostic_metrics.json").exists()
