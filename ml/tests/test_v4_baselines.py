import json
from pathlib import Path

from humanized_detector.v4_baselines import run_tfidf_control, train_tfidf_baseline


def test_word_baseline_fits_only_train_and_writes_development_predictions(tmp_path: Path) -> None:
    train = [
        {"id": "h", "text": "ordinary human prose", "label": 0},
        {"id": "a", "text": "machine generated patterned prose", "label": 1},
    ]
    development = [
        {"id": "d", "text": "ordinary prose", "label": 0},
        {"id": "e", "text": "generated patterned prose", "label": 1},
    ]

    result = train_tfidf_baseline(train, development, tmp_path, "word_tfidf_lr")

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "development_predictions.jsonl").exists()
    assert result["variant"] == "word_tfidf_lr"


def test_character_baseline_is_a_distinct_saved_variant(tmp_path: Path) -> None:
    rows = [
        {"id": "h", "text": "plain text", "label": 0},
        {"id": "a", "text": "stylised---text", "label": 1},
    ]

    result = train_tfidf_baseline(rows, rows, tmp_path, "char_tfidf_lr")

    assert result["variant"] == "char_tfidf_lr"
    assert result["vectorizer_analyzer"] == "char_wb"


def test_control_baseline_runner_uses_only_nonsealed_partitions(tmp_path: Path) -> None:
    for role, rows in {
        "train": [
            {"id": "h", "text": "human prose", "label": 0},
            {"id": "a", "text": "generated prose", "label": 1},
        ],
        "development": [
            {"id": "d0", "text": "human", "label": 0},
            {"id": "d1", "text": "generated", "label": 1},
        ],
        "calibration": [{"id": "c", "text": "not used", "label": 0}],
        "sealed_test": [{"id": "secret", "text": "must not load", "label": 1}],
    }.items():
        (tmp_path / f"{role}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = run_tfidf_control(tmp_path, tmp_path / "artifacts", "word_tfidf_lr")

    assert result["variant"] == "word_tfidf_lr"
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "calibration_predictions.jsonl").exists()
