import json
from pathlib import Path

import pytest

from humanized_detector.v4_control import load_v4_control_partitions, model_config_for_capacity
from humanized_detector.v4_train import train_v4_transformer


def test_control_loader_reads_only_nonsealed_roles(tmp_path: Path) -> None:
    for role in ("train", "development", "calibration"):
        (tmp_path / f"{role}.jsonl").write_text('{"id":"' + role + '"}\n', encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text('{"id":"secret"}\n', encoding="utf-8")

    loaded = load_v4_control_partitions(tmp_path)

    assert set(loaded) == {"train", "development", "calibration"}
    assert loaded["train"] == [{"id": "train"}]


def test_capacity_presets_keep_attention_dimensions_valid() -> None:
    small = model_config_for_capacity(4000, "5m")
    larger = model_config_for_capacity(4000, "12m")

    assert small.hidden_size == 192
    assert larger.hidden_size > small.hidden_size
    assert larger.hidden_size % larger.heads == 0
    assert larger.layers >= small.layers


def test_v4_train_rejects_unknown_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        train_v4_transformer(Path("data"), Path("artifacts"), "70m", 1, 2, 1e-4, 0.01)


def test_v4_transformer_writes_calibration_predictions_without_retaining_corpus(tmp_path: Path) -> None:
    partitions = {
        "train": [
            {"id": "h1", "text": "A person writes an ordinary short paragraph.", "label": 0, "source": "test"},
            {"id": "a1", "text": "A generated response uses patterned formal wording.", "label": 1, "source": "test"},
        ],
        "development": [
            {"id": "h2", "text": "A human writes a different natural sentence.", "label": 0, "source": "test"},
            {"id": "a2", "text": "Generated content repeats a formal response pattern.", "label": 1, "source": "test"},
        ],
        "calibration": [
            {"id": "h3", "text": "Another human sentence for calibration.", "label": 0, "source": "test"},
            {"id": "a3", "text": "Another generated sentence for calibration.", "label": 1, "source": "test"},
        ],
    }
    for role, rows in partitions.items():
        (tmp_path / f"{role}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text("not valid JSON and must not be read\n", encoding="utf-8")

    result = train_v4_transformer(tmp_path, tmp_path / "artifacts", "5m", 1, 2, 1e-4, 0.01)

    assert result["capacity"] == "5m"
    assert (tmp_path / "artifacts" / "calibration_predictions.jsonl").exists()
    assert not (tmp_path / "artifacts" / "tokenizer" / "training_corpus.txt").exists()
