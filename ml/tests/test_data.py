from pathlib import Path

import pytest

from humanized_detector.data import DataConfig, prepare_records


def test_prepare_records_is_seeded_balanced_and_reports_row_split_limitation(tmp_path: Path) -> None:
    rows = [
        {"idx": index, "sentence": f"human passage {index}", "label": 0}
        for index in range(20)
    ] + [
        {"idx": 100 + index, "sentence": f"humanized ai passage {index}", "label": 1}
        for index in range(20)
    ]

    first = prepare_records(rows, tmp_path / "first", DataConfig(samples_per_class=10, seed=7))
    second = prepare_records(rows, tmp_path / "second", DataConfig(samples_per_class=10, seed=7))

    assert first.class_counts == {0: 10, 1: 10}
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (tmp_path / "second" / "train.jsonl").read_bytes()
    assert "row-level" in (tmp_path / "first" / "report.json").read_text(encoding="utf-8")


def test_prepare_records_rejects_unexpected_padben_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="idx, sentence, and label"):
        prepare_records([{"idx": 1, "text": "wrong field", "label": 0}], tmp_path, DataConfig(samples_per_class=1))
