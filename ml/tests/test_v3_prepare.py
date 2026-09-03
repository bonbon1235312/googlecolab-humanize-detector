import json
from pathlib import Path

from humanized_detector.beemo import BeemoRecord
from humanized_detector.v3_prepare import V3DataConfig, prepare_v3_dataset


def test_v3_default_padben_sample_size_fits_the_task5_class_capacity() -> None:
    assert V3DataConfig().padben_samples_per_class == 15_000


def _rows() -> list[dict[str, object]]:
    return ([{"idx": f"h{index}", "sentence": f"padben human example letter{chr(97 + index)} with enough distinct words", "label": 0} for index in range(12)] + [{"idx": f"a{index}", "sentence": f"padben paraphrased ai example letter{chr(97 + index)} with enough distinct words", "label": 1} for index in range(12)])


def test_prepare_v3_dataset_preserves_sources_and_atomic_beemo_lineages(tmp_path: Path) -> None:
    beemo = [BeemoRecord(f"human response {group}", 0, str(group), "human", "human") for group in range(20)]
    beemo += [BeemoRecord(f"raw ai response {group}", 1, str(group), "raw_ai", "raw_ai") for group in range(20)]

    report = prepare_v3_dataset(_rows(), beemo, tmp_path, V3DataConfig(seed=7, padben_samples_per_class=8))

    assert report["source_counts"]["train"]["padben"] == 16
    partitions = {name: [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()] for name in ("train", "development", "calibration")}
    membership = {row["group_id"]: name for name, rows in partitions.items() for row in rows if row["source"] == "beemo"}
    assert len(membership) == 20
    assert all({name for name, rows in partitions.items() for row in rows if row["source"] == "beemo" and row["group_id"] == group} == {membership[group]} for group in membership)
    assert all({"source", "provenance", "group_id"}.issubset(row) for row in partitions["train"])
