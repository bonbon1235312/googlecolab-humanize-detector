import json
from pathlib import Path

from humanized_detector.beemo import BeemoRecord
from humanized_detector.v4_control_data import V4ControlDataConfig, prepare_padben_diagnostic, prepare_v4_control_dataset


def _padben_rows() -> list[dict[str, object]]:
    return [
        {"idx": 1, "sentence": "A genuine human sentence.", "label": 0},
        {"idx": 2, "sentence": "Another human sentence.", "label": 0},
        {"idx": 3, "sentence": "A deeply rewritten AI sentence.", "label": 1},
        {"idx": 4, "sentence": "Another rewritten AI sentence.", "label": 1},
    ]


def _beemo_rows_for_three_prompt_groups() -> list[BeemoRecord]:
    rows: list[BeemoRecord] = []
    for group_id in ("prompt-a", "prompt-b", "prompt-c"):
        rows.extend((
            BeemoRecord(f"Human response {group_id}.", 0, group_id, "human", "human"),
            BeemoRecord(f"Raw AI response {group_id}.", 1, group_id, "raw_ai", "raw_ai"),
            BeemoRecord(f"Expert edit {group_id}.", 1, group_id, "expert_edit", "expert_edited_ai"),
        ))
    return rows


def test_control_preparation_keeps_beemo_variants_atomic_and_records_the_raw_ai_parent(tmp_path: Path) -> None:
    report = prepare_v4_control_dataset(
        _padben_rows(),
        _beemo_rows_for_three_prompt_groups(),
        tmp_path,
        V4ControlDataConfig(padben_samples_per_class=1),
    )
    rows = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    expert = next(row for row in rows if row["provenance"] == "expert_edited_ai")

    assert report["split_counts"]
    assert expert["parent_id"] in {row["id"] for row in rows}
    assert len({row["split"] for row in rows if row["lineage_id"] == expert["lineage_id"]}) == 1


def test_padben_diagnostic_uses_only_control_unused_rows(tmp_path: Path) -> None:
    rows = _padben_rows() + [
        {"idx": 5, "sentence": "A third genuine human sentence.", "label": 0},
        {"idx": 6, "sentence": "A third deep AI paraphrase sentence.", "label": 1},
    ]
    prepare_v4_control_dataset(rows, _beemo_rows_for_three_prompt_groups(), tmp_path / "control", V4ControlDataConfig(seed=4, padben_samples_per_class=1))

    report = prepare_padben_diagnostic(rows, tmp_path / "control", tmp_path / "diagnostic", samples_per_class=1, seed=9)
    diagnostic = [json.loads(line) for line in (tmp_path / "diagnostic" / "padben_diagnostic.jsonl").read_text(encoding="utf-8").splitlines()]
    control_ids = {
        json.loads(line)["id"]
        for path in (tmp_path / "control").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }

    assert report["selected_rows"] == 2
    assert {row["label"] for row in diagnostic} == {0, 1}
    assert not ({row["id"] for row in diagnostic} & control_ids)
    assert all(row["split"] == "padben_diagnostic" and not row["train_eligible"] for row in diagnostic)
