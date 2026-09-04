import json
from pathlib import Path

from humanized_detector.beemo import BeemoRecord
from humanized_detector.v4_control_data import V4ControlDataConfig, prepare_v4_control_dataset


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
