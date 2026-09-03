import json
from pathlib import Path

from humanized_detector.beemo import BeemoRecord, group_split, load_beemo_rows


def test_group_split_keeps_prompt_versions_together() -> None:
    records = [BeemoRecord(f"text-{group}-{variant}", variant == "edited", group, variant) for group in range(20) for variant in ("human", "edited")]

    splits = group_split(records, seed=7)

    memberships = {record.group_id: split for split, rows in splits.items() for record in rows for _ in [0]}
    assert len(memberships) == 20
    assert {record.group_id for record in splits["train"]}.isdisjoint({record.group_id for record in splits["development"]})
    assert {record.group_id for record in splits["train"]}.isdisjoint({record.group_id for record in splits["test"]})
    assert {record.group_id for record in splits["development"]}.isdisjoint({record.group_id for record in splits["test"]})


def test_load_beemo_rows_expands_editing_variants() -> None:
    rows = [{"prompt_id": "p1", "human_output": "a human answer", "human_edits": "an expert edit", "model_output": "raw AI", "llama-3.1-70b_edits": "[{'P1': 'llama edit'}]", "gpt-4o_edits": "[{'P1': 'gpt edit'}]"}]

    records = load_beemo_rows(rows)

    assert [(record.variant, record.label) for record in records] == [("human", 0), ("expert_edit", 1), ("llama-3.1-70b_P1", 1), ("gpt-4o_P1", 1)]
