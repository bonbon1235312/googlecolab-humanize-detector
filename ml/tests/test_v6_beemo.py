import pytest

from humanized_detector.v6_beemo import adapt_beemo_rows, write_beemo_candidates


def source_row() -> dict[str, object]:
    return {
        "prompt_id": "prompt-1",
        "model_name": "llama-3.1-70b",
        "human_output": "A genuine human response.",
        "model_output": "A raw machine response.",
        "human_edits": "A human expert revision of the machine response.",
        "llama-3.1-70b_edits": "[{'P1': 'An LLM revision.'}]",
        "gpt-4o_edits": "[]",
    }


def test_beemo_adapter_derives_labels_from_origin_not_editor() -> None:
    candidates = adapt_beemo_rows([source_row()])

    assert [(candidate.origin, candidate.edit_actor, candidate.label) for candidate in candidates] == [
        ("human", "not_applicable", 0),
        ("ai", "not_applicable", 1),
        ("ai", "human_expert", 1),
        ("ai", "llm", 1),
    ]
    assert {candidate.lineage_id for candidate in candidates} == {"beemo:prompt-1"}
    assert candidates[2].parent_lineage_id == "beemo:prompt-1"
    assert candidates[1].generator_family == "llama-3.1-70b"


def test_beemo_adapter_rejects_ai_rows_without_an_explicit_generator_family() -> None:
    row = source_row()
    row.pop("model_name")

    with pytest.raises(ValueError, match="generator family"):
        adapt_beemo_rows([row])


def test_beemo_writer_emits_unsplit_canonical_candidate_rows(tmp_path) -> None:
    output = tmp_path / "beemo_v6_candidates.jsonl"

    count = write_beemo_candidates(output, adapt_beemo_rows([source_row()]))

    assert count == 4
    first = output.read_text(encoding="utf-8").splitlines()[0]
    assert '"split"' not in first
    assert '"origin":"human"' in first
