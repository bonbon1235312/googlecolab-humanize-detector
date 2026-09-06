from humanized_detector.v6_jfleg import adapt_jfleg_rows


def test_jfleg_adapter_preserves_human_origin_for_source_and_human_corrections() -> None:
    rows = [{
        "sentence": "An English learner sentence.",
        "corrections": ["A human-written fluent revision.", "Another human revision."],
    }]

    candidates = adapt_jfleg_rows(rows, split_name="dev")

    assert [(candidate.origin, candidate.edit_actor, candidate.label) for candidate in candidates] == [
        ("human", "not_applicable", 0),
        ("human", "human_expert", 0),
        ("human", "human_expert", 0),
    ]
    assert {candidate.lineage_id for candidate in candidates} == {"jfleg:dev:0"}
    assert candidates[1].editor_family == "jfleg_human_ref0"


def test_jfleg_adapter_rejects_rows_without_human_source_or_corrections() -> None:
    try:
        adapt_jfleg_rows([{"sentence": "", "corrections": []}], split_name="dev")
    except ValueError as error:
        assert "no usable human source or corrections" in str(error)
    else:
        raise AssertionError("expected provenance-incomplete row to fail closed")


def test_jfleg_adapter_can_preserve_original_source_indexes_after_an_explicit_filter() -> None:
    candidates = adapt_jfleg_rows(
        [{"sentence": "Kept.", "corrections": ["Human correction."]}],
        split_name="test",
        source_indices=[755],
    )

    assert {candidate.lineage_id for candidate in candidates} == {"jfleg:test:755"}
