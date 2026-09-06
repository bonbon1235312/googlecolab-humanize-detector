from humanized_detector.v6_sampling import cap_variants_per_lineage


def test_variant_cap_retains_one_deterministic_llm_edit_per_lineage() -> None:
    rows = [
        {"id": "raw", "lineage_id": "lineage-1", "transformation": "raw_generation", "text": "raw"},
        {"id": "edit-a", "lineage_id": "lineage-1", "transformation": "llm_edit", "text": "alpha"},
        {"id": "edit-b", "lineage_id": "lineage-1", "transformation": "llm_edit", "text": "beta"},
    ]

    selected = cap_variants_per_lineage(rows, caps={"llm_edit": 1}, seed="20260906")

    assert {row["id"] for row in selected} & {"raw"} == {"raw"}
    assert len([row for row in selected if row["transformation"] == "llm_edit"]) == 1
    assert selected == cap_variants_per_lineage(rows, caps={"llm_edit": 1}, seed="20260906")


def test_variant_cap_rejects_an_unestablished_cap() -> None:
    try:
        cap_variants_per_lineage([], caps={"llm_edit": 0}, seed="20260906")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected invalid cap to fail closed")
