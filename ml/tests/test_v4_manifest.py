import pytest

from humanized_detector.v4_audit import audit_v4_records
from humanized_detector.v4_manifest import V4Record, manifest_digest, metadata_manifest


def make_row(
    identifier: str,
    lineage_id: str,
    text: str,
    split: str,
    *,
    label: int = 0,
    parent_id: str | None = None,
    sealed: bool = False,
    train_eligible: bool = True,
) -> dict[str, object]:
    return {
        "id": identifier,
        "lineage_id": lineage_id,
        "text": text,
        "text_sha256": "provided-by-source",
        "label": label,
        "source": "example",
        "domain": "essay",
        "provenance": "human" if label == 0 else "ai_humanized",
        "generator_family": "human" if label == 0 else "example-llm",
        "editor_family": "none" if label == 0 else "example-editor",
        "transformation_family": "none" if label == 0 else "paraphrase",
        "split": split,
        "sealed": sealed,
        "train_eligible": train_eligible,
        "parent_id": parent_id,
    }


def test_v4_record_normalises_text_and_calculates_its_hash() -> None:
    record = V4Record.from_mapping(make_row("human:1", "lineage:1", "A human\r\npassage.", "train"))

    assert record.text == "A human\npassage."
    assert record.text_sha256 != "provided-by-source"


def test_v4_record_rejects_invalid_binary_label() -> None:
    row = make_row("human:1", "lineage:1", "A human passage.", "train", label=2)

    with pytest.raises(ValueError, match="label"):
        V4Record.from_mapping(row)


def test_metadata_manifest_excludes_text_and_has_a_stable_digest() -> None:
    record = V4Record.from_mapping(make_row("human:1", "lineage:1", "Hidden source text.", "train"))

    first = metadata_manifest([record], {"source_locator": "https://example.test", "revision": "v1"})
    second = metadata_manifest([record], {"revision": "v1", "source_locator": "https://example.test"})

    assert "Hidden source text." not in str(first)
    assert manifest_digest(first) == manifest_digest(second)


def test_v4_record_preserves_text_safe_source_fields() -> None:
    row = make_row("raid:a", "raid:source", "A paraphrased passage.", "sealed_test", sealed=True, train_eligible=False)
    row["source_fields"] = {"raid_adv_source_id": "adv-1", "raid_model": "gpt-4", "raid_attack": "paraphrase"}

    record = V4Record.from_mapping(row)

    assert record.source_fields["raid_adv_source_id"] == "adv-1"
    assert record.to_row()["source_fields"]["raid_model"] == "gpt-4"


def test_audit_rejects_a_train_eligible_sealed_record() -> None:
    row = make_row("sealed:1", "lineage:1", "sealed human text", "sealed_test", sealed=True, train_eligible=True)

    with pytest.raises(ValueError, match="sealed_test"):
        audit_v4_records([V4Record.from_mapping(row)])


def test_audit_rejects_a_lineage_crossing_split_boundaries() -> None:
    train = V4Record.from_mapping(make_row("a", "lineage:x", "one two three four five six", "train"))
    development = V4Record.from_mapping(make_row("b", "lineage:x", "different lineage variant", "development"))

    with pytest.raises(ValueError, match="lineage crosses"):
        audit_v4_records([train, development])


def test_audit_rejects_a_parent_crossing_split_boundaries() -> None:
    parent = V4Record.from_mapping(make_row("parent", "lineage:x", "one two three four five six", "train"))
    child = V4Record.from_mapping(make_row("child", "lineage:y", "a generated variant", "development", parent_id="parent"))

    with pytest.raises(ValueError, match="parent"):
        audit_v4_records([parent, child])


def test_audit_rejects_exact_duplicates_across_splits() -> None:
    train = V4Record.from_mapping(make_row("a", "la", "one two three four five six", "train"))
    development = V4Record.from_mapping(make_row("b", "lb", "one two three four five six", "development"))

    with pytest.raises(ValueError, match="exact duplicate"):
        audit_v4_records([train, development])


def test_audit_rejects_cross_split_near_duplicates() -> None:
    words = [f"word{index}" for index in range(100)]
    train = V4Record.from_mapping(make_row("a", "la", " ".join(words), "train"))
    words[-1] = "replacement"
    development = V4Record.from_mapping(make_row("b", "lb", " ".join(words), "development"))

    with pytest.raises(ValueError, match="near duplicate"):
        audit_v4_records([train, development])
