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


def test_metadata_manifest_excludes_text_and_has_a_stable_digest() -> None:
    record = V4Record.from_mapping(make_row("human:1", "lineage:1", "Hidden source text.", "train"))

    first = metadata_manifest([record], {"source_locator": "https://example.test", "revision": "v1"})
    second = metadata_manifest([record], {"revision": "v1", "source_locator": "https://example.test"})

    assert "Hidden source text." not in str(first)
    assert manifest_digest(first) == manifest_digest(second)
