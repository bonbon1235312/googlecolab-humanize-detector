from humanized_detector.v6_dedup import remove_cross_lineage_near_duplicates
from humanized_detector.v6_manifest import V6Record


def record(identifier: str, lineage_id: str, text: str) -> V6Record:
    return V6Record.from_mapping(
        {
            "id": identifier,
            "lineage_id": lineage_id,
            "parent_lineage_id": None,
            "text": text,
            "dataset": "mage",
            "domain": "essay",
            "source_family": "mage-source",
            "template_id": "not_applicable",
            "origin": "ai",
            "edit_actor": "not_applicable",
            "editor_family": "not_applicable",
            "generator_family": "mage-source",
            "transformation": "raw_generation",
            "split": "train",
            "sampling_weight": 1.0,
        }
    )


def test_cross_lineage_near_duplicate_filter_removes_one_deterministically() -> None:
    original = " ".join(f"token{index}" for index in range(220))
    rows = [
        record("a", "lineage-a", original),
        record("b", "lineage-b", original.replace("token219", "replacement")),
        record("c", "lineage-c", "A genuinely unrelated prose document. " * 20),
    ]

    kept, removals = remove_cross_lineage_near_duplicates(rows, seed="20260906")

    assert len(kept) == 2
    assert len(removals) == 1
    assert removals[0]["reason"] == "cross_lineage_near_duplicate"
    assert removals[0]["jaccard"] >= 0.90
    assert (kept, removals) == remove_cross_lineage_near_duplicates(rows, seed="20260906")
