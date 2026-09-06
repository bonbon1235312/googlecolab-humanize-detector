from humanized_detector.v6_split import assign_mage_source_splits


def mage_record(source: str, index: int, origin: str) -> dict[str, object]:
    return {"id": f"mage:{source}:{index}", "lineage_id": f"mage:{source}:{index}", "origin": origin}


def test_mage_split_assigns_whole_source_groups_and_each_origin_to_all_partitions() -> None:
    rows = [
        mage_record(f"human_{index}", index, "human") for index in range(10)
    ] + [
        mage_record(f"machine_{index}", index, "ai") for index in range(10)
    ]

    assigned = assign_mage_source_splits(rows, seed="20260906")

    by_source = {}
    for row in assigned:
        by_source.setdefault(row["lineage_id"].rsplit(":", 1)[0], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())
    assert {row["split"] for row in assigned if row["origin"] == "human"} == {"train", "selection_dev", "calibration"}
    assert {row["split"] for row in assigned if row["origin"] == "ai"} == {"train", "selection_dev", "calibration"}


def test_mage_split_is_seed_deterministic() -> None:
    rows = [mage_record(f"human_{index}", index, "human") for index in range(10)] + [
        mage_record(f"machine_{index}", index, "ai") for index in range(10)
    ]

    assert assign_mage_source_splits(rows, seed="fixed") == assign_mage_source_splits(rows, seed="fixed")
