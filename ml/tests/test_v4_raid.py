from humanized_detector.v4_audit import audit_v4_records
from humanized_detector.v4_raid import collect_raid_paraphrase_candidates, build_raid_paraphrase_pairs, select_raid_paraphrase_pairs


def _row(identifier: str, source_id: str, model: str, attack: str, generation: str) -> dict[str, object]:
    return {
        "id": identifier,
        "source_id": source_id,
        "adv_source_id": f"adv:{source_id}",
        "model": model,
        "attack": attack,
        "domain": "abstracts",
        "generation": generation,
    }


def test_raid_adapter_builds_one_provenance_preserving_pair_per_source_family() -> None:
    rows = [
        _row("human-1", "source-1", "human", "none", "A genuine human abstract."),
        _row("gpt-1", "source-1", "gpt-4", "paraphrase", "A paraphrased AI abstract."),
        _row("other-1", "source-1", "gpt-4", "synonym", "An ignored attack."),
        _row("gpt-2", "source-2", "gpt-4", "paraphrase", "Ignored without a human pair."),
    ]

    records = build_raid_paraphrase_pairs(rows)

    assert [(record.id, record.label) for record in records] == [("raid:human-1", 0), ("raid:gpt-1", 1)]
    assert records[0].lineage_id == records[1].lineage_id == "raid:source-1"
    assert records[0].provenance == "human"
    assert records[1].provenance == "ai_paraphrased"
    assert records[1].parent_id == "raid:human-1"
    assert records[1].source_fields == {
        "raid_adv_source_id": "adv:source-1",
        "raid_model": "gpt-4",
        "raid_attack": "paraphrase",
        "raid_domain": "abstracts",
    }
    audit_v4_records(records)


def test_raid_selector_uses_a_stable_source_family_sample() -> None:
    rows = []
    for index in range(4):
        rows.extend((
            _row(f"human-{index}", f"source-{index}", "human", "none", f"Human {index}."),
            _row(f"gpt-{index}", f"source-{index}", "gpt-4", "paraphrase", f"AI {index}."),
        ))

    first = select_raid_paraphrase_pairs(rows, target_pairs=2, seed=19)
    second = select_raid_paraphrase_pairs(list(reversed(rows)), target_pairs=2, seed=19)

    assert [record.id for record in first] == [record.id for record in second]
    assert len(first) == 4
    assert {record.lineage_id for record in first} == {first[0].lineage_id, first[2].lineage_id}
    assert first[0].lineage_id != first[2].lineage_id


def test_raid_streaming_collector_keeps_only_one_deterministic_pair_per_family() -> None:
    rows = [
        _row("human-z", "source-1", "human", "none", "Human z."),
        _row("human-a", "source-1", "human", "none", "Human a."),
        _row("model-z", "source-1", "mistral", "paraphrase", "Mistral."),
        _row("model-a", "source-1", "gpt-4", "paraphrase", "GPT."),
    ]

    candidates, snapshot_hash = collect_raid_paraphrase_candidates(iter(rows))

    assert [row["id"] for row in candidates] == ["human-a", "model-a"]
    assert len(snapshot_hash) == 64
