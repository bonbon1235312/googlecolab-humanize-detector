"""Convert selected RAID source families into V4 sealed-test records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .v4_manifest import V4Record


RAID_REQUIRED_FIELDS = frozenset({"id", "source_id", "adv_source_id", "model", "attack", "domain", "generation"})


def _source_fields(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "raid_adv_source_id": str(row["adv_source_id"]),
        "raid_model": str(row["model"]),
        "raid_attack": str(row["attack"]),
        "raid_domain": str(row["domain"]),
    }


def _record(row: Mapping[str, object], *, label: int, provenance: str, parent_id: str | None) -> V4Record:
    identifier = f"raid:{row['id']}"
    return V4Record.from_mapping({
        "id": identifier,
        "lineage_id": f"raid:{row['source_id']}",
        "text": str(row["generation"]),
        "label": label,
        "source": "raid",
        "domain": str(row["domain"]),
        "provenance": provenance,
        "generator_family": str(row["model"]),
        "editor_family": "raid_adversarial_attack" if label == 1 else "none",
        "transformation_family": str(row["attack"]),
        "split": "sealed_test",
        "sealed": True,
        "train_eligible": False,
        "parent_id": parent_id,
        "source_fields": _source_fields(row),
    })


def build_raid_paraphrase_pairs(rows: Sequence[Mapping[str, object]]) -> list[V4Record]:
    """Return one human/AI-paraphrase pair for every eligible RAID source family.

    The caller supplies only the selected external RAID cohort. This function never
    downloads RAID, writes a benchmark, or exposes data to V4 training paths.
    """
    families: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        missing = sorted(RAID_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"RAID record is missing required fields: {', '.join(missing)}")
        families[str(row["source_id"])].append(row)

    records: list[V4Record] = []
    for source_id in sorted(families):
        family = families[source_id]
        humans = sorted((row for row in family if str(row["model"]) == "human" and str(row["attack"]) == "none"), key=lambda row: str(row["id"]))
        paraphrases = sorted((row for row in family if str(row["model"]) != "human" and str(row["attack"]) == "paraphrase"), key=lambda row: (str(row["model"]), str(row["id"])))
        if not humans or not paraphrases:
            continue
        human = _record(humans[0], label=0, provenance="human", parent_id=None)
        ai = _record(paraphrases[0], label=1, provenance="ai_paraphrased", parent_id=human.id)
        records.extend((human, ai))
    return records
