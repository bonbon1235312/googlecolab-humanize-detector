"""Convert selected RAID source families into V4 sealed-test records."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

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


def collect_raid_paraphrase_candidates(rows: Iterable[Mapping[str, object]]) -> tuple[list[dict[str, object]], str]:
    """Stream RAID rows while retaining one deterministic human/paraphrase candidate per source."""
    families: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        missing = sorted(RAID_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"RAID record is missing required fields: {', '.join(missing)}")
        model = str(row["model"])
        attack = str(row["attack"])
        if not ((model == "human" and attack == "none") or (model != "human" and attack == "paraphrase")):
            continue
        source_id = str(row["source_id"])
        stored = {field: row[field] for field in RAID_REQUIRED_FIELDS}
        role = "human" if model == "human" else "paraphrase"
        candidate_key = (str(stored["id"]),) if role == "human" else (str(stored["model"]), str(stored["id"]))
        current = families[source_id].get(role)
        current_key = None if current is None else ((str(current["id"]),) if role == "human" else (str(current["model"]), str(current["id"])))
        if current_key is None or candidate_key < current_key:
            families[source_id][role] = stored

    candidates: list[dict[str, object]] = []
    for source_id in sorted(families):
        family = families[source_id]
        if "human" in family and "paraphrase" in family:
            candidates.extend((family["human"], family["paraphrase"]))
    digest = hashlib.sha256()
    for row in candidates:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return candidates, digest.hexdigest()


def select_raid_paraphrase_pairs(rows: Sequence[Mapping[str, object]], *, target_pairs: int, seed: int) -> list[V4Record]:
    """Select a deterministic, source-family-safe subset of RAID paraphrase pairs."""
    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")
    all_pairs = build_raid_paraphrase_pairs(rows)
    families: dict[str, list[V4Record]] = defaultdict(list)
    for record in all_pairs:
        families[record.lineage_id].append(record)
    ranked_families = sorted(
        families.items(),
        key=lambda item: (hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest(), item[0]),
    )
    selected = ranked_families[:target_pairs]
    return [record for _, family in selected for record in family]
