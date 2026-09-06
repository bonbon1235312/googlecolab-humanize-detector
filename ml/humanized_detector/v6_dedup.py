"""Deterministic pre-split removal of V6 cross-lineage near duplicates."""

from __future__ import annotations

import hashlib
from typing import Sequence

from .v6_manifest import V6Record, _candidate_pairs, _char_shingles, _jaccard, canonical_text


def _rank(seed: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\n{identifier}".encode("utf-8")).hexdigest()


def remove_cross_lineage_near_duplicates(
    records: Sequence[V6Record], *, seed: str, threshold: float = 0.90
) -> tuple[list[V6Record], list[dict[str, object]]]:
    """Remove one record from each live cross-lineage duplicate pair and log why."""
    if not seed.strip():
        raise ValueError("dedup seed must be explicit")
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    shingles = [_char_shingles(record.text) for record in records]
    active = {record.id for record in records}
    removals: list[dict[str, object]] = []
    for left_index, right_index in sorted(_candidate_pairs(records)):
        left, right = records[left_index], records[right_index]
        if left.id not in active or right.id not in active:
            continue
        if left.lineage_id == right.lineage_id:
            continue
        if len(canonical_text(left.text)) < 100 or len(canonical_text(right.text)) < 100:
            continue
        similarity = _jaccard(shingles[left_index], shingles[right_index])
        if similarity < threshold:
            continue
        removed, retained = sorted((left, right), key=lambda record: (_rank(seed, record.id), record.id))
        active.remove(retained.id)
        removals.append(
            {
                "id": retained.id,
                "conflicting_id": removed.id,
                "jaccard": similarity,
                "reason": "cross_lineage_near_duplicate",
            }
        )
    return [record for record in records if record.id in active], removals
