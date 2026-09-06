"""Deterministic V6 sampling guards applied after lineage-safe splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping, Sequence


def _rank(seed: str, row: Mapping[str, object]) -> str:
    material = f"{seed}\n{row.get('lineage_id')}\n{row.get('transformation')}\n{row.get('id')}\n{row.get('text')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def cap_variants_per_lineage(
    rows: Sequence[Mapping[str, object]], *, caps: Mapping[str, int], seed: str
) -> list[dict[str, object]]:
    """Cap declared transformations per lineage without changing any provenance label."""
    if not seed.strip():
        raise ValueError("sampling seed must be explicit")
    if any(not isinstance(cap, int) or cap <= 0 for cap in caps.values()):
        raise ValueError("variant caps must be positive integers")
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        lineage = str(row.get("lineage_id") or "").strip()
        transformation = str(row.get("transformation") or "").strip()
        if not lineage or not transformation:
            raise ValueError("variant capping requires lineage_id and transformation")
        groups[(lineage, transformation)].append(row)
    selected: list[dict[str, object]] = []
    for (lineage, transformation), candidates in sorted(groups.items()):
        del lineage
        limit = caps.get(transformation, len(candidates))
        selected.extend(dict(row) for row in sorted(candidates, key=lambda row: _rank(seed, row))[:limit])
    return selected
