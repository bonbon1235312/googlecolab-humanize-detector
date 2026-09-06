"""Deterministic, source-held partitioning for V6 candidate records."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping, Sequence


_SPLITS = ("train", "selection_dev", "calibration")


def _source_key(row: Mapping[str, object]) -> str:
    lineage = str(row.get("lineage_id") or "")
    if not lineage.startswith("mage:") or lineage.count(":") < 2:
        raise ValueError("MAGE record must expose mage:<declared-source>:<row-index> lineage")
    return lineage.rsplit(":", 1)[0]


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\n{value}".encode("utf-8")).hexdigest()


def _split_group_keys(keys: Sequence[str]) -> dict[str, str]:
    if len(keys) < 3:
        raise ValueError("each MAGE origin requires at least three declared source groups")
    ordered = sorted(keys)
    calibration = max(1, round(len(ordered) * 0.10))
    selection_dev = max(1, round(len(ordered) * 0.20))
    if calibration + selection_dev >= len(ordered):
        raise ValueError("not enough MAGE source groups for train, selection_dev, and calibration")
    assignment = {key: "calibration" for key in ordered[:calibration]}
    assignment.update({key: "selection_dev" for key in ordered[calibration : calibration + selection_dev]})
    assignment.update({key: "train" for key in ordered[calibration + selection_dev :]})
    return assignment


def assign_mage_source_splits(rows: Sequence[Mapping[str, object]], *, seed: str) -> list[dict[str, object]]:
    """Split MAGE by its declared source group, stratified by provenance origin."""
    if not seed.strip():
        raise ValueError("split seed must be explicit")
    keys_by_origin: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        origin = str(row.get("origin") or "").casefold()
        if origin not in {"human", "ai"}:
            raise ValueError("MAGE records require an explicit human or ai origin")
        keys_by_origin[origin].add(_source_key(row))
    assignments: dict[str, str] = {}
    for origin in ("human", "ai"):
        ranked = sorted(keys_by_origin[origin], key=lambda key: _stable_rank(seed, key))
        assignments.update(_split_group_keys(ranked))
    result: list[dict[str, object]] = []
    for row in rows:
        assigned = dict(row)
        assigned["split"] = assignments[_source_key(row)]
        result.append(assigned)
    return result
