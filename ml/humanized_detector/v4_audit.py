"""Integrity checks for V4 provenance manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence
import unicodedata

from .v4_manifest import V4Record


VALID_SPLITS = frozenset({"train", "development", "calibration", "sealed_test"})


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = _canonical(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


@dataclass(frozen=True)
class V4Audit:
    checked_records: int
    split_counts: Mapping[str, int]


def audit_v4_records(records: Sequence[V4Record], near_duplicate_threshold: float = 0.85) -> V4Audit:
    """Reject invalid sealing state before a V4 dataset is written or consumed."""
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    by_id: dict[str, V4Record] = {}
    lineage_split: dict[str, str] = {}
    text_split: dict[str, tuple[str, str]] = {}
    seen_shingles: dict[tuple[str, ...], set[int]] = {}
    indexed: list[tuple[V4Record, set[tuple[str, ...]]]] = []
    for record in records:
        if record.id in by_id:
            raise ValueError(f"duplicate record ID: {record.id}")
        by_id[record.id] = record
        if record.split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {record.split}")
        if record.split == "sealed_test" and (not record.sealed or record.train_eligible):
            raise ValueError("sealed_test records must be sealed and not train eligible")
        if record.split != "sealed_test" and record.sealed:
            raise ValueError("only sealed_test records may be sealed")
        previous_split = lineage_split.setdefault(record.lineage_id, record.split)
        if previous_split != record.split:
            raise ValueError(f"lineage crosses {previous_split} and {record.split}: {record.lineage_id}")
        canonical = _canonical(record.text)
        if not canonical:
            raise ValueError(f"blank text in record {record.id}")
        duplicate = text_split.get(canonical)
        if duplicate is not None and duplicate[0] != record.split:
            raise ValueError(f"exact duplicate across {duplicate[0]} and {record.split}: {duplicate[1]} / {record.id}")
        text_split.setdefault(canonical, (record.split, record.id))
        signatures = _shingles(record.text)
        candidates = {candidate for signature in signatures for candidate in seen_shingles.get(signature, set())}
        for candidate in candidates:
            other, other_signatures = indexed[candidate]
            if other.split == record.split:
                continue
            union = signatures | other_signatures
            similarity = len(signatures & other_signatures) / len(union) if union else 0.0
            if similarity >= near_duplicate_threshold:
                raise ValueError(f"near duplicate across {other.split} and {record.split}: {other.id} / {record.id} ({similarity:.3f})")
        index = len(indexed)
        indexed.append((record, signatures))
        for signature in signatures:
            seen_shingles.setdefault(signature, set()).add(index)
    for record in records:
        if record.parent_id is None:
            continue
        parent = by_id.get(record.parent_id)
        if parent is None:
            raise ValueError(f"parent is missing for record {record.id}: {record.parent_id}")
        if parent.split != record.split:
            raise ValueError(f"parent crosses {parent.split} and {record.split}: {parent.id} / {record.id}")
    return V4Audit(checked_records=len(records), split_counts=dict(sorted(Counter(record.split for record in records).items())))
