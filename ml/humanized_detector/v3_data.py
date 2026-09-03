"""Data-integrity utilities for the V3 mixed-domain experiment."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class BoundaryAudit:
    checked_records: int


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = _canonical(text).split()
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def audit_split_boundaries(splits: Mapping[str, Sequence[Mapping[str, object]]], threshold: float = 0.85) -> BoundaryAudit:
    """Reject exact or high-overlap text found across different partitions."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    seen_text: dict[str, tuple[str, str]] = {}
    seen_shingles: dict[tuple[str, ...], set[int]] = {}
    indexed: list[tuple[str, str, set[tuple[str, ...]]]] = []
    for split_name, rows in splits.items():
        for row in rows:
            identifier = str(row.get("id", "<missing-id>"))
            text = str(row.get("text", ""))
            canonical = _canonical(text)
            if not canonical:
                raise ValueError(f"blank text in {split_name} record {identifier}")
            prior = seen_text.get(canonical)
            if prior is not None and prior[0] != split_name:
                raise ValueError(f"exact duplicate across {prior[0]} and {split_name}: {prior[1]} / {identifier}")
            seen_text.setdefault(canonical, (split_name, identifier))
            signatures = _shingles(text)
            candidates = {candidate for signature in signatures for candidate in seen_shingles.get(signature, set())}
            for candidate in candidates:
                other_split, other_identifier, other_signatures = indexed[candidate]
                if other_split == split_name:
                    continue
                union = signatures | other_signatures
                similarity = len(signatures & other_signatures) / len(union) if union else 0.0
                if similarity >= threshold:
                    raise ValueError(f"near duplicate across {other_split} and {split_name}: {other_identifier} / {identifier} ({similarity:.3f})")
            index = len(indexed)
            indexed.append((split_name, identifier, signatures))
            for signature in signatures:
                seen_shingles.setdefault(signature, set()).add(index)
    return BoundaryAudit(checked_records=len(indexed))
