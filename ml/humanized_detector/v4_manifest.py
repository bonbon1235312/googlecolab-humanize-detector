"""Typed, provenance-preserving records for V4 experiment data."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence


def _normalise_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V4Record:
    id: str
    lineage_id: str
    text: str
    text_sha256: str
    label: int
    source: str
    domain: str
    provenance: str
    generator_family: str
    editor_family: str
    transformation_family: str
    split: str
    sealed: bool
    train_eligible: bool
    parent_id: str | None

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "V4Record":
        text = _normalise_text(str(record["text"]))
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        return cls(
            id=str(record["id"]),
            lineage_id=str(record["lineage_id"]),
            text=text,
            text_sha256=_text_sha256(text),
            label=label,
            source=str(record["source"]),
            domain=str(record["domain"]),
            provenance=str(record["provenance"]),
            generator_family=str(record["generator_family"]),
            editor_family=str(record["editor_family"]),
            transformation_family=str(record["transformation_family"]),
            split=str(record["split"]),
            sealed=bool(record["sealed"]),
            train_eligible=bool(record["train_eligible"]),
            parent_id=str(record["parent_id"]) if record["parent_id"] is not None else None,
        )

    def to_row(self) -> dict[str, object]:
        return {
            "id": self.id,
            "lineage_id": self.lineage_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "label": self.label,
            "source": self.source,
            "domain": self.domain,
            "provenance": self.provenance,
            "generator_family": self.generator_family,
            "editor_family": self.editor_family,
            "transformation_family": self.transformation_family,
            "split": self.split,
            "sealed": self.sealed,
            "train_eligible": self.train_eligible,
            "parent_id": self.parent_id,
        }


def metadata_manifest(records: Sequence[V4Record], source_metadata: Mapping[str, object]) -> dict[str, object]:
    """Create publicly shareable record metadata without copying source text."""
    rows = [{key: value for key, value in record.to_row().items() if key != "text"} for record in records]
    return {
        "source": dict(source_metadata),
        "records": rows,
        "counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
    }


def manifest_digest(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 digest of canonical JSON metadata."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
