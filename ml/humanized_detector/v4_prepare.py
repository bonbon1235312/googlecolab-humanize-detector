"""Write validated V4 role partitions and their public metadata manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .v4_audit import VALID_SPLITS, audit_v4_records
from .v4_manifest import V4Record, manifest_digest, metadata_manifest


SEALED_SOURCE_METADATA_FIELDS = frozenset({
    "source_locator",
    "revision",
    "raw_download_sha256",
    "row_selection_rule",
    "selection_seed",
    "sealed_at",
})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_v4_dataset(records: Sequence[V4Record], output_dir: Path, source_metadata: Mapping[str, object]) -> dict[str, object]:
    """Audit and write one JSONL file for every V4 role plus public metadata."""
    audit = audit_v4_records(records)
    if any(record.split == "sealed_test" for record in records):
        missing = sorted(SEALED_SOURCE_METADATA_FIELDS - set(source_metadata))
        if missing:
            raise ValueError(f"sealed source metadata is missing: {', '.join(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(VALID_SPLITS):
        _write_jsonl(output_dir / f"{split}.jsonl", [record.to_row() for record in records if record.split == split])
    metadata = metadata_manifest(records, source_metadata)
    digest = manifest_digest(metadata)
    _write_json(output_dir / "metadata_manifest.json", {**metadata, "sha256": digest})
    report: dict[str, object] = {
        "checked_records": audit.checked_records,
        "split_counts": dict(audit.split_counts),
        "metadata_manifest_sha256": digest,
    }
    _write_json(output_dir / "report.json", report)
    return report


def load_v4_partition(data_dir: Path, split: str) -> list[dict[str, object]]:
    """Load only train/development/calibration records for non-final V4 workflows."""
    if split not in {"train", "development", "calibration"}:
        raise ValueError("non-final loaders cannot load sealed partitions")
    path = data_dir / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
