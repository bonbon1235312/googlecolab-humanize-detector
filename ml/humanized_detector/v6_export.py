"""Export a manifest-authorized V6 pool into V4.8's non-sealed input contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .v6_manifest import V6Record


_ROLE_NAMES = {"train": "train", "selection_dev": "development", "calibration": "calibration"}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _training_row(record: V6Record) -> dict[str, object]:
    return {
        "id": record.id,
        "lineage_id": record.lineage_id,
        "text": record.text,
        "label": record.label,
        "source": record.dataset,
        "domain": record.domain,
        "origin": record.origin,
        "edit_actor": record.edit_actor,
        "editor_family": record.editor_family,
        "generator_family": record.generator_family,
        "transformation": record.transformation,
        "sampling_weight": record.sampling_weight,
    }


def export_v6_partitions(
    records: Sequence[V6Record], manifest: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    """Write train/development/calibration JSONLs only after the V6 audit authorizes training."""
    audit = manifest.get("pretrain_audit")
    if not isinstance(audit, Mapping) or audit.get("TRAINING AUTHORIZED") != "YES":
        raise ValueError("V6 training export requires an authorized pretrain manifest")
    for field in ("manifest_sha256", "sampling_policy_sha256", "dedup_policy_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"manifest is missing established {field}")
    lineage_map = manifest.get("lineage_map")
    if not isinstance(lineage_map, list):
        raise ValueError("manifest is missing the lineage map")
    expected = {str(row.get("id")): row for row in lineage_map if isinstance(row, Mapping)}
    if len(expected) != len(records) or any(expected.get(record.id) != record.metadata() for record in records):
        raise ValueError("V6 records do not exactly match the authorized manifest lineage map")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, name in _ROLE_NAMES.items():
        _write_jsonl(output_dir / f"{name}.jsonl", [_training_row(record) for record in records if record.split == split])
    contract = {
        "protocol_version": "v6",
        "manifest_sha256": manifest["manifest_sha256"],
        "sampling_policy_sha256": manifest["sampling_policy_sha256"],
        "dedup_policy_sha256": manifest["dedup_policy_sha256"],
        "partition_counts": {name: sum(record.split == split for record in records) for split, name in _ROLE_NAMES.items()},
        "sealed_data_loaded": False,
    }
    (output_dir / "v6_training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract
