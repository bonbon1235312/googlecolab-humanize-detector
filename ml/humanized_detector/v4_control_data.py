"""Build the eligible PADBen/Beemo V4 control manifest."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .beemo import BeemoRecord, v3_group_split
from .v4_manifest import V4Record
from .v4_prepare import write_v4_dataset


@dataclass(frozen=True)
class V4ControlDataConfig:
    seed: int = 20260904
    padben_samples_per_class: int = 15_000


def _padben_records(rows: list[dict[str, object]], config: V4ControlDataConfig) -> list[V4Record]:
    by_label: dict[int, list[V4Record]] = {0: [], 1: []}
    seen: set[str] = set()
    for row in rows:
        if not {"idx", "sentence", "label"}.issubset(row):
            raise ValueError("PADBen records must contain idx, sentence, and label")
        label = int(row["label"])
        if label not in by_label:
            raise ValueError(f"PADBen label must be 0 or 1, got {label!r}")
        text = str(row["sentence"])
        canonical = " ".join(text.casefold().split())
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        identifier = f"padben:{row['idx']}"
        by_label[label].append(V4Record.from_mapping({
            "id": identifier,
            "lineage_id": identifier,
            "text": text,
            "label": label,
            "source": "padben",
            "domain": "unknown",
            "provenance": "human" if label == 0 else "ai_humanized",
            "generator_family": "human" if label == 0 else "unknown",
            "editor_family": "none" if label == 0 else "padben_humanizer",
            "transformation_family": "none" if label == 0 else "deep_paraphrase",
            "split": "train",
            "sealed": False,
            "train_eligible": True,
            "parent_id": None,
            "source_fields": {"padben_idx": str(row["idx"])},
        }))
    rng = random.Random(config.seed)
    selected: list[V4Record] = []
    for label in (0, 1):
        candidates = by_label[label]
        if len(candidates) < config.padben_samples_per_class:
            raise ValueError(f"PADBen label {label} has {len(candidates)} usable rows; need {config.padben_samples_per_class}")
        selected.extend(rng.sample(candidates, config.padben_samples_per_class))
    return selected


def _beemo_record(record: BeemoRecord, split: str, index: int, raw_ai_id: str | None) -> V4Record:
    identifier = f"beemo:{record.group_id}:{record.variant}:{index}"
    if record.provenance == "human":
        provenance, generator, editor, transformation, parent_id = "human", "human", "none", "none", None
    elif record.provenance == "raw_ai":
        provenance, generator, editor, transformation, parent_id = "raw_ai", "unknown", "none", "none", None
    elif record.provenance == "expert_edited_ai":
        provenance, generator, editor, transformation, parent_id = "expert_edited_ai", "unknown", "human_expert", "style_rewrite", raw_ai_id
    else:
        provenance, generator, editor, transformation, parent_id = "llm_edited_ai", "unknown", "llm_editor", "style_rewrite", raw_ai_id
    return V4Record.from_mapping({
        "id": identifier,
        "lineage_id": f"beemo:{record.group_id}",
        "text": record.text,
        "label": record.label,
        "source": "beemo",
        "domain": "unknown",
        "provenance": provenance,
        "generator_family": generator,
        "editor_family": editor,
        "transformation_family": transformation,
        "split": split,
        "sealed": False,
        "train_eligible": True,
        "parent_id": parent_id,
        "source_fields": {"beemo_prompt_id": record.group_id, "beemo_variant": record.variant},
    })


def _beemo_records(records: list[BeemoRecord], split: str) -> list[V4Record]:
    by_group: dict[str, list[BeemoRecord]] = {}
    for record in records:
        by_group.setdefault(record.group_id, []).append(record)
    output: list[V4Record] = []
    for group_id in sorted(by_group):
        group = by_group[group_id]
        raw_index = next((index for index, record in enumerate(group) if record.provenance == "raw_ai"), None)
        raw_ai_id = None if raw_index is None else f"beemo:{group_id}:{group[raw_index].variant}:{raw_index}"
        output.extend(_beemo_record(record, split, index, raw_ai_id) for index, record in enumerate(group))
    return output


def prepare_v4_control_dataset(
    padben_rows: list[dict[str, object]],
    beemo_records: list[BeemoRecord],
    output_dir: Path,
    config: V4ControlDataConfig = V4ControlDataConfig(),
) -> dict[str, object]:
    """Create the V4 control dataset from existing, non-sealed source data."""
    beemo_splits = v3_group_split(beemo_records, seed=config.seed)
    records = _padben_records(padben_rows, config)
    for split in ("train", "development", "calibration"):
        records.extend(_beemo_records(beemo_splits[split], split))
    source_metadata = {
        "purpose": "V4 control manifest from eligible PADBen and Beemo sources",
        "padben_samples_per_class": config.padben_samples_per_class,
        "selection_seed": config.seed,
        "split_rule": "PADBen train only; Beemo prompt_id atomic split",
    }
    return write_v4_dataset(records, output_dir, source_metadata)
