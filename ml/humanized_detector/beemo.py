"""Beemo expansion and prompt-disjoint evaluation splits."""

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeemoRecord:
    text: str
    label: int
    group_id: str
    variant: str
    provenance: str = "unknown"


def _llm_variants(value: str, model: str) -> list[tuple[str, str]]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [(f"{model}_{key}", item[key]) for item in parsed if isinstance(item, dict) for key in ("P1", "P2", "P3") if isinstance(item.get(key), str) and item[key].strip()]


def load_beemo_rows(rows: list[dict[str, object]]) -> list[BeemoRecord]:
    records: list[BeemoRecord] = []
    for row in rows:
        group_id = str(row["prompt_id"])
        human = str(row.get("human_output") or "").strip()
        raw_ai = str(row.get("model_output") or "").strip()
        expert = str(row.get("human_edits") or "").strip()
        if human:
            records.append(BeemoRecord(human, 0, group_id, "human", "human"))
        if raw_ai:
            records.append(BeemoRecord(raw_ai, 1, group_id, "raw_ai", "raw_ai"))
        if expert:
            records.append(BeemoRecord(expert, 1, group_id, "expert_edit", "expert_edited_ai"))
        for key, model in (("llama-3.1-70b_edits", "llama-3.1-70b"), ("gpt-4o_edits", "gpt-4o")):
            records.extend(BeemoRecord(text, 1, group_id, variant, "llm_edited_ai") for variant, text in _llm_variants(str(row.get(key) or ""), model))
    return records


def load_beemo_parquet(path: Path) -> list[BeemoRecord]:
    import pyarrow.parquet as parquet

    return load_beemo_rows(parquet.read_table(path).to_pylist())


def group_split(records: list[BeemoRecord], seed: int = 20260903, train_fraction: float = 0.70, development_fraction: float = 0.15) -> dict[str, list[BeemoRecord]]:
    """Split complete prompt groups; no prompt's variants cross a partition."""
    groups: dict[str, list[BeemoRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    test_count = max(1, round(len(group_ids) * (1 - train_fraction - development_fraction)))
    development_count = max(1, round(len(group_ids) * development_fraction))
    partitions = {"train": group_ids[test_count + development_count :], "development": group_ids[test_count : test_count + development_count], "test": group_ids[:test_count]}
    return {name: [record for group_id in ids for record in groups[group_id]] for name, ids in partitions.items()}


def v3_group_split(records: list[BeemoRecord], seed: int = 20260903, train_fraction: float = 0.70, development_fraction: float = 0.15) -> dict[str, list[BeemoRecord]]:
    """Make prompt-lineage-safe train, development, and calibration partitions."""
    if train_fraction <= 0 or development_fraction <= 0 or train_fraction + development_fraction >= 1:
        raise ValueError("train and development fractions must be positive and leave room for calibration")
    groups: dict[str, list[BeemoRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    calibration_count = max(1, round(len(group_ids) * (1 - train_fraction - development_fraction)))
    development_count = max(1, round(len(group_ids) * development_fraction))
    partitions = {
        "calibration": group_ids[:calibration_count],
        "development": group_ids[calibration_count : calibration_count + development_count],
        "train": group_ids[calibration_count + development_count :],
    }
    return {name: [record for group_id in ids for record in groups[group_id]] for name, ids in partitions.items()}


def write_records(splits: dict[str, list[BeemoRecord]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps({"text": record.text, "label": record.label, "group_id": record.group_id, "variant": record.variant, "provenance": record.provenance}, ensure_ascii=False, separators=(",", ":")) + "\n")
