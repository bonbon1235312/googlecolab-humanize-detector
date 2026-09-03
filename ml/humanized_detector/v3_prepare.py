"""Build the provenance-preserving mixed-domain V3 JSONL partitions."""

import json
import random
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .beemo import BeemoRecord, v3_group_split
from .v3_data import audit_split_boundaries


@dataclass(frozen=True)
class V3DataConfig:
    seed: int = 20260903
    padben_samples_per_class: int = 15_000


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _padben_rows(records: list[dict[str, object]], config: V3DataConfig) -> list[dict[str, object]]:
    classes: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    seen: set[str] = set()
    for record in records:
        if not {"idx", "sentence", "label"}.issubset(record):
            raise ValueError("PADBen records must contain idx, sentence, and label")
        label = int(record["label"])
        if label not in classes:
            raise ValueError(f"PADBen label must be 0 or 1, got {label!r}")
        text = _normalise(str(record["sentence"]))
        canonical = " ".join(text.casefold().split())
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        provenance = "human" if label == 0 else "padben_humanized_ai"
        classes[label].append({"id": f"padben:{record['idx']}", "text": text, "label": label, "source": "padben", "group_id": f"padben:{record['idx']}", "provenance": provenance})
    rng = random.Random(config.seed)
    selected: list[dict[str, object]] = []
    for label, rows in classes.items():
        if len(rows) < config.padben_samples_per_class:
            raise ValueError(f"PADBen label {label} has {len(rows)} usable rows; need {config.padben_samples_per_class}")
        selected.extend(rng.sample(rows, config.padben_samples_per_class))
    return selected


def _beemo_rows(records: list[BeemoRecord], split_name: str) -> list[dict[str, object]]:
    return [{"id": f"beemo:{record.group_id}:{record.variant}:{index}", "text": record.text, "label": record.label, "source": "beemo", "group_id": record.group_id, "provenance": record.provenance} for index, record in enumerate(records)]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_v3_dataset(padben_records: list[dict[str, object]], beemo_records: list[BeemoRecord], output_dir: Path, config: V3DataConfig = V3DataConfig()) -> dict[str, object]:
    """Prepare source-labelled train, development, and calibration partitions."""
    beemo_splits = v3_group_split(beemo_records, seed=config.seed)
    splits = {"train": _padben_rows(padben_records, config) + _beemo_rows(beemo_splits["train"], "train"), "development": _beemo_rows(beemo_splits["development"], "development"), "calibration": _beemo_rows(beemo_splits["calibration"], "calibration")}
    audit_split_boundaries(splits)
    rng = random.Random(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rows in splits.values():
        rng.shuffle(rows)
    for name, rows in splits.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    source_counts = {name: {source: sum(row["source"] == source for row in rows) for source in ("padben", "beemo")} for name, rows in splits.items()}
    report = {"config": asdict(config), "split_counts": {name: len(rows) for name, rows in splits.items()}, "source_counts": source_counts, "beemo_groups": {name: len({row["group_id"] for row in rows if row["source"] == "beemo"}) for name, rows in splits.items()}, "boundary_audit": "exact and 5-word-shingle near-duplicate checks passed"}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
