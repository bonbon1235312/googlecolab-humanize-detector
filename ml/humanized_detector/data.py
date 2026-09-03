"""PADBen preparation with reproducible, data-safe row-level splits."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    seed: int = 20260903
    samples_per_class: int = 12_000
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


@dataclass(frozen=True)
class DatasetReport:
    seed: int
    input_rows: int
    blank_rows: int
    duplicate_rows: int
    class_counts: dict[int, int]
    split_counts: dict[str, int]
    split_strategy: str


def _normalise(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _split(rows: list[dict[str, object]], config: DataConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    test_count = max(1, round(len(rows) * (1 - config.train_fraction - config.validation_fraction)))
    validation_count = max(1, round(len(rows) * config.validation_fraction))
    return rows[test_count + validation_count :], rows[test_count : test_count + validation_count], rows[:test_count]


def prepare_records(records: list[dict[str, object]], output_dir: Path, config: DataConfig = DataConfig()) -> DatasetReport:
    """Validate PADBen rows and write deterministic, balanced JSONL splits."""
    selected: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    seen: set[str] = set()
    blank_rows = duplicate_rows = 0
    for record in records:
        if not {"idx", "sentence", "label"}.issubset(record):
            raise ValueError("PADBen records must contain idx, sentence, and label")
        label = record["label"]
        if label not in (0, 1):
            raise ValueError(f"PADBen label must be 0 or 1, got {label!r}")
        text = _normalise(str(record["sentence"]))
        if not text.strip():
            blank_rows += 1
            continue
        if text in seen:
            duplicate_rows += 1
            continue
        seen.add(text)
        selected[int(label)].append({"id": str(record["idx"]), "text": text, "label": int(label)})
    for label, rows in selected.items():
        if len(rows) < config.samples_per_class:
            raise ValueError(f"label {label} has {len(rows)} usable rows; need {config.samples_per_class}")
    rng = random.Random(config.seed)
    chosen = {label: rng.sample(rows, config.samples_per_class) for label, rows in selected.items()}
    splits = {"train": [], "validation": [], "test": []}
    for rows in chosen.values():
        rng.shuffle(rows)
        train, validation, test = _split(rows, config)
        splits["train"].extend(train)
        splits["validation"].extend(validation)
        splits["test"].extend(test)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    report = DatasetReport(config.seed, len(records), blank_rows, duplicate_rows, {0: len(chosen[0]), 1: len(chosen[1])}, {name: len(rows) for name, rows in splits.items()}, "row-level stratified; PADBen has no source-pair identifier")
    (output_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    return report
