"""V6 provenance-first adapter for the Apache-2.0 MAGE corpus."""

from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Sequence

from .v6_beemo import V6BeemoCandidate, write_beemo_candidates
from .v6_manifest import _length_bucket


def _origin_from_source(source: str) -> str:
    if source.endswith("_human"):
        return "human"
    if "_machine_" in source:
        return "ai"
    raise ValueError(f"MAGE source has no declared author provenance: {source!r}")


def adapt_mage_rows(rows: Sequence[Mapping[str, object]]) -> list[V6BeemoCandidate]:
    """Normalize raw MAGE rows and reject source/label contradictions."""
    output: list[V6BeemoCandidate] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or "").strip()
        source = str(row.get("source") or "").strip()
        if not text or not source:
            raise ValueError(f"MAGE row {index} lacks text or source provenance")
        origin = _origin_from_source(source)
        try:
            mage_label = int(row["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"MAGE row {index} has no valid declared label") from error
        expected_mage_label = 1 if origin == "human" else 0
        if mage_label != expected_mage_label:
            raise ValueError(f"MAGE source provenance conflicts with label at row {index}")
        domain = source.split("_", 1)[0]
        output.append(V6BeemoCandidate(
            id=f"mage:{index}",
            lineage_id=f"mage:{source}:{index}",
            parent_lineage_id=None,
            text=text,
            dataset="mage",
            domain=domain,
            source_family="mage",
            template_id="not_applicable",
            origin=origin,
            edit_actor="not_applicable",
            editor_family="not_applicable",
            generator_family="human" if origin == "human" else f"mage:{source}",
            transformation="untouched" if origin == "human" else "raw_generation",
            sequence_length_bucket=_length_bucket(text),
        ))
    return output


def main() -> None:
    parser = ArgumentParser(description="Normalize a MAGE CSV into unsplit V6 candidates.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = adapt_mage_rows(rows)
    print(json.dumps({"candidates": write_beemo_candidates(args.output_jsonl, candidates), "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
