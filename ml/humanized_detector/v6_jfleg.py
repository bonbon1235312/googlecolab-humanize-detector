"""V6 diagnostic-only adapter for the human-authored JFLEG correction corpus."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Sequence

from .v6_beemo import V6BeemoCandidate, write_beemo_candidates
from .v6_manifest import _length_bucket


def _candidate(*, identifier: str, lineage_id: str, text: str, edit_actor: str, editor_family: str, transformation: str) -> V6BeemoCandidate:
    return V6BeemoCandidate(
        id=identifier,
        lineage_id=lineage_id,
        parent_lineage_id=None if transformation == "untouched" else lineage_id,
        text=text,
        dataset="jfleg",
        domain="learner_english",
        source_family="jfleg",
        template_id="not_applicable",
        origin="human",
        edit_actor=edit_actor,
        editor_family=editor_family,
        generator_family="human",
        transformation=transformation,
        sequence_length_bucket=_length_bucket(text),
    )


def adapt_jfleg_rows(rows: Sequence[Mapping[str, object]], *, split_name: str) -> list[V6BeemoCandidate]:
    """Emit source and correction branches as human-origin records; never infer machine authorship."""
    if not split_name.strip():
        raise ValueError("JFLEG split name must be explicit")
    output: list[V6BeemoCandidate] = []
    for index, row in enumerate(rows):
        lineage_id = f"jfleg:{split_name}:{index}"
        source = str(row.get("sentence") or "").strip()
        corrections = row.get("corrections")
        if not isinstance(corrections, Sequence) or isinstance(corrections, (str, bytes)):
            corrections = []
        usable = [str(correction).strip() for correction in corrections if str(correction).strip()]
        if not source and not usable:
            raise ValueError(f"JFLEG row {index} has no usable human source or corrections")
        if source:
            output.append(_candidate(
                identifier=f"{lineage_id}:source",
                lineage_id=lineage_id,
                text=source,
                edit_actor="not_applicable",
                editor_family="not_applicable",
                transformation="untouched",
            ))
        for ref_index, correction in enumerate(usable):
            output.append(_candidate(
                identifier=f"{lineage_id}:human_correction:{ref_index}",
                lineage_id=lineage_id,
                text=correction,
                edit_actor="human_expert",
                editor_family=f"jfleg_human_ref{ref_index}",
                transformation="human_fluency_correction",
            ))
    return output


def main() -> None:
    parser = ArgumentParser(description="Normalize a JFLEG split into diagnostic-only V6 candidates.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    candidates = adapt_jfleg_rows(rows, split_name=args.split_name)
    print(json.dumps({"candidates": write_beemo_candidates(args.output_jsonl, candidates), "split": args.split_name, "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
