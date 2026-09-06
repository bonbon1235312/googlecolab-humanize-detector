"""V6 provenance-first adapter for the Beemo source corpus."""

from __future__ import annotations

import ast
import hashlib
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .v6_manifest import _length_bucket


@dataclass(frozen=True)
class V6BeemoCandidate:
    id: str
    lineage_id: str
    parent_lineage_id: str | None
    text: str
    dataset: str
    domain: str
    source_family: str
    template_id: str
    origin: str
    edit_actor: str
    editor_family: str
    generator_family: str
    transformation: str
    sequence_length_bucket: str

    @property
    def label(self) -> int:
        return 0 if self.origin == "human" else 1

    def to_mapping(self, *, split: str, sampling_weight: float) -> dict[str, object]:
        return {
            "id": self.id,
            "lineage_id": self.lineage_id,
            "parent_lineage_id": self.parent_lineage_id,
            "text": self.text,
            "dataset": self.dataset,
            "domain": self.domain,
            "source_family": self.source_family,
            "template_id": self.template_id,
            "origin": self.origin,
            "edit_actor": self.edit_actor,
            "editor_family": self.editor_family,
            "generator_family": self.generator_family,
            "transformation": self.transformation,
            "sequence_length_bucket": self.sequence_length_bucket,
            "split": split,
            "sampling_weight": sampling_weight,
        }


def _text(row: Mapping[str, object], field: str) -> str:
    return str(row.get(field) or "").strip()


def _generator_family(row: Mapping[str, object]) -> str:
    for field in ("generator_family", "model_name", "model"):
        value = _text(row, field)
        if value and value.casefold() not in {"unknown", "tbd", "none", "n/a"}:
            return value
    raise ValueError("Beemo AI-origin row has no established generator family")


def _llm_outputs(value: object, editor: str) -> list[tuple[str, str]]:
    try:
        parsed = ast.literal_eval(str(value or ""))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    outputs: list[tuple[str, str]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            continue
        for prompt_variant in ("P1", "P2", "P3"):
            text = _text(item, prompt_variant)
            if text:
                outputs.append((f"{editor}:{prompt_variant}:{index}", text))
    return outputs


def _candidate(
    *,
    identifier: str,
    lineage_id: str,
    text: str,
    origin: str,
    edit_actor: str,
    editor_family: str,
    generator_family: str,
    transformation: str,
    parent_lineage_id: str | None,
) -> V6BeemoCandidate:
    return V6BeemoCandidate(
        id=identifier,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        text=text,
        dataset="beemo",
        domain="instruction_response",
        source_family="no_robots",
        template_id="not_applicable",
        origin=origin,
        edit_actor=edit_actor,
        editor_family=editor_family,
        generator_family=generator_family,
        transformation=transformation,
        sequence_length_bucket=_length_bucket(text),
    )


def adapt_beemo_rows(rows: Sequence[Mapping[str, object]]) -> list[V6BeemoCandidate]:
    """Emit unsplit V6 candidates from Beemo rows with provenance-derived labels."""
    output: list[V6BeemoCandidate] = []
    for source_index, row in enumerate(rows):
        prompt_id = _text(row, "prompt_id")
        if not prompt_id:
            raise ValueError(f"Beemo row {source_index} has no prompt_id")
        lineage_id = f"beemo:{prompt_id}"
        human = _text(row, "human_output")
        raw_ai = _text(row, "model_output")
        expert_edit = _text(row, "human_edits")
        if human:
            output.append(_candidate(
                identifier=f"{lineage_id}:human",
                lineage_id=lineage_id,
                text=human,
                origin="human",
                edit_actor="not_applicable",
                editor_family="not_applicable",
                generator_family="human",
                transformation="untouched",
                parent_lineage_id=None,
            ))
        if raw_ai or expert_edit or _text(row, "llama-3.1-70b_edits") or _text(row, "gpt-4o_edits"):
            generator = _generator_family(row)
        else:
            generator = "not_applicable"
        if raw_ai:
            output.append(_candidate(
                identifier=f"{lineage_id}:raw_ai",
                lineage_id=lineage_id,
                text=raw_ai,
                origin="ai",
                edit_actor="not_applicable",
                editor_family="not_applicable",
                generator_family=generator,
                transformation="raw_generation",
                parent_lineage_id=lineage_id,
            ))
        if expert_edit:
            output.append(_candidate(
                identifier=f"{lineage_id}:expert_edit",
                lineage_id=lineage_id,
                text=expert_edit,
                origin="ai",
                edit_actor="human_expert",
                editor_family="beemo_human_expert",
                generator_family=generator,
                transformation="expert_edit",
                parent_lineage_id=lineage_id,
            ))
        for field, editor in (("llama-3.1-70b_edits", "llama-3.1-70b"), ("gpt-4o_edits", "gpt-4o")):
            for variant, text in _llm_outputs(row.get(field), editor):
                output.append(_candidate(
                    identifier=f"{lineage_id}:llm_edit:{variant}",
                    lineage_id=lineage_id,
                    text=text,
                    origin="ai",
                    edit_actor="llm",
                    editor_family=editor,
                    generator_family=generator,
                    transformation="llm_edit",
                    parent_lineage_id=lineage_id,
                ))
    return output


def beemo_download_sha256(path: Path) -> str:
    """Hash the exact locally acquired Beemo source before any transformation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_beemo_candidates(path: Path, candidates: Sequence[V6BeemoCandidate]) -> int:
    """Write unsplit V6 candidate rows; only split assignment may add a split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            row = {
                "id": candidate.id,
                "lineage_id": candidate.lineage_id,
                "parent_lineage_id": candidate.parent_lineage_id,
                "text": candidate.text,
                "dataset": candidate.dataset,
                "domain": candidate.domain,
                "source_family": candidate.source_family,
                "template_id": candidate.template_id,
                "origin": candidate.origin,
                "edit_actor": candidate.edit_actor,
                "editor_family": candidate.editor_family,
                "generator_family": candidate.generator_family,
                "transformation": candidate.transformation,
                "sequence_length_bucket": candidate.sequence_length_bucket,
                "label": candidate.label,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(candidates)


def main() -> None:
    parser = ArgumentParser(description="Normalize a pinned Beemo parquet file into unsplit V6 candidates.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as parquet

    candidates = adapt_beemo_rows(parquet.read_table(args.input_parquet).to_pylist())
    print(json.dumps({
        "candidates": write_beemo_candidates(args.output_jsonl, candidates),
        "raw_download_sha256": beemo_download_sha256(args.input_parquet),
        "output": str(args.output_jsonl),
    }, indent=2))


if __name__ == "__main__":
    main()
