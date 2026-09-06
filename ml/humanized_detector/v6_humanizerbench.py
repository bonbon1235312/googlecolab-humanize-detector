"""V6 provenance-first adapter for published HumanizerBench cycles."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Sequence

from .v6_beemo import V6BeemoCandidate, write_beemo_candidates
from .v6_manifest import _length_bucket


def _value(row: Mapping[str, object], field: str) -> str:
    return str(row.get(field) or "").strip()


def _candidate(
    *,
    identifier: str,
    lineage_id: str,
    text: str,
    template_id: str,
    domain: str,
    generator_family: str,
    edit_actor: str,
    editor_family: str,
    transformation: str,
) -> V6BeemoCandidate:
    return V6BeemoCandidate(
        id=identifier,
        lineage_id=lineage_id,
        parent_lineage_id=lineage_id,
        text=text,
        dataset="humanizerbench",
        domain=domain,
        source_family="humanizerbench",
        template_id=template_id,
        origin="ai",
        edit_actor=edit_actor,
        editor_family=editor_family,
        generator_family=generator_family,
        transformation=transformation,
        sequence_length_bucket=_length_bucket(text),
    )


def adapt_humanizerbench_cycle(
    samples: Sequence[Mapping[str, object]], tests: Sequence[Mapping[str, object]], *, cycle: str
) -> list[V6BeemoCandidate]:
    """Normalize one published cycle without reading detector scores as training evidence."""
    if not cycle.strip():
        raise ValueError("HumanizerBench cycle must be explicit")
    sample_by_id = {_value(sample, "id"): sample for sample in samples if _value(sample, "id")}
    output: list[V6BeemoCandidate] = []
    emitted_raw: set[str] = set()
    for test_index, test in enumerate(tests):
        test_id = _value(test, "id")
        sample_id = _value(test, "sample_id")
        if not test_id or not sample_id:
            raise ValueError(f"HumanizerBench test {test_index} lacks id or sample_id")
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"HumanizerBench test {test_id} has missing source sample {sample_id}")
        template_id = _value(sample, "prompt_slug")
        domain = _value(sample, "category")
        generator_family = _value(sample, "source_model_slug")
        raw_text = _value(sample, "input_text")
        if not template_id or not domain or not generator_family or not raw_text:
            raise ValueError(f"HumanizerBench sample {sample_id} has incomplete provenance")
        declared_input = _value(test, "input_text")
        if declared_input and declared_input != raw_text:
            raise ValueError(f"HumanizerBench test {test_id} input disagrees with source sample {sample_id}")
        lineage_id = f"humanizerbench:{cycle}:{sample_id}"
        if sample_id not in emitted_raw:
            output.append(_candidate(
                identifier=f"{lineage_id}:raw_ai",
                lineage_id=lineage_id,
                text=raw_text,
                template_id=template_id,
                domain=domain,
                generator_family=generator_family,
                edit_actor="not_applicable",
                editor_family="not_applicable",
                transformation="raw_generation",
            ))
            emitted_raw.add(sample_id)
        editor_family = _value(test, "humanizer_slug")
        rewritten = _value(test, "output_text")
        if not editor_family or not rewritten:
            raise ValueError(f"HumanizerBench test {test_id} has incomplete editor provenance")
        output.append(_candidate(
            identifier=f"{lineage_id}:humanizer:{test_id}",
            lineage_id=lineage_id,
            text=rewritten,
            template_id=template_id,
            domain=domain,
            generator_family=generator_family,
            edit_actor="llm",
            editor_family=editor_family,
            transformation="humanizer_rewrite",
        ))
    return output


def main() -> None:
    parser = ArgumentParser(description="Normalize a published HumanizerBench cycle into unsplit V6 candidates.")
    parser.add_argument("--cycle-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads((args.cycle_dir / "samples.json").read_text(encoding="utf-8"))
    tests = json.loads((args.cycle_dir / "tests.json").read_text(encoding="utf-8"))
    cycle = args.cycle_dir.name
    candidates = adapt_humanizerbench_cycle(samples, tests, cycle=cycle)
    print(json.dumps({"candidates": write_beemo_candidates(args.output_jsonl, candidates), "cycle": cycle, "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
