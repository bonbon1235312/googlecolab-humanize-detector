"""Seal a reproducible RAID-derived paraphrase cohort for V4 evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .v4_prepare import write_v4_dataset
from .v4_raid import collect_raid_paraphrase_candidates, select_raid_paraphrase_pairs


RAID_SOURCE_LOCATOR = "https://huggingface.co/datasets/liamdugan/raid"


def seal_raid_rows(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    target_pairs: int,
    seed: int,
    revision: str,
    sealed_at: str,
) -> dict[str, object]:
    """Write a pinned, source-family-safe RAID paraphrase cohort without scoring it."""
    if (output_dir / "metadata_manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing sealed metadata_manifest: {output_dir / 'metadata_manifest.json'}")
    candidates, source_snapshot_sha256 = collect_raid_paraphrase_candidates(rows)
    records = select_raid_paraphrase_pairs(candidates, target_pairs=target_pairs, seed=seed)
    if len(records) != target_pairs * 2:
        raise ValueError(f"RAID yielded {len(records) // 2} eligible source families; need {target_pairs}")
    return write_v4_dataset(records, output_dir, {
        "source_locator": RAID_SOURCE_LOCATOR,
        "revision": revision,
        "source_snapshot_sha256": source_snapshot_sha256,
        "row_selection_rule": "one human attack=none and one non-human attack=paraphrase row per source_id; deterministic SHA-256 source-family rank",
        "selection_seed": seed,
        "sealed_at": sealed_at,
        "dataset_config": "raid",
        "dataset_split": "train",
        "claim_scope": "RAID-derived unseen-source paraphrase cohort; not RAID's official hidden test split",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a RAID-derived V4 paraphrase cohort without evaluating a model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pairs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--revision", type=str)
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = args.revision or HfApi().dataset_info("liamdugan/raid", repo_type="dataset").sha
    rows = load_dataset("liamdugan/raid", "raid", split="train", streaming=True, revision=revision)
    print(f"Streaming pinned RAID train source at revision {revision}; no model evaluation will run.")

    def progress_rows() -> Iterable[Mapping[str, object]]:
        for index, row in enumerate(rows, start=1):
            if index % 100_000 == 0:
                print(f"Scanned {index:,} source rows...")
            yield row

    report = seal_raid_rows(
        progress_rows(),
        args.output_dir,
        target_pairs=args.target_pairs,
        seed=args.seed,
        revision=revision,
        sealed_at=datetime.now(UTC).isoformat(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
