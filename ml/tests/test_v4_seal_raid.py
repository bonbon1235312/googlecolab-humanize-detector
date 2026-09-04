import json
from pathlib import Path

import pytest

from humanized_detector.v4_seal_raid import seal_raid_rows


def _row(identifier: str, source_id: str, model: str, attack: str, generation: str) -> dict[str, object]:
    return {
        "id": identifier,
        "source_id": source_id,
        "adv_source_id": f"adv:{source_id}",
        "model": model,
        "attack": attack,
        "domain": "abstracts",
        "generation": generation,
    }


def test_sealer_writes_a_hash_pinned_raid_derived_cohort(tmp_path: Path) -> None:
    rows = [
        _row("human-1", "source-1", "human", "none", "Human one."),
        _row("gpt-1", "source-1", "gpt-4", "paraphrase", "AI one."),
        _row("human-2", "source-2", "human", "none", "Human two."),
        _row("gpt-2", "source-2", "gpt-4", "paraphrase", "AI two."),
    ]

    report = seal_raid_rows(rows, tmp_path, target_pairs=1, seed=23, revision="pinned-revision", sealed_at="2026-09-04T12:00:00Z")

    metadata = json.loads((tmp_path / "metadata_manifest.json").read_text(encoding="utf-8"))
    assert report["split_counts"] == {"sealed_test": 2}
    assert metadata["source"]["revision"] == "pinned-revision"
    assert len(metadata["source"]["source_snapshot_sha256"]) == 64
    assert "Human one." not in (tmp_path / "metadata_manifest.json").read_text(encoding="utf-8")


def test_sealer_refuses_to_overwrite_an_existing_sealed_manifest(tmp_path: Path) -> None:
    rows = [
        _row("human-1", "source-1", "human", "none", "Human one."),
        _row("gpt-1", "source-1", "gpt-4", "paraphrase", "AI one."),
    ]
    seal_raid_rows(rows, tmp_path, target_pairs=1, seed=23, revision="pinned-revision", sealed_at="2026-09-04T12:00:00Z")

    with pytest.raises(FileExistsError, match="metadata_manifest"):
        seal_raid_rows(rows, tmp_path, target_pairs=1, seed=23, revision="pinned-revision", sealed_at="2026-09-04T12:00:00Z")
