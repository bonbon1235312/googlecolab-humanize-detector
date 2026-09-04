import json
from pathlib import Path

from humanized_detector.v4_manifest import V4Record
import pytest

from humanized_detector.v4_prepare import load_v4_partition, write_v4_dataset


def make_row(
    identifier: str,
    lineage_id: str,
    text: str,
    split: str,
    *,
    sealed: bool = False,
    train_eligible: bool = True,
) -> dict[str, object]:
    return {
        "id": identifier,
        "lineage_id": lineage_id,
        "text": text,
        "text_sha256": "provided-by-source",
        "label": 0,
        "source": "example",
        "domain": "essay",
        "provenance": "human",
        "generator_family": "human",
        "editor_family": "none",
        "transformation_family": "none",
        "split": split,
        "sealed": sealed,
        "train_eligible": train_eligible,
        "parent_id": None,
    }


def test_writer_outputs_role_files_and_text_free_metadata(tmp_path: Path) -> None:
    records = [
        V4Record.from_mapping(make_row("train:1", "l1", "Train secret", "train")),
        V4Record.from_mapping(make_row("sealed:1", "l2", "Sealed secret", "sealed_test", sealed=True, train_eligible=False)),
    ]

    report = write_v4_dataset(records, tmp_path, {
        "source_locator": "https://example.test",
        "revision": "v1",
        "raw_download_sha256": "abc123",
        "row_selection_rule": "complete lineages only",
        "selection_seed": 7,
        "sealed_at": "2026-09-04T12:00:00Z",
    })

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "sealed_test.jsonl").exists()
    assert "Train secret" not in (tmp_path / "metadata_manifest.json").read_text(encoding="utf-8")
    assert report["metadata_manifest_sha256"]
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["checked_records"] == 2


def test_non_final_loader_refuses_to_load_a_sealed_partition(tmp_path: Path) -> None:
    (tmp_path / "sealed_test.jsonl").write_text('{"id":"sealed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sealed"):
        load_v4_partition(tmp_path, "sealed_test")


def test_writer_requires_reproducibility_metadata_for_a_sealed_source(tmp_path: Path) -> None:
    record = V4Record.from_mapping(make_row("sealed:1", "l2", "Sealed secret", "sealed_test", sealed=True, train_eligible=False))

    with pytest.raises(ValueError, match="raw_download_sha256"):
        write_v4_dataset([record], tmp_path, {"source_locator": "https://example.test", "revision": "v1"})


def test_v4_data_protocol_documents_sealed_manifest_rules() -> None:
    protocol = Path(__file__).parents[2] / "docs" / "v4-data-protocol.md"

    text = protocol.read_text(encoding="utf-8")
    assert "sealed_test" in text
    assert "GRADTEX" in text
    assert "metadata_manifest.json" in text
