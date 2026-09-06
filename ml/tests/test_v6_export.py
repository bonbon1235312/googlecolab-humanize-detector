import json

from humanized_detector.v6_export import export_v6_partitions
from humanized_detector.v6_manifest import V6Record


def record(identifier: str, split: str, label: int) -> V6Record:
    origin = "ai" if label else "human"
    return V6Record.from_mapping(
        {
            "id": identifier,
            "lineage_id": f"lineage:{identifier}",
            "parent_lineage_id": None,
            "text": f"A documented {origin} training record {identifier}. " * 8,
            "dataset": "mage",
            "domain": "essay",
            "source_family": f"mage-{identifier}",
            "template_id": "not_applicable",
            "origin": origin,
            "edit_actor": "not_applicable",
            "editor_family": "not_applicable",
            "generator_family": "mage" if label else "human",
            "transformation": "raw_generation" if label else "untouched",
            "split": split,
            "sampling_weight": 0.5 if label else 1.0,
        }
    )


def test_export_v6_partitions_preserves_split_mapping_weight_and_manifest_identity(tmp_path) -> None:
    records = [record("train", "train", 1), record("dev", "selection_dev", 0), record("cal", "calibration", 1)]
    manifest = {
        "manifest_sha256": "a" * 64,
        "sampling_policy_sha256": "b" * 64,
        "dedup_policy_sha256": "c" * 64,
        "pretrain_audit": {"TRAINING AUTHORIZED": "YES"},
        "lineage_map": [row.metadata() for row in records],
    }

    contract = export_v6_partitions(records, manifest, tmp_path)

    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(line) for line in (tmp_path / "development.jsonl").read_text(encoding="utf-8").splitlines()]
    calibration = [json.loads(line) for line in (tmp_path / "calibration.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train[0]["sampling_weight"] == 0.5
    assert dev[0]["id"] == "dev"
    assert calibration[0]["id"] == "cal"
    assert contract["manifest_sha256"] == "a" * 64
    assert not (tmp_path / "sealed_test.jsonl").exists()
