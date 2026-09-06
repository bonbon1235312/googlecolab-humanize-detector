"""Train the frozen V4.8 architecture from a manifest-authorized V6 export."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from .v3_train import explicit_sampling_weights, train_v3_model
from .v4_control import load_v4_control_partitions, model_config_for_capacity
from .v4_train import _write_calibration_predictions


def load_v6_training_contract(path: Path) -> dict[str, object]:
    """Load the pretrain-approved V6 contract and reject any sealed-data claim."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol_version") != "v6":
        raise ValueError("training contract must declare protocol_version v6")
    for field in ("manifest_sha256", "sampling_policy_sha256", "dedup_policy_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"training contract is missing established {field}")
    if payload.get("sealed_data_loaded") is not False:
        raise ValueError("V6 training contract must prove sealed data was not loaded")
    return payload


def train_v6_base(
    data_dir: Path,
    artifacts_dir: Path,
    *,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
) -> dict[str, object]:
    """Run V4.8's 5M masked-mean fusion recipe using frozen V6 sampling weights."""
    contract = load_v6_training_contract(data_dir / "v6_training_contract.json")
    partitions = load_v4_control_partitions(data_dir)
    expected_counts = contract.get("partition_counts")
    if isinstance(expected_counts, Mapping):
        actual_counts = {name: len(partitions[name]) for name in ("train", "development", "calibration")}
        if actual_counts != dict(expected_counts):
            raise ValueError("V6 partition counts do not match the authorized training contract")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "v6_training_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = replace(model_config_for_capacity(4_000, "5m"), token_pooling="masked_mean")
    result = train_v3_model(
        data_dir / "train.jsonl",
        data_dir / "development.jsonl",
        artifacts_dir,
        config,
        "fusion_concat",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        sampling_weights=explicit_sampling_weights(partitions["train"]),
        checkpoint_provenance=contract,
    )
    (artifacts_dir / "tokenizer" / "training_corpus.txt").unlink(missing_ok=True)
    calibration_predictions = _write_calibration_predictions(artifacts_dir, partitions["calibration"], batch_size)
    return {
        "variant": "fusion_concat",
        "capacity": "5m",
        "token_pooling": "masked_mean",
        "checkpoint": str(result.checkpoint),
        "development_metrics": str(result.metrics_path),
        "calibration_predictions": str(calibration_predictions),
        "manifest_sha256": contract["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V4.8-DATA-BASE only from an authorized V6 export.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(train_v6_base(
        args.data_dir,
        args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    ), indent=2))


if __name__ == "__main__":
    main()
