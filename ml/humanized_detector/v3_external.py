"""One-time, scenario-level evaluation of the frozen GRADTEX Test C benchmark."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from .model import ModelConfig
from .v3_evaluate import evaluate_binary, metrics_by_field
from .v3_features import FEATURE_NAMES, FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _predict


def gradtex_test_c_rows(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Convert GRADTEX's HWT=1/MIX=0 convention to this project's human=0/AI=1 labels."""
    rows = []
    for record in records:
        if str(record["test_split"]) != "C":
            continue
        rows.append({
            "id": str(len(rows)),
            "text": str(record["text"]),
            "label": 1 - int(record["binary_label"]),
            "scenario": str(record["scenario"]),
            "scenario_family": str(record["scenario_family"]),
            "generator_model": str(record.get("generator_model") or "human"),
        })
    if not rows:
        raise ValueError("the benchmark contains no GRADTEX Test C records")
    return rows


def evaluate_gradtex_test_c(benchmark_file: Path, artifact_dir: Path) -> dict[str, object]:
    """Apply a calibrated selected checkpoint once to the frozen Test C parquet file."""
    calibration_path = artifact_dir / "calibration.json"
    checkpoint_path = artifact_dir / "model.pt"
    normalizer_path = artifact_dir / "feature_normalizer.json"
    tokenizer_dir = artifact_dir / "tokenizer"
    for path in (benchmark_file, calibration_path, checkpoint_path, normalizer_path, tokenizer_dir):
        if not path.exists():
            raise FileNotFoundError(f"required external-evaluation artifact is missing: {path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold = float(calibration["threshold"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    variant = str(payload["variant"])
    normalizer_data = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if normalizer_data["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("feature normalizer does not match the current V3 feature schema")
    normalizer = FeatureNormalizer(
        mean=np.asarray(normalizer_data["mean"], dtype=np.float32),
        scale=np.asarray(normalizer_data["scale"], dtype=np.float32),
    )
    rows = gradtex_test_c_rows(pq.read_table(benchmark_file).to_pylist())
    dataset = V3EncodedDataset(rows, tokenizer_dir, config.max_tokens, normalizer)
    loader = DataLoader(dataset, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(config, variant).to(device)
    model.load_state_dict(payload["state_dict"])
    labels, probabilities = _predict(model, variant, loader, device)
    result: dict[str, object] = {
        "benchmark": "GRADTEX Test C",
        "variant": variant,
        "calibrated_threshold": threshold,
        "aggregate": evaluate_binary(labels, probabilities, threshold),
        "per_scenario": metrics_by_field(rows, probabilities, "scenario", threshold),
        "per_scenario_family": metrics_by_field(rows, probabilities, "scenario_family", threshold),
    }
    (artifact_dir / "gradtex_test_c_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_gradtex_test_c(args.benchmark_file, args.artifacts_dir), indent=2))


if __name__ == "__main__":
    main()
