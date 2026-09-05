"""Artifact dispatch shared by V5.1 comparison and calibration tools."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import ModelConfig
from .v3_features import FEATURE_NAMES, FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _predict
from .v5_1_model import V51FusedClassifier, V51ModelConfig


V51_VARIANT = "v5_1_film_cross_attention"


def create_artifact_model(payload: Mapping[str, object]) -> nn.Module:
    """Construct a model from either the deployed V4.8 or V5.1 checkpoint."""
    variant = str(payload["variant"])
    config = payload["model_config"]
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint model_config must be an object")
    if variant == V51_VARIANT:
        return V51FusedClassifier(V51ModelConfig(**config), len(FEATURE_NAMES))
    return _create_model(ModelConfig(**config), variant)


def score_artifact_rows(artifact_dir: Path, rows: Sequence[dict[str, object]], batch_size: int = 64) -> np.ndarray:
    """Score aligned rows without using labels for model loading or inference."""
    checkpoint_path = artifact_dir / "model.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = create_artifact_model(payload)
    model.load_state_dict(payload["state_dict"])
    config = payload["model_config"]
    assert isinstance(config, Mapping)
    max_tokens = int(config["max_tokens"])
    data = json.loads((artifact_dir / "feature_normalizer.json").read_text(encoding="utf-8"))
    if data["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("feature normalizer does not match the V5.1 feature schema")
    normalizer = FeatureNormalizer(np.asarray(data["mean"], dtype=np.float32), np.asarray(data["scale"], dtype=np.float32))
    dataset = V3EncodedDataset(list(rows), artifact_dir / "tokenizer", max_tokens, normalizer)
    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if str(payload["variant"]) == V51_VARIANT:
        model.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for windows, features, _ in loader:
                predictions.extend(torch.sigmoid(model(windows.to(device), features.to(device))).cpu().tolist())
        return np.asarray(predictions, dtype=float)
    _, probabilities = _predict(model, str(payload["variant"]), loader, device)
    return probabilities
