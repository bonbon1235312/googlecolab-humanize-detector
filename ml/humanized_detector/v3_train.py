"""Source-balanced training for the V3 multi-window ablation suite."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import FusedMultiWindowClassifier, ModelConfig, MultiWindowClassifier, StructuralOnlyClassifier
from .tokenizer import load_tokenizer, train_tokenizer
from .train import smooth_binary_labels
from .v3_evaluate import evaluate_binary
from .v3_features import FEATURE_NAMES, FeatureNormalizer, extract_structural_features


@dataclass(frozen=True)
class V3TrainingResult:
    checkpoint: Path
    metrics_path: Path
    normalizer_path: Path


def load_v3_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_token_windows(token_ids: list[int], max_tokens: int, window_count: int = 3) -> list[list[int]]:
    """Select up to beginning/middle/end windows and pad to a fixed shape."""
    if max_tokens <= 0 or window_count <= 0:
        raise ValueError("max_tokens and window_count must be positive")
    if len(token_ids) <= max_tokens:
        starts = [0]
    else:
        starts = list(dict.fromkeys((0, (len(token_ids) - max_tokens) // 2, len(token_ids) - max_tokens)))
    windows = [token_ids[start : start + max_tokens] for start in starts]
    windows = [window + [0] * (max_tokens - len(window)) for window in windows]
    return (windows + [[0] * max_tokens for _ in range(window_count)])[:window_count]


def source_label_weights(rows: list[dict[str, object]]) -> list[float]:
    """Give every non-empty (source, binary-label) training stratum equal mass."""
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["source"]), int(row["label"]))
        counts[key] = counts.get(key, 0) + 1
    return [1.0 / counts[(str(row["source"]), int(row["label"]))] for row in rows]


def is_eligible_checkpoint(metrics: dict[str, object]) -> bool:
    """Select by ranking quality; operating thresholds belong to calibration."""
    roc_auc = metrics.get("roc_auc")
    return isinstance(roc_auc, (int, float))


def _checkpoint_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    if not is_eligible_checkpoint(metrics):
        return (float("-inf"), float("-inf"), float("-inf"))
    return (float(metrics["roc_auc"]), float(metrics.get("pr_auc") or float("-inf")), -float(metrics["human_fpr"]))


class V3EncodedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, object]], tokenizer_dir: Path, max_tokens: int, normalizer: FeatureNormalizer) -> None:
        tokenizer = load_tokenizer(tokenizer_dir)
        self.windows = [make_token_windows(tokenizer.encode(str(row["text"])).ids, max_tokens) for row in rows]
        raw_features = np.asarray([extract_structural_features(str(row["text"])) for row in rows])
        self.features = normalizer.transform(raw_features).astype(np.float32)
        self.labels = [int(row["label"]) for row in rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.tensor(self.windows[index], dtype=torch.long), torch.tensor(self.features[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)


def _create_model(config: ModelConfig, variant: str) -> nn.Module:
    if variant == "text_mean":
        return MultiWindowClassifier(config, pooling="mean")
    if variant == "text_attention":
        return MultiWindowClassifier(config, pooling="attention")
    if variant == "structural":
        return StructuralOnlyClassifier(len(FEATURE_NAMES))
    if variant == "fusion_concat":
        return FusedMultiWindowClassifier(config, len(FEATURE_NAMES), pooling="attention", gated=False)
    if variant == "fusion_gated":
        return FusedMultiWindowClassifier(config, len(FEATURE_NAMES), pooling="attention", gated=True)
    raise ValueError(f"unknown V3 variant: {variant}")


def _logits(model: nn.Module, variant: str, windows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if variant.startswith("text_"):
        return model(windows)
    if variant == "structural":
        return model(features)
    return model(windows, features)


def _predict(model: nn.Module, variant: str, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for windows, features, batch_labels in loader:
            logits = _logits(model, variant, windows.to(device), features.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def train_v3_model(train_file: Path, development_file: Path, artifact_dir: Path, config: ModelConfig, variant: str, epochs: int = 6, batch_size: int = 64, learning_rate: float = 3e-5, weight_decay: float = 0.01, label_smoothing: float = 0.1) -> V3TrainingResult:
    """Train one predeclared V3 ablation and retain its best development-F1 checkpoint."""
    torch.manual_seed(20260903)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_v3_jsonl(train_file)
    development_rows = load_v3_jsonl(development_file)
    tokenizer_dir = train_tokenizer(train_file, artifact_dir / "tokenizer", config.vocab_size)
    config = replace(config, vocab_size=len(load_tokenizer(tokenizer_dir).get_vocab()))
    train_feature_values = np.asarray([extract_structural_features(str(row["text"])) for row in train_rows])
    normalizer = FeatureNormalizer.fit(train_feature_values)
    normalizer_path = artifact_dir / "feature_normalizer.json"
    normalizer_path.write_text(json.dumps({"feature_names": FEATURE_NAMES, "mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist()}, indent=2), encoding="utf-8")
    np.savez_compressed(artifact_dir / "structural_feature_cache.npz", train_ids=np.asarray([str(row["id"]) for row in train_rows]), train_features=train_feature_values, development_ids=np.asarray([str(row["id"]) for row in development_rows]), development_features=np.asarray([extract_structural_features(str(row["text"])) for row in development_rows]))
    train_dataset = V3EncodedDataset(train_rows, tokenizer_dir, config.max_tokens, normalizer)
    development_dataset = V3EncodedDataset(development_rows, tokenizer_dir, config.max_tokens, normalizer)
    sampler = WeightedRandomSampler(source_label_weights(train_rows), num_samples=len(train_rows), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    development_loader = DataLoader(development_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(config, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    checkpoint = artifact_dir / "model.pt"
    metrics_path = artifact_dir / "development_metrics.json"
    for epoch in range(1, epochs + 1):
        model.train()
        for windows, features, labels in train_loader:
            optimizer.zero_grad()
            logits = _logits(model, variant, windows.to(device), features.to(device))
            loss = loss_fn(logits, smooth_binary_labels(labels.to(device), label_smoothing))
            loss.backward()
            optimizer.step()
        labels, probabilities = _predict(model, variant, development_loader, device)
        metrics = evaluate_binary(labels, probabilities)
        print(f"Epoch {epoch}/{epochs}: development_roc_auc={metrics['roc_auc']!s} human_fpr={metrics['human_fpr']!s}")
        candidate_key = _checkpoint_key(metrics)
        if candidate_key > best_key:
            best_key = candidate_key
            torch.save({"model_config": config.__dict__, "variant": variant, "state_dict": model.state_dict()}, checkpoint)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        scheduler.step()
    if not checkpoint.exists():
        raise RuntimeError("development data did not contain both binary classes")
    return V3TrainingResult(checkpoint, metrics_path, normalizer_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("text_mean", "text_attention", "structural", "fusion_concat", "fusion_gated"), required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    result = train_v3_model(args.data_dir / "train.jsonl", args.data_dir / "development.jsonl", args.artifacts_dir, ModelConfig(vocab_size=4_000), args.variant, args.epochs, args.batch_size, args.lr, args.weight_decay)
    print(f"Saved checkpoint: {result.checkpoint}")
    print(f"Saved development metrics: {result.metrics_path}")
    print(f"Saved feature normalizer: {result.normalizer_path}")


if __name__ == "__main__":
    main()
