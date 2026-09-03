"""Colab-friendly model training and ONNX export."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import ModelConfig, TinyTransformerClassifier
from .tokenizer import load_tokenizer, train_tokenizer


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    metrics_path: Path


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [str(row["text"]) for row in rows], [int(row["label"]) for row in rows]


class EncodedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: list[str], labels: list[int], tokenizer_dir: Path, max_tokens: int) -> None:
        tokenizer = load_tokenizer(tokenizer_dir)
        self.tokens = [tokenizer.encode(text).ids[:max_tokens] for text in texts]
        self.labels = labels
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokens[index] + [0] * (self.max_tokens - len(self.tokens[index]))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.float32)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)), "confusion_matrix": confusion_matrix(labels, predicted).tolist()}


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); labels: list[float] = []; probabilities: list[float] = []
    with torch.no_grad():
        for ids, batch_labels in loader:
            probabilities.extend(torch.sigmoid(model(ids.to(device))).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def smooth_binary_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Move binary targets slightly toward 0.5 to reduce overconfident fits."""
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def train_model(
    train_file: Path,
    validation_file: Path,
    artifact_dir: Path,
    config: ModelConfig,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    scheduler: str = "cosine",
    label_smoothing: float = 0.1,
) -> TrainingResult:
    torch.manual_seed(20260903); artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = train_tokenizer(train_file, artifact_dir / "tokenizer", config.vocab_size)
    config = replace(config, vocab_size=len(load_tokenizer(tokenizer_dir).get_vocab()))
    train_texts, train_labels = load_jsonl(train_file); val_texts, val_labels = load_jsonl(validation_file)
    train_loader = DataLoader(EncodedDataset(train_texts, train_labels, tokenizer_dir, config.max_tokens), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EncodedDataset(val_texts, val_labels, tokenizer_dir, config.max_tokens), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformerClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if scheduler == "cosine" else None
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1 = -1.0; checkpoint = artifact_dir / "model.pt"; metrics_path = artifact_dir / "validation_metrics.json"
    for epoch in range(1, epochs + 1):
        model.train()
        for ids, labels in train_loader:
            optimizer.zero_grad()
            smoothed = smooth_binary_labels(labels.to(device), label_smoothing)
            loss = loss_fn(model(ids.to(device)), smoothed); loss.backward(); optimizer.step()
        labels, probabilities = _predict(model, val_loader, device); metrics = _metrics(labels, probabilities)
        print(f"Epoch {epoch}/{epochs}: validation_f1={metrics['f1']:.4f}")
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"]); torch.save({"model_config": config.__dict__, "state_dict": model.state_dict()}, checkpoint); metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if lr_scheduler is not None:
            lr_scheduler.step()
    return TrainingResult(checkpoint, metrics_path)


def run_experiment(data_dir: Path, artifact_dir: Path, epochs: int = 6, batch_size: int = 64, learning_rate: float = 3e-5, scheduler: str = "cosine", dropout: float = 0.15, weight_decay: float = 0.01, label_smoothing: float = 0.1) -> dict[str, Path]:
    config = ModelConfig(vocab_size=4_000, dropout=dropout)
    result = train_model(data_dir / "train.jsonl", data_dir / "validation.jsonl", artifact_dir, config, epochs, batch_size, learning_rate, weight_decay, scheduler, label_smoothing)
    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=True); model_config = ModelConfig(**payload["model_config"]); model = TinyTransformerClassifier(model_config); model.load_state_dict(payload["state_dict"])
    texts, labels = load_jsonl(data_dir / "test.jsonl"); loader = DataLoader(EncodedDataset(texts, labels, artifact_dir / "tokenizer", model_config.max_tokens), batch_size=batch_size)
    actual, probabilities = _predict(model, loader, torch.device("cpu")); test_path = artifact_dir / "test_metrics.json"; test_path.write_text(json.dumps(_metrics(actual, probabilities), indent=2), encoding="utf-8")
    destination = artifact_dir / "model.onnx"; torch.onnx.export(model, torch.ones((1, model_config.max_tokens), dtype=torch.long), destination, input_names=["input_ids"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    return {"checkpoint": result.checkpoint, "validation_metrics": result.metrics_path, "test_metrics": test_path, "onnx_model": destination}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--artifacts-dir", type=Path, required=True); parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--lr", type=float, default=3e-5); parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine"); parser.add_argument("--dropout", type=float, default=0.15); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--label-smoothing", type=float, default=0.1)
    args = parser.parse_args()
    for name, path in run_experiment(args.data_dir, args.artifacts_dir, args.epochs, args.batch_size, args.lr, args.scheduler, args.dropout, args.weight_decay, args.label_smoothing).items(): print(f"Saved {name}: {path}")


if __name__ == "__main__": main()
