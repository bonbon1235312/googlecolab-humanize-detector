"""Timed two-stage V5.1 training without sealed-data access."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .tokenizer import load_tokenizer, train_tokenizer
from .train import smooth_binary_labels
from .v3_features import FEATURE_NAMES, FeatureNormalizer, extract_structural_features
from .v3_train import V3EncodedDataset, source_label_weights
from .v4_prepare import load_v4_partition
from .v5_1_model import V51FusedClassifier, V51ModelConfig
from .v5_sampling import curriculum_mix, hierarchical_weights
from .v5_selection import humanizer_development_report, humanizer_selection_key


_SEED = 20260905
_VARIANT = "v5_1_film_cross_attention"


@dataclass(frozen=True)
class V51RunConfig:
    base_epochs: int = 6
    curriculum_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    label_smoothing: float = 0.02
    warmup_fraction: float = 0.05
    grad_clip_norm: float = 1.0


def estimated_total_seconds(first_base_epoch_seconds: float, config: V51RunConfig) -> float:
    """Conservative linear runtime estimate used before committing the job."""
    if first_base_epoch_seconds <= 0:
        raise ValueError("first_base_epoch_seconds must be positive")
    return float(first_base_epoch_seconds * (config.base_epochs + config.curriculum_epochs))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_normalizer(path: Path) -> FeatureNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FeatureNormalizer(mean=np.asarray(payload["mean"], dtype=np.float32), scale=np.asarray(payload["scale"], dtype=np.float32))


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    with torch.no_grad():
        for windows, features, batch_labels in loader:
            scores.extend(torch.sigmoid(model(windows.to(device), features.to(device))).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, learning_rate: float) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return (1e-6 / learning_rate) + (1 - 1e-6 / learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _checkpoint(model: nn.Module, model_config: V51ModelConfig, epoch: int, report: dict[str, object], stage: str, config: V51RunConfig) -> dict[str, object]:
    return {"model_config": asdict(model_config), "variant": _VARIANT, "state_dict": model.state_dict(), "epoch": epoch, "development_report": report, "stage": stage, "training_config": asdict(config)}


def _train_stage(
    model: nn.Module, train_dataset: V3EncodedDataset, development_rows: list[dict[str, object]], development_loader: DataLoader,
    output_dir: Path, model_config: V51ModelConfig, run_config: V51RunConfig, stage: str, epochs: int,
    weights_for_epoch, start_epoch: int = 1, timing_only: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
    steps_per_epoch = math.ceil(len(train_dataset) / run_config.batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = _scheduler(optimizer, total_steps, round(total_steps * run_config.warmup_fraction), run_config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    _, baseline_probabilities = _predict(model, development_loader, device)
    best_report = humanizer_development_report(development_rows, baseline_probabilities)
    best_key = humanizer_selection_key(best_report)
    best_epoch = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint(model, model_config, 0, best_report, stage, run_config), output_dir / "model.pt")
    history = [{"epoch": 0, "training_loss": None, "elapsed_seconds": None, "improved": True, **best_report}]
    completed = 0
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(_SEED + epoch + (0 if stage == "base" else 10_000))
        sampler = WeightedRandomSampler(weights_for_epoch(epoch), num_samples=len(train_dataset), replacement=True, generator=generator)
        loader = DataLoader(train_dataset, batch_size=run_config.batch_size, sampler=sampler)
        model.train()
        total_loss = 0.0
        for windows, features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows.to(device), features.to(device))
            loss = loss_fn(logits, smooth_binary_labels(labels.to(device), run_config.label_smoothing))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.grad_clip_norm)
            optimizer.step(); scheduler.step()
            total_loss += float(loss.detach().cpu())
        _, probabilities = _predict(model, development_loader, device)
        report = humanizer_development_report(development_rows, probabilities)
        key = humanizer_selection_key(report)
        improved = key > best_key
        if improved:
            best_key, best_report, best_epoch = key, report, epoch
            torch.save(_checkpoint(model, model_config, epoch, report, stage, run_config), output_dir / "model.pt")
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "training_loss": total_loss / len(loader), "elapsed_seconds": elapsed, "improved": improved, **report}
        history.append(row); completed = epoch
        print(f"{stage.title()} epoch {epoch}/{epochs}: macro_edit_auc={key[0]:.6f} expert_auc={report['subtypes']['expert_edited_ai']['roc_auc']:.6f} llm_auc={report['subtypes']['llm_edited_ai']['roc_auc']:.6f} elapsed_seconds={elapsed:.1f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "completed_epoch": completed, "best_epoch": best_epoch, "best_key": best_key}, output_dir / "last_state.pt")
        if timing_only:
            break
    (output_dir / "history.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in history), encoding="utf-8")
    return {"best_epoch": best_epoch, "completed_epochs": completed, "best_report": best_report, "first_epoch_seconds": history[1]["elapsed_seconds"] if len(history) > 1 else None}


def train_v51(data_dir: Path, output_dir: Path, run_config: V51RunConfig = V51RunConfig(), stage: str = "base", timing_only: bool = False) -> dict[str, object]:
    """Run one V5.1 stage. Base must be completed before curriculum starts."""
    if stage not in {"base", "curriculum"}:
        raise ValueError("stage must be 'base' or 'curriculum'")
    if run_config.base_epochs <= 0 or run_config.curriculum_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    torch.manual_seed(_SEED)
    train_rows = load_v4_partition(data_dir, "train")
    development_rows = load_v4_partition(data_dir, "development")
    base_dir = output_dir / "base"
    curriculum_dir = output_dir / "curriculum"
    if stage == "base":
        tokenizer_dir = train_tokenizer(data_dir / "train.jsonl", base_dir / "tokenizer", 4_000)
        model_config = V51ModelConfig(vocab_size=len(load_tokenizer(tokenizer_dir).get_vocab()))
        raw_features = np.asarray([extract_structural_features(str(row["text"])) for row in train_rows])
        normalizer = FeatureNormalizer.fit(raw_features)
        _write_json(base_dir / "feature_normalizer.json", {"feature_names": FEATURE_NAMES, "mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist()})
        model = V51FusedClassifier(model_config, len(FEATURE_NAMES))
        weights = source_label_weights(train_rows)
        result = _train_stage(model, V3EncodedDataset(train_rows, tokenizer_dir, model_config.max_tokens, normalizer), development_rows, DataLoader(V3EncodedDataset(development_rows, tokenizer_dir, model_config.max_tokens, normalizer), batch_size=run_config.batch_size), base_dir, model_config, run_config, "base", run_config.base_epochs, lambda _epoch: weights, timing_only=timing_only)
        if timing_only and result["first_epoch_seconds"] is not None:
            result["estimated_total_seconds"] = estimated_total_seconds(float(result["first_epoch_seconds"]), run_config)
        _write_json(base_dir / "run_manifest.json", {"stage": "base", "timing_only": timing_only, "config": asdict(run_config), **result})
        return result
    payload = torch.load(base_dir / "model.pt", map_location="cpu", weights_only=True)
    model_config = V51ModelConfig(**payload["model_config"])
    if not (base_dir / "tokenizer" / "vocab.json").is_file():
        raise FileNotFoundError("V5.1 curriculum requires a completed base tokenizer")
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_dir / "tokenizer", curriculum_dir / "tokenizer", dirs_exist_ok=True)
    shutil.copy2(base_dir / "feature_normalizer.json", curriculum_dir / "feature_normalizer.json")
    normalizer = _load_normalizer(base_dir / "feature_normalizer.json")
    model = V51FusedClassifier(model_config, len(FEATURE_NAMES)); model.load_state_dict(payload["state_dict"])
    result = _train_stage(model, V3EncodedDataset(train_rows, base_dir / "tokenizer", model_config.max_tokens, normalizer), development_rows, DataLoader(V3EncodedDataset(development_rows, base_dir / "tokenizer", model_config.max_tokens, normalizer), batch_size=run_config.batch_size), curriculum_dir, model_config, run_config, "curriculum", run_config.curriculum_epochs, lambda epoch: hierarchical_weights(train_rows, curriculum_mix(epoch)))
    _write_json(curriculum_dir / "run_manifest.json", {"stage": "curriculum", "base_checkpoint": str(base_dir / "model.pt"), "config": asdict(run_config), **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("base", "curriculum"), required=True); parser.add_argument("--base-epochs", type=int, default=6); parser.add_argument("--curriculum-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--lr", type=float, default=3e-5); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--label-smoothing", type=float, default=0.02); parser.add_argument("--warmup-fraction", type=float, default=0.05); parser.add_argument("--grad-clip-norm", type=float, default=1.0); parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    config = V51RunConfig(args.base_epochs, args.curriculum_epochs, args.batch_size, args.lr, args.weight_decay, args.label_smoothing, args.warmup_fraction, args.grad_clip_norm)
    print(json.dumps(train_v51(args.data_dir, args.artifacts_dir, config, args.stage, args.timing_only), indent=2, default=str))


if __name__ == "__main__":
    main()
