"""Resumable curriculum continuation for the H1 humanizer release model."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import ModelConfig
from .train import smooth_binary_labels
from .v3_features import FeatureNormalizer
from .v3_train import V3EncodedDataset, _create_model, _logits, _predict
from .v4_prepare import load_v4_partition
from .v5_preflight import inspect_h1_inputs
from .v5_sampling import curriculum_mix, hierarchical_weights
from .v5_selection import humanizer_development_report, humanizer_selection_key


_VARIANT = "fusion_concat"
_SEED = 20260904


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_normalizer(path: Path) -> FeatureNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FeatureNormalizer(mean=np.asarray(payload["mean"], dtype=np.float32), scale=np.asarray(payload["scale"], dtype=np.float32))


def _checkpoint_payload(
    model: nn.Module,
    config: ModelConfig,
    epoch: int,
    report: dict[str, object],
    source_sha256: str,
    training_config: dict[str, object],
) -> dict[str, object]:
    return {
        "model_config": config.__dict__,
        "variant": _VARIANT,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "development_report": report,
        "source_checkpoint_sha256": source_sha256,
        "training_config": training_config,
    }


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, minimum_lr: float, peak_lr: float
) -> torch.optim.lr_scheduler.LambdaLR:
    minimum_scale = min(1.0, minimum_lr / peak_lr)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return minimum_scale + (1.0 - minimum_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _write_calibration_predictions(
    rows: list[dict[str, object]], model: nn.Module, config: ModelConfig, normalizer: FeatureNormalizer,
    tokenizer_dir: Path, output_path: Path, batch_size: int, device: torch.device,
) -> None:
    dataset = V3EncodedDataset(rows, tokenizer_dir, config.max_tokens, normalizer)
    labels, probabilities = _predict(model, _VARIANT, DataLoader(dataset, batch_size=batch_size), device)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, label, probability in zip(rows, labels, probabilities, strict=True):
            handle.write(json.dumps({"id": str(row["id"]), "label": int(label), "probability": float(probability)}, separators=(",", ":")) + "\n")


def continue_h1(
    data_dir: Path,
    source_artifacts: Path,
    output_dir: Path,
    epochs: int = 4,
    batch_size: int = 64,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.02,
    warmup_fraction: float = 0.05,
    grad_clip_norm: float = 1.0,
    amp: bool = True,
    progress_every: int = 100,
    resume: bool = False,
) -> dict[str, object]:
    """Continue a complete V4.8 checkpoint through the fixed H1 curriculum."""
    if epochs <= 0 or batch_size <= 0 or progress_every <= 0:
        raise ValueError("epochs, batch_size, and progress_every must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    preflight = inspect_h1_inputs(data_dir, source_artifacts, output_dir, allow_existing=resume)
    last_path = output_dir / "last_state.pt"
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"resume state is missing: {last_path}")

    torch.manual_seed(_SEED)
    source_payload = torch.load(source_artifacts / "model.pt", map_location="cpu", weights_only=True)
    config = ModelConfig(**source_payload["model_config"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    if not resume:
        shutil.copytree(source_artifacts / "tokenizer", tokenizer_dir)
        shutil.copy2(source_artifacts / "feature_normalizer.json", output_dir / "feature_normalizer.json")
    normalizer = _load_normalizer(output_dir / "feature_normalizer.json")
    train_rows = load_v4_partition(data_dir, "train")
    development_rows = load_v4_partition(data_dir, "development")
    calibration_rows = load_v4_partition(data_dir, "calibration")
    train_dataset = V3EncodedDataset(train_rows, tokenizer_dir, config.max_tokens, normalizer)
    development_loader = DataLoader(V3EncodedDataset(development_rows, tokenizer_dir, config.max_tokens, normalizer), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(config, _VARIANT).to(device)
    model.load_state_dict(source_payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = round(total_steps * warmup_fraction)
    scheduler = _cosine_scheduler(optimizer, total_steps, warmup_steps, 1e-6, learning_rate)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = nn.BCEWithLogitsLoss()
    training_config: dict[str, object] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "grad_clip_norm": grad_clip_norm,
        "amp": amp_enabled,
        "seed": _SEED,
        "curriculum": {str(epoch): curriculum_mix(epoch) for epoch in range(1, epochs + 1)},
    }

    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "model.pt"
    start_epoch = 1
    best_epoch = 0
    completed_epoch = 0
    epochs_without_improvement = 0
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state["source_checkpoint_sha256"] != preflight["source_checkpoint_sha256"]:
            raise ValueError("resume state source checkpoint does not match")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        start_epoch = int(state["completed_epoch"]) + 1
        completed_epoch = int(state["completed_epoch"])
        best_epoch = int(state["best_epoch"])
        best_report = state["best_report"]
        best_key = tuple(float(value) for value in state["best_key"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
    else:
        labels, probabilities = _predict(model, _VARIANT, development_loader, device)
        best_report = humanizer_development_report(development_rows, probabilities)
        best_key = humanizer_selection_key(best_report)
        _atomic_torch_save(_checkpoint_payload(model, config, 0, best_report, str(preflight["source_checkpoint_sha256"]), training_config), best_path)
        history_path.write_text(json.dumps({"epoch": 0, "training_loss": None, **best_report}, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"Epoch 0/{epochs}: macro_edit_auc={best_key[0]:.6f} expert_auc={best_report['subtypes']['expert_edited_ai']['roc_auc']:.6f} llm_auc={best_report['subtypes']['llm_edited_ai']['roc_auc']:.6f}", flush=True)
        _write_json(output_dir / "run_manifest.json", {"preflight": preflight, "training_config": training_config})

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        mix = curriculum_mix(epoch)
        generator = torch.Generator().manual_seed(_SEED + epoch)
        sampler = WeightedRandomSampler(hierarchical_weights(train_rows, mix), num_samples=len(train_dataset), replacement=True, generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        model.train()
        total_loss = 0.0
        for step, (windows, features, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = _logits(model, _VARIANT, windows.to(device), features.to(device))
                loss = loss_fn(logits, smooth_binary_labels(labels.to(device), label_smoothing))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite H1 loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            if step % progress_every == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs} step {step}/{len(train_loader)} loss={total_loss / step:.5f} lr={optimizer.param_groups[0]['lr']:.7g}", flush=True)

        dev_labels, dev_probabilities = _predict(model, _VARIANT, development_loader, device)
        report = humanizer_development_report(development_rows, dev_probabilities)
        candidate_key = humanizer_selection_key(report)
        improved = candidate_key > best_key
        if improved:
            best_key = candidate_key
            best_report = report
            best_epoch = epoch
            epochs_without_improvement = 0
            _atomic_torch_save(_checkpoint_payload(model, config, epoch, report, str(preflight["source_checkpoint_sha256"]), training_config), best_path)
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - epoch_started
        row = {"epoch": epoch, "training_loss": total_loss / len(train_loader), "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed, "improved": improved, "curriculum_mix": mix, **report}
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        completed_epoch = epoch
        last_state: dict[str, object] = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "completed_epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_report": best_report,
            "best_key": best_key,
            "epochs_without_improvement": epochs_without_improvement,
            "source_checkpoint_sha256": preflight["source_checkpoint_sha256"],
            "training_config": training_config,
        }
        _atomic_torch_save(last_state, last_path)
        print(f"Epoch {epoch}/{epochs}: macro_edit_auc={candidate_key[0]:.6f} expert_auc={report['subtypes']['expert_edited_ai']['roc_auc']:.6f} llm_auc={report['subtypes']['llm_edited_ai']['roc_auc']:.6f} improved={improved} saved_last_state=True", flush=True)
        if epoch >= 3 and epochs_without_improvement >= 2:
            print("Early stopping: no primary improvement for two completed epochs.", flush=True)
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_payload["state_dict"])
    model.to(device)
    _write_json(output_dir / "development_metrics.json", best_report)
    _write_calibration_predictions(calibration_rows, model, config, normalizer, tokenizer_dir, output_dir / "calibration_predictions.jsonl", batch_size, device)
    summary = {
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "best_macro_edit_auc": best_key[0],
        "best_expert_auc": best_report["subtypes"]["expert_edited_ai"]["roc_auc"],
        "best_llm_auc": best_report["subtypes"]["llm_edited_ai"]["roc_auc"],
        "checkpoint": str(best_path),
        "resume_checkpoint": str(last_path),
        "calibration_predictions": str(output_dir / "calibration_predictions.jsonl"),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue the V4.8 humanizer through the fixed H1 curriculum.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = continue_h1(
        args.data_dir, args.source_artifacts, args.artifacts_dir, args.epochs, args.batch_size,
        args.lr, args.weight_decay, args.label_smoothing, args.warmup_fraction, args.grad_clip_norm,
        not args.no_amp, args.progress_every, args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
