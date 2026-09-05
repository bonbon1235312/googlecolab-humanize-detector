import json
from pathlib import Path

import torch

from humanized_detector.model import ModelConfig
from humanized_detector.v3_train import train_v3_model
from humanized_detector.v5_train import continue_h1


def _rows(suffix: str) -> list[dict[str, object]]:
    return [
        {"id": f"ph-{suffix}", "lineage_id": f"ph-{suffix}", "text": f"natural human essay {suffix}", "label": 0, "source": "padben", "provenance": "human"},
        {"id": f"pa-{suffix}", "lineage_id": f"pa-{suffix}", "text": f"deeply paraphrased generated essay {suffix}", "label": 1, "source": "padben", "provenance": "ai_humanized"},
        {"id": f"bh-{suffix}", "lineage_id": f"bp-{suffix}", "text": f"genuine answer {suffix}", "label": 0, "source": "beemo", "provenance": "human"},
        {"id": f"br-{suffix}", "lineage_id": f"bp-{suffix}", "text": f"raw generated answer {suffix}", "label": 1, "source": "beemo", "provenance": "raw_ai"},
        {"id": f"be-{suffix}", "lineage_id": f"bp-{suffix}", "text": f"expert edited generated answer {suffix}", "label": 1, "source": "beemo", "provenance": "expert_edited_ai"},
        {"id": f"bl-{suffix}", "lineage_id": f"bp-{suffix}", "text": f"language model edited answer {suffix}", "label": 1, "source": "beemo", "provenance": "llm_edited_ai"},
    ]


def test_h1_continuation_preserves_source_and_writes_recoverable_bundle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source = tmp_path / "source"
    output = tmp_path / "h1"
    data_dir.mkdir()
    partitions = {"train": _rows("train"), "development": _rows("dev"), "calibration": _rows("cal")}
    for role, rows in partitions.items():
        (data_dir / f"{role}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    train_v3_model(
        data_dir / "train.jsonl", data_dir / "development.jsonl", source,
        ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8, token_pooling="masked_mean"),
        "fusion_concat", epochs=1, batch_size=2,
    )
    source_hash_before = (source / "model.pt").read_bytes()

    result = continue_h1(data_dir, source, output, epochs=1, batch_size=2, warmup_fraction=0.0, amp=False, progress_every=1)
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]

    assert (source / "model.pt").read_bytes() == source_hash_before
    assert result["best_epoch"] in {0, 1}
    assert payload["model_config"]["token_pooling"] == "masked_mean"
    assert [row["epoch"] for row in history] == [0, 1]
    last_state = torch.load(output / "last_state.pt", map_location="cpu", weights_only=True)
    assert "cuda_rng_state_all" in last_state
    assert (output / "calibration_predictions.jsonl").exists()
    assert (output / "tokenizer" / "vocab.json").exists()


def test_h1_resume_continues_after_last_completed_epoch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source = tmp_path / "source"
    output = tmp_path / "h1"
    data_dir.mkdir()
    for role, suffix in (("train", "train"), ("development", "dev"), ("calibration", "cal")):
        (data_dir / f"{role}.jsonl").write_text("\n".join(json.dumps(row) for row in _rows(suffix)) + "\n", encoding="utf-8")
    train_v3_model(
        data_dir / "train.jsonl", data_dir / "development.jsonl", source,
        ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8, token_pooling="masked_mean"),
        "fusion_concat", epochs=1, batch_size=2,
    )
    continue_h1(data_dir, source, output, epochs=1, batch_size=2, warmup_fraction=0.0, amp=False)

    result = continue_h1(data_dir, source, output, epochs=2, batch_size=2, warmup_fraction=0.0, amp=False, resume=True)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["completed_epoch"] == 2
    assert [row["epoch"] for row in history] == [0, 1, 2]
