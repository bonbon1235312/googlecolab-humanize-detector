import json
from pathlib import Path

import torch

from humanized_detector.model import ModelConfig, TinyTransformerClassifier
from humanized_detector.train import train_model


def test_model_returns_one_logit_per_text() -> None:
    model = TinyTransformerClassifier(ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=16))
    assert model(torch.ones((3, 16), dtype=torch.long)).shape == (3,)


def test_train_model_exports_checkpoint_and_metrics(tmp_path: Path) -> None:
    rows = [{"text": "genuine human writing varies naturally", "label": 0}, {"text": "another human authored this response", "label": 0}, {"text": "paraphrased ai text follows generated patterns", "label": 1}, {"text": "humanized generated response retains artefacts", "label": 1}]
    for name in ("train", "validation"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = train_model(tmp_path / "train.jsonl", tmp_path / "validation.jsonl", tmp_path / "artifacts", ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=16), epochs=1, batch_size=2)
    assert result.checkpoint.exists()
    assert result.metrics_path.exists()
