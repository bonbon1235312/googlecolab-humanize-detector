import json
from pathlib import Path

import torch

from humanized_detector.model import FusedMultiWindowClassifier, ModelConfig, MultiWindowClassifier, StructuralOnlyClassifier, TinyTransformerClassifier
from humanized_detector.train import smooth_binary_labels, train_model


def test_binary_label_smoothing_moves_targets_toward_half() -> None:
    actual = smooth_binary_labels(torch.tensor([0.0, 1.0]), 0.1)
    torch.testing.assert_close(actual, torch.tensor([0.05, 0.95]))


def test_model_returns_one_logit_per_text() -> None:
    model = TinyTransformerClassifier(ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=16, dropout=0.15))
    assert model(torch.ones((3, 16), dtype=torch.long)).shape == (3,)
    assert model.encoder.layers[0].dropout.p == 0.15


def test_multi_window_model_pools_shared_encoder_windows() -> None:
    model = MultiWindowClassifier(ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), pooling="mean")
    windows = torch.ones((2, 3, 8), dtype=torch.long)

    assert model(windows).shape == (2,)


def test_multi_window_attention_ignores_all_padding_window() -> None:
    torch.manual_seed(3)
    model = MultiWindowClassifier(ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), pooling="attention").eval()
    one_window = torch.ones((1, 1, 8), dtype=torch.long)
    padded_window = torch.cat([one_window, torch.zeros((1, 1, 8), dtype=torch.long)], dim=1)

    torch.testing.assert_close(model(one_window), model(padded_window))


def test_structural_only_model_returns_one_logit_per_text() -> None:
    model = StructuralOnlyClassifier(feature_count=6, hidden_size=12)

    assert model(torch.ones((3, 6))).shape == (3,)


def test_fusion_models_return_one_logit_per_text() -> None:
    windows = torch.ones((2, 3, 8), dtype=torch.long)
    features = torch.ones((2, 6))
    config = ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8)

    assert FusedMultiWindowClassifier(config, feature_count=6, gated=False)(windows, features).shape == (2,)
    assert FusedMultiWindowClassifier(config, feature_count=6, gated=True)(windows, features).shape == (2,)


def test_train_model_exports_checkpoint_and_metrics(tmp_path: Path) -> None:
    rows = [{"text": "genuine human writing varies naturally", "label": 0}, {"text": "another human authored this response", "label": 0}, {"text": "paraphrased ai text follows generated patterns", "label": 1}, {"text": "humanized generated response retains artefacts", "label": 1}]
    for name in ("train", "validation"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = train_model(tmp_path / "train.jsonl", tmp_path / "validation.jsonl", tmp_path / "artifacts", ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=16), epochs=1, batch_size=2, learning_rate=3e-5, weight_decay=0.01, scheduler="cosine", label_smoothing=0.1)
    assert result.checkpoint.exists()
    assert result.metrics_path.exists()
