import torch

from humanized_detector.v5_1_model import V51FusedClassifier, V51ModelConfig


def test_v51_model_returns_one_logit_per_row() -> None:
    model = V51FusedClassifier(V51ModelConfig(vocab_size=32, hidden_size=32, heads=4, layers=1, max_tokens=8), 23)
    logits = model(torch.ones((2, 3, 8), dtype=torch.long), torch.zeros((2, 23)))
    assert logits.shape == (2,)


def test_v51_model_handles_an_empty_window() -> None:
    model = V51FusedClassifier(V51ModelConfig(vocab_size=32, hidden_size=32, heads=4, layers=1, max_tokens=8), 23).eval()
    windows = torch.tensor([[[1, 2, 0, 0, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]])
    logits = model(windows, torch.zeros((1, 23)))
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_v51_model_uses_film_and_cross_window_attention() -> None:
    model = V51FusedClassifier(V51ModelConfig(vocab_size=32, hidden_size=32, heads=4, layers=1, max_tokens=8), 23)
    assert model.cross_window_attention.batch_first is True
    assert model.film[-1].out_features == 64
