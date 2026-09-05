from dataclasses import asdict

from humanized_detector.v5_1_inference import create_artifact_model
from humanized_detector.v5_1_model import V51FusedClassifier, V51ModelConfig


def test_create_artifact_model_dispatches_the_v5_1_checkpoint() -> None:
    config = V51ModelConfig(vocab_size=50, hidden_size=32, heads=4, layers=1)
    model = create_artifact_model({"variant": "v5_1_film_cross_attention", "model_config": asdict(config)})

    assert isinstance(model, V51FusedClassifier)
