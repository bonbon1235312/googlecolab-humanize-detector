import json
from pathlib import Path

import numpy as np

from humanized_detector.model import ModelConfig
from humanized_detector.v3_train import is_eligible_checkpoint, make_token_windows, source_label_weights, train_v3_model


def test_make_token_windows_selects_beginning_middle_and_end() -> None:
    windows = make_token_windows(list(range(1, 13)), max_tokens=4, window_count=3)

    assert windows == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


def test_source_label_weights_balance_each_nonempty_stratum() -> None:
    rows = [{"source": "padben", "label": 0}, {"source": "padben", "label": 0}, {"source": "padben", "label": 1}, {"source": "beemo", "label": 0}, {"source": "beemo", "label": 1}, {"source": "beemo", "label": 1}]
    weights = source_label_weights(rows)
    totals = {}
    for row, weight in zip(rows, weights, strict=True):
        totals[row["source"], row["label"]] = totals.get((row["source"], row["label"]), 0.0) + weight

    np.testing.assert_allclose(list(totals.values()), [1.0, 1.0, 1.0, 1.0])


def test_checkpoint_selection_requires_development_roc_auc_and_human_fpr_guard() -> None:
    assert is_eligible_checkpoint({"roc_auc": 0.7, "human_fpr": 0.04}, max_human_fpr=0.05)
    assert not is_eligible_checkpoint({"roc_auc": 0.7, "human_fpr": 0.06}, max_human_fpr=0.05)
    assert not is_eligible_checkpoint({"roc_auc": None, "human_fpr": 0.01}, max_human_fpr=0.05)


def test_train_v3_model_writes_best_checkpoint_and_feature_normalizer(tmp_path: Path) -> None:
    rows = [{"id": f"h{index}", "text": f"human authored response letter {chr(97 + index)}", "label": 0, "source": "padben", "group_id": f"h{index}", "provenance": "human"} for index in range(4)] + [{"id": f"a{index}", "text": f"generated paraphrased response letter {chr(97 + index)}", "label": 1, "source": "beemo", "group_id": f"a{index}", "provenance": "raw_ai"} for index in range(4)]
    for name in ("train", "development"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = train_v3_model(tmp_path / "train.jsonl", tmp_path / "development.jsonl", tmp_path / "artifacts", ModelConfig(vocab_size=300, hidden_size=24, heads=4, layers=1, max_tokens=8), variant="text_mean", epochs=1, batch_size=2)

    assert result.checkpoint.exists()
    assert result.normalizer_path.exists()
