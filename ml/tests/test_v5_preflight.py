import json
from pathlib import Path

import pytest
import torch

from humanized_detector.model import ModelConfig
from humanized_detector.v3_train import _create_model
from humanized_detector.v5_preflight import inspect_h1_inputs


def test_preflight_rejects_missing_train_before_gpu_work(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        inspect_h1_inputs(tmp_path, tmp_path / "source", tmp_path / "output")


def test_preflight_validates_checkpoint_and_never_reads_sealed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    source = tmp_path / "source"
    output = tmp_path / "output"
    data_dir.mkdir()
    source.mkdir()
    for role, lineage in (("train", "train-family"), ("development", "dev-family"), ("calibration", "cal-family")):
        rows = [
            {"id": f"{role}-ph", "lineage_id": f"{lineage}-ph", "text": "PADBen human", "label": 0, "source": "padben", "provenance": "human"},
            {"id": f"{role}-pa", "lineage_id": f"{lineage}-pa", "text": "PADBen AI", "label": 1, "source": "padben", "provenance": "ai_humanized"},
            {"id": f"{role}-bh", "lineage_id": lineage, "text": "Beemo human", "label": 0, "source": "beemo", "provenance": "human"},
            {"id": f"{role}-be", "lineage_id": lineage, "text": "expert edit", "label": 1, "source": "beemo", "provenance": "expert_edited_ai"},
            {"id": f"{role}-bl", "lineage_id": lineage, "text": "LLM edit", "label": 1, "source": "beemo", "provenance": "llm_edited_ai"},
            {"id": f"{role}-br", "lineage_id": lineage, "text": "raw AI", "label": 1, "source": "beemo", "provenance": "raw_ai"},
        ]
        (data_dir / f"{role}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (data_dir / "sealed_test.jsonl").write_text("invalid and forbidden", encoding="utf-8")
    tokenizer = source / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.json").write_text("{}", encoding="utf-8")
    (tokenizer / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (source / "feature_normalizer.json").write_text('{"mean": [0], "scale": [1]}', encoding="utf-8")
    config = ModelConfig(vocab_size=20, hidden_size=24, heads=4, layers=1, max_tokens=8, token_pooling="masked_mean")
    model = _create_model(config, "fusion_concat")
    torch.save({"model_config": config.__dict__, "variant": "fusion_concat", "state_dict": model.state_dict()}, source / "model.pt")

    report = inspect_h1_inputs(data_dir, source, output)

    assert report["partition_counts"] == {"calibration": 6, "development": 6, "train": 6}
    assert report["model"]["token_pooling"] == "masked_mean"
    assert report["model"]["parameter_count"] > 0


def test_preflight_rejects_lineage_overlap(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {"id": "x", "lineage_id": "shared", "text": "text", "label": 0, "source": "beemo", "provenance": "human"}
    for role in ("train", "development", "calibration"):
        (data_dir / f"{role}.jsonl").write_text(json.dumps({**row, "id": role}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lineage overlap"):
        inspect_h1_inputs(data_dir, tmp_path / "source", tmp_path / "output")


def test_h1_colab_notebook_uses_v48_source_resume_and_no_external_evaluation() -> None:
    notebook = Path(__file__).parents[2] / "notebooks" / "train_v5_h1.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in payload["cells"])

    assert payload["nbformat"] == 4
    assert "humanized_detector.v5_train" in source
    assert "/v4-artifacts/v4-8/masked_mean_base" in source
    assert "--resume" in source
    assert "humanized_detector.v4_calibrate" in source
    assert "Do not run GRADTEX" in source
    assert "v3_external" not in source
