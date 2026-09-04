from pathlib import Path

from humanized_detector.v4_control import load_v4_control_partitions, model_config_for_capacity


def test_control_loader_reads_only_nonsealed_roles(tmp_path: Path) -> None:
    for role in ("train", "development", "calibration"):
        (tmp_path / f"{role}.jsonl").write_text('{"id":"' + role + '"}\n', encoding="utf-8")
    (tmp_path / "sealed_test.jsonl").write_text('{"id":"secret"}\n', encoding="utf-8")

    loaded = load_v4_control_partitions(tmp_path)

    assert set(loaded) == {"train", "development", "calibration"}
    assert loaded["train"] == [{"id": "train"}]


def test_capacity_presets_keep_attention_dimensions_valid() -> None:
    small = model_config_for_capacity(4000, "5m")
    larger = model_config_for_capacity(4000, "12m")

    assert small.hidden_size == 192
    assert larger.hidden_size > small.hidden_size
    assert larger.hidden_size % larger.heads == 0
    assert larger.layers >= small.layers
