from humanized_detector.v5_1_train import V51RunConfig, copy_frozen_tokenizer, estimated_total_seconds


def test_timing_estimate_scales_one_balanced_epoch_to_full_two_stage_run() -> None:
    config = V51RunConfig(base_epochs=6, curriculum_epochs=4)
    assert estimated_total_seconds(12.5, config) == 125.0


def test_timing_estimate_rejects_non_positive_epoch_duration() -> None:
    config = V51RunConfig(base_epochs=6, curriculum_epochs=4)
    try:
        estimated_total_seconds(0.0, config)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive duration must be rejected")


def test_copy_frozen_tokenizer_preserves_the_v4_artifact_files(tmp_path) -> None:
    source = tmp_path / "v4-tokenizer"
    source.mkdir()
    (source / "vocab.json").write_text('{"<pad>": 0}', encoding="utf-8")
    (source / "merges.txt").write_text('#version: 0.2\n', encoding="utf-8")

    destination = copy_frozen_tokenizer(source, tmp_path / "v5-tokenizer")

    assert (destination / "vocab.json").read_text(encoding="utf-8") == '{"<pad>": 0}'
    assert (destination / "merges.txt").is_file()
