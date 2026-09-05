from humanized_detector.v5_1_train import V51RunConfig, estimated_total_seconds


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
