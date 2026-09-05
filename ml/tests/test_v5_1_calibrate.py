from humanized_detector.v5_1_calibrate import calibration_summary


def test_calibration_summary_records_each_requested_human_fpr() -> None:
    report = calibration_summary([0, 0, 1, 1], [0.1, 0.4, 0.8, 0.9], (0.01, 0.05))

    assert set(report["operating_points"]) == {"1pct", "5pct"}
