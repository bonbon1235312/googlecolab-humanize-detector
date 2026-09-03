import pytest

from humanized_detector.v3_data import audit_split_boundaries


def test_boundary_audit_rejects_exact_text_in_different_splits() -> None:
    splits = {
        "train": [{"id": "a", "text": "The same passage appears twice."}],
        "development": [{"id": "b", "text": "The same passage appears twice."}],
    }

    with pytest.raises(ValueError, match="exact duplicate"):
        audit_split_boundaries(splits)


def test_boundary_audit_rejects_near_duplicate_in_different_splits() -> None:
    stem = " ".join(f"word{index}" for index in range(20))
    altered = " ".join([*(f"word{index}" for index in range(19)), "replacement"])
    splits = {
        "train": [{"id": "a", "text": stem}],
        "calibration": [{"id": "b", "text": altered}],
    }

    with pytest.raises(ValueError, match="near duplicate"):
        audit_split_boundaries(splits)


def test_boundary_audit_allows_related_rows_inside_one_partition() -> None:
    audit = audit_split_boundaries({"train": [{"id": "a", "text": "one two three four five six"}, {"id": "b", "text": "one two three four five seven"}]})

    assert audit.checked_records == 2
