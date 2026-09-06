import pytest

from humanized_detector.v6_mage import adapt_mage_rows


def test_mage_adapter_uses_declared_source_and_checks_the_dataset_label() -> None:
    candidates = adapt_mage_rows([
        {"text": "A human document.", "label": 1, "src": "cmv_human"},
        {"text": "A machine document.", "label": 0, "src": "roct_machine_continuation_flan_t5_large"},
    ])

    assert [(candidate.origin, candidate.label) for candidate in candidates] == [("human", 0), ("ai", 1)]
    assert candidates[0].domain == "cmv"
    assert candidates[0].source_family == "cmv_human"
    assert candidates[1].source_family == "roct_machine_continuation_flan_t5_large"
    assert candidates[1].generator_family == "mage:roct_machine_continuation_flan_t5_large"
    assert candidates[1].transformation == "raw_generation"


def test_mage_adapter_fails_closed_when_source_provenance_conflicts_with_label() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        adapt_mage_rows([{"text": "bad", "label": 1, "src": "roct_machine_flan"}])
