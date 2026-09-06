import json

import pytest

from humanized_detector.v6_train import load_v6_training_contract


def test_v6_training_contract_requires_the_authorized_text_free_export(tmp_path) -> None:
    path = tmp_path / "v6_training_contract.json"
    payload = {
        "protocol_version": "v6",
        "manifest_sha256": "a" * 64,
        "sampling_policy_sha256": "b" * 64,
        "dedup_policy_sha256": "c" * 64,
        "sealed_data_loaded": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_v6_training_contract(path) == payload


def test_v6_training_contract_rejects_a_sealed_input_claim(tmp_path) -> None:
    path = tmp_path / "v6_training_contract.json"
    path.write_text(json.dumps({"protocol_version": "v6", "manifest_sha256": "a" * 64, "sampling_policy_sha256": "b" * 64, "dedup_policy_sha256": "c" * 64, "sealed_data_loaded": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed"):
        load_v6_training_contract(path)
