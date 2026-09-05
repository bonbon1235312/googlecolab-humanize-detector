import pytest

from humanized_detector.v5_sampling import curriculum_mix, hierarchical_weights


def _row(identifier: str, source: str, label: int, provenance: str, lineage: str) -> dict[str, object]:
    return {"id": identifier, "source": source, "label": label, "provenance": provenance, "lineage_id": lineage}


def test_hierarchical_weights_preserve_top_level_mass_and_balance_beemo_lineages() -> None:
    rows = [
        _row("ph", "padben", 0, "human", "ph"),
        _row("pa", "padben", 1, "ai_humanized", "pa"),
        _row("bh", "beemo", 0, "human", "p1"),
        _row("r", "beemo", 1, "raw_ai", "p1"),
        _row("e", "beemo", 1, "expert_edited_ai", "p1"),
        _row("l1", "beemo", 1, "llm_edited_ai", "p1"),
        _row("l2a", "beemo", 1, "llm_edited_ai", "p2"),
        _row("l2b", "beemo", 1, "llm_edited_ai", "p2"),
    ]
    weights = hierarchical_weights(rows, {"raw_ai": 0.2, "expert_edited_ai": 0.5, "llm_edited_ai": 0.3})
    total = sum(weights)

    def mass(predicate):
        return sum(weight for row, weight in zip(rows, weights, strict=True) if predicate(row)) / total

    assert mass(lambda row: row["source"] == "padben" and row["label"] == 0) == pytest.approx(0.25)
    assert mass(lambda row: row["source"] == "padben" and row["label"] == 1) == pytest.approx(0.25)
    assert mass(lambda row: row["source"] == "beemo" and row["label"] == 0) == pytest.approx(0.25)
    assert mass(lambda row: row["provenance"] == "expert_edited_ai") == pytest.approx(0.125)
    assert weights[5] == pytest.approx(weights[6] + weights[7])


def test_curriculum_mix_moves_toward_expert_edits() -> None:
    assert curriculum_mix(1) == {"expert_edited_ai": 0.25, "llm_edited_ai": 0.50, "raw_ai": 0.25}
    assert curriculum_mix(2)["expert_edited_ai"] == 0.40
    assert curriculum_mix(4)["expert_edited_ai"] == 0.50


def test_hierarchical_weights_reject_missing_required_stratum() -> None:
    rows = [_row("ph", "padben", 0, "human", "ph")]
    with pytest.raises(ValueError, match="missing training stratum"):
        hierarchical_weights(rows, curriculum_mix(1))
