from humanized_detector.v5_selection import humanizer_development_report, humanizer_selection_key


def test_selection_gives_expert_and_llm_edits_equal_weight() -> None:
    rows = [
        {"label": 0, "provenance": "human"},
        {"label": 0, "provenance": "human"},
        {"label": 1, "provenance": "expert_edited_ai"},
        {"label": 1, "provenance": "expert_edited_ai"},
        {"label": 1, "provenance": "llm_edited_ai"},
        {"label": 1, "provenance": "llm_edited_ai"},
        {"label": 1, "provenance": "raw_ai"},
    ]
    report = humanizer_development_report(rows, [0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.95])

    assert report["selection"]["macro_edit_auc"] == 1.0
    assert set(report["subtypes"]) == {"expert_edited_ai", "llm_edited_ai", "raw_ai"}
    assert humanizer_selection_key(report)[0] == 1.0


def test_selection_prefers_expert_llm_macro_over_aggregate_auc() -> None:
    balanced = {"selection": {"macro_edit_auc": 0.80, "macro_edit_partial_auc_5pct": 0.60}, "aggregate": {"roc_auc": 0.80}}
    majority_favoured = {"selection": {"macro_edit_auc": 0.79, "macro_edit_partial_auc_5pct": 0.99}, "aggregate": {"roc_auc": 0.99}}

    assert humanizer_selection_key(balanced) > humanizer_selection_key(majority_favoured)
