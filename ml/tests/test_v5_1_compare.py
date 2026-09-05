from humanized_detector.v5_1_compare import lineage_paired_bootstrap, promotion_decision


def test_lineage_paired_bootstrap_reports_a_clear_candidate_gain() -> None:
    rows = []
    baseline = []
    candidate = []
    for group in range(10):
        rows.extend([
            {"id": f"h-{group}", "lineage_id": f"p-{group}", "label": 0, "provenance": "human"},
            {"id": f"e-{group}", "lineage_id": f"p-{group}", "label": 1, "provenance": "expert_edited_ai"},
            {"id": f"l-{group}", "lineage_id": f"p-{group}", "label": 1, "provenance": "llm_edited_ai"},
        ])
        baseline.extend([0.6, 0.55, 0.9])
        candidate.extend([0.2, 0.8, 0.95])

    report = lineage_paired_bootstrap(rows, baseline, candidate, iterations=100, seed=7)

    assert report["point_estimate"]["macro_edit_auc_delta"] > 0
    assert report["confidence_intervals"]["macro_edit_auc_delta"]["lower"] > 0
    assert report["expert_edit_disposition"] == "improved"


def test_promotion_requires_a_credible_macro_gain_and_no_subtype_regression() -> None:
    report = {
        "confidence_intervals": {
            "macro_edit_auc_delta": {"lower": 0.02, "upper": 0.08},
            "expert_edited_ai_auc_delta": {"lower": -0.005, "upper": 0.04},
            "llm_edited_ai_auc_delta": {"lower": -0.004, "upper": 0.03},
            "expert_edited_ai_tpr_at_5pct_fpr_delta": {"lower": -0.01, "upper": 0.05},
            "llm_edited_ai_tpr_at_5pct_fpr_delta": {"lower": -0.01, "upper": 0.05},
        }
    }

    decision = promotion_decision(report, auc_noninferiority_tolerance=0.01, tpr_noninferiority_tolerance=0.02)

    assert decision["promote"] is True
