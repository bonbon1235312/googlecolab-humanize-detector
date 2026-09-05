"""Subtype-aware development reporting and checkpoint selection for H1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .v3_calibrate import calibration_record
from .v3_evaluate import evaluate_binary


_SUBTYPES = ("expert_edited_ai", "llm_edited_ai", "raw_ai")
_SELECTION_SUBTYPES = ("expert_edited_ai", "llm_edited_ai")


def _subtype_report(rows: Sequence[Mapping[str, object]], probabilities: np.ndarray, subtype: str) -> dict[str, object]:
    indexes = [index for index, row in enumerate(rows) if int(row["label"]) == 0 or str(row.get("provenance")) == subtype]
    labels = np.asarray([int(rows[index]["label"]) for index in indexes], dtype=int)
    scores = probabilities[indexes]
    if len(np.unique(labels)) != 2:
        raise ValueError(f"development subtype {subtype} requires both classes")
    report = evaluate_binary(labels, scores)
    operating_point = calibration_record(labels, scores, 0.05)
    operating_metrics = operating_point["metrics"]
    assert isinstance(operating_metrics, dict)
    report["partial_auc_5pct"] = float(roc_auc_score(labels, scores, max_fpr=0.05))
    report["tpr_at_5pct_fpr"] = float(operating_metrics["recall"])
    report["human_fpr_at_5pct"] = float(operating_metrics["human_fpr"])
    report["human_false_positives_at_5pct"] = int(round(float(operating_metrics["human_fpr"]) * int((labels == 0).sum())))
    return report


def humanizer_development_report(rows: Sequence[Mapping[str, object]], probabilities: Sequence[float]) -> dict[str, object]:
    """Report aggregate and positive-subtype development metrics."""
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("rows and probabilities must be non-empty and have the same length")
    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    subtypes = {name: _subtype_report(rows, scores, name) for name in _SUBTYPES}
    macro_auc = float(np.mean([float(subtypes[name]["roc_auc"]) for name in _SELECTION_SUBTYPES]))
    macro_partial = float(np.mean([float(subtypes[name]["partial_auc_5pct"]) for name in _SELECTION_SUBTYPES]))
    return {
        "aggregate": evaluate_binary(labels, scores),
        "subtypes": subtypes,
        "selection": {
            "primary_metric": "macro_edit_auc",
            "macro_edit_auc": macro_auc,
            "macro_edit_partial_auc_5pct": macro_partial,
        },
    }


def humanizer_selection_key(report: Mapping[str, object]) -> tuple[float, float, float]:
    """Return the predeclared H1 checkpoint ordering key."""
    selection = report["selection"]
    aggregate = report["aggregate"]
    assert isinstance(selection, Mapping) and isinstance(aggregate, Mapping)
    return (
        float(selection["macro_edit_auc"]),
        float(selection["macro_edit_partial_auc_5pct"]),
        float(aggregate["roc_auc"]),
    )
