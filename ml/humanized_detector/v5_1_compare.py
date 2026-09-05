"""Lineage-aware V5.1 promotion analysis.

This module deliberately compares already-scored development rows.  It never
opens calibration or sealed evaluation data, so architecture selection remains
separate from deployment calibration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .v3_calibrate import select_operating_threshold
from .v4_prepare import load_v4_partition
from .v5_1_inference import score_artifact_rows


_EDIT_SUBTYPES = ("expert_edited_ai", "llm_edited_ai")


def _group_indexes(rows: Sequence[Mapping[str, object]]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        lineage = str(row.get("lineage_id") or row.get("group_id") or row.get("id"))
        grouped.setdefault(lineage, []).append(index)
    return [np.asarray(indexes, dtype=int) for indexes in grouped.values()]


def _metrics(rows: Sequence[Mapping[str, object]], scores: np.ndarray, max_human_fpr: float) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    threshold = select_operating_threshold(labels, scores, max_human_fpr)
    values: dict[str, float] = {}
    aucs: list[float] = []
    for subtype in _EDIT_SUBTYPES:
        indexes = np.asarray([i for i, row in enumerate(rows) if int(row["label"]) == 0 or str(row.get("provenance")) == subtype], dtype=int)
        subtype_labels = labels[indexes]
        subtype_scores = scores[indexes]
        auc = float(roc_auc_score(subtype_labels, subtype_scores))
        tpr = float((subtype_scores[subtype_labels == 1] >= threshold).mean())
        values[f"{subtype}_auc"] = auc
        values[f"{subtype}_tpr_at_{int(max_human_fpr * 100)}pct_fpr"] = tpr
        aucs.append(auc)
    values["macro_edit_auc"] = float(np.mean(aucs))
    return values


def _interval(values: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=float)
    return {"lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975))}


def lineage_paired_bootstrap(
    rows: Sequence[Mapping[str, object]], baseline_probabilities: Sequence[float], candidate_probabilities: Sequence[float],
    iterations: int = 2000, seed: int = 20260905, max_human_fpr: float = 0.05,
) -> dict[str, object]:
    """Bootstrap paired score differences by complete prompt lineage.

    A group is resampled as a unit, preserving the correlation among a human
    response and its raw/edited variants.  The FPR threshold is global within
    each model draw: deployment does not know an edit subtype beforehand.
    """
    if len(rows) < 2 or len(rows) != len(baseline_probabilities) or len(rows) != len(candidate_probabilities):
        raise ValueError("rows and both probability sequences must be non-empty and aligned")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    baseline = np.asarray(baseline_probabilities, dtype=float)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    groups = _group_indexes(rows)
    point_base = _metrics(rows, baseline, max_human_fpr)
    point_candidate = _metrics(rows, candidate, max_human_fpr)
    keys = tuple(point_base)
    deltas: dict[str, list[float]] = {key: [] for key in keys}
    generator = np.random.default_rng(seed)
    for _ in range(iterations):
        chosen = generator.integers(0, len(groups), size=len(groups))
        indexes = np.concatenate([groups[index] for index in chosen])
        sampled_rows = [rows[index] for index in indexes]
        sampled_base = _metrics(sampled_rows, baseline[indexes], max_human_fpr)
        sampled_candidate = _metrics(sampled_rows, candidate[indexes], max_human_fpr)
        for key in keys:
            deltas[key].append(sampled_candidate[key] - sampled_base[key])
    point_delta = {f"{key}_delta": point_candidate[key] - point_base[key] for key in keys}
    intervals = {f"{key}_delta": _interval(deltas[key]) for key in keys}
    expert_interval = intervals["expert_edited_ai_auc_delta"]
    expert_disposition = "improved" if expert_interval["lower"] > 0 else "regressed" if expert_interval["upper"] < 0 else "inconclusive"
    return {
        "method": "paired_lineage_bootstrap",
        "iterations": iterations,
        "max_human_fpr": max_human_fpr,
        "point_estimate": point_delta,
        "confidence_intervals": intervals,
        "expert_edit_disposition": expert_disposition,
    }


def promotion_decision(
    bootstrap_report: Mapping[str, object], auc_noninferiority_tolerance: float = 0.01,
    tpr_noninferiority_tolerance: float = 0.02,
) -> dict[str, object]:
    """Apply the predeclared V5.1 gate; an inconclusive result is no promotion."""
    intervals = bootstrap_report["confidence_intervals"]
    assert isinstance(intervals, Mapping)
    def lower(name: str) -> float:
        interval = intervals[name]
        assert isinstance(interval, Mapping)
        return float(interval["lower"])
    macro_improved = lower("macro_edit_auc_delta") > 0
    auc_safe = all(lower(f"{subtype}_auc_delta") >= -auc_noninferiority_tolerance for subtype in _EDIT_SUBTYPES)
    tpr_safe = all(lower(f"{subtype}_tpr_at_5pct_fpr_delta") >= -tpr_noninferiority_tolerance for subtype in _EDIT_SUBTYPES)
    return {
        "promote": macro_improved and auc_safe and tpr_safe,
        "criteria": {
            "credible_macro_edit_auc_improvement": macro_improved,
            "auc_noninferiority": auc_safe,
            "tpr_at_5pct_fpr_noninferiority": tpr_safe,
            "auc_noninferiority_tolerance": auc_noninferiority_tolerance,
            "tpr_noninferiority_tolerance": tpr_noninferiority_tolerance,
        },
        "default_on_inconclusive": "retain_baseline",
    }


def compare_artifacts(
    data_dir: Path, baseline_artifacts: Path, candidate_artifacts: Path, output_dir: Path,
    iterations: int = 2000, max_human_fpr: float = 0.05,
) -> dict[str, object]:
    """Score the untouched development partition and write a promotion record."""
    rows = load_v4_partition(data_dir, "development")
    baseline = score_artifact_rows(baseline_artifacts, rows)
    candidate = score_artifact_rows(candidate_artifacts, rows)
    bootstrap = lineage_paired_bootstrap(rows, baseline, candidate, iterations=iterations, max_human_fpr=max_human_fpr)
    record = {
        "baseline_artifacts": str(baseline_artifacts),
        "candidate_artifacts": str(candidate_artifacts),
        "partition": "development.jsonl",
        "bootstrap": bootstrap,
        "decision": promotion_decision(bootstrap),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "promotion_decision.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-artifacts", type=Path, required=True)
    parser.add_argument("--candidate-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare_artifacts(args.data_dir, args.baseline_artifacts, args.candidate_artifacts, args.output_dir, args.iterations), indent=2))


if __name__ == "__main__":
    main()
