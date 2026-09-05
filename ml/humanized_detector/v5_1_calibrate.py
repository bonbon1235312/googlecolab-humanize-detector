"""Post-selection operating-point calibration for a V5.1 artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .v3_calibrate import calibration_record
from .v3_evaluate import evaluate_binary
from .v4_prepare import load_v4_partition
from .v5_1_inference import score_artifact_rows


def calibration_summary(labels: Sequence[int], probabilities: Sequence[float], fpr_targets: Sequence[float] = (0.01, 0.02, 0.05)) -> dict[str, object]:
    """Produce frozen operating points; threshold fitting uses calibration only."""
    points: dict[str, object] = {}
    for target in fpr_targets:
        key = f"{int(target * 100)}pct"
        points[key] = calibration_record(labels, probabilities, target)
    return {"ranking_metrics": evaluate_binary(labels, probabilities), "operating_points": points}


def calibrate_artifact(data_dir: Path, artifact_dir: Path) -> dict[str, object]:
    rows = load_v4_partition(data_dir, "calibration")
    labels = [int(row["label"]) for row in rows]
    probabilities = score_artifact_rows(artifact_dir, rows)
    report = calibration_summary(labels, probabilities)
    report.update({"partition": "calibration.jsonl", "checkpoint": "model.pt", "artifact_dir": str(artifact_dir)})
    (artifact_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate_artifact(args.data_dir, args.artifacts_dir), indent=2))


if __name__ == "__main__":
    main()
