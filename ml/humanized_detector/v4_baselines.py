"""Classical, reproducible V4 control baselines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .v3_evaluate import evaluate_binary
from .v4_control import load_v4_control_partitions


def _vectorizer(variant: str) -> TfidfVectorizer:
    if variant == "word_tfidf_lr":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    if variant == "char_tfidf_lr":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    raise ValueError("unknown V4 baseline variant")


def _write_predictions(path: Path, rows: Sequence[dict[str, object]], probabilities: Sequence[float]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, probability in zip(rows, probabilities, strict=True):
            prediction = {"id": str(row["id"]), "label": int(row["label"]), "probability": float(probability)}
            handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")


def train_tfidf_baseline(
    train_rows: Sequence[dict[str, object]],
    development_rows: Sequence[dict[str, object]],
    artifact_dir: Path,
    variant: str,
    calibration_rows: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Fit a TF-IDF + logistic-regression control model on training rows only."""
    if not train_rows or not development_rows:
        raise ValueError("train and development rows must be non-empty")
    vectorizer = _vectorizer(variant)
    model = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260904)),
    ])
    train_texts = [str(row["text"]) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]
    development_texts = [str(row["text"]) for row in development_rows]
    development_labels = [int(row["label"]) for row in development_rows]
    model.fit(train_texts, train_labels)
    probabilities = model.predict_proba(development_texts)[:, 1].tolist()
    metrics: dict[str, object] = {
        "variant": variant,
        "vectorizer_analyzer": str(vectorizer.analyzer),
        **evaluate_binary(development_labels, probabilities),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "development_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_predictions(artifact_dir / "development_predictions.jsonl", development_rows, probabilities)
    if calibration_rows is not None:
        calibration_probabilities = model.predict_proba([str(row["text"]) for row in calibration_rows])[:, 1].tolist()
        _write_predictions(artifact_dir / "calibration_predictions.jsonl", calibration_rows, calibration_probabilities)
    return metrics


def run_tfidf_control(data_dir: Path, artifact_dir: Path, variant: str) -> dict[str, object]:
    """Run a baseline against V4's non-sealed control partitions."""
    partitions = load_v4_control_partitions(data_dir)
    return train_tfidf_baseline(partitions["train"], partitions["development"], artifact_dir, variant, partitions["calibration"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("word_tfidf_lr", "char_tfidf_lr"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_tfidf_control(args.data_dir, args.artifacts_dir, args.variant), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
