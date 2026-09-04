"""Classical, reproducible V4 control baselines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .v3_evaluate import evaluate_binary


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
    return metrics
