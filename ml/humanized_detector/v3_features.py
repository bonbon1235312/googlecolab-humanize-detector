"""Deterministic, dependency-free structural features for V3 ablations."""

from dataclasses import dataclass
import math
import re

import numpy as np


FEATURE_NAMES = (
    "word_count", "sentence_count", "sentence_length_mean", "sentence_length_std", "sentence_length_skew",
    "paragraph_count", "paragraph_length_std", "type_token_ratio", "hapax_rate", "word_length_mean",
    "word_length_std", "repeated_token_rate", "contraction_rate", "function_word_ratio", "newline_per_100_words",
    "semicolon_per_100_words", "colon_per_100_words", "hyphen_per_100_words", "comma_per_100_words",
    "exclamation_per_100_words", "question_per_100_words", "character_entropy_mean", "character_entropy_std",
)

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE = re.compile(r"[.!?]+")
_PARAGRAPH = re.compile(r"\n\s*\n")
_FUNCTION_WORDS = frozenset("a an and are as at be by for from has have he her him his i in is it its me my nor not of on or our she so that the their them they this to us was we were with you your".split())


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    frequencies = {character: text.count(character) / len(text) for character in set(text)}
    return float(-sum(probability * math.log2(probability) for probability in frequencies.values()))


def extract_structural_features(text: str) -> np.ndarray:
    """Return the fixed V3 feature vector without calling a parser or remote service."""
    words = [word.casefold() for word in _WORD.findall(text)]
    word_count = len(words)
    denominator = max(word_count, 1)
    sentences = [sentence for sentence in _SENTENCE.split(text) if sentence.strip()]
    sentence_lengths = [len(_WORD.findall(sentence)) for sentence in sentences]
    paragraphs = [paragraph for paragraph in _PARAGRAPH.split(text) if paragraph.strip()]
    paragraph_lengths = [len(_WORD.findall(paragraph)) for paragraph in paragraphs]
    counts = {word: words.count(word) for word in set(words)}
    sentence_std = _safe_std([float(value) for value in sentence_lengths])
    sentence_mean = _safe_mean([float(value) for value in sentence_lengths])
    skew = _safe_mean([((value - sentence_mean) / sentence_std) ** 3 for value in sentence_lengths]) if sentence_std else 0.0
    entropy_windows = [_entropy(text[index : index + 128]) for index in range(0, max(len(text), 1), 128)]
    per_100 = lambda count: 100.0 * count / denominator
    values = (
        float(word_count), float(len(sentences)), sentence_mean, sentence_std, skew,
        float(len(paragraphs)), _safe_std([float(value) for value in paragraph_lengths]),
        len(counts) / denominator, sum(count == 1 for count in counts.values()) / denominator,
        _safe_mean([float(len(word)) for word in words]), _safe_std([float(len(word)) for word in words]),
        1.0 - len(counts) / denominator, sum("'" in word for word in words) / denominator,
        sum(word in _FUNCTION_WORDS for word in words) / denominator, per_100(text.count("\n")),
        per_100(text.count(";")), per_100(text.count(":")), per_100(text.count("-")), per_100(text.count(",")),
        per_100(text.count("!")), per_100(text.count("?")), _safe_mean(entropy_windows), _safe_std(entropy_windows),
    )
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("values must be a non-empty matrix")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        return cls(mean=mean, scale=np.where(scale == 0, 1.0, scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale
