import numpy as np

from humanized_detector.v3_features import FEATURE_NAMES, FeatureNormalizer, extract_structural_features


def test_structural_features_are_fixed_size_and_capture_layout() -> None:
    values = extract_structural_features("Short sentence! A longer sentence has: punctuation; and contractions don't vanish.\n\nNew paragraph?")

    assert values.shape == (len(FEATURE_NAMES),)
    assert values[FEATURE_NAMES.index("sentence_count")] == 3
    assert values[FEATURE_NAMES.index("semicolon_per_100_words")] > 0
    assert values[FEATURE_NAMES.index("newline_per_100_words")] > 0


def test_feature_normalizer_fits_training_values_only() -> None:
    normalizer = FeatureNormalizer.fit(np.asarray([[0.0, 2.0], [2.0, 4.0]]))

    np.testing.assert_allclose(normalizer.transform(np.asarray([[3.0, 5.0]])), [[2.0, 2.0]])


def test_feature_normalizer_keeps_constant_training_features_finite() -> None:
    normalizer = FeatureNormalizer.fit(np.asarray([[1.0], [1.0]]))

    np.testing.assert_allclose(normalizer.transform(np.asarray([[1.0], [3.0]])), [[0.0], [2.0]])
