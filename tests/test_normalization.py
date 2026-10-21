from __future__ import annotations

import numpy as np
import pytest

from als_insight.normalization import (
    LowCountFilter,
    MedianOfRatiosLog,
    log_geometric_means,
    median_of_ratios_size_factors,
    normalized_log_counts,
)


@pytest.fixture
def counts() -> np.ndarray:
    rng = np.random.default_rng(1)
    base = rng.integers(20, 500, size=50)
    sf = np.array([0.5, 1.0, 2.0, 4.0])
    return np.outer(sf, base).astype(float)


def test_size_factors_recover_scaling(counts: np.ndarray) -> None:
    sf = median_of_ratios_size_factors(counts, log_geometric_means(counts))
    np.testing.assert_allclose(sf / sf[1], [0.5, 1.0, 2.0, 4.0])
    # After normalisation every sample has the same profile.
    logexpr = normalized_log_counts(counts)
    np.testing.assert_allclose(logexpr, np.broadcast_to(logexpr[0], logexpr.shape))


def test_zero_counts_are_ignored_by_reference() -> None:
    counts = np.array([[10.0, 0.0, 30.0], [20.0, 5.0, 60.0]])
    assert np.isneginf(log_geometric_means(counts)[1])
    sf = median_of_ratios_size_factors(counts, log_geometric_means(counts))
    assert sf[1] / sf[0] == pytest.approx(2.0)


def test_no_shared_feature_raises() -> None:
    counts = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="expressed in all"):
        median_of_ratios_size_factors(counts, log_geometric_means(counts))


def test_normalizer_uses_frozen_training_reference(counts: np.ndarray) -> None:
    train, test = counts[:3], counts[3:]
    norm = MedianOfRatiosLog().fit(train)
    # A test sample is normalised on its own: the result does not depend on other test rows.
    batch = norm.transform(np.vstack([test, test * 3]))
    np.testing.assert_allclose(norm.transform(test), batch[:1])
    # Refitting on different data changes the reference (state is learnt in fit only).
    other = MedianOfRatiosLog().fit(counts[:, ::-1])
    assert not np.allclose(other.log_reference_, norm.log_reference_)


def test_low_count_filter_is_learnt_on_training_samples_only() -> None:
    train = np.array([[50.0, 0.0, 12.0], [40.0, 0.0, 15.0], [60.0, 1.0, 11.0]])
    test = np.array([[50.0, 10_000.0, 12.0]])
    filt = LowCountFilter(min_count=10, min_sample_fraction=0.5).fit(train)
    assert filt.support_.tolist() == [True, False, True]
    # The feature that only the test sample expresses is still removed.
    assert filt.transform(test).tolist() == [[50.0, 12.0]]


def test_low_count_filter_rejects_empty_result() -> None:
    with pytest.raises(ValueError, match="removed every feature"):
        LowCountFilter(min_count=100).fit(np.ones((3, 4)))
