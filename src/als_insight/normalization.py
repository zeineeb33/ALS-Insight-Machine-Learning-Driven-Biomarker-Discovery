"""Count normalisation and filtering as scikit-learn transformers.

Every statistic that depends on more than one sample (the low-count filter and the
DESeq2 median-of-ratios reference) is learnt in ``fit`` and only *applied* in
``transform``. Used inside a ``Pipeline`` this guarantees that held-out samples never
influence preprocessing during cross-validation.
"""

from __future__ import annotations

import math
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def low_count_mask(
    counts: NDArray[np.number], min_count: int, min_sample_fraction: float
) -> NDArray[np.bool_]:
    """Keep features with >= ``min_count`` reads in at least a fraction of samples."""
    min_samples = max(1, math.ceil(min_sample_fraction * counts.shape[0]))
    return np.asarray((counts >= min_count).sum(axis=0) >= min_samples)


def log_geometric_means(counts: NDArray[np.number]) -> NDArray[np.float64]:
    """Per-feature log geometric mean; ``-inf`` for features with any zero count."""
    with np.errstate(divide="ignore"):
        return np.log(counts.astype(np.float64)).mean(axis=0)


def median_of_ratios_size_factors(
    counts: NDArray[np.number], log_reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """DESeq2 size factors of ``counts`` against a (possibly external) reference.

    Only features expressed in every reference sample are used; within a sample,
    features with zero counts are ignored.
    """
    usable = np.isfinite(log_reference)
    if not usable.any():
        raise ValueError("No feature is expressed in all reference samples")
    with np.errstate(divide="ignore"):
        log_ratios = np.log(counts[:, usable].astype(np.float64)) - log_reference[usable]
    log_ratios[~np.isfinite(log_ratios)] = np.nan
    if np.isnan(log_ratios).all(axis=1).any():
        raise ValueError("A sample has zero counts for every reference feature")
    return np.exp(np.nanmedian(log_ratios, axis=1))


def normalized_log_counts(
    counts: NDArray[np.number], pseudocount: float = 1.0
) -> NDArray[np.float64]:
    """Median-of-ratios normalised ``log2(count + pseudocount)`` using all given samples."""
    sf = median_of_ratios_size_factors(counts, log_geometric_means(counts))
    return np.log2(counts / sf[:, None] + pseudocount)


class LowCountFilter(TransformerMixin, BaseEstimator):
    """Drop features that are not expressed in enough *training* samples."""

    def __init__(self, min_count: int = 10, min_sample_fraction: float = 0.1) -> None:
        self.min_count = min_count
        self.min_sample_fraction = min_sample_fraction

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        counts = np.asarray(X)
        self.support_ = low_count_mask(counts, self.min_count, self.min_sample_fraction)
        if not self.support_.any():
            raise ValueError("Low-count filter removed every feature")
        self.n_features_in_ = counts.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.number]:
        check_is_fitted(self, "support_")
        return np.asarray(X)[:, self.support_]


class MedianOfRatiosLog(TransformerMixin, BaseEstimator):
    """DESeq2-style size-factor normalisation followed by ``log2(x + pseudocount)``.

    The geometric-mean reference is computed on the training samples; new samples are
    normalised against that frozen reference.
    """

    def __init__(self, pseudocount: float = 1.0) -> None:
        self.pseudocount = pseudocount

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        counts = np.asarray(X)
        self.log_reference_ = log_geometric_means(counts)
        if not np.isfinite(self.log_reference_).any():
            raise ValueError("No feature is expressed in all training samples")
        self.n_features_in_ = counts.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "log_reference_")
        counts = np.asarray(X, dtype=np.float64)
        sf = median_of_ratios_size_factors(counts, self.log_reference_)
        return np.log2(counts / sf[:, None] + self.pseudocount)
