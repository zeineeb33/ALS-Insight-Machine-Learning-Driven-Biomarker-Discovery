"""Unsupervised exploration: PCA on log-normalised counts and robust outlier flagging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from als_insight.config import ExplorationConfig
from als_insight.normalization import normalized_log_counts

logger = logging.getLogger(__name__)

_MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class PCAResult:
    scores: pd.DataFrame  # samples x PCs, plus distance / robust_z / outlier columns
    explained_variance_ratio: np.ndarray
    n_features_used: int


def robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); 0 everywhere when the MAD is zero."""
    median = np.median(values)
    mad = _MAD_TO_SD * np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return (values - median) / mad


def run_pca(counts: pd.DataFrame, cfg: ExplorationConfig) -> PCAResult:
    """PCA on the most variable features, outliers flagged by robust z of PC-space distance.

    The distance uses the first ``outlier_components`` PCs, each scaled to unit variance
    (i.e. a Mahalanobis distance in PC space). This is label-agnostic.
    """
    logexpr = normalized_log_counts(counts.to_numpy())
    n_top = min(cfg.n_top_variable, logexpr.shape[1])
    top = np.argsort(logexpr.var(axis=0))[::-1][:n_top]
    n_components = min(cfg.n_components, logexpr.shape[0] - 1, n_top)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(logexpr[:, top])
    columns = [f"PC{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(scores, index=counts.index, columns=columns)

    k = min(cfg.outlier_components, n_components)
    scaled = scores[:, :k] / scores[:, :k].std(axis=0, ddof=1)
    distance = np.sqrt((scaled**2).sum(axis=1))
    df["distance"] = distance
    df["robust_z"] = robust_z(distance)
    df["outlier"] = df["robust_z"] > cfg.outlier_robust_z
    logger.info(
        "PCA on %d features: PC1-PC3 explain %s; %d outlier(s) flagged (robust z > %.1f)",
        n_top,
        np.round(pca.explained_variance_ratio_[:3] * 100, 1).tolist(),
        int(df["outlier"].sum()),
        cfg.outlier_robust_z,
    )
    return PCAResult(df, pca.explained_variance_ratio_, n_top)
