"""Exploration (PCA/outliers) and differential expression on synthetic counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from als_insight.config import ContrastConfig, DifferentialConfig, ExplorationConfig
from als_insight.differential import run_deseq2, summarize
from als_insight.exploration import robust_z, run_pca


def _counts(n_samples: int = 30, n_genes: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.lognormal(4, 1, size=n_genes)
    mu = np.outer(rng.uniform(0.7, 1.3, n_samples), base)
    counts = rng.negative_binomial(10, 10 / (10 + mu))
    return pd.DataFrame(
        counts,
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"G{j}" for j in range(n_genes)],
    )


def test_robust_z() -> None:
    z = robust_z(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert z[-1] > 10
    assert abs(z[2]) < 1e-12
    assert robust_z(np.ones(4)).tolist() == [0.0] * 4


def test_pca_flags_planted_outlier() -> None:
    counts = _counts()
    rng = np.random.default_rng(3)
    genes = rng.choice(counts.shape[1], 40, replace=False)
    counts.iloc[5, genes] *= 30  # strongly perturbed profile
    result = run_pca(counts, ExplorationConfig(n_components=5, outlier_components=3))
    assert result.scores.loc["S5", "outlier"]
    assert result.scores["outlier"].sum() <= 2
    assert result.explained_variance_ratio[0] >= result.explained_variance_ratio[1]
    assert {"PC1", "PC5", "distance", "robust_z"} <= set(result.scores.columns)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_deseq2_recovers_planted_signal() -> None:
    counts = _counts(n_samples=24, n_genes=150, seed=4)
    condition = np.array(["ALS"] * 12 + ["Control"] * 12)
    signal = ["G0", "G1", "G2", "G3"]
    counts.loc[condition == "ALS", signal] *= 4
    meta = pd.DataFrame(
        {"condition": condition, "region": ["frontal", "motor"] * 12}, index=counts.index
    )
    contrast = ContrastConfig(case="a", control="b")
    res = run_deseq2(counts, meta, contrast, DifferentialConfig(n_cpus=1))
    assert (res.loc[signal, "log2FoldChange"] > 1).all()  # direction: case vs control
    assert (res.loc[signal, "padj"] < 1e-3).all()
    assert (
        res.loc[signal, "log2FoldChange"].abs() <= res.loc[signal, "log2FoldChange_mle"].abs()
    ).all()
    summary = summarize(res, alpha=0.05, lfc_threshold=1.0)
    assert summary["n_up"] >= 4
    assert summary["n_significant"] < 20  # few false positives among null genes
