"""Differential expression with PyDESeq2 (negative-binomial GLM, Wald test, BH FDR)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from als_insight.config import ContrastConfig, DifferentialConfig

logger = logging.getLogger(__name__)

DE_COLUMNS = ["baseMean", "log2FoldChange", "log2FoldChange_mle", "lfcSE", "stat", "pvalue", "padj"]


def design_formula(covariates: list[str]) -> str:
    return "~ " + " + ".join([*covariates, "condition"])


def run_deseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    contrast: ContrastConfig,
    cfg: DifferentialConfig,
) -> pd.DataFrame:
    """Return a per-feature DE table for ``case_label`` vs ``control_label``.

    ``log2FoldChange`` is apeGLM-shrunk when ``cfg.shrink_lfc`` is set (better for
    ranking); the unshrunk MLE is kept in ``log2FoldChange_mle``. P-values are from the
    Wald test and ``padj`` is Benjamini-Hochberg adjusted, with DESeq2's independent
    filtering and Cooks-distance outlier handling.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    obs = metadata[["condition", *cfg.covariates]].copy()
    # The control group is the reference level of the treatment coding.
    obs["condition"] = pd.Categorical(
        obs["condition"], categories=[contrast.control_label, contrast.case_label]
    )
    for cov in cfg.covariates:
        obs[cov] = obs[cov].astype(str)

    inference = DefaultInference(n_cpus=cfg.n_cpus)
    design = design_formula(cfg.covariates)
    logger.info("Running PyDESeq2 (%s) on %d samples x %d features", design, *counts.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dds = DeseqDataSet(
            counts=counts,
            metadata=obs,
            design=design,
            refit_cooks=True,
            inference=inference,
            quiet=True,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds,
            contrast=["condition", contrast.case_label, contrast.control_label],
            alpha=cfg.alpha,
            inference=inference,
            quiet=True,
        )
        stats.summary()
        results = stats.results_df.copy()
        results["log2FoldChange_mle"] = results["log2FoldChange"]
        if cfg.shrink_lfc:
            stats.lfc_shrink(coeff=f"condition[T.{contrast.case_label}]")
            results["log2FoldChange"] = stats.results_df["log2FoldChange"]
            results["lfcSE"] = stats.results_df["lfcSE"]
    results.index.name = "feature"
    return results[DE_COLUMNS]


def summarize(results: pd.DataFrame, alpha: float, lfc_threshold: float) -> dict[str, int]:
    padj = results["padj"].to_numpy()
    lfc = results["log2FoldChange"].to_numpy()
    lfc_mle = results["log2FoldChange_mle"].to_numpy()
    sig = np.nan_to_num(padj, nan=1.0) < alpha
    strong = sig & (np.abs(lfc) >= lfc_threshold)
    return {
        "n_tested": len(results),
        "n_with_padj": int(np.isfinite(padj).sum()),
        "n_significant": int(sig.sum()),
        "n_up": int((sig & (lfc > 0)).sum()),
        "n_down": int((sig & (lfc < 0)).sum()),
        "n_significant_lfc": int(strong.sum()),
        "n_significant_lfc_mle": int((sig & (np.abs(lfc_mle) >= lfc_threshold)).sum()),
    }
