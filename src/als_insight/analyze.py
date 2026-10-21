"""Pipeline step ``analyze``: PCA + outlier detection, then PyDESeq2 differential expression."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from als_insight import cache, differential, preprocess
from als_insight.config import Config
from als_insight.exploration import run_pca

logger = logging.getLogger(__name__)

PCA_FILE = "pca.csv"
PCA_VARIANCE_FILE = "pca_variance.json"
DE_FILE = "de_results.csv"
SUMMARY_FILE = "summary.json"
OUTPUTS = (PCA_FILE, PCA_VARIANCE_FILE, DE_FILE, SUMMARY_FILE)


def retained_samples(pca: pd.DataFrame, remove_outliers: bool) -> pd.Index:
    return pca.index[~pca["outlier"]] if remove_outliers else pca.index


def run(cfg: Config, force: bool = False) -> Path:
    in_dir = cfg.step_dir("preprocess")
    out_dir = cfg.step_dir("analyze")
    fp = cache.fingerprint(
        cfg.section_fingerprint("contrast", "exploration", "differential"),
        [in_dir / name for name in preprocess.OUTPUTS],
    )
    if not force and cache.is_fresh(out_dir, fp, OUTPUTS):
        logger.info("analyze: up to date (%s)", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = preprocess.load_dataset(in_dir)
    kept_features = ds.features.index[ds.features["passes_filter"]]
    counts = ds.counts.loc[:, kept_features]

    pca = run_pca(counts, cfg.exploration)
    pca_table = pca.scores.join(ds.metadata[["condition", "region", "subject_id"]])
    pca_table.to_csv(out_dir / PCA_FILE)
    (out_dir / PCA_VARIANCE_FILE).write_text(
        json.dumps(
            {
                "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
                "n_features_used": pca.n_features_used,
            },
            indent=2,
        )
    )

    keep = retained_samples(pca.scores, cfg.exploration.remove_outliers)
    outliers = pca.scores.index[pca.scores["outlier"]]
    if cfg.exploration.remove_outliers and len(outliers):
        logger.info("Excluding %d PCA outlier(s): %s", len(outliers), list(outliers))
    de = differential.run_deseq2(
        counts.loc[keep], ds.metadata.loc[keep], cfg.contrast, cfg.differential
    )
    de.insert(0, "feature_type", ds.features.loc[de.index, "feature_type"])
    de.to_csv(out_dir / DE_FILE)

    de_summary = differential.summarize(de, cfg.differential.alpha, cfg.differential.lfc_threshold)
    summary = {
        "n_samples_analyzed": len(keep),
        "outliers": [
            {
                "sample_id": s,
                "condition": ds.metadata.loc[s, "condition"],
                "robust_z": round(float(pca.scores.loc[s, "robust_z"]), 2),
            }
            for s in outliers
        ],
        "outliers_removed": cfg.exploration.remove_outliers,
        "samples_per_condition": ds.metadata.loc[keep, "condition"].value_counts().to_dict(),
        "design": differential.design_formula(cfg.differential.covariates),
        "alpha": cfg.differential.alpha,
        "lfc_threshold": cfg.differential.lfc_threshold,
        **de_summary,
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    logger.info(
        "DE: %d/%d features significant at FDR < %.2f (%d up, %d down; %d with |LFC| >= %.1f)",
        de_summary["n_significant"],
        de_summary["n_tested"],
        cfg.differential.alpha,
        de_summary["n_up"],
        de_summary["n_down"],
        de_summary["n_significant_lfc"],
        cfg.differential.lfc_threshold,
    )
    cache.mark_done(out_dir, fp, OUTPUTS)
    return out_dir
