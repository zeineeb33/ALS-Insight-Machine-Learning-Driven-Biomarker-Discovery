"""Pipeline step ``select``: nested-CV elastic net + XGBoost, stability, consensus ranking."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from als_insight import analyze, cache, preprocess
from als_insight.config import Config
from als_insight.consensus import consensus_ranking
from als_insight.modeling import MODELS, CVResult, nested_cv, permutation_test

logger = logging.getLogger(__name__)

FOLDS_FILE = "cv_folds.csv"
PREDICTIONS_FILE = "cv_predictions.csv"
IMPORTANCE_FILE = "importance_{model}.csv"
CONSENSUS_FILE = "consensus.csv"
PERMUTATION_FILE = "permutation.json"
SUMMARY_FILE = "summary.json"
OUTPUTS = (
    FOLDS_FILE,
    PREDICTIONS_FILE,
    CONSENSUS_FILE,
    PERMUTATION_FILE,
    SUMMARY_FILE,
    *(IMPORTANCE_FILE.format(model=m) for m in MODELS),
)


def run(cfg: Config, force: bool = False) -> Path:
    pre_dir, ana_dir = cfg.step_dir("preprocess"), cfg.step_dir("analyze")
    out_dir = cfg.step_dir("select")
    fp = cache.fingerprint(
        cfg.section_fingerprint("preprocess", "exploration", "selection", "consensus"),
        [pre_dir / preprocess.COUNTS_FILE, pre_dir / preprocess.METADATA_FILE]
        + [ana_dir / name for name in analyze.OUTPUTS],
    )
    if not force and cache.is_fresh(out_dir, fp, OUTPUTS):
        logger.info("select: up to date (%s)", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = preprocess.load_dataset(pre_dir)
    pca = pd.read_csv(ana_dir / analyze.PCA_FILE, index_col="sample_id")
    keep = analyze.retained_samples(pca, cfg.exploration.remove_outliers)
    meta = ds.metadata.loc[keep]
    # Raw counts of *all* features: filtering happens inside the CV folds.
    X = ds.counts.loc[keep].to_numpy(dtype=float)
    y = (meta["target"].to_numpy() == 1).astype(int)
    groups = meta["subject_id"].to_numpy()
    features = list(ds.counts.columns)
    logger.info(
        "Modelling %d samples from %d subjects (%d case / %d control) x %d features",
        len(y),
        len(set(groups)),
        int(y.sum()),
        int((1 - y).sum()),
        len(features),
    )

    results: dict[str, CVResult] = {}
    for model in MODELS:
        results[model] = nested_cv(model, X, y, groups, features, cfg.preprocess, cfg.selection)
        results[model].importance_table().to_csv(out_dir / IMPORTANCE_FILE.format(model=model))

    pd.concat([r.metrics_table() for r in results.values()]).to_csv(
        out_dir / FOLDS_FILE, index=False
    )
    pd.concat([r.predictions(list(keep)) for r in results.values()]).to_csv(
        out_dir / PREDICTIONS_FILE, index=False
    )

    perm: dict[str, object] = {"n_permutations": 0}
    if cfg.selection.permutation.n_permutations > 0:
        target = cfg.selection.permutation.model
        observed = results[target].metrics_table().query("repeat == 0")["roc_auc"].mean()
        perm = permutation_test(
            target, X, y, groups, features, cfg.preprocess, cfg.selection, float(observed)
        )
    (out_dir / PERMUTATION_FILE).write_text(json.dumps(perm, indent=2))

    de = pd.read_csv(ana_dir / analyze.DE_FILE, index_col="feature")
    consensus = consensus_ranking(
        de,
        results["elastic_net"].importance_table(),
        results["xgboost"].importance_table(),
        cfg.consensus.weights,
    )
    consensus.to_csv(out_dir / CONSENSUS_FILE)

    summary = {
        "n_samples": len(y),
        "n_subjects": len(set(groups)),
        "n_case": int(y.sum()),
        "n_control": int((1 - y).sum()),
        "cv": {
            "outer_folds": cfg.selection.outer_folds,
            "outer_repeats": cfg.selection.outer_repeats,
            "inner_folds": cfg.selection.inner_folds,
            "grouping": "subject_id",
            "k_features": cfg.selection.k_features,
        },
        "models": {
            model: {
                "metrics": r.metrics_summary(),
                "jaccard_stability": r.jaccard_stability(),
                "mean_selected_features": float(
                    pd.Series([int((f.importance != 0).sum()) for f in r.folds]).mean()
                ),
                "best_params": [f.best_params for f in r.folds],
            }
            for model, r in results.items()
        },
        "permutation_p_value": perm.get("p_value"),
        "top_consensus": consensus.index[: cfg.consensus.top_n].tolist(),
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, default=str))
    cache.mark_done(out_dir, fp, OUTPUTS)
    return out_dir
