"""Leakage-free multivariate gene selection with nested, subject-grouped cross-validation.

Design choices:

* The model input is the **raw count matrix**. Low-count filtering, median-of-ratios
  normalisation, univariate (ANOVA F) pre-selection of ``k`` features, scaling and the
  classifier form a single scikit-learn ``Pipeline``; every step is re-fitted on the
  training part of each split, so held-out samples never inform preprocessing.
* Splits are ``StratifiedGroupKFold`` on the **subject id**: several cortex regions of the
  same donor are never split between training and test (no donor leakage).
* Hyper-parameters are tuned by an inner grouped CV on the outer-training fold only;
  the outer folds are used exclusively for performance estimation (nested CV).
* Stability: per-gene selection frequency across all outer folds and repeats, plus the
  mean pairwise Jaccard index of selected-gene sets.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from als_insight.config import PreprocessConfig, SelectionConfig
from als_insight.normalization import LowCountFilter, MedianOfRatiosLog

logger = logging.getLogger(__name__)

ModelName = Literal["elastic_net", "xgboost"]
MODELS: tuple[ModelName, ...] = ("elastic_net", "xgboost")
METRICS = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "mcc")


@dataclass
class FoldResult:
    model: str
    repeat: int
    fold: int
    test_index: NDArray[np.intp]
    y_true: NDArray[np.int_]
    y_score: NDArray[np.float64]
    metrics: dict[str, float]
    best_params: dict[str, Any]
    importance: pd.Series  # signed coefficient (elastic net) or gain (xgboost), selected only
    n_convergence_warnings: int = 0


@dataclass
class CVResult:
    model: str
    folds: list[FoldResult] = field(default_factory=list)

    def metrics_table(self) -> pd.DataFrame:
        rows = [
            {
                "model": f.model,
                "repeat": f.repeat,
                "fold": f.fold,
                **f.metrics,
                **_flat(f.best_params),
            }
            for f in self.folds
        ]
        return pd.DataFrame(rows)

    def metrics_summary(self) -> dict[str, dict[str, float]]:
        table = self.metrics_table()
        return {
            m: {"mean": float(table[m].mean()), "std": float(table[m].std(ddof=1))} for m in METRICS
        }

    def predictions(self, sample_ids: Sequence[str]) -> pd.DataFrame:
        ids = np.asarray(sample_ids)
        frames = [
            pd.DataFrame(
                {
                    "model": f.model,
                    "repeat": f.repeat,
                    "fold": f.fold,
                    "sample_id": ids[f.test_index],
                    "y_true": f.y_true,
                    "y_score": f.y_score,
                }
            )
            for f in self.folds
        ]
        return pd.concat(frames, ignore_index=True)

    def importance_table(self) -> pd.DataFrame:
        """Per-feature stability summary across all outer folds (zeros when not selected)."""
        n = len(self.folds)
        imp = pd.concat([f.importance.rename(i) for i, f in enumerate(self.folds)], axis=1).fillna(
            0.0
        )
        table = pd.DataFrame(
            {
                "selection_frequency": (imp != 0).sum(axis=1) / n,
                "mean_importance": imp.sum(axis=1) / n,
                "mean_abs_importance": imp.abs().sum(axis=1) / n,
            }
        )
        if self.model == "elastic_net":
            # Fraction of selections where the coefficient is positive (ALS-associated).
            table["sign_consistency"] = (imp > 0).sum(axis=1) / (imp != 0).sum(axis=1)
        table.index.name = "feature"
        return table.sort_values(["mean_abs_importance", "selection_frequency"], ascending=False)

    def jaccard_stability(self) -> float:
        """Mean pairwise Jaccard index between the selected-feature sets of all folds."""
        sets = [set(f.importance.index[f.importance != 0]) for f in self.folds]
        pairs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
        return float(np.mean(pairs)) if pairs else float("nan")


def _flat(params: dict[str, Any]) -> dict[str, Any]:
    return {f"param_{k.removeprefix('clf__')}": v for k, v in params.items()}


def build_search_space(
    model: ModelName, pp: PreprocessConfig, sel: SelectionConfig
) -> tuple[Pipeline, dict[str, list[Any]]]:
    clf: Any
    if model == "elastic_net":
        clf = LogisticRegression(
            l1_ratio=sel.elastic_net.l1_ratio[0],
            solver="saga",
            class_weight="balanced",
            max_iter=sel.elastic_net.max_iter,
            random_state=sel.seed,
        )
        grid: dict[str, list[Any]] = {
            "clf__C": list(sel.elastic_net.C),
            "clf__l1_ratio": list(sel.elastic_net.l1_ratio),
        }
    else:
        from xgboost import XGBClassifier

        xgb = sel.xgboost
        clf = XGBClassifier(
            n_estimators=xgb.n_estimators,
            learning_rate=xgb.learning_rate,
            subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
            importance_type="gain",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=sel.seed,
        )
        grid = {"clf__max_depth": list(xgb.max_depth)}
    pipe = Pipeline(
        [
            ("filter", LowCountFilter(pp.min_count, pp.min_sample_fraction)),
            ("normalize", MedianOfRatiosLog()),
            ("select", SelectKBest(f_classif, k=sel.k_features)),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe, grid


def selected_importance(pipe: Pipeline, feature_names: NDArray[np.str_], model: str) -> pd.Series:
    """Map the fitted classifier's coefficients / gains back to feature names."""
    kept = np.flatnonzero(pipe.named_steps["filter"].support_)
    kept = kept[pipe.named_steps["select"].get_support()]
    clf = pipe.named_steps["clf"]
    values = clf.coef_.ravel() if model == "elastic_net" else clf.feature_importances_
    series = pd.Series(values, index=feature_names[kept], dtype=float)
    return series[series != 0]


def classification_metrics(
    y_true: NDArray[np.int_], y_score: NDArray[np.float64]
) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def outer_splits(
    y: NDArray[np.int_], groups: NDArray[Any], n_folds: int, repeats: int, seed: int
) -> list[tuple[int, int, NDArray[np.intp], NDArray[np.intp]]]:
    splits = []
    for repeat in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            splits.append((repeat, fold, train, test))
    return splits


def fit_fold(
    model: ModelName,
    X: NDArray[np.number],
    y: NDArray[np.int_],
    groups: NDArray[Any],
    feature_names: NDArray[np.str_],
    split: tuple[int, int, NDArray[np.intp], NDArray[np.intp]],
    pp: PreprocessConfig,
    sel: SelectionConfig,
) -> FoldResult:
    repeat, fold, train, test = split
    for part, idx in (("train", train), ("test", test)):
        if len(np.unique(y[idx])) < 2:
            raise ValueError(f"Outer {part} fold {repeat}/{fold} contains a single class")
    pipe, grid = build_search_space(model, pp, sel)
    inner = StratifiedGroupKFold(n_splits=sel.inner_folds, shuffle=True, random_state=sel.seed)
    search = GridSearchCV(pipe, grid, scoring=sel.scoring, cv=inner, n_jobs=1, refit=True)
    fit_params: dict[str, Any] = {"groups": groups[train]}
    if model == "xgboost":
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y[train])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=r"k=\d+ is greater than n_features")
        search.fit(X[train], y[train], **fit_params)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    y_score = search.predict_proba(X[test])[:, 1]
    return FoldResult(
        model=model,
        repeat=repeat,
        fold=fold,
        test_index=test,
        y_true=y[test],
        y_score=y_score,
        metrics=classification_metrics(y[test], y_score),
        best_params=search.best_params_,
        importance=selected_importance(search.best_estimator_, feature_names, model),
        n_convergence_warnings=n_conv,
    )


def nested_cv(
    model: ModelName,
    X: NDArray[np.number],
    y: NDArray[np.int_],
    groups: NDArray[Any],
    feature_names: Sequence[str],
    pp: PreprocessConfig,
    sel: SelectionConfig,
    repeats: int | None = None,
) -> CVResult:
    names = np.asarray(feature_names, dtype=str)
    splits = outer_splits(y, groups, sel.outer_folds, repeats or sel.outer_repeats, sel.seed)
    logger.info("%s: nested CV over %d outer folds", model, len(splits))
    folds = Parallel(n_jobs=sel.n_jobs)(
        delayed(fit_fold)(model, X, y, groups, names, split, pp, sel) for split in splits
    )
    result = CVResult(model=model, folds=list(folds))
    summary = result.metrics_summary()
    n_conv = sum(f.n_convergence_warnings for f in result.folds)
    logger.info(
        "%s: ROC AUC %.3f +/- %.3f, balanced accuracy %.3f +/- %.3f%s",
        model,
        summary["roc_auc"]["mean"],
        summary["roc_auc"]["std"],
        summary["balanced_accuracy"]["mean"],
        summary["balanced_accuracy"]["std"],
        f" ({n_conv} convergence warnings)" if n_conv else "",
    )
    return result


def permute_subject_labels(
    y: NDArray[np.int_], groups: NDArray[Any], rng: np.random.Generator
) -> NDArray[np.int_]:
    """Shuffle labels between subjects (labels stay constant within a subject)."""
    subjects, first = np.unique(groups, return_index=True)
    shuffled = dict(zip(subjects, rng.permutation(y[first]), strict=True))
    return np.array([shuffled[g] for g in groups])


def permutation_test(
    model: ModelName,
    X: NDArray[np.number],
    y: NDArray[np.int_],
    groups: NDArray[Any],
    feature_names: Sequence[str],
    pp: PreprocessConfig,
    sel: SelectionConfig,
    observed_auc: float,
) -> dict[str, Any]:
    """Re-run the whole nested CV on subject-permuted labels to get a null AUC distribution.

    ``observed_auc`` must come from the same scheme (one repeat with seed ``sel.seed``).
    """
    n_perm = sel.permutation.n_permutations
    rng = np.random.default_rng(sel.seed)
    names = np.asarray(feature_names, dtype=str)
    tasks = []
    for p in range(n_perm):
        y_perm = permute_subject_labels(y, groups, rng)
        for split in outer_splits(y_perm, groups, sel.outer_folds, 1, sel.seed):
            tasks.append((p, y_perm, split))
    logger.info("%s: permutation test with %d permutations (%d fits)", model, n_perm, len(tasks))
    folds = Parallel(n_jobs=sel.n_jobs)(
        delayed(fit_fold)(model, X, y_perm, groups, names, split, pp, sel)
        for _, y_perm, split in tasks
    )
    per_perm: dict[int, list[float]] = {}
    for (p, _, _), fold in zip(tasks, folds, strict=True):
        per_perm.setdefault(p, []).append(fold.metrics["roc_auc"])
    null = np.array([np.mean(v) for _, v in sorted(per_perm.items())])
    p_value = (1 + int((null >= observed_auc).sum())) / (n_perm + 1)
    logger.info(
        "%s: observed AUC %.3f vs null %.3f +/- %.3f -> permutation p = %.4f",
        model,
        observed_auc,
        null.mean(),
        null.std(ddof=1) if n_perm > 1 else 0.0,
        p_value,
    )
    return {
        "model": model,
        "n_permutations": n_perm,
        "observed_roc_auc": observed_auc,
        "null_roc_auc": null.tolist(),
        "null_mean": float(null.mean()),
        "p_value": p_value,
    }
