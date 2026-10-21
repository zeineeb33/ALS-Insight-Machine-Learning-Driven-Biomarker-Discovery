"""Consensus biomarker ranking combining univariate (DE) and multivariate (model) evidence."""

from __future__ import annotations

import pandas as pd

from als_insight.config import EvidenceSource


def _normalized_rank(values: pd.Series, ascending: bool) -> pd.Series:
    """Rank in (0, 1]; ties share their average rank, missing values rank last."""
    ranks = values.rank(ascending=ascending, method="average", na_option="bottom")
    return ranks / len(values)


def consensus_ranking(
    de: pd.DataFrame,
    elastic_net: pd.DataFrame,
    xgboost: pd.DataFrame,
    weights: dict[EvidenceSource, float],
) -> pd.DataFrame:
    """Weighted mean of normalised ranks over the features tested for DE.

    * DE evidence: Wald-test p-value (smaller is better).
    * Elastic net: mean |standardised coefficient| over all outer folds (0 when not
      selected), which rewards both effect size and selection stability.
    * XGBoost: mean gain importance over all outer folds (0 when unused).

    Features never selected by a model share the same (worst) rank for that source, so
    the consensus favours genes supported by several independent lines of evidence.
    """
    table = de.copy()
    en = elastic_net.reindex(table.index)
    xgb = xgboost.reindex(table.index)
    table["en_selection_frequency"] = en["selection_frequency"].fillna(0.0)
    table["en_mean_coef"] = en["mean_importance"].fillna(0.0)
    table["en_mean_abs_coef"] = en["mean_abs_importance"].fillna(0.0)
    table["xgb_selection_frequency"] = xgb["selection_frequency"].fillna(0.0)
    table["xgb_mean_gain"] = xgb["mean_importance"].fillna(0.0)

    ranks: dict[EvidenceSource, pd.Series] = {
        "de": _normalized_rank(table["pvalue"], ascending=True),
        "elastic_net": _normalized_rank(table["en_mean_abs_coef"], ascending=False),
        "xgboost": _normalized_rank(table["xgb_mean_gain"], ascending=False),
    }
    table["de_rank"] = table["pvalue"].rank(method="min", na_option="bottom").astype(int)
    table["en_rank"] = table["en_mean_abs_coef"].rank(ascending=False, method="min").astype(int)
    table["xgb_rank"] = table["xgb_mean_gain"].rank(ascending=False, method="min").astype(int)

    total = sum(weights.values())
    score = sum(weights.get(src, 0.0) * r for src, r in ranks.items()) / total
    table["consensus_score"] = score
    table = table.sort_values(["consensus_score", "pvalue"])
    table["consensus_rank"] = range(1, len(table) + 1)
    return table
