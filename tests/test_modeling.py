"""Nested CV, grouping, stability and leakage checks on synthetic count data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from als_insight.config import PreprocessConfig, SelectionConfig
from als_insight.consensus import consensus_ranking
from als_insight.modeling import (
    nested_cv,
    outer_splits,
    permutation_test,
    permute_subject_labels,
)
from synthetic import CASE, CONTROL, make_synthetic_geo

PP = PreprocessConfig()
SEL = SelectionConfig.model_validate(
    {
        "outer_folds": 3,
        "outer_repeats": 1,
        "inner_folds": 2,
        "k_features": 30,
        "n_jobs": 1,
        "elastic_net": {"C": [0.2, 1.0], "l1_ratio": [0.5], "max_iter": 2000},
        "xgboost": {"max_depth": [2], "n_estimators": 40},
        "permutation": {"n_permutations": 2},
    }
)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    geo = make_synthetic_geo(tmp_path_factory.mktemp("geo"), n_genes=200, fold_change=3.0)
    keep = [i for i, s in enumerate(geo.samples) if s.group in {CASE, CONTROL}]
    return {
        "X": geo.counts[keep].astype(float),
        "y": np.array([int(geo.samples[i].group == CASE) for i in keep]),
        "groups": np.array([geo.samples[i].subject for i in keep]),
        "features": geo.features,
        "signal": geo.signal_features,
    }


def test_outer_splits_never_split_a_subject(data: dict) -> None:
    splits = outer_splits(data["y"], data["groups"], n_folds=3, repeats=2, seed=0)
    assert len(splits) == 6
    for _, _, train, test in splits:
        assert not set(data["groups"][train]) & set(data["groups"][test])
        assert set(data["y"][test]) == {0, 1}


@pytest.mark.parametrize("model", ["elastic_net", "xgboost"])
def test_nested_cv_finds_planted_signal(data: dict, model: str) -> None:
    result = nested_cv(model, data["X"], data["y"], data["groups"], data["features"], PP, SEL)
    summary = result.metrics_summary()
    assert summary["roc_auc"]["mean"] > 0.9
    importance = result.importance_table()
    top = set(importance.index[:15])
    assert len(top & set(data["signal"])) >= 5
    assert 0 < result.jaccard_stability() <= 1
    preds = result.predictions([f"s{i}" for i in range(len(data["y"]))])
    assert preds["sample_id"].is_unique  # every sample predicted exactly once per repeat


def test_no_leakage_on_pure_noise(tmp_path_factory: pytest.TempPathFactory) -> None:
    """With labels independent of the data, in-fold feature selection must give AUC ~ 0.5.

    Selecting the top-k genes on the full dataset before CV would make this AUC
    spuriously high; the pipeline re-fits the selection inside each training fold.
    """
    geo = make_synthetic_geo(
        tmp_path_factory.mktemp("noise"),
        n_case_subjects=15,
        n_control_subjects=15,
        n_other_subjects=0,
        n_genes=400,
        n_signal=0,
        seed=7,
    )
    rng = np.random.default_rng(0)
    groups = np.array([s.subject for s in geo.samples])
    y = permute_subject_labels(np.array([int(s.group == CASE) for s in geo.samples]), groups, rng)
    sel = SEL.model_copy(update={"outer_repeats": 3})
    result = nested_cv("elastic_net", geo.counts.astype(float), y, groups, geo.features, PP, sel)
    assert result.metrics_summary()["roc_auc"]["mean"] < 0.7


def test_permute_subject_labels_keeps_subjects_consistent() -> None:
    groups = np.array(["a", "a", "b", "b", "c", "d"])
    y = np.array([1, 1, 0, 0, 1, 0])
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = permute_subject_labels(y, groups, rng)
        assert perm[0] == perm[1] and perm[2] == perm[3]
        assert sorted(perm[[0, 2, 4, 5]]) == [0, 0, 1, 1]


def test_permutation_test(data: dict) -> None:
    out = permutation_test(
        "elastic_net",
        data["X"],
        data["y"],
        data["groups"],
        data["features"],
        PP,
        SEL,
        observed_auc=0.99,
    )
    assert out["n_permutations"] == 2
    assert len(out["null_roc_auc"]) == 2
    assert 1 / 3 <= out["p_value"] <= 1


def test_consensus_ranking_combines_evidence() -> None:
    de = pd.DataFrame(
        {"pvalue": [1e-8, 1e-3, 0.5, 0.9], "padj": [1e-6, 0.01, 0.8, 0.95]},
        index=["G_all", "G_de_only", "G_model_only", "G_none"],
    )
    en = pd.DataFrame(
        {
            "selection_frequency": [1.0, 0.8],
            "mean_importance": [0.5, -0.4],
            "mean_abs_importance": [0.5, 0.4],
        },
        index=["G_all", "G_model_only"],
    )
    xgb = pd.DataFrame(
        {
            "selection_frequency": [1.0, 1.0],
            "mean_importance": [3.0, 2.0],
            "mean_abs_importance": [3.0, 2.0],
        },
        index=["G_all", "G_model_only"],
    )
    table = consensus_ranking(de, en, xgb, {"de": 1.0, "elastic_net": 1.0, "xgboost": 1.0})
    assert table.index[0] == "G_all"
    assert table.index[-1] == "G_none"
    assert table.loc["G_de_only", "en_selection_frequency"] == 0.0
    assert table["consensus_rank"].tolist() == [1, 2, 3, 4]

    de_only = consensus_ranking(de, en, xgb, {"de": 1.0, "elastic_net": 0.0, "xgboost": 0.0})
    assert de_only.index.tolist() == ["G_all", "G_de_only", "G_model_only", "G_none"]
