"""Pydantic response models (they also drive the OpenAPI docs at ``/docs``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    results_version: str
    n_genes: int


class DifferentialExpression(BaseModel):
    base_mean: float | None = Field(description="Mean of normalised counts across samples")
    log2_fold_change: float | None = Field(
        description="apeGLM-shrunk log2 fold change, case vs control"
    )
    log2_fold_change_mle: float | None = Field(description="Unshrunk (MLE) log2 fold change")
    lfc_se: float | None
    pvalue: float | None = Field(description="Wald test p-value")
    padj: float | None = Field(description="Benjamini-Hochberg adjusted p-value")
    significant: bool = Field(description="padj below the analysis FDR threshold")
    rank: int = Field(description="Rank by p-value (1 = most significant)")


class ElasticNetEvidence(BaseModel):
    selection_frequency: float = Field(description="Fraction of outer CV folds selecting the gene")
    mean_coefficient: float = Field(description="Mean standardised coefficient (0 when unselected)")
    rank: int


class XGBoostEvidence(BaseModel):
    selection_frequency: float = Field(description="Fraction of outer CV folds using the gene")
    mean_gain: float = Field(description="Mean gain importance (0 when unused)")
    rank: int


class Consensus(BaseModel):
    score: float | None = Field(description="Weighted mean normalised rank (lower is better)")
    rank: int


class Gene(BaseModel):
    symbol: str
    feature_type: Literal["gene", "TE"]
    differential_expression: DifferentialExpression
    elastic_net: ElasticNetEvidence
    xgboost: XGBoostEvidence
    consensus: Consensus


class GeneMatch(BaseModel):
    symbol: str
    feature_type: str
    consensus_rank: int


class Biomarkers(BaseModel):
    results_version: str
    total_ranked: int
    items: list[Gene]


class VolcanoData(BaseModel):
    alpha: float
    lfc_threshold: float
    symbol: list[str]
    log2_fold_change: list[float]
    neg_log10_padj: list[float]
    significant: list[bool]


class PCASample(BaseModel):
    sample_id: str
    pc1: float
    pc2: float
    pc3: float | None = None
    condition: str
    region: str
    outlier: bool


class PCAData(BaseModel):
    explained_variance_ratio: list[float]
    samples: list[PCASample]


class NotFound(BaseModel):
    detail: str
    suggestions: list[str] = []


class Summary(BaseModel):
    results_version: str
    generated_at: str
    dataset: dict[str, Any]
    contrast: dict[str, Any]
    samples: dict[str, Any]
    differential_expression: dict[str, Any]
    classification: dict[str, Any]
    top_biomarkers: list[str]
    disclaimer: str
