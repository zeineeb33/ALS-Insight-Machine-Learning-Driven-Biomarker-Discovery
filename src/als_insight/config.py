"""Typed, validated pipeline configuration loaded from YAML."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    accession: str = Field(pattern=r"^GSE\d+$")
    miniml_url: HttpUrl
    counts_url: HttpUrl
    counts_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    miniml_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_samples: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")


class ContrastConfig(_Strict):
    """Which sample groups are compared. Samples from any other group are dropped."""

    case: str
    control: str
    case_label: str = Field(default="ALS", pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    control_label: str = Field(default="Control", pattern=r"^[A-Za-z][A-Za-z0-9]*$")

    @model_validator(mode="after")
    def _distinct(self) -> ContrastConfig:
        if self.case == self.control or self.case_label == self.control_label:
            raise ValueError("case and control groups must differ")
        return self


class PreprocessConfig(_Strict):
    include_transposable_elements: bool = True
    min_library_size: int = Field(default=1_000_000, ge=0)
    # Gene-level filter (applied for DE, and re-fitted inside every CV fold for ML).
    min_count: int = Field(default=10, ge=1)
    min_sample_fraction: float = Field(default=0.1, gt=0, le=1)
    # Maps raw "cns subregion" values to a coarse covariate used in the DE design.
    region_map: dict[str, str] = Field(default_factory=dict)


class ExplorationConfig(_Strict):
    n_components: int = Field(default=10, ge=2)
    outlier_components: int = Field(default=4, ge=1)
    outlier_robust_z: float = Field(default=3.5, gt=0)
    remove_outliers: bool = True

    @model_validator(mode="after")
    def _components(self) -> ExplorationConfig:
        if self.outlier_components > self.n_components:
            raise ValueError("outlier_components must be <= n_components")
        return self


class DifferentialConfig(_Strict):
    covariates: list[str] = Field(default_factory=lambda: ["region"])
    alpha: float = Field(default=0.05, gt=0, lt=1)
    lfc_threshold: float = Field(default=1.0, ge=0)
    shrink_lfc: bool = True
    n_cpus: int = Field(default=4, ge=1)


class ElasticNetConfig(_Strict):
    C: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0], min_length=1)
    l1_ratio: list[float] = Field(default_factory=lambda: [0.5, 0.9], min_length=1)
    max_iter: int = Field(default=5000, ge=100)

    @field_validator("l1_ratio")
    @classmethod
    def _ratio(cls, v: list[float]) -> list[float]:
        if any(not 0 <= r <= 1 for r in v):
            raise ValueError("l1_ratio values must be in [0, 1]")
        return v


class XGBoostConfig(_Strict):
    max_depth: list[int] = Field(default_factory=lambda: [2, 3], min_length=1)
    n_estimators: int = Field(default=300, ge=10)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.5, gt=0, le=1)


class PermutationConfig(_Strict):
    n_permutations: int = Field(default=0, ge=0)
    model: Literal["elastic_net", "xgboost"] = "elastic_net"


class SelectionConfig(_Strict):
    outer_folds: int = Field(default=5, ge=2)
    outer_repeats: int = Field(default=3, ge=1)
    inner_folds: int = Field(default=3, ge=2)
    k_features: int = Field(default=2000, ge=1)
    seed: int = 42
    n_jobs: int = -1
    scoring: Literal["roc_auc", "balanced_accuracy", "average_precision"] = "roc_auc"
    elastic_net: ElasticNetConfig = ElasticNetConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    permutation: PermutationConfig = PermutationConfig()


EvidenceSource = Literal["de", "elastic_net", "xgboost"]


def _equal_weights() -> dict[EvidenceSource, float]:
    return {"de": 1.0, "elastic_net": 1.0, "xgboost": 1.0}


class ConsensusConfig(_Strict):
    top_n: int = Field(default=100, ge=1)
    weights: dict[EvidenceSource, float] = Field(default_factory=_equal_weights)

    @field_validator("weights")
    @classmethod
    def _positive(cls, v: dict[EvidenceSource, float]) -> dict[EvidenceSource, float]:
        if not v or any(w < 0 for w in v.values()) or sum(v.values()) == 0:
            raise ValueError("weights must be non-negative and not all zero")
        return v


class ReportConfig(_Strict):
    top_n_table: int = Field(default=25, ge=1)


class Config(_Strict):
    dataset: DatasetConfig
    contrast: ContrastConfig
    paths: PathsConfig = PathsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    differential: DifferentialConfig = DifferentialConfig()
    selection: SelectionConfig = SelectionConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    report: ReportConfig = ReportConfig()

    @property
    def raw_dir(self) -> Path:
        return self.paths.data_dir / "raw" / self.dataset.accession

    def step_dir(self, step: str) -> Path:
        return self.paths.artifacts_dir / step

    def section_fingerprint(self, *sections: str) -> str:
        """Stable hash of the given config sections, used to invalidate cached steps."""
        payload = {s: getattr(self, s).model_dump(mode="json") for s in sections}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and validate it; ``overrides`` are deep-merged on top."""
    with Path(path).open() as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
