"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from als_insight.config import Config
from synthetic import ACCESSION, SyntheticGEO, config_dict, make_synthetic_geo


@pytest.fixture
def synthetic_geo(tmp_path: Path) -> SyntheticGEO:
    return make_synthetic_geo(tmp_path / "data" / "raw" / ACCESSION)


@pytest.fixture
def synthetic_config(tmp_path: Path, synthetic_geo: SyntheticGEO) -> Config:
    return Config.model_validate(config_dict(tmp_path, n_samples=len(synthetic_geo.samples)))
