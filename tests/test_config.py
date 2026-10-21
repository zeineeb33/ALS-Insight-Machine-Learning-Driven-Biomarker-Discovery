from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from als_insight.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_valid() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.dataset.accession == "GSE124439"
    assert cfg.dataset.counts_sha256 is not None
    assert cfg.raw_dir == Path("data/raw/GSE124439")
    assert set(cfg.preprocess.region_map.values()) == {"frontal", "motor"}


def test_overrides_are_deep_merged() -> None:
    cfg = load_config(
        REPO_ROOT / "configs" / "default.yaml",
        overrides={"selection": {"outer_folds": 3}},
    )
    assert cfg.selection.outer_folds == 3
    assert cfg.selection.inner_folds == 3  # untouched sibling keys survive


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text((REPO_ROOT / "configs" / "default.yaml").read_text() + "\nunknown: 1\n")
    with pytest.raises(ValidationError):
        load_config(bad)


@pytest.mark.parametrize(
    "patch",
    [
        {"contrast": {"case": "A", "control": "A"}},
        {"selection": {"elastic_net": {"l1_ratio": [1.5]}}},
        {"exploration": {"n_components": 2, "outlier_components": 3}},
        {"consensus": {"weights": {"de": 0.0}}},
        {"dataset": {"accession": "not-an-accession"}},
    ],
)
def test_invalid_values_are_rejected(patch: dict[str, dict[str, object]]) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").model_dump(mode="json")
    for section, values in patch.items():
        cfg[section] = {**cfg[section], **values}
    with pytest.raises(ValidationError):
        Config.model_validate(cfg)


def test_section_fingerprint_changes_with_config() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    other = load_config(
        REPO_ROOT / "configs" / "default.yaml", overrides={"preprocess": {"min_count": 5}}
    )
    assert cfg.section_fingerprint("preprocess") != other.section_fingerprint("preprocess")
    assert cfg.section_fingerprint("selection") == other.section_fingerprint("selection")
