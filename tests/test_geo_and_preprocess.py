from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from als_insight import cache, preprocess
from als_insight.config import Config
from als_insight.geo import is_transposable_element, parse_miniml, read_count_archive
from synthetic import CASE, CONTROL, SyntheticGEO, count_table, write_tar


def test_parse_miniml(synthetic_geo: SyntheticGEO) -> None:
    meta = parse_miniml(synthetic_geo.raw_dir / "GSE000001_family.xml")
    assert len(meta) == len(synthetic_geo.samples)
    assert {"sample_group", "subject_id", "cns_subregion", "tissue_type"} <= set(meta.columns)
    first = synthetic_geo.samples[0]
    row = meta.loc[first.gsm]
    assert row["sample_group"] == first.group
    assert row["subject_id"] == first.subject
    assert row["tissue_type"] == "post-mortem tissue"  # whitespace/newlines stripped


def test_parse_miniml_without_samples(tmp_path: Path) -> None:
    xml = tmp_path / "empty.xml"
    xml.write_text('<MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML"/>')
    with pytest.raises(ValueError, match="No <Sample>"):
        parse_miniml(xml)


def test_read_count_archive(synthetic_geo: SyntheticGEO) -> None:
    matrix = read_count_archive(synthetic_geo.raw_dir / "GSE000001_RAW.tar")
    assert matrix.shape == synthetic_geo.counts.shape
    assert list(matrix.columns) == synthetic_geo.features  # quotes stripped from feature names
    first = synthetic_geo.samples[0].gsm
    np.testing.assert_array_equal(matrix.loc[first].to_numpy(), synthetic_geo.counts[0])


def test_read_count_archive_rejects_mismatched_features(tmp_path: Path) -> None:
    tar = tmp_path / "bad.tar"
    write_tar(
        tar,
        {
            "GSM1_a_counts.txt.gz": count_table(["A", "B"], np.array([1, 2])),
            "GSM2_b_counts.txt.gz": count_table(["A", "C"], np.array([1, 2])),
        },
    )
    with pytest.raises(ValueError, match="feature set differs"):
        read_count_archive(tar)


def test_read_count_archive_rejects_negative_counts(tmp_path: Path) -> None:
    tar = tmp_path / "neg.tar"
    write_tar(tar, {"GSM1_a_counts.txt.gz": count_table(["A", "B"], np.array([1, -2]))})
    with pytest.raises(ValueError, match="negative"):
        read_count_archive(tar)


def test_read_count_archive_ignores_unrelated_members(tmp_path: Path) -> None:
    tar = tmp_path / "mixed.tar"
    write_tar(
        tar,
        {
            "GSM1_a_counts.txt.gz": count_table(["A"], np.array([3])),
            "README.txt": b"not a count file",
        },
    )
    assert read_count_archive(tar).shape == (1, 1)


def test_transposable_element_detection() -> None:
    mask = is_transposable_element(pd.Index(["A1BG", "L1HS:L1:LINE"]))
    assert mask.tolist() == [False, True]


def test_preprocess_step(synthetic_config: Config, synthetic_geo: SyntheticGEO) -> None:
    out_dir = preprocess.run(synthetic_config)
    ds = preprocess.load_dataset(out_dir)
    n_expected = sum(s.group in {CASE, CONTROL} for s in synthetic_geo.samples)
    assert len(ds.metadata) == n_expected  # "other" group dropped
    assert set(ds.metadata["condition"]) == {"ALS", "Control"}
    assert set(ds.metadata["region"]) == {"frontal", "motor"}
    assert ds.counts.index.equals(ds.metadata.index)
    assert ds.y.sum() == (ds.metadata["sample_group"] == CASE).sum()
    assert (ds.features["feature_type"] == "TE").sum() == 20
    # near-zero features planted by the fixture are flagged by the low-count filter
    assert not ds.features.loc[synthetic_geo.features[:5], "passes_filter"].any()
    fp = (out_dir / cache.STEP_FILE).read_text()
    preprocess.run(synthetic_config)  # cached: no recomputation, record unchanged
    assert (out_dir / cache.STEP_FILE).read_text() == fp


def test_preprocess_can_drop_transposable_elements(
    synthetic_config: Config, synthetic_geo: SyntheticGEO
) -> None:
    cfg = synthetic_config.model_copy(
        update={
            "preprocess": synthetic_config.preprocess.model_copy(
                update={"include_transposable_elements": False}
            )
        }
    )
    ds = preprocess.load_dataset(preprocess.run(cfg))
    assert (ds.features["feature_type"] == "gene").all()


def test_preprocess_validates_expected_sample_count(
    synthetic_config: Config, synthetic_geo: SyntheticGEO
) -> None:
    cfg = synthetic_config.model_copy(
        update={"dataset": synthetic_config.dataset.model_copy(update={"expected_samples": 3})}
    )
    with pytest.raises(ValueError, match="Expected 3 annotated samples"):
        preprocess.run(cfg)


def test_preprocess_requires_download(tmp_path: Path, synthetic_config: Config) -> None:
    cfg = synthetic_config.model_copy(
        update={"paths": synthetic_config.paths.model_copy(update={"data_dir": tmp_path / "x"})}
    )
    with pytest.raises(FileNotFoundError, match="als-insight download"):
        preprocess.run(cfg)
