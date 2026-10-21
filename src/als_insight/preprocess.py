"""Join counts with annotations, restrict to the configured contrast and run sample/gene QC."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from als_insight import cache
from als_insight.config import Config
from als_insight.download import raw_files
from als_insight.geo import is_transposable_element, parse_miniml, read_count_archive

logger = logging.getLogger(__name__)

COUNTS_FILE = "counts.parquet"
METADATA_FILE = "metadata.csv"
FEATURES_FILE = "features.csv"
SUMMARY_FILE = "summary.json"
OUTPUTS = (COUNTS_FILE, METADATA_FILE, FEATURES_FILE, SUMMARY_FILE)


@dataclass(frozen=True)
class Dataset:
    """Samples x features raw counts with aligned per-sample metadata."""

    counts: pd.DataFrame
    metadata: pd.DataFrame
    features: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        """Binary target: 1 for the case group, 0 for controls."""
        return (self.metadata["target"].to_numpy() == 1).astype(int)

    @property
    def groups(self) -> np.ndarray:
        return self.metadata["subject_id"].to_numpy()


def low_count_mask(counts: np.ndarray, min_count: int, min_sample_fraction: float) -> np.ndarray:
    """Keep features with >= ``min_count`` reads in at least a fraction of samples."""
    min_samples = max(1, math.ceil(min_sample_fraction * counts.shape[0]))
    return np.asarray((counts >= min_count).sum(axis=0) >= min_samples)


def build_dataset(counts: pd.DataFrame, annotations: pd.DataFrame, cfg: Config) -> Dataset:
    """Pure transformation from parsed GEO inputs to a QC'd, contrast-restricted dataset."""
    contrast, pp = cfg.contrast, cfg.preprocess
    expected = cfg.dataset.expected_samples
    if expected is not None and len(annotations) != expected:
        raise ValueError(f"Expected {expected} annotated samples, found {len(annotations)}")
    missing_counts = annotations.index.difference(counts.index)
    missing_annot = counts.index.difference(annotations.index)
    if len(missing_counts) or len(missing_annot):
        raise ValueError(
            f"Samples without counts: {list(missing_counts)}; "
            f"count files without annotation: {list(missing_annot)}"
        )
    for column in ("sample_group", "subject_id", "cns_subregion"):
        if column not in annotations.columns:
            raise ValueError(f"MINiML annotations lack the '{column}' characteristic")

    meta = annotations.loc[annotations["sample_group"].isin([contrast.case, contrast.control])]
    dropped_groups = annotations.loc[~annotations.index.isin(meta.index), "sample_group"]
    if len(dropped_groups):
        logger.info(
            "Dropping samples outside the contrast: %s", dropped_groups.value_counts().to_dict()
        )
    for group in (contrast.case, contrast.control):
        if not (meta["sample_group"] == group).any():
            raise ValueError(f"No samples found for group '{group}'")

    meta = meta.copy()
    meta["target"] = (meta["sample_group"] == contrast.case).astype(int)
    meta["condition"] = np.where(meta["target"] == 1, contrast.case_label, contrast.control_label)
    if pp.region_map:
        unmapped = set(meta["cns_subregion"]) - set(pp.region_map)
        if unmapped:
            raise ValueError(f"region_map has no entry for {sorted(unmapped)}")
        meta["region"] = meta["cns_subregion"].map(pp.region_map)
    else:
        meta["region"] = meta["cns_subregion"]

    x = counts.loc[meta.index]
    te_mask = is_transposable_element(x.columns)
    if not pp.include_transposable_elements:
        x = x.loc[:, ~te_mask]
        te_mask = np.zeros(x.shape[1], dtype=bool)

    meta["library_size"] = x.sum(axis=1)
    meta["detected_features"] = (x > 0).sum(axis=1)
    meta["te_fraction"] = x.loc[:, te_mask].sum(axis=1) / meta["library_size"]
    low_lib = meta["library_size"] < pp.min_library_size
    if low_lib.any():
        logger.warning(
            "Removing %d samples with library size < %d", low_lib.sum(), pp.min_library_size
        )
    meta = meta.loc[~low_lib]
    x = x.loc[meta.index]

    features = pd.DataFrame(
        {
            "feature_type": np.where(te_mask, "TE", "gene"),
            "total_counts": x.sum(axis=0).to_numpy(),
            "passes_filter": low_count_mask(x.to_numpy(), pp.min_count, pp.min_sample_fraction),
        },
        index=x.columns,
    )
    features.index.name = "feature"
    logger.info(
        "Dataset: %d samples (%s), %d features (%d pass the low-count filter)",
        len(meta),
        meta["condition"].value_counts().to_dict(),
        len(features),
        int(features["passes_filter"].sum()),
    )
    columns = [
        "condition",
        "target",
        "sample_group",
        "subject_id",
        "region",
        "cns_subregion",
        "library_size",
        "detected_features",
        "te_fraction",
    ]
    return Dataset(counts=x, metadata=meta[columns], features=features)


def save_dataset(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if ds.counts.to_numpy().max() >= np.iinfo(np.int32).max:
        raise OverflowError("counts exceed int32 range")
    ds.counts.astype(np.int32).to_parquet(out_dir / COUNTS_FILE)
    ds.metadata.to_csv(out_dir / METADATA_FILE)
    ds.features.to_csv(out_dir / FEATURES_FILE)
    meta = ds.metadata
    summary = {
        "n_samples": len(meta),
        "n_subjects": int(meta["subject_id"].nunique()),
        "samples_per_condition": meta["condition"].value_counts().to_dict(),
        "subjects_per_condition": meta.groupby("condition")["subject_id"].nunique().to_dict(),
        "samples_per_region": meta["region"].value_counts().to_dict(),
        "n_features": int(ds.features.shape[0]),
        "n_features_te": int((ds.features["feature_type"] == "TE").sum()),
        "n_features_passing_filter": int(ds.features["passes_filter"].sum()),
        "library_size_median": float(meta["library_size"].median()),
        "library_size_min": int(meta["library_size"].min()),
        "library_size_max": int(meta["library_size"].max()),
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))


def load_dataset(out_dir: Path) -> Dataset:
    counts = pd.read_parquet(out_dir / COUNTS_FILE)
    metadata = pd.read_csv(out_dir / METADATA_FILE, index_col="sample_id")
    features = pd.read_csv(out_dir / FEATURES_FILE, index_col="feature")
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    return Dataset(counts=counts, metadata=metadata, features=features)


def run(cfg: Config, force: bool = False) -> Path:
    files = raw_files(cfg)
    out_dir = cfg.step_dir("preprocess")
    fp = cache.fingerprint(
        cfg.section_fingerprint("dataset", "contrast", "preprocess"),
        [files.miniml_xml, files.counts_tar],
    )
    if not force and cache.is_fresh(out_dir, fp, OUTPUTS):
        logger.info("preprocess: up to date (%s)", out_dir)
        return out_dir
    annotations = parse_miniml(files.miniml_xml)
    counts = read_count_archive(files.counts_tar)
    ds = build_dataset(counts, annotations, cfg)
    save_dataset(ds, out_dir)
    cache.mark_done(out_dir, fp, OUTPUTS)
    return out_dir
