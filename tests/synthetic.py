"""Synthetic GEO data: tiny MINiML files and count archives (no network access)."""

from __future__ import annotations

import gzip
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ACCESSION = "GSE000001"
CASE = "ALS Spectrum MND"
CONTROL = "Non-Neurological Control"
OTHER = "Other Neurological Disorders"
REGIONS = ("Frontal Cortex", "Motor Cortex (Medial)")


@dataclass(frozen=True)
class SyntheticSample:
    gsm: str
    group: str
    subject: str
    region: str


def miniml_xml(samples: list[SyntheticSample]) -> str:
    blocks = []
    for s in samples:
        blocks.append(
            f"""
  <Sample iid="{s.gsm}">
    <Title>{s.gsm}-title</Title>
    <Channel position="1">
      <Source>{s.region}</Source>
      <Characteristics tag="tissue type">
post-mortem tissue
      </Characteristics>
      <Characteristics tag="cns subregion">
{s.region}
      </Characteristics>
      <Characteristics tag="subject id">
{s.subject}
      </Characteristics>
      <Characteristics tag="sample group">
{s.group}
      </Characteristics>
    </Channel>
  </Sample>"""
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML" version="0.5.0">'
        + "".join(blocks)
        + "\n</MINiML>\n"
    )


def count_table(features: list[str], counts: np.ndarray) -> bytes:
    lines = ["gene/TE\t../mapping/sample.bam"]
    lines += [f'"{f}"\t{int(c)}' for f, c in zip(features, counts, strict=True)]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


@dataclass
class SyntheticGEO:
    raw_dir: Path
    samples: list[SyntheticSample]
    features: list[str]
    counts: np.ndarray  # samples x features
    signal_features: list[str]


def make_synthetic_geo(
    raw_dir: Path,
    *,
    n_case_subjects: int = 20,
    n_control_subjects: int = 12,
    n_other_subjects: int = 2,
    n_genes: int = 300,
    n_te: int = 20,
    n_signal: int = 10,
    fold_change: float = 4.0,
    seed: int = 0,
) -> SyntheticGEO:
    """Negative-binomial counts with ``n_signal`` genes up-regulated in the case group.

    Every subject contributes two samples (two cortex regions), mimicking GSE124439.
    """
    rng = np.random.default_rng(seed)
    samples: list[SyntheticSample] = []
    gsm = 1000
    for group, n_subjects, prefix in (
        (CASE, n_case_subjects, "ALS"),
        (CONTROL, n_control_subjects, "CTL"),
        (OTHER, n_other_subjects, "OND"),
    ):
        for subj in range(n_subjects):
            for region in REGIONS:
                samples.append(SyntheticSample(f"GSM{gsm}", group, f"{prefix}{subj:03d}", region))
                gsm += 1
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    tes = [f"TE{i:03d}:L1:LINE" for i in range(n_te)]
    features = genes + tes
    base = rng.lognormal(mean=4.0, sigma=1.5, size=len(features))
    base[:5] = 0.05  # a few near-zero features that the low-count filter must drop
    signal_idx = np.arange(10, 10 + n_signal)
    is_case = np.array([s.group == CASE for s in samples])
    subject_effect = {s.subject: rng.normal(0, 0.15) for s in samples}
    size_factors = rng.uniform(0.6, 1.6, size=len(samples))
    mu = np.outer(size_factors, base) * 50
    mu *= np.exp([subject_effect[s.subject] for s in samples])[:, None]
    mu[np.ix_(is_case, signal_idx)] *= fold_change
    dispersion = 0.1
    p = 1 / (1 + mu * dispersion)
    counts = rng.negative_binomial(1 / dispersion, p)
    members = {
        f"{s.gsm}_CGND-HRA-{i:05d}_counts.txt.gz": count_table(features, counts[i])
        for i, s in enumerate(samples)
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_tar(raw_dir / f"{ACCESSION}_RAW.tar", members)
    (raw_dir / f"{ACCESSION}_family.xml").write_text(miniml_xml(samples))
    return SyntheticGEO(
        raw_dir=raw_dir,
        samples=samples,
        features=features,
        counts=counts,
        signal_features=[features[i] for i in signal_idx],
    )


def config_dict(root: Path, n_samples: int | None = None) -> dict[str, Any]:
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE000nnn/GSE000001"
    return {
        "dataset": {
            "accession": ACCESSION,
            "miniml_url": f"{base}/miniml/{ACCESSION}_family.xml.tgz",
            "counts_url": f"{base}/suppl/{ACCESSION}_RAW.tar",
            "expected_samples": n_samples,
        },
        "paths": {
            "data_dir": str(root / "data"),
            "artifacts_dir": str(root / "artifacts"),
            "results_dir": str(root / "results"),
        },
        "contrast": {"case": CASE, "control": CONTROL},
        "preprocess": {
            "min_library_size": 1000,
            "region_map": {"Frontal Cortex": "frontal", "Motor Cortex (Medial)": "motor"},
        },
        "exploration": {"n_components": 4, "outlier_components": 3},
        "differential": {"n_cpus": 1},
        "selection": {
            "outer_folds": 3,
            "outer_repeats": 1,
            "inner_folds": 2,
            "k_features": 50,
            "n_jobs": 1,
            "elastic_net": {"C": [0.1, 1.0], "l1_ratio": [0.5], "max_iter": 2000},
            "xgboost": {"max_depth": [2], "n_estimators": 50},
            "permutation": {"n_permutations": 2},
        },
        "consensus": {"top_n": 20},
        "report": {"top_n_table": 10},
    }
