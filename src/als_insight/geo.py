"""Parsers for GEO MINiML sample annotations and per-sample count tables."""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MINIML_NS = "{http://www.ncbi.nlm.nih.gov/geo/info/MINiML}"
COUNT_FILE_RE = re.compile(r"^(?P<gsm>GSM\d+)_[^/]*counts\.txt(?:\.gz)?$")
# TEtranscripts encodes transposable elements as "name:family:class".
TE_SEPARATOR = ":"


def _snake(tag: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", tag.strip().lower()).strip("_")


def parse_miniml(xml_path: Path) -> pd.DataFrame:
    """Return one row per GEO sample with its title, source and all characteristics.

    Characteristic tags are normalised to snake_case column names, e.g.
    ``"sample group"`` -> ``sample_group``.
    """
    root = ET.parse(xml_path).getroot()
    rows: list[dict[str, str | None]] = []
    for sample in root.iter(f"{MINIML_NS}Sample"):
        row: dict[str, str | None] = {"sample_id": sample.attrib.get("iid")}
        title = sample.find(f"{MINIML_NS}Title")
        row["title"] = title.text.strip() if title is not None and title.text else None
        source = sample.find(f".//{MINIML_NS}Source")
        row["source"] = source.text.strip() if source is not None and source.text else None
        for char in sample.iter(f"{MINIML_NS}Characteristics"):
            tag = char.attrib.get("tag")
            if tag:
                row[_snake(tag)] = " ".join((char.text or "").split()) or None
        rows.append(row)
    if not rows:
        raise ValueError(f"No <Sample> elements found in {xml_path}")
    meta = pd.DataFrame(rows).set_index("sample_id")
    if meta.index.has_duplicates:
        raise ValueError("Duplicate sample accessions in MINiML file")
    logger.info("Parsed %d samples with columns %s", len(meta), list(meta.columns))
    return meta


def _read_count_table(handle: IO[bytes], name: str) -> pd.Series:
    raw = handle.read()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    table = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0)
    if table.shape[1] != 1:
        raise ValueError(f"{name}: expected 2 columns (feature, count), got {table.shape[1] + 1}")
    counts = table.iloc[:, 0]
    if counts.isna().any():
        raise ValueError(f"{name}: missing count values")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"{name}: counts must be integers, got {counts.dtype}")
    if (counts < 0).any():
        raise ValueError(f"{name}: negative counts")
    if counts.index.has_duplicates:
        raise ValueError(f"{name}: duplicated feature identifiers")
    return counts


def read_count_archive(tar_path: Path) -> pd.DataFrame:
    """Read every ``GSM*_counts.txt[.gz]`` member of a GEO ``_RAW.tar`` without extracting it.

    Returns a samples x features integer matrix. All samples must share the same features.
    """
    columns: dict[str, pd.Series] = {}
    reference: pd.Index | None = None
    with tarfile.open(tar_path) as archive:
        for member in archive:
            match = COUNT_FILE_RE.match(Path(member.name).name)
            if not member.isfile() or match is None:
                logger.debug("Skipping archive member %s", member.name)
                continue
            gsm = match["gsm"]
            if gsm in columns:
                raise ValueError(f"Sample {gsm} appears twice in {tar_path.name}")
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - isfile() guarantees a handle
                continue
            with handle:
                counts = _read_count_table(handle, member.name)
            if reference is None:
                reference = counts.index
            elif not counts.index.equals(reference):
                raise ValueError(f"{member.name}: feature set differs from other samples")
            columns[gsm] = counts
    if not columns:
        raise ValueError(f"No count files found in {tar_path}")
    matrix = pd.DataFrame(columns).T.astype(np.int64)
    matrix.index.name = "sample_id"
    matrix.columns.name = "feature"
    logger.info("Read count matrix: %d samples x %d features", *matrix.shape)
    return matrix


def is_transposable_element(features: pd.Index) -> np.ndarray:
    return np.asarray(features.astype(str).str.contains(TE_SEPARATOR, regex=False))
