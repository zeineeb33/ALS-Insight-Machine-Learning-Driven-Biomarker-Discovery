"""Cached, checksum-verified download of a GEO series (MINiML family file + raw counts)."""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from als_insight.cache import sha256_file
from als_insight.config import Config

logger = logging.getLogger(__name__)

CHECKSUMS_FILE = "checksums.json"
_RETRIES = 3


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its expected SHA-256."""


@dataclass(frozen=True)
class RawFiles:
    miniml_xml: Path
    counts_tar: Path


def fetch(url: str, dest: Path, timeout: float) -> None:
    """Stream ``url`` to ``dest`` atomically (via a ``.part`` file), with retries."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, _RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(dest)
            return
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            if attempt == _RETRIES:
                raise
            wait = 2**attempt
            logger.warning("Download of %s failed (%s); retrying in %ss", url, exc, wait)
            time.sleep(wait)


def ensure_file(url: str, dest: Path, sha256: str | None, timeout: float) -> str:
    """Download ``url`` unless ``dest`` already exists with the expected checksum."""
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        logger.info("Using cached %s", dest)
    else:
        logger.info("Downloading %s -> %s", url, dest)
        fetch(url, dest, timeout)
    digest = sha256_file(dest)
    if sha256 is not None and digest != sha256:
        dest.unlink()
        raise ChecksumError(f"{dest.name}: expected sha256 {sha256}, got {digest}")
    return digest


def _extract_family_xml(tgz: Path, accession: str, dest_dir: Path) -> Path:
    """Extract only ``<accession>_family.xml`` from the MINiML archive (no path traversal)."""
    member_name = f"{accession}_family.xml"
    with tarfile.open(tgz, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError(f"{tgz.name} does not contain {member_name}") from exc
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"{member_name} in {tgz.name} is not a regular file")
        out = dest_dir / member_name
        with source, out.open("wb") as fh:
            shutil.copyfileobj(source, fh)
    return out


def download(cfg: Config) -> RawFiles:
    """Fetch the configured GEO series into ``data/raw/<accession>`` and record checksums."""
    ds = cfg.dataset
    raw_dir = cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    miniml_tgz = raw_dir / Path(ds.miniml_url.path or "").name
    counts_tar = raw_dir / Path(ds.counts_url.path or "").name
    checksums = {
        miniml_tgz.name: ensure_file(
            str(ds.miniml_url), miniml_tgz, ds.miniml_sha256, ds.timeout_seconds
        ),
        counts_tar.name: ensure_file(
            str(ds.counts_url), counts_tar, ds.counts_sha256, ds.timeout_seconds
        ),
    }
    if not tarfile.is_tarfile(counts_tar):
        raise ValueError(f"{counts_tar} is not a valid tar archive")
    xml_path = _extract_family_xml(miniml_tgz, ds.accession, raw_dir)

    (raw_dir / CHECKSUMS_FILE).write_text(json.dumps(checksums, indent=2, sort_keys=True))
    for name, digest in checksums.items():
        logger.info("sha256 %s  %s", digest, name)
    return RawFiles(miniml_xml=xml_path, counts_tar=counts_tar)


def raw_files(cfg: Config) -> RawFiles:
    """Locate previously downloaded files, failing with a helpful message if absent."""
    xml_path = cfg.raw_dir / f"{cfg.dataset.accession}_family.xml"
    counts_tar = cfg.raw_dir / Path(cfg.dataset.counts_url.path or "").name
    missing = [p for p in (xml_path, counts_tar) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files {[str(p) for p in missing]}; run `als-insight download` first."
        )
    return RawFiles(miniml_xml=xml_path, counts_tar=counts_tar)
