from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from als_insight import download
from als_insight.cache import sha256_file
from als_insight.config import Config
from synthetic import ACCESSION, SyntheticGEO


@pytest.fixture
def remote(tmp_path: Path, synthetic_geo: SyntheticGEO) -> dict[str, Path]:
    """Files a fake GEO server would serve, keyed by URL basename."""
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    tgz = remote_dir / f"{ACCESSION}_family.xml.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(
            synthetic_geo.raw_dir / f"{ACCESSION}_family.xml", arcname=f"{ACCESSION}_family.xml"
        )
    shutil.move(synthetic_geo.raw_dir / f"{ACCESSION}_RAW.tar", remote_dir / f"{ACCESSION}_RAW.tar")
    (synthetic_geo.raw_dir / f"{ACCESSION}_family.xml").unlink()
    return {p.name: p for p in remote_dir.iterdir()}


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch, remote: dict[str, Path]) -> list[str]:
    calls: list[str] = []

    def _fetch(url: str, dest: Path, timeout: float) -> None:
        calls.append(url)
        shutil.copy(remote[url.rsplit("/", 1)[-1]], dest)

    monkeypatch.setattr(download, "fetch", _fetch)
    return calls


def test_download_fetches_verifies_and_caches(
    synthetic_config: Config, remote: dict[str, Path], fake_fetch: list[str]
) -> None:
    tar_sha = sha256_file(remote[f"{ACCESSION}_RAW.tar"])
    cfg = synthetic_config.model_copy(
        update={"dataset": synthetic_config.dataset.model_copy(update={"counts_sha256": tar_sha})}
    )
    files = download.download(cfg)
    assert files.miniml_xml.read_text().startswith("<?xml")
    assert files.counts_tar.exists()
    checksums = json.loads((cfg.raw_dir / download.CHECKSUMS_FILE).read_text())
    assert checksums[f"{ACCESSION}_RAW.tar"] == tar_sha
    assert len(fake_fetch) == 2

    download.download(cfg)  # second call is served from the local cache
    assert len(fake_fetch) == 2
    assert download.raw_files(cfg) == files


def test_download_rejects_checksum_mismatch(
    synthetic_config: Config, fake_fetch: list[str]
) -> None:
    cfg = synthetic_config.model_copy(
        update={"dataset": synthetic_config.dataset.model_copy(update={"counts_sha256": "0" * 64})}
    )
    with pytest.raises(download.ChecksumError):
        download.download(cfg)
    assert not (cfg.raw_dir / f"{ACCESSION}_RAW.tar").exists()  # corrupt file is removed


def test_download_rejects_archive_without_family_file(
    synthetic_config: Config, remote: dict[str, Path], fake_fetch: list[str]
) -> None:
    tgz = remote[f"{ACCESSION}_family.xml.tgz"]
    with tarfile.open(tgz, "w:gz") as tar:
        info = tarfile.TarInfo("../evil.xml")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="does not contain"):
        download.download(synthetic_config)


def test_fetch_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts: list[int] = []

    def _urlopen(url: str, timeout: float) -> io.BytesIO:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("connection reset")
        return io.BytesIO(b"payload")

    monkeypatch.setattr(download.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)
    dest = tmp_path / "file.bin"
    download.fetch("https://example.org/file.bin", dest, timeout=1)
    assert dest.read_bytes() == b"payload"
    assert len(attempts) == 2
    assert not dest.with_suffix(".bin.part").exists()
