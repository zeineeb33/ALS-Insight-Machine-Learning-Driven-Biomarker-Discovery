"""End-to-end: CLI ``run`` on a synthetic GEO series, then serve the bundle via the API."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from als_insight.cli import app as cli
from als_insight.report import GENES_FILE, MANIFEST_FILE, PCA_FILE
from als_insight.serving.app import create_app
from synthetic import ACCESSION, config_dict, make_synthetic_geo


@pytest.mark.slow
def test_full_pipeline_and_api(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / ACCESSION
    geo = make_synthetic_geo(raw_dir, n_genes=150, n_te=10)
    # Pre-seed the "downloaded" MINiML archive so the download step is a cache hit.
    with tarfile.open(raw_dir / f"{ACCESSION}_family.xml.tgz", "w:gz") as tar:
        tar.add(raw_dir / f"{ACCESSION}_family.xml", arcname=f"{ACCESSION}_family.xml")
    cfg = config_dict(tmp_path, n_samples=len(geo.samples))
    cfg["paths"]["report_dir"] = str(tmp_path / "reports")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    result = CliRunner().invoke(
        cli, ["--log-level", "WARNING", "run", "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output

    results_dir = tmp_path / "results"
    for name in (MANIFEST_FILE, GENES_FILE, PCA_FILE):
        assert (results_dir / name).exists()
    manifest = json.loads((results_dir / MANIFEST_FILE).read_text())
    assert manifest["dataset"]["accession"] == ACCESSION
    assert manifest["classification"]["cv"]["grouping"] == "subject_id"
    assert manifest["classification"]["permutation_test"]["n_permutations"] == 2
    assert set(manifest["top_biomarkers"][:5]) & set(geo.signal_features)
    report = (tmp_path / "reports" / "report.md").read_text()
    assert "Consensus biomarker candidates" in report
    assert (tmp_path / "reports" / "figures" / "volcano.png").stat().st_size > 1000

    # Second invocation is fully cached.
    again = CliRunner().invoke(cli, ["run", "--config", str(config_path)])
    assert again.exit_code == 0
    assert "up to date" in again.output

    client = TestClient(create_app(results_dir))
    assert client.get("/health").json()["n_genes"] > 100
    signal = client.get(f"/genes/{geo.signal_features[0]}").json()
    assert signal["differential_expression"]["log2_fold_change"] > 0.5
    assert signal["differential_expression"]["significant"]
    top = client.get("/biomarkers", params={"top": 10}).json()["items"]
    assert len({g["symbol"] for g in top} & set(geo.signal_features)) >= 5
