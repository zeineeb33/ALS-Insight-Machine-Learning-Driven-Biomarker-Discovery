from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from als_insight.serving.app import create_app
from als_insight.serving.store import ResultsError

HEADER = [
    "symbol",
    "feature_type",
    "base_mean",
    "log2_fold_change",
    "log2_fold_change_mle",
    "lfc_se",
    "pvalue",
    "padj",
    "en_selection_frequency",
    "en_mean_coef",
    "xgb_selection_frequency",
    "xgb_mean_gain",
    "de_rank",
    "en_rank",
    "xgb_rank",
    "consensus_score",
    "consensus_rank",
]
ROWS = [
    [
        "TBX3",
        "gene",
        61.8,
        -1.32,
        -1.42,
        0.25,
        2.5e-8,
        7.4e-5,
        0.8,
        -0.12,
        0.6,
        1.5,
        1,
        1,
        2,
        0.01,
        1,
    ],
    [
        "ANKRD20A9P",
        "gene",
        12.6,
        -1.93,
        -2.0,
        0.27,
        1.8e-13,
        3.6e-9,
        0.4,
        -0.05,
        0.9,
        2.1,
        2,
        3,
        1,
        0.02,
        2,
    ],
    ["L1HS:L1:LINE", "TE", 900.0, 0.4, 0.45, 0.1, 1e-3, 0.04, 0.0, 0.0, 0.2, 0.3, 3, 4, 3, 0.3, 3],
    ["GENEX", "gene", 3.0, 0.01, 0.02, 0.5, 0.9, "", 0.0, 0.0, 0.0, 0.0, 4, 4, 4, 0.8, 4],
]


def write_bundle(directory: Path, schema_version: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(HEADER)
    writer.writerows(ROWS)
    (directory / "genes.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode()))
    pca = {
        "explained_variance_ratio": [0.2, 0.15, 0.09],
        "samples": [
            {
                "sample_id": "GSM1",
                "pc1": 1.0,
                "pc2": 2.0,
                "pc3": 0.5,
                "condition": "ALS",
                "region": "motor",
                "outlier": False,
            },
            {
                "sample_id": "GSM2",
                "pc1": -1.0,
                "pc2": 0.0,
                "pc3": 0.1,
                "condition": "Control",
                "region": "frontal",
                "outlier": True,
            },
        ],
    }
    (directory / "pca.json").write_text(json.dumps(pca))
    manifest = {
        "schema_version": schema_version,
        "results_version": "GSE124439-v0.1.0-test",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "dataset": {"accession": "GSE124439", "url": "https://example.org", "citation": None},
        "contrast": {"case": "ALS Spectrum MND", "control": "Non-Neurological Control"},
        "samples": {"n_samples_analyzed": 2, "samples_per_condition": {"ALS": 1, "Control": 1}},
        "differential_expression": {
            "alpha": 0.05,
            "lfc_threshold": 1.0,
            "n_significant": 3,
            "n_tested": 4,
        },
        "classification": {
            "models": {
                "elastic_net": {"metrics": {"roc_auc": {"mean": 0.7, "std": 0.1}}},
                "xgboost": {"metrics": {"roc_auc": {"mean": 0.69, "std": 0.2}}},
            },
            "permutation_test": {"n_permutations": 0},
        },
        "top_biomarkers": ["TBX3", "ANKRD20A9P"],
        "disclaimer": "Not a diagnostic tool.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(write_bundle(tmp_path / "results")))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "results_version": "GSE124439-v0.1.0-test",
        "n_genes": 4,
    }


def test_gene_lookup_is_case_insensitive(client: TestClient) -> None:
    body = client.get("/genes/tbx3").json()
    assert body["symbol"] == "TBX3"
    assert body["differential_expression"]["significant"] is True
    assert body["differential_expression"]["log2_fold_change"] == pytest.approx(-1.32)
    assert body["elastic_net"] == {"selection_frequency": 0.8, "mean_coefficient": -0.12, "rank": 1}
    assert body["consensus"]["rank"] == 1


def test_transposable_element_and_missing_padj(client: TestClient) -> None:
    assert client.get("/genes/L1HS:L1:LINE").json()["feature_type"] == "TE"
    genex = client.get("/genes/GENEX").json()["differential_expression"]
    assert genex["padj"] is None
    assert genex["significant"] is False


def test_unknown_gene_returns_404_with_suggestions(client: TestClient) -> None:
    response = client.get("/genes/TBX99")
    assert response.status_code == 404
    assert response.json()["suggestions"] == ["TBX3"]


def test_search(client: TestClient) -> None:
    symbols = [m["symbol"] for m in client.get("/genes", params={"q": "an"}).json()]
    assert symbols == ["ANKRD20A9P"]
    assert client.get("/genes", params={"q": ""}).status_code == 422


def test_biomarkers(client: TestClient) -> None:
    body = client.get("/biomarkers", params={"top": 2}).json()
    assert [g["symbol"] for g in body["items"]] == ["TBX3", "ANKRD20A9P"]
    assert body["total_ranked"] == 4
    te = client.get("/biomarkers", params={"top": 5, "feature_type": "TE"}).json()
    assert [g["symbol"] for g in te["items"]] == ["L1HS:L1:LINE"]
    assert client.get("/biomarkers", params={"top": 0}).status_code == 422


def test_plots(client: TestClient) -> None:
    volcano = client.get("/plots/volcano").json()
    assert len(volcano["symbol"]) == 3  # GENEX has no adjusted p-value
    assert volcano["significant"] == [True, True, True]
    pca = client.get("/plots/pca").json()
    assert len(pca["samples"]) == 2
    assert pca["samples"][1]["outlier"] is True


def test_summary_and_ui(client: TestClient) -> None:
    summary = client.get("/summary").json()
    assert summary["dataset"]["accession"] == "GSE124439"
    assert summary["disclaimer"] == "Not a diagnostic tool."
    page = client.get("/")
    assert page.status_code == 200
    assert "ALS Insight" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "ALS Insight API"


def test_gzip_for_large_payloads(client: TestClient) -> None:
    response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert response.headers.get("content-encoding") == "gzip"


def test_incompatible_bundle_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="schema"):
        create_app(write_bundle(tmp_path / "r", schema_version=99))
    with pytest.raises(ResultsError, match="Incomplete"):
        create_app(tmp_path / "missing")


def test_results_dir_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALS_INSIGHT_RESULTS_DIR", str(write_bundle(tmp_path / "env")))
    assert TestClient(create_app()).get("/health").status_code == 200
