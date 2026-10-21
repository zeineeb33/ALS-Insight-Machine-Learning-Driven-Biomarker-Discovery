"""FastAPI application: JSON API + a static single-page UI, served by one process.

Run locally with ``uvicorn --factory als_insight.serving.app:create_app``. The results
directory defaults to ``./results`` and can be overridden with ``ALS_INSIGHT_RESULTS_DIR``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from fastapi import Path as PathParam
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from als_insight import __version__
from als_insight.serving.schemas import (
    Biomarkers,
    Consensus,
    DifferentialExpression,
    ElasticNetEvidence,
    Gene,
    GeneMatch,
    Health,
    NotFound,
    PCAData,
    Summary,
    VolcanoData,
    XGBoostEvidence,
)
from als_insight.serving.store import GeneRow, ResultsStore

logger = logging.getLogger("als_insight.serving")
STATIC_DIR = Path(__file__).parent / "static"
RESULTS_ENV = "ALS_INSIGHT_RESULTS_DIR"


def _gene(row: GeneRow, alpha: float) -> Gene:
    v, r = row.values, row.ranks
    padj = v["padj"]
    return Gene(
        symbol=row.symbol,
        feature_type="TE" if row.feature_type == "TE" else "gene",
        differential_expression=DifferentialExpression(
            base_mean=v["base_mean"],
            log2_fold_change=v["log2_fold_change"],
            log2_fold_change_mle=v["log2_fold_change_mle"],
            lfc_se=v["lfc_se"],
            pvalue=v["pvalue"],
            padj=padj,
            significant=padj is not None and padj < alpha,
            rank=r["de_rank"],
        ),
        elastic_net=ElasticNetEvidence(
            selection_frequency=v["en_selection_frequency"] or 0.0,
            mean_coefficient=v["en_mean_coef"] or 0.0,
            rank=r["en_rank"],
        ),
        xgboost=XGBoostEvidence(
            selection_frequency=v["xgb_selection_frequency"] or 0.0,
            mean_gain=v["xgb_mean_gain"] or 0.0,
            rank=r["xgb_rank"],
        ),
        consensus=Consensus(score=v["consensus_score"], rank=r["consensus_rank"]),
    )


def _volcano_bytes(store: ResultsStore) -> bytes:
    rows = [g for g in store.genes if g.values["padj"] is not None]
    payload = VolcanoData(
        alpha=store.alpha,
        lfc_threshold=store.lfc_threshold,
        symbol=[g.symbol for g in rows],
        log2_fold_change=[round(g.values["log2_fold_change"] or 0.0, 4) for g in rows],
        neg_log10_padj=[round(-math.log10(max(g.values["padj"] or 1.0, 1e-300)), 4) for g in rows],
        significant=[(g.values["padj"] or 1.0) < store.alpha for g in rows],
    )
    return payload.model_dump_json().encode()


def create_app(results_dir: Path | None = None) -> FastAPI:
    store = ResultsStore(results_dir or Path(os.environ.get(RESULTS_ENV, "results")))
    volcano = _volcano_bytes(store)
    pca = PCAData.model_validate(store.pca).model_dump_json().encode()
    logger.info("Loaded results %s (%d genes)", store.version, len(store.genes))

    app = FastAPI(
        title="ALS Insight API",
        version=__version__,
        description=(
            "Precomputed ALS vs control RNA-seq biomarker evidence (GEO "
            f"{store.manifest['dataset']['accession']}): differential expression, "
            "nested-CV model importances and a consensus ranking. " + store.manifest["disclaimer"]
        ),
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=Health, tags=["meta"])
    def health() -> Health:
        return Health(results_version=store.version, n_genes=len(store.genes))

    @app.get("/summary", response_model=Summary, tags=["meta"])
    def summary() -> Summary:
        m = store.manifest
        return Summary(
            results_version=store.version,
            **{k: m[k] for k in Summary.model_fields if k != "results_version"},
        )

    @app.get("/genes", response_model=list[GeneMatch], tags=["genes"])
    def search_genes(
        q: Annotated[str, Query(min_length=1, max_length=64, description="Symbol prefix")],
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[GeneMatch]:
        """Autocomplete: case-insensitive prefix (then substring) matches."""
        return [
            GeneMatch(
                symbol=g.symbol,
                feature_type=g.feature_type,
                consensus_rank=g.ranks["consensus_rank"],
            )
            for g in store.search(q, limit)
        ]

    @app.get(
        "/genes/{symbol}",
        response_model=Gene,
        responses={404: {"model": NotFound}},
        tags=["genes"],
    )
    def get_gene(
        symbol: Annotated[str, PathParam(min_length=1, max_length=64, examples=["TBX3"])],
    ) -> Gene | JSONResponse:
        """All evidence for one gene or transposable element (case-insensitive)."""
        row = store.get(symbol)
        if row is None:
            suggestions = store.suggest(symbol, 5)
            body = NotFound(detail=f"'{symbol}' was not tested", suggestions=suggestions)
            return JSONResponse(status_code=404, content=body.model_dump())
        return _gene(row, store.alpha)

    @app.get("/biomarkers", response_model=Biomarkers, tags=["genes"])
    def biomarkers(
        top: Annotated[int, Query(ge=1, le=500, description="Number of candidates")] = 20,
        feature_type: Annotated[Literal["gene", "TE"] | None, Query()] = None,
    ) -> Biomarkers:
        """Top-N consensus biomarker candidates (best first)."""
        rows = store.top(top, feature_type)
        return Biomarkers(
            results_version=store.version,
            total_ranked=len(store.genes),
            items=[_gene(r, store.alpha) for r in rows],
        )

    @app.get(
        "/plots/volcano",
        responses={200: {"model": VolcanoData}},
        response_class=Response,
        tags=["plots"],
    )
    def volcano_data() -> Response:
        """Columnar data for the volcano plot (one entry per tested feature)."""
        return Response(content=volcano, media_type="application/json")

    @app.get(
        "/plots/pca",
        responses={200: {"model": PCAData}},
        response_class=Response,
        tags=["plots"],
    )
    def pca_data() -> Response:
        """PC1-PC3 coordinates of every sample, with condition, region and outlier flag."""
        return Response(content=pca, media_type="application/json")

    return app
