"""Command-line interface: ``als-insight download|preprocess|analyze|select|report|run``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from als_insight.config import Config, load_config
from als_insight.logging_utils import setup_logging

app = typer.Typer(
    help="Reproducible ALS RNA-seq biomarker discovery pipeline.",
    no_args_is_help=True,
    add_completion=False,
)
logger = logging.getLogger("als_insight")

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Pipeline YAML config."),
]
ForceOpt = Annotated[bool, typer.Option("--force", help="Recompute even if cached.")]

_state: dict[str, str] = {"log_level": "INFO"}


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    _state["log_level"] = log_level.upper()
    setup_logging(_state["log_level"])


def _load(path: Path) -> Config:
    cfg = load_config(path)
    logger.info("Loaded config %s (dataset %s)", path, cfg.dataset.accession)
    return cfg


@app.command()
def download(config: ConfigOpt = Path("configs/default.yaml")) -> None:
    """Download the GEO series (MINiML annotations + raw count archive)."""
    from als_insight.download import download as _download

    _download(_load(config))


@app.command()
def preprocess(config: ConfigOpt = Path("configs/default.yaml"), force: ForceOpt = False) -> None:
    """Build the samples x features count matrix and run sample/gene QC."""
    from als_insight import preprocess as step

    step.run(_load(config), force=force)


@app.command()
def analyze(config: ConfigOpt = Path("configs/default.yaml"), force: ForceOpt = False) -> None:
    """PCA + outlier detection, then PyDESeq2 differential expression."""
    from als_insight import analyze as step

    step.run(_load(config), force=force)


@app.command()
def select(config: ConfigOpt = Path("configs/default.yaml"), force: ForceOpt = False) -> None:
    """Nested, subject-grouped CV of elastic net + XGBoost and consensus gene ranking."""
    from als_insight import selection as step

    step.run(_load(config), force=force)


@app.command()
def report(config: ConfigOpt = Path("configs/default.yaml"), force: ForceOpt = False) -> None:
    """Render figures + Markdown/HTML report and export the versioned results bundle."""
    from als_insight import report as step

    step.run(_load(config), force=force)


@app.command()
def run(config: ConfigOpt = Path("configs/default.yaml"), force: ForceOpt = False) -> None:
    """Run every step (download -> preprocess -> analyze -> select -> report)."""
    from als_insight import analyze as analyze_step
    from als_insight import preprocess as preprocess_step
    from als_insight import report as report_step
    from als_insight import selection as selection_step
    from als_insight.download import download as download_step

    cfg = _load(config)
    download_step(cfg)
    preprocess_step.run(cfg, force=force)
    analyze_step.run(cfg, force=force)
    selection_step.run(cfg, force=force)
    report_step.run(cfg, force=force)


if __name__ == "__main__":  # pragma: no cover
    app()
