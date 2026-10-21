"""Logging setup shared by the CLI and the pipeline."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
    # Third-party libraries are noisy at INFO level.
    for noisy in ("matplotlib", "PIL", "urllib3", "numba", "anndata", "formulaic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
