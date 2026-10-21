"""In-memory index of the results bundle, loaded once at startup with the standard library.

Deliberately free of numpy/pandas so the runtime image stays small (Render free tier).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILE = "manifest.json"
GENES_FILE = "genes.csv.gz"
PCA_FILE = "pca.json"
SUPPORTED_SCHEMA = 1

_FLOAT_FIELDS = (
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
    "consensus_score",
)
_INT_FIELDS = ("de_rank", "en_rank", "xgb_rank", "consensus_rank")


class ResultsError(RuntimeError):
    """The results bundle is missing or incompatible."""


def _float(value: str) -> float | None:
    if value in {"", "nan", "NaN"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class GeneRow:
    symbol: str
    feature_type: str
    values: dict[str, float | None]
    ranks: dict[str, int]


class ResultsStore:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        try:
            self.manifest: dict[str, Any] = json.loads((results_dir / MANIFEST_FILE).read_text())
            pca_raw = (results_dir / PCA_FILE).read_text()
            genes_raw = gzip.decompress((results_dir / GENES_FILE).read_bytes()).decode()
        except FileNotFoundError as exc:
            raise ResultsError(f"Incomplete results bundle in {results_dir}: {exc}") from exc
        schema = self.manifest.get("schema_version")
        if schema != SUPPORTED_SCHEMA:
            raise ResultsError(f"Unsupported results schema {schema!r}")
        self.pca: dict[str, Any] = json.loads(pca_raw)
        self.genes: list[GeneRow] = [
            GeneRow(
                symbol=row["symbol"],
                feature_type=row["feature_type"],
                values={f: _float(row[f]) for f in _FLOAT_FIELDS},
                ranks={f: int(row[f]) for f in _INT_FIELDS},
            )
            for row in csv.DictReader(io.StringIO(genes_raw))
        ]
        self.genes.sort(key=lambda g: g.ranks["consensus_rank"])
        self._by_symbol = {g.symbol.upper(): g for g in self.genes}
        self._sorted_symbols = sorted(self._by_symbol)

    @property
    def version(self) -> str:
        return str(self.manifest["results_version"])

    @property
    def alpha(self) -> float:
        return float(self.manifest["differential_expression"]["alpha"])

    @property
    def lfc_threshold(self) -> float:
        return float(self.manifest["differential_expression"]["lfc_threshold"])

    def get(self, symbol: str) -> GeneRow | None:
        return self._by_symbol.get(symbol.strip().upper())

    def search(self, query: str, limit: int) -> list[GeneRow]:
        """Case-insensitive prefix matches first, then substring matches."""
        q = query.strip().upper()
        if not q:
            return []
        prefix = [s for s in self._sorted_symbols if s.startswith(q)]
        contains = [s for s in self._sorted_symbols if q in s and not s.startswith(q)]
        return [self._by_symbol[s] for s in (prefix + contains)[:limit]]

    def suggest(self, symbol: str, limit: int) -> list[str]:
        """Symbols sharing the longest possible prefix (>= 2 characters) with ``symbol``."""
        q = symbol.strip()
        for n in range(len(q), 1, -1):
            matches = self.search(q[:n], limit)
            if matches:
                return [g.symbol for g in matches]
        return []

    def top(self, n: int, feature_type: str | None = None) -> list[GeneRow]:
        rows = (g for g in self.genes if feature_type is None or g.feature_type == feature_type)
        return [row for _, row in zip(range(n), rows, strict=False)]
