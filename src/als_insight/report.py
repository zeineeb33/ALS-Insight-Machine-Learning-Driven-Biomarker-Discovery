"""Pipeline step ``report``: figures, Markdown/HTML report and the versioned results bundle.

The results bundle (``results/``) is the contract between the offline pipeline and the
web service: small, deterministic files that the API loads with the standard library.
"""

from __future__ import annotations

import gzip
import html
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from als_insight import __version__, analyze, cache, figures, preprocess, selection
from als_insight.config import Config
from als_insight.normalization import normalized_log_counts

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
GENES_FILE = "genes.csv.gz"
PCA_FILE = "pca.json"
DISCLAIMER = (
    "Research and educational analysis of public data. Not a diagnostic tool and not "
    "medical advice; candidate genes are statistical associations that require "
    "independent validation."
)
GENE_COLUMNS = {
    "feature_type": "feature_type",
    "baseMean": "base_mean",
    "log2FoldChange": "log2_fold_change",
    "log2FoldChange_mle": "log2_fold_change_mle",
    "lfcSE": "lfc_se",
    "pvalue": "pvalue",
    "padj": "padj",
    "en_selection_frequency": "en_selection_frequency",
    "en_mean_coef": "en_mean_coef",
    "xgb_selection_frequency": "xgb_selection_frequency",
    "xgb_mean_gain": "xgb_mean_gain",
    "de_rank": "de_rank",
    "en_rank": "en_rank",
    "xgb_rank": "xgb_rank",
    "consensus_score": "consensus_score",
    "consensus_rank": "consensus_rank",
}


@dataclass(frozen=True)
class Inputs:
    pre_summary: dict[str, Any]
    ana_summary: dict[str, Any]
    sel_summary: dict[str, Any]
    permutation: dict[str, Any]
    pca: pd.DataFrame
    pca_variance: list[float]
    de: pd.DataFrame
    consensus: pd.DataFrame
    folds: pd.DataFrame
    predictions: pd.DataFrame
    en_importance: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def load_inputs(cfg: Config) -> Inputs:
    pre, ana, sel = (cfg.step_dir(s) for s in ("preprocess", "analyze", "select"))
    return Inputs(
        pre_summary=_read_json(pre / preprocess.SUMMARY_FILE),
        ana_summary=_read_json(ana / analyze.SUMMARY_FILE),
        sel_summary=_read_json(sel / selection.SUMMARY_FILE),
        permutation=_read_json(sel / selection.PERMUTATION_FILE),
        pca=pd.read_csv(ana / analyze.PCA_FILE, index_col="sample_id"),
        pca_variance=_read_json(ana / analyze.PCA_VARIANCE_FILE)["explained_variance_ratio"],
        de=pd.read_csv(ana / analyze.DE_FILE, index_col="feature"),
        consensus=pd.read_csv(sel / selection.CONSENSUS_FILE, index_col="feature"),
        folds=pd.read_csv(sel / selection.FOLDS_FILE),
        predictions=pd.read_csv(sel / selection.PREDICTIONS_FILE),
        en_importance=pd.read_csv(
            sel / selection.IMPORTANCE_FILE.format(model="elastic_net"), index_col="feature"
        ),
    )


# --------------------------------------------------------------------------- bundle


def _gzip_csv(df: pd.DataFrame, path: Path) -> None:
    # mtime=0 keeps the archive byte-identical across runs with identical content.
    path.write_bytes(gzip.compress(df.to_csv(float_format="%.6g").encode(), mtime=0))


def build_manifest(cfg: Config, inp: Inputs) -> dict[str, Any]:
    fp = cfg.section_fingerprint(
        "dataset", "contrast", "preprocess", "exploration", "differential", "selection", "consensus"
    )
    models = {
        name: {
            "metrics": m["metrics"],
            "jaccard_stability": m["jaccard_stability"],
            "mean_selected_features": m["mean_selected_features"],
        }
        for name, m in inp.sel_summary["models"].items()
    }
    ana = inp.ana_summary
    return {
        "schema_version": SCHEMA_VERSION,
        "results_version": f"{cfg.dataset.accession}-v{__version__}-{fp[:8]}",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "config_fingerprint": fp,
        "dataset": {
            "accession": cfg.dataset.accession,
            "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={cfg.dataset.accession}",
            "citation": cfg.dataset.citation,
            "counts_sha256": cfg.dataset.counts_sha256,
        },
        "contrast": cfg.contrast.model_dump(),
        "samples": {
            "n_samples_in_contrast": inp.pre_summary["n_samples"],
            "n_subjects_in_contrast": inp.pre_summary["n_subjects"],
            "n_samples_analyzed": ana["n_samples_analyzed"],
            "samples_per_condition": ana["samples_per_condition"],
            "subjects_per_condition": inp.pre_summary["subjects_per_condition"],
            "outliers_removed": [o["sample_id"] for o in ana["outliers"]]
            if ana["outliers_removed"]
            else [],
            "n_features": inp.pre_summary["n_features"],
            "n_features_tested": ana["n_tested"],
        },
        "differential_expression": {
            k: ana[k]
            for k in (
                "design",
                "alpha",
                "lfc_threshold",
                "n_tested",
                "n_significant",
                "n_up",
                "n_down",
                "n_significant_lfc",
                "n_significant_lfc_mle",
            )
        },
        "classification": {
            "cv": inp.sel_summary["cv"],
            "positive_class": cfg.contrast.case_label,
            "models": models,
            "permutation_test": {
                k: inp.permutation.get(k)
                for k in ("model", "n_permutations", "observed_roc_auc", "null_mean", "p_value")
            },
        },
        "top_biomarkers": inp.consensus.index[:10].tolist(),
        "disclaimer": DISCLAIMER,
    }


def write_bundle(cfg: Config, inp: Inputs) -> Path:
    out = cfg.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)
    genes = inp.consensus[list(GENE_COLUMNS)].rename(columns=GENE_COLUMNS)
    genes.index.name = "symbol"
    _gzip_csv(genes, out / GENES_FILE)

    pcs = [c for c in ("PC1", "PC2", "PC3") if c in inp.pca.columns]
    samples = [
        {
            "sample_id": sid,
            **{pc.lower(): round(float(row[pc]), 4) for pc in pcs},
            "condition": row["condition"],
            "region": row["region"],
            "outlier": bool(row["outlier"]),
        }
        for sid, row in inp.pca.iterrows()
    ]
    pca_payload = {"explained_variance_ratio": inp.pca_variance[: len(pcs)], "samples": samples}
    (out / PCA_FILE).write_text(json.dumps(pca_payload, separators=(",", ":")))
    (out / MANIFEST_FILE).write_text(json.dumps(build_manifest(cfg, inp), indent=2))
    logger.info("Wrote results bundle to %s", out)
    return out


# --------------------------------------------------------------------------- report


def _metric(m: dict[str, dict[str, float]], key: str) -> str:
    return f"{m[key]['mean']:.3f} ± {m[key]['std']:.3f}"


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def render_figures(cfg: Config, inp: Inputs, fig_dir: Path) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [cfg.contrast.case_label, cfg.contrast.control_label]
    ds = preprocess.load_dataset(cfg.step_dir("preprocess"))
    paths = {
        "library": figures.library_sizes(ds.metadata, labels, fig_dir / "qc_library_size.png"),
        "pca": figures.pca_scatter(inp.pca, inp.pca_variance, labels, fig_dir / "pca.png"),
        "volcano": figures.volcano(
            inp.de, cfg.differential.alpha, cfg.differential.lfc_threshold, fig_dir / "volcano.png"
        ),
        "roc": figures.roc_curves(inp.predictions, fig_dir / "cv_roc.png"),
        "metrics": figures.cv_metrics(
            inp.folds,
            inp.permutation.get("null_roc_auc", []),
            inp.permutation.get("observed_roc_auc"),
            fig_dir / "cv_metrics.png",
        ),
        "stability": figures.stability(inp.en_importance, fig_dir / "stability.png"),
    }
    keep = inp.pca.index[~inp.pca["outlier"]] if cfg.exploration.remove_outliers else inp.pca.index
    features = ds.features.index[ds.features["passes_filter"]]
    counts = ds.counts.loc[keep, features]
    logexpr = pd.DataFrame(normalized_log_counts(counts.to_numpy()), index=keep, columns=features)
    top = inp.consensus.index[:30].tolist()
    paths["heatmap"] = figures.top_gene_heatmap(
        logexpr, ds.metadata.loc[keep, "condition"], top, fig_dir / "top_genes_heatmap.png"
    )
    return {k: f"figures/{v.name}" for k, v in paths.items()}


def _top_table(cfg: Config, inp: Inputs) -> tuple[list[str], list[list[str]]]:
    header = [
        "Rank",
        "Gene",
        "log2FC",
        "padj",
        "EN freq.",
        "EN coef",
        "XGB freq.",
        "Consensus",
    ]
    rows = []
    for name, r in inp.consensus.head(cfg.report.top_n_table).iterrows():
        rows.append(
            [
                str(int(r["consensus_rank"])),
                str(name),
                _fmt(r["log2FoldChange"], 2),
                _fmt(r["padj"]),
                _fmt(r["en_selection_frequency"], 2),
                _fmt(r["en_mean_coef"], 3),
                _fmt(r["xgb_selection_frequency"], 2),
                _fmt(r["consensus_score"], 4),
            ]
        )
    return header, rows


def build_sections(cfg: Config, inp: Inputs, figs: dict[str, str]) -> list[tuple[str, Any]]:
    pre, ana, sel, perm = inp.pre_summary, inp.ana_summary, inp.sel_summary, inp.permutation
    case, ctrl = cfg.contrast.case_label, cfg.contrast.control_label
    en, xgb = sel["models"]["elastic_net"]["metrics"], sel["models"]["xgboost"]["metrics"]
    cv = sel["cv"]
    outliers = ", ".join(o["sample_id"] for o in ana["outliers"]) or "none"
    blocks: list[tuple[str, Any]] = [
        ("h1", f"ALS Insight report: {cfg.dataset.accession}"),
        ("p", f"_{DISCLAIMER}_"),
        ("h2", "Dataset and quality control"),
        (
            "p",
            f"Contrast **{cfg.contrast.case}** ({case}) vs **{cfg.contrast.control}** ({ctrl}). "
            f"{pre['n_samples']} samples from {pre['n_subjects']} donors "
            f"({pre['samples_per_condition']}; donors {pre['subjects_per_condition']}), "
            f"{pre['n_features']:,} features ({pre['n_features_te']:,} transposable elements), "
            f"{pre['n_features_passing_filter']:,} passing the low-count filter "
            f"(>= {cfg.preprocess.min_count} reads in >= "
            f"{cfg.preprocess.min_sample_fraction:.0%} of samples). Median library size "
            f"{pre['library_size_median'] / 1e6:.1f} M assigned reads.",
        ),
        ("img", (figs["library"], "Library sizes")),
        ("h2", "Exploration"),
        (
            "p",
            f"PCA on the {cfg.exploration.n_top_variable} most variable log-normalised features. "
            f"Outliers are samples whose Mahalanobis distance in the first "
            f"{cfg.exploration.outlier_components} PCs has a robust z-score > "
            f"{cfg.exploration.outlier_robust_z}: **{outliers}**"
            f"{' (excluded downstream)' if ana['outliers_removed'] else ''}.",
        ),
        ("img", (figs["pca"], "PCA")),
        ("h2", "Differential expression (PyDESeq2)"),
        (
            "p",
            f"Design `{ana['design']}`, Wald test, Benjamini-Hochberg FDR. On "
            f"{ana['n_samples_analyzed']} samples, **{ana['n_significant']:,} of "
            f"{ana['n_tested']:,}** features have FDR < {ana['alpha']} ({ana['n_up']:,} higher, "
            f"{ana['n_down']:,} lower in {case}); {ana['n_significant_lfc']} also have "
            f"|shrunk log2FC| >= {ana['lfc_threshold']} ({ana['n_significant_lfc_mle']} using "
            f"the unshrunk MLE).",
        ),
        ("img", (figs["volcano"], "Volcano plot")),
        ("h2", "Multivariate selection with nested, donor-grouped cross-validation"),
        (
            "p",
            f"{cv['outer_repeats']}x repeated {cv['outer_folds']}-fold outer CV with "
            f"{cv['inner_folds']}-fold inner CV for hyper-parameters, both stratified and "
            f"grouped by donor. Filtering, normalisation, ANOVA pre-selection of "
            f"{cv['k_features']} features and scaling are re-fitted inside every training "
            f"fold. Positive class: {case}.",
        ),
        (
            "table",
            (
                [
                    "Model",
                    "ROC AUC",
                    "Balanced acc.",
                    "Sensitivity",
                    "Specificity",
                    "MCC",
                    "Jaccard",
                ],
                [
                    [
                        name.replace("_", " "),
                        _metric(m, "roc_auc"),
                        _metric(m, "balanced_accuracy"),
                        _metric(m, "sensitivity"),
                        _metric(m, "specificity"),
                        _metric(m, "mcc"),
                        f"{sel['models'][key]['jaccard_stability']:.3f}",
                    ]
                    for key, name, m in (
                        ("elastic_net", "elastic_net", en),
                        ("xgboost", "xgboost", xgb),
                    )
                ],
            ),
        ),
    ]
    if perm.get("n_permutations"):
        blocks.append(
            (
                "p",
                f"Permutation test ({perm['model'].replace('_', ' ')}, "
                f"{perm['n_permutations']} donor-level label permutations, full nested CV each): "
                f"observed AUC {perm['observed_roc_auc']:.3f} vs null mean "
                f"{perm['null_mean']:.3f}, **p = {perm['p_value']:.3f}**.",
            )
        )
    blocks += [
        ("img", (figs["roc"], "ROC curves")),
        ("img", (figs["metrics"], "Fold metrics")),
        ("img", (figs["stability"], "Selection stability")),
        ("h2", "Consensus biomarker candidates"),
        (
            "p",
            "Weighted mean of normalised ranks from DE p-values, elastic-net mean |coefficient| "
            f"and XGBoost mean gain across outer folds (weights {dict(cfg.consensus.weights)}).",
        ),
        ("table", _top_table(cfg, inp)),
        ("img", (figs["heatmap"], "Top genes heatmap")),
        ("h2", "Reproducibility"),
        (
            "p",
            f"Generated by `als-insight run --config configs/default.yaml` (package "
            f"{__version__}). Raw counts archive sha256 `{cfg.dataset.counts_sha256}`.",
        ),
    ]
    if cfg.dataset.citation:
        blocks.append(("p", f"Data: {cfg.dataset.citation}"))
    return blocks


def to_markdown(blocks: list[tuple[str, Any]]) -> str:
    out: list[str] = []
    for kind, content in blocks:
        if kind == "h1":
            out.append(f"# {content}")
        elif kind == "h2":
            out.append(f"## {content}")
        elif kind == "p":
            out.append(str(content))
        elif kind == "img":
            out.append(f"![{content[1]}]({content[0]})")
        elif kind == "table":
            header, rows = content
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "---|" * len(header))
            out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n\n".join(out).replace("|\n\n|", "|\n|") + "\n"


def _inline_md(text: str) -> str:
    """Escape, then render the tiny subset of Markdown used in the report."""
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return re.sub(r"(?<![\w])_(.+?)_(?![\w])", r"<em>\1</em>", text)


def to_html(blocks: list[tuple[str, Any]], title: str) -> str:
    body: list[str] = []
    for kind, content in blocks:
        if kind in {"h1", "h2"}:
            body.append(f"<{kind}>{html.escape(str(content))}</{kind}>")
        elif kind == "p":
            body.append(f"<p>{_inline_md(str(content))}</p>")
        elif kind == "img":
            body.append(
                f'<figure><img src="{content[0]}" alt="{html.escape(content[1])}"></figure>'
            )
        elif kind == "table":
            header, rows = content
            head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
            trs = "".join(
                "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows
            )
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>")
    style = (
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:960px;"
        "margin:2rem auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7;line-height:1.5}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:.9rem;"
        "font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #e1e0d9;"
        "padding:4px 8px;text-align:left}code{background:#f0efec;padding:0 3px}"
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{style}</style></head>"
        f"<body>{''.join(body)}</body></html>\n"
    )


def run(cfg: Config, force: bool = False) -> Path:
    sel_dir = cfg.step_dir("select")
    report_dir = cfg.paths.report_dir
    outputs = ("report.md", "report.html")
    fp = cache.fingerprint(
        cfg.section_fingerprint("report", "differential", "consensus"),
        [sel_dir / cache.STEP_FILE, cfg.step_dir("analyze") / cache.STEP_FILE],
    )
    bundle_ok = all(
        (cfg.paths.results_dir / f).exists() for f in (MANIFEST_FILE, GENES_FILE, PCA_FILE)
    )
    if not force and bundle_ok and cache.is_fresh(report_dir, fp, outputs):
        logger.info("report: up to date (%s)", report_dir)
        return report_dir
    inp = load_inputs(cfg)
    write_bundle(cfg, inp)
    figs = render_figures(cfg, inp, report_dir / "figures")
    blocks = build_sections(cfg, inp, figs)
    (report_dir / "report.md").write_text(to_markdown(blocks))
    (report_dir / "report.html").write_text(
        to_html(blocks, f"ALS Insight report {cfg.dataset.accession}")
    )
    cache.mark_done(report_dir, fp, outputs)
    logger.info("Wrote report to %s", report_dir / "report.md")
    return report_dir
