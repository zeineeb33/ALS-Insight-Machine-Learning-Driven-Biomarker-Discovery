"""Static report figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

# Validated categorical slots and chart chrome (light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
UP, DOWN, NEUTRAL = "#e34948", "#2a78d6", "#c3c2b7"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


def library_sizes(metadata: pd.DataFrame, labels: list[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        values = metadata.loc[metadata["condition"] == label, "library_size"] / 1e6
        x = i + rng.uniform(-0.15, 0.15, len(values))
        ax.scatter(x, values, s=16, color=SERIES[i], alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
        ax.hlines(values.median(), i - 0.28, i + 0.28, color=INK, linewidth=2)
    ax.set_xticks(
        range(len(labels)), [f"{lab} (n={(metadata['condition'] == lab).sum()})" for lab in labels]
    )
    ax.set_ylabel("Assigned reads (millions)")
    ax.set_title("Library size per sample (bar = median)")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def pca_scatter(pca: pd.DataFrame, variance: list[float], labels: list[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    panels = (("condition", labels), ("region", sorted(pca["region"].unique())))
    outliers = pca[pca["outlier"]]
    for ax, (column, levels) in zip(axes, panels, strict=True):
        for i, level in enumerate(levels):
            sub = pca[pca[column] == level]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=26,
                color=SERIES[i],
                alpha=0.8,
                edgecolors=SURFACE,
                linewidths=0.8,
                label=f"{level} (n={len(sub)})",
            )
        if len(outliers):
            ax.scatter(
                outliers["PC1"],
                outliers["PC2"],
                s=150,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.2,
                label="flagged outlier",
            )
        ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
        ax.set_title(f"Coloured by {column}")
        ax.legend(loc="best")
    axes[0].set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
    fig.suptitle("PCA of log-normalised expression (top variable features)", color=INK, fontsize=12)
    return _save(fig, path)


def volcano(de: pd.DataFrame, alpha: float, lfc: float, path: Path, n_labels: int = 12) -> Path:
    df = de.dropna(subset=["padj"]).copy()
    df["neglog"] = -np.log10(df["padj"].clip(lower=1e-300))
    sig = df["padj"] < alpha
    up = sig & (df["log2FoldChange"] > 0)
    down = sig & (df["log2FoldChange"] < 0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(
        df.loc[~sig, "log2FoldChange"],
        df.loc[~sig, "neglog"],
        s=5,
        color=NEUTRAL,
        alpha=0.6,
        linewidths=0,
        label=f"not significant ({(~sig).sum():,})",
    )
    ax.scatter(
        df.loc[up, "log2FoldChange"],
        df.loc[up, "neglog"],
        s=8,
        color=UP,
        alpha=0.8,
        linewidths=0,
        label=f"higher in ALS ({up.sum():,})",
    )
    ax.scatter(
        df.loc[down, "log2FoldChange"],
        df.loc[down, "neglog"],
        s=8,
        color=DOWN,
        alpha=0.8,
        linewidths=0,
        label=f"lower in ALS ({down.sum():,})",
    )
    ax.axhline(-np.log10(alpha), color=MUTED, linewidth=0.8)
    for x in (-lfc, lfc):
        ax.axvline(x, color=GRID, linewidth=0.8)
    for name, row in df.nsmallest(n_labels, "pvalue").iterrows():
        ax.annotate(
            str(name),
            (row["log2FoldChange"], row["neglog"]),
            fontsize=7.5,
            color=INK_2,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("log2 fold change, ALS vs control (apeGLM-shrunk)")
    ax.set_ylabel("-log10 adjusted p-value (BH)")
    ax.set_title(f"Differential expression (PyDESeq2), FDR < {alpha}")
    ax.legend(loc="upper left")
    return _save(fig, path)


def roc_curves(predictions: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    grid = np.linspace(0, 1, 101)
    for i, (model, sub) in enumerate(predictions.groupby("model", sort=False)):
        tprs = []
        for _, rep in sub.groupby("repeat"):
            fpr, tpr, _ = roc_curve(rep["y_true"], rep["y_score"])
            tprs.append(np.interp(grid, fpr, tpr))
            ax.plot(fpr, tpr, color=SERIES[i], alpha=0.25, linewidth=1)
        ax.plot(
            grid, np.mean(tprs, axis=0), color=SERIES[i], linewidth=2, label=model.replace("_", " ")
        )
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1)
    ax.set_xlabel("False positive rate (controls called ALS)")
    ax.set_ylabel("True positive rate (ALS detected)")
    ax.set_title("Out-of-fold ROC (pooled per repeat)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    return _save(fig, path)


def cv_metrics(
    folds: pd.DataFrame, null_auc: list[float], observed: float | None, path: Path
) -> Path:
    metrics = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"]
    models = list(dict.fromkeys(folds["model"]))
    ncols = 2 if null_auc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4), squeeze=False)
    ax = axes[0, 0]
    width = 0.36
    rng = np.random.default_rng(0)
    for i, model in enumerate(models):
        sub = folds[folds["model"] == model]
        for j, metric in enumerate(metrics):
            x = j + (i - (len(models) - 1) / 2) * width
            vals = sub[metric].to_numpy()
            ax.scatter(
                x + rng.uniform(-0.06, 0.06, len(vals)),
                vals,
                s=12,
                color=SERIES[i],
                alpha=0.6,
                linewidths=0,
                label=model.replace("_", " ") if j == 0 else None,
            )
            ax.hlines(vals.mean(), x - 0.14, x + 0.14, color=SERIES[i], linewidth=2.2)
    ax.set_xticks(range(len(metrics)), [m.replace("_", "\n") for m in metrics])
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Outer-fold metrics (bar = mean)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    if null_auc:
        ax2 = axes[0, 1]
        ax2.hist(
            null_auc,
            bins=15,
            color=NEUTRAL,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=f"label-permuted ({len(null_auc)} runs)",
        )
        if observed is not None:
            ax2.axvline(observed, color=SERIES[0], linewidth=2, label=f"observed ({observed:.2f})")
        ax2.set_xlabel("Mean outer-fold ROC AUC")
        ax2.set_ylabel("Permutations")
        ax2.set_title("Permutation test (elastic net)")
        ax2.legend(loc="upper left")
    return _save(fig, path)


def stability(importance: pd.DataFrame, path: Path, top: int = 25) -> Path:
    data = importance.sort_values(
        ["selection_frequency", "mean_abs_importance"], ascending=False
    ).head(top)
    fig, ax = plt.subplots(figsize=(6.5, 0.26 * len(data) + 1.2))
    colors = [UP if c > 0 else DOWN for c in data["mean_importance"]]
    ax.barh(range(len(data)), data["selection_frequency"], color=colors, height=0.7)
    ax.set_yticks(range(len(data)), data.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency across outer folds")
    ax.set_title("Elastic-net stability (red: higher in ALS, blue: lower)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def top_gene_heatmap(
    logexpr: pd.DataFrame, condition: pd.Series, genes: list[str], path: Path
) -> Path:
    order = condition.sort_values(kind="stable").index
    z = logexpr.loc[order, genes]
    z = (z - z.mean()) / z.std(ddof=1)
    fig, ax = plt.subplots(figsize=(9, 0.24 * len(genes) + 1.5))
    cmap = LinearSegmentedColormap.from_list("diverging", [DOWN, "#f0efec", UP])
    image = ax.imshow(
        z.T.clip(-3, 3), aspect="auto", cmap=cmap, vmin=-3, vmax=3, interpolation="nearest"
    )
    ax.set_yticks(range(len(genes)), genes, fontsize=8)
    counts = condition.loc[order].value_counts(sort=False)
    edges = np.cumsum(counts.to_numpy())
    for e in edges[:-1]:
        ax.axvline(e - 0.5, color=INK, linewidth=1.2)
    centers = edges - counts.to_numpy() / 2
    ax.set_xticks(centers, [f"{k} (n={v})" for k, v in counts.items()])
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("z-score (log2 normalised)", color=INK_2)
    ax.set_title("Top consensus genes across samples")
    return _save(fig, path)
