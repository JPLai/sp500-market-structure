"""Figures for the market map. Each function returns a matplotlib figure; the caller saves it."""

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

HEATMAP_FLOOR = -0.4  # correlations below this share the bluest colour; almost none are this low


def sector_colors(sectors: list[str]) -> dict[str, tuple[float, ...]]:
    palette = colormaps["tab20"]
    return {name: palette(i % 20) for i, name in enumerate(sorted(set(sectors)))}


def leaf_order(tree: np.ndarray) -> np.ndarray:
    return np.array(dendrogram(tree, no_plot=True)["leaves"])


def cut_height(tree: np.ndarray, k: int) -> float:
    """A height that cuts the tree into ``k`` clusters (halfway between two merges)."""
    n = len(tree) + 1
    if k <= 1:
        return float(tree[-1, 2]) * 1.02
    if k >= n:
        return float(tree[0, 2]) / 2
    return float((tree[n - k - 1, 2] + tree[n - k, 2]) / 2)


def heatmap_figure(
    corr: np.ndarray, tree: np.ndarray, clusters: np.ndarray, sectors: np.ndarray, title: str
) -> Figure:
    """Correlation matrix with stocks ordered by the tree; black lines mark cluster borders and the
    coloured strips show each stock's GICS sector (not used to build the clusters)."""
    order = leaf_order(tree)
    ordered = corr[np.ix_(order, order)]
    ordered_clusters = clusters[order]
    ordered_sectors = sectors[order]
    colors = sector_colors(list(sectors))
    strip = np.array([colors[s][:3] for s in ordered_sectors])[:, None, :]

    fig = Figure(figsize=(10.5, 9.5), layout="constrained")
    grid = fig.add_gridspec(
        2, 3, width_ratios=[0.025, 1, 0.03], height_ratios=[0.025, 1], wspace=0.01, hspace=0.01
    )
    ax = fig.add_subplot(grid[1, 1])
    top = fig.add_subplot(grid[0, 1], sharex=ax)
    left = fig.add_subplot(grid[1, 0], sharey=ax)
    cax = fig.add_subplot(grid[1, 2])

    norm = TwoSlopeNorm(vmin=HEATMAP_FLOOR, vcenter=0.0, vmax=1.0)
    image = ax.imshow(ordered, cmap="RdBu_r", norm=norm, interpolation="nearest", aspect="auto")
    top.imshow(strip.transpose(1, 0, 2), aspect="auto", interpolation="nearest")
    left.imshow(strip, aspect="auto", interpolation="nearest")
    boundaries = np.flatnonzero(np.diff(ordered_clusters)) + 0.5
    for edge in boundaries:
        ax.axhline(edge, color="black", linewidth=0.5)
        ax.axvline(edge, color="black", linewidth=0.5)
    for axis in (ax, top, left):
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, cax=cax, label=f"correlation (colour floor {HEATMAP_FLOOR})")
    ax.set_xlabel(f"{len(corr)} stocks, ordered by the clustering tree")
    top.set_title(title, fontsize=11)
    handles = [Patch(color=c, label=name) for name, c in colors.items()]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=4,
        fontsize=7.5,
        title="GICS sector (strip)",
        title_fontsize=8,
    )
    return fig


def dendrogram_figure(tree: np.ndarray, k: int, method: str) -> Figure:
    fig = Figure(figsize=(11, 5), layout="constrained")
    ax = fig.add_subplot()
    height = cut_height(tree, k)
    dendrogram(tree, no_labels=True, color_threshold=height, above_threshold_color="0.5", ax=ax)
    ax.axhline(height, color="black", linestyle="--", linewidth=0.8, label=f"cut into {k} clusters")
    ax.set_ylabel(r"merge distance  $\sqrt{2(1-\rho)}$")
    ax.set_xlabel("stocks")
    ax.set_title(f"Clustering tree ({method} linkage)")
    ax.legend(loc="upper right")
    return fig


def sensitivity_figure(sensitivity: dict[str, pd.DataFrame], chosen: str, k: int) -> Figure:
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    left, right = fig.subplots(1, 2)
    for method, table in sensitivity.items():
        width = 2.2 if method == chosen else 1.0
        left.plot(table["k"], table["silhouette"], label=method, linewidth=width)
        right.plot(table["k"], table["ari_vs_sectors"], label=method, linewidth=width)
    for axis in (left, right):
        axis.axvline(k, color="black", linestyle=":", linewidth=0.8)
        axis.set_xlabel("number of clusters (k)")
    left.set_ylabel("mean silhouette (higher = better separated)")
    right.set_ylabel("adjusted Rand index vs GICS sectors")
    left.set_title("Cluster separation")
    right.set_title("Agreement with sectors (not used to choose k)")
    left.legend(title="linkage")
    return fig


def composition_figure(crosstab: pd.DataFrame) -> Figure:
    """Sectors (rows) by clusters (columns); cell text is the number of stocks."""
    fig = Figure(figsize=(max(7.0, 0.5 * crosstab.shape[1] + 3), 5), layout="constrained")
    ax = fig.add_subplot()
    shares = crosstab.div(crosstab.sum(axis=0), axis=1)
    image = ax.imshow(shares.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(crosstab.shape[1]), [str(c) for c in crosstab.columns])
    ax.set_yticks(range(crosstab.shape[0]), list(crosstab.index))
    sizes = crosstab.sum(axis=0).to_numpy()
    ax.set_xlabel("cluster (largest first; size shown in brackets)")
    ax.set_xticklabels([f"{c}\n({s})" for c, s in zip(crosstab.columns, sizes, strict=True)], fontsize=7)
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            count = crosstab.iat[i, j]
            if count:
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if shares.iat[i, j] > 0.6 else "black",
                )
    fig.colorbar(image, ax=ax, label="share of the cluster")
    ax.set_title("Sector make-up of each cluster (number of stocks)")
    return fig


def sector_matrix_figure(matrix: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(8.5, 7), layout="constrained")
    ax = fig.add_subplot()
    values = matrix.to_numpy()
    image = ax.imshow(values, cmap="viridis")
    ax.set_xticks(range(len(matrix)), list(matrix.columns), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), list(matrix.index), fontsize=8)
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if values[i, j] > midpoint else "white",
            )
    fig.colorbar(image, ax=ax, label="mean correlation")
    ax.set_title("Average correlation between sectors (diagonal: within the sector)")
    return fig
