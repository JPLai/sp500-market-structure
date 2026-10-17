"""Write the market-map tables, figures and a generated report with the numbers of a run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from .step1 import Step1Result

FIGURES = {
    "correlation_heatmap.png": "Correlation matrix ordered by the clustering tree",
    "dendrogram.png": "Clustering tree and the chosen cut",
    "k_sensitivity.png": "Sensitivity to the number of clusters",
    "sector_composition.png": "Sector make-up of each cluster",
    "sector_correlation.png": "Average correlation between sectors",
}


def markdown_table(table: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """A plain markdown table; ``formats`` maps column names to format strings such as ``{:.2f}``."""
    formats = formats or {}
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        cells = []
        for name, value in row.items():
            if isinstance(value, float | np.floating) and np.isnan(value):
                cells.append("")
            elif name in formats:
                cells.append(formats[name].format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _save_tables(result: Step1Result, folder: Path) -> None:
    long_sensitivity = pd.concat(
        [t.assign(linkage=m) for m, t in result.sensitivity.items()], ignore_index=True
    )
    tables = {
        "membership": result.membership,
        "cluster_summary": result.clusters,
        "sector_by_cluster": result.sector_crosstab.reset_index(),
        "sector_correlation": result.sector_matrix.rename_axis("sector").reset_index(),
        "pair_summary": result.pairs,
        "strongest_pairs": result.strongest,
        "most_negative_pairs": result.most_negative,
        "stock_connectivity": result.connectivity,
        "linkage_comparison": result.comparison,
        "linkage_agreement": result.agreement.rename_axis("linkage").reset_index(),
        "k_sensitivity": long_sensitivity,
        "excluded_stocks": result.excluded,
    }
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False, float_format="%.4f")
    if len(result.bootstrap_scores):
        scores = pd.DataFrame(
            {
                "resample": range(1, len(result.bootstrap_scores) + 1),
                "adjusted_rand_index": result.bootstrap_scores,
            }
        )
        scores.to_csv(folder / "bootstrap_stability.csv", index=False, float_format="%.4f")


def _save_figures(result: Step1Result, folder: Path) -> None:
    sectors = result.membership.set_index("symbol").loc[result.symbols, "sector"].to_numpy()
    tree = result.trees[result.linkage]
    figures = {
        "correlation_heatmap.png": plotting.heatmap_figure(
            result.corr,
            tree,
            result.labels,
            sectors,
            f"S&P 500 stock correlations, {result.first_return_date} to {result.last_return_date} "
            f"({result.linkage} linkage, {result.k} clusters)",
        ),
        "dendrogram.png": plotting.dendrogram_figure(tree, result.k, result.linkage),
        "k_sensitivity.png": plotting.sensitivity_figure(result.sensitivity, result.linkage, result.k),
        "sector_composition.png": plotting.composition_figure(result.sector_crosstab),
        "sector_correlation.png": plotting.sector_matrix_figure(result.sector_matrix),
    }
    for name, figure in figures.items():
        figure.savefig(folder / name, dpi=150)


def _stability_text(scores: np.ndarray) -> str:
    if not len(scores):
        return "Not run (bootstrap disabled)."
    low, median, high = np.percentile(scores, [5, 50, 95])
    return (
        f"{len(scores)} resamples of the trading days (with replacement), rebuilding the tree each "
        f"time and comparing its cut with the full-sample clusters by adjusted Rand "
        f"index: median {median:.2f}, 5th to 95th percentile {low:.2f} to {high:.2f} "
        "(1 = identical grouping)."
    )


def build_report(result: Step1Result) -> str:
    cfg = result.config
    pair = result.pairs.set_index("pairs")["mean_correlation"]
    primary = result.sensitivity[result.linkage].set_index("k").loc[result.k]
    kept = len(result.symbols)
    sizes = result.clusters["size"]
    trees_tested = ", ".join(result.trees)
    ward_note = (
        "Ward linkage was included because the correlation matrix is positive semi-definite, so the "
        "distance is Euclidean."
        if "ward" in result.trees
        else "Ward linkage was left out because the correlation matrix is not positive semi-definite."
    )
    dropped = (
        f" {result.dropped_dates} date(s) with prices for fewer than half the stocks were ignored."
        if result.dropped_dates
        else ""
    )
    edge_note = []
    if cfg.k is None and result.k in (cfg.k_min, cfg.k_max):
        edge_note = [
            f"> **Note:** the best k sits at the edge of the allowed range ({cfg.k_min}-{cfg.k_max}). The "
            "silhouette has no clear peak, so the data show no natural number of clusters; this k is "
            "a convenient cut, not a discovered one. See the sensitivity plot.",
            "",
        ]
    singular_note = (
        [
            f"- There are more stocks ({kept}) than days ({result.days}), so the correlation matrix is "
            "singular (rank below the number of stocks). It is still valid for clustering, but "
            "individual correlations are noisy."
        ]
        if kept > result.days
        else []
    )
    lines = [
        "# Market map results",
        "",
        "_Generated by `market-structure step1`; do not edit by hand. Interpretation is in "
        "[docs/step1_market_map.md](../../docs/step1_market_map.md)._",
        "",
        "## Data",
        "",
        f"- Window requested: {result.start} to {result.end} (end exclusive); daily returns from "
        f"{result.first_return_date} to {result.last_return_date} ({result.days} days).{dropped}",
        f"- Universe: {result.universe_size} current S&P 500 members; {kept} have a return on every "
        f"day and are clustered, {len(result.excluded)} are excluded "
        "([excluded_stocks.csv](excluded_stocks.csv)).",
        "- Returns are daily log returns of dividend- and split-adjusted closes. Nothing is filled in.",
        "",
        "## Method",
        "",
        "- Correlation: Pearson correlation of daily returns, `C[i,j]`.",
        "- Distance: `d[i,j] = sqrt(2 (1 - C[i,j]))` (0 for correlation 1, 1.41 for 0, 2 for -1).",
        f"- Linkages compared: {trees_tested}. {ward_note}",
        f"- Choice of linkage and number of clusters k: {result.selection}. Sector labels are never "
        "used for this.",
        "",
        "## Linkage comparison",
        "",
        markdown_table(
            result.comparison,
            {
                "cophenetic_correlation": "{:.3f}",
                "silhouette": "{:.3f}",
                "largest_cluster_share": "{:.1%}",
                "ari_vs_sectors": "{:.3f}",
            },
        ),
        "",
        "`best_k` is the k with the highest silhouette in the allowed range; the last three columns "
        "describe that cut. Cophenetic correlation says how faithfully the tree keeps the original "
        "distances.",
        "",
        "## Chosen clustering",
        "",
        f"**{result.linkage} linkage, {result.k} clusters** (silhouette {primary['silhouette']:.3f}, "
        f"largest cluster {primary['largest_cluster_share']:.0%} of the stocks, "
        f"{int(primary['singletons'])} single-stock clusters). Cluster sizes: "
        f"{', '.join(str(s) for s in sizes)}.",
        "",
        *edge_note,
        markdown_table(
            result.clusters,
            {"mean_within_correlation": "{:.3f}", "dominant_sector_share": "{:.0%}"},
        ),
        "",
        "![Correlation matrix](correlation_heatmap.png)",
        "",
        "![Clustering tree](dendrogram.png)",
        "",
        "![Sector make-up of each cluster](sector_composition.png)",
        "",
        "## Sensitivity",
        "",
        "![Sensitivity to k](k_sensitivity.png)",
        "",
        f"- Agreement between linkages at k = {result.k} (adjusted Rand index):",
        "",
        markdown_table(
            result.agreement.rename_axis("linkage").reset_index(),
            {m: "{:.2f}" for m in result.agreement.columns},
        ),
        "",
        f"- Stability to the sample of days: {_stability_text(result.bootstrap_scores)}",
        "",
        "## Sectors and correlation",
        "",
        markdown_table(
            result.pairs,
            {"mean_correlation": "{:.3f}", "median_correlation": "{:.3f}"},
        ),
        "",
        f"Average correlation within a sector is {pair['same sector']:.3f} against "
        f"{pair['different sector']:.3f} across sectors.",
        "",
        "![Sector correlation](sector_correlation.png)",
        "",
        f"## Strongest pairs (top {min(10, len(result.strongest))})",
        "",
        markdown_table(
            result.strongest.head(10)[
                [
                    "rank",
                    "symbol_a",
                    "symbol_b",
                    "correlation",
                    "same_company",
                    "same_sector",
                    "same_cluster",
                ]
            ],
            {"correlation": "{:.3f}"},
        ),
        "",
        "`same_company` marks two share classes of one company. All pairs are in "
        "[strongest_pairs.csv](strongest_pairs.csv) and [most_negative_pairs.csv]"
        "(most_negative_pairs.csv).",
        "",
        "## Caveats",
        "",
        "- The universe is today's S&P 500 applied to the past year, so stocks that left the index "
        "are missing (survivorship bias).",
        f"- {len(result.excluded)} stocks without a complete history are not clustered.",
        f"- The window has {result.days} daily observations, so correlations are noisy, and the result "
        "describes this window only.",
        *singular_note,
        "- Correlation is not causation. Clusters describe stocks that moved together, nothing more.",
        f"- The bootstrap treats days as independent (seed {cfg.seed}); returns are not exactly so.",
        "- This is research output, not investment advice.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: Step1Result, output_dir: Path | None = None) -> Path:
    """Write tables, figures, the generated report and a small run summary; return the folder."""
    folder = Path(output_dir or result.config.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    _save_tables(result, folder)
    _save_figures(result, folder)
    (folder / "step1_report.md").write_text(build_report(result), encoding="utf-8")
    summary = {
        "window_requested": [result.start, result.end],
        "first_return_date": result.first_return_date,
        "last_return_date": result.last_return_date,
        "days": result.days,
        "universe_size": result.universe_size,
        "stocks_clustered": len(result.symbols),
        "stocks_excluded": len(result.excluded),
        "euclidean": result.euclidean,
        "linkage": result.linkage,
        "k": result.k,
        "selection": result.selection,
        "seed": result.config.seed,
        "bootstrap_resamples": int(len(result.bootstrap_scores)),
    }
    (folder / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return folder
