"""The market map: correlation, distance and hierarchical clustering of one window of daily returns."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import clustering as cl
from . import summaries as sm
from .data import load_prices, load_universe
from .returns import log_returns, price_table, select_complete

MIN_PRICE_COVERAGE = 0.5  # a date needs prices for at least this share of the stocks to count


@dataclass
class Step1Config:
    start: str = ""
    end: str = ""
    data_dir: Path = Path("data")
    output_dir: Path = Path("output/step1")
    linkage: str | None = None  # None: choose by silhouette
    k: int | None = None  # None: choose by silhouette
    k_min: int = 5
    k_max: int = 30
    k_scan_max: int = 40
    bootstrap: int = 100
    seed: int = 0
    top_pairs: int = 20
    refresh_data: bool = False

    def resolved_window(self) -> tuple[str, str]:
        """Default window: the last 365 days up to today (the end date is exclusive)."""
        try:
            end = date.fromisoformat(self.end) if self.end else date.today()
            start = date.fromisoformat(self.start) if self.start else end - timedelta(days=365)
        except ValueError:
            raise ValueError("Dates must be written as YYYY-MM-DD") from None
        if start >= end:
            raise ValueError("The start date must be before the end date")
        return start.isoformat(), end.isoformat()

    def validate(self) -> None:
        if self.k_min < 2 or self.k_max < self.k_min:
            raise ValueError("Need 2 <= k-min <= k-max")
        if self.k_scan_max < self.k_max:
            raise ValueError("The scanned range of k must include k-min to k-max")
        if self.k is not None and self.k < 2:
            raise ValueError("k must be at least 2")
        if self.linkage is not None and self.linkage not in cl.LINKAGES:
            raise ValueError(f"Unknown linkage {self.linkage!r}; choose from {', '.join(cl.LINKAGES)}")
        if self.bootstrap < 0 or self.top_pairs < 1:
            raise ValueError("bootstrap must be >= 0 and top-pairs >= 1")


@dataclass
class Step1Result:
    config: Step1Config
    start: str
    end: str
    first_return_date: str
    last_return_date: str
    days: int
    universe_size: int
    symbols: list[str]
    excluded: pd.DataFrame
    dropped_dates: int
    euclidean: bool
    corr: np.ndarray
    dist: np.ndarray
    trees: dict[str, np.ndarray]
    sensitivity: dict[str, pd.DataFrame]
    comparison: pd.DataFrame
    agreement: pd.DataFrame
    linkage: str
    k: int
    selection: str
    labels: np.ndarray
    membership: pd.DataFrame
    clusters: pd.DataFrame
    sector_crosstab: pd.DataFrame
    sector_matrix: pd.DataFrame
    pairs: pd.DataFrame
    strongest: pd.DataFrame
    most_negative: pd.DataFrame
    connectivity: pd.DataFrame
    bootstrap_scores: np.ndarray = field(default_factory=lambda: np.array([]))


def prepare_returns(
    prices: pd.DataFrame, universe_symbols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Complete-history returns, the exclusion report, and how many partial dates were dropped."""
    closes = price_table(prices)
    coverage = closes.notna().mean(axis=1)
    thin = coverage < MIN_PRICE_COVERAGE
    closes = closes[~thin]
    returns, excluded = select_complete(log_returns(closes), universe_symbols)
    return returns, excluded, int(thin.sum())


def analyse(returns: pd.DataFrame, universe: pd.DataFrame, config: Step1Config) -> dict[str, object]:
    """All of the clustering work on a complete returns table (no I/O)."""
    symbols = list(returns.columns)
    if len(symbols) < config.k_scan_max + 1:
        raise ValueError(
            f"Only {len(symbols)} stocks have a complete history; need more than "
            f"{config.k_scan_max} to scan {config.k_scan_max} clusters"
        )
    if len(returns) < 30:
        raise ValueError(f"Only {len(returns)} daily returns; need at least 30")

    corr = cl.correlation_matrix(returns)
    dist = cl.correlation_distance(corr)
    allowed = cl.usable_linkages(corr)
    if config.linkage is not None and config.linkage not in allowed:
        raise ValueError(
            f"{config.linkage!r} linkage needs a Euclidean distance, "
            "but this correlation matrix is not positive semi-definite"
        )

    info = universe.drop_duplicates("symbol").set_index("symbol")
    sector_labels = info["sector"].reindex(symbols).fillna("Unknown").to_numpy()
    trees = {method: cl.build_tree(dist, method) for method in allowed}
    ks = range(2, config.k_scan_max + 1)
    sensitivity = {m: cl.k_sensitivity(t, dist, sector_labels, ks) for m, t in trees.items()}
    comparison = cl.linkage_comparison(trees, dist, sensitivity, config.k_min, config.k_max)

    if config.linkage is None:
        linkage, k = cl.choose_primary(comparison)
        selection = f"highest silhouette over {', '.join(allowed)} and k = {config.k_min}-{config.k_max}"
    else:
        linkage = config.linkage
        k = cl.best_k(sensitivity[linkage], config.k_min, config.k_max)
        selection = f"linkage set by the user; k by highest silhouette in {config.k_min}-{config.k_max}"
    if config.k is not None:
        k = config.k
        selection = (
            "linkage and k set by the user"
            if config.linkage
            else "linkage by highest silhouette; k set by the user"
        )
    if k > config.k_scan_max:
        raise ValueError(f"k = {k} is above the scanned range (up to {config.k_scan_max})")

    labels = sm.renumber_by_size(cl.cut_tree(trees[linkage], k))
    membership = sm.membership_table(symbols, labels, universe)
    scores = (
        cl.bootstrap_stability(returns, linkage, k, labels, config.bootstrap, config.seed)
        if config.bootstrap
        else np.array([])
    )
    return {
        "symbols": symbols,
        "corr": corr,
        "dist": dist,
        "euclidean": cl.is_euclidean(corr),
        "trees": trees,
        "sensitivity": sensitivity,
        "comparison": comparison,
        "agreement": cl.linkage_agreement(trees, k),
        "linkage": linkage,
        "k": k,
        "selection": selection,
        "labels": labels,
        "membership": membership,
        "clusters": sm.cluster_summary(membership, corr, symbols),
        "sector_crosstab": sm.sector_by_cluster(membership),
        "sector_matrix": sm.sector_correlation_matrix(corr, symbols, membership),
        "pairs": sm.pair_summary(corr, symbols, membership),
        "strongest": sm.extreme_pairs(corr, symbols, membership, config.top_pairs, highest=True),
        "most_negative": sm.extreme_pairs(corr, symbols, membership, config.top_pairs, highest=False),
        "connectivity": sm.stock_connectivity(corr, symbols, membership),
        "bootstrap_scores": scores,
    }


def run_step1(
    config: Step1Config,
    universe: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> Step1Result:
    """Load data (cached), run the analysis and return everything; nothing is written here.

    ``universe`` and ``prices`` can be passed in to skip loading (used by the tests).
    """
    config.validate()
    start, end = config.resolved_window()
    raw = Path(config.data_dir) / "raw"
    if universe is None:
        universe = load_universe(raw / "sp500_universe.csv", refresh=config.refresh_data)
    if prices is None:
        prices = load_prices(
            raw / f"sp500_prices_{start}_{end}.csv",
            universe["symbol"],
            start,
            end,
            refresh=config.refresh_data,
        )
    returns, excluded, dropped_dates = prepare_returns(prices, list(universe["symbol"]))
    if returns.empty:
        raise ValueError("No stock has a complete return history in this window")
    parts = analyse(returns, universe, config)
    return Step1Result(
        config=config,
        start=start,
        end=end,
        first_return_date=returns.index[0].date().isoformat(),
        last_return_date=returns.index[-1].date().isoformat(),
        days=len(returns),
        universe_size=len(universe),
        excluded=excluded,
        dropped_dates=dropped_dates,
        **parts,
    )
