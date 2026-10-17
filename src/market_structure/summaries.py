"""Tables that describe the clusters, using the sector labels only to compare afterwards."""

import numpy as np
import pandas as pd


def renumber_by_size(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 1, 2, 3, ... from largest to smallest (ties go to the smaller original label)."""
    unique, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    new_label = np.empty(len(unique), dtype=int)
    new_label[order] = np.arange(1, len(unique) + 1)
    return new_label[codes]


def membership_table(symbols: list[str], labels: np.ndarray, universe: pd.DataFrame) -> pd.DataFrame:
    """One row per clustered stock with its cluster and its sector labels."""
    table = pd.DataFrame({"symbol": symbols, "cluster": labels})
    info = universe.drop_duplicates("symbol").set_index("symbol")
    table = table.join(info[["name", "sector", "sub_industry"]], on="symbol")
    table[["name", "sector", "sub_industry"]] = table[["name", "sector", "sub_industry"]].fillna("Unknown")
    columns = ["cluster", "symbol", "name", "sector", "sub_industry"]
    return table[columns].sort_values(["cluster", "symbol"], ignore_index=True)


def _mean_off_diagonal(block: np.ndarray) -> float:
    n = len(block)
    return float((block.sum() - np.trace(block)) / (n * (n - 1))) if n > 1 else np.nan


def cluster_summary(membership: pd.DataFrame, corr: np.ndarray, symbols: list[str]) -> pd.DataFrame:
    """Size, how tightly each cluster moves together, and which sectors it is made of."""
    position = {symbol: i for i, symbol in enumerate(symbols)}
    rows = []
    for cluster, group in membership.groupby("cluster"):
        idx = [position[s] for s in group["symbol"]]
        block = corr[np.ix_(idx, idx)]
        sector_counts = group["sector"].value_counts()
        n = len(group)
        if n > 1:
            centrality = (block.sum(axis=1) - 1) / (n - 1)
            order = np.argsort(-centrality, kind="stable")[:3]
            central = ", ".join(group["symbol"].iloc[order])
        else:
            central = group["symbol"].iloc[0]
        rows.append(
            {
                "cluster": cluster,
                "size": n,
                "mean_within_correlation": _mean_off_diagonal(block),
                "dominant_sector": sector_counts.index[0],
                "dominant_sector_share": sector_counts.iloc[0] / n,
                "sectors": len(sector_counts),
                "most_central": central,
            }
        )
    return pd.DataFrame(rows)


def sector_by_cluster(membership: pd.DataFrame) -> pd.DataFrame:
    """Number of stocks of each sector (rows) in each cluster (columns)."""
    return pd.crosstab(membership["sector"], membership["cluster"])


def sector_correlation_matrix(corr: np.ndarray, symbols: list[str], membership: pd.DataFrame) -> pd.DataFrame:
    """Average pairwise correlation between every pair of sectors (the diagonal is within-sector)."""
    sectors = membership.set_index("symbol").loc[symbols, "sector"].to_numpy()
    names = sorted(set(sectors))
    masks = {name: sectors == name for name in names}
    values = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            block = corr[np.ix_(masks[a], masks[b])]
            values[i, j] = _mean_off_diagonal(block) if a == b else float(block.mean())
    return pd.DataFrame(values, index=names, columns=names)


def pair_summary(corr: np.ndarray, symbols: list[str], membership: pd.DataFrame) -> pd.DataFrame:
    """Mean correlation of all stock pairs, split by whether they share a sector or a cluster."""
    info = membership.set_index("symbol").loc[symbols]
    upper = np.triu_indices(len(symbols), k=1)
    values = corr[upper]
    sector = info["sector"].to_numpy()
    cluster = info["cluster"].to_numpy()
    same_sector = sector[upper[0]] == sector[upper[1]]
    same_cluster = cluster[upper[0]] == cluster[upper[1]]
    groups = {
        "all pairs": np.ones_like(values, dtype=bool),
        "same sector": same_sector,
        "different sector": ~same_sector,
        "same cluster": same_cluster,
        "different cluster": ~same_cluster,
    }
    return pd.DataFrame(
        [
            {
                "pairs": label,
                "count": int(mask.sum()),
                "mean_correlation": float(values[mask].mean()) if mask.any() else np.nan,
                "median_correlation": float(np.median(values[mask])) if mask.any() else np.nan,
            }
            for label, mask in groups.items()
        ]
    )


def extreme_pairs(
    corr: np.ndarray, symbols: list[str], membership: pd.DataFrame, n: int, highest: bool
) -> pd.DataFrame:
    """The ``n`` most (or least) correlated pairs, flagging share classes of the same company."""
    info = membership.set_index("symbol").loc[symbols]
    company = info["name"].str.replace(r"\s*\(.*\)\s*$", "", regex=True).to_numpy()  # drop "(Class A)"
    first, second = np.triu_indices(len(symbols), k=1)
    values = corr[first, second]
    order = np.argsort(-values if highest else values, kind="stable")[:n]
    rows = []
    for rank, o in enumerate(order, start=1):
        a, b = first[o], second[o]
        rows.append(
            {
                "rank": rank,
                "symbol_a": symbols[a],
                "symbol_b": symbols[b],
                "correlation": values[o],
                "same_company": company[a] == company[b],
                "same_sector": info["sector"].iloc[a] == info["sector"].iloc[b],
                "same_cluster": info["cluster"].iloc[a] == info["cluster"].iloc[b],
                "sector_a": info["sector"].iloc[a],
                "sector_b": info["sector"].iloc[b],
            }
        )
    return pd.DataFrame(rows)


def stock_connectivity(corr: np.ndarray, symbols: list[str], membership: pd.DataFrame) -> pd.DataFrame:
    """Per stock: average correlation with all others, with its sector peers and with its cluster."""
    info = membership.set_index("symbol").loc[symbols]
    others = ~np.eye(len(symbols), dtype=bool)

    def mean_with(mask_matrix: np.ndarray) -> np.ndarray:
        mask = mask_matrix & others
        counts = mask.sum(axis=1)
        return np.divide(
            (corr * mask).sum(axis=1), counts, out=np.full(len(symbols), np.nan), where=counts > 0
        )

    sector = info["sector"].to_numpy()
    cluster = info["cluster"].to_numpy()
    table = info.reset_index()[["cluster", "symbol", "name", "sector"]].copy()
    table["mean_correlation_all"] = mean_with(np.ones_like(others))
    table["mean_correlation_sector_peers"] = mean_with(sector[:, None] == sector[None, :])
    table["mean_correlation_cluster"] = mean_with(cluster[:, None] == cluster[None, :])
    return table.sort_values("mean_correlation_all", ascending=False, ignore_index=True)
