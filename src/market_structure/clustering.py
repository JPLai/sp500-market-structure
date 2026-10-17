"""Correlation distance, hierarchical clustering and the measures used to judge the clusters.

Everything here works on a plain returns table (days x stocks) with no missing values and never
sees sector labels, except in the clearly named comparison functions that take a second labelling.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform

LINKAGES = ("single", "complete", "average", "ward")


def correlation_matrix(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Pearson correlation of the columns (stocks x stocks), exactly symmetric with a unit diagonal."""
    values = np.asarray(returns, dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.clip((corr + corr.T) / 2, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Distance ``d = sqrt(2 (1 - rho))``: 0 for rho = 1, 1.41 for rho = 0, 2 for rho = -1.

    For standardized return series this is the ordinary straight-line distance between them (up to
    a constant), so it is a true metric and Ward linkage is valid.
    """
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def is_euclidean(corr: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Whether the correlation matrix is positive semi-definite, i.e. the distance is Euclidean."""
    return bool(np.linalg.eigvalsh(corr).min() >= -tolerance)


def build_tree(dist: np.ndarray, method: str) -> np.ndarray:
    """Agglomerative clustering tree (scipy linkage matrix) from a distance matrix."""
    if method not in LINKAGES:
        raise ValueError(f"Unknown linkage {method!r}; choose from {', '.join(LINKAGES)}")
    return linkage(squareform(dist, checks=False), method=method)


def usable_linkages(corr: np.ndarray) -> tuple[str, ...]:
    """All linkages, except Ward when the distance is not Euclidean (it assumes it is)."""
    return LINKAGES if is_euclidean(corr) else tuple(m for m in LINKAGES if m != "ward")


def cut_tree(tree: np.ndarray, k: int) -> np.ndarray:
    """Cluster labels (1..k) from cutting the tree into at most ``k`` clusters."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return fcluster(tree, t=k, criterion="maxclust")


def cophenetic_correlation(tree: np.ndarray, dist: np.ndarray) -> float:
    """How faithfully the tree keeps the original distances (1 is perfect)."""
    value, _ = cophenet(tree, squareform(dist, checks=False))
    return float(value)


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette: how much closer stocks are to their own cluster than to the next one.

    Ranges from -1 to 1; stocks alone in their cluster score 0. Needs at least two clusters.
    """
    _, codes = np.unique(labels, return_inverse=True)
    n_clusters = codes.max() + 1
    if n_clusters < 2:
        raise ValueError("The silhouette needs at least two clusters")
    membership = np.eye(n_clusters)[codes]  # stocks x clusters
    counts = membership.sum(axis=0)
    totals = dist @ membership  # summed distance from each stock to each cluster
    own = totals[np.arange(len(codes)), codes]
    within = np.divide(own, counts[codes] - 1, out=np.zeros_like(own), where=counts[codes] > 1)
    others = totals / counts
    others[np.arange(len(codes)), codes] = np.inf
    nearest = others.min(axis=1)
    scale = np.maximum(within, nearest)
    score = np.divide(nearest - within, scale, out=np.zeros_like(scale), where=scale > 0)
    score[counts[codes] == 1] = 0.0
    return float(score.mean())


def _contingency(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) != len(second):
        raise ValueError("The two labelings must have the same length")
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    return table


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Agreement of two clusterings: 1 identical, about 0 for unrelated, can be below 0."""
    table = _contingency(first, second)

    def pairs(values: np.ndarray) -> float:
        return float((values * (values - 1) / 2).sum())

    n = table.sum()
    together = pairs(table)
    rows, columns = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * columns / (n * (n - 1) / 2)
    best = (rows + columns) / 2
    if best == expected:
        return 1.0
    return (together - expected) / (best - expected)


def normalized_mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    """Shared information of two clusterings, from 0 (unrelated) to 1 (identical)."""
    table = _contingency(first, second)
    joint = table / table.sum()
    p_first, p_second = joint.sum(axis=1), joint.sum(axis=0)
    nonzero = joint > 0
    outer = np.outer(p_first, p_second)
    mutual = float((joint[nonzero] * np.log(joint[nonzero] / outer[nonzero])).sum())

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    scale = (entropy(p_first) + entropy(p_second)) / 2
    return 1.0 if scale == 0 else mutual / scale


def k_sensitivity(
    tree: np.ndarray,
    dist: np.ndarray,
    sectors: np.ndarray,
    ks: range | list[int],
) -> pd.DataFrame:
    """How the clustering changes with the number of clusters.

    Returns, for each k, the silhouette (uses returns only) and the agreement with the sector
    labels (reported afterwards, never used to choose anything).
    """
    rows = []
    for k in ks:
        labels = cut_tree(tree, k)
        sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
        rows.append(
            {
                "k": k,
                "clusters": len(sizes),
                "silhouette": silhouette(dist, labels) if len(sizes) > 1 else np.nan,
                "ari_vs_sectors": adjusted_rand_index(labels, sectors),
                "nmi_vs_sectors": normalized_mutual_information(labels, sectors),
                "largest_cluster_share": sizes.max() / sizes.sum(),
                "singletons": int((sizes == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_k(table: pd.DataFrame, k_min: int, k_max: int) -> int:
    """The k in the range with the highest silhouette (the smaller k wins a tie)."""
    window = table[(table["k"] >= k_min) & (table["k"] <= k_max)].dropna(subset=["silhouette"])
    if window.empty:
        raise ValueError(f"No clustering with {k_min} to {k_max} clusters is available")
    return int(window.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])


def linkage_comparison(
    trees: dict[str, np.ndarray],
    dist: np.ndarray,
    sensitivity: dict[str, pd.DataFrame],
    k_min: int,
    k_max: int,
) -> pd.DataFrame:
    """One row per linkage: tree fidelity and its best cut by silhouette."""
    rows = []
    for method, tree in trees.items():
        table = sensitivity[method]
        k = best_k(table, k_min, k_max)
        row = table[table["k"] == k].iloc[0]
        rows.append(
            {
                "linkage": method,
                "cophenetic_correlation": cophenetic_correlation(tree, dist),
                "best_k": k,
                "silhouette": row["silhouette"],
                "largest_cluster_share": row["largest_cluster_share"],
                "singletons": int(row["singletons"]),
                "ari_vs_sectors": row["ari_vs_sectors"],
            }
        )
    return pd.DataFrame(rows)


def choose_primary(comparison: pd.DataFrame) -> tuple[str, int]:
    """The linkage and k with the highest silhouette (a rule that never looks at sectors).

    A tie goes to the tree that keeps the distances best (higher cophenetic correlation).
    """
    ranked = comparison.sort_values(["silhouette", "cophenetic_correlation"], ascending=False, kind="stable")
    best = ranked.iloc[0]
    return str(best["linkage"]), int(best["best_k"])


def linkage_agreement(trees: dict[str, np.ndarray], k: int) -> pd.DataFrame:
    """Adjusted Rand index between the linkages' cuts into ``k`` clusters."""
    labels = {method: cut_tree(tree, k) for method, tree in trees.items()}
    return pd.DataFrame(
        [[adjusted_rand_index(labels[a], labels[b]) for b in labels] for a in labels],
        index=list(labels),
        columns=list(labels),
    )


def bootstrap_stability(
    returns: pd.DataFrame,
    method: str,
    k: int,
    reference: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Agreement (adjusted Rand index) between the full-sample clusters and clusters rebuilt from
    days resampled with replacement. Close to 1 means the grouping does not depend on which days
    happened to be in the sample. Days are treated as independent, which is only approximately true.
    """
    values = returns.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        dist = correlation_distance(correlation_matrix(sample))
        labels = cut_tree(build_tree(dist, method), k)
        scores[b] = adjusted_rand_index(reference, labels)
    return scores
