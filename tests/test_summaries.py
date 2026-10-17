import numpy as np
import pandas as pd
import pytest

from market_structure import clustering as cl
from market_structure import summaries as sm


@pytest.fixture
def clustered(market):
    returns, universe, planted = market
    symbols = list(returns.columns)
    corr = cl.correlation_matrix(returns)
    dist = cl.correlation_distance(corr)
    labels = sm.renumber_by_size(cl.cut_tree(cl.build_tree(dist, "average"), 3))
    membership = sm.membership_table(symbols, labels, universe)
    return symbols, corr, membership, planted


def test_renumber_by_size_orders_largest_first_and_keeps_ties_stable():
    labels = np.array([7, 7, 3, 9, 9, 9, 3, 1])
    assert list(sm.renumber_by_size(labels)) == [3, 3, 2, 1, 1, 1, 2, 4]
    assert list(sm.renumber_by_size(np.array([5, 4, 5, 4]))) == [2, 1, 2, 1]


def test_membership_table_has_sector_labels_and_fills_unknown(market):
    returns, universe, _ = market
    symbols = list(returns.columns) + ["NEW"]
    table = sm.membership_table(symbols, np.arange(len(symbols)) % 3 + 1, universe)
    assert list(table.columns) == ["cluster", "symbol", "name", "sector", "sub_industry"]
    assert table.loc[table["symbol"] == "NEW", "sector"].iloc[0] == "Unknown"
    assert len(table) == len(symbols)


def test_cluster_summary(clustered):
    symbols, corr, membership, _ = clustered
    summary = sm.cluster_summary(membership, corr, symbols)
    assert list(summary["size"]) == [15, 15, 15]
    assert (summary["dominant_sector_share"] == 1.0).all()
    assert (summary["mean_within_correlation"] > 0.5).all()
    assert (summary["sectors"] == 1).all()
    assert all(len(row.split(", ")) == 3 for row in summary["most_central"])


def test_cluster_summary_single_stock_cluster(clustered):
    symbols, corr, membership, _ = clustered
    membership = membership.copy()
    membership.loc[membership["symbol"] == symbols[0], "cluster"] = 99
    row = sm.cluster_summary(membership, corr, symbols).set_index("cluster").loc[99]
    assert row["size"] == 1 and np.isnan(row["mean_within_correlation"])
    assert row["most_central"] == symbols[0]


def test_sector_by_cluster_is_diagonal_for_planted_groups(clustered):
    _, _, membership, _ = clustered
    table = sm.sector_by_cluster(membership)
    assert table.to_numpy().sum() == 45
    assert (table.to_numpy() > 0).sum() == 3


def test_sector_correlation_matrix_within_exceeds_between(clustered):
    symbols, corr, membership, _ = clustered
    matrix = sm.sector_correlation_matrix(corr, symbols, membership)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    within = np.diag(matrix)
    between = matrix.to_numpy()[~np.eye(3, dtype=bool)]
    assert within.min() > 0.5 and abs(between).max() < 0.2


def test_pair_summary_counts_and_means(clustered):
    symbols, corr, membership, _ = clustered
    table = sm.pair_summary(corr, symbols, membership).set_index("pairs")
    assert table.loc["all pairs", "count"] == 45 * 44 // 2
    assert table.loc["same sector", "count"] == 3 * (15 * 14 // 2)
    assert table.loc["same sector", "count"] == table.loc["same cluster", "count"]
    assert table.loc["same sector", "mean_correlation"] > table.loc["different sector", "mean_correlation"]
    upper = corr[np.triu_indices(45, k=1)]
    assert table.loc["all pairs", "mean_correlation"] == pytest.approx(upper.mean())


def test_extreme_pairs_sorted_and_flag_share_classes(clustered):
    symbols, corr, membership, _ = clustered
    membership = membership.copy()
    membership.loc[membership["symbol"] == "A00", "name"] = "Alpha Co (Class A)"
    membership.loc[membership["symbol"] == "A01", "name"] = "Alpha Co (Class C)"
    highest = sm.extreme_pairs(corr, symbols, membership, 5, highest=True)
    lowest = sm.extreme_pairs(corr, symbols, membership, 5, highest=False)
    assert highest["correlation"].is_monotonic_decreasing
    assert lowest["correlation"].is_monotonic_increasing
    assert highest["correlation"].iloc[0] >= lowest["correlation"].iloc[0]
    both = sm.extreme_pairs(corr, symbols, membership, 45 * 44 // 2, highest=True)
    flagged = both[both["same_company"]]
    assert set(flagged["symbol_a"]) | set(flagged["symbol_b"]) == {"A00", "A01"}
    assert len(flagged) == 1


def test_stock_connectivity(clustered):
    symbols, corr, membership, _ = clustered
    table = sm.stock_connectivity(corr, symbols, membership)
    assert len(table) == 45
    assert table["mean_correlation_all"].is_monotonic_decreasing
    assert (table["mean_correlation_sector_peers"] > table["mean_correlation_all"]).all()
    first = table.set_index("symbol").loc[symbols[0]]
    assert first["mean_correlation_all"] == pytest.approx((corr[0].sum() - 1) / 44)
    assert isinstance(table, pd.DataFrame)
