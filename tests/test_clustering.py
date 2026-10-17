import numpy as np
import pandas as pd
import pytest

from market_structure import clustering as cl

NON_EUCLIDEAN = np.array([[1, 0.9, 0], [0.9, 1, 0.9], [0, 0.9, 1.0]])  # smallest eigenvalue < 0


def test_correlation_matrix_is_symmetric_with_unit_diagonal(market):
    returns, _, _ = market
    corr = cl.correlation_matrix(returns)
    assert corr.shape == (len(returns.columns),) * 2
    assert np.array_equal(corr, corr.T)
    assert np.all(np.diag(corr) == 1.0)
    assert corr.min() >= -1 and corr.max() <= 1
    assert corr[0, 1] == pytest.approx(returns.iloc[:, 0].corr(returns.iloc[:, 1]))


def test_distance_values_and_properties():
    corr = np.array([[1, 1, 0, -1], [1, 1, 0, -1], [0, 0, 1, 0], [-1, -1, 0, 1.0]])
    dist = cl.correlation_distance(corr)
    assert dist[0, 1] == pytest.approx(0)
    assert dist[0, 2] == pytest.approx(np.sqrt(2))
    assert dist[0, 3] == pytest.approx(2)
    assert np.all(np.diag(dist) == 0)
    assert np.array_equal(dist, dist.T)


def test_distance_is_euclidean_distance_between_standardized_series(market):
    returns, _, _ = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    z = (returns - returns.mean()) / returns.std()  # sample standard deviation
    straight_line = np.linalg.norm(z.iloc[:, 0] - z.iloc[:, 5]) / np.sqrt(len(returns) - 1)
    assert dist[0, 5] == pytest.approx(straight_line)


def test_euclidean_check_and_ward_guard(market):
    returns, _, _ = market
    corr = cl.correlation_matrix(returns)
    assert cl.is_euclidean(corr)
    assert "ward" in cl.usable_linkages(corr)
    assert not cl.is_euclidean(NON_EUCLIDEAN)
    assert cl.usable_linkages(NON_EUCLIDEAN) == ("single", "complete", "average")


def test_correlation_matrix_with_more_stocks_than_days_is_still_euclidean():
    returns = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 60)))
    assert cl.is_euclidean(cl.correlation_matrix(returns))


@pytest.mark.parametrize("method", ["complete", "average", "ward"])
def test_planted_groups_are_recovered_exactly(market, method):
    returns, _, planted = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    labels = cl.cut_tree(cl.build_tree(dist, method), 3)
    assert cl.adjusted_rand_index(labels, planted) == pytest.approx(1.0)


def test_unknown_linkage_and_bad_k(market):
    returns, _, _ = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    with pytest.raises(ValueError, match="Unknown linkage"):
        cl.build_tree(dist, "median")
    with pytest.raises(ValueError, match="at least 1"):
        cl.cut_tree(cl.build_tree(dist, "average"), 0)


def test_cophenetic_correlation_is_high_for_clear_structure(market):
    returns, _, _ = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    assert cl.cophenetic_correlation(cl.build_tree(dist, "average"), dist) > 0.9


def brute_force_silhouette(dist, labels):
    scores = []
    for i in range(len(labels)):
        own = [j for j in range(len(labels)) if labels[j] == labels[i] and j != i]
        if not own:
            scores.append(0.0)
            continue
        a = np.mean([dist[i, j] for j in own])
        b = min(
            np.mean([dist[i, j] for j in range(len(labels)) if labels[j] == other])
            for other in set(labels)
            if other != labels[i]
        )
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def test_silhouette_matches_brute_force_including_singletons():
    rng = np.random.default_rng(3)
    points = rng.standard_normal((25, 3))
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    labels = rng.integers(0, 4, 25)
    labels[0] = 9  # a cluster with one member
    assert cl.silhouette(dist, labels) == pytest.approx(brute_force_silhouette(dist, labels))


def test_silhouette_is_high_for_planted_and_needs_two_clusters(market):
    returns, _, planted = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    assert cl.silhouette(dist, planted) > 0.3
    with pytest.raises(ValueError, match="two clusters"):
        cl.silhouette(dist, np.ones(len(planted), dtype=int))


def test_adjusted_rand_index_known_values():
    assert cl.adjusted_rand_index([0, 0, 1, 1], [5, 5, 7, 7]) == pytest.approx(1.0)
    assert cl.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)
    assert cl.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert cl.adjusted_rand_index([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same length"):
        cl.adjusted_rand_index([0, 1], [0, 1, 2])


def test_normalized_mutual_information_known_values():
    assert cl.normalized_mutual_information([0, 0, 1, 1], [3, 3, 4, 4]) == pytest.approx(1.0)
    assert cl.normalized_mutual_information([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(0.8)
    assert cl.normalized_mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0, abs=1e-12)
    assert cl.normalized_mutual_information([0, 0, 0], [1, 1, 1]) == 1.0


def test_k_sensitivity_and_best_k(market):
    returns, _, planted = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    tree = cl.build_tree(dist, "average")
    table = cl.k_sensitivity(tree, dist, planted, range(2, 8))
    assert list(table["k"]) == list(range(2, 8))
    assert table.loc[table["k"] == 3, "ari_vs_sectors"].iloc[0] == pytest.approx(1.0)
    assert cl.best_k(table, 2, 7) == 3
    assert cl.best_k(table, 4, 7) in range(4, 8)
    with pytest.raises(ValueError, match="No clustering"):
        cl.best_k(table, 20, 30)


def test_best_k_prefers_smaller_k_on_ties():
    table = pd.DataFrame({"k": [2, 3, 4], "silhouette": [0.5, 0.5, 0.1]})
    assert cl.best_k(table, 2, 4) == 2


def test_linkage_comparison_choice_and_agreement(market):
    returns, _, planted = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    trees = {m: cl.build_tree(dist, m) for m in cl.LINKAGES}
    sensitivity = {m: cl.k_sensitivity(t, dist, planted, range(2, 8)) for m, t in trees.items()}
    comparison = cl.linkage_comparison(trees, dist, sensitivity, 2, 7)
    assert set(comparison["linkage"]) == set(cl.LINKAGES)
    linkage, k = cl.choose_primary(comparison)
    assert k == 3 and linkage != "single"
    agreement = cl.linkage_agreement(trees, 3)
    assert agreement.loc["average", "ward"] == pytest.approx(1.0)
    assert np.allclose(np.diag(agreement), 1.0)


def test_bootstrap_is_reproducible_and_stable_for_clear_structure(market):
    returns, _, planted = market
    dist = cl.correlation_distance(cl.correlation_matrix(returns))
    reference = cl.cut_tree(cl.build_tree(dist, "average"), 8)
    first = cl.bootstrap_stability(returns, "average", 8, reference, 15, seed=7)
    second = cl.bootstrap_stability(returns, "average", 8, reference, 15, seed=7)
    other = cl.bootstrap_stability(returns, "average", 8, reference, 15, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.max() <= 1.0
    clear = cl.bootstrap_stability(returns, "average", 3, planted, 15, seed=7)
    assert clear.min() > 0.9
