import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import returns_to_prices

from market_structure import cli, plotting
from market_structure.report import build_report, markdown_table, write_outputs
from market_structure.step1 import Step1Config, run_step1


def run(market, config, **overrides):
    returns, universe, _ = market
    for name, value in overrides.items():
        setattr(config, name, value)
    return run_step1(config, universe=universe, prices=returns_to_prices(returns))


def test_default_window_is_one_year_ending_today():
    start, end = Step1Config().resolved_window()
    assert end == date.today().isoformat()
    assert date.fromisoformat(end) - date.fromisoformat(start) == timedelta(days=365)


@pytest.mark.parametrize(
    "overrides",
    [
        {"k_min": 1},
        {"k_min": 6, "k_max": 5},
        {"k_scan_max": 4},
        {"k": 1},
        {"linkage": "median"},
        {"bootstrap": -1},
        {"top_pairs": 0},
        {"start": "2026-01-01", "end": "2025-01-01"},
        {"start": "banana"},
    ],
)
def test_invalid_settings_are_rejected(small_config, market, overrides):
    with pytest.raises(ValueError):
        run(market, small_config, **overrides)


def test_end_to_end_recovers_planted_groups(market, small_config):
    result = run(market, small_config)
    assert (result.k, result.euclidean) == (3, True)
    assert result.linkage in {"complete", "average", "ward"}
    assert len(result.symbols) == 45 and result.days == 250
    assert cl_ari(result.labels, market[2]) == pytest.approx(1.0)
    assert list(result.clusters["size"]) == [15, 15, 15]
    assert result.excluded.empty
    assert result.bootstrap_scores.min() > 0.9


def cl_ari(a, b):
    from market_structure.clustering import adjusted_rand_index

    return adjusted_rand_index(a, b)


def test_choices_can_be_forced(market, small_config):
    result = run(market, small_config, linkage="complete", k=5)
    assert (result.linkage, result.k) == ("complete", 5)
    assert result.clusters.shape[0] == 5
    assert "set by the user" in result.selection


def test_stocks_with_incomplete_history_are_excluded_and_listed(market, small_config):
    returns, universe, _ = market
    returns = returns.copy()
    returns.iloc[:5, 0] = np.nan
    extra = pd.concat(
        [
            universe,
            pd.DataFrame({"symbol": ["GONE"], "name": ["Gone"], "sector": ["Alpha"], "sub_industry": ["x"]}),
        ]
    )
    result = run_step1(small_config, universe=extra, prices=returns_to_prices(returns))
    assert len(result.symbols) == 44
    assert set(result.excluded["symbol"]) == {returns.columns[0], "GONE"}


def test_too_few_stocks_or_days(market, small_config):
    returns, universe, _ = market
    with pytest.raises(ValueError, match="complete history"):
        run_step1(small_config, universe=universe.head(8), prices=returns_to_prices(returns.iloc[:, :8]))
    with pytest.raises(ValueError, match="at least 30"):
        run_step1(small_config, universe=universe, prices=returns_to_prices(returns.head(20)))


def test_edge_note_when_best_k_is_at_the_limit(market, small_config):
    result = run(market, small_config, k_min=3, k_max=3)
    assert "edge of the allowed range" in build_report(result)
    assert "edge of the allowed range" not in build_report(run(market, small_config, k_min=2, k_max=8, k=4))


def test_write_outputs_creates_every_file_and_report_is_consistent(market, small_config, tmp_path):
    result = run(market, small_config)
    folder = write_outputs(result, tmp_path / "results")
    names = {p.name for p in folder.iterdir()}
    expected = {
        "membership.csv",
        "cluster_summary.csv",
        "sector_by_cluster.csv",
        "sector_correlation.csv",
        "pair_summary.csv",
        "strongest_pairs.csv",
        "most_negative_pairs.csv",
        "stock_connectivity.csv",
        "linkage_comparison.csv",
        "linkage_agreement.csv",
        "k_sensitivity.csv",
        "excluded_stocks.csv",
        "bootstrap_stability.csv",
        "step1_report.md",
        "run_summary.json",
        *plotting_files(),
    }
    assert expected <= names
    assert (folder / "correlation_heatmap.png").stat().st_size > 5000
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["k"] == result.k and summary["stocks_clustered"] == 45
    report = (folder / "step1_report.md").read_text(encoding="utf-8")
    assert f"{result.linkage} linkage, {result.k} clusters" in report
    for image in plotting_files():
        assert f"]({image})" in report
    membership = pd.read_csv(folder / "membership.csv")
    assert len(membership) == 45 and membership["cluster"].nunique() == 3
    assert not (folder / "correlation_matrix.csv").exists()


def plotting_files():
    from market_structure.report import FIGURES

    return list(FIGURES)


def test_markdown_table_formats_and_blanks_missing_values():
    table = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]})
    text = markdown_table(table, {"a": "{:.2f}"})
    assert "| 1.23 | x |" in text and "|  | y |" in text


def test_cut_height_separates_exactly_k_clusters(market):
    from scipy.cluster.hierarchy import fcluster

    from market_structure import clustering as cl

    returns, _, _ = market
    tree = cl.build_tree(cl.correlation_distance(cl.correlation_matrix(returns)), "average")
    for k in (1, 2, 5, 45):
        labels = fcluster(tree, plotting.cut_height(tree, k), criterion="distance")
        assert len(set(labels)) == k


def test_cli_runs_offline_from_saved_data_and_reports_errors(market, tmp_path, capsys):
    returns, universe, _ = market
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    universe.to_csv(raw / "sp500_universe.csv", index=False)
    returns_to_prices(returns).to_csv(raw / "sp500_prices_2025-01-01_2026-01-01.csv", index=False)
    out = tmp_path / "out"
    args = [
        "step1",
        "--start",
        "2025-01-01",
        "--end",
        "2026-01-01",
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(out),
        "--k-min",
        "2",
        "--k-max",
        "8",
        "--bootstrap",
        "5",
    ]
    # k_scan_max defaults to 40, which needs more than 40 stocks; the 45 here are enough
    assert cli.main(args) == 0
    assert "45 stocks, 250 days" in capsys.readouterr().out
    assert (out / "step1_report.md").exists()
    assert cli.main([*args[:-2], "--bootstrap", "-3"]) == 1
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["step1", "--linkage", "nope"])
    with pytest.raises(SystemExit):
        cli.main([])
