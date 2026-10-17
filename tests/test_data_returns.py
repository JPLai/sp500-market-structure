import numpy as np
import pandas as pd
import pytest

from market_structure import data
from market_structure.returns import log_returns, price_table, select_complete

HTML = """
<html><body><table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>AAA</td><td>Alpha Corp</td><td>Tech</td><td>Software</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td></tr>
<tr><td>AAA</td><td>Alpha Corp</td><td>Tech</td><td>Software</td></tr>
</table></body></html>
"""


def test_fetch_universe_parses_cleans_and_deduplicates():
    universe = data.fetch_universe(fetch_html=lambda url: HTML)
    assert list(universe.columns) == data.UNIVERSE_COLUMNS
    assert list(universe["symbol"]) == ["AAA", "BRK-B"]
    assert universe.loc[0, "sector"] == "Tech"


def test_fetch_universe_rejects_unexpected_page():
    with pytest.raises(ValueError):
        data.fetch_universe(
            fetch_html=lambda url: "<table><tr><th>Symbol</th></tr><tr><td>X</td></tr></table>"
        )


def fake_downloader(calls):
    def download(tickers, **kwargs):
        calls.append(tickers)
        index = pd.date_range("2025-01-02", periods=3, tz="America/New_York")
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close", "Volume"]])
        frame = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4) + 1, index=index, columns=columns)
        frame.loc[index[0], ("BBB", "Close")] = np.nan  # BBB starts a day later
        return frame

    return download


def test_download_prices_long_format_ignores_unknown_symbols():
    prices = data.download_prices(["aaa", "BBB", "ZZZ", " "], "2025-01-01", "2025-02-01", fake_downloader([]))
    assert list(prices.columns) == ["timestamp", "symbol", "close"]
    assert set(prices["symbol"]) == {"AAA", "BBB"}
    assert len(prices) == 3 + 2
    assert prices["timestamp"].dt.tz is None
    assert not prices["close"].isna().any()


def test_download_prices_single_symbol_and_errors():
    def single(tickers, **kwargs):
        index = pd.date_range("2025-01-02", periods=2)
        return pd.DataFrame({"Close": [1.0, 2.0], "Volume": [5, 6]}, index=index)

    assert set(data.download_prices(["AAA"], "a", "b", single)["symbol"]) == {"AAA"}
    with pytest.raises(ValueError, match="At least one"):
        data.download_prices([], "a", "b", single)
    with pytest.raises(ValueError, match="empty"):
        data.download_prices(["AAA"], "a", "b", lambda **k: pd.DataFrame())


def test_load_prices_uses_the_saved_copy_unless_refreshed(tmp_path):
    calls = []
    path = tmp_path / "raw" / "prices.csv"
    kwargs = {"downloader": fake_downloader(calls)}
    first = data.load_prices(path, ["AAA", "BBB"], "a", "b", **kwargs)
    again = data.load_prices(path, ["AAA", "BBB"], "a", "b", **kwargs)
    assert len(calls) == 1
    assert len(first) == len(again) and path.exists()
    data.load_prices(path, ["AAA", "BBB"], "a", "b", refresh=True, **kwargs)
    assert len(calls) == 2


def test_partial_download_is_rejected_and_not_saved(tmp_path):
    path = tmp_path / "prices.csv"
    symbols = ["AAA", "BBB"] + [f"X{i}" for i in range(8)]  # the fake only knows AAA and BBB
    with pytest.raises(ValueError, match="only 2 of 10"):
        data.load_prices(path, symbols, "a", "b", downloader=fake_downloader([]))
    assert not path.exists()


def test_load_universe_caches(tmp_path):
    path = tmp_path / "u.csv"
    assert len(data.load_universe(path, fetch_html=lambda url: HTML)) == 2
    assert len(data.load_universe(path, fetch_html=lambda url: pytest.fail("fetched again"))) == 2


def long_prices(table):
    return (
        table.rename_axis("timestamp")
        .reset_index()
        .melt("timestamp", var_name="symbol", value_name="close")
        .dropna()
    )


def test_log_returns_and_price_table():
    index = pd.date_range("2025-01-01", periods=3)
    closes = price_table(
        long_prices(pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [1.0, 0.0, 2.0]}, index=index))
    )
    returns = log_returns(closes)
    assert len(returns) == 2
    assert returns["A"].iloc[0] == pytest.approx(np.log(1.1))
    assert returns["B"].isna().all()  # a zero price gives non-finite returns, treated as missing


def test_price_table_rejects_duplicates():
    prices = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-01-01"] * 2), "symbol": ["A", "A"], "close": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        price_table(prices)


def test_select_complete_keeps_only_full_histories_and_explains_the_rest():
    index = pd.date_range("2025-01-01", periods=10)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.standard_normal((10, 4)), index=index, columns=["FULL", "GAPPY", "SHORT", "FLAT"]
    )
    returns.loc[index[3], "GAPPY"] = np.nan
    returns.loc[index[:7], "SHORT"] = np.nan
    returns["FLAT"] = 0.0
    kept, excluded = select_complete(returns, ["FULL", "GAPPY", "SHORT", "FLAT", "GONE"])
    assert list(kept.columns) == ["FULL"]
    reasons = excluded.set_index("symbol")["reason"]
    assert reasons["GAPPY"] == "missing 1 of 10 daily returns"
    assert reasons["SHORT"] == "missing 7 of 10 daily returns"
    assert reasons["FLAT"].startswith("constant")
    assert reasons["GONE"] == "no price data"
    assert kept.notna().all().all()
