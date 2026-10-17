"""Small synthetic market with planted groups, so every test runs offline in a moment."""

import numpy as np
import pandas as pd
import pytest

GROUPS = ["Alpha", "Beta", "Gamma"]
PER_GROUP = 15
DAYS = 250


def make_market(seed: int = 1, per_group: int = PER_GROUP, days: int = DAYS):
    """Returns (days x stocks), the universe table and the planted group of every stock.

    Each stock = 0.8 x its group's factor + 0.6 x its own noise, so the true within-group
    correlation is 0.64 and the between-group correlation is 0.
    """
    rng = np.random.default_rng(seed)
    symbols, planted = [], []
    columns = []
    for g, group in enumerate(GROUPS):
        factor = rng.standard_normal(days)
        for i in range(per_group):
            columns.append(0.01 * (0.8 * factor + 0.6 * rng.standard_normal(days)))
            symbols.append(f"{group[:1]}{i:02d}")
            planted.append(g)
    index = pd.bdate_range("2025-01-02", periods=days)
    returns = pd.DataFrame(np.column_stack(columns), index=index, columns=symbols)
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "name": [f"{s} Inc." for s in symbols],
            "sector": [GROUPS[g] for g in planted],
            "sub_industry": [f"{GROUPS[g]} sub" for g in planted],
        }
    )
    return returns, universe, np.array(planted)


def returns_to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Long price table (timestamp, symbol, close) whose log returns equal ``returns``."""
    start = returns.index[0] - pd.offsets.BDay(1)
    log_prices = returns.cumsum()
    log_prices.loc[start] = 0.0
    closes = 100 * np.exp(log_prices.sort_index())
    long = (
        closes.rename_axis("timestamp").reset_index().melt("timestamp", var_name="symbol", value_name="close")
    )
    return long.sort_values(["symbol", "timestamp"], ignore_index=True)


@pytest.fixture
def market():
    return make_market()


@pytest.fixture
def small_config(tmp_path):
    from market_structure.step1 import Step1Config

    return Step1Config(
        start="2025-01-01",
        end="2026-01-01",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        k_min=2,
        k_max=8,
        k_scan_max=10,
        bootstrap=10,
        top_pairs=5,
    )
