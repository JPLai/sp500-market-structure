"""Daily log returns and the rule for which stocks can be clustered."""

import numpy as np
import pandas as pd


def price_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide table of closes: one row per date, one column per symbol."""
    if prices.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("The prices contain duplicate (symbol, timestamp) rows")
    return prices.pivot(index="timestamp", columns="symbol", values="close").sort_index()


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, ``ln(P_t / P_{t-1})``. Gaps stay missing; nothing is filled.

    Non-positive prices give non-finite returns, which are treated as missing.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(closes).diff().iloc[1:]
    return returns.where(np.isfinite(returns))


def select_complete(returns: pd.DataFrame, universe_symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the stocks with a return on every day; report every other stock and why.

    A complete history makes the correlation matrix exactly valid (positive semi-definite), so the
    correlation distance is a true Euclidean distance. Returns are never filled or imputed.
    """
    days = len(returns)
    reasons: dict[str, tuple[int, str]] = {}
    for symbol in universe_symbols:
        if symbol not in returns.columns:
            reasons[symbol] = (0, "no price data")
            continue
        column = returns[symbol]
        available = int(column.notna().sum())
        if available < days:
            reasons[symbol] = (available, f"missing {days - available} of {days} daily returns")
        elif column.std() == 0:
            reasons[symbol] = (available, "constant price (no variation)")
    exclusions = pd.DataFrame(
        [(symbol, n, reason) for symbol, (n, reason) in reasons.items()],
        columns=["symbol", "days_available", "reason"],
    )
    kept = [s for s in universe_symbols if s in returns.columns and s not in reasons]
    return returns[kept], exclusions.sort_values("symbol", ignore_index=True)
