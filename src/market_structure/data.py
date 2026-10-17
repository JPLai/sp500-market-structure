"""The S&P 500 universe (with GICS sectors) and daily prices from Yahoo Finance, cached on disk."""

from collections.abc import Callable, Iterable
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "sp500-market-structure/0.1 (research project)"
UNIVERSE_COLUMNS = ["symbol", "name", "sector", "sub_industry"]
MIN_SYMBOL_SHARE = 0.9  # a download with fewer of the requested symbols than this is treated as failed


def _fetch_html(url: str) -> str:
    # Wikipedia rejects the default Python user agent.
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_universe(fetch_html: Callable[[str], str] = _fetch_html) -> pd.DataFrame:
    """Current S&P 500 members: ``symbol`` (Yahoo format), ``name``, GICS ``sector`` and ``sub_industry``.

    This is *today's* membership; using it for a past window has survivorship bias.
    """
    tables = pd.read_html(StringIO(fetch_html(SP500_URL)), match="Symbol")
    needed = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    if not tables or not needed <= set(tables[0].columns):
        raise ValueError("Could not find the S&P 500 table with Symbol and GICS columns")
    table = tables[0].dropna(subset=["Symbol"])
    universe = pd.DataFrame(
        {
            "symbol": table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
            "name": table["Security"].astype(str),
            "sector": table["GICS Sector"].astype(str),
            "sub_industry": table["GICS Sub-Industry"].astype(str),
        }
    )
    universe = universe.drop_duplicates("symbol").reset_index(drop=True)
    if universe.empty:
        raise ValueError("The S&P 500 table is empty")
    return universe[UNIVERSE_COLUMNS]


def download_prices(
    symbols: Iterable[str],
    start: str,
    end: str,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Adjusted daily closes in long format: ``timestamp``, ``symbol``, ``close``.

    ``end`` is exclusive. Closes are adjusted for splits and dividends. Symbols Yahoo has no data
    for are simply absent from the result.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    cleaned = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not cleaned:
        raise ValueError("At least one symbol is required")
    data = downloader(
        tickers=cleaned,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if data.empty:
        raise ValueError("Yahoo Finance returned an empty dataset")

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        available = set(data.columns.get_level_values(0))
        series = {s: data[s]["Close"] for s in cleaned if s in available}
    else:  # a single symbol comes back without the symbol level
        series = {cleaned[0]: data["Close"]}
    for symbol, close in series.items():
        close = close.dropna()
        index = pd.DatetimeIndex(close.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frames.append(pd.DataFrame({"timestamp": index, "symbol": symbol, "close": close.to_numpy()}))
    if not frames:
        raise ValueError("Yahoo Finance returned no price data for the requested symbols")
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["symbol", "timestamp"], ignore_index=True)


def load_universe(path: Path, refresh: bool = False, **kwargs) -> pd.DataFrame:
    """Read the saved universe, or fetch and save it if missing (or ``refresh``)."""
    if path.exists() and not refresh:
        return pd.read_csv(path)
    universe = fetch_universe(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(path, index=False)
    return universe


def load_prices(
    path: Path,
    symbols: Iterable[str],
    start: str,
    end: str,
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Read the saved prices, or download and save them if missing (or ``refresh``).

    A download that returns data for too few of the requested symbols (for example when Yahoo is
    rate-limiting) is rejected instead of being saved, so a bad copy is never reused.
    """
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["timestamp"])
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    prices = download_prices(requested, start, end, **kwargs)
    found = prices["symbol"].nunique()
    if found < MIN_SYMBOL_SHARE * len(requested):
        raise ValueError(
            f"Yahoo Finance returned data for only {found} of {len(requested)} symbols; "
            "nothing was saved. Try again in a few minutes."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)
    return prices
