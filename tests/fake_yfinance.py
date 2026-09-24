"""A stand-in for the ``yfinance`` module that serves histories in Yahoo's exact format.

Built from the synthetic provider's raw histories, then converted the way Yahoo reports them:
  * ``Close`` and ``Dividends`` are split-adjusted; ``Stock Splits`` is 0 on non-split days;
  * ``Adj Close`` uses Yahoo's multiplicative dividend factor (1 - D_t / Close_{t-1}), which
    differs slightly from a reinvested-total-return index;
  * the index is timezone-aware (America/New_York);
  * OHLC/Volume and a ``Capital Gains`` column are present.
Injected quirks (per ticker) mirror real Yahoo data problems:
  * VWO:  a dividend reported on a non-trading day (Saturday row with NaN prices);
  * SCHD: a capital-gain distribution (reflected in Adj Close);
  * VOO:  a duplicated last row;
  * BND:  one isolated Adj Close glitch day;
  * QQQ:  a rate-limit error on the first request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from robo_advisor.data.providers import SyntheticProvider

_SYN = SyntheticProvider(inject_gaps=False)
CALLS: dict[str, int] = {}
SPLIT_UNADJUSTED: set[str] = set()      # tickers served with a missed split adjustment
REPAIR_LIBS_MISSING = False              # simulate yfinance[repair] extras not installed


class YFRateLimitError(Exception):
    pass


def yahoo_frame(ticker: str) -> pd.DataFrame:
    raw = _SYN._make(ticker)
    raw = raw[raw.index <= pd.Timestamp.today().normalize()]             # Yahoo only has the past
    split = raw["split_ratio"].to_numpy()
    future = pd.Series(split[::-1]).cumprod()[::-1].shift(-1).fillna(1.0).to_numpy()
    close = raw["close"].to_numpy() / future
    divs = raw["dividend"].to_numpy() / future
    cap_gains = np.zeros(len(raw))
    if ticker == "SCHD":
        k = len(raw) // 2
        cap_gains[k] = 0.004 * close[k - 1]
    dist = divs + cap_gains
    m = np.ones(len(raw))
    m[1:] = 1 - dist[1:] / close[:-1]
    factor = np.append(np.cumprod(m[::-1])[::-1][1:], 1.0)          # product of LATER factors
    adj = close * factor
    if ticker in SPLIT_UNADJUSTED:
        adj = raw["close"].to_numpy() * factor                         # forgot to adjust for splits
    idx = raw.index.tz_localize("America/New_York")
    df = pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": np.round(close, 6),
        "Adj Close": np.round(adj, 6), "Volume": 1_000_000,
        "Dividends": divs, "Stock Splits": np.where(split == 1.0, 0.0, split), "Capital Gains": cap_gains,
    }, index=idx)
    if ticker == "VWO":                   # dividend on a Saturday row with no prices
        ex = next(i for i in np.flatnonzero(divs > 0) if df.index[i].weekday() == 0)   # a Monday ex-date
        sat = df.index[ex] - pd.Timedelta(days=2)
        row = df.iloc[[ex]].copy()
        row.index = [sat]
        row[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = np.nan
        df.iloc[ex, df.columns.get_loc("Dividends")] = 0.0
        df = pd.concat([df, row]).sort_index()
    if ticker == "VOO":
        df = pd.concat([df, df.iloc[[-1]]])
    if ticker == "BND":
        k = len(df) // 3
        df.iloc[k, df.columns.get_loc("Adj Close")] *= 1.01
    return df


class Ticker:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, period="1mo", interval="1d", auto_adjust=True, actions=True, repair=False,
                raise_errors=False, **kw) -> pd.DataFrame:
        assert period == "max" and interval == "1d" and auto_adjust is False and actions is True
        CALLS[self.symbol] = CALLS.get(self.symbol, 0) + 1
        if repair and REPAIR_LIBS_MISSING:     # what yfinance does without scikit-learn
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        if self.symbol == "QQQ" and CALLS[self.symbol] == 1:
            raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")
        return yahoo_frame(self.symbol)
