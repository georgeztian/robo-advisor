"""Market-data providers.

Canonical per-ticker daily frame (DatetimeIndex, ascending):

    close        raw traded close (NOT split-adjusted)
    adj_close    total-return adjusted close (splits + distributions)
    dividend     cash distribution per share on the ex-date (raw, per share at that time)
    split_ratio  new shares per old share on the split date (1.0 otherwise)

Providers must never return observations after the requested end date (no look-ahead).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay

from ..universe import CATALOG

COLUMNS = ["close", "adj_close", "dividend", "split_ratio"]


class DataProvider(Protocol):
    name: str
    synthetic: bool

    def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        ...


def _clip(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)), COLUMNS]


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular NYSE holidays (incl. Good Friday; no Columbus/Veterans Day) plus the one-off
    closures inside the estimation window (national days of mourning, Hurricane Sandy)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ] + [Holiday(f"Special closure {d}", year=d.year, month=d.month, day=d.day) for d in (
        dt.date(2004, 6, 11), dt.date(2007, 1, 2), dt.date(2012, 10, 29), dt.date(2012, 10, 30),
        dt.date(2018, 12, 5), dt.date(2025, 1, 9))]


_BDAY = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def trading_calendar(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    """Approximate NYSE trading calendar."""
    return pd.date_range(start, end, freq=_BDAY)


# --------------------------------------------------------------------------- synthetic

# Base-asset calibration (annual arithmetic mean of log-return drift, annual vol).
_BASE = ["US", "GROWTH", "VALUE", "INTL", "EM", "REIT", "AGG", "LTSY", "CASH", "GOLD"]
_BASE_MU = np.array([0.100, 0.125, 0.092, 0.075, 0.085, 0.090, 0.036, 0.045, 0.018, 0.070])
_BASE_VOL = np.array([0.165, 0.210, 0.150, 0.170, 0.220, 0.230, 0.050, 0.140, 0.004, 0.160])
_BASE_CORR = np.array([
    #  US    GRO   VAL   INTL  EM    REIT  AGG   LTSY  CASH  GOLD
    [1.00, 0.92, 0.93, 0.85, 0.75, 0.72, 0.05, -0.25, 0.00, 0.05],
    [0.92, 1.00, 0.78, 0.78, 0.72, 0.60, 0.02, -0.22, 0.00, 0.03],
    [0.93, 0.78, 1.00, 0.82, 0.70, 0.75, 0.08, -0.20, 0.00, 0.06],
    [0.85, 0.78, 0.82, 1.00, 0.85, 0.68, 0.10, -0.15, 0.00, 0.15],
    [0.75, 0.72, 0.70, 0.85, 1.00, 0.60, 0.10, -0.12, 0.00, 0.22],
    [0.72, 0.60, 0.75, 0.68, 0.60, 1.00, 0.20, 0.05, 0.00, 0.10],
    [0.05, 0.02, 0.08, 0.10, 0.10, 0.20, 1.00, 0.85, 0.10, 0.30],
    [-0.25, -0.22, -0.20, -0.15, -0.12, 0.05, 0.85, 1.00, 0.05, 0.25],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.05, 1.00, 0.02],
    [0.05, 0.03, 0.06, 0.15, 0.22, 0.10, 0.30, 0.25, 0.02, 1.00],
])

# Stress episodes: (start, end, total log-return overlay per base asset, vol multiplier)
_EPISODES = [
    ("2008-09-15", "2009-03-09", {"US": -0.55, "GROWTH": -0.50, "VALUE": -0.60, "INTL": -0.60,
                                  "EM": -0.65, "REIT": -0.85, "LTSY": 0.15, "GOLD": 0.05}, 2.5),
    ("2020-02-20", "2020-03-23", {"US": -0.38, "GROWTH": -0.30, "VALUE": -0.42, "INTL": -0.36,
                                  "EM": -0.35, "REIT": -0.50, "LTSY": 0.10, "AGG": -0.03}, 3.0),
    ("2022-01-03", "2022-10-14", {"US": -0.22, "GROWTH": -0.38, "VALUE": -0.10, "INTL": -0.25,
                                  "EM": -0.28, "REIT": -0.30, "AGG": -0.16, "LTSY": -0.40,
                                  "GOLD": -0.08}, 1.4),
]

# ETF construction: base-asset mix, idiosyncratic vol, annual distribution yield, dist. frequency
_ETF_SPEC: dict[str, tuple[dict[str, float], float, float, int]] = {
    "VOO": ({"US": 1.0}, 0.004, 0.015, 4),
    "VTI": ({"US": 1.0}, 0.012, 0.015, 4),
    "BND": ({"AGG": 1.0}, 0.006, 0.030, 12),
    "TLT": ({"LTSY": 1.0}, 0.010, 0.032, 12),
    "BIL": ({"CASH": 1.0}, 0.001, 0.017, 12),
    "SGOV": ({"CASH": 1.0}, 0.001, 0.030, 12),
    "GLD": ({"GOLD": 1.0}, 0.010, 0.000, 0),
    "QQQ": ({"GROWTH": 1.0}, 0.010, 0.007, 4),
    "VXUS": ({"INTL": 0.78, "EM": 0.22}, 0.012, 0.030, 4),
    "VT": ({"US": 0.60, "INTL": 0.30, "EM": 0.10}, 0.008, 0.021, 4),
    "VWO": ({"EM": 1.0}, 0.015, 0.032, 4),
    "VNQ": ({"REIT": 1.0}, 0.012, 0.038, 4),
    "SCHH": ({"REIT": 1.0}, 0.015, 0.030, 4),
    "SCHD": ({"VALUE": 1.0}, 0.030, 0.033, 4),
    "VYM": ({"VALUE": 1.0}, 0.025, 0.030, 4),
    "DGRO": ({"VALUE": 0.6, "US": 0.4}, 0.025, 0.023, 4),
}


def _near_psd_corr(c: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((c + c.T) / 2)
    c2 = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


class SyntheticProvider:
    """Deterministic synthetic ETF histories for offline demos and tests.

    Calibrated to each ETF's real inception date, typical long-run return/volatility,
    cross-asset correlations, distribution yield/frequency and three historical stress
    episodes (GFC, COVID crash, 2022 rate shock). Prices are generated once over a fixed
    calendar (2000 through 2030) and then truncated to the request, so moving the as-of
    date never changes earlier history (look-ahead-safe). Raw closes split 2:1 whenever
    they exceed $500. A few missing observations are injected into SCHH / VWO so data
    validation is exercised.

    THIS IS NOT REAL MARKET DATA. Reports built on it are watermarked SYNTHETIC.
    """

    name = "synthetic"
    synthetic = True
    _GEN_START = dt.date(2000, 1, 3)
    _GEN_END = dt.date(2030, 12, 31)

    def __init__(self, seed: int = 26, inject_gaps: bool = True):
        self.seed = seed
        self.inject_gaps = inject_gaps
        self._cache: dict[str, pd.DataFrame] = {}
        self._base: pd.DataFrame | None = None

    def _base_returns(self) -> pd.DataFrame:
        if self._base is not None:
            return self._base
        idx = trading_calendar(self._GEN_START, self._GEN_END)
        n, k = len(idx), len(_BASE)
        rng = np.random.default_rng(self.seed)
        dvol = _BASE_VOL / np.sqrt(252)
        chol = np.linalg.cholesky(_near_psd_corr(_BASE_CORR))
        # fat tails: multivariate t with 5 dof, scaled to unit variance
        z = rng.standard_normal((n, k)) @ chol.T
        chi = rng.chisquare(5, size=(n, 1)) / 5
        z = z / np.sqrt(chi) * np.sqrt(3 / 5)
        volmult = np.ones((n, 1))
        overlay = np.zeros((n, k))
        for s, e, shocks, vm in _EPISODES:
            mask = (idx >= s) & (idx <= e)
            m = mask.sum()
            volmult[mask] = vm
            for a, tot in shocks.items():
                overlay[mask, _BASE.index(a)] += tot / m
        # episodes add drawdowns; lift the calm-period drift so full-sample means stay near target
        n_years = n / 252
        epi_total = np.array([sum(sh.get(a, 0.0) for _, _, sh, _ in _EPISODES) for a in _BASE])
        drift = (_BASE_MU - 0.5 * _BASE_VOL**2) / 252 - epi_total / n_years / 252
        logret = drift + z * dvol * volmult + overlay
        self._base = pd.DataFrame(logret, index=idx, columns=_BASE)
        return self._base

    def _make(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        if ticker not in CATALOG:
            raise KeyError(f"no synthetic calibration for {ticker}")
        base = self._base_returns()
        info = CATALOG[ticker]
        rng = np.random.default_rng(self.seed * 1000 + sum(map(ord, ticker)))
        if ticker == "TQQQ":   # 3x daily QQQ, minus fees and financing (cash) cost
            qqq = np.expm1(self._total_log_returns("QQQ", rng))
            cash = np.expm1(base["CASH"].to_numpy())
            simple = 3 * qqq - 2 * cash - info.expense_ratio / 252
            tr = np.log1p(np.clip(simple, -0.95, None))
            yld, freq = 0.002, 4
        else:
            tr = self._total_log_returns(ticker, rng)
            _, _, yld, freq = _ETF_SPEC[ticker]
        idx = base.index
        live = idx >= pd.Timestamp(info.inception)
        idx, tr = idx[live], tr[live]
        tri = np.exp(np.cumsum(tr))          # total-return index
        tri = tri / tri[0]
        # distributions on the last trading day of each distribution period
        div_flag = np.zeros(len(idx), bool)
        if freq:
            per = idx.to_period("M") if freq == 12 else idx.to_period("Q")
            last = pd.Series(np.arange(len(idx)), index=idx).groupby(per).max().to_numpy()
            div_flag[last[:-1]] = True
        close = np.empty(len(idx))
        div = np.zeros(len(idx))
        split = np.ones(len(idx))
        price = {"BIL": 91.0, "SGOV": 100.0, "TQQQ": 40.0}.get(ticker, 50.0 + len(ticker) * 10)
        close[0] = price
        for i in range(1, len(idx)):
            gross = tri[i] / tri[i - 1]
            p = close[i - 1] * gross               # total value per old share
            d = close[i - 1] * yld / freq if div_flag[i] else 0.0
            p -= d
            if p > 500:                            # 2:1 split: raw close and dividend per new share
                split[i] = 2.0
                p /= 2.0
                d /= 2.0
            close[i], div[i] = p, d
        # adjusted close: total-return index rebased so last value equals last raw close
        adj = tri * close[-1] / tri[-1]
        df = pd.DataFrame({"close": close, "adj_close": adj, "dividend": div, "split_ratio": split},
                          index=idx)
        if self.inject_gaps and ticker in ("SCHH", "VWO"):
            g = np.random.default_rng(self.seed + len(ticker))
            quiet = np.flatnonzero((df["dividend"].to_numpy() == 0) & (df["split_ratio"].to_numpy() == 1))
            quiet = quiet[(quiet > 10) & (quiet < len(df) - 10)]
            quiet = quiet[np.isin(quiet + 1, quiet)]      # the following day is quiet too
            drop = g.choice(quiet, size=6, replace=False)
            if ticker == "VWO":                    # one 2-day gap as well
                drop = np.concatenate([drop, [drop[0] + 1]])
            df = df.drop(df.index[np.unique(drop)])
        self._cache[ticker] = df
        return df

    def _total_log_returns(self, ticker: str, rng: np.random.Generator) -> np.ndarray:
        base = self._base_returns()
        mix, idio, _, _ = _ETF_SPEC[ticker]
        info = CATALOG[ticker]
        simple = sum(w * np.expm1(base[a].to_numpy()) for a, w in mix.items())
        simple = simple + rng.standard_normal(len(base)) * idio / np.sqrt(252) - info.expense_ratio / 252
        return np.log1p(simple)

    def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        return {t: _clip(self._make(t), start, end) for t in tickers}


# --------------------------------------------------------------------------- CSV


class CSVProvider:
    """Reads ``<dir>/<TICKER>.csv`` with columns date, close, adj_close[, dividend, split_ratio]."""

    name = "csv"
    synthetic = False

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        out = {}
        for t in tickers:
            path = self.dir / f"{t}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing price file {path}")
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            if "dividend" not in df:
                df["dividend"] = 0.0
            if "split_ratio" not in df:
                df["split_ratio"] = 1.0
            df["split_ratio"] = df["split_ratio"].replace(0, 1.0).fillna(1.0)
            df["dividend"] = df["dividend"].fillna(0.0)
            out[t] = _clip(df, start, end)
        return out


# --------------------------------------------------------------------------- Yahoo


class DataUnavailableError(RuntimeError):
    """Market data could not be obtained (network, rate limit, unknown ticker)."""


def normalize_yahoo(h: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Convert a ``yfinance`` ``history(auto_adjust=False, actions=True)`` frame to the canonical
    raw convention.

    Yahoo reports Close and Dividends split-adjusted; they are converted back to raw values
    (x product of all LATER splits) so split handling can be validated. Fund capital-gain
    distributions are cash distributions and are added to ``dividend``. Rows without prices
    (Yahoo sometimes emits a dividend on a non-trading date) pass their distributions and splits
    to the next priced row. Duplicate dates keep the last row.
    """
    if h is None or h.empty:
        raise DataUnavailableError(f"Yahoo returned no data for {ticker}")
    missing = {"Close", "Adj Close"} - set(h.columns)
    if missing:
        raise DataUnavailableError(f"Yahoo data for {ticker} lacks columns {sorted(missing)}; "
                                   "request it with auto_adjust=False")
    h = h.copy()
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    dist = h.get("Dividends", pd.Series(0.0, index=h.index)).fillna(0.0)
    if "Capital Gains" in h:
        dist = dist + h["Capital Gains"].fillna(0.0)
    splits = h.get("Stock Splits", pd.Series(0.0, index=h.index)).fillna(0.0).replace(0.0, 1.0)
    priced = h["Close"].notna() & h["Adj Close"].notna() & (h["Close"] > 0) & (h["Adj Close"] > 0)
    # carry events on unpriced rows to the next priced row
    group = priced[::-1].cumsum()[::-1]            # rows share a group with the next priced row
    dist = dist.groupby(group).transform("sum")
    splits = splits.groupby(group).transform("prod")
    h, dist, splits = h[priced], dist[priced], splits[priced]
    if h.empty:
        raise DataUnavailableError(f"Yahoo returned no priced rows for {ticker}")
    future = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return pd.DataFrame({
        "close": h["Close"] * future,
        "adj_close": h["Adj Close"],
        "dividend": dist * future,
        "split_ratio": splits,
    })


class YahooProvider:
    """Yahoo Finance via ``yfinance`` (optional dependency: ``pip install -e ".[yahoo]"``).

    * Full daily history (``period="max"``), unadjusted Close + Adj Close + actions, with
      yfinance's price repair (100x errors, bad dividend adjustments) enabled.
    * Retries with exponential backoff on rate limits / transient network errors.
    * Cached per ticker as CSV; a cache that ends before the requested end date is refreshed
      (at most once per ``refresh_hours``), so repeated runs work offline.
    """

    name = "yahoo"
    synthetic = False

    def __init__(self, cache_dir: str | Path | None = None, retries: int = 4, backoff: float = 2.0,
                 repair: bool = True, refresh_hours: float = 12.0):
        self.cache = Path(cache_dir) if cache_dir else None
        self.retries, self.backoff, self.repair, self.refresh_hours = retries, backoff, repair, refresh_hours
        self.notes: list[str] = []                 # surfaced as data-quality warnings

    @staticmethod
    def _yf():
        try:
            import yfinance as yf  # optional dependency
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise DataUnavailableError(
                "the Yahoo provider needs yfinance: pip install -e \".[yahoo]\"") from e
        return yf

    def _download(self, t: str) -> pd.DataFrame:
        import time

        yf = self._yf()
        last: Exception | None = None
        attempt = 0
        while attempt < self.retries:
            try:
                h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False, actions=True,
                                         repair=self.repair, raise_errors=True)
                return normalize_yahoo(h, t)
            except DataUnavailableError:
                raise
            except ImportError as e:
                # yfinance's price repair needs optional libraries (scipy, scikit-learn). Without
                # them, download unrepaired data rather than fail; validation still checks it.
                if not self.repair:
                    raise DataUnavailableError(f"yfinance failed for {t}: {e}") from e
                self.repair = False
                self.notes.append(f"Yahoo price repair disabled (missing library: {e.name or e}); "
                                  "reinstall with pip install -e \".[yahoo]\" to enable it")
                continue                           # retry immediately, not counted as an attempt
            except (TypeError, ValueError, KeyError, AttributeError) as e:
                # programming / data-format errors are not transient: do not retry
                raise DataUnavailableError(f"unexpected Yahoo data for {t}: {type(e).__name__}: {e}") from e
            except Exception as e:                 # rate limits, timeouts, transient HTTP errors
                last = e
                attempt += 1
                if attempt < self.retries:
                    time.sleep(self.backoff * 2**(attempt - 1))
        raise DataUnavailableError(f"could not download {t} from Yahoo after {self.retries} attempts: "
                                   f"{type(last).__name__}: {last}")

    def _cache_fresh(self, path: Path, df: pd.DataFrame, end: dt.date) -> bool:
        import time

        last = df.index[-1].date() if len(df) else dt.date.min
        wanted = min(end, dt.date.today())
        behind = len(trading_calendar(last + dt.timedelta(days=1), wanted)) if wanted > last else 0
        if behind <= 1:                            # today's bar may not exist yet
            return True
        age_h = (time.time() - path.stat().st_mtime) / 3600
        return age_h < self.refresh_hours

    def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        out = {}
        for t in tickers:
            path = self.cache / f"{t}.csv" if self.cache else None
            df = None
            if path is not None and path.exists():
                df = pd.read_csv(path, parse_dates=["date"], index_col="date")
                if not self._cache_fresh(path, df, end):
                    df = None
            if df is None:
                df = self._download(t)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.rename_axis("date").to_csv(path)
            out[t] = _clip(df, start, end)
        return out


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


def fetch_treasury_rate(series: str, start: dt.date, end: dt.date, cache_dir: str | Path | None = None,
                        reader=None) -> pd.Series:
    """Daily Treasury rate from FRED (e.g. DTB3 = 3-month T-bill), as an annual decimal rate.
    ``reader`` (url -> DataFrame) is injectable for tests; defaults to ``pandas.read_csv``."""
    path = Path(cache_dir) / f"FRED_{series}.csv" if cache_dir else None
    if path is not None and path.exists():
        cached = pd.read_csv(path, parse_dates=["date"], index_col="date")["rate"]
        if len(cached) and cached.index[0].date() <= start + dt.timedelta(days=7) and \
                cached.index[-1].date() >= min(end, dt.date.today()) - dt.timedelta(days=7):
            return cached[(cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))]
    try:
        raw = (reader or pd.read_csv)(FRED_URL.format(series=series, start=start, end=end))
    except Exception as e:
        raise DataUnavailableError(f"could not download FRED series {series}: {e}") from e
    date_col = next(c for c in raw.columns if c.lower() in ("date", "observation_date"))
    rate = pd.to_numeric(raw[series], errors="coerce") / 100
    out = pd.Series(rate.to_numpy(), index=pd.to_datetime(raw[date_col]), name="rate").dropna()
    if out.empty:
        raise DataUnavailableError(f"FRED series {series} returned no observations")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("date").to_csv(path)
    return out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]


def make_provider(kind: str, *, csv_dir: str = "data/prices", cache_dir: str | None = None,
                  seed: int = 26, retries: int = 4, refresh_hours: float = 12.0) -> DataProvider:
    if kind == "synthetic":
        return SyntheticProvider(seed=seed)
    if kind == "csv":
        return CSVProvider(csv_dir)
    if kind == "yahoo":
        return YahooProvider(cache_dir, retries=retries, refresh_hours=refresh_hours)
    raise ValueError(f"unknown data provider {kind!r}")
