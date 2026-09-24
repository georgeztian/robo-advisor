"""S&P 500 benchmark comparison (spec §14).

The recommended portfolio and the S&P 500 (VOO total return) are run through the SAME
monthly backtest: same start/end month-ends (most recent 10 years), same initial investment,
same monthly contributions, same return convention (monthly total returns, pre-tax), with the
portfolio rebalanced per the selected rule. If a held ETF has a shorter history, the
comparison starts at the first month all held ETFs traded, and this is flagged.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import RebalancingCfg
from .estimation import max_drawdown, month_end_prices
from .models import BenchmarkResult
from .rebalancing import due

BENCH_LABEL = "S&P 500"


def backtest(monthly: pd.DataFrame, w: np.ndarray, W0: float, C: float, rebal: RebalancingCfg) -> tuple[np.ndarray, np.ndarray]:
    """Return (lump-sum wealth path, wealth path with contributions), length len(monthly)+1."""
    R = monthly.to_numpy()
    paths = []
    for c in (0.0, C):
        V = W0 * w.copy()
        out = [V.sum()]
        for t, r in enumerate(R, start=1):
            V = V * (1 + r) + c * w
            total = V.sum()
            if total <= 0:
                V = np.zeros_like(V)
                out.extend([0.0] * (len(R) - t + 1))
                break
            if due(rebal, t, V / total, w):
                V = total * w
            out.append(V.sum())
        paths.append(np.array(out))
    return paths[0], paths[1]


def compare(frames: dict[str, pd.DataFrame], tickers: list[str], w: np.ndarray, W0: float, C: float,
            as_of: dt.date, years: int, rebal: RebalancingCfg, benchmark: str, rf_ticker: str,
            rf_fallback: float, rf_series: pd.Series | None = None) -> BenchmarkResult:
    held = [t for t, x in zip(tickers, w) if abs(x) > 1e-9]
    wh = np.array([x for x in w if abs(x) > 1e-9])
    start_req = pd.Timestamp(as_of) - pd.DateOffset(years=years)
    need = sorted(set(held) | {benchmark})
    px = pd.concat({t: frames[t]["adj_close"] for t in need}, axis=1, sort=True).sort_index()
    px = px[(px.index >= start_req - pd.DateOffset(months=1)) & (px.index <= pd.Timestamp(as_of))]
    mpx = month_end_prices(px.ffill(limit=5))
    # the first month-end on/after the requested start; later if a held ETF started later
    first_ok = mpx.dropna().index
    if len(first_ok) < 13:
        raise ValueError("fewer than 12 months of common history for the benchmark comparison")
    start_p = max(pd.Timestamp(start_req).to_period("M"), first_ok[0])
    mpx = mpx.loc[start_p:]
    monthly = mpx.pct_change(fill_method=None).iloc[1:]
    notes = []
    period_years = len(monthly) / 12
    if period_years < years - 1 / 12:
        late = [t for t in held if frames[t].index[0] > start_req]
        notes.append(f"Comparison covers {period_years:.1f} years (not {years}) because "
                     f"{', '.join(late)} started trading later.")
    rf_m = None
    if rf_ticker in frames:
        r = month_end_prices(frames[rf_ticker][["adj_close"]])["adj_close"].pct_change()
        rf_m = r.reindex(monthly.index).dropna()
    rf = float(rf_m.mean() * 12) if rf_m is not None and len(rf_m) > 6 else rf_fallback
    if rf_series is not None:                         # Treasury yield series, same period
        rs = rf_series[(rf_series.index > monthly.index[0].start_time) & (rf_series.index <= pd.Timestamp(as_of))]
        if len(rs) > 20:
            rf = float(rs.mean())

    # time-weighted index (unit lump sum) drives every return metric, so the comparison is
    # well defined even with a zero initial investment; wealth paths scale it / add contributions
    port_idx = backtest(monthly[held], wh, 1.0, 0.0, rebal)[0]
    port_contrib = backtest(monthly[held], wh, W0, C, rebal)[1]
    one = np.array([1.0])
    bench_idx = backtest(monthly[[benchmark]], one, 1.0, 0.0, rebal)[0]
    bench_contrib = backtest(monthly[[benchmark]], one, W0, C, rebal)[1]
    idx = [start_p] + list(monthly.index)
    total_contrib = W0 + C * len(monthly)

    def metrics(index: np.ndarray, contrib: np.ndarray) -> dict:
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(index[:-1] > 0, index[1:] / index[:-1] - 1, 0.0)
        n = len(rets)
        cum = index[-1] / index[0] - 1
        ann = (1 + cum) ** (12 / n) - 1
        vol = rets.std(ddof=1) * np.sqrt(12)
        s = pd.Series(rets, index=monthly.index)
        yearly = (1 + s).groupby(s.index.year).prod() - 1
        counts = s.groupby(s.index.year).size()
        full = yearly[counts == 12]
        yr = full if len(full) else yearly
        return {
            "Cumulative return": cum, "Annualized return": ann, "Annualized volatility": vol,
            "Sharpe ratio": (rets.mean() * 12 - rf) / vol if vol > 0 else np.nan,
            "Maximum drawdown": max_drawdown(index),
            "Ending wealth (initial investment only)": W0 * index[-1],
            "Ending wealth (with monthly contributions)": contrib[-1],
            "Total contributed": total_contrib,
            "Best year": float(yr.max()), "Best year (calendar)": int(yr.idxmax()),
            "Worst year": float(yr.min()), "Worst year (calendar)": int(yr.idxmin()),
            "_yearly": yearly,
        }

    mp, mb = metrics(port_idx, port_contrib), metrics(bench_idx, bench_contrib)
    months_in_year = pd.Series(1, index=monthly.index).groupby(monthly.index.year).size()
    cal = pd.DataFrame({"Portfolio": mp.pop("_yearly"), BENCH_LABEL: mb.pop("_yearly"),
                        "Months": months_in_year})
    table = pd.DataFrame({"Portfolio": mp, BENCH_LABEL: mb})
    g10 = pd.DataFrame({"Portfolio": port_idx * 10000, BENCH_LABEL: bench_idx * 10000}, index=pd.PeriodIndex(idx, freq="M"))
    wc = pd.DataFrame({"Portfolio": port_contrib, BENCH_LABEL: bench_contrib,
                       "Contributed": W0 + C * np.arange(len(idx))}, index=pd.PeriodIndex(idx, freq="M"))
    return BenchmarkResult(start=px[px.index.to_period("M") == start_p].index[-1].date(), end=px.index[-1].date(),
                           years=period_years, initial_investment=W0, monthly_contribution=C,
                           metrics=table, growth_of_10k=g10, wealth_with_contributions=wc,
                           calendar_year_returns=cal, notes=notes)
