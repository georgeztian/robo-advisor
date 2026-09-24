"""ETF data validation (spec §3 "ETF Data Validation").

Checks per ticker: inception date, availability of the 20-year window, missing observations,
trading history, adjusted-price consistency with splits and distributions, expense ratio,
dividend distributions, and whether the ETF existed throughout the estimation window.
Short histories are *flagged* and the maximum available history is used.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import DataCfg
from ..models import DataQualityReport, TickerQuality
from ..universe import CATALOG
from .providers import trading_calendar


def total_return_from_raw(df: pd.DataFrame) -> pd.Series:
    """Daily total return reconstructed from raw close, distributions and splits:
    r_t = (close_t + div_t) * split_t / close_{t-1} - 1."""
    return (df["close"] + df["dividend"]) * df["split_ratio"] / df["close"].shift(1) - 1


def validate(frames: dict[str, pd.DataFrame], window_start: dt.date, as_of: dt.date,
             cfg: DataCfg) -> DataQualityReport:
    cal = trading_calendar(window_start, as_of)
    out: dict[str, TickerQuality] = {}
    blocking: list[str] = []
    warnings: list[str] = []
    for t, df in frames.items():
        info = CATALOG.get(t)
        issues: list[str] = []
        if df.empty:
            q = TickerQuality(t, info.inception if info else None, None, None, 0.0, False, False,
                              0, 1.0, 0, float("nan"), 0, 0, info.expense_ratio if info else None,
                              ["no observations in window"])
            out[t] = q
            blocking.append(f"{t}: no observations in the estimation window")
            continue
        first, last = df.index[0].date(), df.index[-1].date()
        if last > as_of:
            issues.append("observations after as-of date (look-ahead)")
            blocking.append(f"{t}: data after as-of date {as_of} (look-ahead)")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            issues.append("unsorted or duplicate dates")
            blocking.append(f"{t}: unsorted or duplicate dates")
        if (df[["close", "adj_close"]] <= 0).any().any() or df[["close", "adj_close"]].isna().any().any():
            issues.append("non-positive or missing prices")
            blocking.append(f"{t}: non-positive or missing prices")
        expected = cal[(cal >= df.index[0]) & (cal <= df.index[-1])]
        missing_idx = expected.difference(df.index)
        missing_frac = len(missing_idx) / max(len(expected), 1)
        max_gap = 0
        if len(missing_idx):
            pos = np.searchsorted(expected, missing_idx)
            runs = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
            max_gap = max(len(r) for r in runs)
            issues.append(f"{len(missing_idx)} missing trading days (max gap {max_gap})")
            if missing_frac > cfg.max_missing_fraction:
                blocking.append(f"{t}: {missing_frac:.2%} missing observations exceeds "
                                f"{cfg.max_missing_fraction:.2%}")
            elif max_gap > cfg.max_ffill_days:
                blocking.append(f"{t}: gap of {max_gap} trading days exceeds fill limit {cfg.max_ffill_days}")
            else:
                warnings.append(f"{t}: {len(missing_idx)} missing day(s) forward-filled for alignment")
        adj_ret = df["adj_close"].pct_change()
        raw_ret = total_return_from_raw(df)
        err = float((adj_ret - raw_ret).abs().max(skipna=True)) if len(df) > 1 else 0.0
        if err > cfg.adj_consistency_tol:
            issues.append(f"adjusted close inconsistent with splits/distributions (max err {err:.4f})")
            blocking.append(f"{t}: adjusted prices inconsistent with splits/distributions (max err {err:.4f})")
        if len(df) < cfg.min_observations:
            issues.append(f"only {len(df)} observations")
            blocking.append(f"{t}: only {len(df)} daily observations up to {as_of}; at least "
                            f"{cfg.min_observations} are needed to estimate risk")
        n_splits = int((df["split_ratio"] != 1).sum())
        n_div = int((df["dividend"] > 0).sum())
        years = (last - max(first, window_start)).days / 365.25
        meets = years >= cfg.min_history_years - 0.02
        inception = info.inception if info else first
        existed = inception <= window_start
        if not meets:
            warnings.append(f"{t}: only {years:.1f} years of history (inception {inception}); "
                            f"using maximum available history")
        if info is None:
            warnings.append(f"{t}: not in ETF catalog (no expense ratio / tax metadata)")
        stale = len(cal[(cal > df.index[-1])])
        if stale > 5:
            blocking.append(f"{t}: last observation {last} is {stale} trading days before as-of")
        yrs_span = max((last - first).days / 365.25, 1e-9)
        yld = float((df["dividend"] / df["close"].shift(1)).sum() / yrs_span)
        if yld > 0.20:
            warnings.append(f"{t}: implausible distribution yield {yld:.1%}")
        out[t] = TickerQuality(
            ticker=t, inception=inception, first_obs=first, last_obs=last,
            years_available=round(years, 2), meets_min_history=meets, existed_full_window=existed,
            missing_days=len(missing_idx), missing_fraction=missing_frac, max_gap_days=max_gap,
            adj_consistency_max_error=err, n_splits=n_splits, n_distributions=n_div,
            expense_ratio=info.expense_ratio if info else None, issues=issues)
    return DataQualityReport(window_start, as_of, out, blocking, warnings)
