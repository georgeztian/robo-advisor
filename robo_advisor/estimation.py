"""Parameter estimation (spec §6).

* Daily adjusted (total-return) prices -> risk: volatility, covariance, downside statistics.
* Monthly total returns -> expected returns (arithmetic, annualized x12) for long horizons.
* Risk-free rate from a Treasury-bill ETF (BIL) over the same window.
* Consistent observation dates: all series aligned on one trading calendar; each ETF uses
  its maximum available history inside the window; pairwise-overlap covariance is repaired
  to the nearest positive semi-definite matrix.
* No look-ahead: callers pass data already truncated at the as-of date and this module
  asserts it.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .config import DataCfg, EstimationCfg
from .models import Estimates


class EstimationError(ValueError):
    pass


def align_prices(frames: dict[str, pd.DataFrame], tickers: list[str], max_ffill: int,
                 column: str = "adj_close") -> pd.DataFrame:
    """Union calendar; short gaps forward-filled only inside each ETF's live range."""
    px = pd.concat({t: frames[t][column] for t in tickers}, axis=1, sort=True).sort_index()
    filled = px.ffill(limit=max_ffill)
    for t in tickers:  # never fill past the last real observation
        last = frames[t].index[-1]
        filled.loc[filled.index > last, t] = np.nan
    return filled


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, bool]:
    """Clip negative eigenvalues of the correlation matrix, keep the variances."""
    cov = (cov + cov.T) / 2
    vals = np.linalg.eigvalsh(cov)
    if vals.min() >= -eps * max(1.0, vals.max()):
        return cov, False
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    w, v = np.linalg.eigh(corr)
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    s = np.sqrt(np.diag(corr))
    corr = corr / np.outer(s, s)
    return corr * np.outer(d, d), True


def max_drawdown(values: np.ndarray | pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1).min())


def var_cvar(returns: np.ndarray, conf: float) -> tuple[float, float]:
    """Historical VaR / Expected Shortfall as positive loss fractions."""
    r = np.sort(returns[~np.isnan(returns)])
    if len(r) == 0:
        return float("nan"), float("nan")
    k = max(1, int(np.floor((1 - conf) * len(r))))
    return float(-np.quantile(r, 1 - conf)), float(-r[:k].mean())


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.groupby(daily.index.to_period("M")).last()


def estimate(frames: dict[str, pd.DataFrame], tickers: list[str], as_of: dt.date,
             window_start: dt.date, data_cfg: DataCfg, cfg: EstimationCfg) -> Estimates:
    for t in tickers:
        if len(frames[t]) and frames[t].index[-1].date() > as_of:
            raise EstimationError(f"look-ahead: {t} has data after {as_of}")
    td = cfg.trading_days
    px = align_prices(frames, tickers, data_cfg.max_ffill_days)
    daily = px.pct_change(fill_method=None)
    daily = daily.iloc[1:]
    sigma = daily.std().to_numpy() * np.sqrt(td)
    cov_df = daily.cov(min_periods=cfg.min_overlap_days) * td
    if cov_df.isna().any().any():
        bad = [(a, b) for a in tickers for b in tickers if np.isnan(cov_df.loc[a, b])]
        raise EstimationError(f"insufficient overlapping history for covariance: {bad[:5]}")
    cov, repaired = nearest_psd(cov_df.to_numpy())
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    mpx = month_end_prices(px)
    monthly = mpx.pct_change(fill_method=None).iloc[1:]
    mu = monthly.mean().to_numpy() * 12
    if cfg.mu_shrinkage > 0:
        mu = (1 - cfg.mu_shrinkage) * mu + cfg.mu_shrinkage * mu.mean()

    # distributions: monthly income yield = sum(div / previous raw close) within the month
    inc_daily = pd.concat({t: frames[t]["dividend"] / frames[t]["close"].shift(1) for t in tickers}, sort=True,
                          axis=1).reindex(px.index).fillna(0.0)
    monthly_income = inc_daily.groupby(inc_daily.index.to_period("M")).sum().reindex(monthly.index)
    monthly_income = monthly_income.where(monthly.notna())
    income_yield = (monthly_income.mean() * 12).fillna(0.0).to_numpy()

    rf_t = data_cfg.risk_free_ticker
    if rf_t in frames and len(frames[rf_t]) > 60:
        rf_m = month_end_prices(frames[rf_t][["adj_close"]])["adj_close"].pct_change().dropna()
        risk_free = float(rf_m.mean() * 12)
    else:
        risk_free = data_cfg.risk_free_fallback

    bench_t = data_cfg.benchmark
    bench = frames[bench_t]["adj_close"].pct_change() if bench_t in frames else None
    rows = []
    hist_years = {}
    for i, t in enumerate(tickers):
        r = daily[t].dropna()
        live = px[t].dropna()
        yrs = (live.index[-1] - live.index[0]).days / 365.25
        hist_years[t] = round(yrs, 2)
        cagr = (live.iloc[-1] / live.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else float("nan")
        dn = r.clip(upper=0)
        downside = float(np.sqrt((dn**2).mean()) * np.sqrt(td))
        v, cv = var_cvar(r.to_numpy(), cfg.var_confidence)
        mv, mcv = var_cvar(monthly[t].dropna().to_numpy(), cfg.var_confidence)
        if bench is not None:
            j = pd.concat([r, bench], axis=1, join="inner").dropna()
            beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 20 else float("nan")
        else:
            beta = float("nan")
        rows.append({
            "ticker": t, "years": hist_years[t], "first_obs": live.index[0].date(),
            "exp_return": mu[i], "cagr": cagr, "volatility": sigma[i], "downside_dev": downside,
            "max_drawdown": max_drawdown(live.to_numpy()), "var_daily": v, "cvar_daily": cv,
            "var_monthly": mv, "cvar_monthly": mcv, "beta": beta,
            "sharpe": (mu[i] - risk_free) / sigma[i] if sigma[i] > 0 else float("nan"),
            "sortino": (mu[i] - risk_free) / downside if downside > 0 else float("nan"),
            "income_yield": income_yield[i],
        })
    stats = pd.DataFrame(rows).set_index("ticker")
    return Estimates(tickers=list(tickers), mu=mu, sigma=sigma, cov=cov, corr=corr,
                     risk_free=risk_free, daily_returns=daily, monthly_returns=monthly,
                     income_yield=income_yield, stats=stats, window_start=window_start,
                     window_end=as_of, history_years=hist_years, monthly_income=monthly_income,
                     psd_repaired=repaired)


def portfolio_risk_stats(w: np.ndarray, est: Estimates, conf: float = 0.95) -> dict:
    """Historical downside statistics for fixed weights over the common live period."""
    common = est.daily_returns.dropna()
    out = {"common_start": common.index[0].date() if len(common) else None,
           "common_days": len(common)}
    if len(common) < 20:
        return out | {"max_drawdown": float("nan"), "var_daily": float("nan"),
                      "cvar_daily": float("nan"), "downside_dev": float("nan")}
    rp = common.to_numpy() @ w
    idx = np.cumprod(1 + rp)
    v, cv = var_cvar(rp, conf)
    dn = np.clip(rp, None, 0)
    return out | {"max_drawdown": max_drawdown(idx), "var_daily": v, "cvar_daily": cv,
                  "downside_dev": float(np.sqrt((dn**2).mean()) * np.sqrt(252))}
