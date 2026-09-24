"""Independent recomputations used by the reviewer agent.

Deliberately written from scratch: this module must not import the production estimation,
optimization, simulation or benchmark code (enforced by tests/test_review_isolation.py), so a
bug in those modules cannot silently confirm itself.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def tr_index_from_raw(df: pd.DataFrame) -> pd.Series:
    """Total-return index from RAW close, distributions and splits (not from adj_close)."""
    close = df["close"].to_numpy(float)
    gross = np.ones(len(close))
    gross[1:] = (close[1:] + df["dividend"].to_numpy(float)[1:]) * df["split_ratio"].to_numpy(float)[1:] / close[:-1]
    return pd.Series(np.cumprod(gross), index=df.index)


def mu_sigma(frames: dict[str, pd.DataFrame], tickers: list[str], trading_days: int,
             shrink: float, as_of: dt.date) -> tuple[np.ndarray, np.ndarray]:
    """Convention: a month still in progress at the as-of date contributes no monthly return."""
    in_progress = (pd.Timestamp(as_of) + pd.offsets.BDay(1)).month == as_of.month
    mus, sigmas = [], []
    for t in tickers:
        idx = tr_index_from_raw(frames[t])
        daily = idx.pct_change().dropna()
        sigmas.append(daily.std() * np.sqrt(trading_days))
        month_end = idx.groupby([idx.index.year, idx.index.month]).last()
        if in_progress and month_end.index[-1] == (as_of.year, as_of.month):
            month_end = month_end.iloc[:-1]
        mus.append(month_end.pct_change().dropna().mean() * 12)
    mu = np.array(mus)
    if shrink > 0:
        mu = (1 - shrink) * mu + shrink * mu.mean()
    return mu, np.array(sigmas)


def weighted_score(options_by_q: dict[str, dict[str, float]], weights: dict[str, float],
                   answers: dict[str, str]) -> float:
    return round(sum(weights[q] * options_by_q[q][answers[q]] for q in weights), 2)


def fv_by_recursion(W0: float, C: float, r: float, T: int) -> float:
    v = W0
    for _ in range(T):
        v = v * (1 + r) + C
    return v


def mc_terminal(w: np.ndarray, mu_annual: np.ndarray, cov_annual: np.ndarray, W0: float, C: float,
                T: int, every: int | None, threshold: float | None, n: int, seed: int,
                history: np.ndarray | None = None) -> np.ndarray:
    """Independent Monte Carlo (pre-tax). Monthly simple returns are log-normal with the
    same first two moments as mu/12 and Sigma/12; or iid bootstrap rows of ``history``."""
    rng = np.random.default_rng(seed)
    a = 1 + mu_annual / 12
    logcov = np.log(1 + (cov_annual / 12) / np.outer(a, a))
    vals, vecs = np.linalg.eigh(logcov)
    root = vecs * np.sqrt(np.clip(vals, 0, None))
    logmean = np.log(a) - 0.5 * np.diag(logcov)
    hold = np.outer(np.full(n, W0), w)
    for month in range(1, T + 1):
        if history is not None:
            r = history[rng.integers(0, len(history), n)]
        else:
            r = np.exp(logmean + rng.standard_normal((n, len(w))) @ root.T) - 1
        hold = hold * (1 + r) + C * w
        tot = hold.sum(axis=1, keepdims=True)
        if every is not None:
            if month % every == 0:
                hold = tot * w
        elif threshold is not None:
            drift = np.abs(hold / np.where(tot == 0, 1, tot) - w).max(axis=1) > threshold
            hold[drift] = (tot * w)[drift]
        hold[tot[:, 0] <= 0] = 0.0
    return hold.sum(axis=1)


def month_end_series(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.Series:
    idx = tr_index_from_raw(df)
    idx = idx[(idx.index >= pd.Timestamp(start)) & (idx.index <= pd.Timestamp(end))]
    return idx.groupby(idx.index.to_period("M")).last()


def single_asset_wealth(me: pd.Series, W0: float, C: float) -> tuple[float, float]:
    """(ending wealth lump sum, ending wealth with contributions) investing at month ends."""
    r = me.pct_change().dropna().to_numpy()
    lump, contrib = W0, W0
    for x in r:
        lump *= 1 + x
        contrib = contrib * (1 + x) + C
    return lump, contrib


def is_psd(m: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.linalg.eigvalsh((m + m.T) / 2).min() >= -tol * max(1.0, np.abs(m).max()))


# ----------------------------------------------------------------------------- optimization
# An independent, deliberately plain re-solve used to verify the production optimizer's
# objective value. Variables: w (long-only) or [p, q] with w = p - q (shorts).

def _space(n: int, short: bool):
    if short:
        return 2 * n, (lambda x: x[:n] - x[n:])
    return n, (lambda x: x)


def resolve(kind: str, mu_long: np.ndarray, mu_short: np.ndarray, cov: np.ndarray, short: bool,
            max_pos: float, gross: float, vol_cap: float, rf: float = 0.0, min_return: float | None = None,
            starts: int = 8, seed: int = 4242,
            groups: list[tuple[np.ndarray, float]] = ()) -> tuple[np.ndarray | None, float]:
    """kind: 'max_return' | 'min_vol' | 'max_sharpe'. Returns (w, objective value in natural units:
    return for max_return, volatility for min_vol, Sharpe for max_sharpe). ``groups`` are
    (0/1 membership vector, limit) pairs: sum of |w| over each group <= limit."""
    from scipy.optimize import minimize

    n = len(mu_long)
    d, to_w = _space(n, short)

    def ret(x):
        return float(x[:n] @ mu_long - x[n:] @ mu_short) if short else float(x @ mu_long)

    def vol(x):
        w = to_w(x)
        return float(np.sqrt(max(w @ cov @ w, 1e-18)))

    obj = {"max_return": lambda x: -ret(x), "min_vol": lambda x: vol(x) ** 2,
           "max_sharpe": lambda x: -(ret(x) - rf) / vol(x)}[kind]
    cons = [{"type": "eq", "fun": lambda x: to_w(x).sum() - 1}]
    if kind != "min_vol":
        cons.append({"type": "ineq", "fun": lambda x: vol_cap**2 - vol(x) ** 2})
    if short:
        cons.append({"type": "ineq", "fun": lambda x: gross - x.sum()})
    if min_return is not None:
        cons.append({"type": "ineq", "fun": lambda x: ret(x) - min_return})
    for member, lim in groups:
        g = np.concatenate([member, member]) if short else member
        cons.append({"type": "ineq", "fun": lambda x, g=g, lim=lim: lim - g @ x})
    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    for k in range(starts):
        w0 = np.full(n, 1 / n) if k == 0 else rng.dirichlet(np.ones(n))
        w0 = np.minimum(w0, max_pos)
        w0 = w0 / w0.sum()
        x0 = np.concatenate([w0, np.zeros(n)]) if short else w0
        r = minimize(obj, x0, method="SLSQP", bounds=[(0, max_pos)] * d, constraints=cons,
                     options={"maxiter": 800, "ftol": 1e-12})
        x = r.x
        w = to_w(x)
        ok = (abs(w.sum() - 1) < 1e-6 and np.abs(w).max() <= max_pos + 1e-6
              and (kind == "min_vol" or vol(x) <= vol_cap + 1e-6)
              and (not short or np.abs(w).sum() <= gross + 1e-6)
              and (min_return is None or ret(x) >= min_return - 1e-7)
              and all(m @ np.abs(w) <= lim + 1e-6 for m, lim in groups))
        if ok and r.fun < best_val:
            best, best_val = w, r.fun
    if best is None:
        return None, float("nan")
    x = np.concatenate([np.clip(best, 0, None), np.clip(-best, 0, None)]) if short else best
    val = {"max_return": ret(x), "min_vol": vol(x), "max_sharpe": (ret(x) - rf) / vol(x)}[kind]
    return best, val


# ----------------------------------------------------------------------------- backtest

def portfolio_wealth(month_end: pd.DataFrame, w: np.ndarray, W0: float, C: float,
                     every: int | None, threshold: float | None) -> tuple[float, float]:
    """(ending wealth lump sum, ending wealth with contributions) for a rebalanced portfolio."""
    rets = month_end.pct_change().iloc[1:].to_numpy()
    ends = []
    for c in (0.0, C):
        hold = W0 * w.astype(float)
        for m, r in enumerate(rets, start=1):
            hold = hold * (1 + r) + c * w
            tot = hold.sum()
            if tot <= 0:
                hold = hold * 0
                break
            if (every is not None and m % every == 0) or (
                    threshold is not None and np.abs(hold / tot - w).max() > threshold):
                hold = tot * w
        ends.append(float(hold.sum()))
    return ends[0], ends[1]
