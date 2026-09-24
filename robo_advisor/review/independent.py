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
             shrink: float) -> tuple[np.ndarray, np.ndarray]:
    mus, sigmas = [], []
    for t in tickers:
        idx = tr_index_from_raw(frames[t])
        daily = idx.pct_change().dropna()
        sigmas.append(daily.std() * np.sqrt(trading_days))
        month_end = idx.groupby([idx.index.year, idx.index.month]).last()
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


def sample_long_only(n: int, max_pos: float, k: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.dirichlet(np.full(n, 0.3), size=k)
    w = np.vstack([w, np.eye(n)[:, :] if max_pos >= 1 else np.empty((0, n))])
    return w[(w <= max_pos + 1e-12).all(axis=1)]


def is_psd(m: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.linalg.eigvalsh((m + m.T) / 2).min() >= -tol * max(1.0, np.abs(m).max()))
