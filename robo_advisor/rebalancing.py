"""Rebalancing rules (spec §17), shared by the Monte Carlo engine and the historical backtest."""
from __future__ import annotations

import numpy as np

from .config import RebalancingCfg

_EVERY = {"monthly": 1, "quarterly": 3, "annual": 12}


def describe(cfg: RebalancingCfg) -> str:
    if cfg.type == "calendar":
        return f"Calendar-based: rebalance {cfg.frequency} to target weights"
    return f"Threshold-based: rebalance when any ETF drifts more than {cfg.threshold:.0%} from target"


def due(cfg: RebalancingCfg, month: int, current_w: np.ndarray, target_w: np.ndarray) -> np.ndarray | bool:
    """Whether to rebalance at the end of ``month`` (1-based). For threshold rules the answer is
    per path (current_w may be 2-D: paths x assets)."""
    if cfg.type == "calendar":
        return month % _EVERY[cfg.frequency] == 0
    drift = np.abs(current_w - target_w)
    return drift.max(axis=-1) > cfg.threshold
