"""Deterministic projection (spec §11):

    FV_T = W0 (1 + r)^T + C [ ((1 + r)^T - 1) / r ]

r is the monthly rate implied by the portfolio's expected annual return, T in months.
The primary projection is the Monte Carlo (simulation.py); this is the simplified path.
"""
from __future__ import annotations

from .models import Projection


def future_value(W0: float, C: float, r: float, T: int) -> float:
    if abs(r) < 1e-12:
        return W0 + C * T
    g = (1 + r) ** T
    return W0 * g + C * (g - 1) / r


def project(W0: float, C: float, annual_return: float, T: int, inflation: float) -> Projection:
    r = annual_return / 12             # arithmetic monthly rate consistent with mu/12 in the MC
    fv = future_value(W0, C, r, T)
    real = fv / (1 + inflation) ** (T / 12)
    return Projection(months=T, monthly_rate=r, fv_nominal=fv, fv_real=real, total_contributed=W0 + C * T)
