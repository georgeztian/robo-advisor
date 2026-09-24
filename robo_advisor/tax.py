"""Estimated tax model (spec §5). NOT individualized tax advice.

Two uses:
1. ``after_tax_returns``: the after-tax monthly return series the optimizer uses when taxes
   are enabled: income taxed each month at the ETF's income rate (interest / non-qualified
   at ordinary rates, qualified dividends at the qualified rate), plus an estimated drag from
   realized gains (assumed turnover x positive price return x capital-gains rate).
2. ``TaxRates``: per-asset rates consumed by the Monte Carlo tax engine (simulation.py), which
   tracks cost basis, realized short/long-term gains from rebalancing, loss netting,
   carry-forward and the annual ordinary-income offset for net capital losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TaxCfg
from .models import Estimates, TaxInput
from .universe import CATALOG

DISCLAIMER = ("Taxes are an ESTIMATED model based on the rates you supplied and simplified "
              "assumptions (taxable account, average-cost basis, annual settlement). Actual "
              "taxation depends on account type and individual circumstances; this is not "
              "individualized tax advice.")


@dataclass
class TaxRates:
    enabled: bool
    income: np.ndarray        # per-asset tax rate on distributions
    lt: np.ndarray            # per-asset long-term capital-gains rate (collectibles for GLD)
    st: float                 # short-term capital-gains rate
    ordinary: float
    loss_offset: float        # annual ordinary-income offset for net capital losses ($)
    liquidate_at_horizon: bool
    turnover: float

    @staticmethod
    def disabled(n: int) -> "TaxRates":
        z = np.zeros(n)
        return TaxRates(False, z, z, 0.0, 0.0, 0.0, False, 0.0)


def resolve_rates(tickers: list[str], tax_in: TaxInput, cfg: TaxCfg) -> TaxRates:
    if not tax_in.enabled:
        return TaxRates.disabled(len(tickers))

    def pick(v, d):
        return d if v is None else v

    state = pick(tax_in.state_rate, cfg.state_rate)
    ordinary = pick(tax_in.ordinary_rate, cfg.ordinary_rate) + state
    qual = pick(tax_in.qualified_dividend_rate, cfg.qualified_dividend_rate) + state
    ltcg = pick(tax_in.ltcg_rate, cfg.ltcg_rate) + state
    stcg = pick(tax_in.stcg_rate, cfg.stcg_rate) + state
    collect = min(cfg.collectibles_rate, ordinary - state) + state
    inc, lt = [], []
    for t in tickers:
        info = CATALOG[t]
        if info.income_type == "interest":
            inc.append(ordinary)
        elif info.income_type == "none":
            inc.append(0.0)
        else:
            q = info.qualified_fraction
            inc.append(q * qual + (1 - q) * ordinary)
        lt.append(collect if info.collectible else ltcg)
    return TaxRates(True, np.array(inc), np.array(lt), stcg, ordinary,
                    cfg.capital_loss_ordinary_offset,
                    pick(tax_in.liquidate_at_horizon, cfg.liquidate_at_horizon), cfg.assumed_turnover)


def after_tax_returns(est: Estimates, rates: TaxRates):
    """Return (monthly after-tax return DataFrame, annualized after-tax mu)."""
    if not rates.enabled:
        return est.monthly_returns, est.mu.copy()
    inc = est.monthly_income.fillna(0.0).to_numpy()
    r = est.monthly_returns.to_numpy()
    price_ret_annual = est.mu - est.income_yield
    drag = rates.turnover * np.clip(price_ret_annual, 0, None) * rates.lt / 12
    after = r - inc * rates.income - drag
    df = est.monthly_returns.copy()
    df.loc[:, :] = np.where(np.isnan(r), np.nan, after)
    # annual tax drag measured on the series, applied to the (possibly shrunk) pre-tax mu
    tax_drag = (est.monthly_returns - df).mean().to_numpy() * 12
    return df, est.mu - tax_drag
