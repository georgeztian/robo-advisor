"""Estimated tax model (spec §5). NOT individualized tax advice.

Return of capital (ROC): the part of an ETF's distributions that is a return of the investor's
own money (``roc_fraction`` in the catalog). It is not taxed when paid; it lowers the cost basis
instead, so it is taxed later as a capital gain when the shares are sold. The rest of the
distribution is taxed by its income character as below.

Three uses:
1. ``after_tax_returns``: the after-tax monthly return series the optimizer uses when taxes
   are enabled: income taxed each month at the ETF's income rate (interest / non-qualified
   at ordinary rates, qualified dividends at the qualified rate; Treasury interest free of state
   tax, municipal interest free of federal tax; ROC untaxed), plus an estimated drag from
   realized gains (assumed turnover x positive (price return + ROC) x capital-gains rate: ROC
   lowers basis, so it adds to the gain realized on a sale).
2. ``TaxRates``: per-asset rates consumed by the Monte Carlo tax engine (simulation.py), which
   tracks cost basis (reinvested distributions add basis only for their taxable part),
   realized short/long-term gains from rebalancing, loss netting, carry-forward and the annual
   ordinary-income offset for net capital losses.
3. ``tax_schedule`` / ``etf_tax_rates``: the federal / state rates behind both, also shown in the
   report's per-ETF tax-rate table (``resolve_rates`` is built from them, so they always agree).
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
    income: np.ndarray        # per-asset tax rate on the whole distribution (ROC part untaxed)
    lt: np.ndarray            # per-asset long-term capital-gains rate (collectibles rate for GLD, SLV)
    st: float                 # short-term capital-gains rate
    ordinary: float
    loss_offset: float        # annual ordinary-income offset for net capital losses ($)
    liquidate_at_horizon: bool
    turnover: float
    roc: np.ndarray | None = None   # per-asset return-of-capital share of distributions (None -> 0)

    def __post_init__(self):
        if self.roc is None:
            self.roc = np.zeros(len(self.income))

    @staticmethod
    def disabled(n: int) -> TaxRates:
        z = np.zeros(n)
        return TaxRates(False, z, z, 0.0, 0.0, 0.0, False, 0.0, z)


SCHEDULE_LABELS = {"ordinary": "Ordinary income (interest, non-qualified dividends)",
                   "qualified": "Qualified dividends", "ltcg": "Long-term capital gains",
                   "stcg": "Short-term capital gains", "collectibles": "Long-term gains on collectibles (gold, silver)"}


def tax_schedule(tax_in: TaxInput, cfg: TaxCfg) -> dict[str, tuple[float, float]]:
    """(federal, state) rate for each kind of taxable income; all zero when taxes are off. One
    state rate applies to every kind; etf_tax_rates handles the exemptions (Treasury interest
    is state-exempt, municipal interest federal-exempt)."""
    if not tax_in.enabled:
        return {k: (0.0, 0.0) for k in SCHEDULE_LABELS}

    def pick(v, d):
        return d if v is None else v

    state = pick(tax_in.state_rate, cfg.state_rate)
    ordinary = pick(tax_in.ordinary_rate, cfg.ordinary_rate)
    return {"ordinary": (ordinary, state),
            "qualified": (pick(tax_in.qualified_dividend_rate, cfg.qualified_dividend_rate), state),
            "ltcg": (pick(tax_in.ltcg_rate, cfg.ltcg_rate), state),
            "stcg": (pick(tax_in.stcg_rate, cfg.stcg_rate), state),
            "collectibles": (min(cfg.collectibles_rate, ordinary), state)}   # capped at the ordinary rate


def etf_tax_rates(ticker: str, sched: dict[str, tuple[float, float]]) -> dict:
    """How one ETF's income is taxed: (federal, state) rates on its distributions (on the whole
    distribution, so a return-of-capital part lowers them), long-term and short-term gains, and
    a description of the distribution treatment."""
    info = CATALOG[ticker]
    (o_f, o_s), (q_f, q_s) = sched["ordinary"], sched["qualified"]
    q, roc = info.qualified_fraction, info.roc_fraction
    kind = info.income_type
    if kind == "interest":
        treatment, income = "Interest, taxed as ordinary income", (o_f, o_s)
    elif kind == "treasury":
        treatment, income = "US Treasury interest: federal tax only (state-exempt)", (o_f, 0.0)
    elif kind == "tax_exempt":
        treatment, income = "Municipal-bond interest: state tax only (federal-exempt)", (0.0, o_s)
    elif kind == "none":
        treatment, income = "No distributions", (0.0, 0.0)
    else:
        label = {"qualified": "Dividends", "reit": "REIT distributions"}.get(kind, "Distributions")
        treatment = f"{label}: {q:.0%} qualified, {1 - q:.0%} ordinary income"
        if roc > 0:
            treatment = (f"{label}: {roc:.0%} return of capital (not taxed when paid; lowers the cost "
                         f"basis), the rest {q:.0%} qualified, {1 - q:.0%} ordinary income")
        income = (q * q_f + (1 - q) * o_f, q * q_s + (1 - q) * o_s)
    if roc > 0 and kind in ("interest", "treasury", "tax_exempt"):
        treatment = (f"{roc:.0%} return of capital (not taxed when paid; lowers the cost basis); "
                     f"the rest: {treatment}")
    income = tuple((1 - roc) * r for r in income)
    return {"ticker": ticker, "treatment": treatment, "income": income,
            "lt": sched["collectibles" if info.collectible else "ltcg"], "st": sched["stcg"],
            "collectible": info.collectible, "roc": roc}


def resolve_rates(tickers: list[str], tax_in: TaxInput, cfg: TaxCfg) -> TaxRates:
    if not tax_in.enabled:
        return TaxRates.disabled(len(tickers))
    sched = tax_schedule(tax_in, cfg)
    by_etf = [etf_tax_rates(t, sched) for t in tickers]
    return TaxRates(True, np.array([sum(e["income"]) for e in by_etf]), np.array([sum(e["lt"]) for e in by_etf]),
                    sum(sched["stcg"]), sum(sched["ordinary"]), cfg.capital_loss_ordinary_offset,
                    tax_in.liquidate_at_horizon if tax_in.liquidate_at_horizon is not None
                    else cfg.liquidate_at_horizon, cfg.assumed_turnover,
                    np.array([e["roc"] for e in by_etf]))


def after_tax_returns(est: Estimates, rates: TaxRates):
    """Return (monthly after-tax return DataFrame, annualized after-tax mu)."""
    if not rates.enabled:
        return est.monthly_returns, est.mu.copy()
    inc = est.monthly_income.fillna(0.0).to_numpy()
    r = est.monthly_returns.to_numpy()
    # gains build up from price growth plus return of capital (which lowers basis instead of
    # being taxed when paid); the assumed turnover realizes part of them each year
    gain_annual = est.mu - est.income_yield + rates.roc * est.income_yield
    drag = rates.turnover * np.clip(gain_annual, 0, None) * rates.lt / 12
    after = r - inc * rates.income - drag
    df = est.monthly_returns.copy()
    df.loc[:, :] = np.where(np.isnan(r), np.nan, after)
    # annual tax drag measured on the series, applied to the (possibly shrunk) pre-tax mu
    tax_drag = (est.monthly_returns - df).mean().to_numpy() * 12
    return df, est.mu - tax_drag
