"""Return of capital (ROC): not taxed when paid, lowers the cost basis, so it is taxed as a
capital gain when the shares are sold."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from conftest import AS_OF
from robo_advisor.config import RebalancingCfg, TaxCfg
from robo_advisor.estimation import estimate
from robo_advisor.models import TaxInput
from robo_advisor.report.html import render
from robo_advisor.review import Reviewer
from robo_advisor.simulation import MCModel, simulate
from robo_advisor.tax import TaxRates, after_tax_returns, etf_tax_rates, resolve_rates, tax_schedule
from robo_advisor.universe import CATALOG

TI = TaxInput(enabled=True, ordinary_rate=0.32, qualified_dividend_rate=0.15, ltcg_rate=0.15,
              stcg_rate=0.32, state_rate=0.05)
W0, MU, YIELD, YEARS = 100_000.0, 0.10, 0.12, 10


def test_roc_part_of_a_distribution_is_untaxed(settings):
    sched = tax_schedule(TI, TaxCfg())
    info = CATALOG["SPYI"]
    if_all_taxable = info.qualified_fraction * 0.20 + (1 - info.qualified_fraction) * 0.37
    row = etf_tax_rates("SPYI", sched)
    assert info.roc_fraction > 0.9 and row["roc"] == info.roc_fraction
    assert sum(row["income"]) == pytest.approx((1 - info.roc_fraction) * if_all_taxable)
    assert "return of capital" in row["treatment"]
    voo = etf_tax_rates("VOO", sched)                       # no ROC: unchanged
    assert voo["roc"] == 0 and "return of capital" not in voo["treatment"]
    rates = resolve_rates(["VOO", "SPYI"], TI, TaxCfg())
    assert rates.roc.tolist() == [0.0, info.roc_fraction]
    assert not resolve_rates(["SPYI"], TaxInput(enabled=False), TaxCfg()).roc.any()
    assert Reviewer(settings)._income_rate("SPYI", TI) == pytest.approx(rates.income[1])   # independent check agrees


def _one_fund(roc: float, liquidate: bool = True):
    """One fund, no volatility: 10 %/yr total return including a 12 % distribution yield
    (reinvested), so its price drifts down 2 %/yr like an option-income fund. Distributions'
    taxable part is taxed at 30 %; long-term gains at 20 %."""
    model = MCModel.from_moments(np.array([MU]), np.zeros((1, 1)), np.array([YIELD]))
    rates = TaxRates(True, np.array([0.30 * (1 - roc)]), np.array([0.20]), 0.37, 0.37, 3000.0,
                     liquidate, 0.10, np.array([roc]))
    return simulate(np.array([1.0]), model, W0, 0.0, 12 * YEARS, RebalancingCfg(), rates, 3, 1, record=False)


def test_full_roc_defers_all_tax_to_the_sale():
    growth = W0 * (1 + MU / 12) ** (12 * YEARS)
    raw = _one_fund(1.0)
    assert raw.terminal == pytest.approx(growth)                      # nothing taxed along the way
    # the basis is still the 100,000 invested: every reinvested ROC dollar lowered it as much
    assert raw.taxes_paid == pytest.approx(0.20 * (growth - W0))
    assert raw.terminal_after_liq == pytest.approx(growth - 0.20 * (growth - W0))
    assert _one_fund(1.0, liquidate=False).taxes_paid == pytest.approx(0.0)   # deferred, not exempt


def test_more_roc_means_less_tax_now_and_more_gain_at_the_sale():
    runs = {roc: (_one_fund(roc), _one_fund(roc, liquidate=False)) for roc in (0.0, 0.5, 0.93, 1.0)}
    held = [k[1].taxes_paid[0] for k in runs.values()]                # taxes paid while holding
    sale = [k[0].taxes_paid[0] - k[1].taxes_paid[0] for k in runs.values()]   # tax due on the sale
    after = [k[0].terminal_after_liq[0] for k in runs.values()]
    assert held == sorted(held, reverse=True) and held[0] > 0
    assert sale == sorted(sale)
    # deferral plus the lower long-term rate leave more after all taxes
    assert after == sorted(after) and after[0] < after[-1]


def test_after_tax_returns_count_roc_as_deferred_gain(provider, settings, monkeypatch):
    t = ["VOO", "SPYI"]
    ws = dt.date(2006, 9, 23)
    est = estimate(provider.fetch(t, ws, AS_OF), t, AS_OF, ws, settings.data, settings.estimation)
    rates = resolve_rates(t, TI, settings.tax)
    _, mu_after = after_tax_returns(est, rates)
    y, roc = est.income_yield[1], rates.roc[1]
    gain = est.mu[1] - y + roc * y                            # price change plus basis-lowering ROC
    expected = rates.income[1] * y + rates.turnover * max(gain, 0) * rates.lt[1]
    assert est.mu[1] - mu_after[1] == pytest.approx(expected, rel=1e-9)
    monkeypatch.setitem(CATALOG, "SPYI", dataclasses.replace(CATALOG["SPYI"], roc_fraction=0.0))
    _, mu_no_roc = after_tax_returns(est, resolve_rates(t, TI, settings.tax))
    assert mu_after[1] > mu_no_roc[1] + 0.01                   # ROC makes SPYI far cheaper in tax
    assert mu_after[0] == pytest.approx(mu_no_roc[0])          # other ETFs unaffected


def test_roc_in_the_full_workflow(no_target_run):
    st = no_target_run.state
    by = {r["ticker"]: r for r in st["tax"].by_etf}
    tickers = st["request"].tickers
    assert {"SPYI", "QQQI", "VNQ"} <= set(tickers)
    for i, tk in enumerate(tickers):
        assert st["tax"].rates.roc[i] == CATALOG[tk].roc_fraction
        assert sum(by[tk]["income"]) == pytest.approx(st["tax"].rates.income[i])
    assert all(r.ok for r in no_target_run.latest_reviews())      # incl. R-TAX-02 / R-TAX-03
    html = render(no_target_run)
    assert "93% return of capital" in html and "lowers your cost basis" in html


def test_reviewer_catches_roc_ignored_in_the_table(settings, no_target_run):
    st = dict(no_target_run.state)
    rows = [dict(r) for r in st["tax"].by_etf]
    i = next(k for k, r in enumerate(rows) if r["ticker"] == "SPYI")
    rows[i]["income"] = tuple(v / (1 - CATALOG["SPYI"].roc_fraction) for v in rows[i]["income"])
    st["tax"] = dataclasses.replace(st["tax"], by_etf=rows)
    assert "R-TAX-03" in {f.rule_id for f in Reviewer(settings).review_final(st).blocking}
