import datetime as dt

import numpy as np
import pytest

from conftest import AS_OF
from robo_advisor.benchmark import compare
from robo_advisor.config import RebalancingCfg, TaxCfg
from robo_advisor.models import TaxInput
from robo_advisor.projection import future_value, project
from robo_advisor.simulation import MCModel, simulate, summarize
from robo_advisor.tax import TaxRates, resolve_rates

MU = np.array([0.08, 0.03])
COV = np.array([[0.16**2, 0.0], [0.0, 0.05**2]])


def test_fv_formula_matches_recursion():
    v = 10000.0
    for _ in range(240):
        v = v * 1.005 + 500
    assert future_value(10000, 500, 0.005, 240) == pytest.approx(v, rel=1e-12)
    assert future_value(10000, 500, 0.0, 12) == 16000
    p = project(10000, 500, 0.06, 240, 0.025)
    assert p.fv_real == pytest.approx(p.fv_nominal / 1.025**20)


def test_zero_volatility_mc_equals_deterministic_fv():
    model = MCModel.from_moments(MU, np.zeros((2, 2)), np.zeros(2))
    w = np.array([0.6, 0.4])
    raw = simulate(w, model, 10000, 500, 120, RebalancingCfg(frequency="monthly"), TaxRates.disabled(2), 50, 1)
    r = (w @ MU) / 12
    assert np.allclose(raw.terminal, future_value(10000, 500, r, 120), rtol=1e-6)


def test_mc_summary_and_moments():
    model = MCModel.from_moments(MU, COV, np.zeros(2))
    draws = np.vstack([model.draw(np.random.default_rng(i), 20000) for i in range(3)])
    assert draws.mean(0) == pytest.approx(MU / 12, abs=5e-4)
    assert np.cov(draws.T) == pytest.approx(COV / 12, abs=2e-4)
    raw = simulate(np.array([.7, .3]), model, 100000, 1000, 120, RebalancingCfg(), TaxRates.disabled(2), 5000, 3)
    s = summarize(raw, 120, 100000, 1000, 250000, [10, 25, 50, 75, 90], 3, "lognormal")
    v = list(s.percentiles.values())
    assert v == sorted(v) and s.band.shape == (121, 6)
    assert s.prob_target == pytest.approx((raw.terminal >= 250000).mean())
    assert s.prob_loss_principal == pytest.approx((raw.terminal < 220000).mean())
    assert -1 < s.max_drawdown_p95 <= s.max_drawdown_median <= 0


def test_taxes_reduce_wealth_and_threshold_rebalancing_runs():
    model = MCModel.from_moments(MU, COV, np.array([0.015, 0.03]))
    w = np.array([0.6, 0.4])
    rates = resolve_rates(["VOO", "BND"], TaxInput(enabled=True), TaxCfg(liquidate_at_horizon=True))
    for rb in (RebalancingCfg(), RebalancingCfg(type="threshold", threshold=0.05)):
        pre = simulate(w, model, 100000, 1000, 120, rb, TaxRates.disabled(2), 3000, 4, record=False)
        post = simulate(w, model, 100000, 1000, 120, rb, rates, 3000, 4, record=False)
        assert np.median(post.terminal) < np.median(pre.terminal)
        # liquidation can only add wealth through the capped ordinary-income offset of a net loss
        assert (post.terminal_after_liq <= post.terminal + rates.loss_offset * rates.ordinary + 1e-6).all()
        assert np.median(post.terminal_after_liq) < np.median(post.terminal)
        assert np.median(post.taxes_paid) > 0


def test_benchmark_identical_portfolio_matches_sp500(provider):
    fr = provider.fetch(["VOO", "BND", "BIL"], dt.date(2006, 9, 1), AS_OF)
    b = compare(fr, ["VOO", "BND"], np.array([1.0, 0.0]), 100000, 2000, AS_OF, 10, RebalancingCfg(), "VOO", "BIL", 0.02)
    m = b.metrics
    assert np.allclose(m["Portfolio"].to_numpy(float), m["S&P 500"].to_numpy(float))
    assert b.years == pytest.approx(10, abs=0.1) and not b.notes
    assert m.loc["Total contributed", "Portfolio"] == 100000 + 2000 * 120


def test_benchmark_flags_short_history(provider):
    fr = provider.fetch(["VOO", "SGOV", "BIL"], dt.date(2006, 9, 1), AS_OF)
    b = compare(fr, ["VOO", "SGOV"], np.array([.5, .5]), 1000, 10, AS_OF, 10, RebalancingCfg(), "VOO", "BIL", 0.02)
    assert b.years < 7 and "SGOV" in b.notes[0]
