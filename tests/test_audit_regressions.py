"""Regression tests for findings of the independent code audit."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from conftest import load_client
from robo_advisor.agents.advisory import build_advisory_graph
from robo_advisor.config import RebalancingCfg, TaxCfg
from robo_advisor.data.providers import trading_calendar
from robo_advisor.graph.engine import GraphHalted
from robo_advisor.models import ResolvedConstraints, TaxInput
from robo_advisor.optimization.methods import Optimizer
from robo_advisor.review import Reviewer
from robo_advisor.simulation import MCModel, simulate
from robo_advisor.tax import resolve_rates


def run(settings, provider, name="client_target.json", goal=None, **upd):
    c = load_client(name, **upd)
    if goal:
        c = c.model_copy(update={"goal": c.goal.model_copy(update=goal)})
    return build_advisory_graph(settings, provider).run({"client": c})


def all_ok(res):
    return all(r.ok for r in res.latest_reviews()) and not any(r.warnings for r in res.latest_reviews())


EDGE_CASES = {
    "zero_initial": ("client_target.json", {"initial_investment": 0.0, "monthly_contribution": 4000,
                                            "target_amount": 450000}, {}),
    "zero_initial_no_target": ("client_no_target.json", {"initial_investment": 0.0}, {}),
    "two_month_horizon": ("client_no_target.json", {"horizon_years": 2 / 12}, {}),
    "zero_contribution_infeasible": ("client_target.json", {"monthly_contribution": 0.0}, {}),
    "target_below_w0": ("client_target.json", {"target_amount": 90000}, {}),
    "single_etf": ("client_no_target.json", {}, {"universe": ["BND"], "constraints": {"max_position": 1.0},
                                                 "tolerance_answers": None}),
    "shorts_with_taxes": ("client_no_target.json", {}, {"constraints": {"allow_short": True}}),
    "cvar_goal": ("client_target.json", {}, {"preferences": {"goal_risk_metric": "cvar"}}),
    "threshold_taxed_target": ("client_target.json", {}, {"taxes": {"enabled": True},
                                                          "preferences": {"rebalancing_type": "threshold"}}),
    "target_return_method": ("client_no_target.json", {}, {"preferences": {"optimization_method": "target_return",
                                                                           "target_return": 0.06}}),
}


@pytest.mark.parametrize("case", list(EDGE_CASES))
def test_edge_cases_complete_and_pass_review(settings, provider, case):
    name, goal, upd = EDGE_CASES[case]
    upd = {k: v for k, v in upd.items() if v is not None}
    res = run(settings, provider, name, goal, **upd)
    assert all_ok(res), [f.message for r in res.latest_reviews() for f in r.findings if not f.passed]
    b = res.state["benchmark"].metrics
    assert np.isfinite(b.loc["Annualized return"].astype(float)).all()


def test_bootstrap_distribution(provider):
    from conftest import FAST
    from robo_advisor.config import load_settings
    s = load_settings(overrides=FAST | {"simulation": {"n_paths": 2000, "distribution": "bootstrap"}})
    res = run(s, provider, "client_no_target.json")
    assert all_ok(res) and res.state["simulation"].distribution == "bootstrap"


def test_short_positions_get_no_tax_refund():
    # flat market: the shorted asset yields 6% but has zero total return
    model = MCModel.from_moments(np.zeros(2), np.zeros((2, 2)), np.array([0.0, 0.06]))
    rates = resolve_rates(["VOO", "BND"], TaxInput(enabled=True, ordinary_rate=0.37), TaxCfg())
    raw = simulate(np.array([1.5, -0.5]), model, 100000, 0, 24, RebalancingCfg(), rates, 10, 1, record=False)
    assert (raw.taxes_paid >= -1e-9).all() and (raw.terminal <= 100000 + 1e-6).all()


def test_short_leg_uses_pretax_return():
    mu_after, mu_pre = np.array([0.08, 0.02]), np.array([0.09, 0.05])
    cov = np.diag([0.04, 0.01])
    rc = ResolvedConstraints(["A", "B"], True, 1.5, 2.0, 0.5)
    o = Optimizer(mu_after, cov, rc, 0.0, mu_short=mu_pre)
    assert o.port_return(np.array([1.5, -0.5])) == pytest.approx(1.5 * 0.08 - 0.5 * 0.05)


def test_validation_findings_surface_before_estimation(settings, provider):
    with pytest.raises(GraphHalted) as e:
        run(settings, provider, universe=["VOO", "SGOV", "BND"], as_of="2019-06-28", goal={"target_date": dt.date(2030, 1, 1)})
    assert e.value.node == "review_data_gate" and "SGOV" in str(e.value)


def test_nyse_calendar_excludes_good_friday():
    cal = trading_calendar(dt.date(2024, 3, 25), dt.date(2024, 4, 5))
    assert dt.datetime(2024, 3, 29) not in cal and dt.datetime(2024, 3, 28) in cal
    assert dt.datetime(2023, 10, 9) in trading_calendar(dt.date(2023, 10, 6), dt.date(2023, 10, 10))  # Columbus Day


# ---------------------------------------------------------------- reviewer catches injected errors

def blockers(rep):
    return {f.rule_id for f in rep.blocking}


def test_reviewer_blocks_suboptimal_case_b(settings, no_target_run):
    st = dict(no_target_run.state)
    p, est, tax = st["portfolio"], st["estimates"], st["tax"]
    w = Optimizer(tax.mu_after_tax, est.cov, p.constraints, est.risk_free).min_vol_portfolio()
    ret = float(w @ tax.mu_after_tax)
    st["portfolio"] = dataclasses.replace(
        p, weights=w, expected_return=ret, expected_return_pretax=float(w @ est.mu),
        volatility=float(np.sqrt(w @ est.cov @ w)),
        initial_allocation={t: x * st["request"].W0 for t, x in zip(p.tickers, w)},
        monthly_allocation={t: x * st["request"].C for t, x in zip(p.tickers, w)})
    assert "R-PORT-09" in blockers(Reviewer(settings).review_portfolio(st))


def test_reviewer_blocks_goal_portfolio_that_is_not_minimal_risk(settings, target_run):
    st = dict(target_run.state)
    p, est = st["portfolio"], st["estimates"]
    w = Optimizer(est.mu, est.cov, p.constraints, est.risk_free).mean_variance().weights   # max return at cap
    st["portfolio"] = dataclasses.replace(
        p, weights=w, expected_return=float(w @ est.mu), expected_return_pretax=float(w @ est.mu),
        volatility=float(np.sqrt(w @ est.cov @ w)),
        initial_allocation={t: x * st["request"].W0 for t, x in zip(p.tickers, w)},
        monthly_allocation={t: x * st["request"].C for t, x in zip(p.tickers, w)})
    assert "R-PORT-13" in blockers(Reviewer(settings).review_portfolio(st))


def test_reviewer_blocks_wrong_portfolio_backtest(settings, target_run):
    st = dict(target_run.state)
    b = st["benchmark"]
    m = b.metrics.copy()
    m.loc["Ending wealth (with monthly contributions)", "Portfolio"] *= 1.3
    st["benchmark"] = dataclasses.replace(b, metrics=m)
    assert "R-BM-05" in blockers(Reviewer(settings).review_final(st))


def test_reviewer_blocks_missing_taxes(settings, no_target_run):
    st = dict(no_target_run.state)
    st["simulation"] = dataclasses.replace(st["simulation"], taxes_paid_median=0.0)
    assert "R-TAX-02" in blockers(Reviewer(settings).review_final(st))


def test_remediated_review_counts_as_passed():
    from robo_advisor.graph.engine import RunResult
    from robo_advisor.models import ReviewFinding, ReviewReport
    bad = ReviewReport("portfolio", [ReviewFinding("R", "§", "BLOCKER", False, "x")])
    good = ReviewReport("portfolio", [ReviewFinding("R", "§", "BLOCKER", True, "x")])
    r = RunResult({}, [], [bad, good])
    assert r.latest_reviews() == [good] and r.superseded_reviews() == [bad]
