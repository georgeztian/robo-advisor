import numpy as np
import pytest

from robo_advisor.models import ResolvedConstraints
from robo_advisor.optimization.constraints import InfeasibleError
from robo_advisor.optimization.goal import goal_search
from robo_advisor.optimization.methods import METHODS, Optimizer
from robo_advisor.simulation import MCModel, simulate
from robo_advisor.tax import TaxRates

MU = np.array([0.09, 0.11, 0.035, 0.018, 0.07])
SIG = np.array([0.16, 0.22, 0.05, 0.005, 0.16])
CORR = np.array([[1, .85, .05, 0, .05], [.85, 1, 0, 0, .03], [.05, 0, 1, .1, .3], [0, 0, .1, 1, 0], [.05, .03, .3, 0, 1]])
COV = CORR * np.outer(SIG, SIG)
T = ["A", "B", "C", "D", "E"]
SCEN = np.random.default_rng(0).multivariate_normal(MU / 12, COV / 12, 240)


def rc(short=False, cap=0.12, maxpos=0.5, L=1.5):
    return ResolvedConstraints(T, short, maxpos, L if short else 1.0, cap)


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("method", [m for m in METHODS if m != "target_return"] + ["target_return"])
def test_every_method_respects_constraints(method, short):
    c = rc(short)
    o = Optimizer(MU, COV, c, 0.02, SCEN)
    w = o.run(method, target=0.06).weights
    assert w.sum() == pytest.approx(1, abs=1e-6)
    assert np.abs(w).max() <= c.max_position + 1e-6
    assert np.sqrt(w @ COV @ w) <= c.max_volatility + 1e-6
    if short:
        assert np.abs(w).sum() <= c.max_gross_leverage + 1e-6
    else:
        assert w.min() >= -1e-9


def test_mean_variance_is_optimal_vs_random_portfolios():
    c = rc(cap=0.10)
    w = Optimizer(MU, COV, c, 0.02).run("mean_variance").weights
    assert np.sqrt(w @ COV @ w) == pytest.approx(0.10, abs=1e-5)          # risk budget binds
    S = np.random.default_rng(1).dirichlet(np.full(5, .4), 50000)
    S = S[(S <= .5).all(1) & (np.sqrt(np.einsum("ij,jk,ik->i", S, COV, S)) <= .10)]
    assert (S @ MU).max() <= MU @ w + 1e-4


def test_shorting_increases_attainable_return():
    lo = Optimizer(MU, COV, rc(False), 0.02).run("mean_variance").weights
    hi = Optimizer(MU, COV, rc(True), 0.02).run("mean_variance").weights
    assert MU @ hi >= MU @ lo - 1e-8 and hi.min() < -1e-4


def test_infeasible_inputs_raise():
    with pytest.raises(InfeasibleError, match="cannot sum"):
        Optimizer(MU[:2], COV[:2, :2], ResolvedConstraints(T[:2], False, 0.4, 1.0, 0.2), 0.02)
    with pytest.raises(InfeasibleError, match="lowest-volatility"):
        Optimizer(MU[:2], COV[:2, :2], ResolvedConstraints(T[:2], False, 1.0, 1.0, 0.05), 0.02).run("mean_variance")


def _p(w, model, target, T_, seed=99, n=4000):
    from robo_advisor.config import RebalancingCfg
    raw = simulate(w, model, 50000, 1000, T_, RebalancingCfg(), TaxRates.disabled(5), n, seed, record=False)
    return (raw.terminal >= target).mean()


def test_goal_search_meets_probability_with_lowest_risk():
    from robo_advisor.config import RebalancingCfg
    o = Optimizer(MU, COV, rc(cap=0.17), 0.02)
    model = MCModel.from_moments(MU, COV, np.zeros(5))
    res = goal_search(o, model, 50000, 1000, 120, 200000, 0.8, RebalancingCfg(), TaxRates.disabled(5),
                      1000, 5, 10, 0.01, verify_paths=4000, verify_seed=99)
    w = res.weights
    assert not res.diagnostics["infeasible"]
    assert _p(w, model, 200000, 120) >= 0.8
    # any lower-volatility frontier point misses the probability requirement
    lower = [e for e in res.diagnostics["frontier"] if e["vol"] < np.sqrt(w @ COV @ w) - 1e-3]
    assert all(e["prob"] < 0.81 for e in lower)


def test_goal_search_reports_unreachable_target():
    from robo_advisor.config import RebalancingCfg
    o = Optimizer(MU, COV, rc(cap=0.05), 0.02)
    res = goal_search(o, MCModel.from_moments(MU, COV, np.zeros(5)), 50000, 1000, 60, 500000, 0.8,
                      RebalancingCfg(), TaxRates.disabled(5), 500, 5, 6, 0.01)
    assert res.diagnostics["infeasible"] and res.diagnostics["required_monthly_contribution"] > 1000
    assert any("No portfolio" in n for n in res.notes)


def cats_rc(short=False, limits=((("A", "B"), 0.30), (("E",), 0.10))):
    from robo_advisor.models import CategoryCap
    caps = [CategoryCap(f"cat{i}", lim, list(ts)) for i, (ts, lim) in enumerate(limits)]
    return ResolvedConstraints(T, short, 0.5, 1.5 if short else 1.0, 0.12, caps)


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("method", [m for m in METHODS if m != "target_return"] + ["target_return"])
def test_every_method_respects_category_limits(method, short):
    o = Optimizer(MU, COV, cats_rc(short), 0.02, SCEN)
    w = o.run(method, target=0.05).weights
    assert abs(w[0]) + abs(w[1]) <= 0.30 + 1e-6          # category of A, B (the two equity-like assets)
    assert abs(w[4]) <= 0.10 + 1e-6
    assert w.sum() == pytest.approx(1, abs=1e-6)


def test_category_limits_bind_and_infeasible_limits_are_explained():
    free = Optimizer(MU, COV, rc(cap=0.12), 0.02).run("mean_variance").weights
    capped = Optimizer(MU, COV, cats_rc(), 0.02).run("mean_variance").weights
    assert free[0] + free[1] > 0.30 and capped[0] + capped[1] == pytest.approx(0.30, abs=1e-5)
    with pytest.raises(InfeasibleError, match="category limits"):
        Optimizer(MU, COV, cats_rc(limits=((("A", "B", "C"), 0.2), (("D", "E"), 0.2))), 0.02)
