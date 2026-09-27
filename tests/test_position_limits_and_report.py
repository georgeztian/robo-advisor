"""Per-ETF position limits, the optimization write-up in the report, and questionnaire shape."""
import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import load_client
from robo_advisor.agents.advisory import build_advisory_graph
from robo_advisor.models import ClientInput
from robo_advisor.report.html import render
from robo_advisor.review import Reviewer

LIMITS = {"BND": {"min": 0.15, "max": 0.30}, "voo": {"max": 0.20}, "GLD": {"min": 0.05}}


def run_with(settings, provider, name, constraints, **prefs):
    d = load_client(name).model_dump()
    d["constraints"] |= constraints
    d["preferences"] |= prefs
    return build_advisory_graph(settings, provider).run({"client": ClientInput.model_validate(d)})


@pytest.fixture(scope="module")
def limited_run(settings, provider):
    return run_with(settings, provider, "client_target.json", {"position_limits": LIMITS})


def test_client_position_limits_are_applied_and_reviewed(limited_run):
    port = limited_run.state["portfolio"]
    w = port.weight_map()
    rc = port.constraints
    assert rc.position_min["BND"] == 0.15 and rc.position_max["BND"] == 0.30
    assert rc.position_max["VOO"] == 0.20 and rc.position_min["VOO"] == 0.0     # ticker keys are case-insensitive
    assert rc.position_min["QQQ"] == 0.0 and rc.position_max["QQQ"] == 0.5      # defaults 0 and 50%
    assert 0.15 - 1e-6 <= w["BND"] <= 0.30 + 1e-6 and w["VOO"] <= 0.20 + 1e-6 and w["GLD"] >= 0.05 - 1e-6
    assert all(r.ok for r in limited_run.latest_reviews())


def test_client_defaults_apply_to_every_etf(settings, provider):
    res = run_with(settings, provider, "client_no_target.json", {"min_position": 0.02, "max_position": 0.3})
    w = res.state["portfolio"].weights
    assert w.min() >= 0.02 - 1e-6 and w.max() <= 0.3 + 1e-6
    assert all(r.ok for r in res.latest_reviews())


def test_reviewer_blocks_weight_below_its_minimum(settings, limited_run):
    st = dict(limited_run.state)
    p = st["portfolio"]
    w = p.weights.copy()
    i, j = p.tickers.index("BND"), int(np.argmax(p.weights))
    w[j] += w[i] - 0.05
    w[i] = 0.05                                                  # BND below its 15% minimum
    st["portfolio"] = dataclasses.replace(
        p, weights=w, initial_allocation={t: x * st["request"].W0 for t, x in zip(p.tickers, w)},
        monthly_allocation={t: x * st["request"].C for t, x in zip(p.tickers, w)})
    rep = Reviewer(settings).review_portfolio(st)
    assert "R-PORT-03" in {f.rule_id for f in rep.blocking}


def test_reviewer_blocks_resolved_limits_that_ignore_the_profile(settings, limited_run):
    st = dict(limited_run.state)
    rc = st["constraints"]
    st["constraints"] = dataclasses.replace(rc, position_max={**rc.position_max, "BND": 0.5})
    assert "R-CON-01" in {f.rule_id for f in Reviewer(settings).review_inputs(st).blocking}


@pytest.mark.parametrize("constraints, match", [
    ({"position_limits": {"SPY": {"max": 0.2}}}, "not selected"),
    ({"min_position": 0.6, "max_position": 0.9}, "minimum positions add up"),
    ({"position_limits": {"BND": {"min": 0.4}}, "max_position": 0.3}, "above its maximum"),
])
def test_impossible_position_limits_are_input_problems(settings, provider, constraints, match):
    with pytest.raises(ValueError, match=match):
        run_with(settings, provider, "client_target.json", constraints)


def test_unknown_profile_keys_are_rejected():
    with pytest.raises(ValidationError, match="maximum"):
        load_client("client_target.json", constraints={"position_limits": {"BND": {"maximum": 0.3}}})
    with pytest.raises(ValidationError, match="max_postion"):
        load_client("client_target.json", constraints={"max_postion": 0.3})


# ---------------------------------------------------------------- optimization write-up

def _ids(res):
    return [c["id"] for c in res.state["explanation"].optimization["constraints"]]


def test_goal_report_writes_out_the_problem(limited_run):
    o = limited_run.state["explanation"].optimization
    assert _ids(limited_run) == ["budget", "positions", "no_short", "categories", "risk", "target"]
    assert "minimize" in o["objective"]["math"] and o["objective"]["plain"] and o["summary"]
    bnd = next(r for r in o["positions"] if r["ticker"] == "BND")
    assert (bnd["min"], bnd["max"]) == (0.15, 0.30)
    html = render(limited_run)
    for text in ("How the optimizer chose this allocation", "In plain English", "The optimization problem",
                 "Position limits for each ETF", "What the symbols mean"):
        assert text in html


@pytest.mark.parametrize("method, extra, objective", [
    ("mean_variance", {}, "maximize"), ("min_volatility", {}, "minimize"), ("max_sharpe", {}, "maximize"),
    ("cvar", {}, "CVaR"), ("target_return", {"target_return": 0.04}, "minimize"),
    ("risk_parity", {}, "RC"), ("max_diversification", {}, "maximize"),
])
def test_every_method_is_written_out(settings, provider, method, extra, objective):
    res = run_with(settings, provider, "client_no_target.json", {"allow_short": method in ("max_sharpe",)},
                   optimization_method=method, **extra)
    o = res.state["explanation"].optimization
    assert objective in o["objective"]["math"]
    ids = _ids(res)
    assert {"budget", "positions", "risk"} <= set(ids)
    assert ("gross" in ids) == (method == "max_sharpe") and ("target_return" in ids) == (method == "target_return")
    assert all(r.ok for r in res.latest_reviews())


def test_reviewer_blocks_a_missing_constraint_in_the_write_up(settings, limited_run):
    st = dict(limited_run.state)
    exp = st["explanation"]
    o = dict(exp.optimization, constraints=[c for c in exp.optimization["constraints"] if c["id"] != "positions"])
    st["explanation"] = dataclasses.replace(exp, optimization=o)
    assert "R-EXP-08" in {f.rule_id for f in Reviewer(settings).review_final(st).blocking}


# ---------------------------------------------------------------- questionnaire

def test_every_question_has_five_options_and_the_attitude_question_exists(settings):
    q = settings.questionnaire
    assert all(len(x.options) == 5 for x in q.capacity + q.tolerance)
    att = next(x for x in q.tolerance if x.id == "risk_attitude")
    assert att.label("avoid_losses").startswith("I strongly prefer avoiding losses")
    assert list(att.options.values()) == sorted(att.options.values())
    assert "held_but_nervous" in next(x for x in q.tolerance if x.id == "crash_behavior").options
