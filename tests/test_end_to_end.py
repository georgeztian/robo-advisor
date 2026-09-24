import json

import pytest

from conftest import load_client
from robo_advisor.agents.advisory import build_advisory_graph
from robo_advisor.agents.monitor import build_monitoring_graph
from robo_advisor.cli import main
from robo_advisor.graph.engine import GraphHalted
from robo_advisor.report.html import audit_bundle, render


def test_target_client_workflow(target_run, settings):
    st = target_run.state
    req, port, sim = st["request"], st["portfolio"], st["simulation"]
    assert port.case == "target" and port.method == "goal_min_volatility"
    assert sim.prob_target >= req.target_probability            # verified on the reported sample
    assert port.volatility <= st["risk"].max_volatility
    assert set(port.tickers) == set(req.tickers)
    assert sim.n_paths == settings.simulation.n_paths
    html = render(target_run, settings, "flowchart TD")
    for s in ("Probability of reaching $500,000", "S&amp;P 500 comparison", "Independent review passed",
              "SYNTHETIC DATA", "Risk capacity", "Scenario analysis (not forecasts)"):
        assert s in html
    json.dumps(audit_bundle(target_run))


def test_no_target_taxed_client_workflow(no_target_run):
    st = no_target_run.state
    port, tax = st["portfolio"], st["tax"]
    assert port.case == "no_target" and port.method == "mean_variance"
    assert tax.mu_after_tax is not None and port.expected_return < port.expected_return_pretax
    assert port.volatility == pytest.approx(st["risk"].max_volatility, abs=1e-4)   # return maximized at the cap
    assert st["simulation"].prob_target is None and st["simulation"].taxes_paid_median > 0


def test_parallel_and_sequential_runs_agree(settings, provider):
    c = load_client("client_target.json", universe=["VOO", "BND", "BIL"])
    g = build_advisory_graph(settings, provider)
    a, b = g.run({"client": c}), g.run({"client": c}, parallel=False)
    assert (a.state["portfolio"].weights == b.state["portfolio"].weights).all()
    assert a.state["simulation"].percentiles == b.state["simulation"].percentiles


def test_unattainable_risk_limit_halts_with_explanation(settings, provider):
    c = load_client("client_target.json", universe=["QQQ", "TQQQ"],
                    tolerance_answers={k: v for k, v in load_client("client_target.json").tolerance_answers.items()}
                    | {"decline_reaction": "sell_at_10", "temporary_losses": "not_willing", "equity_comfort": "very_uncomfortable",
                       "stable_vs_volatile": "stable_low", "preserve_vs_growth": "preserve", "crash_behavior": "sold_everything"})
    with pytest.raises(Exception, match="lowest-volatility portfolio"):
        build_advisory_graph(settings, provider).run({"client": c})


def test_monitoring_triggers(tmp_path, target_run, settings, provider):
    prior = audit_bundle(target_run)
    g = build_monitoring_graph(settings, provider)
    same = g.run({"client": load_client("client_target.json"), "prior": prior}).state["monitoring"]
    assert not same.review_required
    changed = load_client("client_target.json")
    changed = changed.model_copy(update={"goal": changed.goal.model_copy(update={"monthly_contribution": 500})})
    rep = g.run({"client": changed, "prior": prior}).state["monitoring"]
    codes = {t.code for t in rep.triggers}
    assert {"CONTRIBUTION_CHANGED", "TARGET_AT_RISK"} <= codes


def test_cli_run_and_graph(tmp_path, capsys):
    cfg = tmp_path / "fast.yaml"
    cfg.write_text("simulation: {n_paths: 1500}\nscenarios: {n_paths: 500}\n"
                   "optimization: {goal: {search_paths: 500, frontier_points: 8}}\nreview: {mc_paths: 1500}\n")
    rc = main(["run", "--profile", "examples/client_target.json", "--out", str(tmp_path), "--config", str(cfg)])
    assert rc == 0
    assert (tmp_path / "alex_target_report.html").exists()
    audit = json.loads((tmp_path / "alex_target_audit.json").read_text())
    assert all(r["ok"] for r in audit["reviews"])
    assert main(["graph"]) == 0 and "review_portfolio_gate" in capsys.readouterr().out
