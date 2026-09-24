import json

import pytest

from conftest import FIXTURES, load_client
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
                   "optimization: {goal: {search_paths: 500, frontier_points: 8}}\nreview: {mc_paths: 1500}\n",
                   encoding="utf-8")
    rc = main(["run", "--profile", str(FIXTURES / "client_target.json"), "--out", str(tmp_path), "--config", str(cfg),
               "--provider", "synthetic"])
    assert rc == 0
    assert (tmp_path / "alex_target_report.html").exists()
    audit = json.loads((tmp_path / "alex_target_audit.json").read_text(encoding="utf-8"))
    assert all(r["ok"] for r in audit["reviews"])
    assert main(["graph"]) == 0 and "review_portfolio_gate" in capsys.readouterr().out


def test_cli_never_relies_on_platform_default_encoding(tmp_path):
    """Windows defaults to cp1252 for files; every read/write must state UTF-8 explicitly.
    EncodingWarning (python -X warn_default_encoding) flags any call that doesn't."""
    import subprocess
    import sys

    cfg = tmp_path / "fast.yaml"
    cfg.write_text("simulation: {n_paths: 800}\nscenarios: {n_paths: 300}\n"
                   "optimization: {goal: {search_paths: 300, frontier_points: 6}}\nreview: {mc_paths: 800}\n",
                   encoding="utf-8")
    strict = [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-m", "robo_advisor"]
    out = tmp_path / "out"
    run = subprocess.run(strict + ["run", "--provider", "synthetic", "--profile", str(FIXTURES / "client_target.json"), "--out", str(out),
                                   "--config", str(cfg)], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr[-2000:]
    html = (out / "alex_target_report.html").read_text(encoding="utf-8")
    assert "⚠" in html                                   # the non-ASCII banner that crashed on Windows
    mon = subprocess.run(strict + ["monitor", "--provider", "synthetic", "--prior", str(out / "alex_target_audit.json"), "--config", str(cfg)],
                         capture_output=True, text=True, encoding="utf-8")
    assert mon.returncode in (0, 1), mon.stderr[-2000:]      # 1 = review triggers found


def test_profiles_saved_with_bom_are_accepted(tmp_path):
    from robo_advisor.cli import _read
    p = tmp_path / "profile.json"
    with open(str(FIXTURES / "client_target.json"), "rb") as fh:
        p.write_bytes(b"\xef\xbb\xbf" + fh.read())            # Notepad-style UTF-8 with BOM
    assert _read(p).startswith("{")


def test_readme_profile_skeleton_matches_the_app():
    """The README's Route B skeleton must stay in sync with the questionnaire and the model."""
    import re

    from pydantic import ValidationError

    from conftest import ROOT
    from robo_advisor.config import load_settings
    from robo_advisor.models import ClientInput

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skeleton = json.loads(re.search(r"```json\n(.*?)```", readme, re.S).group(1))
    s = load_settings()
    assert set(skeleton["capacity_answers"]) == {q.id for q in s.questionnaire.capacity if not q.derive_from_goal}
    assert set(skeleton["tolerance_answers"]) == {q.id for q in s.questionnaire.tolerance}
    assert set(skeleton) <= set(ClientInput.model_fields)
    with pytest.raises(ValidationError, match="target_amount"):   # unfilled placeholders are rejected
        ClientInput.model_validate(skeleton)
