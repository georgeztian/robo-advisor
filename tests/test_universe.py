"""The ETF universe: exactly the configured 25 ETFs in 9 categories, chosen by category."""
import dataclasses

import numpy as np

import pytest

from robo_advisor.cli import choose_etfs
from robo_advisor.review import Reviewer
from robo_advisor.universe import CATALOG, RISK_NOTE_PREFIX, UniverseError, category_of, resolve_universe

EXPECTED = {
    "Equity ETFs": ["SPY", "VOO", "VTI", "QQQ", "TQQQ"],
    "Bond ETFs": ["BND", "TLT", "HYG"],
    "Risk-free Short-term Treasury ETFs": ["BIL", "SGOV"],
    "Commodity ETFs": ["GLD", "SLV"],
    "International Equity ETFs": ["VXUS", "IEFA", "VWO"],
    "Real Estate ETFs": ["VNQ", "SCHH"],
    "Dividend ETFs": ["SCHD", "VYM", "DGRO"],
    "Income ETFs": ["SPYI", "QQQI", "JEPQ", "JEPI"],
    "Crypto ETFs": ["IBIT"],
}


def test_configured_universe_is_the_agreed_list(settings):
    assert settings.universe.categories == EXPECTED
    assert len(settings.universe.tickers) == 25 == len(set(settings.universe.tickers))
    assert set(CATALOG) == set(settings.universe.tickers)          # catalog has no stale or missing ETFs


def test_catalog_facts_are_sane():
    for t, e in CATALOG.items():
        assert 0 < e.expense_ratio < 0.01 and 0 <= e.qualified_fraction <= 1
    assert CATALOG["GLD"].collectible and CATALOG["SLV"].collectible and not CATALOG["IBIT"].collectible
    assert {t for t, e in CATALOG.items() if e.risk_note} == {"TQQQ", "SPYI", "QQQI", "JEPQ", "JEPI", "IBIT"}


def test_resolve_by_category_and_ticker():
    got = resolve_universe(["bond etfs", "SPY", "IBIT", "HYG"], EXPECTED)
    assert got == ["SPY", "BND", "TLT", "HYG", "IBIT"]                  # configured order, no duplicates
    assert category_of("HYG", EXPECTED) == "Bond ETFs"
    with pytest.raises(UniverseError, match="Crypto ETFs"):
        resolve_universe(None, EXPECTED)                                  # the client must choose
    with pytest.raises(UniverseError, match="VT"):
        resolve_universe(["VT"], EXPECTED)                                # no longer offered


def test_interactive_selection_by_category(settings, monkeypatch, capsys):
    # Equity: 1,2 | Bond: all | Treasury: bad input then 2 | the rest skipped
    answers = iter(["1,2", "all", "9", "2", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert choose_etfs(settings) == ["SPY", "VOO", "BND", "TLT", "HYG", "SGOV"]
    assert "25 ETFs in 9 categories" in capsys.readouterr().out


def test_invalid_universe_config_rejected():
    from robo_advisor.config import load_settings
    with pytest.raises(ValueError, match="no entry in the ETF catalog"):
        load_settings(overrides={"universe": {"categories": {"Equity ETFs": ["SPY", "XYZ"]}}})
    with pytest.raises(ValueError, match="listed in both"):
        load_settings(overrides={"universe": {"categories": {"A": ["SPY"], "B": ["SPY"]}}})


def test_reviewer_blocks_missing_special_risk_disclosure(settings, no_target_run):
    st = dict(no_target_run.state)
    held = [t for t, w in st["portfolio"].weight_map().items() if w > 1e-6 and CATALOG[t].risk_note]
    assert held, "example no-target client should hold a special-risk ETF"
    ex = st["explanation"]
    st["explanation"] = dataclasses.replace(
        ex, disclosures=[d for d in ex.disclosures if not d.startswith(RISK_NOTE_PREFIX)])
    blockers = {f.rule_id for f in Reviewer(settings).review_final(st).blocking}
    assert "R-EXP-07" in blockers


def test_category_limits_from_config_and_client(settings, provider):
    from conftest import load_client
    from robo_advisor.agents.advisory import build_advisory_graph
    c = load_client("client_no_target.json")
    c = c.model_copy(update={"constraints": c.constraints.model_copy(
        update={"category_limits": {"Crypto ETFs": 0.02, "Equity ETFs": 0.40}})})
    res = build_advisory_graph(settings, provider).run({"client": c})
    caps = {cap.category: cap.limit for cap in res.state["constraints"].category_caps}
    assert caps == {"Equity ETFs": 0.40, "Income ETFs": 0.25, "Commodity ETFs": 0.20,
                    "Real Estate ETFs": 0.20, "Crypto ETFs": 0.02}  # config defaults + client overrides
    w = res.state["portfolio"].weight_map()
    assert w["IBIT"] <= 0.02 + 1e-6
    assert sum(w[t] for t in ("SPY", "VOO", "VTI", "QQQ", "TQQQ")) <= 0.40 + 1e-6
    assert all(r.ok for r in res.latest_reviews())


def test_reviewer_blocks_category_limit_breach(settings, no_target_run):
    st = dict(no_target_run.state)
    p = st["portfolio"]
    w = np.zeros(len(p.tickers))
    w[p.tickers.index("IBIT")], w[p.tickers.index("BND")] = 0.3, 0.7
    st["portfolio"] = dataclasses.replace(p, weights=w)
    assert "R-PORT-15" in {f.rule_id for f in Reviewer(settings).review_portfolio(st).blocking}


def test_unknown_category_limit_is_an_input_problem(settings, provider):
    from conftest import load_client
    from robo_advisor.agents.advisory import build_advisory_graph
    c = load_client("client_target.json")
    c = c.model_copy(update={"constraints": c.constraints.model_copy(update={"category_limits": {"Crypto": 0.1}})})
    with pytest.raises(ValueError, match="unknown categories"):
        build_advisory_graph(settings, provider).run({"client": c})


def test_optimizer_question_lists_every_method(settings, monkeypatch, capsys):
    from robo_advisor.cli import _ask_optimizer
    from robo_advisor.optimization.methods import METHODS
    answers = iter(["5", "0.06"])                          # 5 = target_return, then the return
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert _ask_optimizer(settings, has_target=False) == {"optimization_method": "target_return",
                                                           "target_return": 0.06}
    out = capsys.readouterr().out
    assert all(f"{i}." in out for i in range(1, len(METHODS) + 1))
    answers = iter(["90", "2"])                            # target client: probability, CVaR
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert _ask_optimizer(settings, has_target=True) == {"target_probability": 0.9, "goal_risk_metric": "cvar"}
