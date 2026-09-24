"""The ETF universe: exactly the configured 25 ETFs in 9 categories, chosen by category."""
import dataclasses

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
