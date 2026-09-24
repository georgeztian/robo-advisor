import datetime as dt

import numpy as np
import pytest

from conftest import AS_OF
from robo_advisor.data.providers import CSVProvider, SyntheticProvider
from robo_advisor.data.validation import validate
from robo_advisor.estimation import estimate
from robo_advisor.models import TaxInput
from robo_advisor.tax import after_tax_returns, resolve_rates

WS = dt.date(2006, 9, 23)
TICKERS = ["VOO", "BND", "GLD", "SCHH", "VWO", "DGRO", "BIL"]


@pytest.fixture(scope="module")
def frames(provider):
    return provider.fetch(TICKERS, WS, AS_OF)


def test_synthetic_history_is_look_ahead_safe(provider):
    early = provider.fetch(["VOO"], WS, dt.date(2020, 1, 31))["VOO"]
    late = provider.fetch(["VOO"], WS, AS_OF)["VOO"]
    assert early.index[-1] <= dt.datetime(2020, 1, 31)
    assert np.allclose(early["adj_close"] / early["adj_close"].iloc[0],
                       late.loc[early.index, "adj_close"] / late["adj_close"].iloc[0])


def test_validation_flags_short_history_and_gaps(frames, settings):
    dq = validate(frames, WS, AS_OF, settings.data)
    assert not dq.blocking
    assert not dq.tickers["DGRO"].meets_min_history and dq.tickers["VWO"].meets_min_history
    assert any(w.startswith("DGRO:") and "history" in w for w in dq.warnings)
    assert dq.tickers["SCHH"].missing_days == 6 and dq.tickers["VWO"].max_gap_days == 2
    assert dq.tickers["VOO"].n_splits >= 0 and dq.tickers["BND"].n_distributions > 100


def test_validation_detects_bad_adjusted_prices_and_look_ahead(frames, settings):
    bad = {t: df.copy() for t, df in frames.items()}
    i = len(bad["VOO"]) // 2
    bad["VOO"].iloc[i:, bad["VOO"].columns.get_loc("adj_close")] *= 1.5       # missed split adjustment
    dq = validate(bad, WS, AS_OF, settings.data)
    assert any("VOO" in b and "adjusted" in b for b in dq.blocking)
    dq = validate(frames, WS, AS_OF - dt.timedelta(days=30), settings.data)
    assert any("look-ahead" in b for b in dq.blocking)


def test_csv_provider_roundtrip(tmp_path, frames):
    frames["BND"].rename_axis("date").to_csv(tmp_path / "BND.csv")
    got = CSVProvider(tmp_path).fetch(["BND"], WS, AS_OF)["BND"]
    assert np.allclose(got["adj_close"], frames["BND"]["adj_close"])


def test_estimates_annualization_and_psd(frames, settings):
    t = ["VOO", "BND", "GLD", "DGRO"]
    est = estimate(frames, t, AS_OF, WS, settings.data, settings.estimation)
    r = frames["BND"]["adj_close"].pct_change().dropna()
    assert est.sigma[1] == pytest.approx(r.std() * np.sqrt(252), rel=0.01)
    me = frames["BND"]["adj_close"].groupby(frames["BND"].index.to_period("M")).last().pct_change().dropna()
    assert est.mu[1] == pytest.approx(me.mean() * 12, abs=1e-12)
    assert np.linalg.eigvalsh(est.cov).min() >= -1e-12
    assert np.allclose(np.sqrt(np.diag(est.cov)), est.sigma)
    assert est.history_years["DGRO"] < 13 < est.history_years["VOO"]
    assert 0.005 < est.risk_free < 0.04
    for col in ("max_drawdown", "var_daily", "cvar_daily", "beta", "sharpe", "sortino", "downside_dev"):
        assert est.stats[col].notna().all()
    assert est.stats.loc["VOO", "beta"] == pytest.approx(1.0, abs=1e-9)


def test_estimation_refuses_look_ahead(frames, settings):
    with pytest.raises(ValueError, match="look-ahead"):
        estimate(frames, ["VOO"], AS_OF - dt.timedelta(days=10), WS, settings.data, settings.estimation)


def test_after_tax_returns_lower_and_character_aware(frames, settings):
    t = ["VOO", "BND", "GLD", "SCHH"]
    est = estimate(frames, t, AS_OF, WS, settings.data, settings.estimation)
    rates = resolve_rates(t, TaxInput(enabled=True, ordinary_rate=0.32, qualified_dividend_rate=0.15), settings.tax)
    _, mu_after = after_tax_returns(est, rates)
    assert (mu_after < est.mu).all()
    assert rates.income[1] == pytest.approx(0.32)            # bond interest at ordinary rate
    assert rates.income[2] == 0 and rates.lt[2] == pytest.approx(0.28)   # GLD collectible
    assert rates.income[0] < rates.income[3]                 # qualified VOO vs REIT (mostly ordinary)
    assert not resolve_rates(t, TaxInput(enabled=False), settings.tax).enabled
