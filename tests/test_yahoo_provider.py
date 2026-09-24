"""The Yahoo path end to end, offline: real provider code, fake yfinance serving Yahoo-format data."""
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

import fake_yfinance
from conftest import AS_OF, load_client
from robo_advisor.agents.advisory import build_advisory_graph
from robo_advisor.data.providers import (DataUnavailableError, SyntheticProvider, YahooProvider,
                                         fetch_treasury_rate, normalize_yahoo)
from robo_advisor.data.validation import validate

WS = dt.date(2006, 9, 23)


@pytest.fixture
def yahoo(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    fake_yfinance.CALLS.clear()
    fake_yfinance.SPLIT_UNADJUSTED.clear()
    fake_yfinance.REPAIR_LIBS_MISSING = False
    return YahooProvider(tmp_path / "cache", backoff=0.0)


def test_normalize_recovers_raw_prices_splits_and_distributions():
    syn = SyntheticProvider(inject_gaps=False)._make("TQQQ")          # has several 2:1 splits
    got = normalize_yahoo(fake_yfinance.yahoo_frame("TQQQ"), "TQQQ")
    syn = syn.loc[got.index]
    assert (got["split_ratio"] > 1).sum() == (syn["split_ratio"] > 1).sum() >= 2
    assert np.allclose(got["close"], syn["close"], rtol=1e-6)
    assert np.allclose(got["dividend"], syn["dividend"], rtol=1e-6, atol=1e-9)
    assert got.index.tz is None


def test_normalize_handles_yahoo_quirks():
    vwo = normalize_yahoo(fake_yfinance.yahoo_frame("VWO"), "VWO")
    assert vwo["close"].notna().all() and vwo.index.dayofweek.max() < 5     # Saturday row folded in
    assert (vwo["dividend"] > 0).sum() == (fake_yfinance.yahoo_frame("VWO")["Dividends"] > 0).sum()
    voo = normalize_yahoo(fake_yfinance.yahoo_frame("VOO"), "VOO")
    assert not voo.index.has_duplicates
    schd = normalize_yahoo(fake_yfinance.yahoo_frame("SCHD"), "SCHD")
    cg = fake_yfinance.yahoo_frame("SCHD")["Capital Gains"]
    assert schd["dividend"].sum() == pytest.approx(
        (fake_yfinance.yahoo_frame("SCHD")["Dividends"] + cg).sum(), rel=1e-9)
    with pytest.raises(DataUnavailableError):
        normalize_yahoo(pd.DataFrame(), "XYZ")


def test_yahoo_format_passes_validation_with_isolated_glitch_tolerated(yahoo):
    frames = yahoo.fetch(["VOO", "BND", "VWO", "SCHD", "TQQQ"], WS, AS_OF)
    dq = validate(frames, WS, AS_OF, __import__("robo_advisor.config", fromlist=["x"]).load_settings().data)
    assert not dq.blocking, dq.blocking
    assert any(w.startswith("BND:") and "tolerated" in w for w in dq.warnings)


def test_missed_split_adjustment_still_blocks(yahoo, settings):
    fake_yfinance.SPLIT_UNADJUSTED.add("TQQQ")
    dq = validate(yahoo.fetch(["TQQQ"], WS, AS_OF), WS, AS_OF, settings.data)
    assert any(b.startswith("TQQQ:") for b in dq.blocking)


def test_retry_cache_and_refresh(yahoo):
    yahoo.fetch(["QQQ"], WS, AS_OF)                    # first request is rate limited, then retried
    assert fake_yfinance.CALLS["QQQ"] == 2
    yahoo.fetch(["QQQ"], WS, AS_OF)                    # served from cache
    assert fake_yfinance.CALLS["QQQ"] == 2
    path = yahoo.cache / "QQQ.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.iloc[:-30].to_csv(path)                         # cache now ends ~6 weeks early
    old = time.time() - 24 * 3600
    os.utime(path, (old, old))
    yahoo.fetch(["QQQ"], WS, AS_OF)
    assert fake_yfinance.CALLS["QQQ"] == 3             # stale cache refreshed


def test_download_failure_is_reported_cleanly(monkeypatch, tmp_path):
    class Broken:
        class Ticker:
            def __init__(self, s):
                pass

            def history(self, **kw):
                raise ConnectionError("CONNECT tunnel failed, response 403")

    monkeypatch.setitem(sys.modules, "yfinance", Broken)
    with pytest.raises(DataUnavailableError, match="after 2 attempts"):
        YahooProvider(tmp_path, retries=2, backoff=0.0).fetch(["VOO"], WS, AS_OF)


def test_full_workflow_on_yahoo_format_data(yahoo, settings):
    res = build_advisory_graph(settings, yahoo).run({"client": load_client("client_target.json")})
    assert all(r.ok for r in res.latest_reviews())
    assert not res.state["market"].synthetic and res.state["market"].source == "yahoo"
    assert not any("SYNTHETIC" in d for d in res.state["explanation"].disclosures)
    assert res.state["simulation"].prob_target >= 0.8


def test_fred_treasury_rate_parsing_and_cache(tmp_path):
    csv = pd.DataFrame({"observation_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                        "DTB3": ["5.20", ".", "5.10"]})
    calls = []

    def reader(url):
        calls.append(url)
        return csv

    s = fetch_treasury_rate("DTB3", dt.date(2024, 1, 1), dt.date(2024, 1, 5), tmp_path, reader)
    assert list(s.round(4)) == [0.052, 0.051] and "id=DTB3" in calls[0]

    def fail(url):
        raise OSError("network down")

    with pytest.raises(DataUnavailableError):
        fetch_treasury_rate("DTB3", dt.date(2020, 1, 1), dt.date(2024, 1, 5), None, fail)


def test_missing_repair_libraries_fall_back_to_unrepaired_download(yahoo, settings):
    """Real-world failure: yfinance's repair imports scikit-learn, which is optional."""
    fake_yfinance.REPAIR_LIBS_MISSING = True
    frames = yahoo.fetch(["BND", "VOO"], WS, AS_OF)
    assert fake_yfinance.CALLS == {"BND": 2, "VOO": 1}           # one immediate fallback, no retry storm
    assert yahoo.repair is False and "sklearn" in yahoo.notes[0]
    assert not validate(frames, WS, AS_OF, settings.data).blocking


def test_non_transient_errors_are_not_retried(monkeypatch, tmp_path):
    class Bad:
        calls = 0

        class Ticker:
            def __init__(self, s):
                pass

            def history(self, **kw):
                Bad.calls += 1
                raise KeyError("Adj Close")

    monkeypatch.setitem(sys.modules, "yfinance", Bad)
    with pytest.raises(DataUnavailableError, match="unexpected Yahoo data"):
        YahooProvider(tmp_path, retries=4, backoff=0.0).fetch(["VOO"], WS, AS_OF)
    assert Bad.calls == 1
