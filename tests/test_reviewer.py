import dataclasses

import numpy as np
import pandas as pd

from robo_advisor.models import MarketData
from robo_advisor.review import Reviewer


def failed(rep):
    return {f.rule_id for f in rep.blocking}


def test_clean_run_passes_every_rule(target_run, no_target_run):
    for run in (target_run, no_target_run):
        assert [r.stage for r in run.reviews] == ["data", "inputs", "portfolio", "final"]
        assert all(r.ok for r in run.reviews) and not any(r.warnings for r in run.reviews)


def test_blocks_weights_not_summing_to_one(settings, no_target_run):
    st = dict(no_target_run.state)
    p = st["portfolio"]
    st["portfolio"] = dataclasses.replace(p, weights=p.weights * 1.05)
    assert {"R-PORT-01", "R-PORT-07"} <= failed(Reviewer(settings).review_portfolio(st))


def test_blocks_risk_limit_breach_and_short_positions(settings, target_run):
    st = dict(target_run.state)
    p, est = st["portfolio"], st["estimates"]
    w = np.zeros(len(p.tickers))
    w[p.tickers.index("QQQ")] = 0.5
    w[p.tickers.index("VNQ")] = 0.7
    w[p.tickers.index("BIL")] = -0.2
    st["portfolio"] = dataclasses.replace(p, weights=w, volatility=float(np.sqrt(w @ est.cov @ w)))
    assert {"R-PORT-02", "R-PORT-03", "R-PORT-06"} <= failed(Reviewer(settings).review_portfolio(st))


def test_blocks_look_ahead_data(settings, target_run):
    st = dict(target_run.state)
    m = st["market"]
    frames = dict(m.frames)
    df = frames["VOO"]
    extra = df.iloc[[-1]].copy()
    extra.index = [df.index[-1] + pd.Timedelta(days=5)]
    frames["VOO"] = pd.concat([df, extra])
    st["market"] = MarketData(frames, m.as_of, m.source, m.synthetic)
    assert "R-DATA-01" in failed(Reviewer(settings).review_data(st))


def test_blocks_wrong_risk_mapping(settings, target_run):
    st = dict(target_run.state)
    r = st["risk"]
    st["risk"] = dataclasses.replace(r, mapped_score=max(r.capacity_score, r.tolerance_score))
    assert {"R-RISK-03"} <= failed(Reviewer(settings).review_inputs(st))


def test_blocks_tampered_estimates(settings, target_run):
    st = dict(target_run.state)
    e = st["estimates"]
    st["estimates"] = dataclasses.replace(e, mu=e.mu + 0.01)
    assert "R-EST-01" in failed(Reviewer(settings).review_inputs(st))


def test_blocks_benchmark_with_different_contributions(settings, target_run):
    st = dict(target_run.state)
    b = st["benchmark"]
    m = b.metrics.copy()
    m.loc["Total contributed", "S&P 500"] += 12000
    m.loc["Ending wealth (with monthly contributions)", "S&P 500"] *= 1.01
    st["benchmark"] = dataclasses.replace(b, metrics=m)
    assert {"R-BM-01", "R-BM-03"} <= failed(Reviewer(settings).review_final(st))


def test_blocks_missing_disclosures_and_misreported_probability(settings, target_run):
    st = dict(target_run.state)
    ex, sim = st["explanation"], st["simulation"]
    st["explanation"] = dataclasses.replace(ex, disclosures=[d for d in ex.disclosures if "SYNTHETIC" not in d
                                                             and "historical estimates" not in d])
    st["simulation"] = dataclasses.replace(sim, prob_target=0.95)
    assert {"R-EXP-02", "R-EXP-05", "R-SIM-03"} <= failed(Reviewer(settings).review_final(st))
