"""Client report (spec §15 dashboard + §19 step 11) and the JSON audit bundle."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..explain import REPORT_DISCLAIMER, method_label
from ..graph.engine import RunResult
from . import charts

_ENV = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))


def _money(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"${v:,.0f}"


def _bench_rows(m: pd.DataFrame) -> list[tuple[str, tuple[str, str]]]:
    fmt = {"Cumulative return": "pct", "Annualized return": "pct", "Annualized volatility": "pct",
           "Sharpe ratio": "num", "Maximum drawdown": "pct", "Best year": "pct", "Worst year": "pct",
           "Best year (calendar)": "int", "Worst year (calendar)": "int"}
    rows = []
    for k in m.index:
        f = fmt.get(k, "money")
        vals = []
        for c in m.columns:
            v = float(m.loc[k, c])
            vals.append(f"{v:.1%}" if f == "pct" else f"{v:.2f}" if f == "num" else f"{int(v)}" if f == "int" else _money(v))
        rows.append((k, tuple(vals)))
    return rows


def render(result: RunResult, settings=None, mermaid: str = "") -> str:
    st = result.state
    req, risk, port, sim = st["request"], st["risk"], st["portfolio"], st["simulation"]
    exp, bench, est, dq = st["explanation"], st["benchmark"], st["estimates"], st["data_quality"]
    s = settings
    table = exp.etf_rationale
    held = [(t, r) for t, r in table.iterrows() if abs(r["Weight"]) > 1e-6]
    unheld = [(t, r) for t, r in table.iterrows() if abs(r["Weight"]) <= 1e-6]
    held.sort(key=lambda x: -abs(x[1]["Weight"]))
    cap_by_cat = {c.category: c.limit for c in port.constraints.category_caps}
    category_rows = []
    for cat, grp in table[table["Weight"].abs() > 1e-6].groupby("Category", sort=False):
        category_rows.append({"name": cat, "tickers": ", ".join(grp.index), "weight": float(grp["Weight"].sum()),
                              "initial": float(grp["Initial"].sum()), "monthly": float(grp["Monthly"].sum()),
                              "limit": cap_by_cat.get(cat)})
    category_rows.sort(key=lambda c: -abs(c["weight"]))

    alloc = charts.bar_chart_h([t for t, _ in held], [r["Weight"] for _, r in held], charts.pct,
                               "ETF weights", notes=[f"{_money(r['Initial'])} initial, {_money(r['Monthly'])}/month"
                                                     for _, r in held])
    bands = [(b.max_score, b.profile) for b in (s.risk_bands if s else [])]
    risk_chart = charts.score_bars([("Risk capacity", risk.capacity_score), ("Risk tolerance", risk.tolerance_score),
                                    ("Mapped risk score", risk.mapped_score)], bands)
    b = sim.band
    labels = [str(p) for p in b.index]
    fan = charts.line_chart(
        labels, {"Median": (b["p50"].tolist(), "--seq-550"), "Total contributed": (b["contributed"].tolist(), "--text-muted")},
        hline=(req.target, f"Target {_money(req.target)}") if req.has_target else None,
        bands=[(b["p10"].tolist(), b["p90"].tolist(), "--seq-150", "10th-90th percentile"),
               (b["p25"].tolist(), b["p75"].tolist(), "--seq-250", "25th-75th percentile")],
        label="Monte Carlo projection", x_every=max(1, len(labels) // 8))
    term = sim.terminal_after_liquidation if sim.terminal_after_liquidation is not None else sim.terminal
    hist = charts.histogram(term.tolist(), req.target, label="terminal value distribution") if req.has_target else ""
    g = bench.growth_of_10k
    gl = [str(p) for p in g.index]
    growth = charts.line_chart(gl, {"Portfolio": (g["Portfolio"].tolist(), "--series-1"),
                                    "S&P 500": (g["S&P 500"].tolist(), "--series-2")}, label="growth of $10,000")
    wc = bench.wealth_with_contributions
    contrib = charts.line_chart(gl, {"Portfolio": (wc["Portfolio"].tolist(), "--series-1"),
                                     "S&P 500": (wc["S&P 500"].tolist(), "--series-2"),
                                     "Contributed": (wc["Contributed"].tolist(), "--text-muted")},
                                label="wealth with contributions")
    reviews = result.latest_reviews()
    final = [r for r in reviews if r.stage == "final"]
    review_ok = all(r.ok for r in reviews) and bool(final)
    n = sum(len(r.findings) for r in reviews)
    passed = sum(f.passed for r in reviews for f in r.findings)
    superseded = result.superseded_reviews()
    ctx = dict(
        client_name=req.client.profile.name, as_of=req.as_of, data_source=st["market"].source,
        synthetic=st["market"].synthetic, exp=exp, risk=risk, port=port, sim=sim, proj=st["projection"],
        scenarios=st["scenarios"], bench=bench, has_target=req.has_target, target=req.target,
        target_date=req.target_date.strftime("%B %Y") if req.target_date else None,
        target_probability=req.target_probability, months=req.months, W0=req.W0, C=req.C,
        taxes=req.client.taxes.enabled, money=_money, method_label=method_label(port.method),
        held_rows=held, unheld_rows=unheld, category_rows=category_rows, weight_total=float(port.weights.sum()),
        prisk=st["portfolio_risk"], bands=s.risk_bands if s else [],
        alloc_chart=alloc, risk_chart=risk_chart, fan_chart=fan, hist_chart=hist,
        growth_chart=growth, contrib_chart=contrib, bench_rows=_bench_rows(bench.metrics),
        stats_rows=list(est.stats.iterrows()), corr_tickers=est.tickers,
        corr_rows=list(zip(est.tickers, est.corr.tolist())), dq_rows=list(dq.tickers.values()),
        reviews=reviews, superseded=superseded, review_ok=review_ok,
        review_summary=f"{passed}/{n} checks passed" + (
            f" (after {len(superseded)} remediated attempt(s))" if superseded else ""),
        trace=result.trace, mermaid=mermaid, disclaimer=REPORT_DISCLAIMER,
        calculations=json.dumps(exp.calculations, indent=2, default=_json_default))
    return _ENV.get_template("report.html.j2").render(**ctx)


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (dt.date, pd.Period, pd.Timestamp)):
        return str(o)
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="split", date_format="iso"))
    return str(o)


def _review_json(r) -> dict:
    return {"stage": r.stage, "ok": r.ok,
            "findings": [{"rule": f.rule_id, "spec": f.spec_ref, "severity": f.severity,
                          "passed": f.passed, "message": f.message, "evidence": f.evidence}
                         for f in r.findings]}


def audit_bundle(result: RunResult) -> dict:
    """Everything needed to audit the run and to monitor it later (spec §18)."""
    st = result.state
    req, port, est, sim, risk = st["request"], st["portfolio"], st["estimates"], st.get("simulation"), st["risk"]
    out = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "client": json.loads(req.client.model_dump_json()),
        "as_of": str(req.as_of), "months": req.months,
        "data": {"source": st["market"].source, "synthetic": st["market"].synthetic,
                 "window_start": str(req.window_start), "quality_warnings": st["data_quality"].warnings},
        "risk": {"capacity": risk.capacity_score, "tolerance": risk.tolerance_score,
                 "mapped": risk.mapped_score, "profile": risk.profile, "max_volatility": risk.max_volatility},
        "portfolio": {"method": port.method, "weights": port.weight_map(),
                      "expected_return": port.expected_return, "volatility": port.volatility,
                      "sharpe": port.sharpe, "initial_allocation": port.initial_allocation,
                      "monthly_allocation": port.monthly_allocation, "notes": port.notes,
                      "goal_search": {k: v for k, v in (port.goal_search or {}).items() if k != "frontier"}},
        "estimates": {"tickers": est.tickers, "mu": est.mu.tolist(), "sigma": est.sigma.tolist(),
                      "corr": est.corr.tolist(), "risk_free": est.risk_free},
        "reviews": [_review_json(r) for r in result.latest_reviews()],
        "superseded_reviews": [_review_json(r) for r in result.superseded_reviews()],
        "trace": [{"wave": e.wave, "node": e.node, "kind": e.kind, "attempt": e.attempt, "status": e.status,
                   "seconds": round(e.seconds, 4), "outputs": e.outputs, "detail": e.detail} for e in result.trace],
    }
    if sim is not None:
        out["simulation"] = {"n_paths": sim.n_paths, "percentiles": sim.percentiles,
                             "prob_target": sim.prob_target, "prob_loss_principal": sim.prob_loss_principal,
                             "expected_terminal": sim.expected_terminal,
                             "max_drawdown_median": sim.max_drawdown_median}
    if "benchmark" in st:
        out["benchmark"] = json.loads(st["benchmark"].metrics.to_json())
    return json.loads(json.dumps(out, default=_json_default))
