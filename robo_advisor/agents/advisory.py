"""Advisory agents and the advisory workflow graph (spec §19 "Complete User Workflow").

Each agent is a small, single-responsibility node that reads declared blackboard keys and
writes its own. The graph engine derives the dependency structure from these contracts,
runs independent agents in parallel, and places the independent reviewer at four gates.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import benchmark as bm
from ..config import RebalancingCfg, Settings
from ..data.providers import DataProvider, DataUnavailableError, fetch_treasury_rate
from ..data.validation import validate
from ..estimation import estimate, portfolio_risk_stats
from ..explain import explain
from ..graph.engine import Graph, Node
from ..models import (CategoryCap, ClientInput, MarketData, Portfolio, Request, ResolvedConstraints,
                      TaxContext)
from ..optimization.goal import goal_search
from ..optimization.methods import METHODS, Optimizer
from ..projection import project
from ..questionnaire import assess_risk
from ..review import Reviewer
from ..simulation import MCModel, run_scenarios, simulate, summarize
from ..tax import DISCLAIMER, after_tax_returns, resolve_rates
from ..universe import resolve_universe


def last_business_day(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def years_before(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


@dataclass
class Services:
    settings: Settings
    provider: DataProvider


# ---------------------------------------------------------------------------------- agents


class IntakeAgent:
    """Step 1-2, 5: validate the client input and normalize the request."""

    name, requires, provides = "intake", ("client",), ("request",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st: Mapping[str, Any]) -> dict:
        s = self.sv.settings
        client: ClientInput = st["client"]
        as_of = last_business_day(client.as_of or dt.date.today())
        if as_of > dt.date.today():
            raise ValueError(f"as_of {as_of} is in the future; market data only exists up to today")
        tickers = resolve_universe(client.universe, s.universe.categories)
        pref = client.preferences
        method = pref.optimization_method or s.optimization.default_method
        if method not in METHODS:
            raise ValueError(f"unknown optimization method {method!r}; choose from {METHODS}")
        if not client.goal.has_target and method == "target_return" and pref.target_return is None:
            raise ValueError("the target_return method needs preferences.target_return (annual return)")
        rb = s.rebalancing.model_copy(update={k: v for k, v in {
            "type": pref.rebalancing_type, "frequency": pref.rebalancing_frequency,
            "threshold": pref.rebalancing_threshold}.items() if v is not None})
        g = client.goal
        req = Request(
            client=client, as_of=as_of, window_start=years_before(as_of, s.data.lookback_years),
            months=g.months(as_of), W0=g.initial_investment, C=g.monthly_contribution,
            target=g.target_amount if g.has_target else None,
            target_date=g.target_date if g.has_target else None, has_target=g.has_target,
            tickers=tickers, method="goal" if g.has_target else method, rebalancing=RebalancingCfg(**rb.model_dump()),
            target_probability=pref.target_probability or s.optimization.goal.target_probability,
            goal_risk_metric=pref.goal_risk_metric or s.optimization.goal.risk_metric,
            data_tickers=sorted(set(tickers) | {s.data.benchmark, s.data.risk_free_ticker}))
        return {"request": req}


class RiskProfilerAgent:
    """Step 3: capacity and tolerance scored separately; mapped = min."""

    name, requires, provides = "risk_profiler", ("request",), ("risk",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req: Request = st["request"]
        return {"risk": assess_risk(self.sv.settings, req.client, req.as_of)}


class MarketDataAgent:
    """Step 6a: retrieve adjusted total-return histories (selected ETFs + benchmark + T-bills)."""

    name, requires, provides = "market_data", ("request",), ("market",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req: Request = st["request"]
        p, d = self.sv.provider, self.sv.settings.data
        frames = p.fetch(req.data_tickers, req.window_start, req.as_of)
        md = MarketData(frames, req.as_of, p.name, p.synthetic, notes=list(getattr(p, "notes", [])))
        if d.risk_free_source == "fred":
            if p.synthetic:
                md.notes.append("risk-free: FRED not used with synthetic data; T-bill ETF proxy used")
            else:
                try:
                    md.risk_free_series = fetch_treasury_rate(d.fred_series, req.window_start, req.as_of,
                                                              d.cache_dir)
                    md.risk_free_source = f"FRED {d.fred_series} Treasury bill rate"
                except DataUnavailableError as e:
                    md.notes.append(f"risk-free: {e}; T-bill ETF proxy used instead")
        return {"market": md}


class DataValidationAgent:
    """Step 6b: ETF data validation (spec §3)."""

    name, requires, provides = "data_validation", ("request", "market"), ("data_quality",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req, market = st["request"], st["market"]
        dq = validate(market.frames, req.window_start, req.as_of, self.sv.settings.data)
        dq.warnings.extend(market.notes)
        return {"data_quality": dq}


class EstimationAgent:
    """Step 6c: returns, volatility, covariance, downside risk (spec §6)."""

    name, requires, provides = "estimation", ("request", "market", "review_data"), ("estimates",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req, market = st["request"], st["market"]
        s = self.sv.settings
        return {"estimates": estimate(market.frames, req.tickers, req.as_of, req.window_start, s.data, s.estimation,
                                      market.risk_free_series, market.risk_free_source)}


class TaxAgent:
    """Spec §5: after-tax return series when taxes are enabled."""

    name, requires, provides = "tax_adjust", ("request", "estimates"), ("tax",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req, est = st["request"], st["estimates"]
        rates = resolve_rates(req.tickers, req.client.taxes, self.sv.settings.tax)
        if not rates.enabled:
            return {"tax": TaxContext(rates, None, None, None)}
        monthly, mu_after = after_tax_returns(est, rates)
        return {"tax": TaxContext(rates, mu_after, monthly, DISCLAIMER)}


class ConstraintAgent:
    """Step 5: resolve short-sale / leverage / position / risk-limit constraints."""

    name, requires, provides = "constraints", ("request", "risk"), ("constraints",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req, risk = st["request"], st["risk"]
        s = self.sv.settings
        c, o = req.client.constraints, s.optimization
        unknown = set(c.category_limits or {}) - set(s.universe.categories)
        if unknown:
            raise ValueError(f"category_limits names unknown categories {sorted(unknown)}; "
                             f"categories are: {', '.join(s.universe.categories)}")
        limits = {**o.category_limits, **(c.category_limits or {})}
        caps = [CategoryCap(cat, lim, [t for t in tickers if t in req.tickers])
                for cat, tickers in s.universe.categories.items()
                if (lim := limits.get(cat, 1.0)) < 1.0 and any(t in req.tickers for t in tickers)]
        rc = ResolvedConstraints(
            tickers=req.tickers, allow_short=c.allow_short,
            max_position=c.max_position if c.max_position is not None else o.max_position,
            max_gross_leverage=(c.max_gross_leverage if c.max_gross_leverage is not None
                                else o.max_gross_leverage) if c.allow_short else 1.0,
            max_volatility=risk.max_volatility, category_caps=caps)
        return {"constraints": rc}


class OptimizerAgent:
    """Step 7: Case A (target) minimizes risk subject to the target; Case B maximizes return
    subject to the mapped risk limit (or the client's chosen alternative method)."""

    name = "optimizer"
    requires = ("request", "estimates", "tax", "constraints", "review_inputs")
    optional = ("remediation",)
    provides = ("portfolio",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        s = self.sv.settings
        req: Request = st["request"]
        est, tax, rc = st["estimates"], st["tax"], st["constraints"]
        attempt = st.get("remediation", {}).get("attempt", 0)
        mu = tax.mu_after_tax if tax.mu_after_tax is not None else est.mu
        scen = (tax.monthly_after_tax if tax.monthly_after_tax is not None else est.monthly_returns).dropna().to_numpy()
        # short legs pay the pre-tax return: a short cannot collect the borrowed asset's tax drag
        opt = Optimizer(mu, est.cov, rc, est.risk_free, scen, n_starts=s.optimization.n_starts * 2**attempt,
                        seed=attempt, cvar_alpha=s.optimization.cvar_alpha, mu_short=est.mu,
                        scenarios_short=est.monthly_returns.dropna().to_numpy())
        notes = [f"remediation attempt {attempt}: re-optimized with more solver starts"] if attempt else []
        if req.has_target:
            g = s.optimization.goal
            model = MCModel.from_estimates(est, s.simulation.distribution)
            res = goal_search(opt, model, req.W0, req.C, req.months, req.target, req.target_probability,
                              req.rebalancing, tax.rates, g.search_paths,
                              s.simulation.seed + 1,          # search sample != final verification sample
                              g.frontier_points, g.probability_margin, req.goal_risk_metric, s.optimization.cvar_alpha,
                              verify_paths=s.simulation.n_paths, verify_seed=s.simulation.seed)
            case = "target"
        else:
            res = opt.run(req.method, target=req.client.preferences.target_return)
            case = "no_target"
        w = res.weights
        vol = opt.vol(w)
        ret = opt.port_return(w)
        port = Portfolio(
            tickers=req.tickers, weights=w, method=res.method, case=case, expected_return=ret,
            expected_return_pretax=float(w @ est.mu), volatility=vol,
            sharpe=(ret - est.risk_free) / vol if vol > 0 else float("nan"), constraints=rc,
            initial_allocation={t: float(x * req.W0) for t, x in zip(req.tickers, w)},
            monthly_allocation={t: float(x * req.C) for t, x in zip(req.tickers, w)},
            goal_search=res.diagnostics if req.has_target else None, notes=notes + res.notes)
        return {"portfolio": port}


class SimulationAgent:
    """Step 9: Monte Carlo projection (10,000 paths)."""

    name, requires, provides = "simulation", ("request", "estimates", "tax", "portfolio", "review_portfolio"), ("simulation",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        s = self.sv.settings.simulation
        req, est, tax, port = st["request"], st["estimates"], st["tax"], st["portfolio"]
        model = MCModel.from_estimates(est, s.distribution)
        raw = simulate(port.weights, model, req.W0, req.C, req.months, req.rebalancing, tax.rates, s.n_paths, s.seed)
        return {"simulation": summarize(raw, req.months, req.W0, req.C, req.target, s.percentiles, s.seed,
                                        s.distribution, pd.Period(req.as_of, "M"))}


class ScenarioAgent:
    """Spec §13: conservative / base / optimistic scenarios."""

    name, requires, provides = "scenarios", ("request", "estimates", "tax", "portfolio", "review_portfolio"), ("scenarios",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        s = self.sv.settings
        req, est, tax, port = st["request"], st["estimates"], st["tax"], st["portfolio"]
        return {"scenarios": run_scenarios(port.weights, est, s.scenarios, req.W0, req.C, req.months,
                                           req.rebalancing, tax.rates, req.target, s.simulation.seed + 2,
                                           s.simulation.distribution)}


class BenchmarkAgent:
    """Step 10: S&P 500 comparison over the most recent 10 years."""

    name, requires, provides = "benchmark", ("request", "market", "portfolio", "review_portfolio"), ("benchmark",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        s = self.sv.settings
        req, market, port = st["request"], st["market"], st["portfolio"]
        return {"benchmark": bm.compare(market.frames, req.tickers, port.weights, req.W0, req.C, req.as_of,
                                        s.benchmark.years, req.rebalancing, s.data.benchmark,
                                        s.data.risk_free_ticker, s.data.risk_free_fallback,
                                        market.risk_free_series)}


class ProjectionAgent:
    """Spec §11: deterministic future value."""

    name, requires, provides = "projection", ("request", "portfolio", "review_portfolio"), ("projection",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        req, port = st["request"], st["portfolio"]
        return {"projection": project(req.W0, req.C, port.expected_return, req.months,
                                      self.sv.settings.simulation.inflation)}


class ExplainerAgent:
    """Step 8 & 11 content: recommendation explanation (spec §10, §16)."""

    name = "explainer"
    requires = ("request", "risk", "estimates", "tax", "portfolio", "simulation", "projection",
                "benchmark", "scenarios", "data_quality", "market")
    provides = ("explanation", "portfolio_risk")

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        s = self.sv.settings
        exp = explain(st["request"], st["risk"], st["estimates"], st["tax"], st["portfolio"], st["simulation"],
                      st["projection"], st["benchmark"], st["scenarios"], st["data_quality"],
                      st["market"].synthetic, s.simulation.inflation, s.data.lookback_years,
                      s.universe.categories)
        prs = portfolio_risk_stats(st["portfolio"].weights, st["estimates"], s.estimation.var_confidence)
        return {"explanation": exp, "portfolio_risk": prs}


def _gate(name: str, fn, requires: tuple[str, ...], provides: str, desc: str,
          remediate: tuple[str, ...] = (), retries: int = 0) -> Node:
    def run(st):
        return {provides: fn(st)}
    return Node(name, run, requires, (provides,), kind="gate", description=desc,
                remediate=remediate, max_retries=retries)


def build_advisory_graph(settings: Settings, provider: DataProvider) -> Graph:
    sv = Services(settings, provider)
    rv = Reviewer(settings)
    g = Graph("advisory", inputs=("client",))
    for cls in (IntakeAgent, RiskProfilerAgent, MarketDataAgent, DataValidationAgent, EstimationAgent,
                TaxAgent, ConstraintAgent, OptimizerAgent, SimulationAgent, ScenarioAgent, BenchmarkAgent,
                ProjectionAgent, ExplainerAgent):
        a = cls(sv)
        g.add(Node(a.name, a, tuple(a.requires), tuple(a.provides), tuple(getattr(a, "optional", ())),
                   description=(cls.__doc__ or "").strip().splitlines()[0]))
    g.add(_gate("review_data_gate", rv.review_data, ("request", "market", "data_quality"), "review_data",
                "Reviewer: data validation, look-ahead, universe (before estimation)"))
    g.add(_gate("review_inputs_gate", rv.review_inputs,
                ("request", "risk", "market", "data_quality", "estimates", "tax", "constraints"),
                "review_inputs", "Reviewer: data, estimates, risk mapping, constraints"))
    g.add(_gate("review_portfolio_gate", rv.review_portfolio,
                ("request", "estimates", "tax", "constraints", "portfolio"),
                "review_portfolio", "Reviewer: constraint compliance, allocations, optimality",
                remediate=("optimizer",), retries=2))
    g.add(_gate("final_review_gate", rv.review_final,
                ("request", "risk", "estimates", "tax", "constraints", "portfolio", "simulation", "projection",
                 "benchmark", "scenarios", "explanation", "market"),
                "review_final", "Reviewer: simulation, projection, benchmark, disclosures"))
    g.build()
    return g
