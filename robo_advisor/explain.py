"""Recommendation explanation (spec §10, §16): why each ETF gets its weight, what the risk
profile implies, how the optimizer decided, and the assumptions and limitations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import (BenchmarkResult, DataQualityReport, Estimates, Explanation, Portfolio, Projection,
                     Request, RiskAssessment, ScenarioResult, SimulationResult, TaxContext)
from .optimization.methods import METHOD_LABELS
from .rebalancing import describe as describe_rebalancing
from .universe import CATALOG

DISCLOSURE_ESTIMATES = ("Expected returns, volatilities and correlations are historical estimates / "
                        "model assumptions. They are not guaranteed future returns.")
DISCLOSURE_SCENARIOS = ("Scenarios are illustrative what-if assumptions, not forecasts.")
DISCLOSURE_SYNTHETIC = ("SYNTHETIC DATA: this report was generated from simulated price histories for "
                        "demonstration and testing. The figures do not describe real ETF performance.")
DISCLOSURE_LEVERAGED_PREFIX = "LEVERAGED ETF:"
DISCLOSURE_NOT_ADVICE = ("This is an automated, model-based illustration and not personalized "
                         "investment, legal or tax advice.")

GOAL_METHOD_LABELS = {
    "goal_min_volatility": "Target-based: minimize volatility subject to P(terminal wealth >= target) >= p",
    "goal_min_cvar": "Target-based: minimize CVaR of terminal wealth subject to P(terminal wealth >= target) >= p",
}


def method_label(method: str) -> str:
    return GOAL_METHOD_LABELS.get(method) or METHOD_LABELS.get(method, method)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio variance: w_i (Sigma w)_i / (w' Sigma w); sums to 1."""
    var = float(w @ cov @ w)
    return w * (cov @ w) / var if var > 0 else np.zeros_like(w)


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _reason(i: int, w: np.ndarray, est: Estimates, rc: np.ndarray, corr_to_port: np.ndarray,
            max_pos: float, mu: np.ndarray) -> str:
    t = est.tickers[i]
    if abs(w[i]) < 1e-6:
        held = [j for j in range(len(w)) if w[j] > 1e-6]
        if held:
            j = max(held, key=lambda j: est.corr[i, j])
            if est.corr[i, j] > 0.85 and mu[j] >= mu[i]:
                return (f"Not held: highly correlated with {est.tickers[j]} (rho={est.corr[i, j]:.2f}) "
                        f"with a lower estimated return ({mu[i]:.1%} vs {mu[j]:.1%}).")
        return "Not held: did not improve the risk/return trade-off under your constraints."
    if w[i] < 0:
        return (f"Short position: its estimated return ({mu[i]:.1%}) is low relative to its covariance "
                "with the other holdings; shorting it funds higher-return positions within the "
                "gross-exposure limit.")
    parts = []
    if abs(w[i] - max_pos) < 1e-4:
        parts.append(f"at the {max_pos:.0%} position cap")
    if est.sigma[i] < 0.03:
        parts.append(f"stabilizer: very low volatility ({est.sigma[i]:.1%})")
    elif corr_to_port[i] < 0.3:
        parts.append(f"diversifier: low correlation with the rest of the portfolio (rho={corr_to_port[i]:.2f})")
    else:
        parts.append(f"return driver: estimated return {mu[i]:.1%} at {est.sigma[i]:.1%} volatility")
    parts.append(f"contributes {rc[i]:.0%} of portfolio risk")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


def explain(req: Request, risk: RiskAssessment, est: Estimates, tax: TaxContext, port: Portfolio,
            sim: SimulationResult, proj: Projection, bench: BenchmarkResult,
            scenarios: list[ScenarioResult], dq: DataQualityReport, synthetic: bool,
            inflation: float, lookback_years: int = 20) -> Explanation:
    w = port.weights
    mu = tax.mu_after_tax if tax.mu_after_tax is not None else est.mu
    rc = risk_contributions(w, est.cov)
    port_cov = est.cov @ w
    sp = np.sqrt(w @ est.cov @ w)
    corr_to_port = np.array([port_cov[i] / (est.sigma[i] * sp) if est.sigma[i] > 0 else 0 for i in range(len(w))])
    rows = []
    for i, t in enumerate(est.tickers):
        rows.append({
            "ETF": t, "Name": CATALOG[t].name, "Asset class": CATALOG[t].asset_class,
            "Weight": w[i], "Initial": port.initial_allocation[t], "Monthly": port.monthly_allocation[t],
            "Est. return": mu[i], "Volatility": est.sigma[i], "Corr. to portfolio": corr_to_port[i],
            "Risk contribution": rc[i], "History (yrs)": est.history_years[t],
            "Expense ratio": CATALOG[t].expense_ratio,
            "Rationale": _reason(i, w, est, rc, corr_to_port, port.constraints.max_position, mu),
        })
    table = pd.DataFrame(rows).set_index("ETF")

    risk_text = (
        f"Your risk-capacity score is {risk.capacity_score:.0f} and your risk-tolerance score is "
        f"{risk.tolerance_score:.0f}. Your mapped risk score is therefore {risk.mapped_score:.0f} "
        f"(the lower of the two), which corresponds to the {risk.profile} profile with a maximum "
        f"volatility of {risk.max_volatility:.0%}. The recommended portfolio has an estimated annual "
        f"volatility of {port.volatility:.1%}, which is "
        f"{'within' if port.volatility <= risk.max_volatility + 1e-9 else 'ABOVE'} the risk constraint "
        f"associated with your mapped profile.")
    if req.has_target:
        pt = sim.prob_target or 0.0
        gs = port.goal_search or {}
        intro = (f"Given your initial investment of {_money(req.W0)}, monthly contribution of {_money(req.C)}, "
                 f"and target of {_money(req.target)} by {req.target_date:%B %Y} ({req.months} months), ")
        if not gs.get("infeasible"):
            goal_text = intro + (
                f"the optimizer selected the portfolio that minimizes estimated "
                f"{'CVaR of terminal wealth' if req.goal_risk_metric == 'cvar' else 'volatility'} while "
                f"satisfying the target-achievement constraint P(terminal wealth >= target) >= "
                f"{req.target_probability:.0%}. In {sim.n_paths:,} simulated paths the probability of "
                f"reaching the target is {pt:.1%}.")
        else:
            goal_text = intro + (
                f"No portfolio within your risk limit reaches the required {req.target_probability:.0%} "
                f"probability of meeting the target, so the optimizer selected the portfolio with the highest "
                f"probability: {pt:.1%} in {sim.n_paths:,} simulated paths.")
            if gs.get("required_monthly_contribution"):
                goal_text += (f" Raising the monthly contribution to about "
                              f"{_money(gs['required_monthly_contribution'])} would reach "
                              f"{req.target_probability:.0%} with this portfolio.")
    else:
        goal_text = (
            f"Because you did not specify a target amount, the optimizer maximizes estimated portfolio "
            f"return subject to the {risk.max_volatility:.0%} volatility limit implied by your mapped risk "
            f"profile ({method_label(port.method)}). Over your {req.months / 12:.0f}-year horizon the median "
            f"simulated portfolio value is {_money(sim.percentiles[50])}.")
    portfolio_text = (
        f"Estimated annual return {port.expected_return:.2%}"
        + (f" after tax ({port.expected_return_pretax:.2%} pre-tax)" if tax.mu_after_tax is not None else "")
        + f", volatility {port.volatility:.2%}, Sharpe ratio {port.sharpe:.2f}.")
    headline = (f"{risk.profile} portfolio · est. return {port.expected_return:.1%} · volatility "
                f"{port.volatility:.1%}" + (f" · {sim.prob_target:.0%} chance of reaching "
                                              f"{_money(req.target)}" if req.has_target else ""))

    methodology = [
        f"Optimization: {method_label(port.method)}.",
        f"Estimation window: {est.window_start} to {est.window_end} (most recent {lookback_years} years, "
        "maximum available history for younger ETFs).",
        "Risk (volatility, covariance) from daily adjusted total returns, annualized x252; covariance "
        "estimated pairwise over overlapping history" + (" and repaired to the nearest positive "
                                                         "semi-definite matrix." if est.psd_repaired else "."),
        "Expected returns from monthly total returns (arithmetic mean x12)"
        + (", after tax." if tax.mu_after_tax is not None else "."),
        f"Monte Carlo: {sim.n_paths:,} paths, monthly {sim.distribution} returns, contributions added monthly "
        f"at target weights, {describe_rebalancing(req.rebalancing).lower()}"
        + (", taxes applied." if tax.rates.enabled else "."),
        f"Benchmark: S&P 500 (VOO total return), {bench.start} to {bench.end}, same initial investment, "
        "contributions and monthly return convention.",
    ]
    assumptions = [
        f"Risk-free rate (T-bill ETF proxy): {est.risk_free:.2%} per year.",
        f"Inflation assumption: {inflation:.1%} per year (real value of the deterministic projection: "
        f"{_money(proj.fv_real)}).",
        f"Constraints: {'short sales allowed, gross exposure <= ' + format(port.constraints.max_gross_leverage, '.0%') if port.constraints.allow_short else 'long-only'}"
        f", max position {port.constraints.max_position:.0%}, max volatility {port.constraints.max_volatility:.0%}.",
        "Scenario shifts: " + "; ".join(f"{s.name} {s.mu_shift:+.1%} return, x{s.vol_multiplier:.2f} volatility"
                                         for s in scenarios) + ".",
    ]
    if tax.rates.enabled:
        r = tax.rates
        assumptions.append(f"Taxes: short-term gains {r.st:.0%}, long-term gains {r.lt.min():.0%}"
                           + (f" ({r.lt.max():.0%} collectibles)" if r.lt.max() > r.lt.min() else "")
                           + f", ordinary income {r.ordinary:.0%}; liquidation at horizon "
                           f"{'included' if r.liquidate_at_horizon else 'not included'}.")
    limitations = [
        "Past performance does not guarantee future results; estimates carry substantial error.",
        "Log-normal monthly returns understate extreme tail events." if sim.distribution == "lognormal"
        else "Bootstrap resampling assumes future months resemble historical ones.",
    ]
    short = [t for t, q in dq.tickers.items() if not q.meets_min_history and t in req.tickers]
    if short:
        limitations.append("Less than 20 years of history (maximum available used): " + ", ".join(
            f"{t} ({dq.tickers[t].years_available:.1f}y)" for t in short) + ".")
    limitations += [f"Benchmark note: {n}" for n in bench.notes]
    disclosures = [DISCLOSURE_ESTIMATES, DISCLOSURE_SCENARIOS, DISCLOSURE_NOT_ADVICE]
    lev = [t for t, x in zip(est.tickers, w) if abs(x) > 1e-6 and CATALOG[t].leveraged]
    for t in lev:
        cagr = float(est.stats.loc[t, "cagr"])
        disclosures.append(
            f"{DISCLOSURE_LEVERAGED_PREFIX} {t} resets its leverage daily; over long horizons its compound "
            f"return can differ sharply from the leveraged index return (volatility decay). Its arithmetic "
            f"mean return estimate ({est.mu[est.tickers.index(t)]:.1%}) exceeds its historical compound growth "
            f"({cagr:.1%}), which mean-variance optimization does not penalize.")
    if tax.disclaimer:
        disclosures.append(tax.disclaimer)
    if synthetic:
        disclosures.insert(0, DISCLOSURE_SYNTHETIC)
    calculations = {
        "capacity_score": {"formula": "sum(weight_k * score_k)", "value": risk.capacity_score,
                           "items": {k: round(v["contribution"], 2) for k, v in risk.capacity_detail.items()}},
        "tolerance_score": {"formula": "sum(weight_k * score_k)", "value": risk.tolerance_score,
                            "items": {k: round(v["contribution"], 2) for k, v in risk.tolerance_detail.items()}},
        "mapped_score": {"formula": "min(capacity, tolerance)", "value": risk.mapped_score},
        "max_volatility": risk.max_volatility,
        "expected_return": {"formula": "w' mu", "value": port.expected_return},
        "volatility": {"formula": "sqrt(w' Sigma w)", "value": port.volatility},
        "deterministic_fv": {"formula": "W0(1+r)^T + C[((1+r)^T - 1)/r]", "W0": req.W0, "C": req.C,
                             "r_monthly": proj.monthly_rate, "T": proj.months, "value": proj.fv_nominal},
        "prob_target": {"formula": "share of paths with terminal wealth >= target",
                        "value": sim.prob_target, "paths": sim.n_paths},
    }
    return Explanation(headline, risk_text, goal_text, portfolio_text, methodology, table, assumptions,
                       limitations, disclosures, calculations)
