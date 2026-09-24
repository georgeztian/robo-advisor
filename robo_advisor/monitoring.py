"""Ongoing monitoring (spec §18).

Re-assesses a previous recommendation (its audit bundle) against the client's updated
questionnaire/goal and fresh market data, and raises review triggers when:
  * the client changes their goal,
  * the client changes contributions,
  * the target becomes difficult to achieve,
  * the portfolio exceeds the risk limit,
  * market conditions materially change the portfolio's risk characteristics
    (portfolio volatility or asset correlations shift),
  * the client's questionnaire responses change the mapped risk profile.

Convention: in the updated client input, ``goal.initial_investment`` is the CURRENT
portfolio value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings
from .models import Estimates, Request, RiskAssessment
from .simulation import MCModel, simulate
from .tax import TaxRates


@dataclass
class Trigger:
    code: str
    message: str


@dataclass
class MonitoringReport:
    prior_as_of: str
    as_of: str
    triggers: list[Trigger]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def review_required(self) -> bool:
        return bool(self.triggers)


def assess(prior: dict, req: Request, risk: RiskAssessment, est: Estimates, rates: TaxRates,
           settings: Settings) -> MonitoringReport:
    cfg = settings.monitoring
    trig: list[Trigger] = []
    pw = prior["portfolio"]["weights"]
    w = np.array([pw.get(t, 0.0) for t in est.tickers])
    missing = [t for t, x in pw.items() if abs(x) > 1e-9 and t not in est.tickers]
    if missing:
        trig.append(Trigger("UNIVERSE_CHANGED", f"previously held ETFs no longer selected: {missing}"))
    m: dict[str, Any] = {}

    # goal & contribution changes
    old_goal, new_goal = prior["client"]["goal"], req.client.goal
    if (old_goal["has_target"] != new_goal.has_target
            or (new_goal.has_target and (old_goal.get("target_amount") != new_goal.target_amount
                                         or str(old_goal.get("target_date")) != str(new_goal.target_date)))):
        trig.append(Trigger("GOAL_CHANGED", "the investment goal (target amount/date) changed"))
    oc = old_goal["monthly_contribution"]
    if abs(req.C - oc) > cfg.contribution_change_trigger * max(oc, 1):
        trig.append(Trigger("CONTRIBUTION_CHANGED", f"monthly contribution changed from ${oc:,.0f} to ${req.C:,.0f}"))

    # questionnaire / risk profile
    pr = prior["risk"]
    m["mapped_score"] = {"prior": pr["mapped"], "now": risk.mapped_score}
    if risk.profile != pr["profile"]:
        trig.append(Trigger("RISK_PROFILE_CHANGED",
                            f"mapped risk profile changed from {pr['profile']} to {risk.profile}"))

    # risk limit & market-driven risk changes
    vol_now = float(np.sqrt(w @ est.cov @ w))
    vol_prior = prior["portfolio"]["volatility"]
    m["portfolio_volatility"] = {"prior": vol_prior, "now": vol_now, "limit": risk.max_volatility}
    if vol_now > risk.max_volatility + cfg.risk_limit_tolerance:
        trig.append(Trigger("RISK_LIMIT_EXCEEDED",
                            f"portfolio volatility {vol_now:.1%} exceeds the {risk.max_volatility:.0%} limit"))
    if vol_prior > 0 and abs(vol_now / vol_prior - 1) > cfg.volatility_change_trigger:
        trig.append(Trigger("VOLATILITY_SHIFT", f"estimated portfolio volatility moved from {vol_prior:.1%} to {vol_now:.1%}"))
    pt = prior["estimates"]["tickers"]
    common = [t for t in est.tickers if t in pt]
    if len(common) > 1:
        i_new = [est.tickers.index(t) for t in common]
        i_old = [pt.index(t) for t in common]
        c_new = est.corr[np.ix_(i_new, i_new)]
        c_old = np.array(prior["estimates"]["corr"])[np.ix_(i_old, i_old)]
        iu = np.triu_indices(len(common), 1)
        dc = float(np.abs(c_new[iu] - c_old[iu]).mean())
        m["mean_abs_correlation_change"] = dc
        if dc > cfg.correlation_change_trigger:
            trig.append(Trigger("CORRELATION_SHIFT", f"average pairwise correlation changed by {dc:.2f}"))
    sig_old = dict(zip(pt, prior["estimates"]["sigma"]))
    m["etf_volatility_change"] = {t: (float(est.sigma[est.tickers.index(t)]), sig_old[t]) for t in common}

    # target progress with the CURRENT value and the remaining months
    m["remaining_months"] = req.months
    if req.has_target and req.target:
        model = MCModel.from_estimates(est, settings.simulation.distribution)
        raw = simulate(w / w.sum() if abs(w.sum() - 1) > 1e-9 else w, model, req.W0, req.C, req.months,
                       req.rebalancing, rates, 5000, settings.simulation.seed, record=False)
        term = raw.terminal_after_liq if raw.terminal_after_liq is not None else raw.terminal
        p = float((term >= req.target).mean())
        m["prob_target"] = {"prior": prior.get("simulation", {}).get("prob_target"), "now": p}
        if p < cfg.target_probability_floor:
            trig.append(Trigger("TARGET_AT_RISK", f"probability of reaching the target fell to {p:.0%} "
                                                  f"(floor {cfg.target_probability_floor:.0%})"))
    return MonitoringReport(prior["as_of"], str(req.as_of), trig, m)
