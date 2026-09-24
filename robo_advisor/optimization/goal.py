"""Case A: client has a target amount and date (spec §7).

    min_w Risk(w)   s.t.  P(F_T(w) >= F*) >= p,   Risk(w) <= Risk_max,
                          sum w = 1, short-sale / position constraints

F_T depends on the initial investment W0 and on every monthly contribution C, so P(.) is
evaluated by Monte Carlo on the same path engine used for the final projection (including
rebalancing and taxes). To make the search stable, candidates are evaluated with common
random numbers (same seed for every candidate).

Search: the minimum-variance frontier between the minimum-volatility portfolio and the
maximum-return portfolio at the risk cap is traced; the lowest-risk frontier portfolio that
meets the probability requirement (plus a small margin to absorb MC noise) is refined by
bisection on the frontier's target return. Risk is volatility (default) or CVaR_alpha of
terminal wealth. If no admissible portfolio reaches p, the probability-maximizing portfolio
within the risk cap is returned, flagged infeasible, along with the monthly contribution
that would be required to reach p.
"""
from __future__ import annotations

import numpy as np

from ..config import RebalancingCfg
from ..simulation import MCModel, simulate
from ..tax import TaxRates
from .methods import Optimizer, OptResult


def _terminal(w, model, W0, C, T, rebal, tax, paths, seed):
    raw = simulate(w, model, W0, C, T, rebal, tax, paths, seed, record=False)
    return raw.terminal_after_liq if raw.terminal_after_liq is not None else raw.terminal


def terminal_cvar_risk(term: np.ndarray, alpha: float) -> float:
    """Risk = -(mean of the worst (1-alpha) share of terminal wealth); lower is better."""
    k = max(1, int((1 - alpha) * len(term)))
    return -float(np.sort(term)[:k].mean())


def goal_search(opt: Optimizer, model: MCModel, W0: float, C: float, T: int, target: float,
                p: float, rebal: RebalancingCfg, tax: TaxRates, paths: int, seed: int,
                points: int, margin: float, risk_metric: str = "volatility",
                alpha: float = 0.95, verify_paths: int | None = None,
                verify_seed: int | None = None) -> OptResult:
    opt.check_risk_limit_feasible()
    w_lo = opt.min_vol_portfolio()
    w_hi = opt.mean_variance().weights
    r_lo, r_hi = opt.port_return(w_lo), opt.port_return(w_hi)
    evals: list[dict] = []

    def evaluate(w: np.ndarray, r: float) -> dict:
        term = _terminal(w, model, W0, C, T, rebal, tax, paths, seed)
        e = {"target_return": r, "exp_return": opt.port_return(w), "vol": opt.vol(w),
             "prob": float((term >= target).mean()), "exp_terminal": float(term.mean()),
             "cvar_risk": terminal_cvar_risk(term, alpha), "w": w}
        evals.append(e)
        return e

    frontier = []
    for r in np.linspace(r_lo, r_hi, max(points, 2)):
        if r <= r_lo + 1e-12:
            w = w_lo
        elif r >= r_hi - 1e-12:
            w = w_hi
        else:
            try:
                w = opt.target_return(r).weights
            except ValueError:
                continue
        frontier.append(evaluate(w, r))

    need = p + margin
    feas = [e for e in frontier if e["prob"] >= need]
    infeasible = not feas
    required_c = None
    if infeasible:
        best = max(frontier, key=lambda e: (e["prob"], -e["vol"]))
        required_c = _required_contribution(best["w"], model, W0, C, T, target, need, rebal, tax,
                                            paths, seed)
    elif risk_metric == "cvar":
        best = min(feas, key=lambda e: e["cvar_risk"])
    else:
        best = min(feas, key=lambda e: e["vol"])
        # refine between the adjacent lower-return frontier point and the chosen one
        lower = [e for e in frontier if e["target_return"] < best["target_return"]]
        if lower:
            lo_r, hi_r = max(e["target_return"] for e in lower), best["target_return"]
            for _ in range(6):
                mid = (lo_r + hi_r) / 2
                try:
                    e = evaluate(opt.target_return(mid).weights, mid)
                except ValueError:
                    break
                if e["prob"] >= need:
                    hi_r = mid
                    if e["vol"] <= best["vol"]:
                        best = e
                else:
                    lo_r = mid
    verified = None
    if not infeasible and verify_paths:
        best, verified = _verify(opt, best, frontier, model, W0, C, T, target, p, rebal, tax,
                                 verify_paths, verify_seed if verify_seed is not None else seed + 1, alpha)
        if verified is None:
            infeasible = True
            best = max(frontier, key=lambda e: (e["prob"], -e["vol"]))
            required_c = _required_contribution(best["w"], model, W0, C, T, target, p, rebal, tax,
                                                verify_paths, verify_seed if verify_seed is not None else seed + 1)
    notes = []
    if infeasible:
        notes.append(
            f"No portfolio within the risk limit reaches a {p:.0%} probability of the target; "
            f"showing the portfolio with the highest probability ({best['prob']:.1%}).")
        if required_c is not None:
            notes.append(f"A monthly contribution of about ${required_c:,.0f} would be required to "
                         f"reach a {p:.0%} probability with this portfolio.")
    if best["exp_terminal"] < target:
        notes.append("Expected terminal wealth is below the target.")
    diag = {
        "target_probability": p, "margin": margin, "search_paths": paths, "seed": seed,
        "risk_metric": risk_metric, "infeasible": infeasible,
        "search_prob": best["prob"], "required_monthly_contribution": required_c,
        "frontier": [{k: v for k, v in e.items() if k != "w"} for e in frontier],
        "n_evaluations": len(evals), "verified_prob": verified,
        "verify_paths": verify_paths,
    }
    method = "goal_min_cvar" if risk_metric == "cvar" else "goal_min_volatility"
    return OptResult(best["w"], method, notes, diag)


def _verify(opt, best, frontier, model, W0, C, T, target, p, rebal, tax, paths, seed, alpha):
    """Re-evaluate the chosen portfolio on the full verification sample (the same sample the
    final projection reports). If it falls short of p, move up the frontier (higher return)
    and bisect with the full sample until the constraint holds."""
    def prob(w):
        term = _terminal(w, model, W0, C, T, rebal, tax, paths, seed)
        return float((term >= target).mean()), float(term.mean())

    pb, eb = prob(best["w"])
    if pb >= p:
        return {**best, "prob": pb, "exp_terminal": eb}, pb
    higher = sorted((e for e in frontier if e["target_return"] > best["target_return"]),
                    key=lambda e: e["target_return"])
    lo_r = best["target_return"]
    for e in higher:
        pe, ee = prob(e["w"])
        if pe >= p:
            hi_r, hi, hi_p = e["target_return"], {**e, "prob": pe, "exp_terminal": ee}, pe
            break
        lo_r = e["target_return"]
    else:
        return best, None
    for _ in range(6):
        mid = (lo_r + hi_r) / 2
        try:
            w = opt.target_return(mid).weights
        except ValueError:
            break
        pm, em = prob(w)
        if pm >= p:
            hi_r, hi_p = mid, pm
            hi = {"target_return": mid, "exp_return": opt.port_return(w), "vol": opt.vol(w), "prob": pm,
                  "exp_terminal": em, "cvar_risk": float("nan"), "w": w}
        else:
            lo_r = mid
    return hi, hi_p


def _required_contribution(w, model, W0, C, T, target, need, rebal, tax, paths, seed):
    def prob(c):
        return float((_terminal(w, model, W0, c, T, rebal, tax, paths, seed) >= target).mean())

    lo, hi = C, max(C * 2, target / T)
    for _ in range(20):
        if prob(hi) >= need:
            break
        lo, hi = hi, hi * 2
    else:
        return None
    for _ in range(25):
        mid = (lo + hi) / 2
        if prob(mid) >= need:
            hi = mid
        else:
            lo = mid
    return hi
