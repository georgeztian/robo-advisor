"""The optimization problem behind a recommendation, written out for the client report: the
process in steps, the problem in mathematical form with every constraint that was imposed (with
the client's actual numbers and how the recommended portfolio sits against each one), and the
same problem in plain English.

Math is returned as small HTML fragments (italic symbols, sub/superscripts, Unicode operators) so
the report renders it offline without a math library. Anything that is not our own markup
(category names) is escaped.
"""
from __future__ import annotations

import numpy as np
from markupsafe import escape

from ..explain import money
from ..models import Estimates, Portfolio, Request, RiskAssessment, SimulationResult, TaxContext

TOL = 1e-4
W = "<i>w</i>"
WI = "<i>w</i><sub><i>i</i></sub>"
MU_W = "<i>μ</i><sup>⊤</sup><i>w</i>"
COV = "<b>V</b>"          # covariance matrix: a distinct letter, so it can't be mistaken for ∑ (sum)
VAR = f"<i>w</i><sup>⊤</sup>{COV}<i>w</i>"
VOL = "σ<sub><i>p</i></sub>(<i>w</i>)"
PROB = "P(<i>F</i><sub><i>T</i></sub>(<i>w</i>) ≥ <i>F</i><sup>*</sup>)"


def _pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}%}"


def _objective(port: Portfolio, req: Request, alpha: float) -> tuple[str, str, str]:
    """(math, plain English, short goal phrase) for the objective the optimizer used."""
    m = port.method
    tail = f"{int(round((1 - alpha) * 100))}%"
    if m == "goal_min_volatility":
        return (f"minimize over {W}: &nbsp;{VOL} = √({VAR})",
                "Make the portfolio's ups and downs (volatility) as small as possible.",
                "has the smallest ups and downs")
    if m == "goal_min_cvar":
        return (f"minimize over {W}: &nbsp;− mean of the worst {tail} of <i>F</i><sub><i>T</i></sub>(<i>w</i>)",
                f"Make the bad outcomes as mild as possible: raise the average final value in the worst "
                f"{tail} of simulated futures as high as it can go.",
                f"has the mildest bad outcomes (the best average result in the worst {tail} of futures)")
    if m == "mean_variance":
        return (f"maximize over {W}: &nbsp;{MU_W} = ∑<sub><i>i</i></sub> <i>μ</i><sub><i>i</i></sub> {WI}",
                "Make the portfolio's expected yearly return as high as possible.",
                "has the highest expected return")
    if m == "min_volatility":
        return (f"minimize over {W}: &nbsp;{VAR} &nbsp;(the square of {VOL})",
                "Make the portfolio's ups and downs (volatility) as small as possible, whatever the return.",
                "has the smallest ups and downs")
    if m == "max_sharpe":
        return (f"maximize over {W}: &nbsp;({MU_W} − <i>r</i><sub><i>f</i></sub>) / {VOL}",
                "Get the most expected return above the risk-free rate for each unit of volatility "
                "(the Sharpe ratio).",
                "earns the most return per unit of risk")
    if m == "target_return":
        return (f"minimize over {W}: &nbsp;{VAR}",
                "Make the portfolio's ups and downs as small as possible while still earning the return you chose.",
                "has the smallest ups and downs while earning your chosen return")
    if m == "cvar":
        return (f"minimize over {W}, <i>ζ</i>: &nbsp;<i>ζ</i> + 1/((1−<i>α</i>)<i>S</i>) · ∑<sub><i>s</i>=1..<i>S</i></sub> "
                f"max(0, −<i>r</i><sub><i>s</i></sub><sup>⊤</sup><i>w</i> − <i>ζ</i>) &nbsp;(= CVaR<sub><i>α</i></sub>, <i>α</i> = {alpha:.0%})",
                f"Make the average loss in the worst {tail} of past months as small as possible (tail risk).",
                f"has the smallest average loss in the worst {tail} of months")
    if m == "risk_parity":
        return (f"minimize over {W}: &nbsp;∑<sub><i>i</i></sub> (<i>RC</i><sub><i>i</i></sub> − 1/<i>n</i>)<sup>2</sup>, "
                f"&nbsp;<i>RC</i><sub><i>i</i></sub> = {WI}({COV}<i>w</i>)<sub><i>i</i></sub> / {VAR}",
                "Make every ETF contribute an equal share of the portfolio's total risk.",
                "spreads the risk most evenly across your ETFs")
    if m == "max_diversification":
        return (f"maximize over {W}: &nbsp;(∑<sub><i>i</i></sub> σ<sub><i>i</i></sub> {WI}) / {VOL}",
                "Make the portfolio as diversified as possible: the ETFs' individual ups and downs should "
                "cancel each other out as much as they can.",
                "is the most diversified")
    return (escape(m), escape(m), escape(m))


def describe_optimization(req: Request, risk: RiskAssessment, est: Estimates, tax: TaxContext,
                          port: Portfolio, sim: SimulationResult, n_starts: int,
                          cvar_alpha: float = 0.95) -> dict:
    rc = port.constraints
    w = np.asarray(port.weights, float)
    n = len(w)
    lo, hi = rc.lower(), rc.upper()
    vol = port.volatility
    goal = req.has_target
    gs = port.goal_search or {}
    infeasible = bool(gs.get("infeasible"))
    after_tax = tax.mu_after_tax is not None
    obj_math, obj_plain, obj_phrase = _objective(port, req, cvar_alpha)
    if goal and infeasible:
        obj_math = f"maximize over {W}: &nbsp;{PROB}"
        obj_plain = "Make the chance of reaching your target as high as possible."
        obj_phrase = "gives the highest chance of reaching your target"
    cons: list[dict] = []

    # 1. budget
    cons.append({"id": "budget", "name": "Fully invested",
                 "math": f"∑<sub><i>i</i>=1..<i>n</i></sub> {WI} = 1",
                 "plain": "All of your money is invested: the ETF shares add up to 100%.",
                 "status": f"weights add up to {_pct(w.sum())}"})
    # 2. per-ETF position limits
    at_max = [t for t, x, u in zip(rc.tickers, w, hi) if abs(abs(x) - u) < TOL]
    at_min = [t for t, x, lo_i in zip(rc.tickers, w, lo) if lo_i > 0 and abs(x - lo_i) < TOL]
    status = ", ".join(filter(None, [f"at its maximum: {', '.join(at_max)}" if at_max else "",
                                     f"at its minimum: {', '.join(at_min)}" if at_min else ""])) \
        or "every ETF strictly inside its limits"
    if rc.allow_short:
        pos_math = (f"|{WI}| ≤ <i>u</i><sub><i>i</i></sub> &nbsp;and&nbsp; {WI} ≥ <i>l</i><sub><i>i</i></sub> "
                    "for ETFs with a minimum <i>l</i><sub><i>i</i></sub> &gt; 0, &nbsp;for every <i>i</i>")
        pos_plain = ("Each ETF stays within the minimum and maximum share you set (see the table below). "
                     "A short position may not be larger than the ETF's maximum, and an ETF with a positive "
                     "minimum cannot be shorted.")
    else:
        pos_math = f"<i>l</i><sub><i>i</i></sub> ≤ {WI} ≤ <i>u</i><sub><i>i</i></sub> &nbsp;for every ETF <i>i</i>"
        pos_plain = ("Each ETF gets at least its minimum share and at most its maximum share of the "
                     "portfolio (see the table below).")
    cons.append({"id": "positions", "name": "Position limits", "math": pos_math, "plain": pos_plain,
                 "status": status})
    # 3. long-only or gross exposure
    if rc.allow_short:
        gross = float(np.abs(w).sum())
        cons.append({"id": "gross", "name": "Short-sale exposure limit",
                     "math": f"∑<sub><i>i</i></sub> |{WI}| ≤ <i>L</i> = {rc.max_gross_leverage:.2f}",
                     "plain": (f"Short sales are allowed, but the long and short positions together may not "
                               f"exceed {rc.max_gross_leverage:.0%} of your money."),
                     "status": f"total exposure {_pct(gross)}"})
    else:
        cons.append({"id": "no_short", "name": "No short sales",
                     "math": f"{WI} ≥ 0 &nbsp;for every <i>i</i>",
                     "plain": "No ETF can be sold short: every share is zero or positive.",
                     "status": f"smallest weight {_pct(w.min())}"})
    # 4. category limits
    if rc.category_caps:
        idx = {t: i for i, t in enumerate(rc.tickers)}
        used = [(c, float(sum(abs(w[idx[t]]) for t in c.tickers))) for c in rc.category_caps]
        cons.append({"id": "categories", "name": "Category limits",
                     "math": "<br>".join(f"∑<sub><i>i</i> ∈ {escape(c.category)}</sub> |{WI}| ≤ {c.limit:.2f}"
                                         for c in rc.category_caps),
                     "plain": "Some kinds of ETF are limited to a share of the portfolio: "
                              + "; ".join(f"{escape(c.category)} at most {c.limit:.0%}" for c in rc.category_caps) + ".",
                     "status": "; ".join(f"{escape(c.category)} {_pct(u)}" + (" (limit reached)" if u >= c.limit - TOL else "")
                                         for c, u in used)})
    # 5. risk limit from the mapped profile
    binding = vol >= rc.max_volatility - TOL
    risk_plain = (f"The portfolio's volatility, how much its value typically swings in a year, may not exceed "
                  f"{rc.max_volatility:.0%}. This limit comes from your {risk.profile} risk profile (mapped risk "
                  f"score {risk.mapped_score:.0f}, the lower of your capacity and tolerance scores).")
    if port.method == "min_volatility":
        risk_plain += " This method already finds the lowest volatility possible; the run checks it is within the limit."
    cons.append({"id": "risk", "name": "Risk limit",
                 "math": f"{VOL} = √({VAR}) ≤ σ<sub>max</sub> = {rc.max_volatility:.2f}",
                 "plain": risk_plain,
                 "status": f"volatility {_pct(vol)}" + (" (limit fully used)" if binding else "")})
    # 6. method-specific constraints
    if port.method == "target_return":
        r_star = req.client.preferences.target_return
        cons.append({"id": "target_return", "name": "Required return",
                     "math": f"{MU_W} ≥ <i>R</i><sup>*</sup> = {r_star:.4f}",
                     "plain": f"The expected yearly return must be at least the {r_star:.1%} you chose.",
                     "status": f"expected return {_pct(port.expected_return)}"})
    objective_status = ""
    if goal:
        p = req.target_probability
        reached = f"reached in {_pct(sim.prob_target or 0.0)} of {sim.n_paths:,} simulated futures"
        if infeasible:
            obj_plain += (f" (You asked for at least {p:.0%}, but no portfolio within your limits gets there, so "
                          "the requirement became the goal instead of a rule.)")
            objective_status = reached
        else:
            cons.append({"id": "target", "name": "Reach your target",
                         "math": f"{PROB} ≥ <i>p</i> = {p:.2f}",
                         "plain": (f"In at least {p:.0%} of simulated futures your portfolio must be worth "
                                   f"{money(req.target)} or more on {req.target_date:%B %Y}."),
                         "status": reached})

    symbols = [
        (WI, f"the share of your portfolio in ETF <i>i</i>; these {n} numbers are what the optimizer chooses"),
        ("<i>n</i>", f"the number of ETFs you selected ({n})"),
        ("<i>μ</i><sub><i>i</i></sub>", "estimated yearly return of ETF <i>i</i> from its price history"
         + ((" (after tax; a short position pays the pre-tax return)" if rc.allow_short else " (after tax)")
            if after_tax else "")),
        ("∑", "“sum of”: adds up the terms for the ETFs named underneath it"),
        ("σ<sub><i>i</i></sub>", "the volatility of ETF <i>i</i> on its own: the typical size of its yearly ups and downs"),
        ("<i>ρ</i><sub><i>ij</i></sub>", "the correlation between ETFs <i>i</i> and <i>j</i>, from −1 to +1: how closely "
         "they move together (+1 in lockstep, 0 unrelated, −1 opposite)"),
        (COV, f"the covariance matrix: a table with one row and one column per ETF ({n} × {n}), built from the "
              "volatilities and correlations; the entry for ETFs <i>i</i> and <i>j</i> is "
              "σ<sub><i>i</i></sub> σ<sub><i>j</i></sub> <i>ρ</i><sub><i>ij</i></sub>. "
              "It measures how much the ETFs move and how they move together"),
        (VOL, "the portfolio's volatility: the typical size of its yearly ups and downs"),
        ("<i>l</i><sub><i>i</i></sub>, <i>u</i><sub><i>i</i></sub>", "the minimum and maximum share for ETF <i>i</i>"),
        ("σ<sub>max</sub>", "the highest volatility your risk profile allows"),
    ]
    if port.method == "max_sharpe":
        symbols.append(("<i>r</i><sub><i>f</i></sub>", f"the risk-free rate ({_pct(est.risk_free, 2)} per year)"))
    if port.method == "cvar":
        symbols.append(("<i>r</i><sub><i>s</i></sub>", "the ETFs' returns in historical month <i>s</i> (<i>S</i> months in total)"))
        symbols.append(("<i>ζ</i>", "a helper variable; at the optimum it equals the loss threshold of the worst months"))
    if port.method == "risk_parity":
        symbols.append(("<i>RC</i><sub><i>i</i></sub>", "the share of total portfolio risk that comes from ETF <i>i</i>"))
    if rc.allow_short:
        symbols.append(("<i>L</i>", "the most total exposure (long plus short) allowed"))
    if goal:
        symbols += [
            ("<i>F</i><sub><i>T</i></sub>(<i>w</i>)",
             f"the portfolio's value at your target date in one simulated future: it starts with "
             f"{money(req.W0)}, adds {money(req.C)} every month for {req.months} months"
             + (", and pays taxes along the way" if tax.rates.enabled else "")),
            ("<i>F</i><sup>*</sup>", f"your target amount ({money(req.target)})"),
            ("<i>p</i>", f"the chance of reaching the target you require ({req.target_probability:.0%})"),
        ]

    # the process, step by step
    window = f"{est.window_start:%b %Y} to {est.window_end:%b %Y}"
    steps = [
        ("Measure the ETFs",
         f"From daily prices ({window}; up to 20 years, less for younger ETFs) we estimated each ETF's "
         "average yearly return, how much it swings (volatility), and how the ETFs move together "
         "(correlation)." + (" Returns were reduced for the taxes you would pay." if after_tax else "")),
        ("Set the rules",
         f"Your answers set the rules the portfolio must follow: a {rc.max_volatility:.0%} volatility "
         "limit from your risk profile, the position limits for each ETF"
         + (", the category limits" if rc.category_caps else "")
         + (", and the short-sale limit" if rc.allow_short else ", and no short sales") + "."),
    ]
    if goal:
        fr = gs.get("frontier") or []
        steps += [
            ("Line up candidate portfolios",
             f"The optimizer built {len(fr)} candidate portfolios that obey every rule, from the lowest-risk "
             "one to the highest-return one allowed by your risk limit. Each is the lowest-risk way to earn "
             "its level of return (the \"efficient frontier\")."),
            ("Simulate each candidate",
             f"Each candidate was run through {gs.get('search_paths', 0):,} simulated futures with your "
             f"{money(req.W0)} start and {money(req.C)} monthly contributions, counting how often it "
             f"reaches {money(req.target)} by {req.target_date:%B %Y}."),
            ("Pick and double-check",
             (f"It picked the lowest-risk candidate reaching the target in at least {req.target_probability:.0%} "
              f"of futures (with a small safety margin), fine-tuned it between neighbouring candidates, then "
              f"re-checked it on {sim.n_paths:,} fresh simulated futures.")
             if not infeasible else
             (f"No candidate reached the target in {req.target_probability:.0%} of futures, so it picked the "
              "candidate with the highest chance and worked out the monthly contribution that would be needed.")),
        ]
    else:
        solver = ("a linear-programming solver (HiGHS) over the historical months" if port.method == "cvar"
                  else f"a numerical solver (SLSQP), started from {n_starts} different starting mixes so it does "
                       "not get stuck on a second-best answer")
        steps.append(("Search for the best mix",
                      f"Using {solver}, the optimizer searched all the ways to split your money that follow "
                      f"every rule and kept the one that {obj_phrase}."))
        if any("blended" in x for x in port.notes):
            steps.append(("Respect the risk limit",
                          "This method cannot include the volatility limit directly, so its answer was blended "
                          "with the lowest-volatility portfolio just enough to meet the limit."))
    steps.append(("Independent check",
                  "A separate reviewer, using its own code, re-checked every rule above"
                  + (" and re-solved the problem to confirm no better portfolio was missed"
                     if port.method in ("mean_variance", "min_volatility", "max_sharpe")
                     or (port.method == "goal_min_volatility" and not infeasible) else "")
                  + " before this report was produced."))

    rules_plain = [f"<b>{c['name']}.</b> {c['plain']}" for c in cons]
    summary = (f"Think of the optimizer as trying every possible way to divide your money among your {n} "
               f"ETFs. It picks the mix that {obj_phrase}, but only among mixes that follow all of the rules below.")
    if goal and not infeasible:
        summary = (f"Think of the optimizer as trying every possible way to divide your money among your {n} "
                   f"ETFs. Among the mixes that reach {money(req.target)} by {req.target_date:%B %Y} in at least "
                   f"{req.target_probability:.0%} of simulated futures, it picks the one that {obj_phrase}. "
                   "Every mix must also follow the rules below.")

    positions = [{"ticker": t, "min": float(a), "weight": float(x), "max": float(b)}
                 for t, a, x, b in zip(rc.tickers, lo, w, hi)]
    return {"method": port.method, "objective": {"math": obj_math, "plain": obj_plain, "status": objective_status},
            "constraints": cons, "rules_plain": rules_plain, "summary": summary, "steps": steps,
            "symbols": symbols, "positions": positions, "notes": list(port.notes)}
