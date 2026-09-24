"""Independent reviewer agent.

Validates data and computations and checks that the spec's rules are followed, at four
checkpoints of the advisory graph:

    data       after data validation, before estimation (look-ahead, coverage, consistency)
    inputs     after estimation / tax adjustment / risk profiling
    portfolio  after optimization (can trigger optimizer remediation)
    final      after simulation, scenarios, benchmark, projection and explanation

Every finding cites the spec section it enforces. BLOCKER findings halt the workflow;
WARN findings are surfaced in the report. The reviewer recomputes key figures with its own
code (``independent.py``) rather than trusting the production modules.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import Settings
from ..models import ReviewFinding, ReviewReport
from ..universe import CATALOG, RISK_NOTE_PREFIX
from . import independent as ind

REQUIRED_BENCH_METRICS = [
    "Cumulative return", "Annualized return", "Annualized volatility", "Sharpe ratio",
    "Maximum drawdown", "Ending wealth (initial investment only)",
    "Ending wealth (with monthly contributions)", "Best year", "Worst year"]


class Reviewer:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cfg = settings.review

    # ------------------------------------------------------------------ helpers
    def _income_rate(self, t: str) -> float:
        """Distribution tax rate from the catalog's income character (independent of tax.py)."""
        tc, ti = self.s.tax, self._tax_in
        g = lambda v, d: d if v is None else v          # noqa: E731
        state = g(ti.state_rate, tc.state_rate)
        ordinary = g(ti.ordinary_rate, tc.ordinary_rate) + state
        qual = g(ti.qualified_dividend_rate, tc.qualified_dividend_rate) + state
        info = CATALOG[t]
        if info.income_type == "interest":
            return ordinary
        if info.income_type == "none":
            return 0.0
        return info.qualified_fraction * qual + (1 - info.qualified_fraction) * ordinary

    @staticmethod
    def _f(out: list, rule: str, spec: str, sev: str, ok: bool, msg: str, **ev: Any) -> None:
        out.append(ReviewFinding(rule, spec, sev, bool(ok), msg, {k: _plain(v) for k, v in ev.items()}))

    # ------------------------------------------------------------------ stage: data
    def review_data(self, st: Mapping[str, Any]) -> ReviewReport:
        """Runs before estimation so data problems surface as findings, not crashes."""
        out: list[ReviewFinding] = []
        f = self._f
        req, market, dq = st["request"], st["market"], st["data_quality"]
        # universe: only ETFs offered in the configured categories
        allowed = set(self.s.universe.tickers)
        f(out, "R-UNIV-01", "§3", "BLOCKER", set(req.tickers) <= allowed and set(req.tickers) <= set(market.frames),
          f"optimizer universe {req.tickers} is the client's selection from the offered ETF categories")
        # data (spec §3, §6)
        late = {t: str(df.index[-1].date()) for t, df in market.frames.items()
                if len(df) and df.index[-1].date() > req.as_of}
        f(out, "R-DATA-01", "§6", "BLOCKER", not late,
          "no observation after the as-of date (no look-ahead)" if not late else f"look-ahead data: {late}")
        exp_start = _years_before(req.as_of, self.s.data.lookback_years)
        f(out, "R-DATA-02", "§3/§6", "BLOCKER", abs((req.window_start - exp_start).days) <= 3,
          f"estimation window starts {req.window_start} (expected {exp_start}: most recent "
          f"{self.s.data.lookback_years} years)")
        unflagged = [t for t, q in dq.tickers.items() if not q.meets_min_history
                     and not any(w.startswith(f"{t}:") and "history" in w for w in dq.warnings)]
        f(out, "R-DATA-03", "§3", "BLOCKER", not unflagged,
          "ETFs with < 20 years of history are flagged (maximum available history used)"
          if not unflagged else f"short histories not flagged: {unflagged}")
        f(out, "R-DATA-04", "§3", "BLOCKER", not dq.blocking,
          "data validation passed" if not dq.blocking else "; ".join(dq.blocking))
        bad_adj = {}
        for t in req.tickers:
            df = market.frames.get(t)
            if df is None or len(df) < 2:
                continue                      # reported by R-UNIV-01 / R-DATA-04
            adj_r = df["adj_close"].pct_change().to_numpy()[1:]
            tri = ind.tr_index_from_raw(df).to_numpy()
            raw_r = np.diff(tri) / tri[:-1]
            gap = np.abs(adj_r - raw_r)
            err = float(np.nanmax(gap)) if len(raw_r) else 0.0
            n_bad = int(np.nansum(gap > self.s.data.adj_consistency_tol))
            d = self.s.data
            if err > d.adj_split_error or n_bad > max(d.adj_max_bad_days, d.adj_max_bad_fraction * len(df)):
                bad_adj[t] = {"max_err": round(err, 4), "bad_days": n_bad}
        f(out, "R-DATA-05", "§3", "BLOCKER", not bad_adj,
          "adjusted prices consistent with raw close, splits and distributions (independent check)"
          if not bad_adj else f"inconsistent adjusted prices: {bad_adj}")
        no_er = [t for t in req.tickers if CATALOG.get(t) is None or CATALOG[t].expense_ratio is None]
        f(out, "R-DATA-06", "§3", "WARN", not no_er, "expense ratios available for all ETFs"
          if not no_er else f"missing expense ratio: {no_er}")
        return ReviewReport("data", out)

    # ------------------------------------------------------------------ stage: inputs
    def review_inputs(self, st: Mapping[str, Any]) -> ReviewReport:
        out: list[ReviewFinding] = []
        f = self._f
        req, risk, market, dq, est, tax, rc = (st["request"], st["risk"], st["market"], st["data_quality"],
                                               st["estimates"], st["tax"], st["constraints"])
        qs = self.s.questionnaire
        # risk profiling (spec §2)
        cap_answers = {**risk.derived_answers, **req.client.capacity_answers}
        cap = ind.weighted_score({q.id: q.options for q in qs.capacity}, {q.id: q.weight for q in qs.capacity}, cap_answers)
        tol = ind.weighted_score({q.id: q.options for q in qs.tolerance}, {q.id: q.weight for q in qs.tolerance},
                                 req.client.tolerance_answers)
        f(out, "R-RISK-01", "§2A/§2C", "BLOCKER", abs(cap - risk.capacity_score) < 1e-9,
          f"capacity score recomputed {cap} vs reported {risk.capacity_score}", reviewer=cap, reported=risk.capacity_score)
        f(out, "R-RISK-02", "§2B/§2C", "BLOCKER", abs(tol - risk.tolerance_score) < 1e-9,
          f"tolerance score recomputed {tol} vs reported {risk.tolerance_score}", reviewer=tol, reported=risk.tolerance_score)
        f(out, "R-RISK-03", "§2C", "BLOCKER", risk.mapped_score == min(cap, tol),
          f"mapped score {risk.mapped_score} = min(capacity {cap}, tolerance {tol})")
        band = next(b for b in self.s.risk_bands if risk.mapped_score <= b.max_score)
        f(out, "R-RISK-04", "§2C", "BLOCKER",
          band.profile == risk.profile and abs(band.max_volatility - risk.max_volatility) < 1e-12,
          f"score {risk.mapped_score} maps to {band.profile} / {band.max_volatility:.0%} (reported "
          f"{risk.profile} / {risk.max_volatility:.0%})")
        f(out, "R-RISK-05", "§2", "BLOCKER",
          set(risk.capacity_detail) == {q.id for q in qs.capacity}
          and set(risk.tolerance_detail) == {q.id for q in qs.tolerance},
          "capacity and tolerance measured separately with every configured question answered")
        # estimates (spec §6)
        f(out, "R-EST-00", "§3/§6", "BLOCKER", est.tickers == req.tickers and est.window_end <= req.as_of,
          "estimates cover exactly the selected ETFs and end at the as-of date")
        frames = {t: market.frames[t][market.frames[t].index >= np.datetime64(req.window_start)] for t in req.tickers}
        mu_r, sig_r = ind.mu_sigma(frames, req.tickers, self.s.estimation.trading_days, self.s.estimation.mu_shrinkage,
                                req.as_of)
        dmu = np.abs(mu_r - est.mu)
        dsig = np.abs(sig_r / est.sigma - 1)
        f(out, "R-EST-01", "§6", "BLOCKER", float(dmu.max()) <= self.cfg.mu_abs_tol,
          f"expected returns match independent recomputation (max abs diff {dmu.max():.2e})",
          worst=req.tickers[int(dmu.argmax())])
        f(out, "R-EST-02", "§6", "BLOCKER", float(dsig.max()) <= self.cfg.sigma_rel_tol,
          f"volatilities match independent recomputation (max rel diff {dsig.max():.2%})",
          worst=req.tickers[int(dsig.argmax())])
        cov = est.cov
        f(out, "R-EST-03", "§6", "BLOCKER",
          np.allclose(cov, cov.T) and ind.is_psd(cov) and np.allclose(np.sqrt(np.diag(cov)), est.sigma, rtol=1e-6),
          "covariance matrix symmetric, positive semi-definite and consistent with volatilities")
        f(out, "R-EST-04", "§6", "WARN", 0.0 <= est.risk_free <= 0.15,
          f"risk-free rate {est.risk_free:.2%} plausible")
        # taxes (spec §5)
        if req.client.taxes.enabled:
            ok = tax.mu_after_tax is not None and bool(np.all(tax.mu_after_tax <= est.mu + 1e-12))
            f(out, "R-TAX-01", "§5", "BLOCKER", ok, "after-tax expected returns computed and <= pre-tax")
        else:
            f(out, "R-TAX-01", "§5", "BLOCKER", tax.mu_after_tax is None and not tax.rates.enabled,
              "taxes disabled: optimizer uses pre-tax returns")
        # constraints (spec §4)
        c = req.client.constraints
        exp_pos = c.max_position if c.max_position is not None else self.s.optimization.max_position
        exp_L = (c.max_gross_leverage if c.max_gross_leverage is not None else self.s.optimization.max_gross_leverage) if c.allow_short else 1.0
        f(out, "R-CON-01", "§4/§8", "BLOCKER",
          rc.max_volatility == risk.max_volatility and rc.allow_short == c.allow_short
          and rc.max_position == exp_pos and rc.max_gross_leverage == exp_L,
          f"constraints: max vol {rc.max_volatility:.0%} (from mapped profile), short={rc.allow_short}, "
          f"max position {rc.max_position:.0%}, gross <= {rc.max_gross_leverage:.2f}")
        return ReviewReport("inputs", out)

    # ------------------------------------------------------------------ stage: portfolio
    def _portfolio_rules(self, st: Mapping[str, Any]) -> list[ReviewFinding]:
        out: list[ReviewFinding] = []
        f = self._f
        req, est, tax, rc, port = st["request"], st["estimates"], st["tax"], st["constraints"], st["portfolio"]
        tol = self.s.optimization.tolerance
        w = np.asarray(port.weights, float)
        f(out, "R-PORT-05", "§3", "BLOCKER", port.tickers == req.tickers and len(w) == len(req.tickers)
          and np.all(np.isfinite(w)), "weights cover exactly the client-selected ETFs")
        f(out, "R-PORT-01", "§4", "BLOCKER", abs(w.sum() - 1) <= tol, f"weights sum to {w.sum():.6f} (= 1)")
        if not rc.allow_short:
            f(out, "R-PORT-02", "§4", "BLOCKER", w.min() >= -tol, f"long-only: min weight {w.min():.6f} >= 0")
        else:
            f(out, "R-PORT-04", "§4", "BLOCKER", np.abs(w).sum() <= rc.max_gross_leverage + tol,
              f"gross exposure {np.abs(w).sum():.4f} <= L = {rc.max_gross_leverage:.2f}")
        f(out, "R-PORT-03", "§4", "BLOCKER", np.abs(w).max() <= rc.max_position + tol,
          f"largest position {np.abs(w).max():.2%} <= max position {rc.max_position:.0%}")
        vol = float(np.sqrt(w @ est.cov @ w))
        f(out, "R-PORT-06", "§2C/§7/§8", "BLOCKER", vol <= rc.max_volatility + tol,
          f"portfolio volatility {vol:.2%} <= limit {rc.max_volatility:.0%} of the mapped risk profile")
        mu_opt = tax.mu_after_tax if tax.mu_after_tax is not None else est.mu
        # long legs earn the optimizer's (after-tax) mu; short legs pay the pre-tax mu
        ret = float(np.clip(w, 0, None) @ mu_opt - np.clip(-w, 0, None) @ est.mu)
        f(out, "R-PORT-07", "§9", "BLOCKER",
          abs(port.expected_return - ret) < 1e-9 and abs(port.volatility - vol) < 1e-9
          and abs(port.expected_return_pretax - w @ est.mu) < 1e-9,
          f"E[Rp] = w'mu = {ret:.4%}, sigma_p = sqrt(w' Sigma w) = {vol:.4%} reproduce reported values")
        f(out, "R-PORT-11", "§5", "BLOCKER", (tax.mu_after_tax is not None) == bool(req.client.taxes.enabled),
          "optimizer return inputs are after-tax exactly when taxes are enabled")
        ia, ma = port.initial_allocation, port.monthly_allocation
        rtol = self.cfg.wealth_rel_tol
        ok = (abs(sum(ia.values()) - req.W0) <= rtol * max(req.W0, 1) and abs(sum(ma.values()) - req.C) <= rtol * max(req.C, 1)
              and all(abs(ia[t] - wi * req.W0) <= 1e-6 * max(req.W0, 1) and abs(ma[t] - wi * req.C) <= 1e-6 * max(req.C, 1)
                      for t, wi in zip(port.tickers, w)))
        f(out, "R-PORT-08", "§10", "BLOCKER", ok,
          f"dollar allocations reconcile: initial {sum(ia.values()):,.2f} = {req.W0:,.2f}, "
          f"monthly {sum(ma.values()):,.2f} = {req.C:,.2f}")
        if req.has_target:
            f(out, "R-PORT-10", "§7", "BLOCKER", port.case == "target" and port.method.startswith("goal_")
              and port.goal_search is not None,
              f"target client: risk minimized subject to the target-probability constraint ({port.method})")
        else:
            f(out, "R-PORT-10", "§8", "BLOCKER", port.case == "no_target" and port.method == req.method,
              f"no-target client: {port.method} subject to the mapped risk limit")
        f(out, "R-PORT-12", "§9", "WARN", bool(port.method), "optimization method disclosed")
        return out

    def review_portfolio(self, st: Mapping[str, Any]) -> ReviewReport:
        return ReviewReport("portfolio", self._portfolio_rules(st) + self._optimality_rules(st))

    def _optimality_rules(self, st: Mapping[str, Any]) -> list[ReviewFinding]:
        """Independent re-solves: the recommended portfolio must actually optimize the stated
        objective (Case B) or be the lowest-risk portfolio meeting the target (Case A)."""
        out: list[ReviewFinding] = []
        f = self._f
        req, est, tax, rc, port = st["request"], st["estimates"], st["tax"], st["constraints"], st["portfolio"]
        w = np.asarray(port.weights, float)
        mu_l = tax.mu_after_tax if tax.mu_after_tax is not None else est.mu
        args = (mu_l, est.mu, est.cov, rc.allow_short, rc.max_position, rc.max_gross_leverage, rc.max_volatility)
        vol = float(np.sqrt(w @ est.cov @ w))
        if not req.has_target:
            kind = {"mean_variance": "max_return", "min_volatility": "min_vol", "max_sharpe": "max_sharpe"}.get(port.method)
            if kind is None:
                f(out, "R-PORT-09", "§8/§9", "INFO", True,
                  f"{port.method}: no independent re-solve (constraint and risk-limit rules still apply)")
                return out
            _, best = ind.resolve(kind, *args, rf=est.risk_free)
            if kind == "max_return":
                val, tol, ok_fn, unit = port.expected_return, self.cfg.optimality_tol, lambda b, v, t: v >= b - t, "return"
            elif kind == "max_sharpe":
                val, tol, ok_fn, unit = port.sharpe, 0.01, lambda b, v, t: v >= b - t, "Sharpe"
            else:
                val, tol, ok_fn, unit = vol, 1e-4, lambda b, v, t: v <= b + t, "volatility"
            ok = bool(np.isnan(best)) or ok_fn(best, val, tol)
            f(out, "R-PORT-09", "§8/§9", "BLOCKER", ok,
              f"independent re-solve: optimizer {unit} {val:.4f} vs independent {best:.4f} (tolerance {tol})")
            return out
        gs = port.goal_search or {}
        if req.goal_risk_metric != "volatility" or gs.get("infeasible"):
            f(out, "R-PORT-13", "§7", "INFO", True, "goal minimality check applies to the volatility metric "
              "with a feasible target (see R-SIM-04 for the infeasible case)")
            return out
        # candidate lower-risk portfolios on the reviewer's own frontier, scored by its own MC
        rb = req.rebalancing
        every = {"monthly": 1, "quarterly": 3, "annual": 12}[rb.frequency] if rb.type == "calendar" else None
        thr = rb.threshold if rb.type == "threshold" else None
        w_mv, v_mv = ind.resolve("min_vol", *args)
        r_mv = float(np.clip(w_mv, 0, None) @ mu_l - np.clip(-w_mv, 0, None) @ est.mu)
        n, p, z = self.cfg.mc_paths, req.target_probability, self.cfg.mc_z
        se = np.sqrt(p * (1 - p) / n)
        offenders = []
        for r in np.linspace(r_mv, port.expected_return, 6)[:-1]:
            wc, vc = (w_mv, v_mv) if r <= r_mv + 1e-12 else ind.resolve("min_vol", *args, min_return=r)
            if wc is None or vc > 0.9 * vol:
                continue
            # with taxes on, drift at the after-tax return (the engine taxes each path explicitly)
            term = ind.mc_terminal(wc, mu_l, est.cov, req.W0, req.C, req.months, every, thr, n, 991)
            pc = float((term >= req.target).mean())
            if pc >= p + z * se:
                offenders.append((round(vc, 4), round(pc, 3)))
        sev = "BLOCKER" if not req.client.taxes.enabled else "WARN"   # after-tax drift is an approximation
        f(out, "R-PORT-13", "§7", sev, not offenders,
          f"no portfolio with >=10% lower volatility than {vol:.2%} reaches the target probability "
          f"{p:.0%} (independent frontier + MC)" if not offenders else
          f"lower-risk portfolios also meet the target (vol, P): {offenders}")
        return out

    # ------------------------------------------------------------------ stage: final
    def review_final(self, st: Mapping[str, Any]) -> ReviewReport:
        out: list[ReviewFinding] = self._portfolio_rules(st)
        f = self._f
        req, est, port, sim = st["request"], st["estimates"], st["portfolio"], st["simulation"]
        self._tax_in = req.client.taxes
        proj, bench, scen, exp, market, tax = (st["projection"], st["benchmark"], st["scenarios"],
                                               st["explanation"], st["market"], st["tax"])
        z = self.cfg.mc_z
        f(out, "R-SIM-01", "§12", "BLOCKER", sim.n_paths == self.s.simulation.n_paths and sim.months == req.months,
          f"{sim.n_paths:,} simulated paths over {sim.months} months")
        pv = [sim.percentiles.get(p) for p in (10, 25, 50, 75, 90)]
        f(out, "R-SIM-02", "§12", "BLOCKER", None not in pv and all(a <= b for a, b in zip(pv, pv[1:])),
          "10/25/50/75/90th percentiles reported and monotone")
        term = sim.terminal_after_liquidation if sim.terminal_after_liquidation is not None else sim.terminal
        contributed = req.W0 + req.C * req.months
        f(out, "R-SIM-06", "§12", "BLOCKER", abs(sim.prob_loss_principal - float((term < contributed).mean())) < 1e-12
          and abs(sim.total_contributed - contributed) < 1e-6,
          f"probability of losing principal ({sim.prob_loss_principal:.2%}) recounted against "
          f"total contributions {contributed:,.0f}")
        if req.has_target:
            recount = float((term >= req.target).mean())
            f(out, "R-SIM-03", "§12", "BLOCKER", sim.prob_target is not None and abs(sim.prob_target - recount) < 1e-12,
              f"probability of achieving target = share of paths with terminal wealth >= target ({recount:.2%})")
            p = req.target_probability
            se = np.sqrt(p * (1 - p) / sim.n_paths)
            infeasible = bool((port.goal_search or {}).get("infeasible"))
            ok = (sim.prob_target or 0) >= p - z * se or (infeasible and "No portfolio" in exp.goal_text)
            f(out, "R-SIM-04", "§7", "BLOCKER", ok,
              f"target probability {sim.prob_target:.2%} meets the {p:.0%} requirement"
              if not infeasible else "target unattainable within the risk limit: disclosed with the best "
              "achievable probability and the required contribution")
        # independent Monte Carlo
        w = np.asarray(port.weights, float)
        rb = req.rebalancing
        every = {"monthly": 1, "quarterly": 3, "annual": 12}[rb.frequency] if rb.type == "calendar" else None
        hist = None
        if sim.distribution == "bootstrap":
            h = est.monthly_returns.to_numpy()
            hist = h[~np.isnan(h).any(axis=1)]
        n = self.cfg.mc_paths
        rt = ind.mc_terminal(w, est.mu, est.cov, req.W0, req.C, req.months, every,
                             rb.threshold if rb.type == "threshold" else None, n, 777, hist)
        med_r, med_p = float(np.median(rt)), float(np.median(term))
        if not req.client.taxes.enabled:
            if req.has_target:
                pr = float((rt >= req.target).mean())
                pp = sim.prob_target or 0.0
                se = np.sqrt(max(pr * (1 - pr), 1e-4) / n + max(pp * (1 - pp), 1e-4) / sim.n_paths)
                f(out, "R-SIM-05", "§12", "BLOCKER", abs(pr - pp) <= z * se,
                  f"independent MC probability {pr:.2%} vs engine {pp:.2%} (tolerance {z * se:.2%})")
            f(out, "R-SIM-07", "§12", "WARN", abs(med_p / med_r - 1) <= 0.04,
              f"independent MC median terminal wealth {med_r:,.0f} vs engine {med_p:,.0f}")
        else:
            f(out, "R-SIM-05", "§5/§12", "BLOCKER", med_p <= med_r * 1.02,
              f"after-tax median terminal wealth {med_p:,.0f} does not exceed independent pre-tax median {med_r:,.0f}")
            # lower bound: tax on distributions alone along the median wealth path
            y = est.income_yield
            inc_rate = np.array([self._income_rate(t) for t in port.tickers])
            wl = np.clip(w, 0, None)
            monthly_rate = float(wl @ (y * inc_rate)) / 12
            bound = float(sim.band["p50"].iloc[:-1].sum() * monthly_rate) if sim.band is not None else 0.0
            f(out, "R-TAX-02", "§5", "BLOCKER", bound < 100 or sim.taxes_paid_median >= 0.6 * bound,
              f"median taxes paid {sim.taxes_paid_median:,.0f} >= 60% of the independent distribution-tax "
              f"estimate {bound:,.0f} (capital-gains tax comes on top)")
        # deterministic projection (spec §11)
        r = port.expected_return / 12
        fv = ind.fv_by_recursion(req.W0, req.C, r, req.months)
        f(out, "R-PROJ-01", "§11", "BLOCKER", abs(proj.fv_nominal / fv - 1) <= 1e-9 and proj.months == req.months,
          f"FV_T = W0(1+r)^T + C[((1+r)^T-1)/r] = {proj.fv_nominal:,.2f} matches month-by-month recursion {fv:,.2f}")
        # scenarios (spec §13)
        names = [s.name for s in scen]
        f(out, "R-SCN-01", "§13", "BLOCKER", names == ["conservative", "base", "optimistic"],
          "conservative, base and optimistic scenarios provided")
        meds = [s.median for s in scen]
        f(out, "R-SCN-02", "§13", "WARN", meds[0] <= meds[1] <= meds[2],
          f"scenario medians ordered: {meds[0]:,.0f} <= {meds[1]:,.0f} <= {meds[2]:,.0f}")
        # benchmark (spec §14)
        m = bench.metrics
        f(out, "R-BM-04", "§14", "BLOCKER", all(k in m.index for k in REQUIRED_BENCH_METRICS),
          "all required benchmark metrics reported")
        n_months = len(bench.growth_of_10k) - 1
        exp_contrib = req.W0 + req.C * n_months
        same = (bench.initial_investment == req.W0 and bench.monthly_contribution == req.C
                and all(abs(float(m.loc["Total contributed", c]) - exp_contrib) < 1e-6 for c in m.columns)
                and abs(float(bench.wealth_with_contributions["Contributed"].iloc[-1]) - exp_contrib) < 1e-6)
        f(out, "R-BM-01", "§14", "BLOCKER", same,
          f"both sides use W0 {req.W0:,.0f} and C {req.C:,.0f} over the same {n_months} months "
          f"(total contributed {exp_contrib:,.0f})")
        yrs_ok = bench.years >= self.s.benchmark.years - 1 / 12 or bool(bench.notes)
        f(out, "R-BM-02", "§14", "BLOCKER", yrs_ok and bench.end <= req.as_of,
          f"comparison period {bench.start} to {bench.end} ({bench.years:.1f} years)" +
          ("" if bench.years >= self.s.benchmark.years - 1 / 12 else " - shortened and disclosed"))
        # independent month-end backtests of BOTH sides from raw prices
        tol = self.cfg.backtest_rel_tol
        first = bench.growth_of_10k.index[0].to_timestamp(how="start").date()
        rb = req.rebalancing
        every = {"monthly": 1, "quarterly": 3, "annual": 12}[rb.frequency] if rb.type == "calendar" else None
        thr = rb.threshold if rb.type == "threshold" else None

        def close(a: float, b: float) -> bool:
            return abs(a - b) <= tol * max(abs(b), 1.0)

        bm_t = self.s.data.benchmark
        me = ind.month_end_series(market.frames[bm_t], first, bench.end)
        lump, contrib = ind.single_asset_wealth(me, req.W0, req.C)
        el = float(m.loc["Ending wealth (initial investment only)", "S&P 500"])
        ec = float(m.loc["Ending wealth (with monthly contributions)", "S&P 500"])
        f(out, "R-BM-03", "§14", "BLOCKER", close(el, lump) and close(ec, contrib),
          f"S&P 500 ending wealth recomputed from raw prices: {lump:,.0f} / {contrib:,.0f} "
          f"(reported {el:,.0f} / {ec:,.0f})")
        held = [(t, x) for t, x in zip(port.tickers, port.weights) if abs(x) > 1e-9]
        mes = pd.concat({t: ind.month_end_series(market.frames[t], first, bench.end) for t, _ in held}, axis=1)
        pl, pc = ind.portfolio_wealth(mes, np.array([x for _, x in held]), req.W0, req.C, every, thr)
        rl = float(m.loc["Ending wealth (initial investment only)", "Portfolio"])
        rc_ = float(m.loc["Ending wealth (with monthly contributions)", "Portfolio"])
        f(out, "R-BM-05", "§14", "BLOCKER", close(rl, pl) and close(rc_, pc),
          f"portfolio ending wealth recomputed from raw prices with the same rebalancing rule: "
          f"{pl:,.0f} / {pc:,.0f} (reported {rl:,.0f} / {rc_:,.0f})")
        # explanation & disclosures (spec §2C, §6, §13, §15, §16)
        rk = st["risk"]
        phrases = [f"risk-capacity score is {rk.capacity_score:.0f}", f"risk-tolerance score is {rk.tolerance_score:.0f}",
                   f"mapped risk score is therefore {rk.mapped_score:.0f}"]
        f(out, "R-EXP-01", "§2C/§16", "BLOCKER", all(ph in exp.risk_text for ph in phrases),
          "capacity, tolerance and mapped risk scores displayed separately")
        disc = " ".join(exp.disclosures)
        f(out, "R-EXP-02", "§6", "BLOCKER", "historical estimates" in disc,
          "expected returns labelled as historical estimates / model assumptions")
        f(out, "R-EXP-03", "§13", "BLOCKER", "not forecasts" in disc, "scenarios distinguished from forecasts")
        if req.client.taxes.enabled:
            f(out, "R-EXP-04", "§5", "BLOCKER", tax.disclaimer is not None and tax.disclaimer in disc,
              "tax module presented as an estimated model, not individualized tax advice")
        if market.synthetic:
            f(out, "R-EXP-05", "data", "BLOCKER", "SYNTHETIC" in disc, "synthetic data watermark present")
        special = [t for t, x in zip(port.tickers, port.weights) if abs(x) > 1e-6 and CATALOG[t].risk_note]
        if special:
            f(out, "R-EXP-07", "§16", "BLOCKER",
              all(any(d.startswith(f"{RISK_NOTE_PREFIX} {t} ") for d in exp.disclosures) for t in special),
              f"special risks disclosed for held {', '.join(special)} (leveraged / option-income / crypto)")
        f(out, "R-EXP-06", "§16", "BLOCKER", bool(exp.methodology and exp.assumptions and exp.limitations
                                                  and exp.calculations),
          "methodology, assumptions, limitations and underlying calculations provided")
        return ReviewReport("final", out)


def _years_before(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:            # 29 February
        return d.replace(year=d.year - years, day=28)


def _plain(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
