"""Independent reviewer agent.

Validates data and computations and checks that the spec's rules are followed, at three
checkpoints of the advisory graph:

    inputs     after data validation / estimation / risk profiling
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

from ..config import Settings
from ..models import ReviewFinding, ReviewReport
from ..universe import CATALOG
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
    @staticmethod
    def _f(out: list, rule: str, spec: str, sev: str, ok: bool, msg: str, **ev: Any) -> None:
        out.append(ReviewFinding(rule, spec, sev, bool(ok), msg, {k: _plain(v) for k, v in ev.items()}))

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
        # universe (spec §3)
        allowed = set(self.s.universe.default) | set(self.s.universe.optional_extra)
        f(out, "R-UNIV-01", "§3", "BLOCKER", set(req.tickers) <= allowed and est.tickers == req.tickers,
          f"optimizer universe {req.tickers} is the client's selection within the allowed universe")
        # data (spec §3, §6)
        late = {t: str(df.index[-1].date()) for t, df in market.frames.items()
                if len(df) and df.index[-1].date() > req.as_of}
        f(out, "R-DATA-01", "§6", "BLOCKER", not late and est.window_end <= req.as_of,
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
            df = market.frames[t]
            adj_r = df["adj_close"].pct_change().to_numpy()[1:]
            raw_r = np.diff(ind.tr_index_from_raw(df).to_numpy()) / ind.tr_index_from_raw(df).to_numpy()[:-1]
            err = float(np.nanmax(np.abs(adj_r - raw_r))) if len(raw_r) else 0.0
            if err > self.s.data.adj_consistency_tol:
                bad_adj[t] = err
        f(out, "R-DATA-05", "§3", "BLOCKER", not bad_adj,
          "adjusted prices consistent with raw close, splits and distributions (independent check)"
          if not bad_adj else f"inconsistent adjusted prices: {bad_adj}")
        no_er = [t for t in req.tickers if CATALOG.get(t) is None or CATALOG[t].expense_ratio is None]
        f(out, "R-DATA-06", "§3", "WARN", not no_er, "expense ratios available for all ETFs"
          if not no_er else f"missing expense ratio: {no_er}")
        # estimates (spec §6)
        frames = {t: market.frames[t][market.frames[t].index >= np.datetime64(req.window_start)] for t in req.tickers}
        mu_r, sig_r = ind.mu_sigma(frames, req.tickers, self.s.estimation.trading_days, self.s.estimation.mu_shrinkage)
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
    def review_portfolio(self, st: Mapping[str, Any]) -> ReviewReport:
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
        f(out, "R-PORT-07", "§9", "BLOCKER",
          abs(port.expected_return - w @ mu_opt) < 1e-9 and abs(port.volatility - vol) < 1e-9
          and abs(port.expected_return_pretax - w @ est.mu) < 1e-9,
          f"E[Rp] = w'mu = {w @ mu_opt:.4%}, sigma_p = sqrt(w' Sigma w) = {vol:.4%} reproduce reported values")
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
        # optimality spot-check against random feasible portfolios (independent search)
        if not req.has_target and not rc.allow_short and port.method in ("mean_variance", "max_sharpe", "min_volatility"):
            rng = np.random.default_rng(12345)
            S = ind.sample_long_only(len(w), rc.max_position, self.cfg.optimality_samples, rng)
            vols = np.sqrt(np.einsum("ij,jk,ik->i", S, est.cov, S))
            rets = S @ mu_opt
            feas = vols <= rc.max_volatility
            if port.method == "mean_variance":
                best = float(rets[feas].max()) if feas.any() else -np.inf
                val = float(w @ mu_opt)
                ok = best <= val + self.cfg.optimality_tol
                msg = f"no sampled feasible portfolio beats the optimizer's return {val:.3%} (best sampled {best:.3%})"
            elif port.method == "max_sharpe":
                sh = (rets - est.risk_free) / vols
                best = float(sh[feas].max()) if feas.any() else -np.inf
                val = (w @ mu_opt - est.risk_free) / vol
                ok = best <= val + 0.02
                msg = f"no sampled feasible portfolio beats the optimizer's Sharpe {val:.3f} (best sampled {best:.3f})"
            else:
                best = float(vols.min())
                ok = best >= vol - 1e-3
                msg = f"no sampled portfolio has lower volatility than {vol:.3%} (lowest sampled {best:.3%})"
            f(out, "R-PORT-09", "§8/§9", "WARN", ok, msg, samples=len(S))
        return ReviewReport("portfolio", out)

    # ------------------------------------------------------------------ stage: final
    def review_final(self, st: Mapping[str, Any]) -> ReviewReport:
        out: list[ReviewFinding] = list(self.review_portfolio(st).findings)
        f = self._f
        req, est, port, sim = st["request"], st["estimates"], st["portfolio"], st["simulation"]
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
        same = (bench.initial_investment == req.W0 and bench.monthly_contribution == req.C
                and m.loc["Total contributed", "Portfolio"] == m.loc["Total contributed", "S&P 500"]
                and len(bench.growth_of_10k["Portfolio"].dropna()) == len(bench.growth_of_10k["S&P 500"].dropna()))
        f(out, "R-BM-01", "§14", "BLOCKER", same,
          "portfolio and S&P 500 use the same initial investment, contributions, period and return convention")
        yrs_ok = bench.years >= self.s.benchmark.years - 1 / 12 or bool(bench.notes)
        f(out, "R-BM-02", "§14", "BLOCKER", yrs_ok and bench.end <= req.as_of,
          f"comparison period {bench.start} to {bench.end} ({bench.years:.1f} years)" +
          ("" if bench.years >= self.s.benchmark.years - 1 / 12 else " - shortened and disclosed"))
        bm_t = self.s.data.benchmark
        start_me = bench.growth_of_10k.index[0].to_timestamp(how="end").date()
        me = ind.month_end_series(market.frames[bm_t], start_me.replace(day=1), bench.end)
        lump, contrib = ind.single_asset_wealth(me, req.W0, req.C)
        el = float(m.loc["Ending wealth (initial investment only)", "S&P 500"])
        ec = float(m.loc["Ending wealth (with monthly contributions)", "S&P 500"])
        f(out, "R-BM-03", "§14", "BLOCKER", abs(el / lump - 1) <= 1e-6 and abs(ec / contrib - 1) <= 1e-6,
          f"S&P 500 ending wealth recomputed from raw prices: {lump:,.0f} / {contrib:,.0f} "
          f"(reported {el:,.0f} / {ec:,.0f})")
        # explanation & disclosures (spec §2C, §6, §13, §15, §16)
        rk = st["risk"]
        f(out, "R-EXP-01", "§2C/§16", "BLOCKER",
          all(f"{v:.0f}" in exp.risk_text for v in (rk.capacity_score, rk.tolerance_score, rk.mapped_score)),
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
        lev = [t for t, x in zip(port.tickers, port.weights) if abs(x) > 1e-6 and CATALOG[t].leveraged]
        if lev:
            f(out, "R-EXP-07", "§16", "BLOCKER",
              all(any(d.startswith("LEVERAGED ETF:") and t in d for d in exp.disclosures) for t in lev),
              f"leveraged-ETF risk disclosed for {', '.join(lev)}")
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
