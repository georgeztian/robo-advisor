"""Portfolio optimization methods (spec §8, §9).

Primary:  mean-variance  max w'mu  s.t.  w' Sigma w <= sigma_max^2   (Case B)
Also:     minimum volatility, maximum Sharpe, target-return minimum volatility,
          CVaR minimization (Rockafellar-Uryasev LP), risk parity, maximum diversification.

Every method honours the client's constraints (long-only or gross-leverage-limited shorts,
max position) and the mapped-risk volatility cap. Methods whose natural formulation cannot
carry the quadratic cap (CVaR LP, risk parity, max diversification) are blended toward the
minimum-volatility portfolio just enough to satisfy it; the feasible set is convex so the
blend keeps every other constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize

from ..models import ResolvedConstraints
from .constraints import InfeasibleError, Space

METHODS = ("mean_variance", "min_volatility", "max_sharpe", "cvar", "target_return",
           "risk_parity", "max_diversification")

# Plain-language descriptions shown to clients (questionnaire, README). Every method keeps the
# portfolio within the client's risk limit and position / category limits.
METHOD_DESCRIPTIONS = {
    "mean_variance": "Highest expected return within your risk limit (recommended; the classic approach)",
    "min_volatility": "Smallest ups and downs possible, whatever the return",
    "max_sharpe": "Best return per unit of risk (Sharpe ratio)",
    "cvar": "Smallest average loss in the worst 5% of months (tail-risk focus)",
    "target_return": "Smallest ups and downs that still earn a return you choose",
    "risk_parity": "Every ETF contributes the same share of total risk (balanced, long-only)",
    "max_diversification": "Most diversified mix: least overlap between the ETFs' movements (long-only)",
}
GOAL_RISK_METRICS = {
    "volatility": "Volatility: how much the portfolio value moves up and down (recommended)",
    "cvar": "Tail risk (CVaR): the average final value in the worst 5% of simulated futures",
}

METHOD_LABELS = {
    "mean_variance": "Mean-variance: maximize expected return subject to the volatility limit",
    "min_volatility": "Minimum volatility",
    "max_sharpe": "Maximum Sharpe ratio (subject to the volatility limit)",
    "cvar": "CVaR minimization (historical monthly scenarios)",
    "target_return": "Target-return minimum volatility",
    "risk_parity": "Risk parity (equal risk contribution)",
    "max_diversification": "Maximum diversification ratio",
}


@dataclass
class OptResult:
    weights: np.ndarray
    method: str
    notes: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class Optimizer:
    def __init__(self, mu: np.ndarray, cov: np.ndarray, rc: ResolvedConstraints, risk_free: float,
                 scenarios: np.ndarray | None = None, n_starts: int = 6, seed: int = 0,
                 cvar_alpha: float = 0.95, mu_short: np.ndarray | None = None,
                 scenarios_short: np.ndarray | None = None):
        """``mu`` is the return earned on long positions (after tax when taxes are enabled).
        ``mu_short`` is the return a short position pays away: pre-tax, since a short cannot
        collect the tax drag of the asset it borrows (defaults to ``mu``)."""
        self.mu, self.cov, self.rc, self.rf = np.asarray(mu), np.asarray(cov), rc, risk_free
        self.mu_short = np.asarray(mu_short) if mu_short is not None else self.mu
        self.scenarios_short = scenarios_short if scenarios_short is not None else scenarios
        self.n = len(mu)
        self.space = Space(self.n, rc)
        self.space.check_feasible()
        self.scenarios = scenarios
        self.n_starts = n_starts
        self.rng = np.random.default_rng(seed)
        self.alpha = cvar_alpha
        self._minvol: np.ndarray | None = None

    # ------------------------------------------------------------------ core helpers
    def vol(self, w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ self.cov @ w, 0.0)))

    def port_return(self, w: np.ndarray) -> float:
        """Expected return: long legs earn mu, short legs pay mu_short."""
        return float(np.clip(w, 0, None) @ self.mu - np.clip(-w, 0, None) @ self.mu_short)

    def _ret_x(self, x: np.ndarray) -> float:
        n = self.n
        return float(x[:n] @ self.mu - x[n:] @ self.mu_short) if self.space.short else float(x @ self.mu)

    def _ret_grad_x(self) -> np.ndarray:
        return np.concatenate([self.mu, -self.mu_short]) if self.space.short else self.mu

    def _wfun(self, fun):
        """Lift a (value, gradient) function of the weights to the solver variables."""
        sp = self.space

        def f(x):
            val, g = fun(sp.to_w(x))
            return val, sp.grad_to_x(g)
        return f

    def _solve(self, fun, extra_cons: list[dict] | None = None, vol_cap: bool = True,
               starts: list[np.ndarray] | None = None) -> np.ndarray:
        sp = self.space
        cons = sp.linear_constraints() + (extra_cons or [])
        if vol_cap:
            cons.append(sp.vol_constraint(self.cov, self.rc.max_volatility))

        best, best_val = None, np.inf
        for x0 in starts or sp.starts(self.n_starts, self.rng):
            res = minimize(fun, x0, jac=True, method="SLSQP", bounds=sp.bounds(), constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-12})
            w = sp.to_w(res.x)
            ok = sp.is_feasible(w, 1e-6, self.cov, self.rc.max_volatility if vol_cap else None)
            ok = ok and all(c["fun"](res.x) >= -1e-7 for c in (extra_cons or []) if c["type"] == "ineq")
            if ok and res.fun < best_val:
                best, best_val = w, res.fun
        if best is None:
            raise InfeasibleError("optimizer found no portfolio satisfying all constraints")
        return self._clean(best)

    def _clean(self, w: np.ndarray) -> np.ndarray:
        w = np.where(np.abs(w) < 1e-7, 0.0, w)
        return w / w.sum()

    def min_vol_portfolio(self) -> np.ndarray:
        if self._minvol is None:
            cov = self.cov
            self._minvol = self._solve(self._wfun(lambda w: (w @ cov @ w, 2 * cov @ w)), vol_cap=False)
        return self._minvol

    def check_risk_limit_feasible(self) -> None:
        w = self.min_vol_portfolio()
        if self.vol(w) > self.rc.max_volatility + 1e-6:
            raise InfeasibleError(
                f"the lowest-volatility portfolio of the selected ETFs has {self.vol(w):.2%} volatility, "
                f"above the {self.rc.max_volatility:.2%} limit of the mapped risk profile; "
                "add lower-risk ETFs (e.g. BIL, BND) to the universe")

    def blend_to_cap(self, w: np.ndarray) -> tuple[np.ndarray, float]:
        """Smallest t in [0,1] with vol((1-t)w + t w_minvol) <= sigma_max."""
        cap = self.rc.max_volatility
        if self.vol(w) <= cap:
            return w, 0.0
        mv = self.min_vol_portfolio()
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.vol((1 - mid) * w + mid * mv) <= cap:
                hi = mid
            else:
                lo = mid
        return (1 - hi) * w + hi * mv, hi

    # ------------------------------------------------------------------ methods
    def mean_variance(self) -> OptResult:
        g = -self._ret_grad_x()
        starts = self.space.starts(self.n_starts, self.rng) + [self.space.from_w(self.min_vol_portfolio())]
        w = self._solve(lambda x: (-self._ret_x(x), g), starts=starts)
        return OptResult(w, "mean_variance")

    def min_volatility(self) -> OptResult:
        return OptResult(self.min_vol_portfolio(), "min_volatility")

    def max_sharpe(self) -> OptResult:
        cov, rf, sp = self.cov, self.rf, self.space
        gr = self._ret_grad_x()

        def f(x):
            w = sp.to_w(x)
            s = np.sqrt(max(w @ cov @ w, 1e-16))
            ex = self._ret_x(x) - rf
            return -ex / s, -(gr / s - ex * sp.grad_to_x(cov @ w) / s**3)

        starts = self.space.starts(self.n_starts, self.rng) + [self.space.from_w(self.min_vol_portfolio())]
        return OptResult(self._solve(f, starts=starts), "max_sharpe")

    def target_return(self, target: float) -> OptResult:
        cov, sp = self.cov, self.space
        gr = self._ret_grad_x()
        con = {"type": "ineq", "fun": lambda x: self._ret_x(x) - target, "jac": lambda x: gr}
        starts = [sp.from_w(self.min_vol_portfolio())] + sp.starts(max(2, self.n_starts // 2), self.rng)
        w = self._solve(self._wfun(lambda w: (w @ cov @ w, 2 * cov @ w)), extra_cons=[con], starts=starts)
        return OptResult(w, "target_return", diagnostics={"target_return": target})

    def cvar(self) -> OptResult:
        if self.scenarios is None or len(self.scenarios) < 24:
            raise InfeasibleError("CVaR optimization needs at least 24 monthly return scenarios")
        R = self.scenarios
        S, n = R.shape
        sp = self.space
        a = self.alpha
        d = sp.dim
        # variables: x (d), zeta (1), u (S)
        c = np.concatenate([np.zeros(d), [1.0], np.full(S, 1 / ((1 - a) * S))])
        Rx = np.hstack([R, -self.scenarios_short]) if sp.short else R
        A_ub = np.hstack([-Rx, -np.ones((S, 1)), -np.eye(S)])
        b_ub = np.zeros(S)
        if sp.short:
            A_ub = np.vstack([A_ub, np.concatenate([np.ones(d), [0.0], np.zeros(S)])])
            b_ub = np.append(b_ub, self.rc.max_gross_leverage)
        for _, lim, m in sp.cap_groups():
            g = np.concatenate([m, m]) if sp.short else m
            A_ub = np.vstack([A_ub, np.concatenate([g, [0.0], np.zeros(S)])])
            b_ub = np.append(b_ub, lim)
        sum_row = np.concatenate([np.ones(n), -np.ones(n)]) if sp.short else np.ones(n)
        A_eq = np.concatenate([sum_row, [0.0], np.zeros(S)])[None, :]
        bounds = sp.bounds() + [(None, None)] + [(0, None)] * S
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
        if not res.success:
            raise InfeasibleError(f"CVaR LP failed: {res.message}")
        w = self._clean(sp.to_w(res.x[:d]))
        w, t = self.blend_to_cap(w)
        notes = [f"blended {t:.0%} toward minimum-volatility to satisfy the volatility limit"] if t > 0 else []
        return OptResult(w, "cvar", notes, {"monthly_cvar": float(res.fun), "alpha": a, "scenarios": S})

    def _cap_cons_long(self) -> list[dict]:
        """Category limits for the long-only formulations (risk parity, max diversification)."""
        return [{"type": "ineq", "fun": lambda w, m=m, lim=lim: lim - m @ w, "jac": lambda w, m=m: -m}
                for _, lim, m in self.space.cap_groups()]

    def _long_only_space_note(self, w: np.ndarray) -> list[str]:
        if not self.rc.allow_short:
            return []
        if (w < -1e-9).any():
            return ["solved long-only (method undefined with short positions); blending toward the "
                    "minimum-volatility portfolio to meet the risk limit introduced short positions"]
        return ["long-only by construction (method undefined with short positions)"]

    def risk_parity(self) -> OptResult:
        cov, n, m = self.cov, self.n, self.rc.max_position

        def f(w):
            sw = cov @ w
            var = w @ sw
            rc = w * sw / var
            diff = rc - 1 / n
            # gradient of sum(diff^2) wrt w
            drc = (np.diag(sw) + w[:, None] * cov) / var - np.outer(rc, 2 * sw) / var
            return float(diff @ diff), 2 * drc.T @ diff

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}] + self._cap_cons_long()
        iv = 1 / np.sqrt(np.diag(cov))
        x0 = np.minimum(iv / iv.sum(), m)
        x0 = x0 / x0.sum()
        res = minimize(f, x0, jac=True, method="SLSQP", bounds=[(1e-6, m)] * n, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-14})
        w = self._clean(np.clip(res.x, 0, m))
        w, t = self.blend_to_cap(w)
        notes = self._long_only_space_note(w)
        if t > 0:
            notes.append(f"blended {t:.0%} toward minimum-volatility to satisfy the volatility limit")
        return OptResult(w, "risk_parity", notes)

    def max_diversification(self) -> OptResult:
        cov, n, m = self.cov, self.n, self.rc.max_position
        sig = np.sqrt(np.diag(cov))

        def f(w):
            s = np.sqrt(max(w @ cov @ w, 1e-16))
            num = sig @ w
            return -num / s, -(sig / s - num * (cov @ w) / s**3)

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}] + self._cap_cons_long()
        res = minimize(f, np.full(n, 1 / n), jac=True, method="SLSQP", bounds=[(0, m)] * n,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-14})
        w = self._clean(np.clip(res.x, 0, m))
        w, t = self.blend_to_cap(w)
        notes = self._long_only_space_note(w)
        if t > 0:
            notes.append(f"blended {t:.0%} toward minimum-volatility to satisfy the volatility limit")
        return OptResult(w, "max_diversification", notes)

    def run(self, method: str, target: float | None = None) -> OptResult:
        if method not in METHODS:
            raise ValueError(f"unknown optimization method {method!r}; choose from {METHODS}")
        self.check_risk_limit_feasible()
        if method == "target_return":
            if target is None:
                raise ValueError("target_return method needs a target return")
            return self.target_return(target)
        return getattr(self, method)()
