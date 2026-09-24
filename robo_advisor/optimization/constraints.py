"""Portfolio constraints (spec §4) in a form usable by scipy SLSQP / linprog.

Long-only:   0 <= w_i <= max_position,  sum w_i = 1
Short sales: w = p - q with p, q >= 0 (variable split), so gross exposure is linear:
             sum (p_i + q_i) <= L,  p_i, q_i <= max_position (=> |w_i| <= max_position),
             sum (p_i - q_i) = 1
Risk limit:  w' Sigma w <= sigma_max^2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ResolvedConstraints


class InfeasibleError(ValueError):
    pass


@dataclass
class Space:
    n: int
    rc: ResolvedConstraints

    @property
    def short(self) -> bool:
        return self.rc.allow_short

    @property
    def dim(self) -> int:
        return 2 * self.n if self.short else self.n

    def to_w(self, x: np.ndarray) -> np.ndarray:
        return x[: self.n] - x[self.n:] if self.short else x

    def grad_to_x(self, g: np.ndarray) -> np.ndarray:
        return np.concatenate([g, -g]) if self.short else g

    def from_w(self, w: np.ndarray) -> np.ndarray:
        return np.concatenate([np.clip(w, 0, None), np.clip(-w, 0, None)]) if self.short else w

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, self.rc.max_position)] * self.dim

    def check_feasible(self) -> None:
        if self.n * self.rc.max_position < 1 - 1e-12:
            raise InfeasibleError(
                f"{self.n} ETF(s) with max position {self.rc.max_position:.0%} cannot sum to 100%; "
                "select more ETFs or raise the position limit")

    def linear_constraints(self) -> list[dict]:
        cons = [{"type": "eq", "fun": lambda x: self.to_w(x).sum() - 1.0,
                 "jac": lambda x: self.grad_to_x(np.ones(self.n))}]
        if self.short:
            L = self.rc.max_gross_leverage
            cons.append({"type": "ineq", "fun": lambda x: L - x.sum(),
                         "jac": lambda x: -np.ones(self.dim)})
        return cons

    def vol_constraint(self, cov: np.ndarray, sigma_max: float) -> dict:
        s2 = sigma_max**2
        return {"type": "ineq", "fun": lambda x: s2 - self.to_w(x) @ cov @ self.to_w(x),
                "jac": lambda x: self.grad_to_x(-2 * cov @ self.to_w(x))}

    def starts(self, k: int, rng: np.random.Generator) -> list[np.ndarray]:
        n, m = self.n, self.rc.max_position
        out = [np.full(n, 1.0 / n)]
        for _ in range(k - 1):
            w = rng.dirichlet(np.ones(n))
            for _ in range(50):                     # water-fill into the position cap
                over = w > m
                if not over.any():
                    break
                excess = (w[over] - m).sum()
                w[over] = m
                free = ~over & (w < m)
                w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
            out.append(w)
        return [self.from_w(w) for w in out]

    def is_feasible(self, w: np.ndarray, tol: float = 1e-6, cov: np.ndarray | None = None,
                    sigma_max: float | None = None) -> bool:
        if abs(w.sum() - 1) > tol or np.abs(w).max() > self.rc.max_position + tol:
            return False
        if not self.short and w.min() < -tol:
            return False
        if self.short and np.abs(w).sum() > self.rc.max_gross_leverage + tol:
            return False
        if cov is not None and sigma_max is not None and np.sqrt(max(w @ cov @ w, 0)) > sigma_max + tol:
            return False
        return True
