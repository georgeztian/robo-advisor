"""Monte Carlo wealth simulation (spec §12) and scenarios (spec §13).

Each path, each month:
  1. draw monthly asset total returns from the estimated distribution
     (multivariate log-normal moment-matched to mu/Sigma, or iid bootstrap of history);
  2. apply them to the current holdings (portfolio weights drift);
  3. add the monthly contribution, allocated at the target weights;
  4. rebalance according to the selected rule;
  5. when taxes are enabled: tax distributions monthly, realize gains/losses on rebalancing
     sales (short/long-term by holding age, average-cost basis), net and carry forward
     losses, settle tax at each year end (and optionally liquidate at the horizon);
  6. continue until the target date / horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RebalancingCfg
from .models import Estimates, ScenarioResult, SimulationResult
from .rebalancing import due
from .tax import TaxRates


@dataclass
class MCModel:
    """Monthly return-generating model."""

    m: np.ndarray                    # mean of monthly log returns
    root: np.ndarray                 # matrix square root of the monthly log-return covariance
    income_monthly: np.ndarray       # monthly distribution yield (part of the total return)
    distribution: str = "lognormal"
    history: np.ndarray | None = None   # monthly simple returns for bootstrap (rows = months)

    @staticmethod
    def from_moments(mu_annual: np.ndarray, cov_annual: np.ndarray, income_yield: np.ndarray,
                     mu_shift: float = 0.0, vol_multiplier: float = 1.0,
                     distribution: str = "lognormal", history: np.ndarray | None = None) -> "MCModel":
        mu_m = (np.asarray(mu_annual) + mu_shift) / 12
        cov_m = np.asarray(cov_annual) * vol_multiplier**2 / 12
        g = 1 + mu_m
        if (g <= 0).any():
            raise ValueError("expected monthly return below -100%")
        S = np.log1p(cov_m / np.outer(g, g))
        S = (S + S.T) / 2
        w, v = np.linalg.eigh(S)
        w = np.clip(w, 0, None)
        S = v @ np.diag(w) @ v.T
        m = np.log(g) - np.diag(S) / 2
        hist = None
        if history is not None:
            h = history[~np.isnan(history).any(axis=1)]
            hm = h.mean(axis=0)
            hist = hm + (h - hm) * vol_multiplier + mu_shift / 12
        return MCModel(m, v * np.sqrt(w), np.asarray(income_yield) / 12, distribution, hist)

    @staticmethod
    def from_estimates(est: Estimates, distribution: str = "lognormal", mu_shift: float = 0.0,
                       vol_multiplier: float = 1.0) -> "MCModel":
        return MCModel.from_moments(est.mu, est.cov, est.income_yield, mu_shift, vol_multiplier,
                                    distribution, est.monthly_returns.to_numpy())

    def draw(self, rng: np.random.Generator, n_paths: int) -> np.ndarray:
        if self.distribution == "bootstrap":
            if self.history is None or len(self.history) < 12:
                raise ValueError("bootstrap needs at least 12 months of complete history")
            return self.history[rng.integers(0, len(self.history), n_paths)]
        z = rng.standard_normal((n_paths, len(self.m)))
        return np.expm1(self.m + z @ self.root.T)


@dataclass
class RawSim:
    terminal: np.ndarray
    terminal_after_liq: np.ndarray | None
    wealth: np.ndarray | None          # paths x (T+1) if recorded
    max_drawdown: np.ndarray           # per path, of the time-weighted return index
    taxes_paid: np.ndarray


def simulate(w: np.ndarray, model: MCModel, W0: float, C: float, T: int, rebal: RebalancingCfg,
             tax: TaxRates, n_paths: int, seed: int, record: bool = True) -> RawSim:
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    n = len(w)
    P = n_paths
    V = np.tile(W0 * w, (P, 1))
    taxed = tax.enabled
    if taxed:
        B = V.copy()                   # cost basis (signed; shorts carry negative basis)
        A = np.zeros((P, n))           # dollar-weighted acquisition month
        st_acc = np.zeros(P)
        lt_acc = np.zeros(P)
        lt_rate_acc = np.zeros(P)
        inc_tax_acc = np.zeros(P)
        carry = np.zeros(P)
        offset_used = np.zeros(P)
        paid = np.zeros(P)
    else:
        paid = np.zeros(P)
    wealth = np.empty((P, T + 1)) if record else None
    if record:
        wealth[:, 0] = W0
    index = np.ones(P)
    peak = np.ones(P)
    mdd = np.zeros(P)
    alive = np.ones(P, bool)

    def buy(mask_amt: np.ndarray, month: int):
        # add signed dollars to basis and update the dollar-weighted acquisition month
        nonlocal B, A
        add_abs = np.abs(mask_amt)
        tot = np.abs(B) + add_abs
        A = np.where(tot > 0, (A * np.abs(B) + month * add_abs) / np.where(tot > 0, tot, 1), A)
        B = B + mask_amt

    def realize(close_frac: np.ndarray, month: int):
        nonlocal B
        gain = close_frac * (V - B)
        B = B * (1 - close_frac)
        lt = (month - A) >= 12
        return np.where(lt, 0.0, gain).sum(axis=1), np.where(lt, gain, 0.0), gain

    def settle(month: int, offset_limit: np.ndarray | float | None = None):
        nonlocal carry, V, B, offset_used     # the *_acc arrays are reset in place
        st, lt = st_acc.copy(), lt_acc.copy()
        # net short- against long-term, then apply carried-forward losses
        off = np.minimum(np.clip(-st, 0, None), np.clip(lt, 0, None))
        st, lt = st + off, lt - off
        off = np.minimum(np.clip(-lt, 0, None), np.clip(st, 0, None))
        lt, st = lt + off, st - off
        use = np.minimum(carry, np.clip(st, 0, None))
        st, carry = st - use, carry - use
        use = np.minimum(carry, np.clip(lt, 0, None))
        lt, carry = lt - use, carry - use
        net_loss = np.clip(-(st + lt), 0, None) * ((st <= 0) & (lt <= 0))
        limit = tax.loss_offset if offset_limit is None else offset_limit
        ordinary_offset = np.minimum(net_loss, limit)
        offset_used = ordinary_offset
        carry = carry + net_loss - ordinary_offset
        lt_rate = np.where(lt_acc > 0, lt_rate_acc / np.where(lt_acc > 0, lt_acc, 1), tax.lt.mean())
        lt_rate = np.clip(lt_rate, tax.lt.min(), tax.lt.max())
        due_tax = (np.clip(st, 0, None) * tax.st + np.clip(lt, 0, None) * lt_rate + inc_tax_acc
                   - ordinary_offset * tax.ordinary)
        total = V.sum(axis=1)
        frac = np.where(total > 0, np.clip(due_tax / np.where(total > 0, total, 1), -1, 1), 0)
        # taxes are paid by selling pro rata (basis falls proportionally); a refund is reinvested
        # pro rata and adds its own amount to basis
        B = np.where((frac >= 0)[:, None], B * (1 - frac)[:, None], B - frac[:, None] * V)
        V = V * (1 - frac)[:, None]
        st_acc[:] = 0
        lt_acc[:] = 0
        lt_rate_acc[:] = 0
        inc_tax_acc[:] = 0
        return np.where(total > 0, due_tax, 0.0)

    for t in range(1, T + 1):
        r = model.draw(rng, P)
        prev_total = V.sum(axis=1)
        if taxed:
            # distributions on long positions are taxed and reinvested (adding basis); payments in
            # lieu of dividends on short positions are a non-deductible cost already in the return
            income = np.clip(V, 0, None) * model.income_monthly
            inc_tax_acc += (income * tax.income).sum(axis=1)
            buy(income, t)
        V = V * (1 + r)
        total = V.sum(axis=1)
        rp = np.where(prev_total > 0, total / np.where(prev_total > 0, prev_total, 1) - 1, 0)
        index = index * (1 + rp)
        peak = np.maximum(peak, index)
        mdd = np.minimum(mdd, index / peak - 1)
        broke = (total <= 0) & (prev_total > 0)       # wiped out (not merely not yet funded)
        if broke.any():
            alive &= ~broke
            V[broke] = 0
            if taxed:
                B[broke] = 0
        if C > 0:
            add = np.where(alive[:, None], C * w, 0.0)
            if taxed:
                buy(add, t)
            V = V + add
        total = V.sum(axis=1)
        cur_w = V / np.where(total > 0, total, 1)[:, None]
        reb = due(rebal, t, cur_w, w)
        reb_mask = (np.full(P, bool(reb)) if np.isscalar(reb) or np.ndim(reb) == 0 else reb) & alive
        if reb_mask.any():
            target = total[:, None] * w
            if taxed:
                same = np.sign(V) == np.sign(target)
                shrink = same & (np.abs(target) < np.abs(V))
                close_frac = np.where(shrink, 1 - np.abs(target) / np.where(V != 0, np.abs(V), 1), 0.0)
                close_frac = np.where(reb_mask[:, None], close_frac, 0.0)
                st_g, lt_g, _ = realize(close_frac, t)
                st_acc += st_g
                lt_acc += lt_g.sum(axis=1)
                lt_rate_acc += (lt_g * tax.lt).sum(axis=1)
                grow = np.where(reb_mask[:, None] & same & (np.abs(target) > np.abs(V)), target - V, 0.0)
                buy(grow, t)
            V = np.where(reb_mask[:, None], target, V)
        if taxed and (t % 12 == 0 or t == T):
            paid += settle(t)
        if record:
            wealth[:, t] = V.sum(axis=1)
    terminal = V.sum(axis=1)
    after_liq = None
    if taxed and tax.liquidate_at_horizon:
        st_g, lt_g, _ = realize(np.ones_like(V), T + 1)
        st_acc += st_g
        lt_acc += lt_g.sum(axis=1)
        lt_rate_acc += (lt_g * tax.lt).sum(axis=1)
        # liquidation happens in the same tax year as the final settlement: the annual
        # ordinary-income offset is not available twice
        liq_tax = settle(T + 1, np.clip(tax.loss_offset - offset_used, 0, None))
        after_liq = V.sum(axis=1)
        paid += liq_tax
    return RawSim(terminal, after_liq, wealth, mdd, paid)


def summarize(raw: RawSim, T: int, W0: float, C: float, target: float | None, pct: list[int],
              seed: int, distribution: str, start: pd.Period | None = None) -> SimulationResult:
    term = raw.terminal_after_liq if raw.terminal_after_liq is not None else raw.terminal
    contributed = W0 + C * T
    band = None
    if raw.wealth is not None:
        q = np.percentile(raw.wealth, pct, axis=0).T
        idx = (pd.period_range(start, periods=T + 1, freq="M") if start is not None
               else pd.RangeIndex(T + 1, name="month"))
        band = pd.DataFrame(q, index=idx, columns=[f"p{p}" for p in pct])
        band["contributed"] = W0 + C * np.arange(T + 1)
    return SimulationResult(
        n_paths=len(term), months=T, terminal=raw.terminal, terminal_after_liquidation=raw.terminal_after_liq,
        percentiles={p: float(np.percentile(term, p)) for p in pct}, band=band,
        prob_target=float((term >= target).mean()) if target is not None else None,
        prob_loss_principal=float((term < contributed).mean()), total_contributed=contributed,
        max_drawdown_median=float(np.median(raw.max_drawdown)),
        max_drawdown_p95=float(np.percentile(raw.max_drawdown, 5)),
        expected_terminal=float(term.mean()), taxes_paid_median=float(np.median(raw.taxes_paid)),
        seed=seed, distribution=distribution)


def run_scenarios(w: np.ndarray, est: Estimates, scen_cfg, W0: float, C: float, T: int,
                  rebal: RebalancingCfg, tax: TaxRates, target: float | None, seed: int,
                  distribution: str) -> list[ScenarioResult]:
    out = []
    for name in ("conservative", "base", "optimistic"):
        s = getattr(scen_cfg, name)
        model = MCModel.from_estimates(est, distribution, s.mu_shift, s.vol_multiplier)
        raw = simulate(w, model, W0, C, T, rebal, tax, scen_cfg.n_paths, seed, record=False)
        term = raw.terminal_after_liq if raw.terminal_after_liq is not None else raw.terminal
        out.append(ScenarioResult(
            name=name, mu_shift=s.mu_shift, vol_multiplier=s.vol_multiplier,
            median=float(np.median(term)), p10=float(np.percentile(term, 10)),
            p90=float(np.percentile(term, 90)),
            prob_target=float((term >= target).mean()) if target is not None else None,
            prob_loss_principal=float((term < W0 + C * T).mean())))
    return out
