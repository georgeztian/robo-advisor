"""Domain objects.

Client inputs are pydantic models (validated at the system boundary). Computed artifacts
are plain dataclasses holding numpy / pandas objects.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- client input


class Goal(BaseModel):
    """Spec §1A. With a target, the optimizer minimizes risk subject to reaching it (Case A);
    without one, it maximizes expected return subject to the mapped risk limit (Case B)."""

    has_target: bool
    initial_investment: float = Field(ge=0)
    monthly_contribution: float = Field(ge=0)
    target_amount: float | None = Field(None, gt=0)
    target_date: dt.date | None = None
    horizon_years: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.has_target:
            if self.target_amount is None or self.target_date is None:
                raise ValueError("a target goal requires target_amount and target_date")
        elif self.horizon_years is None:
            raise ValueError("a no-target goal requires horizon_years")
        if self.initial_investment == 0 and self.monthly_contribution == 0:
            raise ValueError("initial investment and monthly contribution cannot both be zero")
        return self

    def months(self, as_of: dt.date) -> int:
        """T = number of monthly periods until the target date / end of horizon."""
        if self.has_target:
            assert self.target_date is not None
            m = (self.target_date.year - as_of.year) * 12 + (self.target_date.month - as_of.month)
            if m < 1:
                raise ValueError("target_date must be at least one month after the as-of date")
            return m
        assert self.horizon_years is not None
        return max(1, int(round(self.horizon_years * 12)))


class Constraints(BaseModel):
    """Spec §4."""

    allow_short: bool = False
    max_position: float | None = Field(None, gt=0, le=1)       # None -> config default
    max_gross_leverage: float | None = Field(None, ge=1)       # None -> config default
    # max share of the portfolio per ETF category, e.g. {"Crypto ETFs": 0.05}; overrides the
    # config defaults for the categories given (1.0 removes a limit)
    category_limits: dict[str, float] | None = None

    @model_validator(mode="after")
    def _limits_in_range(self):
        for cat, lim in (self.category_limits or {}).items():
            if not 0 < lim <= 1:
                raise ValueError(f"category limit for {cat!r} must be in (0, 1], got {lim}")
        return self


class TaxInput(BaseModel):
    """Spec §5. Rates left as None fall back to config defaults."""

    enabled: bool = False
    ordinary_rate: float | None = Field(None, ge=0, lt=1)
    qualified_dividend_rate: float | None = Field(None, ge=0, lt=1)
    ltcg_rate: float | None = Field(None, ge=0, lt=1)
    stcg_rate: float | None = Field(None, ge=0, lt=1)
    state_rate: float | None = Field(None, ge=0, lt=1)
    liquidate_at_horizon: bool | None = None


class Preferences(BaseModel):
    optimization_method: str | None = None             # None -> config default
    target_return: float | None = None                 # for the target_return method (Case B)
    goal_risk_metric: Literal["volatility", "cvar"] | None = None
    target_probability: float | None = Field(None, gt=0, lt=1)
    rebalancing_type: Literal["calendar", "threshold"] | None = None
    rebalancing_frequency: Literal["monthly", "quarterly", "annual"] | None = None
    rebalancing_threshold: float | None = Field(None, gt=0, lt=1)


class ClientProfile(BaseModel):
    """Spec §19 step 1: experience / financial information / horizon (free-form context)."""

    name: str = "Client"
    age: int | None = None
    investment_experience: str | None = None
    notes: str | None = None


class ClientInput(BaseModel):
    profile: ClientProfile = ClientProfile()
    as_of: dt.date | None = None
    goal: Goal
    capacity_answers: dict[str, str]
    tolerance_answers: dict[str, str]
    universe: list[str] | None = None     # tickers and/or category names chosen by the client (required)
    constraints: Constraints = Constraints()
    taxes: TaxInput = TaxInput()
    preferences: Preferences = Preferences()


# --------------------------------------------------------------------------- computed artifacts


@dataclass
class RiskAssessment:
    capacity_score: float
    tolerance_score: float
    mapped_score: float
    profile: str
    max_volatility: float
    capacity_detail: dict[str, dict[str, Any]]
    tolerance_detail: dict[str, dict[str, Any]]
    derived_answers: dict[str, str] = field(default_factory=dict)


@dataclass
class CategoryCap:
    category: str
    limit: float                        # max sum of |w_i| over the category's selected ETFs
    tickers: list[str]


@dataclass
class ResolvedConstraints:
    tickers: list[str]
    allow_short: bool
    max_position: float
    max_gross_leverage: float           # 1.0 when long-only
    max_volatility: float
    category_caps: list[CategoryCap] = field(default_factory=list)


@dataclass
class MarketData:
    """Raw per-ticker daily frames with columns close, adj_close, dividend, split_ratio."""

    frames: dict[str, pd.DataFrame]
    as_of: dt.date
    source: str
    synthetic: bool
    risk_free_series: pd.Series | None = None      # annual decimal Treasury rate (e.g. FRED DTB3)
    risk_free_source: str = "T-bill ETF proxy"
    notes: list[str] = field(default_factory=list)

    def tickers(self) -> list[str]:
        return list(self.frames)


@dataclass
class TickerQuality:
    ticker: str
    inception: dt.date | None
    first_obs: dt.date | None
    last_obs: dt.date | None
    years_available: float
    meets_min_history: bool
    existed_full_window: bool
    missing_days: int
    missing_fraction: float
    max_gap_days: int
    adj_consistency_max_error: float
    n_splits: int
    n_distributions: int
    expense_ratio: float | None
    issues: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    window_start: dt.date
    window_end: dt.date
    tickers: dict[str, TickerQuality]
    blocking: list[str]
    warnings: list[str]


@dataclass
class Estimates:
    tickers: list[str]
    mu: np.ndarray                  # annual arithmetic expected total return (pre-tax)
    sigma: np.ndarray               # annual volatility (daily data)
    cov: np.ndarray                 # annual covariance (daily data, PSD-repaired)
    corr: np.ndarray
    risk_free: float
    daily_returns: pd.DataFrame     # aligned daily simple total returns (NaN before inception)
    monthly_returns: pd.DataFrame   # month-end simple total returns
    income_yield: np.ndarray        # annual distribution yield
    stats: pd.DataFrame             # per-ETF risk statistics (drawdown, VaR, CVaR, beta, ...)
    window_start: dt.date
    window_end: dt.date
    history_years: dict[str, float]
    monthly_income: pd.DataFrame | None = None   # monthly distribution yield per ETF
    mu_after_tax: np.ndarray | None = None
    psd_repaired: bool = False
    risk_free_source: str = "T-bill ETF proxy"


@dataclass
class Portfolio:
    tickers: list[str]
    weights: np.ndarray
    method: str
    case: Literal["target", "no_target"]
    expected_return: float          # w'mu (after-tax when taxes enabled)
    expected_return_pretax: float
    volatility: float               # sqrt(w' Sigma w)
    sharpe: float
    constraints: ResolvedConstraints
    initial_allocation: dict[str, float]
    monthly_allocation: dict[str, float]
    goal_search: dict[str, Any] | None = None     # Case A frontier search diagnostics
    notes: list[str] = field(default_factory=list)

    def weight_map(self) -> dict[str, float]:
        return {t: float(w) for t, w in zip(self.tickers, self.weights)}


@dataclass
class SimulationResult:
    n_paths: int
    months: int
    terminal: np.ndarray                         # terminal wealth per path
    terminal_after_liquidation: np.ndarray | None
    percentiles: dict[int, float]
    band: pd.DataFrame                           # month x percentile wealth bands
    prob_target: float | None
    prob_loss_principal: float
    total_contributed: float
    max_drawdown_median: float
    max_drawdown_p95: float
    expected_terminal: float
    taxes_paid_median: float
    seed: int
    distribution: str


@dataclass
class ScenarioResult:
    name: str
    mu_shift: float
    vol_multiplier: float
    median: float
    p10: float
    p90: float
    prob_target: float | None
    prob_loss_principal: float


@dataclass
class BenchmarkResult:
    start: dt.date
    end: dt.date
    years: float
    initial_investment: float
    monthly_contribution: float
    metrics: pd.DataFrame                         # rows: metric, cols: Portfolio, S&P 500
    growth_of_10k: pd.DataFrame                   # monthly index, cols Portfolio / S&P 500
    wealth_with_contributions: pd.DataFrame
    calendar_year_returns: pd.DataFrame
    notes: list[str] = field(default_factory=list)


@dataclass
class Projection:
    months: int
    monthly_rate: float
    fv_nominal: float
    fv_real: float
    total_contributed: float


@dataclass
class ReviewFinding:
    rule_id: str
    spec_ref: str
    severity: Literal["BLOCKER", "WARN", "INFO"]
    passed: bool
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewReport:
    stage: str
    findings: list[ReviewFinding]

    @property
    def blocking(self) -> list[ReviewFinding]:
        return [f for f in self.findings if not f.passed and f.severity == "BLOCKER"]

    @property
    def warnings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if not f.passed and f.severity == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        n = len(self.findings)
        p = sum(f.passed for f in self.findings)
        return f"[{self.stage}] {p}/{n} rules passed, {len(self.blocking)} blocker(s), {len(self.warnings)} warning(s)"


@dataclass
class Request:
    """Normalized advisory request produced by the intake agent."""

    client: ClientInput
    as_of: dt.date
    window_start: dt.date
    months: int
    W0: float
    C: float
    target: float | None
    target_date: dt.date | None
    has_target: bool
    tickers: list[str]
    method: str
    rebalancing: Any                 # config.RebalancingCfg (client overrides applied)
    target_probability: float
    goal_risk_metric: str
    data_tickers: list[str]          # selected + benchmark + risk-free proxy


@dataclass
class TaxContext:
    rates: Any                       # tax.TaxRates
    mu_after_tax: np.ndarray | None
    monthly_after_tax: pd.DataFrame | None
    disclaimer: str | None


@dataclass
class Explanation:
    headline: str
    risk_text: str
    goal_text: str
    portfolio_text: str
    methodology: list[str]
    etf_rationale: pd.DataFrame      # per ETF: weight, return, vol, corr, risk contribution, reason
    assumptions: list[str]
    limitations: list[str]
    disclosures: list[str]
    calculations: dict[str, Any]
