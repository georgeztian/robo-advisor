"""Typed, validated configuration (loaded from ``config/default.yaml`` + optional overrides)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
# answer codes of the horizon question, which is answered from the goal (derive_from_goal)
HORIZON_CODES = ("under_3y", "3_5y", "5_10y", "10_20y", "over_20y")


class _Cfg(BaseModel):
    """Settings section: an unknown key (e.g. a typo in a --config file) is an error, not ignored."""

    model_config = ConfigDict(extra="forbid")


class RiskBand(_Cfg):
    max_score: float
    profile: str
    max_volatility: float = Field(gt=0)


class Question(_Cfg):
    id: str
    text: str
    weight: float = Field(ge=0)
    options: dict[str, float]
    labels: dict[str, str] = {}         # optional display text per option code (default: the code)
    derive_from_goal: bool = False

    @model_validator(mode="after")
    def _scores_in_range(self):
        for k, v in self.options.items():
            if not 0 <= v <= 100:
                raise ValueError(f"question {self.id}: option {k} score {v} outside 0-100")
        if set(self.labels) - set(self.options):
            raise ValueError(f"question {self.id}: labels for unknown options {sorted(set(self.labels) - set(self.options))}")
        return self

    def label(self, option: str) -> str:
        return self.labels.get(option, option.replace("_", " "))


class Questionnaire(_Cfg):
    capacity: list[Question]
    tolerance: list[Question]

    @model_validator(mode="after")
    def _consistent(self):
        for name in ("capacity", "tolerance"):
            qs = getattr(self, name)
            total = sum(q.weight for q in qs)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"{name} question weights sum to {total}, expected 1.0")
            ids = [q.id for q in qs]
            if len(set(ids)) < len(ids):
                raise ValueError(f"{name} question ids are not unique: "
                                 f"{sorted({i for i in ids if ids.count(i) > 1})}")
        if any(q.derive_from_goal for q in self.tolerance):
            raise ValueError("derive_from_goal is only supported for capacity questions")
        for q in self.capacity:
            missing = [c for c in HORIZON_CODES if c not in q.options]
            if q.derive_from_goal and missing:
                raise ValueError(f"question {q.id} is answered from the goal (derive_from_goal), so its "
                                 f"options must include {', '.join(HORIZON_CODES)}; missing {missing}")
        return self


class UniverseCfg(_Cfg):
    """ETFs offered to clients, grouped into categories (the client chooses from these)."""

    categories: dict[str, list[str]]

    @model_validator(mode="after")
    def _known_and_unique(self):
        from .universe import CATALOG, catalog_problems

        seen: dict[str, str] = {}
        for cat, tickers in self.categories.items():
            if not tickers:
                raise ValueError(f"universe category {cat!r} is empty")
            for t in tickers:
                if t not in CATALOG:
                    raise ValueError(f"universe ticker {t} ({cat}) has no entry in the ETF catalog")
                if t in seen:
                    raise ValueError(f"universe ticker {t} is listed in both {seen[t]!r} and {cat!r}")
                seen[t] = cat
        bad = catalog_problems(seen)
        if bad:
            raise ValueError("; ".join(bad))
        return self

    @property
    def tickers(self) -> list[str]:
        return [t for ts in self.categories.values() for t in ts]


class DataCfg(_Cfg):
    provider: Literal["synthetic", "csv", "yahoo"] = "yahoo"
    csv_dir: str = "data/prices"
    cache_dir: str = ".cache/prices"
    lookback_years: int = 20
    min_history_years: float = 20
    min_observations: int = 252
    max_missing_fraction: float = 0.02
    max_ffill_days: int = 3
    adj_consistency_tol: float = 0.002
    adj_split_error: float = 0.25
    adj_max_bad_days: int = 3
    adj_max_bad_fraction: float = 0.002
    request_retries: int = 4
    cache_refresh_hours: float = 12.0
    benchmark: str = "VOO"
    risk_free_ticker: str = "BIL"
    risk_free_fallback: float = 0.02
    risk_free_source: Literal["etf", "fred"] = "etf"
    fred_series: str = "DTB3"


class EstimationCfg(_Cfg):
    trading_days: int = 252
    mu_shrinkage: float = Field(0.0, ge=0, le=1)
    var_confidence: float = 0.95
    min_overlap_days: int = 252


class GoalCfg(_Cfg):
    target_probability: float = Field(0.80, gt=0, lt=1)
    risk_metric: Literal["volatility", "cvar"] = "volatility"
    frontier_points: int = 25
    search_paths: int = 2000
    probability_margin: float = 0.01


Method = Literal["mean_variance", "min_volatility", "max_sharpe", "cvar", "target_return",
                 "risk_parity", "max_diversification"]


class OptimizationCfg(_Cfg):
    default_method: Method = "mean_variance"
    min_position: float = Field(0.0, ge=0, lt=1)
    max_position: float = Field(0.5, gt=0, le=1)
    max_gross_leverage: float = Field(1.5, ge=1)
    category_limits: dict[str, float] = {        # max share of the portfolio per ETF category
        "Crypto ETFs": 0.05, "Income ETFs": 0.25, "Commodity ETFs": 0.20, "Real Estate ETFs": 0.20}
    n_starts: int = 6
    tolerance: float = 1e-4
    cvar_alpha: float = 0.95
    goal: GoalCfg = GoalCfg()


class SimulationCfg(_Cfg):
    n_paths: int = 10000
    seed: int = 20260924
    distribution: Literal["lognormal", "bootstrap"] = "lognormal"
    percentiles: list[int] = [10, 25, 50, 75, 90]
    inflation: float = 0.025


class RebalancingCfg(_Cfg):
    type: Literal["calendar", "threshold"] = "calendar"
    frequency: Literal["monthly", "quarterly", "annual"] = "quarterly"
    threshold: float = 0.05


class ScenarioShift(_Cfg):
    mu_shift: float
    vol_multiplier: float = Field(gt=0)


class ScenariosCfg(_Cfg):
    conservative: ScenarioShift
    base: ScenarioShift
    optimistic: ScenarioShift
    n_paths: int = 5000


class BenchmarkCfg(_Cfg):
    years: int = 10


class TaxCfg(_Cfg):
    ordinary_rate: float = Field(0.24, ge=0, lt=1)
    qualified_dividend_rate: float = Field(0.15, ge=0, lt=1)
    ltcg_rate: float = Field(0.15, ge=0, lt=1)
    stcg_rate: float = Field(0.24, ge=0, lt=1)
    collectibles_rate: float = Field(0.28, ge=0, lt=1)
    state_rate: float = Field(0.0, ge=0, lt=1)
    capital_loss_ordinary_offset: float = Field(3000, ge=0)
    assumed_turnover: float = Field(0.10, ge=0, le=1)
    liquidate_at_horizon: bool = False


class MonitoringCfg(_Cfg):
    target_probability_floor: float = 0.70
    risk_limit_tolerance: float = 0.005
    volatility_change_trigger: float = 0.25
    correlation_change_trigger: float = 0.20
    contribution_change_trigger: float = 0.10


class ReviewCfg(_Cfg):
    mu_abs_tol: float = 5e-4
    sigma_rel_tol: float = 0.02
    mc_z: float = 4.0
    mc_paths: int = 4000
    optimality_tol: float = 0.0025
    backtest_rel_tol: float = 0.005
    wealth_rel_tol: float = 1e-6


class Settings(_Cfg):
    risk_bands: list[RiskBand]
    questionnaire: Questionnaire
    universe: UniverseCfg
    data: DataCfg = DataCfg()
    estimation: EstimationCfg = EstimationCfg()
    optimization: OptimizationCfg = OptimizationCfg()
    simulation: SimulationCfg = SimulationCfg()
    rebalancing: RebalancingCfg = RebalancingCfg()
    scenarios: ScenariosCfg
    benchmark: BenchmarkCfg = BenchmarkCfg()
    tax: TaxCfg = TaxCfg()
    monitoring: MonitoringCfg = MonitoringCfg()
    review: ReviewCfg = ReviewCfg()

    @model_validator(mode="after")
    def _category_limits_known(self):
        for cat, lim in self.optimization.category_limits.items():
            if cat not in self.universe.categories:
                raise ValueError(f"category limit for unknown category {cat!r}")
            if not 0 < lim <= 1:
                raise ValueError(f"category limit for {cat!r} must be in (0, 1], got {lim}")
        return self

    @model_validator(mode="after")
    def _reference_etfs_known(self):
        from .universe import CATALOG

        for role, t in (("benchmark", self.data.benchmark), ("risk_free_ticker", self.data.risk_free_ticker)):
            if t not in CATALOG:
                raise ValueError(f"data.{role} {t} has no entry in the ETF catalog")
        return self

    @model_validator(mode="after")
    def _bands_cover_0_100(self):
        prev = -1.0
        for b in self.risk_bands:
            if b.max_score <= prev:
                raise ValueError("risk_bands must have strictly increasing max_score")
            prev = b.max_score
        if prev < 100:
            raise ValueError("risk_bands must cover scores up to 100")
        return self

    def band_for(self, score: float) -> RiskBand:
        for b in self.risk_bands:
            if score <= b.max_score:
                return b
        return self.risk_bands[-1]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load the default YAML, deep-merge an optional user YAML file and dict overrides."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if path:
        raw = _deep_merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings.model_validate(raw)
