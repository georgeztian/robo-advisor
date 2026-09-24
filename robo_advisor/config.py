"""Typed, validated configuration (loaded from ``config/default.yaml`` + optional overrides)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


class RiskBand(BaseModel):
    max_score: float
    profile: str
    max_volatility: float = Field(gt=0)


class Question(BaseModel):
    id: str
    text: str
    weight: float = Field(ge=0)
    options: dict[str, float]
    derive_from_goal: bool = False

    @model_validator(mode="after")
    def _scores_in_range(self):
        for k, v in self.options.items():
            if not 0 <= v <= 100:
                raise ValueError(f"question {self.id}: option {k} score {v} outside 0-100")
        return self


class Questionnaire(BaseModel):
    capacity: list[Question]
    tolerance: list[Question]

    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        for name in ("capacity", "tolerance"):
            total = sum(q.weight for q in getattr(self, name))
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"{name} question weights sum to {total}, expected 1.0")
        return self


class UniverseCfg(BaseModel):
    """ETFs offered to clients, grouped into categories (the client chooses from these)."""

    categories: dict[str, list[str]]

    @model_validator(mode="after")
    def _known_and_unique(self):
        from .universe import CATALOG

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
        return self

    @property
    def tickers(self) -> list[str]:
        return [t for ts in self.categories.values() for t in ts]


class DataCfg(BaseModel):
    provider: Literal["synthetic", "csv", "yahoo"] = "synthetic"
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


class EstimationCfg(BaseModel):
    trading_days: int = 252
    mu_shrinkage: float = Field(0.0, ge=0, le=1)
    var_confidence: float = 0.95
    min_overlap_days: int = 252


class GoalCfg(BaseModel):
    target_probability: float = Field(0.80, gt=0, lt=1)
    risk_metric: Literal["volatility", "cvar"] = "volatility"
    frontier_points: int = 25
    search_paths: int = 2000
    probability_margin: float = 0.01


Method = Literal["mean_variance", "min_volatility", "max_sharpe", "cvar", "target_return",
                 "risk_parity", "max_diversification"]


class OptimizationCfg(BaseModel):
    default_method: Method = "mean_variance"
    max_position: float = Field(0.5, gt=0, le=1)
    max_gross_leverage: float = Field(1.5, ge=1)
    n_starts: int = 6
    tolerance: float = 1e-4
    cvar_alpha: float = 0.95
    goal: GoalCfg = GoalCfg()


class SimulationCfg(BaseModel):
    n_paths: int = 10000
    seed: int = 20260924
    distribution: Literal["lognormal", "bootstrap"] = "lognormal"
    percentiles: list[int] = [10, 25, 50, 75, 90]
    inflation: float = 0.025


class RebalancingCfg(BaseModel):
    type: Literal["calendar", "threshold"] = "calendar"
    frequency: Literal["monthly", "quarterly", "annual"] = "quarterly"
    threshold: float = 0.05


class ScenarioShift(BaseModel):
    mu_shift: float
    vol_multiplier: float = Field(gt=0)


class ScenariosCfg(BaseModel):
    conservative: ScenarioShift
    base: ScenarioShift
    optimistic: ScenarioShift
    n_paths: int = 5000


class BenchmarkCfg(BaseModel):
    years: int = 10


class TaxCfg(BaseModel):
    ordinary_rate: float = 0.24
    qualified_dividend_rate: float = 0.15
    ltcg_rate: float = 0.15
    stcg_rate: float = 0.24
    collectibles_rate: float = 0.28
    state_rate: float = 0.0
    capital_loss_ordinary_offset: float = 3000
    assumed_turnover: float = 0.10
    liquidate_at_horizon: bool = False


class MonitoringCfg(BaseModel):
    target_probability_floor: float = 0.70
    risk_limit_tolerance: float = 0.005
    volatility_change_trigger: float = 0.25
    correlation_change_trigger: float = 0.20
    contribution_change_trigger: float = 0.10


class ReviewCfg(BaseModel):
    mu_abs_tol: float = 5e-4
    sigma_rel_tol: float = 0.02
    mc_z: float = 4.0
    mc_paths: int = 4000
    optimality_tol: float = 0.0025
    backtest_rel_tol: float = 0.005
    wealth_rel_tol: float = 1e-6


class Settings(BaseModel):
    risk_bands: list[RiskBand]
    risk_mapping: Literal["min"] = "min"
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
