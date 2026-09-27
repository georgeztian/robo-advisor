"""Monitoring workflow graph (spec §18, §19 step 12): reuses the advisory data agents and the
reviewer's input gate, then evaluates the prior recommendation for review triggers."""
from __future__ import annotations

import dataclasses

from ..config import Settings
from ..data.providers import DataProvider
from ..graph.engine import Graph, Node
from ..monitoring import assess
from ..review import Reviewer
from ..universe import CATALOG
from .advisory import (ConstraintAgent, DataValidationAgent, EstimationAgent, IntakeAgent, MarketDataAgent,
                       RiskProfilerAgent, Services, TaxAgent, add_input_gates)


class MonitorIntakeAgent(IntakeAgent):
    """Intake for monitoring: the prior portfolio's holdings are always evaluated, even when the
    updated profile no longer selects them (they are flagged as a universe change)."""

    requires = ("client", "prior")

    def __call__(self, st):
        req = super().__call__(st)["request"]
        held = [t for t, x in st["prior"]["portfolio"]["weights"].items() if abs(x) > 1e-9]
        dropped = [t for t in held if t not in req.tickers and t in CATALOG]
        if dropped:
            req = dataclasses.replace(req, tickers=req.tickers + dropped, held_not_selected=dropped,
                                      data_tickers=sorted(set(req.data_tickers) | set(dropped)))
        return {"request": req}


class MonitorAgent:
    """Re-assess the prior recommendation and raise review triggers."""

    name, requires, provides = "monitor", ("prior", "request", "risk", "estimates", "tax", "review_inputs"), ("monitoring",)

    def __init__(self, sv: Services):
        self.sv = sv

    def __call__(self, st):
        return {"monitoring": assess(st["prior"], st["request"], st["risk"], st["estimates"], st["tax"].rates,
                                     self.sv.settings)}


def build_monitoring_graph(settings: Settings, provider: DataProvider) -> Graph:
    sv = Services(settings, provider)
    g = Graph("monitoring", inputs=("client", "prior"))
    for cls in (MonitorIntakeAgent, RiskProfilerAgent, MarketDataAgent, DataValidationAgent, EstimationAgent,
                TaxAgent, ConstraintAgent, MonitorAgent):
        a = cls(sv)
        g.add(Node(a.name, a, tuple(a.requires), tuple(a.provides)))
    add_input_gates(g, Reviewer(settings))
    g.build()
    return g
