import datetime as dt
import json
from pathlib import Path

import pytest

from robo_advisor.agents.advisory import build_advisory_graph
from robo_advisor.config import load_settings
from robo_advisor.data.providers import SyntheticProvider
from robo_advisor.models import ClientInput

ROOT = Path(__file__).resolve().parents[1]
AS_OF = dt.date(2026, 9, 23)

# smaller Monte Carlo / search sizes keep the suite fast; logic is identical
FAST = {
    "simulation": {"n_paths": 3000},
    "scenarios": {"n_paths": 1000},
    "optimization": {"goal": {"search_paths": 800, "frontier_points": 10}},
    "review": {"mc_paths": 2000},
}


@pytest.fixture(scope="session")
def settings():
    return load_settings(overrides=FAST)


@pytest.fixture(scope="session")
def provider():
    return SyntheticProvider()


def load_client(name: str, **update) -> ClientInput:
    data = json.loads((ROOT / "examples" / name).read_text())
    for k, v in update.items():
        data[k] = v
    return ClientInput.model_validate(data)


@pytest.fixture(scope="session")
def target_run(settings, provider):
    return build_advisory_graph(settings, provider).run({"client": load_client("client_target.json")})


@pytest.fixture(scope="session")
def no_target_run(settings, provider):
    return build_advisory_graph(settings, provider).run({"client": load_client("client_no_target.json")})
