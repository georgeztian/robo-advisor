import pytest

from conftest import AS_OF, load_client
from robo_advisor.config import load_settings
from robo_advisor.questionnaire import QuestionnaireError, assess_risk, horizon_option


def test_scores_weighted_and_mapped_to_min(settings):
    c = load_client("client_target.json")
    r = assess_risk(settings, c, AS_OF)
    qs = settings.questionnaire
    cap_ans = {**r.derived_answers, **c.capacity_answers}
    cap = sum(q.weight * q.options[cap_ans[q.id]] for q in qs.capacity)
    tol = sum(q.weight * q.options[c.tolerance_answers[q.id]] for q in qs.tolerance)
    assert r.capacity_score == pytest.approx(cap, abs=0.01)
    assert r.tolerance_score == pytest.approx(tol, abs=0.01)
    assert r.mapped_score == min(r.capacity_score, r.tolerance_score)
    assert r.derived_answers == {"investment_horizon": "10_20y"}      # 123 months


@pytest.mark.parametrize("score,profile,vol", [(0, "Very Conservative", .05), (20, "Very Conservative", .05),
                                               (20.5, "Conservative", .08), (60, "Moderate", .12),
                                               (70, "Growth", .17), (100, "Aggressive", .25)])
def test_band_lookup(settings, score, profile, vol):
    b = settings.band_for(score)
    assert (b.profile, b.max_volatility) == (profile, vol)


def test_bands_are_configurable():
    s = load_settings(overrides={"risk_bands": [{"max_score": 50, "profile": "Low", "max_volatility": 0.06},
                                                {"max_score": 100, "profile": "High", "max_volatility": 0.2}]})
    assert s.band_for(70).profile == "High"


def test_invalid_and_missing_answers_rejected(settings):
    c = load_client("client_target.json")
    bad = c.model_copy(update={"tolerance_answers": {**c.tolerance_answers, "experience": "guru"}})
    with pytest.raises(QuestionnaireError, match="experience"):
        assess_risk(settings, bad, AS_OF)
    missing = c.model_copy(update={"capacity_answers": {}})
    with pytest.raises(QuestionnaireError, match="missing"):
        assess_risk(settings, missing, AS_OF)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum"):
        load_settings(overrides={"questionnaire": {"tolerance": [
            {"id": "x", "text": "x", "weight": 0.5, "options": {"a": 10}}]}})


def test_horizon_buckets():
    assert [horizon_option(m) for m in (12, 36, 84, 150, 300)] == ["under_3y", "3_5y", "5_10y", "10_20y", "over_20y"]
