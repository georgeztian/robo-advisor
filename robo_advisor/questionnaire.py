"""Risk profiling (spec §2): capacity and tolerance scored separately, then mapped.

    Risk Capacity Score  = sum_k weight_k * score(answer_k)      (capacity questions)
    Risk Tolerance Score = sum_k weight_k * score(answer_k)      (tolerance questions)
    Mapped Risk Score    = min(Capacity, Tolerance)
"""
from __future__ import annotations

import datetime as dt

from .config import Question, Settings
from .models import ClientInput, RiskAssessment


class QuestionnaireError(ValueError):
    pass


def horizon_option(months: int) -> str:
    years = months / 12
    if years < 3:
        return "under_3y"
    if years < 5:
        return "3_5y"
    if years < 10:
        return "5_10y"
    if years < 20:
        return "10_20y"
    return "over_20y"


def _score_block(questions: list[Question], answers: dict[str, str], derived: dict[str, str],
                 block: str) -> tuple[float, dict]:
    detail: dict[str, dict] = {}
    missing, invalid = [], []
    total = 0.0
    for q in questions:
        ans = answers.get(q.id, derived.get(q.id))
        if ans is None:
            missing.append(q.id)
            continue
        if ans not in q.options:
            invalid.append(f"{q.id}={ans!r} (valid: {', '.join(q.options)})")
            continue
        s = q.options[ans]
        total += q.weight * s
        detail[q.id] = {"question": q.text, "answer": ans, "score": s, "weight": q.weight,
                        "contribution": q.weight * s, "derived": q.id not in answers}
    unknown = set(answers) - {q.id for q in questions}
    if missing or invalid or unknown:
        msg = [f"{block} questionnaire incomplete/invalid:"]
        if missing:
            msg.append(f"missing {missing}")
        if invalid:
            msg.append(f"invalid {invalid}")
        if unknown:
            msg.append(f"unknown questions {sorted(unknown)}")
        raise QuestionnaireError(" ".join(msg))
    return round(total, 2), detail


def assess_risk(settings: Settings, client: ClientInput, as_of: dt.date) -> RiskAssessment:
    qs = settings.questionnaire
    derived: dict[str, str] = {}
    months = client.goal.months(as_of)
    for q in qs.capacity:
        if q.derive_from_goal and q.id not in client.capacity_answers:
            derived[q.id] = horizon_option(months)
    cap, cap_detail = _score_block(qs.capacity, client.capacity_answers, derived, "capacity")
    tol, tol_detail = _score_block(qs.tolerance, client.tolerance_answers, {}, "tolerance")
    mapped = min(cap, tol)   # settings.risk_mapping == "min" (the only supported, conservative rule)
    band = settings.band_for(mapped)
    return RiskAssessment(capacity_score=cap, tolerance_score=tol, mapped_score=mapped,
                          profile=band.profile, max_volatility=band.max_volatility,
                          capacity_detail=cap_detail, tolerance_detail=tol_detail,
                          derived_answers=derived)
