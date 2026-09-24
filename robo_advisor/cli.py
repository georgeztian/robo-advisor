"""Command-line interface.

    robo-advisor run --profile examples/client_target.json [--out out/]
    robo-advisor run --interactive
    robo-advisor monitor --prior out/<name>_audit.json --profile updated.json
    robo-advisor graph [--monitoring]
    robo-advisor questionnaire
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from .agents.advisory import build_advisory_graph
from .agents.monitor import build_monitoring_graph
from .config import Settings, load_settings
from .data.providers import DataUnavailableError, make_provider
from .data.validation import validate
from .graph.engine import GraphHalted
from .models import ClientInput
from .optimization.methods import METHODS
from .report.html import audit_bundle, render


def _settings(args) -> Settings:
    data = {}
    if getattr(args, "provider", None):
        data["provider"] = args.provider
    if getattr(args, "risk_free", None):
        data["risk_free_source"] = args.risk_free
    return load_settings(getattr(args, "config", None), {"data": data} if data else None)


def _provider(s: Settings):
    d = s.data
    return make_provider(d.provider, csv_dir=d.csv_dir, cache_dir=d.cache_dir, retries=d.request_retries,
                         refresh_hours=d.cache_refresh_hours)


def _as_of(value: str | None) -> dt.date | None:
    if not value:
        return None
    return dt.date.today() if value == "today" else dt.date.fromisoformat(value)


def _read(path: str | Path) -> str:
    """Read a user-edited text file: UTF-8, tolerating the BOM some Windows editors add."""
    return Path(path).read_text(encoding="utf-8-sig")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "client"


# ------------------------------------------------------------------ interactive intake


def _ask(prompt: str, default=None, cast=str, choices=None):
    while True:
        d = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{d}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = cast(raw)
        except ValueError:
            print("  invalid value, try again")
            continue
        if choices and v not in choices:
            print(f"  choose one of: {', '.join(map(str, choices))}")
            continue
        return v


def _yes(prompt: str, default: bool) -> bool:
    return _ask(prompt + " (y/n)", "y" if default else "n", str.lower, ["y", "n", "yes", "no"]).startswith("y")


def interactive_client(s: Settings) -> ClientInput:
    print("\n== Step 1-2: Investment goal ==")
    name = _ask("Your name", "Client")
    has_target = _yes("Do you have a specific target amount you want to reach by a specific date?", True)
    goal: dict = {"has_target": has_target}
    if has_target:
        goal["target_amount"] = _ask("Target portfolio value ($)", cast=float)
        goal["target_date"] = _ask("Target date (YYYY-MM-DD)", cast=dt.date.fromisoformat)
    goal["initial_investment"] = _ask("Initial investment ($)", cast=float)
    goal["monthly_contribution"] = _ask("Monthly contribution ($)", 0.0, float)
    if not has_target:
        goal["horizon_years"] = _ask("Investment horizon (years)", cast=float)

    def block(qs, title, skip_derived):
        print(f"\n== Step 3: {title} ==")
        ans = {}
        for q in qs:
            if skip_derived and q.derive_from_goal:
                continue
            opts = list(q.options)
            print(f"{q.text}")
            for i, o in enumerate(opts, 1):
                print(f"   {i}. {o.replace('_', ' ')}")
            k = _ask("  choice", cast=int, choices=list(range(1, len(opts) + 1)))
            ans[q.id] = opts[k - 1]
        return ans

    cap = block(s.questionnaire.capacity, "Risk capacity (financial ability to bear losses)", True)
    tol = block(s.questionnaire.tolerance, "Risk tolerance (willingness to bear losses)", False)
    print("\n== Step 4: Investment universe ==")
    print("Available ETFs: " + ", ".join(s.universe.default))
    raw = _ask("ETFs to allow (comma-separated, blank = all)", "")
    universe = [t.strip().upper() for t in raw.split(",") if t.strip()] or None
    print("\n== Step 5: Constraints ==")
    allow_short = _yes("Are short sales allowed?", False)
    cons = {"allow_short": allow_short,
            "max_position": _ask("Maximum position size (fraction)", s.optimization.max_position, float)}
    if allow_short:
        cons["max_gross_leverage"] = _ask("Maximum gross exposure L", s.optimization.max_gross_leverage, float)
    taxes = {"enabled": _yes("Should taxes be incorporated?", False)}
    if taxes["enabled"]:
        t = s.tax
        taxes |= {"ordinary_rate": _ask("Marginal income-tax rate", t.ordinary_rate, float),
                  "qualified_dividend_rate": _ask("Qualified-dividend rate", t.qualified_dividend_rate, float),
                  "ltcg_rate": _ask("Long-term capital-gains rate", t.ltcg_rate, float),
                  "stcg_rate": _ask("Short-term capital-gains rate", t.stcg_rate, float),
                  "state_rate": _ask("State tax rate", t.state_rate, float)}
    prefs = {}
    if not has_target:
        prefs["optimization_method"] = _ask("Optimization method", s.optimization.default_method,
                                            choices=[m for m in METHODS if m != "target_return"])
    return ClientInput.model_validate({"profile": {"name": name}, "goal": goal, "capacity_answers": cap,
                                       "tolerance_answers": tol, "universe": universe, "constraints": cons,
                                       "taxes": taxes, "preferences": prefs})


# ------------------------------------------------------------------ commands


def cmd_run(args) -> int:
    s = _settings(args)
    client = interactive_client(s) if args.interactive else ClientInput.model_validate_json(_read(args.profile))
    if args.as_of:
        client = client.model_copy(update={"as_of": _as_of(args.as_of)})
    graph = build_advisory_graph(s, _provider(s))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(client.profile.name)
    try:
        res = graph.run({"client": client}, parallel=not args.sequential)
    except DataUnavailableError as e:
        print(f"\nMARKET DATA UNAVAILABLE: {e}\nCheck your internet connection (Yahoo Finance must be "
              "reachable), then retry; cached tickers are reused.", file=sys.stderr)
        return 3
    except GraphHalted as e:
        print(f"\nWORKFLOW HALTED at {e.node}: {e}", file=sys.stderr)
        audit = {"halted_at": e.node, "reason": str(e),
                 "reviews": [{"stage": r.stage, "findings": [vars(f) for f in r.findings]} for r in e.result.reviews],
                 "trace": [vars(t) for t in e.result.trace]}
        (out / f"{slug}_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        return 2
    html_path, audit_path = out / f"{slug}_report.html", out / f"{slug}_audit.json"
    html_path.write_text(render(res, s, graph.to_mermaid()), encoding="utf-8")
    audit_path.write_text(json.dumps(audit_bundle(res), indent=2), encoding="utf-8")
    st = res.state
    print(st["explanation"].headline)
    print(st["explanation"].risk_text)
    print(st["explanation"].goal_text)
    print("\nAllocation:")
    for t, w in sorted(st["portfolio"].weight_map().items(), key=lambda x: -abs(x[1])):
        if abs(w) > 1e-6:
            print(f"  {t:<5} {w:7.2%}  ${w * client.goal.initial_investment:>12,.0f}  "
                  f"${w * client.goal.monthly_contribution:>9,.0f}/mo")
    for r in res.latest_reviews():
        print(r.summary())
    print(f"\nReport: {html_path}\nAudit:  {audit_path}")
    if args.timeline:
        print(res.timeline())
    return 0


def cmd_monitor(args) -> int:
    s = _settings(args)
    prior = json.loads(_read(args.prior))
    client = (ClientInput.model_validate_json(_read(args.profile)) if args.profile
              else ClientInput.model_validate(prior["client"]))
    res = build_monitoring_graph(s, _provider(s)).run({"client": client, "prior": prior})
    rep = res.state["monitoring"]
    print(f"Monitoring {rep.prior_as_of} -> {rep.as_of}")
    print(json.dumps(rep.metrics, indent=2, default=str))
    if rep.triggers:
        print("\nREVIEW REQUIRED:")
        for t in rep.triggers:
            print(f"  [{t.code}] {t.message}")
    else:
        print("\nNo review triggers.")
    return 1 if rep.triggers else 0


def cmd_data(args) -> int:
    """Download (or read from cache) and validate market data without running an analysis."""
    s = _settings(args)
    as_of = _as_of(args.as_of) or dt.date.today()
    while as_of.weekday() >= 5:
        as_of -= dt.timedelta(days=1)
    try:
        start = as_of.replace(year=as_of.year - s.data.lookback_years)
    except ValueError:
        start = as_of.replace(year=as_of.year - s.data.lookback_years, day=28)
    tickers = [t.upper() for t in args.tickers] if args.tickers else list(s.universe.default)
    tickers = sorted(set(tickers) | {s.data.benchmark, s.data.risk_free_ticker})
    prov = _provider(s)
    print(f"Fetching {len(tickers)} tickers from {prov.name} ({start} to {as_of}) ...")
    try:
        frames = prov.fetch(tickers, start, as_of)
    except DataUnavailableError as e:
        print(f"MARKET DATA UNAVAILABLE: {e}", file=sys.stderr)
        return 3
    dq = validate(frames, start, as_of, s.data)
    print(f"\n{'ETF':<6}{'first':>12}{'last':>12}{'years':>7}{'missing':>9}{'splits':>8}{'dists':>7}"
          f"{'adj err':>10}  status")
    for t, q in dq.tickers.items():
        status = "BLOCKED" if any(b.startswith(f"{t}:") for b in dq.blocking) else (
            "short history" if not q.meets_min_history else "ok")
        print(f"{t:<6}{str(q.first_obs):>12}{str(q.last_obs):>12}{q.years_available:>7.1f}{q.missing_days:>9}"
              f"{q.n_splits:>8}{q.n_distributions:>7}{q.adj_consistency_max_error:>10.1e}  {status}")
    for w in dq.warnings:
        print(f"  warning: {w}")
    for b in dq.blocking:
        print(f"  BLOCKER: {b}")
    if s.data.risk_free_source == "fred" and not prov.synthetic:
        from .data.providers import fetch_treasury_rate
        try:
            rf = fetch_treasury_rate(s.data.fred_series, start, as_of, s.data.cache_dir)
            print(f"\nRisk-free: FRED {s.data.fred_series}, {len(rf)} observations, mean {rf.mean():.2%}")
        except DataUnavailableError as e:
            print(f"\nRisk-free: {e} (the T-bill ETF proxy will be used)")
    print("\nData OK." if not dq.blocking else "\nData has blocking issues (see above).")
    return 0 if not dq.blocking else 2


def cmd_graph(args) -> int:
    s = load_settings()
    g = (build_monitoring_graph if args.monitoring else build_advisory_graph)(s, make_provider("synthetic"))
    print(g.to_mermaid())
    print("\n%% execution waves: " + " | ".join(", ".join(w) for w in g.waves()))
    return 0


def cmd_questionnaire(args) -> int:
    s = load_settings(getattr(args, "config", None))
    for block in ("capacity", "tolerance"):
        print(f"\n# {block}")
        for q in getattr(s.questionnaire, block):
            print(f"- {q.id} (weight {q.weight}): {q.text}\n    options: {q.options}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="robo-advisor", description="Goal-based robo-advisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="produce a recommendation")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--profile", help="client profile JSON")
    src.add_argument("--interactive", action="store_true", help="answer the questionnaire in the terminal")
    r.add_argument("--out", default="out")
    r.add_argument("--timeline", action="store_true", help="print the agent execution timeline")
    r.add_argument("--sequential", action="store_true", help="disable parallel agent execution")
    m = sub.add_parser("monitor", help="re-assess a prior recommendation")
    m.add_argument("--prior", required=True, help="audit JSON from a previous run")
    m.add_argument("--profile", help="updated client profile JSON (initial_investment = current value)")
    d = sub.add_parser("data", help="download/validate market data (no analysis)")
    d.add_argument("--tickers", nargs="*", help="default: the full ETF universe")
    for p in (r, m, d):
        p.add_argument("--config", help="YAML overriding config/default.yaml")
        p.add_argument("--provider", choices=["synthetic", "csv", "yahoo"])
        p.add_argument("--risk-free", choices=["etf", "fred"], help="risk-free source (default: config)")
    for p in (r, d):
        p.add_argument("--as-of", help="analysis date YYYY-MM-DD or 'today' (overrides the profile)")
    gp = sub.add_parser("graph", help="print the workflow graph (mermaid)")
    gp.add_argument("--monitoring", action="store_true")
    q = sub.add_parser("questionnaire", help="print the configured questionnaire")
    q.add_argument("--config")
    args = ap.parse_args(argv)
    # never crash on a console that cannot display a character (e.g. legacy Windows code pages)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    return {"run": cmd_run, "monitor": cmd_monitor, "data": cmd_data, "graph": cmd_graph,
            "questionnaire": cmd_questionnaire}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
