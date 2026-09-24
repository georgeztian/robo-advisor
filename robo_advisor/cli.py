"""Command-line interface (run it through the ./ra or .\\ra launcher).

    run --interactive | --profile FILE   produce a recommendation (report + audit + profile)
    data                                 download and validate market data only
    monitor --prior AUDIT [--profile F]  re-assess an earlier recommendation
    etfs                                 print the ETF menu by category
    questionnaire                        print the risk questions and allowed answers
    graph [--monitoring]                 print the agent workflow graph (mermaid)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
from .optimization.methods import GOAL_RISK_METRICS, METHOD_DESCRIPTIONS
from .universe import CATALOG
from .report.html import audit_bundle, render


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _in_project(path: str) -> str:
    """Relative data paths (price cache, CSV folder) live in the project folder, whichever folder
    the command is started from."""
    p = Path(path)
    if p.is_absolute() or not (PROJECT_ROOT / "pyproject.toml").exists():
        return str(p)
    return str(PROJECT_ROOT / p)


def _settings(args) -> Settings:
    data = {}
    if getattr(args, "provider", None):
        data["provider"] = args.provider
    if getattr(args, "risk_free", None):
        data["risk_free_source"] = args.risk_free
    s = load_settings(getattr(args, "config", None), {"data": data} if data else None)
    s.data.cache_dir, s.data.csv_dir = _in_project(s.data.cache_dir), _in_project(s.data.csv_dir)
    return s


def _provider(s: Settings):
    d = s.data
    return make_provider(d.provider, csv_dir=d.csv_dir, cache_dir=d.cache_dir,
                         retries=d.request_retries, refresh_hours=d.cache_refresh_hours)


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


def _etf_line(i: int, t: str) -> str:
    info = CATALOG[t]
    flag = "  [special risk]" if info.risk_note else ""
    return f"   {i}. {t:<5} {info.name} (since {info.inception.year}, fee {info.expense_ratio:.2%}){flag}"


def choose_etfs(s: Settings) -> list[str]:
    """Step 4: the client picks ETFs category by category."""
    cats = s.universe.categories
    print(f"\n== Step 4: Investment universe ({len(s.universe.tickers)} ETFs in {len(cats)} categories) ==")
    print("For each category, type the numbers of the ETFs to include (e.g. 1,3), 'all' for the whole "
          "category, or press Enter to skip it.")
    while True:
        chosen: list[str] = []
        for cat, tickers in cats.items():
            print(f"\n{cat}")
            for i, t in enumerate(tickers, 1):
                print(_etf_line(i, t))
            while True:
                raw = input("  include: ").strip().lower()
                if not raw:
                    break
                if raw == "all":
                    chosen += tickers
                    break
                try:
                    picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
                except ValueError:
                    picks = []
                if picks and all(1 <= k <= len(tickers) for k in picks):
                    chosen += [tickers[k - 1] for k in picks if tickers[k - 1] not in chosen]
                    break
                print(f"  enter numbers between 1 and {len(tickers)} separated by commas, 'all', or Enter")
        if chosen:
            print("\nSelected: " + ", ".join(chosen))
            return chosen
        print("\nSelect at least one ETF.")


def _menu(title: str, options: dict[str, str], default: str) -> str:
    keys = list(options)
    print(title)
    for i, k in enumerate(keys, 1):
        print(f"   {i}. {options[k]}" + ("  [default]" if k == default else ""))
    k = _ask("  choice", keys.index(default) + 1, int, list(range(1, len(keys) + 1)))
    return keys[k - 1]


def _fraction(prompt: str, default: float) -> float:
    while True:
        v = _ask(prompt, default, float)
        if 0 < v <= 1:
            return v
        print("  enter a fraction between 0 and 1, e.g. 0.25 for 25%")


def _rate(prompt: str, default: float) -> float:
    while True:
        v = _ask(prompt, default, float)
        if 0 <= v < 1:
            return v
        print("  enter a decimal between 0 and 1, e.g. 0.24 for 24%")


def _at_least_one(prompt: str, default: float) -> float:
    while True:
        v = _ask(prompt, default, float)
        if v >= 1:
            return v
        print("  enter a number of at least 1, e.g. 1.5")


def _positive(prompt: str) -> float:
    while True:
        v = _ask(prompt, cast=float)
        if v > 0:
            return v
        print("  enter a number greater than 0")


def _amount(prompt: str, default: float | None = None) -> float:
    while True:
        v = _ask(prompt, default, float)
        if v >= 0:
            return v
        print("  enter a non-negative amount")


def _ask_category_limits(s: Settings, universe: list[str]) -> dict[str, float]:
    """Show the category limits that apply to the chosen ETFs; let the client change them."""
    cats = [c for c, ts in s.universe.categories.items() if any(t in universe for t in ts)]
    defaults = s.optimization.category_limits
    print("\nCategory limits (maximum share of the portfolio per category):")
    for c in cats:
        print(f"   {c:<36} {defaults[c]:.0%}" if c in defaults else f"   {c:<36} no limit")
    if _yes("Keep these category limits?", True):
        return {}
    changed = {}
    for c in cats:
        v = _fraction(f"  Max share for {c} (1 = no limit)", defaults.get(c, 1.0))
        if v != defaults.get(c, 1.0):
            changed[c] = v
    return changed


def _ask_optimizer(s: Settings, has_target: bool) -> dict:
    print("\n== Step 6: How the portfolio is optimized ==")
    prefs: dict = {}
    if has_target:
        print("With a target, the optimizer picks the LOWEST-RISK portfolio that reaches the target with\n"
              "the probability you require, within your risk limit.")
        p = _ask("Required probability of reaching the target, in %",
                 round(s.optimization.goal.target_probability * 100), float)
        prefs["target_probability"] = min(max(p, 1.0), 99.0) / 100
        prefs["goal_risk_metric"] = _menu("How should risk be measured?", GOAL_RISK_METRICS,
                                          s.optimization.goal.risk_metric)
        return prefs
    print("Without a target, choose how the portfolio is built. Every option stays within your risk\n"
          "limit and the position / category limits.")
    prefs["optimization_method"] = _menu("Optimization method:", METHOD_DESCRIPTIONS,
                                         s.optimization.default_method)
    if prefs["optimization_method"] == "target_return":
        prefs["target_return"] = _ask("Target annual return (e.g. 0.06 for 6%)", cast=float)
    return prefs


def interactive_client(s: Settings) -> ClientInput:
    print("\n== Step 1-2: Investment goal ==")
    name = _ask("Your name", "Client")
    has_target = _yes("Do you have a specific target amount you want to reach by a specific date?", True)
    goal: dict = {"has_target": has_target}
    if has_target:
        goal["target_amount"] = _positive("Target portfolio value ($)")
        while True:
            d = _ask("Target date (YYYY-MM-DD)", cast=dt.date.fromisoformat)
            if d > dt.date.today() + dt.timedelta(days=31):
                break
            print("  the target date must be more than a month from today")
        goal["target_date"] = d
    goal["initial_investment"] = _amount("Initial investment ($)")
    goal["monthly_contribution"] = _amount("Monthly contribution ($)", 0.0)
    if not has_target:
        goal["horizon_years"] = _positive("Investment horizon (years)")

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
    universe = choose_etfs(s)
    print("\n== Step 5: Constraints ==")
    allow_short = _yes("Are short sales allowed?", False)
    cons = {"allow_short": allow_short,
            "max_position": _fraction("Maximum position size per ETF (fraction, e.g. 0.5 for 50%)",
                                      s.optimization.max_position)}
    if allow_short:
        cons["max_gross_leverage"] = _at_least_one("Maximum gross exposure L (e.g. 1.5 = 150%)",
                                                   s.optimization.max_gross_leverage)
    limits = _ask_category_limits(s, universe)
    if limits:
        cons["category_limits"] = limits
    taxes = {"enabled": _yes("Should taxes be incorporated?", False)}
    if taxes["enabled"]:
        t = s.tax
        print("Enter tax rates as decimals, e.g. 0.24 for 24%.")
        taxes |= {"ordinary_rate": _rate("Marginal income-tax rate", t.ordinary_rate),
                  "qualified_dividend_rate": _rate("Qualified-dividend rate", t.qualified_dividend_rate),
                  "ltcg_rate": _rate("Long-term capital-gains rate", t.ltcg_rate),
                  "stcg_rate": _rate("Short-term capital-gains rate", t.stcg_rate),
                  "state_rate": _rate("State tax rate", t.state_rate)}
    prefs = _ask_optimizer(s, has_target)
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
    if args.interactive:                       # keep the answers: rerun or edit them with --profile
        saved = out / f"{slug}_profile.json"
        saved.write_text(client.model_dump_json(indent=2, exclude={"as_of"}), encoding="utf-8")
        print(f"\nSaved your answers to {saved} (rerun with --profile {saved})")
    try:
        res = graph.run({"client": client}, parallel=not args.sequential)
    except GraphHalted as e:
        halted = out / f"{slug}_halted.json"         # never overwrite the last good audit
        record = {"halted_at": e.node, "reason": str(e),
                  "reviews": [{"stage": r.stage, "findings": [vars(f) for f in r.findings]} for r in e.result.reviews],
                  "trace": [vars(t) for t in e.result.trace]}
        halted.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"\nWORKFLOW HALTED at {e.node}: {e}\nDetails: {halted}", file=sys.stderr)
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
    if not all(k in prior for k in ("client", "portfolio", "risk", "estimates", "as_of")):
        raise ValueError(f"{args.prior} is not the audit file of a completed recommendation "
                         "(use the <name>_audit.json written by a successful run)")
    client = (ClientInput.model_validate_json(_read(args.profile)) if args.profile
              else ClientInput.model_validate(prior["client"]))
    # re-assess as of today (or --as-of), not as of the prior recommendation's date
    client = client.model_copy(update={"as_of": _as_of(args.as_of)})
    try:
        res = build_monitoring_graph(s, _provider(s)).run({"client": client, "prior": prior})
    except GraphHalted as e:
        print(f"\nMONITORING HALTED at {e.node}: {e}", file=sys.stderr)
        return 2
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
    tickers = [t.upper() for t in args.tickers] if args.tickers else list(s.universe.tickers)
    unknown = [t for t in tickers if t not in CATALOG]
    if unknown:
        raise ValueError(f"unknown ticker(s) {unknown}; offered ETFs: {', '.join(s.universe.tickers)}")
    tickers = sorted(set(tickers) | {s.data.benchmark, s.data.risk_free_ticker})
    prov = _provider(s)
    print(f"Fetching {len(tickers)} tickers from {prov.name} ({start} to {as_of}) ...")
    frames = prov.fetch(tickers, start, as_of)
    dq = validate(frames, start, as_of, s.data)
    dq.warnings.extend(getattr(prov, "notes", []))
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


def cmd_etfs(args) -> int:
    """Print the ETF menu (categories) for filling in a profile's "universe"."""
    s = load_settings(getattr(args, "config", None))
    print(f"{len(s.universe.tickers)} ETFs in {len(s.universe.categories)} categories. In a profile, list tickers "
          "and/or whole category names under \"universe\".")
    for cat, tickers in s.universe.categories.items():
        print(f"\n{cat}")
        for i, t in enumerate(tickers, 1):
            print(_etf_line(i, t))
    print("\n[special risk] = leveraged, option-income or crypto ETF; its risks are disclosed in the report.")
    return 0


def cmd_questionnaire(args) -> int:
    s = load_settings(getattr(args, "config", None))
    for block in ("capacity", "tolerance"):
        print(f"\n# {block}")
        for q in getattr(s.questionnaire, block):
            note = "  [answered automatically from the goal; leave it out of profiles]" if q.derive_from_goal else ""
            print(f"- {q.id} (weight {q.weight}): {q.text}{note}\n    options: {q.options}")
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
    for p in (r, d, m):
        p.add_argument("--as-of", help="analysis date YYYY-MM-DD or 'today' (overrides the profile)")
    gp = sub.add_parser("graph", help="print the workflow graph (mermaid)")
    gp.add_argument("--monitoring", action="store_true")
    q = sub.add_parser("questionnaire", help="print the configured questionnaire")
    q.add_argument("--config")
    e = sub.add_parser("etfs", help="print the ETF menu by category")
    e.add_argument("--config")
    args = ap.parse_args(argv)
    # never crash on a console that cannot display a character (e.g. legacy Windows code pages)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    command = {"run": cmd_run, "monitor": cmd_monitor, "data": cmd_data, "graph": cmd_graph,
               "questionnaire": cmd_questionnaire, "etfs": cmd_etfs}[args.cmd]
    if os.environ.get("ROBO_ADVISOR_DEBUG"):
        return command(args)
    try:
        return command(args)
    except BrokenPipeError:                   # output piped into e.g. `head` / `more` that closed early
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except DataUnavailableError as e:
        print(f"\nMARKET DATA UNAVAILABLE: {e}\nCheck your internet connection (Yahoo Finance must be "
              "reachable), then retry; tickers already downloaded are reused.", file=sys.stderr)
        return 3
    except FileNotFoundError as e:
        print(f"\nFILE NOT FOUND: {e.filename or e}", file=sys.stderr)
        return 2
    except ValueError as e:                   # invalid profile, answers, ETF choice, infeasible limits
        print(f"\nINPUT PROBLEM: {e}\n(set ROBO_ADVISOR_DEBUG=1 to see the full traceback)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
