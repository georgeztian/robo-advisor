# robo-advisor

An end-to-end, goal-based robo-advisor in Python, implementing the specification in
`robo-advisor.docx`. The extracted text, with every Word equation converted to LaTeX, is in
[`docs/SPEC.md`](docs/SPEC.md). It covers:

- **Risk profiling.** Risk capacity and risk tolerance are scored separately, and the mapped
  score is min(capacity, tolerance). The score maps to a volatility limit through bands that
  are configurable, not hard-coded.
- **Investment universe.** The client picks any subset of 16 ETFs (VOO, VTI, BND, TLT, BIL,
  GLD, QQQ, TQQQ, VXUS, VT, VWO, VNQ, SCHH, SCHD, VYM, DGRO). SGOV is an optional extra.
- **Data validation.** Inception dates, 20-year coverage, gaps, and whether adjusted prices
  agree with splits and distributions. Look-ahead is guarded against.
- **Estimation.** Expected return, volatility, covariance, drawdown, VaR/CVaR, beta, Sharpe
  and Sortino, from up to 20 years of history.
- **Constraints and taxes.** Short sales with a gross-leverage limit, a maximum position size,
  and an optional estimated tax model.
- **Optimization.** *Target* clients get the minimum-risk portfolio that reaches
  P(F_T ≥ F\*) ≥ p. *No-target* clients get the maximum expected return within their risk
  limit. Also available: minimum volatility, maximum Sharpe, CVaR, target-return, risk parity
  and maximum diversification.
- **Projection.** A 10,000-path Monte Carlo with contributions, rebalancing and taxes;
  scenario analysis; a deterministic FV; and a 10-year S&P 500 comparison using the same
  contributions.
- **Client output.** A self-contained HTML dashboard with explanations, plus a JSON audit
  bundle and ongoing monitoring.

The work is split among **agents** in a contract-checked **workflow graph** that runs
independent steps in parallel. An **independent reviewer agent** checks every stage against
the spec. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

New to Python? Follow the step-by-step guide in [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

Everything runs through the `ra` launcher (Windows: `.\ra`). Installation is automatic:
- On the first run, `ra` creates a private Python environment in `.venv` and installs the app
  into it, with Yahoo Finance support and the test tools.
- On every later run it checks that environment. It reinstalls if the environment is damaged,
  and updates it if the dependency list in `pyproject.toml` changed (e.g. after `git pull`).

There is nothing to activate. `./setup.sh` / `.\setup.bat` installs up front if you prefer,
and `--fresh` rebuilds the environment from scratch.

```bash
./ra run --profile examples/client_target.json --timeline      # Windows: .\ra ...
./ra run --profile examples/client_no_target.json
./ra run --interactive                                         # answer the questionnaire
./ra data --provider yahoo                                     # download + validate real prices
./ra run --profile examples/client_target.json --provider yahoo --as-of today
./ra test                                                      # run the test suite
open out/alex_target_report.html   # the dashboard; out/alex_target_audit.json holds the audit trail
```

If you manage environments yourself, `pip install -e ".[yahoo,dev]"` followed by
`robo-advisor <command>` works as well.

Ongoing monitoring (spec §18) works from a prior audit bundle. In the updated profile,
`initial_investment` is the current portfolio value.

```bash
./ra monitor --prior out/alex_target_audit.json --profile updated_profile.json
```

Other commands:
- `./ra graph` prints the workflow graph as mermaid.
- `./ra questionnaire` prints the configured questions and scores.
- `python tools/extract_docx.py robo-advisor.docx` re-extracts the spec, including OMML equations.

## Data

| Provider | Use |
|---|---|
| `synthetic` (default) | Deterministic simulated histories, calibrated to each ETF's real inception date, typical risk/return, correlations, distributions, splits and three stress episodes. Used for offline demos and tests. **Every report built on it is watermarked SYNTHETIC.** |
| `yahoo` | Real daily histories via `yfinance` (`pip install -e ".[yahoo]"`, `--provider yahoo`), cached as CSV. See below. |
| `csv` | `data/prices/<TICKER>.csv` with `date, close, adj_close[, dividend, split_ratio]`. |

## Using real market data

The default provider is `synthetic`, so the project runs offline. For real prices:

```bash
./ra data --provider yahoo                                  # download, cache and validate the ETFs
./ra run --profile examples/client_target.json --provider yahoo --as-of today
./ra run --profile examples/client_target.json --provider yahoo --risk-free fred   # Treasury rate
```

To make Yahoo the default, set `data.provider: yahoo` (and optionally `risk_free_source: fred`)
in a YAML file and pass it with `--config`.

How Yahoo data is handled (`robo_advisor/data/providers.py`):
- **Download.** It fetches full daily history with unadjusted close, adjusted close,
  dividends, splits and fund capital-gain distributions. yfinance's price repair is switched
  on, which fixes 100× errors and bad dividend adjustments.
- **Normalization.** Split-adjusted prices and dividends are converted back to raw values,
  so the validator can check them against the adjusted series. A dividend Yahoo reports on a
  non-trading day moves to the next trading day. Duplicate rows are dropped, and timezones are
  removed.
- **Retries.** Rate limits and network errors are retried with exponential backoff.
- **Cache.** Each ticker is cached in `.cache/prices/<TICKER>.csv`, so repeat runs work
  offline. A cache that ends before the requested date is refreshed, at most once every
  12 hours. Delete the folder to force a full re-download.
- **Validation.** Validation follows the NYSE trading calendar. An isolated bad day in
  Yahoo's adjusted prices is reported as a warning. A missed split adjustment, or errors on
  many days, blocks the run.
- **Risk-free rate.** With `--risk-free fred`, the risk-free rate is the FRED 3-month T-bill
  rate (series `DTB3`). If FRED can't be reached, it falls back to the BIL T-bill ETF.

`tests/test_yahoo_provider.py` checks this whole path offline. It feeds the real provider code
a fake yfinance that reproduces Yahoo's format and quirks (see `tests/fake_yfinance.py`), then
runs the full workflow, reviewer included.

If `./ra data` reports `MARKET DATA UNAVAILABLE`, Yahoo was not reachable from your
machine: check your connection or proxy and retry. Any tickers already cached are reused.

## Configuration

All cutoffs, limits and assumptions live in
[`robo_advisor/config/default.yaml`](robo_advisor/config/default.yaml): risk bands,
questionnaire weights, the target probability p, the leverage limit L, the maximum position,
Monte Carlo paths, rebalancing, tax rates, scenario shifts, monitoring triggers and reviewer
tolerances. To override any of them, pass `--config my.yaml`; it is deep-merged over the
defaults.

## Tests

```bash
pytest -q
```

The suite covers:
- unit tests for every module;
- graph-engine contracts, parallelism, remediation and halting;
- reviewer tampering tests, where each injected violation must be blocked;
- a reviewer import-isolation test;
- end-to-end runs of both example clients plus monitoring and the CLI.

> Model outputs are historical estimates and model assumptions, not guarantees. The tax
> module is an estimated model, not tax advice. Nothing here is personalized investment advice.
