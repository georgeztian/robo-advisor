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

```bash
pip install -e ".[dev]"            # add ".[yahoo]" for live Yahoo Finance data
robo-advisor run --profile examples/client_target.json --timeline
robo-advisor run --profile examples/client_no_target.json
robo-advisor run --interactive     # answer the questionnaire in the terminal
open out/alex_target_report.html   # dashboard; out/alex_target_audit.json holds the audit trail
```

Ongoing monitoring (spec §18) works from a prior audit bundle. In the updated profile,
`initial_investment` is the current portfolio value.

```bash
robo-advisor monitor --prior out/alex_target_audit.json --profile updated_profile.json
```

Other commands:
- `robo-advisor graph` prints the workflow graph as mermaid.
- `robo-advisor questionnaire` prints the configured questions and scores.
- `python tools/extract_docx.py robo-advisor.docx` re-extracts the spec, including OMML equations.

## Data

| Provider | Use |
|---|---|
| `synthetic` (default) | Deterministic simulated histories, calibrated to each ETF's real inception date, typical risk/return, correlations, distributions, splits and three stress episodes. Used for offline demos and tests. **Every report built on it is watermarked SYNTHETIC.** |
| `yahoo` | Live adjusted histories via `yfinance` (`pip install -e ".[yahoo]"`, `--provider yahoo`), cached as CSV. |
| `csv` | `data/prices/<TICKER>.csv` with `date, close, adj_close[, dividend, split_ratio]`. |

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
