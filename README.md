# robo-advisor

A goal-based robo-advisor in Python. For each client it does the following:

- **Risk profile.** Scores risk *capacity* (the financial ability to take losses) and risk
  *tolerance* (the willingness to) separately, and turns them into a risk limit.
- **Portfolio.** Builds a portfolio from the ETFs the client chooses, category by category.
- **Projection.** Simulates 10,000 possible futures, compares the portfolio with the
  S&P 500 over the last 10 years, and explains the recommendation in a self-contained HTML
  report.

The work is done by a graph of cooperating **agents**. An **independent reviewer agent**
re-checks the data and the calculations, and checks every rule of the specification, before
a report is produced.

- Original specification, with its equations: [`docs/SPEC.md`](docs/SPEC.md).
- Technical design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

> Results are model estimates based on historical data, not guarantees or personalized
> investment, legal or tax advice.

---

## What it does

| Stage | What happens |
|---|---|
| Client input | Goal (a target amount by a date, or no target), initial investment, monthly contribution, risk-capacity and risk-tolerance questions, ETF choice by category, short selling, a minimum and maximum position for each ETF, tax preferences |
| Risk profile | Capacity and tolerance scores (0–100) are kept separate. **Mapped score = min(capacity, tolerance)**, which maps to a volatility limit (5 %–25 %, configurable) |
| Market data | Up to 20 years of real daily prices from Yahoo Finance, validated for gaps, splits, distributions and look-ahead |
| Optimization | **Target client:** the lowest-risk portfolio with at least an 80 % chance (configurable) of reaching the target. **No target:** one of 7 methods; the default is the highest expected return within the risk limit. Always subject to each ETF's **minimum and maximum position** and the **per-category limits**. The report writes out the optimization problem in mathematical form and in plain English |
| Projection | 10,000-path Monte Carlo with contributions, rebalancing and optional taxes; conservative/base/optimistic scenarios; a deterministic future value |
| Benchmark | The last 10 years against the S&P 500, with the same money invested |
| Review | An independent reviewer runs about 60 checks at 4 checkpoints and stops the run on any blocking error |
| Output | HTML report, JSON audit trail, and the client's saved answers; later check-ins raise review triggers |

### The ETF menu

| Category | ETFs |
|---|---|
| Equity ETFs | SPY, VOO, VTI, QQQ, VTV, VB |
| Bond ETFs | BND, TLT, HYG, VTEB, SCHR, SCHP, BNDX |
| Risk-free Short-term Treasury ETFs | BIL, SGOV |
| Commodity ETFs | GLD, SLV |
| International Equity ETFs | VXUS, IEFA, VWO |
| Real Estate ETFs | VNQ, SCHH |
| Dividend ETFs | SCHD, VYM, DGRO |
| Income ETFs | SPYI, QQQI, JEPI |
| Crypto ETFs | IBIT |

Clients choose from this menu category by category; nothing is included unless chosen.
`./ra etfs` always prints the current menu.
Historical data is analyzed using up to 20 years of daily prices. ETFs younger than 20 years use all the history they have and are flagged.

**Category limits** cap how much of the portfolio one category may take:

| Category | Default limit |
|---|---|
| Crypto ETFs | 5 % |
| Income ETFs | 25 % |
| Commodity ETFs | 20 % |
| Real Estate ETFs | 20 % |
| All other categories | no limit (only the per-ETF position limits) |

A client can change any of these, in the questionnaire or in the profile file
(`constraints.category_limits`). With short sales, the limit applies to the category's gross
exposure. If the chosen ETFs can't add up to 100 % under the limits, the run stops and says
so. The independent reviewer re-checks every limit.

**Position limits** set the smallest and largest share each chosen ETF may hold. By default
every ETF may hold between **0 %** (the optimizer may leave it out) and **50 %**. A client can
change them for each ETF: a positive minimum makes sure the ETF is held (and, with short sales,
that it is not shorted); the maximum also caps the size of a short position. The minimums
must add up to 100 % or less, and the maximums to at least 100 %.

### How the portfolio is optimized

| Client | Options |
|---|---|
| **Has a target** | The optimizer always finds the **lowest-risk** portfolio that reaches the target with the required probability (default 80 %). You choose the probability, and how risk is measured: **volatility** (default) or **tail risk** (CVaR: the average final value in the worst 5 % of simulated futures) |
| **No target** | Choose one method (the questionnaire shows this menu). All of them stay within the risk limit and the position / category limits |

| # | Method (`optimization_method`) | What it does |
|---|---|---|
| 1 | `mean_variance` (default) | Highest expected return within your risk limit; the classic approach |
| 2 | `min_volatility` | Smallest ups and downs possible, whatever the return |
| 3 | `max_sharpe` | Best return per unit of risk (Sharpe ratio) |
| 4 | `cvar` | Smallest average loss in the worst 5 % of months (tail-risk focus) |
| 5 | `target_return` | Smallest ups and downs that still earn a return you choose (set `target_return`, e.g. `0.06`) |
| 6 | `risk_parity` | Every ETF contributes the same share of total risk (balanced, long-only) |
| 7 | `max_diversification` | Most diversified mix: least overlap between the ETFs' movements (long-only) |

The menu is defined in `robo_advisor/config/default.yaml`, and each fund's facts are in
`robo_advisor/universe.py`.

---

## How to use it

### Step 1: Install Python and Git (once)
1. **Python 3.10 or newer:** https://www.python.org/downloads/
   - **Windows:** on the installer's first screen, tick **"Add python.exe to PATH"**.
2. **Git:** https://git-scm.com/downloads (the default options are fine).
3. Close and reopen the terminal.

### Step 2: Download the project (once)
```
cd Documents
git clone https://github.com/georgeztian/robo-advisor.git
cd robo-advisor
```
If you don't use Git: on the GitHub page, choose **Code → Download ZIP**, unzip it, and `cd`
into the unzipped folder (set working directory to the project folder).

### Step 3: Install private Python environment
| Mac / Linux | Windows |
|---|---|
| `./ra etfs` | `.\ra etfs` |

You never install anything by hand and never "activate" anything. Every `ra` command first
checks the app's private Python environment (the `.venv` folder):

- **Missing** (first run): it creates the environment and installs the app and its libraries,
  which takes a few minutes, once.
- **Damaged, or outdated** (the dependency list changed, e.g. after `git pull`): it
  reinstalls or updates it.
- **Up to date:** it starts right away.

This first command then prints the ETF menu. Seeing the menu means the installation worked.

### Step 4: Download and check the market data
```
./ra data
```
This downloads about 20 years of real daily prices from Yahoo Finance for every ETF on the menu.
Two of them are also always used behind the scenes, whether or not a client selects them: VOO
as the S&P 500 benchmark, and BIL as the T-bill risk-free proxy. It then validates the data and prints a table ending in `Data OK.`

The prices are cached in `.cache/prices/`, so later runs are fast and work offline. Re-run
this command whenever you want to check the data. Every analysis also refreshes the cache
automatically when it is out of date.

### Step 5: Enter a client's answers: choose **one** of the two routes

#### Route A: Answer the questions on the command line (easiest)
```
./ra run --interactive --out clients/client_name
```
The app asks, in order:

1. The client's name, and whether there is a **target** amount by a specific date. If yes, it
   asks for the target amount and date (YYYY-MM-DD).
2. The initial investment and monthly contribution in dollars, then, if there is no target,
   the investment horizon in years.
3. The **risk-capacity questions**, then the **risk-tolerance questions**. Type the number of
   the answer. The investment-horizon question is answered automatically from the goal.
4. **ETFs, category by category.** For each category, type the numbers of the ETFs
   to include (e.g. `1,3`), `all` for the whole category, or press Enter to skip it.
5. Whether short sales are allowed, then the **position limits**: keep 0 %–50 % for every ETF
   (Enter), or type `n` to set a minimum and maximum for each chosen ETF. It then shows the
   **category limits** that apply to the chosen ETFs, which you can keep (Enter) or change.
6. Whether to include taxes, and if so, the tax rates.
7. **How the portfolio is optimized** (see "How the portfolio is optimized" above):
   - with a target: the required probability of reaching it, and how risk is measured;
   - without a target: a numbered menu of all 7 methods (Enter picks the default).

Results go to the folder given by `--out` (use one folder per client):

| File | Contents |
|---|---|
| `client_name_report.html` | The recommendation report |
| `client_name_profile.json` | The client's answers, saved so you can edit and re-run them (Route B, step B4) |
| `client_name_audit.json` | The full audit trail; needed for later check-ins (Step 8) |

Here and below, `client_name` stands for the client's name as the app writes it in file names:
lower case, with spaces and punctuation replaced by `_`. For example, "Jane Doe" gives
`jane_doe_report.html`. If a run is stopped by the reviewer, the details go to
`client_name_halted.json`, and the last good report and audit are left untouched.

#### Route B: Fill out a profile file
**B1. Create the file**, for example `clients/client_name.json`. Create the `clients` folder first
if it doesn't exist (`mkdir clients`). Either:

- copy a profile saved by Route A (`…_profile.json`) and change it, or
- paste this skeleton (written for the default questions) into a new file and replace every
  `<…>` placeholder:

```json
{
  "profile": {"name": "<client name>"},
  "goal": {
    "has_target": true,
    "target_amount": "<amount, e.g. 500000>",
    "target_date": "<YYYY-MM-DD>",
    "initial_investment": "<amount>",
    "monthly_contribution": "<amount>"
  },
  "capacity_answers": {
    "annual_income": "<code>", "liquid_assets": "<code>", "investable_assets": "<code>",
    "emergency_savings": "<code>", "debt_obligations": "<code>", "income_stability": "<code>",
    "large_expenditures": "<code>", "portfolio_dependence": "<code>", "continue_investing": "<code>"
  },
  "tolerance_answers": {
    "decline_reaction": "<code>", "temporary_losses": "<code>", "stable_vs_volatile": "<code>",
    "equity_comfort": "<code>", "experience": "<code>", "crash_behavior": "<code>",
    "preserve_vs_growth": "<code>", "risk_attitude": "<code>"
  },
  "universe": ["<category name or ticker>", "<category name or ticker>"],
  "constraints": {"allow_short": false},
  "taxes": {"enabled": false},
  "preferences": {}
}
```
Numbers are written without quotes, `$` or commas (e.g. `"initial_investment": 150000`). For
a client **without a target**, set `"has_target": false`, delete `target_amount` and
`target_date`, and add `"horizon_years": <years>`. A placeholder left in place, or a misspelled
field name, is reported as an `INPUT PROBLEM` naming the field.

**B2. Look up the allowed answers and ETF names:**
```
./ra questionnaire      # every risk question, its answer codes and weights
./ra etfs               # the ETF menu by category
```

**B3. Edit the file** in a plain-text editor (Notepad, TextEdit in plain-text mode, VS Code):

| Field | What to enter |
|---|---|
| `profile.name` | Client name (also used for the output file names) |
| `goal.has_target` | `true` for a target amount by a date, `false` otherwise |
| `goal.target_amount`, `goal.target_date` | Only if `has_target` is `true`, e.g. `500000` and `"2036-12-31"` |
| `goal.horizon_years` | Only if `has_target` is `false`, e.g. `15` |
| `goal.initial_investment`, `goal.monthly_contribution` | Dollar amounts (numbers without `$` or commas) |
| `capacity_answers` | One answer code for each capacity question that `./ra questionnaire` lists, except `investment_horizon`, which is derived from the goal |
| `tolerance_answers` | One answer code for each tolerance question that `./ra questionnaire` lists |
| `universe` | **Required.** Tickers and/or whole categories, e.g. `["Bond ETFs", "Dividend ETFs", "SPY", "GLD"]` |
| `constraints` | `allow_short` (`true`/`false`); optional `min_position` and `max_position`, the limits for every ETF (defaults `0` and `0.5`); optional `position_limits` for individual ETFs, e.g. `{"BND": {"min": 0.1, "max": 0.3}, "IBIT": {"max": 0.02}}` (either key can be left out); `max_gross_leverage` when shorting; optional `category_limits`, e.g. `{"Crypto ETFs": 0.02, "Equity ETFs": 0.6}`, which overrides the defaults for the named categories (`1.0` removes a limit) |
| `taxes` | `{"enabled": false}`, or `enabled: true` with `ordinary_rate`, `qualified_dividend_rate`, `ltcg_rate`, `stcg_rate`, `state_rate` (decimals, e.g. `0.24`); optional `liquidate_at_horizon` (`true` to include the tax on selling everything at the end; default `false`) |
| `preferences` (optional) | **No target:** `optimization_method`, one of the 7 methods above (with `target_return` for method 5). **Target:** `target_probability` (e.g. `0.9`) and `goal_risk_metric` (`volatility` / `cvar`). **Both:** `rebalancing_type` (`calendar` / `threshold`), `rebalancing_frequency` (`monthly` / `quarterly` / `annual`), `rebalancing_threshold` (e.g. `0.05`) |

**B4. Run it:**
```
./ra run --profile clients/client_name.json --as-of today --out clients/client_name
```
`--as-of today` analyses with data up to today. You can give a past date instead, e.g.
`--as-of 2026-06-30`.

The profile saved by Route A (`clients/client_name/client_name_profile.json`) works the same way:
edit it and re-run with `--profile`.

### Step 6: Read the report
Open the report in your browser:

| Mac | Windows |
|---|---|
| `open clients/client_name/client_name_report.html` | `start clients\client_name\client_name_report.html` |

From top to bottom:

- **Key figures:** the probability of reaching the target, the median projected value,
  expected return and volatility.
- **Risk profile:** the capacity, tolerance and mapped scores, and the questionnaire scoring.
- **Recommended portfolio:** each ETF's category, weight, initial and monthly dollars, and why
  it was chosen; the allocation by category; ETFs considered but not held.
- **How the optimizer chose this allocation:** a simple explanation and the steps the optimizer
  took. Under **Additional optimization details** (click to expand): the problem in mathematical
  form with every constraint (and where the recommended portfolio sits against each one), what
  each symbol means (including the covariance matrix **V**, built from the ETFs' volatilities and
  correlations), and each ETF's minimum and maximum.
- **Financial projection:** the Monte Carlo range, the terminal-value distribution against the
  target, and the scenarios.
- **S&P 500 comparison:** the last 10 years with the same money invested.
- **How this recommendation was made:** methodology, assumptions, limitations, and disclosures (including the
  special-risk ETF notes). Expandable tables at the end: **Parameter estimates (historical)**,
  **Asset characteristics & data integrity** (each ETF's inception, history, fees and data
  checks), and **Tax rates** (each selected ETF's federal, state and combined rates on its
  distributions, long-term and short-term gains; all 0 % for a pre-tax analysis). The part of a
  distribution that is return of capital is not taxed when paid; it is taxed as a capital gain
  when the shares are sold.
- **Independent review:** every rule the reviewer checked, and the agent workflow trace.
- **Disclaimer:** the outputs are informational only, not financial, investment, legal or
  trading advice.

Optional run flags:
- `--risk-free fred` uses the official 3-month T-bill rate for the risk-free rate.
- `--timeline` prints each agent's execution time.

### Step 7: Adjust and re-run
Change the profile (amounts, answers, ETF choice, constraints) and repeat the Step 5 **B4**
command. Each run overwrites that client's report and audit.

### Step 8: Periodic check-in (e.g. quarterly)
1. Copy the client's profile and set `goal.initial_investment` to the portfolio's **current
   value**. Update anything else that changed.
2. Run:
   ```
   ./ra monitor --prior clients/client_name/client_name_audit.json --profile clients/client_name/client_name_updated.json
   ```
3. It reports the performance since the last recommendation, and any **review triggers**:
   goal changed, contribution changed, target at risk, risk limit exceeded, market risk
   (volatility or correlations) shifted, risk profile changed, or ETFs still held were dropped
   from the selection or are no longer on the ETF menu. If any fire, re-run Step 5 B4 with the
   updated profile.

   Without `--profile`, it re-checks the prior answers against today's data.

### Step 9: Update the app
```
git pull
```
The next `ra` command updates the environment automatically if the dependencies changed.

---

## Command reference

| Command | Purpose |
|---|---|
| `./ra run --interactive [options]` | Questionnaire on the command line → report, audit, saved profile |
| `./ra run --profile FILE [options]` | Run from a profile file → report, audit |
| `./ra data [--tickers T …]` | Download (or read from cache) and validate market data only |
| `./ra monitor --prior AUDIT [--profile FILE]` | Re-assess an earlier recommendation |
| `./ra etfs` | Print the ETF menu by category |
| `./ra questionnaire` | Print the risk questions and allowed answer codes |
| `./ra graph [--monitoring]` | Print the agent workflow graph (mermaid) |
| `./ra test` | Run the test suite |

Common options:

| Option | Meaning |
|---|---|
| `--provider yahoo\|csv\|synthetic` | Data source. The default is `yahoo` (real prices); `synthetic` is simulated data for offline testing |
| `--as-of YYYY-MM-DD\|today` | Analysis date for `run`, `data` and `monitor`; data after it is never used (default: today; for `run`, an `as_of` date written in the profile is used if present) |
| `--out FOLDER` | Where the report, audit and profile are written (default `out`) |
| `--risk-free etf\|fred` | Risk-free rate from the BIL ETF (default) or the FRED 3-month T-bill rate |
| `--config FILE.yaml` | Override settings (see Configuration) |

Exit codes:

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | Monitoring found review triggers (or an unexpected error with a traceback) |
| 2 | Input problem or missing file, blocking data issues (`data`), or the reviewer halted the workflow (the message names the rule) |
| 3 | Market data unavailable |

## Configuration

All thresholds live in [`robo_advisor/config/default.yaml`](robo_advisor/config/default.yaml):

- the ETF menu and its categories;
- the questionnaire (questions, answer scores and weights);
- the risk bands (score → volatility limit);
- the target probability (80 %), the default position limits (minimum 0 %, maximum 50 % per
  ETF), the category limits and the leverage limit;
- the number of Monte Carlo paths and the rebalancing rule;
- tax rates and scenario shifts;
- data-validation limits, monitoring triggers and reviewer tolerances.

Don't edit that file. Put only the settings you want to change in your own file and pass
`--config` with every command. A misspelled setting name is reported as an `INPUT PROBLEM`
naming it, rather than silently ignored:
```yaml
# my.yaml
data: {provider: yahoo, risk_free_source: fred}
optimization:
  max_position: 0.30
  goal: {target_probability: 0.90}
```
```
./ra run --profile clients/client_name.json --config my.yaml --as-of today --out clients/client_name
```

### Customization

| To change | Edit | What to do |
|---|---|---|
| Which ETFs are offered, and their categories | `universe.categories` in your `my.yaml` | List the tickers under each category. A list you give replaces that category's default list, so repeat the tickers you want to keep. A new category name adds a category; give it a limit under `optimization.category_limits` if it needs one. A ticker may be in only one category. A default category can't be removed from `my.yaml`. Every ticker needs a catalog entry (next row) |
| An ETF's facts: name, asset class, inception date, expense ratio | The `CATALOG` list in [`robo_advisor/universe.py`](robo_advisor/universe.py) | Edit its `ETFInfo(...)` line. The inception date is written `D(YYYY, M, D)`; `expense_ratio` is a decimal (`0.0003` = 0.03 %). To add an ETF, copy a similar line, change every field (including the distribution fields in the next row), then add the ticker to a category above. Optional `risk_note="…"` adds a special-risk disclosure to any report that holds the ETF |
| An ETF's distributions: how they are taxed | The same `ETFInfo(...)` line | `income_type`: `qualified` (stock dividends), `mixed` (part qualified), `reit`, `interest` (taxable bonds), `treasury` (federal tax only), `tax_exempt` (municipal bonds: state tax only) or `none` (no distributions). `roc_fraction` (optional, 0–1): the share that is return of capital, from the fund's tax-year reports. It is not taxed when paid; it lowers the cost basis, so it is taxed as a capital gain when the shares are sold. `qualified_fraction`: the share of the rest taxed at the qualified-dividend rate (0–1), from the fund's latest tax-year "QDI %"; the remainder is taxed as ordinary income. `collectible=True` for physical gold or silver (gains taxed at the collectibles rate) |
| Distribution amounts | Nothing for real data | Real dividends come with the market data (Yahoo, or the `dividend` column of your CSV files). For synthetic data only, an ETF's yield and payment frequency can be set in `_ETF_SPEC` in [`robo_advisor/data/providers.py`](robo_advisor/data/providers.py); an ETF not listed there gets a typical value for its type |
| Default tax rates | `tax:` in your `my.yaml` (defaults in `default.yaml`) | Set `ordinary_rate`, `qualified_dividend_rate`, `ltcg_rate`, `stcg_rate` and `state_rate` as decimals. They apply only to clients with `"taxes": {"enabled": true}`, and a rate in the client's `taxes` block overrides the default. Route A saves all five rates in the profile, so later changes here don't affect those profiles. `collectibles_rate` (gold and silver ETFs) can be set only here, and is capped at the client's ordinary rate |
| Risk capacity and risk tolerance questions | `questionnaire.capacity` and `questionnaire.tolerance` in your `my.yaml` | Each question has an `id`, `text`, `weight`, `options` (answer code: score 0–100) and optional `labels` (display text per code). Weights in each list must add up to 1.0. The list you give replaces the whole default list, so copy the full list from `default.yaml` and edit it. Keep the answer codes of the question marked `derive_from_goal` (`under_3y` … `over_20y`), which are filled in from the goal. Put quotes around the codes `"yes"` and `"no"` |
| How the risk score maps to a volatility limit | `risk_bands` in your `my.yaml` | Give the full list with `max_score`, `profile` and `max_volatility`, in increasing `max_score` order; the last band must reach 100 |

Notes:

- Put YAML changes in your own file and pass `--config my.yaml` with every command (see above).
  `./ra etfs --config my.yaml` and `./ra questionnaire --config my.yaml` show the result.
- Changes to `universe.py` apply to every run directly. It is part of the app's code, so
  `git pull` (Step 9) may stop if it conflicts with your edits: keep a copy of your lines first.
- If you add, remove or rename a question `id` or an answer code, update the client profiles to
  match. Otherwise the run stops with an `INPUT PROBLEM` listing what is missing or invalid.
- The settings and the catalog entries of the offered ETFs are checked on every run. A ticker
  with no catalog entry, an unknown `income_type`, an expense ratio of 0.05 or more (usually a
  percent typed by mistake), a tax rate outside 0–1, question weights that don't add up to 1.0,
  a repeated question `id`, a horizon question missing its answer codes, or a score outside
  0–100 is reported with a message naming the problem.

## Market data

| Provider | Use |
|---|---|
| `yahoo` (default) | Real daily prices via `yfinance` |
| `csv` | Your own files: `data/prices/<TICKER>.csv` with columns `date, close, adj_close[, dividend, split_ratio]` |
| `synthetic` | Simulated histories calibrated to each ETF, for offline testing only. Reports built on them are watermarked **SYNTHETIC** |

How Yahoo data is handled:

- **Download.** Full history of unadjusted and adjusted prices, dividends, splits and fund
  capital-gain distributions. yfinance's price repair is on; if its libraries are missing,
  data is downloaded unrepaired and the report notes it.
- **Normalization.** Timezones are removed, duplicate rows dropped, and dividends dated on
  non-trading days moved to the next trading day.
- **Retries.** Temporary errors such as rate limits are retried with backoff.
- **Cache.** Stored in `.cache/prices/` and refreshed when it falls behind the requested date.
  Delete the folder to force a full re-download.
- **Validation.** Follows the NYSE calendar. An isolated bad day in Yahoo's adjusted prices
  is a warning; a missed split adjustment, or systematic errors, stop the run.

## Project layout

```
ra, ra.bat                 launcher: runs the app inside .venv, installing/updating it automatically
setup.sh, setup.bat        the installer that ra calls (can also be run directly; --fresh rebuilds)
robo_advisor/
  config/default.yaml      all settings, including the ETF menu and the questionnaire
  universe.py              ETF facts: inception, fees, tax character, special risks
  questionnaire.py         risk capacity / tolerance scoring and mapping
  data/                    market-data providers (synthetic, Yahoo, CSV, FRED) and validation
  estimation.py, tax.py    return / risk estimation and the estimated tax model
  optimization/            constraints, 7 optimization methods, target-probability goal search,
                           and the report's write-up of the optimization problem
  simulation.py, projection.py, benchmark.py, rebalancing.py, monitoring.py
  explain.py, report/      explanations and the HTML report
  graph/engine.py          workflow-graph engine (parallel waves, review gates, provenance)
  agents/                  the agents and the advisory / monitoring graphs
  review/                  the independent reviewer (its own code, no production imports)
  cli.py                   the commands above
tests/                     automated tests (./ra test); tests/fixtures/ holds two test client profiles
docs/                      SPEC.md (the original specification), ARCHITECTURE.md
```

Client folders (`clients/`), the default output folder (`out/`), the data cache and `.venv`
are ignored by Git.
Client files contain personal financial data, so keep them out of the repository.

## Troubleshooting

| Message | What to do |
|---|---|
| `Python 3.10 or newer was not found` | Install Python (Step 1). On Windows, reinstall with "Add python.exe to PATH" ticked, then open a new terminal |
| `permission denied: ./ra` (Mac/Linux) | Run `chmod +x ra setup.sh` once |
| `INPUT PROBLEM: …` | The profile or answers are invalid, or the choice can't work. The message says why: a field, the `universe` (it lists the categories and ETFs), position or category limits the chosen ETFs can't satisfy, or no mix of the chosen ETFs meeting the client's risk limit. For that last one, add lower-risk ETFs, such as Bond or Risk-free Treasury ETFs |
| `MARKET DATA UNAVAILABLE` | Yahoo couldn't be reached. Check your internet, VPN or firewall, wait a minute (rate limit) and retry |
| `WORKFLOW HALTED at review_…` | The independent reviewer found a real problem, usually in the market data (e.g. an ETF with too little history on the analysis date). The message names the rule, and the details are in `client_name_halted.json` |
| The environment seems broken | `./setup.sh --fresh` (Windows: `.\setup.bat --fresh`) |
| Odd data warnings | Delete `.cache` and re-run `./ra data` |

## Limitations

- **Estimates, not forecasts.** Expected returns and risks come from history. Optimizers
  favour ETFs with short, strong histories (for example IBIT); the per-ETF and per-category
  limits contain this, but the estimates remain uncertain.
- **Estimated taxes.** The tax model is an estimate for a taxable account, not tax advice.
- **Model assumptions.** Monte Carlo returns are log-normal (or bootstrapped), which
  understates extreme events.
- **Your review.** A qualified professional should review the questionnaire scoring, the risk
  bands and the disclosures before the tool is used with real clients.

## Disclaimer

This robo-advisor provides automated, data-driven outputs for informational purposes only and does not constitute financial, investment, legal, or trading advice. All model insights are generated by algorithms based on historical data, which is not a guarantee of future performance or returns. Do not rely on this system as your sole source for making investment decisions. Investing involves risk, including the possible loss of principal. You are solely responsible for conducting your own research and should consult a licensed financial professional before making any investment choices.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
