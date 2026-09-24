# Getting started (no programming experience needed)

This guide takes you from nothing installed to a personal recommendation report built on real
market data.

**The terminal** is the app where you type commands: *Terminal* on a Mac, *PowerShell* on
Windows. Type each command exactly as shown and press Enter.

Commands are shown for both systems:

- **Mac / Linux:** `./ra …`
- **Windows:** `.\ra …`

---

## Part 1: One-time installation (about 15 minutes)

### Step 1: Install Python
1. Download Python 3 (version 3.10 or newer) from https://www.python.org/downloads/ and run
   the installer.
   - **Windows:** on the first screen, tick **"Add python.exe to PATH"** before clicking
     *Install*.
2. Close and reopen your terminal.

### Step 2: Install Git (the tool that downloads the project)
Download it from https://git-scm.com/downloads and install it with the default options.

### Step 3: Download the project
```
cd Documents
git clone https://github.com/georgeztian/robo-advisor.git
cd robo-advisor
```
If the repository is private, Git will ask you to sign in to GitHub.

Without Git, you can instead open the repository page, click **Code → Download ZIP**, unzip
it into *Documents*, and `cd` into the unzipped folder.

### Step 4: Run the setup script
| Mac / Linux | Windows |
|---|---|
| `./setup.sh` | `.\setup.bat` |

The setup script does four things:

1. **Finds Python** 3.10 or newer on your computer. If it can't, it tells you what to install.
2. **Creates a private Python environment** in a hidden `.venv` folder inside the project.
   This keeps the app's libraries separate from anything else on your computer; deleting the
   folder removes them completely.
3. **Installs the app** and its libraries into that environment: the numerical libraries,
   Yahoo Finance support and the test tools.
4. **Checks** that the app starts.

You never need to "activate" the environment. The `ra` launcher always uses it, and if
`.venv` is missing it runs the setup for you.

To confirm everything works (optional, about 1 minute):

| Mac / Linux | Windows |
|---|---|
| `./ra test` | `.\ra test` |

The expected result is `92 passed`.

---

## Part 2: First run on demo data (no internet needed)

### Step 5: Run the example client
| Mac / Linux | Windows |
|---|---|
| `./ra run --profile examples/client_target.json` | `.\ra run --profile examples\client_target.json` |

After about 10 seconds you'll see the recommended allocation and the independent reviewer's
results. Then open the report:

| Mac | Windows |
|---|---|
| `open out/alex_target_report.html` | `start out\alex_target_report.html` |

This run uses **simulated** prices, and the report is marked SYNTHETIC. It only shows that
everything works.

---

## Part 3: Real market data

### Step 6: Download and check real prices
| Mac / Linux | Windows |
|---|---|
| `./ra data --provider yahoo` | `.\ra data --provider yahoo` |

This downloads about 20 years of daily history for the 16 ETFs, plus the S&P 500 benchmark
(VOO) and the T-bill proxy (BIL). It then checks the data and ends with `Data OK.`

The prices are saved in `.cache/prices/`, so later runs are fast and work offline.

### Step 7: Run the example client on real data
```
./ra run --profile examples/client_target.json --provider yahoo --as-of today
```
On Windows, write `.\ra` and `examples\client_target.json` instead. Open the report as in
Step 5: the SYNTHETIC banner is gone.

Optionally, add `--risk-free fred` to use the official 3-month Treasury-bill rate instead of
the BIL ETF.

---

## Part 4: Your own situation

### Step 8a: Answer the questions in the terminal (easiest)
```
./ra run --interactive --provider yahoo
```
The app asks for:

- your goal: a target amount and date, or no target;
- your initial investment and monthly contribution;
- the risk-capacity and risk-tolerance questions (type the number of your answer);
- which ETFs to allow;
- whether short selling is allowed;
- whether to include taxes.

### Step 8b: Or keep your answers in a file you can rerun
1. Copy an example profile to `my_profile.json`:
   - `examples/client_target.json` if you have a target amount and date;
   - `examples/client_no_target.json` if you don't.
2. Open `my_profile.json` in a plain-text editor (Notepad, TextEdit in plain-text mode, or
   VS Code) and change the values:
   - `name`
   - `goal`: `initial_investment` and `monthly_contribution`, plus either `target_amount` and
     `target_date` (YYYY-MM-DD), or `horizon_years`
   - `capacity_answers` and `tolerance_answers`: run `./ra questionnaire` to see every question
     and its allowed answers
   - `universe`: a list of the ETFs to allow, or `null` for all 16
   - `constraints` (for example `"allow_short": false`) and `taxes` (`"enabled": true` plus
     your tax rates)
3. Run:
   ```
   ./ra run --profile my_profile.json --provider yahoo --as-of today
   ```

### Step 9: Read the report
The report has these sections, top to bottom:

- **Key figures:** the probability of reaching your target and the median projected value.
- **Risk profile:** your capacity, tolerance and mapped scores, and the risk limit they imply.
- **Recommended portfolio:** each ETF's weight, the dollar amounts, and why it's included.
- **Financial projection:** the range of simulated outcomes and the conservative, base and
  optimistic scenarios.
- **S&P 500 comparison:** the last 10 years with the same money invested.
- **How it was made:** methodology, assumptions, limitations and disclosures.
- **Independent review:** every rule the reviewer agent checked.

The run also saves `out/<your_name>_audit.json`. Keep it for Step 10.

---

## Part 5: Later on

### Step 10: Periodic check-in (for example quarterly)
1. Copy your profile and set `initial_investment` to your portfolio's **current value**.
   Update anything else that changed (contribution, goal, answers).
2. Run:
   ```
   ./ra monitor --prior out/<your_name>_audit.json --profile my_profile_updated.json --provider yahoo
   ```
   It lists any review triggers, such as target at risk, risk limit exceeded, contribution
   changed or market risk shifted. If any appear, rerun Step 8b.

### Step 11: Every new session
Open a terminal and go to the project folder:
```
cd Documents/robo-advisor
```
Then use `./ra …` (Mac/Linux) or `.\ra …` (Windows). No activation is needed.

### Step 12: Get updates
```
git pull
```
Then re-run the setup script (Step 4), which updates the installed app in place.

### Optional: change the rules
All thresholds live in `robo_advisor/config/default.yaml`: risk bands, the 80% target
probability, the 50% position cap, tax rates and so on.

Rather than editing that file, create `my.yaml` with only the settings you want to change,
for example:
```yaml
optimization:
  max_position: 0.30
  goal: {target_probability: 0.90}
```
Then add `--config my.yaml` to your commands.

---

## Troubleshooting

| Message | What to do |
|---|---|
| `Python 3.10 or newer was not found` | Install Python (Step 1). On Windows, reinstall with "Add python.exe to PATH" ticked, then open a new terminal |
| `permission denied: ./ra` (Mac/Linux) | Run `chmod +x ra setup.sh` once |
| The environment seems broken | Rebuild it: `./setup.sh --fresh` or `.\setup.bat --fresh` |
| `MARKET DATA UNAVAILABLE` | Yahoo couldn't be reached. Check your internet, VPN or company firewall, wait a minute (rate limit) and retry. Tickers already downloaded are reused |
| `as_of ... is in the future` | Use `--as-of today` or a past date |
| `WORKFLOW HALTED at review_...` | The independent reviewer found a real problem. The message names the rule; for example, if no portfolio meets your risk limit with the chosen ETFs, allow more, such as BND or BIL |
| Odd data warnings | Delete the `.cache` folder and rerun `./ra data --provider yahoo` |

> The output is a model based on historical data, not personalized financial, legal or tax
> advice. Treat the recommendations as analysis to review, not orders to place.
