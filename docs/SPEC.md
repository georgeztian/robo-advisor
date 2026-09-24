Fully Functional Robo-Advisor Workflow

1. Client Onboarding and Investment Questionnaire

The robo-advisor begins by collecting the information needed to determine the client's investment objective, financial circumstances, and risk profile.

A. Investment Goal

Ask:

Do you have a specific target amount that you want to reach by a specific date?

- Yes: collect:

- Target portfolio value

- Target date

- Initial investment

- Monthly contribution

The portfolio optimizer should determine the portfolio that minimizes investment risk while achieving the required investment objective, subject to the client's constraints.

- No: collect:

- Initial investment

- Monthly contribution

- Investment horizon

The optimizer should maximize expected return subject to the client's mapped risk constraint.

The target-date formulation should account for both the initial investment and recurring monthly contributions rather than treating the target as a single lump-sum investment.

2. Risk-Profiling Questionnaire

The questionnaire should separately measure:

- Risk Capacity — the client's financial ability to withstand losses.

- Risk Tolerance — the client's psychological willingness to tolerate losses and volatility.

These should be measured separately rather than combining all answers into one score.

A. Risk Capacity Questions

Examples:

- Annual income

- Liquid financial assets

- Total investable assets

- Emergency savings

- Debt obligations

- Stability of employment/income

- Expected large expenditures

- Investment horizon

- Dependence on this portfolio for near-term financial needs

- Ability to continue investing after a large market decline

Each answer is assigned a numerical risk-capacity score, for example from 0–100.

B. Risk Tolerance Questions

Examples:

- Reaction to a hypothetical 10%, 20%, or 40% portfolio decline

- Willingness to accept temporary losses

- Preference between stable lower returns and volatile higher expected returns

- Comfort with equity-heavy portfolios

- Previous investment experience

- Ability to remain invested during market crashes

- Preference for preserving principal versus pursuing higher growth

Each answer is assigned a numerical risk-tolerance score, for example from 0–100.

C. Mapping the Two Scores

Calculate:

Risk Capacity Score = weighted sum of capacity-question scores

Risk Tolerance Score = weighted sum of tolerance-question scores

Then map the two dimensions into an overall allowable investment-risk level.

A conservative mapping can be:

Mapped Risk Score = min(Risk Capacity Score, Risk Tolerance Score)

This prevents the robo-advisor from recommending a portfolio whose risk exceeds either the client's financial ability or psychological willingness to bear losses.

For example:

| Risk Score | Risk Profile | Illustrative Maximum Volatility |

| 0–20 | Very Conservative | 5% |

| 21–40 | Conservative | 8% |

| 41–60 | Moderate | 12% |

| 61–80 | Growth | 17% |

| 81–100 | Aggressive | 25%+ |



The exact cutoffs and volatility limits should be configurable rather than hard-coded.

The system should display the capacity score, tolerance score, and mapped risk score separately, so the client can understand why the portfolio receives a particular risk constraint.

3. Investment Universe Selection

The client can choose which ETFs the robo-advisor is allowed to consider.

Candidate ETF Universe

- VOO

- VTI

- BND

- TLT

- BIL

- GLD

- QQQ

- TQQQ

- VXUS

- VT

- VWO

- VNQ

- SCHH

- SCHD

- VYM

- DGRO

The application should initially present the complete universe and allow the client to select any subset.

The optimizer may allocate only among the ETFs selected by the client.

ETF Data Validation

Before optimization, the system should check:

- ETF inception date

- Availability of at least 20 years of observations

- Missing observations

- Trading history

- Adjusted prices

- Splits and distributions

- Expense ratios

- Dividend distributions

- Whether the ETF existed throughout the requested estimation period

The application should estimate parameters using the recent 20-year history data. For ETFs with less than 20 years of history, the application should estimate parameters using the maximum available history, or

4. Short-Sale Constraint

The client chooses:

Are short sales allowed?

If No

Portfolio weights must satisfy:

$${w}_{i}\ge 0$$

and

$$\sum_{i} {w}_{i}=1$$

If Yes

Negative portfolio weights are permitted:

$${w}_{i}<0$$

subject to configurable leverage constraints such as:

$$\sum_{i} |{w}_{i}|\le L$$

where  $L$ is the maximum gross exposure.

The application should also allow a maximum individual position size, such as:

$${w}_{i}\le 50\%$$

or a configurable value.

5. Tax Consideration

Ask:

Should taxes be incorporated into the portfolio analysis?

If No

The optimizer operates on pre-tax returns.

If Yes

The system should incorporate:

- Ordinary income tax on interest

- Qualified dividend treatment

- Long-term capital-gains tax

- Short-term capital-gains tax

- Capital losses

- Portfolio turnover

- Realized versus unrealized gains

- Tax impact of distributions

- Rebalancing-related taxes

The user should be asked for the relevant tax assumptions, including marginal income-tax rate and long-term/short-term capital-gains rates.

The system should distinguish between:

Pre-tax expected return

and

After-tax expected return

The optimizer should use the after-tax return series when taxes are enabled.

Because actual taxation depends on account type and individual circumstances, the tax module should be presented as an estimated tax model, not individualized tax advice.

6. Historical Data and Parameter Estimation

Use historical data to estimate the inputs required by the optimizer.

Estimation Window

Use the most recent 20 years of available historical data, subject to each ETF's actual inception date.

The preferred methodology is:

- Daily adjusted prices for risk estimation

- Monthly returns for long-horizon projections

- Total returns including distributions

- Risk-free rate from Treasury instruments

- Consistent observation dates across ETFs

The system should avoid look-ahead bias.

For a portfolio created at date  $t$ , parameter estimation must use only information available through  $t$ .

Parameters Estimated

For each ETF:

$${\mu }_{i}=expected return$$

$${\sigma }_{i}=volatility$$

and

$$\Sigma =covariance matrix$$

The system may also estimate:

- Downside deviation

- Maximum drawdown

- Value-at-Risk

- Expected Shortfall / CVaR

- Beta

- Sharpe ratio

- Sortino ratio

- Correlations

Expected returns should not simply be interpreted as guaranteed future returns. The interface should clearly label them as historical estimates / model assumptions.

7. Portfolio Optimizer

The optimizer uses the client's:

- investment goal

- target amount, if applicable

- target date, if applicable

- initial investment

- monthly contribution

- mapped risk score

- selected ETFs

- short-sale preference

- tax preference

- maximum individual position size preference

- historical parameter estimates

to determine portfolio weights.

Case A: Client Has a Target Amount and Target Date

The objective is:

Minimize portfolio risk subject to achieving the client's required investment target.

Let:

- ${W}_{0}$ = initial investment

- $C$ = monthly contribution

- $T$ = number of months until target date

- ${F}^{*}$ = target portfolio value

- $w$ = ETF portfolio weights

The optimization problem can be expressed as:

$$\underset{w}{min} Risk\left( w \right)$$

subject to:

$$E\left[ {F}_{T}\left( w \right) \right]\ge {F}^{*}$$

$$\sum_{i} {w}_{i}=1$$

plus:

$$Risk\left( w \right)\le Ris{k}_{max}$$

and the short-sale/position constraints.

A more robust formulation should consider the probability of achieving the target, rather than relying exclusively on expected terminal wealth:

$$\underset{w}{max} P\left( {F}_{T}\ge {F}^{*} \right)$$

subject to the client's maximum allowable risk.

Alternatively, the application can minimize downside risk:

$$\underset{w}{min}CVa{R}_{\alpha }\left( {F}_{T} \right)$$

subject to a target-achievement probability such as:

$$P\left( {F}_{T}\ge {F}^{*} \right)\ge p$$

where  $p$ might be 80%, 90%, or another configurable threshold.

This makes the target-date robo-advisor more realistic than simply solving a traditional mean-variance problem.

8. Case B: No Specific Target Amount

When the client does not specify a target amount, the optimizer should:

Maximize expected return subject to the client's mapped risk level.

For example:

$$\underset{w}{max} E\left[ {R}_{p} \right]$$

subject to:

$${\sigma }_{p}\le {\sigma }_{max}$$

$$\sum_{i} {w}_{i}=1$$

plus short-sale and concentration constraints.

The risk constraint can be determined from the client's mapped risk score.

For example:

$$RiskScore=70$$

might translate into:

$${\sigma }_{max}=17\%$$

The exact mapping should be configurable.

9. Portfolio Optimization Methods

The application should support more than one optimization method.

Primary Method

Mean-Variance Optimization

Use:

$$E\left[ {R}_{p} \right]={w}^{'}\mu$$

and

$${\sigma }_{p}^{2}={w}^{'}\Sigma w$$

Alternative Risk Models

The application can also support:

- Minimum volatility

- Maximum Sharpe ratio

- CVaR minimization

- Target-return minimum volatility

- Risk-parity

- Maximum diversification

For the client-facing robo-advisor, the selected methodology should be displayed clearly rather than hiding the optimization process.

10. Recommended Portfolio

The system should present the resulting portfolio in a transparent format.

Example:

| ETF | Weight | Initial Investment | Monthly Contribution |

| VOO | 35% | $35,000 | $700 |

| VXUS | 15% | $15,000 | $300 |

| BND | 25% | $25,000 | $500 |

| SGOV | 15% | $15,000 | $300 |

| GLD | 10% | $10,000 | $200 |

| Total | 100% | $100,000 | $2,000 |



The application should also explain why each ETF receives its allocation, based on its estimated return, volatility, covariance with other assets, and contribution to overall portfolio risk.

11. Financial Projection

After determining the recommended portfolio, the robo-advisor should project future wealth.

Inputs include:

- Initial investment

- Monthly contribution

- Investment horizon

- Expected portfolio return

- Portfolio volatility

- Inflation assumption

- Taxes, if enabled

Calculate:

$$F{V}_{T}={W}_{0}{\left( 1+r \right)}^{T}+C\left[ \frac{{\left( 1+r \right)}^{T}-1}{r} \right]$$

for the simplified deterministic projection.

However, the primary projection should be probabilistic rather than relying only on one expected-return path.

12. Monte Carlo Simulation

Run a Monte Carlo simulation of future portfolio values.

For example:

10,000 simulated paths

For each path:

- Draw monthly asset returns based on the estimated return distribution.

- Apply the portfolio weights.

- Add the monthly contribution.

- Rebalance according to the selected rebalancing assumption.

- Apply taxes when enabled.

- Continue until the target date.

Generate:

- Median terminal wealth

- 10th percentile

- 25th percentile

- 50th percentile

- 75th percentile

- 90th percentile

- Probability of exceeding the target

- Probability of losing principal

- Maximum simulated drawdown

- Distribution of terminal portfolio values

For a client with a target:

Probability of achieving target = percentage of simulations in which terminal wealth ≥ target amount.

This should be one of the most prominent outputs.

13. Scenario Analysis

In addition to the baseline Monte Carlo simulation, provide scenarios such as:

Conservative Scenario

Lower expected returns / higher adverse-return assumptions.

Base Scenario

Historical/model-implied assumptions.

Optimistic Scenario

Higher expected returns / favorable-return assumptions.

The application should clearly distinguish these scenarios from forecasts.

14. S&P 500 Benchmark Comparison

The recommended portfolio should be compared with an S&P 500 benchmark.

Use VOO or an appropriate S&P 500 total-return benchmark for the historical comparison, depending on the implementation.

Historical Comparison Period

Use the most recent 10 years for the benchmark comparison.

Report:

- Cumulative return

- Annualized return

- Annualized volatility

- Sharpe ratio

- Maximum drawdown

- Ending wealth from the same initial investment

- Ending wealth with the same monthly contributions

- Best year

- Worst year

The comparison must use the same initial investment, monthly contributions, time period, and return convention for both portfolios.

For example:

"If you invested $100,000 initially and contributed $2,000 per month, how would the recommended portfolio have performed versus the S&P 500 over the last 10 years?"

This provides an intuitive wealth comparison rather than only comparing percentages.

15. Portfolio Visualization

The dashboard should include:

Asset Allocation

Pie or bar chart showing ETF weights.

Historical Performance

Growth of $10,000 or the user's actual initial investment.

Future Projection

Monte Carlo confidence bands showing:

- 10th percentile

- 25th percentile

- Median

- 75th percentile

- 90th percentile

Target Progress

For users with a target:

$$Target=\$X$$

with the simulated terminal-value distribution displayed relative to the target.

Risk Profile

Display:

- Risk Capacity Score

- Risk Tolerance Score

- Mapped Risk Score

- Portfolio volatility

- Maximum drawdown

- Downside-risk measures

16. Recommendation Explanation

The robo-advisor should not simply output portfolio weights.

It should explain the recommendation in terms such as:

Your risk-capacity score is 72 and your risk-tolerance score is 64. Your mapped risk score is therefore 64. The recommended portfolio has an estimated annual volatility of 11.8%, which is within the risk constraint associated with your mapped profile.

For a target-based investor:

Given your initial investment of $100,000, monthly contribution of $2,000, and target of $500,000 by December 2036, the optimizer selected the portfolio that minimizes estimated risk while satisfying the target-achievement constraint.

For a non-target investor:

Because you did not specify a target amount, the optimizer maximizes estimated portfolio return subject to the risk limit implied by your mapped risk profile.

The system should provide the underlying calculations and assumptions so the client can understand how the recommendation was generated.

17. Rebalancing

The robo-advisor should establish a rebalancing rule.

Possible choices:

Calendar-based: rebalance monthly, quarterly, or annually.

Threshold-based: rebalance whenever an ETF deviates from its target allocation by more than a specified percentage.

When taxes are enabled, the system should incorporate the estimated tax consequences of selling appreciated positions.

18. Ongoing Monitoring

After the initial recommendation, the system should periodically reassess:

- Portfolio performance

- Risk

- Asset correlations

- Target progress

- Contribution changes

- Remaining time to target

- Changes in the client's questionnaire responses

- Changes in ETF historical characteristics

The system should trigger a review when:

- The client changes their goal

- The client changes contributions

- The target becomes difficult to achieve

- The portfolio exceeds the risk limit

- Significant market conditions change the portfolio's risk characteristics

19. Complete User Workflow

The complete application flow should therefore be:

Step 1 — Client Profile

→ Investment experience
→ Financial information
→ Investment horizon

Step 2 — Investment Goal

→ Target amount/date OR no specific target
→ Initial investment
→ Monthly contribution

Step 3 — Risk Questionnaire

→ Risk capacity questions
→ Risk tolerance questions
→ Capacity score
→ Tolerance score
→ Mapped risk score

Step 4 — Investment Universe

→ Client selects ETFs from the 20-ETF list

Step 5 — Investment Constraints

→ Short selling allowed?
→ Taxes considered?
→ Maximum position size
→ Optional leverage constraint

Step 6 — Historical Data

→ Retrieve historical adjusted total-return data
→ Use up to 20 years of available data
→ Estimate returns, volatility, covariance, downside risk, etc.

Step 7 — Portfolio Optimization

→ Target specified: minimize risk subject to target requirement
→ No target: maximize return subject to mapped risk constraint

Step 8 — Recommended Portfolio

→ ETF weights
→ Dollar allocations
→ Monthly contribution allocation
→ Expected return
→ Expected volatility
→ Downside risk

Step 9 — Financial Simulation

→ Monte Carlo simulation
→ Projected wealth distribution
→ Target-achievement probability
→ Drawdown analysis

Step 10 — Benchmark Comparison

→ Compare with S&P 500 over the most recent 10 years
→ Same initial investment and monthly contributions
→ Compare return, volatility, drawdown, Sharpe ratio, and ending wealth

Step 11 — Client Report

→ Recommended allocation
→ Risk profile
→ Financial projection
→ Benchmark comparison
→ Explanation of optimization methodology
→ Assumptions and limitations

Step 12 — Ongoing Monitoring

→ Track progress
→ Recalculate when client circumstances or portfolio characteristics change

20. Key Design Principle

The robo-advisor should separate three concepts that are often incorrectly combined:

$$Risk Capacity$$

$$Risk Tolerance$$

$$Investment Objective$$

The risk questionnaire determines how much risk the client can reasonably take, while the investment objective determines what the portfolio needs to accomplish.

Thus:

Target-based client

$$Minimize Risk subject to achieving the target$$

No-target client

$$Maximize Expected Return subject to allowable risk$$

This gives the robo-advisor a coherent decision architecture rather than simply assigning clients to generic "conservative," "moderate," or "aggressive" portfolios.
