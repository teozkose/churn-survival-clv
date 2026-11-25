# Customer Churn, Survival and Lifetime Value

[![CI](https://github.com/teozkose/churn-survival-clv/actions/workflows/ci.yml/badge.svg)](https://github.com/teozkose/churn-survival-clv/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## TL;DR

- **Question:** with a fixed retention budget, which telecom customers should get an offer: the ones most likely to leave, or the ones whose departure would cost the most?
- **Headline:** a log-normal survival model ranks {{MODEL_ROWS}} customers by time to churn with C-index **{{CINDEX}}** (95% CI {{CINDEX_CI}}). Turning each curve into a 5-year lifetime value and choosing offers by expected value gained earns **{{VP_GAIN}}** net from {{VP_N}} offers, against {{RP_GAIN}} from {{RP_N}} offers when the {{BUDGET}} budget goes to the highest churn risks first (difference {{DIFF}}, 95% CI {{DIFF_CI}} over {{REFITS}} model refits).
- **Decision:** make the offer to the {{VP_N}} month-to-month customers the value plan picks, keep the remaining {{UNSPENT}}, and test the save rate before scaling up: below a {{BREAK_EVEN}} save rate no offer pays for itself.

## Problem and decision

A subscription business loses customers every month and has a retention budget: here, one month free in exchange for a {{COMMIT}}-month commitment. Spending it on whoever looks most likely to leave is the usual rule, but a customer about to leave who pays little is worth less than a steady high-value one with a moderate risk. The retention team needs three things: when customers leave (not only whether), what each active customer is worth, and an offer list that maximises expected value within the budget. The deciding metrics are the concordance index (C-index) for ranking time to churn, calibration of 12-month churn for the probabilities, and expected net gain in dollars for the plan.

## Data

- **Source:** [IBM Telco customer churn](https://github.com/IBM/telco-customer-churn-on-icp4d) sample (Apache 2.0), one CSV downloaded by `make data` and never committed.
- **Size and range:** {{ROWS}} customers of a fictional telecom provider in one snapshot, tenure 0 to 72 months, churn rate {{CHURN_RATE}} (churn = left in the snapshot month).
- **Fields:** tenure, contract (month-to-month, one year, two year), phone and internet services and six internet add-ons, paperless billing, payment method, monthly and total charges, gender, senior citizen, partner, dependents.
- **Validation** (`src/churn/validate.py`): required columns, allowed codes for every categorical field, no nulls outside TotalCharges, unique ids, tenure in 0-120, monthly charges in $1-500, add-ons only for customers with internet, multiple lines only with phone service. Any failure stops the run with the full list. Soft findings are counted: {{BLANK_TOTAL}} blank TotalCharges (all at tenure 0), {{GAP_FLAGGED}} customers whose TotalCharges is more than 35% away from tenure × monthly price (median gap {{GAP_MEDIAN}}), {{DUPES}} identical profiles under different ids.
- **Cleaning** (counts from the last run):

{{CLEANING_TABLE}}

Monthly price is almost a linear function of the services held (R² {{PRICE_R2}}), so price is left out of the hazard models (it would only duplicate the service flags) and used for value instead.

## Approach

```mermaid
flowchart LR
    A[download CSV] --> B[validate codes, ranges, nulls, duplicates]
    B --> C[drop tenure 0, encode 0/1 covariates]
    C --> D{stratified 75/25 hold-out}
    D --> E[Kaplan-Meier + log-rank]
    D --> F[Cox PH, penalty tuned by 5-fold CV]
    F --> G[Schoenfeld test -> Cox stratified by contract]
    D --> H[AFT: Weibull / log-normal / log-logistic by AIC]
    D --> I[XGBoost 12-month classifier]
    E & F & G & H & I --> J[C-index, 12-month AUC, Brier, calibration with bootstrap CIs]
    H --> K[conditional survival -> CLV per active customer]
    K --> L[knapsack on expected gain vs highest-risk-first]
    L --> M[budget and save-rate sensitivity, 100 refits]
    J & M --> N[README + report page]
```

- **Survival setup:** duration = tenure in months, event = churned. Active customers are right-censored at their current tenure.
- **Validation:** a stratified 75/25 hold-out ({{TRAIN}} / {{TEST}} customers); the Cox penalty ({{PENALIZER}}) and the XGBoost depth ({{XGB_DEPTH}}) and learning rate ({{XGB_LR}}) are tuned by 5-fold CV inside the training part. C-index uses every hold-out customer; 12-month metrics use the {{TEST_KNOWN}} whose 12-month outcome is known.
- **Models:** a baseline that knows only the contract type (one Kaplan-Meier curve per contract); Cox proportional hazards; Cox stratified by contract (one baseline hazard per contract) where the Schoenfeld test rejects proportional hazards; an accelerated failure time (AFT) model, family chosen by AIC, which extrapolates beyond 72 months; and an XGBoost classifier for churn within 12 months, the usual alternative. Gender is never a model input; it is kept for the segment check.
- **Uncertainty:** 500 paired bootstrap resamples of hold-out customers for every metric and model difference; Cox hazard-ratio CIs from the fitted model; {{REFITS}} bootstrap refits of the lifetime model for the plan values.
- **Lifetime value:** for an active customer at tenure t, P(still paying k months from now) = S(t+k)/S(t). CLV = Σ {{MARGIN}} × monthly bill × S(t+k)/S(t) × discount^k over {{CLV_HORIZON}} months at {{DISCOUNT}} a year.
- **Targeting:** gain of an offer = save rate × (CLV if kept {{COMMIT}} months, then following its own curve − CLV now) − one monthly bill. The value plan solves a 0/1 knapsack with PuLP (maximise total gain, total cost within the budget); the comparison plan offers to the highest 12-month churn probabilities until the budget runs out.
- **Sensitivity check:** the save rate (10% to 50%) and the budget ({{BUDGET_LIST}}) are varied; the {{SAVE_RATE}} plan is also re-valued at each other save rate as if it had been kept unchanged.

## Results

### Who leaves, and when

{{KM_TABLE}}

Retention differs sharply by contract (log-rank χ² {{LOGRANK}}, 2 df, p < 1e-16) and by payment method (χ² {{LOGRANK_PAY}}). Only {{KM_M2M_72}} of month-to-month customers reach 72 months, against {{KM_2Y_72}} on two-year contracts.

![Kaplan-Meier retention by contract and payment method](results/km_curves.png)

### Model comparison (hold-out)

{{MODEL_TABLE}}

Every covariate model beats contract alone by a wide margin (AFT − baseline C-index {{AFT_VS_KM}}, CI {{AFT_VS_KM_CI}}), and the three survival models rank customers almost identically. The 12-month classifier is slightly sharper on its own question (ROC-AUC +{{XGB_GAP}}, CI {{XGB_GAP_CI}}) and better calibrated ({{XGB_ECE}} vs {{AFT_ECE}}), but it was trained on {{XGB_ROWS}} customers after dropping {{XGB_DROPPED}} whose outcome is not known yet, and it says nothing about month 13 onward, where most lifetime value sits. That is why the survival model, not the classifier, feeds the value calculation.

![Predicted against observed 12-month churn](results/calibration.png)

### Proportional hazards

The Schoenfeld test (rank time) rejects proportional hazards at p < 0.01 for {{PH_FAIL_N}} covariates of the plain Cox model: {{PH_FAIL}}. Stratifying by contract absorbs the contract terms; {{PH_FAIL_S_N}} add-on terms still drift ({{PH_FAIL_S}}), meaning their protective effect fades with tenure. Stratification changes the C-index by {{STRAT_DIFF}} (CI {{STRAT_DIFF_CI}}), so the ranking is robust to the assumption, and the value model is an AFT, which does not assume it.

{{PH_TABLE}}

### Lifetime value

The AFT family with the lowest AIC on the training customers is {{AFT_FAMILY}}:

{{AIC_TABLE}}

{{ACTIVE}} active customers carry an expected {{CLV_TOTAL}} of margin over five years (mean {{CLV_MEAN}}, median {{CLV_MEDIAN}}, {{MONTHS_MEAN}} expected months of the next 60):

{{CONTRACT_TABLE}}

### Who gets the offer

At a {{SAVE_RATE}} save rate only {{POSITIVE}} active customers have a positive expected gain, all on month-to-month contracts. The value plan takes all of them; the risk plan spends the whole budget and includes customers whose offer costs more than it saves.

{{PLAN_TABLE}}

The plans overlap on {{OVERLAP}} customers: every customer the value plan picks is also in the risk plan. With this offer the expected gain rises with churn risk almost everywhere, because high-value customers are high-value precisely because they are unlikely to leave. So the answer to the opening question is "risk first, but know where to stop": the knapsack's contribution is the stopping point, not a different list. The value plan wins in {{DIFF_SHARE}} of {{REFITS}} refits of the lifetime model, returns {{VP_ROI}} on spend against {{RP_ROI}}, and keeps {{UNSPENT}} of the budget.

![Active customers by churn risk and lifetime value, coloured by plan](results/targeting_map.png)

### Sensitivity to the save rate and the budget

{{SENSITIVITY_TABLE}}

{{BUDGET_TABLE}}

The save rate drives everything. At 10% no offer pays (break-even is {{BREAK_EVEN}} for the best customer), and spending the budget on the riskiest customers loses {{RP_GAIN_10}}. At higher save rates both plans grow and converge, because more offers pay off. The budget curve shows the other failure of the risk-first rule: at {{BIG_BUDGET}} it falls to {{BIG_RP_GAIN}}, while the value plan stays at {{BIG_VP_GAIN}} because it stops when offers stop paying.

![Net gain across budgets and save rates](results/plan_curves.png)

## What drives it / error analysis

Hazard ratios from the Cox model stratified by contract, fitted on every customer (top 10 by distance from 1 among p < 0.05):

{{HR_TABLE}}

Fiber internet carries {{HR_FIBER}}× the churn hazard of no internet (DSL {{HR_DSL}}×), electronic-check payment {{HR_ECHECK}}×; online security ({{HR_SECURITY}}×), online backup ({{HR_BACKUP}}×), tech support ({{HR_TECH}}×) and having a partner ({{HR_PARTNER}}×) go with staying. Add-ons look like switching costs: the more of the bundle a customer uses, the longer they stay.

![Hazard ratios](results/hazard_ratios.png)

Errors by segment on the hold-out customers (lifetime model):

{{SEGMENT_TABLE}}

Within one contract type the ranking is much weaker than overall: contract carries most of the signal, and among one- and two-year customers there are too few churners for a precise C-index. Predicted 12-month churn tracks the observed rate in most segments; the model over-predicts for seniors and under-predicts for customers without internet and for mailed-check payers.

![C-index and 12-month churn by segment](results/segments.png)

## Limitations and next steps

- **One snapshot.** Tenure is observed once, and customers who left before the snapshot month are not in the file. Long-tenure retention is therefore likely overstated, and every curve beyond 72 months is extrapolation by the parametric model. A panel of monthly customer records would allow proper left truncation.
- **Current attributes.** Contract, services and payment method are today's values; a customer who upgraded may look stickier than they were.
- **Assumed economics.** Margin ({{MARGIN}}), discount rate, offer cost and above all the save rate are assumptions in `config.yaml`. The save rate is not measured here; a randomised test of the offer on a few hundred customers would replace it and would also show whether the offer works better for some segments (uplift modelling).
- **Teaching sample.** The data is a public sample built for demonstrations, not a real operator's book.
- **Next:** time-varying covariates once monthly records exist; a cure-rate model for customers who will effectively never leave; uplift modelling once offer data exist.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make all     # data -> features -> run -> report -> notebook, about {{RUNTIME}} s for the run stage
make test    # 14 tests on synthetic customers, no download
```

`run_all.sh` runs the same steps. Every number above is read from `results/metrics.json` by `src/churn/report.py`; the report page is `docs/index.html`.

## Project structure

```
├── config.yaml              paths, model grid, CLV and offer assumptions
├── Makefile / run_all.sh    data, features, run, report, notebook
├── data/download.py         fetches the CSV (data files are not committed)
├── src/churn/
│   ├── data.py              download and tidy
│   ├── validate.py          schema, codes, ranges, nulls, duplicates
│   ├── features.py          survival table, 0/1 covariates, 12-month label
│   ├── survival.py          Kaplan-Meier, Cox, stratified Cox, AFT, log-rank, PH test
│   ├── models.py            XGBoost 12-month classifier with inner-CV tuning
│   ├── evaluate.py          C-index, AUC, Brier, calibration, bootstrap CIs, segments
│   ├── clv.py               conditional survival, expected lifetime, CLV
│   ├── targeting.py         expected gain, knapsack (PuLP), risk-first plan
│   ├── figures.py           PNG figures
│   ├── pipeline.py          stages
│   └── report.py            README.md and docs/index.html from metrics.json
├── tests/                   pytest on synthetic customers
├── notebooks/01_analysis.ipynb
├── results/                 metrics.json and figures
└── docs/index.html          report page
```
