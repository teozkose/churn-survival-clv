"""Render README.md (from README.template.md) and docs/index.html from results/metrics.json,
so every number in the write-up comes from the last run."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from churn.config import ROOT, resolve

MODEL_LABELS = {"km_contract": "Baseline: Kaplan-Meier by contract only", "cox": "Cox proportional hazards",
                "cox_stratified": "Cox, stratified by contract", "aft": "Log-normal AFT (lifetime model)",
                "xgboost_12m": "XGBoost, 12-month churn classifier"}
SEGMENT_LABELS = {"contract": "Contract", "internet_service": "Internet", "payment_method": "Payment",
                  "senior_label": "Age", "gender": "Gender"}
FIGURES = [
    ("km_curves", "Kaplan-Meier retention by contract and payment method with 95% bands"),
    ("hazard_ratios", "Hazard ratios with 95% CIs from the Cox model stratified by contract"),
    ("calibration", "Predicted against observed 12-month churn by decile on hold-out customers"),
    ("targeting_map", "Active customers by churn risk and lifetime value, coloured by plan"),
    ("plan_curves", "Net gain of both plans across budgets and save-rate assumptions"),
    ("segments", "Hold-out C-index and 12-month churn by segment"),
]


def pct(x: float, digits: int = 1) -> str:
    """0.1234 -> '12.3%'."""
    return f"{100 * x:.{digits}f}%"


def ci(pair: list[float], digits: int = 3) -> str:
    """[0.1, 0.2] -> '0.100–0.200'."""
    return f"{pair[0]:.{digits}f}–{pair[1]:.{digits}f}"


def usd(x: float) -> str:
    """Dollars, in thousands when large."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1000:,.1f}k" if x >= 1_000 else f"{sign}${x:,.0f}"


def model_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[label, f"{m['models'][k]['cindex']:.3f} ({ci(m['models'][k]['cindex_ci'])})",
             f"{m['models'][k]['auc']:.3f} ({ci(m['models'][k]['auc_ci'])})", f"{m['models'][k]['brier']:.4f}",
             f"{m['models'][k]['ece']:.4f}"] for k, label in MODEL_LABELS.items()]


def km_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for group, label in [("contract", "Contract"), ("payment_method", "Payment")]:
        for level, r in m["km"][group]["levels"].items():
            rows.append([label, level, f"{r['n']:,}", f"{r['events']:,}"] + [pct(r["at"][t]) for t in
                                                                             ["12", "24", "48", "72"]])
    return rows


def ph_rows(m: dict[str, Any]) -> list[list[str]]:
    strat = {r["feature"]: r["p"] for r in m["fit"]["ph_test_stratified"]}
    rows = [[r["feature"], f"{r['statistic']:.1f}", f"{r['p']:.4f}",
             f"{strat[r['feature']]:.4f}" if r["feature"] in strat else "absorbed by strata"]
            for r in sorted(m["fit"]["ph_test"], key=lambda r: r["p"])]
    return rows[:8]


def hr_rows(m: dict[str, Any]) -> list[list[str]]:
    hr = sorted(m["hazard_ratios"], key=lambda r: -abs(r["hr"] - 1) if r["p"] < 0.05 else 0)
    return [[r["feature"], f"{r['hr']:.2f}", f"{r['lo']:.2f}–{r['hi']:.2f}", f"{r['p']:.3g}"] for r in hr[:10]]


def contract_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[r["contract"], f"{r['customers']:,}", pct(r["mean_p12"]), f"{r['mean_months']:.1f}",
             usd(r["mean_clv"]), f"{r['value_plan']:,}", f"{r['risk_plan']:,}"] for r in m["clv"]["by_contract"]]


def plan_rows(m: dict[str, Any]) -> list[list[str]]:
    t = m["targeting"]
    u = t["uncertainty"]
    out = []
    for key, label, band in [("value_plan", "Value plan (knapsack on expected gain)", u["value_plan_ci"]),
                             ("risk_plan", "Highest churn probability first", u["risk_plan_ci"])]:
        p = t[key]
        out.append([label, f"{p['customers']:,}", usd(p["spend"]), f"{usd(p['net_gain'])} ({usd(band[0])} to "
                    f"{usd(band[1])})", f"{p['expected_saves']:.0f}", pct(p["mean_p_churn"]), usd(p["mean_clv"])])
    return out


def sensitivity_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[pct(r["save_rate"], 0), f"{r['value_plan_customers']:,}", usd(r["value_plan_gain"]),
             usd(r["base_plan_gain"]), usd(r["risk_plan_gain"])] for r in m["targeting"]["sensitivity"]]


def budget_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[usd(r["budget"]), f"{r['value_plan_customers']:,}", usd(r["value_plan_spend"]),
             usd(r["value_plan_gain"]), usd(r["risk_plan_gain"])] for r in m["targeting"]["budget_curve"]]


def segment_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[SEGMENT_LABELS[r["segment"]], r["level"], f"{r['n']:,}", f"{r['events']:,}",
             f"{r['cindex']:.3f} ({ci(r['cindex_ci'])})", pct(r["observed"]), pct(r["predicted"])]
            for r in m["segments"]]


def cleaning_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[s["step"], f"{s['rows_before']:,}", f"{s['changed']:,}", f"{s['dropped']:,}", f"{s['rows_after']:,}"]
            for s in m["data"]["cleaning"]]


def aic_rows(m: dict[str, Any]) -> list[list[str]]:
    aic = m["fit"]["aft_aic"]
    best = min(aic.values())
    return [[f, f"{a:,.0f}", f"{a - best:+,.0f}"] for f, a in sorted(aic.items(), key=lambda kv: kv[1])]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Rows of strings -> a GitHub markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def html_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


TABLES = {
    "MODEL_TABLE": (["Model", "C-index (95% CI)", "12-month ROC-AUC (95% CI)", "12-month Brier", "Calib. error"],
                    model_rows),
    "KM_TABLE": (["Group", "Level", "Customers", "Churned", "Still a customer at 12 m", "24 m", "48 m", "72 m"],
                 km_rows),
    "PH_TABLE": (["Covariate", "Schoenfeld χ²", "p (Cox)", "p (stratified Cox)"], ph_rows),
    "HR_TABLE": (["Covariate", "Hazard ratio", "95% CI", "p"], hr_rows),
    "AIC_TABLE": (["AFT family", "AIC", "Δ AIC"], aic_rows),
    "CONTRACT_TABLE": (["Contract", "Active customers", "P(churn, next 12 m)", "Expected months (of 60)",
                        "Mean CLV", "In value plan", "In risk plan"], contract_rows),
    "PLAN_TABLE": (["Plan", "Customers", "Spend", "Net gain (95% CI)", "Expected saves", "Mean churn risk",
                    "Mean CLV"], plan_rows),
    "SENSITIVITY_TABLE": (["Save rate", "Value plan size", "Value plan gain", "Gain of the 25% plan kept as is",
                           "Risk plan gain"], sensitivity_rows),
    "BUDGET_TABLE": (["Budget", "Value plan size", "Value plan spend", "Value plan gain", "Risk plan gain"],
                     budget_rows),
    "SEGMENT_TABLE": (["Segment", "Level", "Customers", "Churned", "C-index (95% CI)", "Observed 12-m churn",
                       "Predicted"], segment_rows),
    "CLEANING_TABLE": (["Step", "Rows in", "Rows flagged", "Rows dropped", "Rows out"], cleaning_rows),
}


def values(m: dict[str, Any]) -> dict[str, str]:
    """Scalar placeholders used by README.template.md and the HTML page."""
    h, mod, cmp, fit, c, t = m["headline"], m["models"], m["comparisons"], m["fit"], m["clv"], m["targeting"]
    d = m["data"]
    km_c = m["km"]["contract"]["levels"]
    hr = {r["feature"]: r for r in m["hazard_ratios"]}
    sens = {round(r["save_rate"], 2): r for r in t["sensitivity"]}
    big = t["budget_curve"][-1]
    vp, rp = t["value_plan"], t["risk_plan"]
    return {
        "ROWS": f"{d['rows']:,}", "MODEL_ROWS": f"{d['modelling_rows']:,}", "CHURN_RATE": pct(d["churn_rate"]),
        "ZERO_TENURE": str(d["zero_tenure"]), "BLANK_TOTAL": str(d["blank_total_charges"]),
        "GAP_FLAGGED": str(d["charges_gap_flagged"]), "GAP_MEDIAN": pct(d["charges_gap_median"]),
        "DUPES": str(d["duplicate_rows_ignoring_id"]), "PRICE_R2": f"{d['price_r2_from_services']:.3f}",
        "MAX_TENURE_SHARE": pct(d["max_tenure_share"]),
        "TRAIN": f"{m['split']['train']:,}", "TEST": f"{m['split']['test']:,}",
        "TEST_KNOWN": f"{m['split']['test_known']:,}", "TEST_EVENTS": f"{m['split']['test_events']:,}",
        "CINDEX": f"{h['cindex']:.3f}", "CINDEX_CI": ci(h["cindex_ci"]),
        "KM_CINDEX": f"{mod['km_contract']['cindex']:.3f}", "COX_CINDEX": f"{mod['cox']['cindex']:.3f}",
        "COXS_CINDEX": f"{mod['cox_stratified']['cindex']:.3f}", "XGB_CINDEX": f"{mod['xgboost_12m']['cindex']:.3f}",
        "AFT_VS_KM": f"{cmp['aft_vs_km_cindex']['mean']:.3f}", "AFT_VS_KM_CI": ci(cmp["aft_vs_km_cindex"]["ci"]),
        "STRAT_DIFF": f"{cmp['cox_stratified_vs_cox_cindex']['mean']:+.4f}",
        "STRAT_DIFF_CI": ci(cmp["cox_stratified_vs_cox_cindex"]["ci"], 4),
        "AFT_AUC": f"{mod['aft']['auc']:.3f}", "XGB_AUC": f"{mod['xgboost_12m']['auc']:.3f}",
        "XGB_GAP": f"{cmp['xgboost_vs_aft_auc']['mean']:.3f}", "XGB_GAP_CI": ci(cmp["xgboost_vs_aft_auc"]["ci"]),
        "AFT_ECE": f"{mod['aft']['ece']:.3f}", "XGB_ECE": f"{mod['xgboost_12m']['ece']:.3f}",
        "COXS_ECE": f"{mod['cox_stratified']['ece']:.3f}",
        "XGB_ROWS": f"{fit['classifier_rows']:,}", "XGB_DROPPED": f"{fit['classifier_dropped']:,}",
        "XGB_DEPTH": str(fit["xgboost_params"]["max_depth"]), "XGB_LR": str(fit["xgboost_params"]["learning_rate"]),
        "PENALIZER": str(fit["cox_penalizer"]), "AFT_FAMILY": fit["aft_family"].replace("lognormal", "log-normal"),
        "PH_FAIL_N": str(len(fit["ph_failing"])), "PH_FAIL": ", ".join(fit["ph_failing"]),
        "PH_FAIL_S_N": str(len(fit["ph_failing_stratified"])),
        "PH_FAIL_S": ", ".join(fit["ph_failing_stratified"]) or "none",
        "KM_M2M_12": pct(km_c["Month-to-month"]["at"]["12"]), "KM_M2M_72": pct(km_c["Month-to-month"]["at"]["72"]),
        "KM_2Y_72": pct(km_c["Two year"]["at"]["72"]), "KM_1Y_72": pct(km_c["One year"]["at"]["72"]),
        "LOGRANK": f"{m['km']['contract']['logrank_statistic']:,.0f}",
        "LOGRANK_PAY": f"{m['km']['payment_method']['logrank_statistic']:,.0f}",
        "HR_FIBER": f"{hr['internet_fiber']['hr']:.2f}", "HR_DSL": f"{hr['internet_dsl']['hr']:.2f}",
        "HR_ECHECK": f"{hr['pay_electronic_check']['hr']:.2f}", "HR_PARTNER": f"{hr['partner']['hr']:.2f}",
        "HR_SECURITY": f"{hr['online_security']['hr']:.2f}", "HR_BACKUP": f"{hr['online_backup']['hr']:.2f}",
        "HR_TECH": f"{hr['tech_support']['hr']:.2f}",
        "ACTIVE": f"{c['active']:,}", "CLV_MEAN": usd(c["mean"]), "CLV_MEDIAN": usd(c["median"]),
        "CLV_TOTAL": f"${c['total'] / 1e6:,.2f}M", "MONTHS_MEAN": f"{c['mean_months']:.1f}",
        "MARGIN": pct(c["gross_margin"], 0), "DISCOUNT": pct(c["annual_discount"], 0),
        "CLV_HORIZON": str(c["horizon_months"]), "POSITIVE": f"{c['positive_gain_customers']:,}",
        "BREAK_EVEN": pct(c["break_even_save_rate"]),
        "BUDGET": usd(t["budget"]), "SAVE_RATE": pct(t["save_rate"], 0),
        "OFFER_MONTHS": str(t["offer_months_free"]), "COMMIT": str(t["commitment_months"]),
        "VP_N": f"{vp['customers']:,}", "VP_SPEND": usd(vp["spend"]), "VP_GAIN": usd(vp["net_gain"]),
        "VP_SAVES": f"{vp['expected_saves']:.0f}", "VP_ROI": pct(vp["net_gain"] / vp["spend"], 0),
        "RP_N": f"{rp['customers']:,}", "RP_SPEND": usd(rp["spend"]), "RP_GAIN": usd(rp["net_gain"]),
        "RP_SAVES": f"{rp['expected_saves']:.0f}", "RP_ROI": pct(rp["net_gain"] / rp["spend"], 0),
        "OVERLAP": f"{t['overlap']:,}", "UNSPENT": usd(t["budget"] - vp["spend"]),
        "DIFF": usd(h["difference"]), "DIFF_CI": f"{usd(h['difference_ci'][0])} to {usd(h['difference_ci'][1])}",
        "DIFF_SHARE": pct(t["uncertainty"]["share_value_better"], 0), "REFITS": str(t["uncertainty"]["refits"]),
        "RP_GAIN_10": usd(sens[0.1]["risk_plan_gain"]), "VP_GAIN_10": usd(sens[0.1]["value_plan_gain"]),
        "BASE_GAIN_50": usd(sens[0.5]["base_plan_gain"]), "VP_GAIN_50": usd(sens[0.5]["value_plan_gain"]),
        "BUDGET_LIST": ", ".join(usd(r["budget"]) for r in t["budget_curve"]),
        "BIG_BUDGET": usd(big["budget"]), "BIG_RP_GAIN": usd(big["risk_plan_gain"]),
        "BIG_VP_GAIN": usd(big["value_plan_gain"]),
        "RUNTIME": f"{m['runtime_s']:.0f}",
    }


def fill(template: str, m: dict[str, Any], table_fn: Any) -> str:
    """Replace {{KEY}} placeholders; tables are rendered with `table_fn(header, rows)`."""
    vals = values(m)
    for key, (header, rows_fn) in TABLES.items():
        vals[key] = table_fn(header, rows_fn(m))
    return re.sub(r"\{\{(\w+)\}\}", lambda mt: vals[mt.group(1)], template)


def _img(path: Path, docs: Path, alt: str) -> str:
    """Copy a figure next to the page (GitHub Pages serves docs/ only) and link it."""
    target = docs / "figures" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return f'<figure><img alt="{html.escape(alt)}" src="figures/{path.name}"></figure>'


def render_html(m: dict[str, Any], results: Path, docs: Path) -> str:
    """One page with its figures beside it: TL;DR, figures, tables, method, limitations."""
    v = values(m)
    t = {k: html_table(h, fn(m)) for k, (h, fn) in TABLES.items()}
    f = {name: _img(results / f"{name}.png", docs, alt) for name, alt in FIGURES}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn, Survival and Lifetime Value</title>
<style>
:root {{ --fg:#1a202c; --muted:#4a5568; --bg:#ffffff; --line:#e2e8f0; --accent:#2b6cb0; --card:#f7fafc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e2e8f0; --muted:#a0aec0; --bg:#171923;
  --line:#2d3748; --accent:#63b3ed; --card:#1a202c; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,sans-serif;
  max-width:960px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.sub {{ color:var(--muted); margin-top:0; }}
.tldr {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:4px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px; }}
.kpi b {{ display:block; font-size:24px; color:var(--accent); }}
.kpi span {{ color:var(--muted); font-size:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:12px 0; display:block; overflow-x:auto; }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }}
figure {{ margin:16px 0; }} img {{ max-width:100%; height:auto; background:#fff; border-radius:4px; }}
</style></head><body>
<h1>Customer Churn, Survival and Lifetime Value</h1>
<p class="sub">When do telecom customers leave, what is each one worth, and who should get a retention offer
when the budget is fixed?</p>
<div class="tldr"><b>TL;DR.</b> A log-normal survival model ranks {v['MODEL_ROWS']} customers by time to churn
with C-index <b>{v['CINDEX']}</b> (95% CI {v['CINDEX_CI']}), {v['AFT_VS_KM']} above contract type alone, and turns
each active customer's curve into a 5-year lifetime value. With a {v['BUDGET']} budget for one-month-free offers
and a {v['SAVE_RATE']} save rate, choosing customers by expected value gained (a knapsack) earns
<b>{v['VP_GAIN']}</b> net from {v['VP_N']} offers ({v['VP_SPEND']} spent), against {v['RP_GAIN']} from
{v['RP_N']} offers when the budget goes to the highest churn risks first. The value plan wins in
{v['DIFF_SHARE']} of {v['REFITS']} model refits (difference {v['DIFF']}, 95% CI {v['DIFF_CI']}) and leaves
{v['UNSPENT']} unspent, because no further offer pays for itself.</div>
<div class="kpis">
<div class="kpi"><b>{v['CINDEX']}</b><span>C-index, lifetime model (hold-out)</span></div>
<div class="kpi"><b>{v['VP_GAIN']}</b><span>net gain, value plan ({v['VP_ROI']} on spend)</span></div>
<div class="kpi"><b>{v['CLV_MEAN']}</b><span>mean 5-year CLV per active customer</span></div>
<div class="kpi"><b>{v['BREAK_EVEN']}</b><span>save rate below which no offer pays</span></div>
</div>
<h2>Who leaves, and when</h2>
<p>Kaplan-Meier retention differs sharply by contract (log-rank χ² {v['LOGRANK']}, 2 df) and payment method
(χ² {v['LOGRANK_PAY']}, 3 df). {v['KM_M2M_12']} of month-to-month customers are still there after a year and
{v['KM_M2M_72']} after six; two-year contracts keep {v['KM_2Y_72']}.</p>
{f['km_curves']}
{t['KM_TABLE']}
<h2>Model comparison</h2>
<p>{v['TRAIN']} customers train the models; {v['TEST']} held out ({v['TEST_EVENTS']} churned) score them.
C-index uses every hold-out customer; 12-month metrics use the {v['TEST_KNOWN']} whose 12-month outcome is known.
95% CIs from 500 paired bootstrap resamples.</p>
{t['MODEL_TABLE']}
<p>The 12-month classifier is slightly sharper on its own question (ROC-AUC +{v['XGB_GAP']}, CI {v['XGB_GAP_CI']})
but had to drop {v['XGB_DROPPED']} recent customers whose outcome is not known yet, and it cannot say anything
about month 13 onward, which is where most lifetime value sits.</p>
{f['calibration']}
<h2>Proportional hazards check</h2>
<p>The Schoenfeld test rejects proportional hazards (p &lt; 0.01) for {v['PH_FAIL_N']} covariates in the plain
Cox model: {v['PH_FAIL']}. Stratifying by contract gives each contract its own baseline hazard and clears the
contract terms; {v['PH_FAIL_S_N']} add-on service terms still drift ({v['PH_FAIL_S']}). The ranking barely moves
(C-index change {v['STRAT_DIFF']}, CI {v['STRAT_DIFF_CI']}), and lifetime value uses the AFT model, which does
not assume proportional hazards.</p>
{t['PH_TABLE']}
<h2>What drives churn</h2>
<p>Hazard ratios from the Cox model stratified by contract, fitted on every customer. Fiber internet carries
{v['HR_FIBER']}× the churn hazard of no internet; electronic-check payers {v['HR_ECHECK']}×; online security
({v['HR_SECURITY']}×), online backup ({v['HR_BACKUP']}×) and tech support ({v['HR_TECH']}×) go with staying.</p>
{f['hazard_ratios']}
{t['HR_TABLE']}
<h2>Lifetime value</h2>
<p>For an active customer at tenure t, the chance of still paying k months from now is S(t+k)/S(t) from the
{v['AFT_FAMILY']} AFT model (lowest AIC of three families). CLV is {v['MARGIN']} margin on the monthly bill,
discounted at {v['DISCOUNT']} a year, over at most {v['CLV_HORIZON']} months.</p>
{t['AIC_TABLE']}
{t['CONTRACT_TABLE']}
<h2>Who gets the offer</h2>
<p>The offer: {v['OFFER_MONTHS']} month free for a {v['COMMIT']}-month commitment, costing one monthly bill.
Expected gain = save rate × (CLV if kept {v['COMMIT']} months − CLV now) − cost. Only {v['POSITIVE']} active
customers have a positive expected gain at a {v['SAVE_RATE']} save rate. All {v['OVERLAP']} of them are also in
the risk-first plan: high-value customers are high-value because they rarely leave, so gain rises with risk.
The knapsack's contribution is knowing where to stop.</p>
{t['PLAN_TABLE']}
{f['targeting_map']}
<h2>Budget and save-rate sensitivity</h2>
<p>The save rate is the assumption that matters most. At 10% the value plan makes no offers ({v['VP_GAIN_10']})
while spending the budget on the riskiest customers loses {v['RP_GAIN_10']}. At 50% both plans grow and the
{v['SAVE_RATE']} plan kept as is earns {v['BASE_GAIN_50']} against {v['VP_GAIN_50']} re-optimised. With a
{v['BIG_BUDGET']} budget the risk-first plan falls to {v['BIG_RP_GAIN']}, while the value plan stays at
{v['BIG_VP_GAIN']} because it stops when offers stop paying.</p>
{f['plan_curves']}
{t['SENSITIVITY_TABLE']}
{t['BUDGET_TABLE']}
<h2>Errors by segment</h2>
<p>Within one contract type the ranking is much harder: the contract carries most of the signal.</p>
{f['segments']}
{t['SEGMENT_TABLE']}
<h2>Data</h2>
<p>IBM Telco customer churn sample: {v['ROWS']} customers of a fictional telecom provider, 21 fields, one
snapshot. Churn rate {v['CHURN_RATE']}. Monthly price is almost fully explained by the services held
(R² {v['PRICE_R2']}), so price stays out of the hazard models.</p>
{t['CLEANING_TABLE']}
<h2>Limitations</h2>
<ul>
<li>One snapshot: tenure is observed once per customer, and customers who left before the snapshot month are not
in the file, so long-tenure retention is likely overstated and curves past 72 months are extrapolation.</li>
<li>Contract, services and payment method are current values; a customer may have switched them.</li>
<li>Margin, discount rate, offer cost and the save rate are assumptions. The save rate is not measured; an A/B
test of the offer would replace it.</li>
<li>The data is a sample built for teaching, not a real operator's book.</li>
</ul>
</body></html>
"""


def render_all(cfg: dict[str, Any]) -> None:
    """Write README.md and docs/index.html."""
    results = resolve(cfg["paths"]["results"])
    m = json.loads((results / "metrics.json").read_text())
    template = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(fill(template, m, md_table))
    docs = resolve(cfg["paths"]["docs"])
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(render_html(m, results, docs))
    print("rendered README.md and docs/index.html")
