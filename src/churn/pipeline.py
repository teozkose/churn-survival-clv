"""Stages: data -> features -> run (survival models, evaluation, lifetime value, targeting) -> report.

Usage: python -m churn.pipeline {data,features,run,report}
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import clv, evaluate, figures, targeting
from churn.config import load_config, resolve
from churn.data import fetch_and_cache, load_raw
from churn.features import EVENT, FEATURES, SEGMENTS, build_table, price_explained_by_services
from churn.models import HorizonClassifier
from churn.survival import AFTModel, CoxModel, KMByGroup, choose_aft, km_table, tune_cox_penalizer
from churn.validate import validate_raw

MODEL_ORDER = ["km_contract", "cox", "cox_stratified", "aft", "xgboost_12m"]
KM_TIMES = np.array([12, 24, 48, 72])


def fit_models(train: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the baseline, both Cox models, the best AFT family and the classifier on the training part."""
    seed, mc = cfg["validation"]["seed"], cfg["models"]
    pen, pen_scores = tune_cox_penalizer(train, FEATURES, mc["cox"]["penalizers"], cfg["validation"]["folds"], seed)
    cox = CoxModel(FEATURES, pen).fit(train)
    ph = cox.ph_test()
    failing = ph[ph["p"] < mc["cox"]["ph_alpha"]]["feature"].tolist()
    cox_s = CoxModel(FEATURES, pen, mc["cox"]["strata"]).fit(train)
    ph_s = cox_s.ph_test()
    aft, aic = choose_aft(train, FEATURES, mc["aft"]["candidates"], mc["aft"]["penalizer"])
    xgb = HorizonClassifier(FEATURES, cfg["validation"]["horizon_months"], cfg, seed).fit(train)
    models = {"km_contract": KMByGroup().fit(train), "cox": cox, "cox_stratified": cox_s, "aft": aft,
              "xgboost_12m": xgb}
    info = {
        "cox_penalizer": pen, "cox_penalizer_cv": {str(k): v for k, v in pen_scores.items()},
        "ph_test": ph.round(5).to_dict(orient="records"), "ph_failing": failing,
        "ph_test_stratified": ph_s.round(5).to_dict(orient="records"),
        "ph_failing_stratified": ph_s[ph_s["p"] < mc["cox"]["ph_alpha"]]["feature"].tolist(),
        "aft_aic": aic, "aft_family": aft.family,
        "xgboost_params": xgb.params_, "xgboost_inner_auc": xgb.inner_auc_,
        "classifier_rows": xgb.n_train_, "classifier_dropped": xgb.n_dropped_,
    }
    return models, info


def plan_uncertainty(table: pd.DataFrame, active: pd.DataFrame, chosen: dict[str, np.ndarray], family: str,
                     cfg: dict[str, Any]) -> dict[str, Any]:
    """Refit the lifetime model on bootstrap resamples and re-value the two fixed plans each time."""
    rng = np.random.default_rng(cfg["validation"]["seed"])
    t = cfg["targeting"]
    rows = []
    for _ in range(cfg["clv"]["refits"]):
        sample = table.iloc[rng.integers(0, len(table), len(table))].reset_index(drop=True)
        model = AFTModel(FEATURES, family, cfg["models"]["aft"]["penalizer"]).fit(sample)
        vals = clv.customer_values(model, active, cfg)
        gain = targeting.expected_gain(vals, t["save_rate"])
        rows.append((gain[chosen["value"]].sum(), gain[chosen["risk"]].sum(), vals["clv"].mean()))
    arr = np.array(rows)
    diff = arr[:, 0] - arr[:, 1]
    pct = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    return {"refits": len(arr), "value_plan_ci": pct(arr[:, 0]), "risk_plan_ci": pct(arr[:, 1]),
            "difference_ci": pct(diff), "difference_mean": float(diff.mean()),
            "share_value_better": float((diff > 0).mean()), "mean_clv_ci": pct(arr[:, 2])}


def run_experiment(table: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Everything after feature building. Returns JSON-ready metrics and in-memory artefacts for figures."""
    v, ev, t = cfg["validation"], cfg["evaluation"], cfg["targeting"]
    seed, h = v["seed"], v["horizon_months"]
    table = table.reset_index(drop=True)
    train, test = train_test_split(table, test_size=v["test_share"], stratify=table[EVENT], random_state=seed)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    km = {g: km_table(table, g, KM_TIMES) for g in ["contract", "payment_method"]}
    models, info = fit_models(train, cfg)
    preds = {name: {"risk": m.risk(test), "p": m.churn_by(test, h)} for name, m in models.items()}
    scores, draws = evaluate.score_models(test, preds, h, ev["n_boot"], seed, ev["calibration_bins"])
    comparisons = {
        "aft_vs_km_cindex": evaluate.paired_difference(draws, "aft", "km_contract", "cindex"),
        "cox_stratified_vs_cox_cindex": evaluate.paired_difference(draws, "cox_stratified", "cox", "cindex"),
        "aft_vs_cox_stratified_cindex": evaluate.paired_difference(draws, "aft", "cox_stratified", "cindex"),
        "xgboost_vs_aft_auc": evaluate.paired_difference(draws, "xgboost_12m", "aft", "auc"),
    }
    known = test[f"known_{h}m"].to_numpy()
    yk = test[f"churn_{h}m"].to_numpy()[known]
    curves = {m: evaluate.calibration_curve(yk, preds[m]["p"][known], ev["calibration_bins"])
              for m in ["cox_stratified", "aft", "xgboost_12m"]}
    segments = evaluate.segment_table(test, preds["aft"]["risk"], preds["aft"]["p"], SEGMENTS, h,
                                      ev["segment_boot"], seed)

    # final models on every customer: hazard ratios, lifetime value and the retention plan
    final_cox = CoxModel(FEATURES, info["cox_penalizer"], cfg["models"]["cox"]["strata"]).fit(table)
    hr = final_cox.hazard_ratios()
    final_aft = AFTModel(FEATURES, info["aft_family"], cfg["models"]["aft"]["penalizer"]).fit(table)
    active = table[table[EVENT] == 0].reset_index(drop=True)
    values = clv.customer_values(final_aft, active, cfg)
    base = targeting.compare_plans(values, t["save_rate"], t["budget"])
    chosen = base.pop("_chosen")
    uncertainty = plan_uncertainty(table, active, chosen, info["aft_family"], cfg)
    sensitivity = []
    for s in t["save_rate_sensitivity"]:
        r = targeting.compare_plans(values, s, t["budget"])
        gain = targeting.expected_gain(values, s)
        sensitivity.append({"save_rate": s, "value_plan_gain": r["value_plan"]["net_gain"],
                            "value_plan_customers": r["value_plan"]["customers"],
                            "risk_plan_gain": r["risk_plan"]["net_gain"],
                            "base_plan_gain": float(gain[chosen["value"]].sum()),
                            "overlap_with_base": int((r["_chosen"]["value"] & chosen["value"]).sum())})
    budget_curve = []
    for b in t["budget_curve"]:
        r = targeting.compare_plans(values, t["save_rate"], b)
        budget_curve.append({"budget": b, "value_plan_gain": r["value_plan"]["net_gain"],
                             "risk_plan_gain": r["risk_plan"]["net_gain"],
                             "value_plan_customers": r["value_plan"]["customers"],
                             "value_plan_spend": r["value_plan"]["spend"]})
    gain = targeting.expected_gain(values, t["save_rate"])
    uplift = values["clv_if_kept"] - values["clv"]
    by_contract = (values.assign(gain=gain, value_plan=chosen["value"], risk_plan=chosen["risk"])
                   .groupby("contract")
                   .agg(customers=("customer_id", "size"), mean_clv=("clv", "mean"),
                        mean_months=("expected_months", "mean"), mean_p12=("p_churn_12m", "mean"),
                        value_plan=("value_plan", "sum"), risk_plan=("risk_plan", "sum"))
                   .reset_index())

    vp, rp = base["value_plan"], base["risk_plan"]
    metrics = {
        "headline": {
            "cindex": scores["aft"]["cindex"], "cindex_ci": scores["aft"]["cindex_ci"],
            "value_plan_gain": vp["net_gain"], "risk_plan_gain": rp["net_gain"],
            "gain_ratio": vp["net_gain"] / rp["net_gain"] if rp["net_gain"] > 0 else None,
            "difference": vp["net_gain"] - rp["net_gain"], "difference_ci": uncertainty["difference_ci"],
            "budget": t["budget"], "save_rate": t["save_rate"],
        },
        "split": {"train": len(train), "test": len(test), "test_known": int(known.sum()),
                  "test_events": int(test[EVENT].sum())},
        "km": {g: {"logrank_statistic": k["logrank_statistic"], "logrank_p": k["logrank_p"],
                   "pairwise": k["pairwise"],
                   "levels": {lv: {"n": c["n"], "events": c["events"], "at": c["at"]}
                              for lv, c in k["curves"].items()}} for g, k in km.items()},
        "models": scores, "comparisons": comparisons, "fit": info,
        "calibration": {m: c.round(4).to_dict(orient="list") for m, c in curves.items()},
        "segments": segments,
        "hazard_ratios": hr.round(5).to_dict(orient="records"),
        "clv": {"active": len(values), "mean": float(values["clv"].mean()),
                "median": float(values["clv"].median()), "total": float(values["clv"].sum()),
                "mean_months": float(values["expected_months"].mean()),
                "mean_p12": float(values["p_churn_12m"].mean()),
                "by_contract": by_contract.round(4).to_dict(orient="records"),
                "gross_margin": cfg["clv"]["gross_margin"], "annual_discount": cfg["clv"]["annual_discount"],
                "horizon_months": cfg["clv"]["horizon_months"],
                "positive_gain_customers": int((gain > 0).sum()),
                "break_even_save_rate": float((values["offer_cost"] / uplift)[uplift > 0].min())},
        "targeting": {**base, "uncertainty": uncertainty, "sensitivity": sensitivity,
                      "budget_curve": budget_curve, "offer_months_free": t["offer_months_free"],
                      "commitment_months": t["commitment_months"]},
    }
    artefacts = {"km": km, "hr": hr, "curves": curves, "values": values.assign(gain=gain, **{
        "value_plan": chosen["value"], "risk_plan": chosen["risk"]})}
    return metrics, artefacts


def stage_data(cfg: dict[str, Any]) -> None:
    raw = fetch_and_cache(cfg)
    summary = validate_raw(raw, cfg["validation"]["charges_gap_tolerance"])
    print(f"data: {summary['rows']:,} customers, churn rate {summary['churn_rate']:.3f}")


def stage_features(cfg: dict[str, Any]) -> None:
    raw = load_raw(resolve(cfg["data"]["raw"]))
    summary = validate_raw(raw, cfg["validation"]["charges_gap_tolerance"])
    table, log = build_table(raw, cfg)
    table.to_parquet(resolve(cfg["data"]["table"]), index=False)
    results = resolve(cfg["paths"]["results"])
    results.mkdir(exist_ok=True)
    summary.update({"cleaning": log, "modelling_rows": len(table),
                    "price_r2_from_services": price_explained_by_services(table)})
    (results / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"features: {len(table):,} rows x {len(FEATURES)} covariates")


def stage_run(cfg: dict[str, Any]) -> None:
    t0 = time.time()
    table = pd.read_parquet(resolve(cfg["data"]["table"]))
    results = resolve(cfg["paths"]["results"])
    metrics, art = run_experiment(table, cfg)
    metrics["data"] = json.loads((results / "data_summary.json").read_text())
    metrics["runtime_s"] = round(time.time() - t0, 1)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cols = ["customer_id", "contract", "tenure", "monthly_charges", "p_churn_12m", "expected_months", "clv",
            "clv_if_kept", "offer_cost", "gain", "value_plan", "risk_plan"]
    art["values"][cols].round(4).to_csv(resolve(cfg["data"]["values"]), index=False)
    figures.make_all(metrics, art, results)
    print(f"run: done in {metrics['runtime_s']:.0f}s")


def stage_report(cfg: dict[str, Any]) -> None:
    from churn.report import render_all

    render_all(cfg)


STAGES = {"data": stage_data, "features": stage_features, "run": stage_run, "report": stage_report}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in STAGES:
        raise SystemExit(f"usage: python -m churn.pipeline {{{','.join(STAGES)}}}")
    STAGES[args[0]](load_config())


if __name__ == "__main__":
    main()
