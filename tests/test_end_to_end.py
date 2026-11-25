"""Whole experiment on synthetic customers: every model trains, metrics are sane, the plan stays in budget."""

import numpy as np

from churn.features import FEATURES, build_table
from churn.pipeline import MODEL_ORDER, run_experiment
from churn.survival import CoxModel
from tests.conftest import make_raw


def test_run_experiment_on_synthetic_customers(cfg):
    table, _ = build_table(make_raw(900, seed=1), cfg)
    metrics, art = run_experiment(table, cfg)
    assert set(metrics["models"]) == set(MODEL_ORDER)
    # the synthetic hazard depends on contract, so every model should rank better than chance
    for name in MODEL_ORDER:
        assert metrics["models"][name]["cindex"] > 0.6
    plan = metrics["targeting"]["value_plan"]
    assert plan["spend"] <= cfg["targeting"]["budget"] + 1e-6
    assert plan["net_gain"] >= metrics["targeting"]["risk_plan"]["net_gain"] - 1e-6
    assert (art["values"]["clv_if_kept"] >= art["values"]["clv"] - 1e-9).all()


def test_stratified_cox_keeps_row_order(cfg):
    table, _ = build_table(make_raw(500, seed=2), cfg)
    model = CoxModel(FEATURES, 0.01, ["contract"]).fit(table)
    shuffled = table.sample(frac=1, random_state=0)
    a = model.survival(shuffled, np.array([12]))[:, 0]
    b = model.survival(table, np.array([12]))[:, 0]
    assert np.allclose(a, b[shuffled.index.to_numpy()])
