"""Who gets a retention offer under a fixed budget.

Expected gain of an offer = save rate x (CLV if kept through the commitment - CLV now) - offer cost.
The value-based plan solves a 0/1 knapsack (maximise total gain, total cost within budget) with PuLP;
the comparison plan offers to the highest churn probabilities until the budget runs out."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pulp


def expected_gain(values: pd.DataFrame, save_rate: float) -> np.ndarray:
    """Dollar gain of making the offer to each customer."""
    uplift = (values["clv_if_kept"] - values["clv"]).to_numpy()
    return save_rate * uplift - values["offer_cost"].to_numpy()


def knapsack(gain: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """0/1 selection maximising total gain with total cost <= budget (CBC via PuLP). Customers with a
    non-positive gain are never worth an offer, so they are left out of the problem."""
    cand = np.flatnonzero(gain > 0)
    chosen = np.zeros(len(gain), dtype=bool)
    if len(cand) == 0:
        return chosen
    if cost[cand].sum() <= budget:
        chosen[cand] = True
        return chosen
    prob = pulp.LpProblem("retention", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in cand]
    prob += pulp.lpSum(float(gain[i]) * v for i, v in zip(cand, x, strict=True))
    prob += pulp.lpSum(float(cost[i]) * v for i, v in zip(cand, x, strict=True)) <= budget
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=1e-6))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"knapsack solver returned {pulp.LpStatus[status]}")
    chosen[cand] = np.array([v.value() > 0.5 for v in x])
    return chosen


def greedy_by(score: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Offer in descending order of `score` while the next offer still fits in the budget."""
    chosen = np.zeros(len(score), dtype=bool)
    spent = 0.0
    for i in np.argsort(-score, kind="stable"):
        if spent + cost[i] <= budget:
            chosen[i] = True
            spent += cost[i]
    return chosen


def plan_summary(chosen: np.ndarray, gain: np.ndarray, cost: np.ndarray, values: pd.DataFrame,
                 save_rate: float) -> dict[str, Any]:
    """Size, spend, expected net gain and expected customers kept for one plan."""
    return {
        "customers": int(chosen.sum()), "spend": float(cost[chosen].sum()),
        "net_gain": float(gain[chosen].sum()),
        "expected_saves": float(save_rate * values["p_churn_12m"].to_numpy()[chosen].sum()),
        "mean_p_churn": float(values["p_churn_12m"].to_numpy()[chosen].mean()) if chosen.any() else 0.0,
        "mean_clv": float(values["clv"].to_numpy()[chosen].mean()) if chosen.any() else 0.0,
        "month_to_month_share": float((values["contract"].to_numpy()[chosen] == "Month-to-month").mean())
        if chosen.any() else 0.0,
    }


def compare_plans(values: pd.DataFrame, save_rate: float, budget: float) -> dict[str, Any]:
    """Value-based knapsack plan against the churn-probability plan at one save rate and budget."""
    gain, cost = expected_gain(values, save_rate), values["offer_cost"].to_numpy()
    by_value = knapsack(gain, cost, budget)
    by_risk = greedy_by(values["p_churn_12m"].to_numpy(), cost, budget)
    return {"save_rate": save_rate, "budget": budget,
            "value_plan": plan_summary(by_value, gain, cost, values, save_rate),
            "risk_plan": plan_summary(by_risk, gain, cost, values, save_rate),
            "overlap": int((by_value & by_risk).sum()),
            "_chosen": {"value": by_value, "risk": by_risk}}
