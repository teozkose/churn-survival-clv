"""Expected remaining lifetime and customer lifetime value from conditional survival curves.

For an active customer at tenure t, the chance of still paying k months from now is S(t + k) / S(t).
Value is the discounted sum of monthly margin times that chance over a fixed horizon."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churn.survival import SurvivalModel


def monthly_discount(annual_rate: float) -> float:
    """Monthly discount factor equivalent to an annual rate."""
    return (1 + annual_rate) ** (-1 / 12)


def conditional_survival(model: SurvivalModel, X: pd.DataFrame, tenure: np.ndarray, horizon: int) -> np.ndarray:
    """n x horizon matrix of S(t + k) / S(t) for k = 1..horizon."""
    tenure = np.asarray(tenure, dtype=int)
    grid = np.arange(0, tenure.max() + horizon + 1)
    curves = model.survival(X, grid)                       # n x len(grid), column j is time j
    rows = np.arange(len(X))[:, None]
    future = curves[rows, tenure[:, None] + np.arange(1, horizon + 1)[None, :]]
    now = np.clip(curves[np.arange(len(X)), tenure], 1e-12, None)
    return np.clip(future / now[:, None], 0.0, 1.0)


def kept_for(cond: np.ndarray, months: int) -> np.ndarray:
    """Conditional survival if the customer is certain to stay `months` more months and then follows their
    own curve from there: 1 up to `months`, then S(t + k) / S(t + months)."""
    out = np.ones_like(cond)
    anchor = np.clip(cond[:, months - 1], 1e-12, None)
    out[:, months:] = np.clip(cond[:, months:] / anchor[:, None], 0.0, 1.0)
    return out


def lifetime_value(cond: np.ndarray, monthly_margin: np.ndarray, annual_rate: float) -> np.ndarray:
    """Discounted margin over the horizon: sum_k margin * survival_k * d^k."""
    d = monthly_discount(annual_rate) ** np.arange(1, cond.shape[1] + 1)
    return (cond * d[None, :]).sum(axis=1) * np.asarray(monthly_margin, dtype=float)


def expected_lifetime(cond: np.ndarray) -> np.ndarray:
    """Expected further months as a customer within the horizon (the area under the conditional curve)."""
    return cond.sum(axis=1)


def customer_values(model: SurvivalModel, active: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per active customer: churn risk over the commitment period, expected remaining months, CLV now and
    CLV if kept through the commitment period."""
    c, t = cfg["clv"], cfg["targeting"]
    cond = conditional_survival(model, active, active["tenure"].to_numpy(), c["horizon_months"])
    margin = active["monthly_charges"].to_numpy() * c["gross_margin"]
    kept = kept_for(cond, t["commitment_months"])
    out = active[["customer_id", "tenure", "monthly_charges", "contract"]].copy()
    out["p_churn_12m"] = 1 - cond[:, t["commitment_months"] - 1]
    out["expected_months"] = expected_lifetime(cond)
    out["clv"] = lifetime_value(cond, margin, c["annual_discount"])
    out["clv_if_kept"] = lifetime_value(kept, margin, c["annual_discount"])
    out["offer_cost"] = t["offer_months_free"] * active["monthly_charges"].to_numpy()
    return out.reset_index(drop=True)
