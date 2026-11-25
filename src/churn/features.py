"""Turn the validated table into survival inputs: duration, event flag and 0/1 covariates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DURATION, EVENT = "tenure", "event"

ADDONS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv",
          "streaming_movies"]
FEATURES: list[str] = [
    "senior", "partner", "dependents", "phone_service", "multiple_lines", "internet_dsl", "internet_fiber",
    *ADDONS, "paperless_billing", "contract_one_year", "contract_two_year", "pay_electronic_check",
    "pay_mailed_check", "pay_credit_card",
]
# Columns the stratified Cox model needs instead of the contract dummies.
CONTRACT_CODES = {"Month-to-month": 0, "One year": 1, "Two year": 2}
SEGMENTS = ["contract", "internet_service", "payment_method", "senior_label", "gender"]


def encode(raw: pd.DataFrame) -> pd.DataFrame:
    """0/1 covariates from the text columns. 'No internet service' and 'No phone service' become 0, so the
    add-on flags never duplicate the internet and phone flags (which would make the Cox model singular)."""
    out = pd.DataFrame(index=raw.index)
    out["senior"] = raw["senior"].astype(int)
    for col in ["partner", "dependents", "phone_service", "paperless_billing", *ADDONS]:
        out[col] = (raw[col] == "Yes").astype(int)
    out["multiple_lines"] = (raw["multiple_lines"] == "Yes").astype(int)
    out["internet_dsl"] = (raw["internet_service"] == "DSL").astype(int)
    out["internet_fiber"] = (raw["internet_service"] == "Fiber optic").astype(int)
    out["contract_one_year"] = (raw["contract"] == "One year").astype(int)
    out["contract_two_year"] = (raw["contract"] == "Two year").astype(int)
    out["pay_electronic_check"] = (raw["payment_method"] == "Electronic check").astype(int)
    out["pay_mailed_check"] = (raw["payment_method"] == "Mailed check").astype(int)
    out["pay_credit_card"] = (raw["payment_method"] == "Credit card (automatic)").astype(int)
    return out[FEATURES]


def price_explained_by_services(table: pd.DataFrame) -> float:
    """R-squared of monthly charges on the service flags. Near 1 means price adds nothing the services do not
    already say, which is why price is left out of the hazard models."""
    cols = ["phone_service", "multiple_lines", "internet_dsl", "internet_fiber", *ADDONS]
    X = np.column_stack([np.ones(len(table)), table[cols].to_numpy(float)])
    y = table["monthly_charges"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - resid.var() / y.var())


def fixed_horizon_label(duration: pd.Series, event: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Churned within `horizon` months, and whether that is known.

    A customer still active at tenure < horizon has an unknown outcome (censored), so a plain classifier
    must drop them; the survival models keep them."""
    churned = (event == 1) & (duration <= horizon)
    known = churned | (duration >= horizon)
    return churned.astype(int), known


def build_table(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cleaning steps with counts, then the modelling table."""
    log: list[dict[str, Any]] = []
    df = raw.copy()
    n0 = len(df)
    zero = df["tenure"] == 0
    df = df.loc[~zero].reset_index(drop=True)
    log.append({"step": "drop customers with tenure 0 (signed up, no month billed; TotalCharges blank)",
                "rows_before": n0, "changed": 0, "dropped": int(zero.sum()), "rows_after": len(df)})
    expected = df["tenure"] * df["monthly_charges"]
    gap = (df["total_charges"] - expected).abs() / expected
    tol = cfg["validation"]["charges_gap_tolerance"]
    flagged = gap > tol
    df["charges_gap_flag"] = flagged.astype(int)
    log.append({"step": f"flag TotalCharges more than {tol:.0%} away from tenure x monthly price (kept)",
                "rows_before": len(df), "changed": int(flagged.sum()), "dropped": 0, "rows_after": len(df)})
    dup = df.drop(columns="customer_id").duplicated()
    log.append({"step": "identical profiles under different ids (kept: distinct customers)",
                "rows_before": len(df), "changed": int(dup.sum()), "dropped": 0, "rows_after": len(df)})

    table = encode(df)
    table.insert(0, "customer_id", df["customer_id"])
    table[DURATION] = df["tenure"].astype(int)
    table[EVENT] = (df["churn"] == "Yes").astype(int)
    table["contract"] = df["contract"]
    table["contract_code"] = df["contract"].map(CONTRACT_CODES).astype(int)
    table["internet_service"] = df["internet_service"]
    table["payment_method"] = df["payment_method"]
    table["gender"] = df["gender"].str.lower()
    table["senior_label"] = np.where(df["senior"] == 1, "senior", "under 65")
    table["monthly_charges"] = df["monthly_charges"]
    table["total_charges"] = df["total_charges"]
    h = cfg["validation"]["horizon_months"]
    table[f"churn_{h}m"], table[f"known_{h}m"] = fixed_horizon_label(table[DURATION], table[EVENT], h)
    return table, log
