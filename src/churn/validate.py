"""Schema, code, range, null and duplicate checks on the tidy table. Any hard failure stops the run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

YES_NO = {"Yes", "No"}
ALLOWED: dict[str, set[Any]] = {
    "gender": {"Male", "Female"},
    "senior": {0, 1},
    "partner": YES_NO,
    "dependents": YES_NO,
    "phone_service": YES_NO,
    "multiple_lines": YES_NO | {"No phone service"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    **{c: YES_NO | {"No internet service"} for c in
       ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv",
        "streaming_movies"]},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": YES_NO,
    "payment_method": {"Electronic check", "Mailed check", "Bank transfer (automatic)",
                       "Credit card (automatic)"},
    "churn": YES_NO,
}
INTERNET_ADDONS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv",
                   "streaming_movies"]


class ValidationError(ValueError):
    """Raised with every problem found, not just the first."""


def validate_raw(df: pd.DataFrame, gap_tolerance: float = 0.35) -> dict[str, Any]:
    """Check the tidy table and return a summary. Raises ValidationError listing all hard failures.

    Hard failures: missing columns, unexpected codes, nulls outside total_charges, duplicate ids, negative
    or implausible tenure and charges, add-on services recorded for customers with no internet. Soft findings
    (blank total charges, charges far from tenure x monthly price) are counted and returned."""
    problems: list[str] = []
    required = set(ALLOWED) | {"customer_id", "tenure", "monthly_charges", "total_charges"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(f"missing columns: {missing}")
    for col, allowed in ALLOWED.items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            problems.append(f"{col} has unexpected values {sorted(map(str, bad))}")
    nulls = df.drop(columns="total_charges").isna().sum()
    if nulls.any():
        problems.append(f"nulls in {nulls[nulls > 0].to_dict()}")
    if df["customer_id"].duplicated().any():
        problems.append(f"{int(df['customer_id'].duplicated().sum())} duplicate customer ids")
    if not df["tenure"].between(0, 120).all():
        problems.append("tenure outside 0-120 months")
    if not df["monthly_charges"].between(1, 500).all():
        problems.append("monthly charges outside $1-500")
    if (df["total_charges"].dropna() < 0).any():
        problems.append("negative total charges")
    no_net = df["internet_service"] == "No"
    for col in INTERNET_ADDONS:
        if (df.loc[no_net, col] != "No internet service").any():
            problems.append(f"{col} recorded for customers without internet")
    no_phone = df["phone_service"] == "No"
    if (df.loc[no_phone, "multiple_lines"] != "No phone service").any():
        problems.append("multiple_lines recorded for customers without phone service")
    if problems:
        raise ValidationError("; ".join(problems))

    blank_total = df["total_charges"].isna()
    expected = df["tenure"] * df["monthly_charges"]
    known = ~blank_total & (df["tenure"] > 0)
    gap = (df.loc[known, "total_charges"] - expected[known]).abs() / expected[known]
    dupes = int(df.drop(columns="customer_id").duplicated().sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "churn_rate": float((df["churn"] == "Yes").mean()),
        "blank_total_charges": int(blank_total.sum()),
        "blank_total_tenure": sorted({int(t) for t in df.loc[blank_total, "tenure"]}),
        "zero_tenure": int((df["tenure"] == 0).sum()),
        "charges_gap_median": float(gap.median()),
        "charges_gap_flagged": int((gap > gap_tolerance).sum()),
        "duplicate_rows_ignoring_id": dupes,
        "tenure_range": [int(df["tenure"].min()), int(df["tenure"].max())],
        "monthly_range": [float(df["monthly_charges"].min()), float(df["monthly_charges"].max())],
        "max_tenure_share": float((df["tenure"] == df["tenure"].max()).mean()),
        "active": int((df["churn"] == "No").sum()),
        "monthly_mean": float(np.round(df["monthly_charges"].mean(), 2)),
    }
