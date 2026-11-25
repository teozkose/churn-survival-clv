"""Synthetic customers shaped like the real file, generated in memory (no download in tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A tidy table with the source's columns and codes; churn hazard rises with month-to-month and fiber."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.22, 0.23])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    def addon() -> np.ndarray:
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))

    hazard = 0.01 + 0.05 * (contract == "Month-to-month") + 0.02 * (internet == "Fiber optic")
    life = rng.exponential(1 / hazard)
    window = rng.integers(1, 73, n)                 # months observed so far
    tenure = np.clip(np.minimum(np.ceil(life), window), 1, 72).astype(int)
    churn = np.where(life <= window, "Yes", "No")
    monthly = np.round(20 + 25 * (internet == "DSL") + 50 * (internet == "Fiber optic") + rng.normal(0, 3, n), 2)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n), "senior": rng.integers(0, 2, n),
        "partner": rng.choice(["Yes", "No"], n), "dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure, "phone_service": phone,
        "multiple_lines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "internet_service": internet,
        "online_security": addon(), "online_backup": addon(), "device_protection": addon(),
        "tech_support": addon(), "streaming_tv": addon(), "streaming_movies": addon(),
        "contract": contract, "paperless_billing": rng.choice(["Yes", "No"], n),
        "payment_method": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                      "Credit card (automatic)"], n),
        "monthly_charges": np.clip(monthly, 18.25, 120), "total_charges": tenure * np.clip(monthly, 18.25, 120),
        "churn": churn,
    })


@pytest.fixture
def cfg() -> dict:
    c = load_config()
    c["evaluation"].update({"n_boot": 20, "segment_boot": 10})
    c["clv"]["refits"] = 3
    c["models"]["xgboost"]["grid"] = {"max_depth": [2], "learning_rate": [0.1]}
    c["models"]["xgboost"]["n_estimators"] = 50
    c["models"]["cox"]["penalizers"] = [0.01]
    c["models"]["aft"]["candidates"] = ["weibull"]
    c["validation"]["folds"] = 3
    c["targeting"].update({"budget": 3000, "budget_curve": [1000, 3000], "save_rate_sensitivity": [0.2, 0.4]})
    return c


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()
