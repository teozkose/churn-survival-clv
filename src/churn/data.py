"""Download the IBM Telco customer churn file and load it with tidy column names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from churn.config import resolve

RENAME: dict[str, str] = {
    "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior", "Partner": "partner",
    "Dependents": "dependents", "tenure": "tenure", "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines", "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup", "DeviceProtection": "device_protection", "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
}


def download(cfg: dict[str, Any], force: bool = False) -> Path:
    """Fetch the CSV once into data/ and return its path."""
    path = resolve(cfg["data"]["csv"])
    if path.exists() and path.stat().st_size >= cfg["data"]["min_csv_bytes"] and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(cfg["data"]["url"], timeout=120)
    resp.raise_for_status()
    if len(resp.content) < cfg["data"]["min_csv_bytes"]:
        raise RuntimeError(f"download too small ({len(resp.content)} bytes); the source may have moved")
    path.write_bytes(resp.content)
    return path


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case. TotalCharges arrives as text with blanks; it becomes float with NaN."""
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"source file is missing columns: {missing}")
    out = df[list(RENAME)].rename(columns=RENAME).copy()
    out["total_charges"] = pd.to_numeric(out["total_charges"].astype(str).str.strip().replace("", None),
                                         errors="coerce")
    out["tenure"] = out["tenure"].astype("int64")
    out["senior"] = out["senior"].astype("int64")
    out["monthly_charges"] = out["monthly_charges"].astype(float)
    return out


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the cached tidy parquet written by `make data`."""
    return pd.read_parquet(path)


def fetch_and_cache(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download, read, tidy and cache the raw table as parquet."""
    df = tidy(pd.read_csv(download(cfg), dtype={"TotalCharges": str}))
    df.to_parquet(resolve(cfg["data"]["raw"]), index=False)
    return df
