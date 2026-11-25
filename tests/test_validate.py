import numpy as np
import pandas as pd
import pytest

from churn.data import tidy
from churn.validate import ValidationError, validate_raw


def test_clean_table_passes_and_counts_blanks(raw):
    raw.loc[:2, "total_charges"] = np.nan
    summary = validate_raw(raw)
    assert summary["rows"] == len(raw)
    assert summary["blank_total_charges"] == 3


def test_every_problem_is_reported(raw):
    raw.loc[0, "contract"] = "Weekly"
    raw.loc[1, "customer_id"] = raw.loc[2, "customer_id"]
    raw.loc[raw["internet_service"] == "No", "tech_support"] = "Yes"
    with pytest.raises(ValidationError) as err:
        validate_raw(raw)
    msg = str(err.value)
    assert "contract" in msg and "duplicate" in msg and "tech_support" in msg


def test_tidy_turns_blank_total_charges_into_nan():
    src = pd.DataFrame({"customerID": ["a"], "gender": ["Male"], "SeniorCitizen": [0], "Partner": ["No"],
                        "Dependents": ["No"], "tenure": [0], "PhoneService": ["Yes"], "MultipleLines": ["No"],
                        "InternetService": ["No"], **{c: ["No internet service"] for c in
                        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                         "StreamingMovies"]}, "Contract": ["Two year"], "PaperlessBilling": ["No"],
                        "PaymentMethod": ["Mailed check"], "MonthlyCharges": [20.0], "TotalCharges": [" "],
                        "Churn": ["No"]})
    out = tidy(src)
    assert out["total_charges"].isna().all()
    assert out.columns[0] == "customer_id"
