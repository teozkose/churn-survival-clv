import numpy as np
import pandas as pd
import pytest

from churn.clv import expected_lifetime, kept_for, lifetime_value, monthly_discount
from churn.evaluate import expected_calibration_error
from churn.survival import cindex
from churn.targeting import expected_gain, greedy_by, knapsack


def test_lifetime_value_matches_closed_form_for_constant_hazard():
    q, months, margin, rate = 0.95, 60, 10.0, 0.10
    cond = q ** np.arange(1, months + 1)[None, :]
    d = monthly_discount(rate)
    expected = margin * sum((q * d) ** k for k in range(1, months + 1))
    assert lifetime_value(cond, np.array([margin]), rate)[0] == pytest.approx(expected)
    assert expected_lifetime(cond)[0] == pytest.approx(sum(q ** k for k in range(1, months + 1)))


def test_kept_for_is_one_during_commitment_then_rescaled():
    cond = np.array([[0.9, 0.8, 0.6, 0.3]])
    kept = kept_for(cond, 2)
    assert kept[0, :2].tolist() == [1.0, 1.0]
    assert kept[0, 2:] == pytest.approx([0.6 / 0.8, 0.3 / 0.8])


def test_knapsack_beats_greedy_when_costs_differ():
    gain = np.array([10.0, 9.0, 9.0, -1.0])
    cost = np.array([10.0, 5.0, 5.0, 1.0])
    chosen = knapsack(gain, cost, budget=10)
    assert chosen.tolist() == [False, True, True, False]
    greedy = greedy_by(gain, cost, budget=10)
    assert gain[chosen].sum() > gain[greedy].sum()


def test_expected_gain_subtracts_offer_cost():
    values = pd.DataFrame({"clv": [100.0, 500.0], "clv_if_kept": [300.0, 520.0], "offer_cost": [50.0, 50.0]})
    assert expected_gain(values, 0.5).tolist() == [50.0, -40.0]


def test_cindex_perfect_and_reversed():
    T = np.array([1, 2, 3, 4])
    E = np.array([1, 1, 1, 1])
    assert cindex(T, E, -T) == pytest.approx(1.0)
    assert cindex(T, E, T) == pytest.approx(0.0)


def test_calibration_error_zero_when_predictions_match_rates():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert expected_calibration_error(y, p, bins=5) == pytest.approx(0.0)
