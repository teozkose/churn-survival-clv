import pandas as pd

from churn.features import FEATURES, build_table, encode, fixed_horizon_label


def test_fixed_horizon_label_marks_censored_as_unknown():
    duration = pd.Series([3, 3, 12, 20, 20])
    event = pd.Series([1, 0, 0, 1, 0])
    churned, known = fixed_horizon_label(duration, event, 12)
    assert churned.tolist() == [1, 0, 0, 0, 0]
    assert known.tolist() == [True, False, True, True, True]


def test_encode_never_duplicates_internet_flag(raw):
    X = encode(raw)
    assert list(X.columns) == FEATURES
    no_net = raw["internet_service"] == "No"
    assert X.loc[no_net, ["online_security", "streaming_tv", "internet_dsl", "internet_fiber"]].sum().sum() == 0
    assert set(X.to_numpy().ravel()) <= {0, 1}


def test_build_table_drops_zero_tenure_and_logs_it(raw, cfg):
    raw.loc[:4, "tenure"] = 0
    table, log = build_table(raw, cfg)
    assert len(table) == len(raw) - 5
    assert log[0]["dropped"] == 5
    assert table["event"].isin([0, 1]).all()
