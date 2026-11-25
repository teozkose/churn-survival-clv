"""The comparison model: a gradient boosting classifier for churn within a fixed horizon.

It answers the same 12-month question as the survival models but has to drop every customer whose 12-month
outcome is not known yet (active with less than 12 months of tenure)."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def make_xgboost(cfg: dict[str, Any], seed: int, **params: Any) -> XGBClassifier:
    """XGBoost with the fixed settings from config.yaml and the tuned depth and learning rate."""
    x = cfg["models"]["xgboost"]
    return XGBClassifier(n_estimators=x["n_estimators"], subsample=x["subsample"],
                         colsample_bytree=x["colsample_bytree"], min_child_weight=x["min_child_weight"],
                         eval_metric="logloss", random_state=seed, n_jobs=4, **params)


def tune_xgboost(X: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any], seed: int
                 ) -> tuple[dict[str, Any], float]:
    """Grid over depth and learning rate scored by inner stratified k-fold ROC-AUC."""
    grid = cfg["models"]["xgboost"]["grid"]
    split = StratifiedKFold(n_splits=cfg["validation"]["folds"], shuffle=True, random_state=seed)
    best: tuple[dict[str, Any], float] = ({}, -1.0)
    for depth, lr in itertools.product(grid["max_depth"], grid["learning_rate"]):
        params = {"max_depth": depth, "learning_rate": lr}
        aucs = []
        for tr, te in split.split(X, y):
            model = make_xgboost(cfg, seed, **params).fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], model.predict_proba(X.iloc[te])[:, 1]))
        if np.mean(aucs) > best[1]:
            best = (params, float(np.mean(aucs)))
    return best


class HorizonClassifier:
    """Wraps the classifier so it can be scored next to the survival models."""

    name = "xgboost_12m"

    def __init__(self, features: list[str], horizon: int, cfg: dict[str, Any], seed: int) -> None:
        self.features, self.horizon, self.cfg, self.seed = features, horizon, cfg, seed

    def fit(self, df: pd.DataFrame) -> HorizonClassifier:
        known = df[df[f"known_{self.horizon}m"]]
        X, y = known[self.features], known[f"churn_{self.horizon}m"].to_numpy()
        self.params_, self.inner_auc_ = tune_xgboost(X, y, self.cfg, self.seed)
        self.model_ = make_xgboost(self.cfg, self.seed, **self.params_).fit(X, y)
        self.n_train_, self.n_dropped_ = len(known), len(df) - len(known)
        return self

    def churn_by(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        if horizon != self.horizon:
            raise ValueError(f"this classifier only answers the {self.horizon}-month question")
        return self.model_.predict_proba(X[self.features])[:, 1]

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.churn_by(X, self.horizon)
