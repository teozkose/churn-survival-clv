"""Survival models behind one small interface: survival curves, a risk score and P(churn by month h).

Every model exposes `survival(X, times)` (n x len(times) array), so evaluation, calibration and lifetime
value never need to know which model they are talking to."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from churn.features import DURATION, EVENT

RMST_TIMES = np.arange(1, 73)  # risk score = minus restricted mean survival over the observed 72 months
AFT_FITTERS = {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter, "loglogistic": LogLogisticAFTFitter}


class SurvivalModel:
    """Base class: subclasses implement `survival`; risk and fixed-horizon churn come for free."""

    name = "model"

    def survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        """Minus the restricted mean survival time up to 72 months: comparable across strata and models."""
        return -self.survival(X, RMST_TIMES).sum(axis=1)

    def churn_by(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """P(churn on or before month `horizon` of tenure)."""
        return 1 - self.survival(X, np.array([horizon]))[:, 0]


class KMByGroup(SurvivalModel):
    """Naive baseline: one Kaplan-Meier curve per contract type, nothing else about the customer."""

    name = "km_contract"

    def __init__(self, group: str = "contract") -> None:
        self.group = group
        self.curves: dict[Any, KaplanMeierFitter] = {}

    def fit(self, df: pd.DataFrame) -> KMByGroup:
        for level, part in df.groupby(self.group):
            self.curves[level] = KaplanMeierFitter().fit(part[DURATION], part[EVENT])
        return self

    def survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        out = np.empty((len(X), len(times)))
        for level, km in self.curves.items():
            rows = (X[self.group] == level).to_numpy()
            out[rows] = km.survival_function_at_times(times).to_numpy()
        return out


class CoxModel(SurvivalModel):
    """Cox proportional hazards, optionally stratified (one baseline hazard per stratum)."""

    def __init__(self, features: list[str], penalizer: float = 0.01, strata: list[str] | None = None) -> None:
        self.features, self.penalizer, self.strata = features, penalizer, strata or []
        self.name = "cox_stratified" if self.strata else "cox"
        self.fitter = CoxPHFitter(penalizer=penalizer)

    def _frame(self, df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
        cols = [c for c in self.features if c not in self._dropped()] + self.strata
        return df[cols + ([DURATION, EVENT] if with_target else [])]

    def _dropped(self) -> set[str]:
        """Dummies of a stratification variable leave the covariates (the strata absorb them)."""
        return {c for c in self.features for s in self.strata if c.startswith(s + "_")}

    def fit(self, df: pd.DataFrame) -> CoxModel:
        self.train_ = self._frame(df)
        self.fitter.fit(self.train_, DURATION, EVENT, strata=self.strata or None)
        return self

    def survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        frame = self._frame(X, with_target=False).reset_index(drop=True)
        sf = self.fitter.predict_survival_function(frame, times=times)
        # a stratified model returns columns grouped by stratum; put them back in row order
        return sf.reindex(columns=frame.index).to_numpy().T

    def hazard_ratios(self) -> pd.DataFrame:
        """Hazard ratio, 95% CI and p-value per covariate."""
        s = self.fitter.summary
        return pd.DataFrame({"feature": s.index, "hr": s["exp(coef)"].to_numpy(),
                             "lo": s["exp(coef) lower 95%"].to_numpy(), "hi": s["exp(coef) upper 95%"].to_numpy(),
                             "p": s["p"].to_numpy()})

    def ph_test(self) -> pd.DataFrame:
        """Schoenfeld-residual test of proportional hazards per covariate (rank time transform)."""
        res = proportional_hazard_test(self.fitter, self.train_, time_transform="rank")
        s = res.summary
        return pd.DataFrame({"feature": [i if isinstance(i, str) else i[0] for i in s.index],
                             "statistic": s["test_statistic"].to_numpy(), "p": s["p"].to_numpy()})


class AFTModel(SurvivalModel):
    """Parametric accelerated failure time model; unlike Cox it extrapolates past the 72 observed months,
    which lifetime value needs."""

    def __init__(self, features: list[str], family: str = "weibull", penalizer: float = 0.01) -> None:
        self.features, self.family, self.name = features, family, f"aft_{family}"
        self.fitter = AFT_FITTERS[family](penalizer=penalizer)

    def fit(self, df: pd.DataFrame) -> AFTModel:
        self.fitter.fit(df[self.features + [DURATION, EVENT]], DURATION, EVENT)
        return self

    def survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        return self.fitter.predict_survival_function(X[self.features], times=times).to_numpy().T

    @property
    def aic(self) -> float:
        return float(self.fitter.AIC_)


def cindex(duration: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance: share of comparable pairs where the higher-risk customer churned first."""
    return float(concordance_index(duration, -np.asarray(risk), event))


def tune_cox_penalizer(df: pd.DataFrame, features: list[str], penalizers: list[float], folds: int, seed: int,
                       strata: list[str] | None = None) -> tuple[float, dict[float, float]]:
    """Pick the L2 penalty with the best mean inner-fold concordance."""
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: dict[float, float] = {}
    for pen in penalizers:
        fold_scores = []
        for tr, te in split.split(df, df[EVENT]):
            m = CoxModel(features, pen, strata).fit(df.iloc[tr])
            part = df.iloc[te]
            fold_scores.append(cindex(part[DURATION].to_numpy(), part[EVENT].to_numpy(), m.risk(part)))
        scores[pen] = float(np.mean(fold_scores))
    best = max(scores, key=scores.get)
    return best, scores


def choose_aft(df: pd.DataFrame, features: list[str], families: list[str], penalizer: float
               ) -> tuple[AFTModel, dict[str, float]]:
    """Fit each AFT family and keep the lowest AIC."""
    fits = {f: AFTModel(features, f, penalizer).fit(df) for f in families}
    aic = {f: m.aic for f, m in fits.items()}
    return fits[min(aic, key=aic.get)], aic


def km_table(df: pd.DataFrame, group: str, times: np.ndarray) -> dict[str, Any]:
    """Kaplan-Meier curves with 95% CIs per level, plus multivariate and pairwise log-rank tests."""
    curves = {}
    for level, part in df.groupby(group):
        km = KaplanMeierFitter().fit(part[DURATION], part[EVENT])
        ci = km.confidence_interval_survival_function_
        curves[str(level)] = {
            "n": len(part), "events": int(part[EVENT].sum()),
            "times": km.survival_function_.index.to_numpy().tolist(),
            "survival": km.survival_function_.iloc[:, 0].to_numpy().tolist(),
            "lo": ci.iloc[:, 0].to_numpy().tolist(), "hi": ci.iloc[:, 1].to_numpy().tolist(),
            "at": {int(t): float(km.survival_function_at_times(t).iloc[0]) for t in times},
        }
    overall = multivariate_logrank_test(df[DURATION], df[group], df[EVENT])
    pairs = pairwise_logrank_test(df[DURATION], df[group], df[EVENT]).summary
    pairwise = [{"a": str(i[0]), "b": str(i[1]), "statistic": float(r["test_statistic"]), "p": float(r["p"])}
                for i, r in pairs.iterrows()]
    return {"curves": curves, "logrank_statistic": float(overall.test_statistic),
            "logrank_p": float(overall.p_value), "pairwise": pairwise}
