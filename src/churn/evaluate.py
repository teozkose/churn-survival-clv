"""Hold-out metrics with paired bootstrap CIs, calibration at the fixed horizon, and errors by segment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from churn.survival import cindex


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """n_boot x n row indices drawn with replacement; shared across models so differences are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Count-weighted mean |observed - predicted| over equal-count bins of the prediction."""
    order = np.argsort(p, kind="stable")
    gaps = [abs(y[idx].mean() - p[idx].mean()) * len(idx) for idx in np.array_split(order, bins) if len(idx)]
    return float(sum(gaps) / len(y))


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean prediction and observed rate per equal-count bin."""
    order = np.argsort(p, kind="stable")
    rows = [{"predicted": float(p[i].mean()), "observed": float(y[i].mean()), "n": len(i)}
            for i in np.array_split(order, bins) if len(i)]
    return pd.DataFrame(rows)


def _ci(draws: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def score_models(test: pd.DataFrame, preds: dict[str, dict[str, np.ndarray]], horizon: int, n_boot: int,
                 seed: int, bins: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """C-index on every hold-out customer; ROC-AUC, Brier and calibration error at the horizon on those
    whose outcome is known. `preds[model]` holds 'risk' (all rows) and 'p' (P(churn by horizon), all rows)."""
    T, E = test["tenure"].to_numpy(), test["event"].to_numpy()
    known = test[f"known_{horizon}m"].to_numpy()
    yk = test[f"churn_{horizon}m"].to_numpy()[known]
    idx_all = bootstrap_indices(len(test), n_boot, seed)
    idx_known = bootstrap_indices(int(known.sum()), n_boot, seed + 1)
    out: dict[str, Any] = {}
    draws: dict[str, np.ndarray] = {}
    for name, pr in preds.items():
        risk, pk = pr["risk"], pr["p"][known]
        c_draws = np.array([cindex(T[i], E[i], risk[i]) for i in idx_all])
        a_draws = np.array([roc_auc_score(yk[i], pk[i]) for i in idx_known])
        draws[f"{name}:cindex"], draws[f"{name}:auc"] = c_draws, a_draws
        out[name] = {
            "cindex": cindex(T, E, risk), "cindex_ci": _ci(c_draws),
            "auc": float(roc_auc_score(yk, pk)), "auc_ci": _ci(a_draws),
            "brier": float(brier_score_loss(yk, pk)), "ece": expected_calibration_error(yk, pk, bins),
            "mean_p": float(pk.mean()),
        }
    return out, draws


def paired_difference(draws: dict[str, np.ndarray], a: str, b: str, metric: str) -> dict[str, Any]:
    """Bootstrap mean and 95% CI of metric(a) - metric(b) on the same resamples."""
    d = draws[f"{a}:{metric}"] - draws[f"{b}:{metric}"]
    return {"mean": float(np.mean(d)), "ci": _ci(d), "share_a_better": float(np.mean(d > 0))}


def segment_table(test: pd.DataFrame, risk: np.ndarray, p: np.ndarray, segments: list[str], horizon: int,
                  n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Per segment level: size, churn at the horizon (known outcomes), mean prediction, C-index with CI."""
    rows = []
    known = test[f"known_{horizon}m"].to_numpy()
    y = test[f"churn_{horizon}m"].to_numpy()
    for seg in segments:
        for level in sorted(test[seg].unique()):
            mask = (test[seg] == level).to_numpy()
            T, E, r = test["tenure"].to_numpy()[mask], test["event"].to_numpy()[mask], risk[mask]
            k = mask & known
            draws = [cindex(T[i], E[i], r[i]) for i in bootstrap_indices(int(mask.sum()), n_boot, seed)
                     if E[i].sum() > 1]
            rows.append({"segment": seg, "level": str(level), "n": int(mask.sum()), "events": int(E.sum()),
                         "observed": float(y[k].mean()) if k.any() else float("nan"),
                         "predicted": float(p[k].mean()) if k.any() else float("nan"),
                         "cindex": cindex(T, E, r) if E.sum() > 1 else float("nan"),
                         "cindex_ci": _ci(np.array(draws)) if draws else [float("nan")] * 2})
    return rows
