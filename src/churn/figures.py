"""Figures written to results/ (PNG, small enough for the README and the report page)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2b6cb0", "#dd6b20", "#2f855a", "#b83280", "#718096"]
LABELS = {"km_contract": "KM by contract (baseline)", "cox": "Cox PH", "cox_stratified": "Cox, stratified",
          "aft": "AFT (lifetime model)", "xgboost_12m": "XGBoost 12-month"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def km_curves(art: dict[str, Any], metrics: dict[str, Any], out: Path) -> None:
    """Kaplan-Meier retention by contract and by payment method with 95% bands and log-rank p-values."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, group, title in zip(axes, ["contract", "payment_method"], ["Contract", "Payment method"], strict=True):
        k = art["km"][group]
        for color, (level, c) in zip(COLORS, k["curves"].items(), strict=False):
            ax.step(c["times"], c["survival"], where="post", color=color, label=f"{level} (n={c['n']:,})")
            ax.fill_between(c["times"], c["lo"], c["hi"], step="post", color=color, alpha=0.15)
        p = k["logrank_p"]
        ax.set_title(f"{title}: log-rank p {'< 1e-16' if p < 1e-16 else f'= {p:.1g}'}")
        ax.set_xlabel("Tenure (months)")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share still a customer")
    _save(fig, out / "km_curves.png")


def hazard_ratios(art: dict[str, Any], out: Path) -> None:
    """Forest plot of the stratified Cox hazard ratios on a log axis."""
    hr = art["hr"].sort_values("hr")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(hr))
    colors = np.where(hr["p"] < 0.05, COLORS[0], COLORS[4])
    ax.errorbar(hr["hr"], y, xerr=[hr["hr"] - hr["lo"], hr["hi"] - hr["hr"]], fmt="none", ecolor=colors)
    ax.scatter(hr["hr"], y, color=colors, zorder=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, hr["feature"])
    ax.set_xlabel("Hazard ratio (95% CI), stratified by contract; grey = p >= 0.05")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out / "hazard_ratios.png")


def calibration(metrics: dict[str, Any], out: Path) -> None:
    """Predicted against observed 12-month churn by decile on the hold-out customers with known outcome."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for color, (name, c) in zip(COLORS, metrics["calibration"].items(), strict=False):
        ax.plot(c["predicted"], c["observed"], "o-", color=color, label=LABELS[name])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted P(churn within 12 months)")
    ax.set_ylabel("Observed share")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out / "calibration.png")


def targeting_map(art: dict[str, Any], out: Path) -> None:
    """Each active customer by churn risk and lifetime value, coloured by which plan offers to them."""
    v = art["values"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    groups = [("neither plan", ~v["value_plan"] & ~v["risk_plan"], "#cbd5e0"),
              ("risk plan only", ~v["value_plan"] & v["risk_plan"], COLORS[1]),
              ("value plan only", v["value_plan"] & ~v["risk_plan"], COLORS[0]),
              ("both plans", v["value_plan"] & v["risk_plan"], COLORS[2])]
    for label, mask, color in groups:
        ax.scatter(v.loc[mask, "p_churn_12m"], v.loc[mask, "clv"], s=6, color=color, alpha=0.7,
                   label=f"{label} ({int(mask.sum()):,})")
    ax.set_xlabel("P(churn in the next 12 months)")
    ax.set_ylabel("Lifetime value now ($, 5-year horizon)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)
    _save(fig, out / "targeting_map.png")


def plan_curves(metrics: dict[str, Any], out: Path) -> None:
    """Net gain against budget, and against the save-rate assumption, for both plans."""
    t = metrics["targeting"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    b = t["budget_curve"]
    axes[0].plot([r["budget"] / 1000 for r in b], [r["value_plan_gain"] / 1000 for r in b], "o-",
                 color=COLORS[0], label="Value plan (knapsack)")
    axes[0].plot([r["budget"] / 1000 for r in b], [r["risk_plan_gain"] / 1000 for r in b], "o-",
                 color=COLORS[1], label="Highest churn risk first")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Retention budget ($k)")
    axes[0].set_ylabel("Expected net gain ($k)")
    axes[0].set_title(f"Save rate {t['save_rate']:.0%}")
    s = t["sensitivity"]
    rates = [r["save_rate"] * 100 for r in s]
    axes[1].plot(rates, [r["value_plan_gain"] / 1000 for r in s], "o-", color=COLORS[0],
                 label="Value plan, re-optimised")
    axes[1].plot(rates, [r["base_plan_gain"] / 1000 for r in s], "s--", color=COLORS[2],
                 label=f"Value plan fixed at {t['save_rate']:.0%}")
    axes[1].plot(rates, [r["risk_plan_gain"] / 1000 for r in s], "o-", color=COLORS[1],
                 label="Highest churn risk first")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Assumed save rate (%)")
    axes[1].set_title(f"Budget ${t['budget'] / 1000:,.0f}k")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, out / "plan_curves.png")


def segments(metrics: dict[str, Any], out: Path) -> None:
    """Hold-out C-index with 95% CI, and predicted vs observed 12-month churn, by segment level."""
    rows = metrics["segments"]
    labels = [f"{r['segment']}: {r['level']}" for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(rows) + 1.5), sharey=True)
    c = np.array([r["cindex"] for r in rows])
    lo = np.array([r["cindex_ci"][0] for r in rows])
    hi = np.array([r["cindex_ci"][1] for r in rows])
    axes[0].errorbar(c, y, xerr=[c - lo, hi - c], fmt="o", color=COLORS[0])
    axes[0].axvline(metrics["models"]["aft"]["cindex"], color="black", lw=0.8, ls="--")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].set_xlabel("C-index (95% CI); dashed = overall")
    axes[1].scatter([r["observed"] for r in rows], y, color=COLORS[1], label="observed", zorder=3)
    axes[1].scatter([r["predicted"] for r in rows], y, marker="x", color=COLORS[0], label="predicted", zorder=3)
    axes[1].set_xlabel("12-month churn (known outcomes)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, axis="x")
    _save(fig, out / "segments.png")


def make_all(metrics: dict[str, Any], art: dict[str, Any], out: Path) -> None:
    """Write every figure."""
    km_curves(art, metrics, out)
    hazard_ratios(art, out)
    calibration(metrics, out)
    targeting_map(art, out)
    plan_curves(metrics, out)
    segments(metrics, out)
