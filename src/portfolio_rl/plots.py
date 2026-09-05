"""Figures for the 20-slide presentation.

Every figure is saved as both PNG (for quick viewing) and PDF (vector, for
LaTeX/Beamer inclusion). Slide-quality plots are explicitly part of the
assessment criteria, so this is not decoration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import mean_ci

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

PALETTE = {
    "DQN": "#1f77b4",
    "PPO": "#d62728",
    "static_60_30_10": "#2ca02c",
    "equal_weight": "#9467bd",
    "buy_and_hold": "#8c564b",
    "inverse_vol": "#e377c2",
    "random": "#7f7f7f",
}


def _fix_date_ticks(fig) -> None:
    """Rotate x tick labels so short date ranges don't overlap into mush."""
    for ax in fig.get_axes():
        labels = ax.get_xticklabels()
        if labels and any("-" in lab.get_text() for lab in labels):
            for lab in labels:
                lab.set_rotation(30)
                lab.set_ha("right")


def _save(fig, out_dir: Path, name: str) -> None:
    _fix_date_ticks(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    logger.info("saved figure %s", name)


# --------------------------------------------------------------------------- #
def plot_prices(prices: pd.DataFrame, out_dir: Path, name: str = "01_prices") -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    normed = prices / prices.iloc[0]
    for col in normed.columns:
        ax.plot(normed.index, normed[col], label=col, lw=1.2)
    ax.set_title("Asset universe (normalised to 1.0 at series start)")
    ax.set_ylabel("Growth of 1 unit")
    ax.legend()
    _save(fig, out_dir, name)


def plot_learning_curves(
    trained: list[dict], out_dir: Path, name: str = "02_learning_curves", smooth: int = 20
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    algos = sorted({m["algo"] for m in trained})
    for ax, algo in zip(axes, algos):
        runs = [m for m in trained if m["algo"] == algo]
        max_len = max((len(m["episode_returns"]) for m in runs), default=0)
        if max_len == 0:
            continue
        mat = np.full((len(runs), max_len), np.nan)
        for i, m in enumerate(runs):
            r = np.asarray(m["episode_returns"], dtype=float)
            mat[i, : len(r)] = r
        sm = pd.DataFrame(mat).T.rolling(smooth, min_periods=1).mean().T.to_numpy()
        mean = np.nanmean(sm, axis=0)
        sd = np.nanstd(sm, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, color=PALETTE.get(algo, "k"), lw=1.5, label=f"{algo} mean")
        ax.fill_between(x, mean - sd, mean + sd, color=PALETTE.get(algo, "k"), alpha=0.2,
                        label="±1 sd across seeds")
        ax.set_title(f"{algo} training episode return")
        ax.set_xlabel("training episode")
        ax.set_ylabel("episode return (scaled reward)")
        ax.legend()
    fig.suptitle("Learning curves, averaged over seeds", y=1.02)
    _save(fig, out_dir, name)


def plot_equity_curves(
    curves: dict, period: str, out_dir: Path, name: str | None = None
) -> None:
    name = name or f"03_equity_{period}"
    fig, ax = plt.subplots(figsize=(9, 4.5))

    rl_by_algo: dict[str, list[pd.Series]] = {}
    for (strategy, seed, per), hist in curves.items():
        if per != period or hist.empty:
            continue
        eq = (1.0 + hist["port_return"]).cumprod()
        if seed == -1:
            ax.plot(eq.index, eq.to_numpy(), lw=1.3, ls="--",
                    color=PALETTE.get(strategy, None), label=strategy)
        else:
            rl_by_algo.setdefault(strategy, []).append(eq)

    for algo, series_list in rl_by_algo.items():
        mat = pd.concat(series_list, axis=1).ffill()
        mean, lo, hi = mat.mean(1), mat.min(1), mat.max(1)
        ax.plot(mean.index, mean.to_numpy(), lw=2.0,
                color=PALETTE.get(algo, None), label=f"{algo} (seed mean)")
        ax.fill_between(mean.index, lo.to_numpy(), hi.to_numpy(),
                        color=PALETTE.get(algo, None), alpha=0.15,
                        label=f"{algo} seed min-max")

    ax.axhline(1.0, color="k", lw=0.7, alpha=0.5)
    ax.set_title(f"Out-of-sample equity curves — {period}")
    ax.set_ylabel("Growth of 1 unit (net of costs)")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_dir, name)


def plot_metric_bars(
    results: pd.DataFrame, metric: str, out_dir: Path, name: str | None = None
) -> None:
    name = name or f"04_{metric}_bars"
    periods = sorted(results["period"].unique())
    strategies = sorted(results["strategy"].unique())

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(periods)), 4.5))
    width = 0.8 / max(len(strategies), 1)
    xs = np.arange(len(periods))

    for i, strat in enumerate(strategies):
        means, errs = [], []
        for per in periods:
            vals = results.loc[
                (results.strategy == strat) & (results.period == per), metric
            ]
            m, lo, hi = mean_ci(vals)
            means.append(m)
            errs.append(max(hi - m, 0.0))
        ax.bar(xs + i * width, means, width, yerr=errs, capsize=2.5,
               label=strat, color=PALETTE.get(strat, None), alpha=0.9)

    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(periods, rotation=20, ha="right")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by strategy and out-of-sample period (95% CI across seeds)")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, out_dir, name)


def plot_weight_allocation(
    curves: dict, strategy: str, seed: int, period: str, out_dir: Path,
    name: str | None = None,
) -> None:
    name = name or f"05_weights_{strategy}_{period}"
    hist = curves.get((strategy, seed, period))
    if hist is None or hist.empty:
        return
    wcols = [c for c in hist.columns if c.startswith("w_")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.stackplot(hist.index, *[hist[c].to_numpy() for c in wcols],
                 labels=[c[2:] for c in wcols], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("portfolio weight")
    ax.set_title(f"{strategy} (seed {seed}) allocation over time — {period}")
    ax.legend(loc="upper right", ncol=len(wcols), fontsize=8)
    _save(fig, out_dir, name)


def plot_generalisation_heatmap(
    results: pd.DataFrame, metric: str, out_dir: Path, name: str | None = None
) -> None:
    name = name or f"06_generalisation_{metric}"
    pivot = (
        results.pivot_table(index="strategy", columns="period", values=metric,
                            aggfunc="mean", observed=True)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(1.5 * len(pivot.columns) + 3, 0.55 * len(pivot) + 2))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Generalisation across regimes — mean {metric}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
    _save(fig, out_dir, name)


def plot_drawdowns(curves: dict, period: str, out_dir: Path, name: str | None = None) -> None:
    name = name or f"07_drawdown_{period}"
    fig, ax = plt.subplots(figsize=(9, 3.6))
    seen = set()
    for (strategy, seed, per), hist in curves.items():
        if per != period or hist.empty:
            continue
        if seed not in (-1, 0):
            continue
        eq = (1.0 + hist["port_return"]).cumprod()
        dd = eq / eq.cummax() - 1.0
        label = strategy if strategy not in seen else None
        seen.add(strategy)
        ax.plot(dd.index, dd.to_numpy(), lw=1.2, label=label,
                color=PALETTE.get(strategy, None),
                ls="--" if seed == -1 else "-")
    ax.set_ylabel("drawdown")
    ax.set_title(f"Drawdown profile — {period} (RL: seed 0)")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, out_dir, name)


def plot_efficiency(trained: list[dict], out_dir: Path, name: str = "08_efficiency") -> None:
    df = pd.DataFrame(trained)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title, unit in zip(
        axes,
        ["train_seconds", "inference_latency_us"],
        ["Wall-clock training time", "Inference latency per action"],
        ["seconds", "microseconds"],
    ):
        algos = sorted(df["algo"].unique())
        data = [df.loc[df.algo == a, col].to_numpy() for a in algos]
        # Note: do NOT pass `labels=` / `tick_labels=` to boxplot. The kwarg was
        # renamed in matplotlib 3.9 and the old name removed in 3.11, so either
        # spelling breaks on some supported version. Setting tick labels
        # afterwards works identically on every version.
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        ax.set_xticks(range(1, len(algos) + 1))
        ax.set_xticklabels(algos)
        for patch, a in zip(bp["boxes"], algos):
            patch.set_facecolor(PALETTE.get(a, "#cccccc"))
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel(unit)
    fig.suptitle("Engineering cost of each algorithm (across seeds)", y=1.03)
    _save(fig, out_dir, name)


def plot_significance(sig: pd.DataFrame, out_dir: Path, name: str = "09_significance") -> None:
    if sig.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sig) + 2.2))
    y = np.arange(len(sig))
    ax.errorbar(
        sig["diff"], y,
        xerr=[sig["diff"] - sig["boot_ci_lo"], sig["boot_ci_hi"] - sig["diff"]],
        fmt="o", capsize=4, color="#333333",
    )
    ax.axvline(0.0, color="crimson", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.period}  (p={r.p_welch:.3f})" for r in sig.itertuples()]
    )
    ax.set_xlabel("DQN − PPO  (bootstrap 95% CI)")
    ax.set_title("Paired comparison: difference in mean Sharpe")
    _save(fig, out_dir, name)


def make_all_figures(
    prices, trained, results, curves, sig, figures_dir: str | Path
) -> None:
    out = Path(figures_dir)
    plot_prices(prices, out)
    plot_learning_curves(trained, out)
    plot_efficiency(trained, out)
    for metric in ("sharpe", "max_drawdown", "cagr"):
        if metric in results:
            plot_metric_bars(results, metric, out)
    plot_generalisation_heatmap(results, "sharpe", out)
    plot_significance(sig, out)

    periods = sorted({p for (_, _, p) in curves})
    for period in periods:
        plot_equity_curves(curves, period, out)
        plot_drawdowns(curves, period, out)
        for algo in ("DQN", "PPO"):
            if (algo, 0, period) in curves:
                plot_weight_allocation(curves, algo, 0, period, out)
