"""Out-of-sample evaluation and statistical comparison.

The evaluation protocol:

1. Each trained model is rolled once, deterministically, through every held-out
   period. One pass = one backtest, start to finish, no episode resampling.
2. Metrics are computed per (algo, seed, period).
3. Across seeds we report mean ± 95% CI and run Welch's t-test between DQN and
   PPO, plus a bootstrap CI on the difference. Welch rather than Student
   because there is no reason to assume the two algorithms have equal variance
   across seeds -- in practice DQN's spread is usually much wider.
4. We also report the fraction of seeds that beat the best static baseline,
   which is a more honest headline than a mean that one lucky seed can carry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics as M
from .baselines import default_baselines, run_policy
from .config import Config
from .env import PortfolioEnv
from .train import ALGO_ACTION_MODE, load_model, make_env

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def rollout(model, env: PortfolioEnv, deterministic: bool = True) -> pd.DataFrame:
    """One full deterministic pass over the period. This is the backtest."""
    env.deterministic_start = True
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        if env.action_mode == "discrete":
            action = int(np.asarray(action).reshape(-1)[0])
        else:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.history_frame()


def evaluate_model_on_period(
    algo: str, model, prices, features, cfg: Config, period_name: str, seed: int,
    in_sample: bool = False,
) -> tuple[dict, pd.DataFrame]:
    env = make_env(
        prices, features, cfg, ALGO_ACTION_MODE[algo], deterministic_start=True
    )
    # A full pass, not a 252-day slice: override the episode cap for evaluation.
    env.episode_length = len(prices) - env.window - 1
    hist = rollout(model, env)
    row = {
        "strategy": algo,
        "kind": "rl",
        "seed": seed,
        "period": period_name,
        "in_sample": in_sample,
        **M.summarise(hist["port_return"], turnover=hist["turnover"]),
    }
    for name in env.asset_names:
        row[f"avg_w_{name}"] = float(hist[f"w_{name}"].mean())
    return row, hist


def evaluate_baselines_on_period(
    prices, features, cfg: Config, period_name: str, action_mode: str = "discrete",
    in_sample: bool = False,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    env = make_env(prices, features, cfg, action_mode, deterministic_start=True)
    env.episode_length = len(prices) - env.window - 1

    rows, hists = [], {}
    for policy in default_baselines(env.n_assets, env.asset_names):
        hist = run_policy(env, policy)
        rows.append(
            {
                "strategy": policy.name,
                "kind": "baseline",
                "seed": -1,
                "period": period_name,
                "in_sample": in_sample,
                **M.summarise(hist["port_return"], turnover=hist["turnover"]),
                **{
                    f"avg_w_{n}": float(hist[f"w_{n}"].mean())
                    for n in env.asset_names
                },
            }
        )
        hists[policy.name] = hist
    return rows, hists


def evaluate_all(
    trained: list[dict], splits: dict, cfg: Config
) -> tuple[pd.DataFrame, dict]:
    """Evaluate every trained model and every baseline on every period."""
    rows: list[dict] = []
    curves: dict[tuple, pd.DataFrame] = {}

    eval_periods = [p for p in splits if p != "train"]

    for period in eval_periods:
        prices = splits[period]["prices"]
        features = splits[period]["features"]
        in_sample = bool(splits[period].get("in_sample", False))
        if len(prices) < cfg.env.window + 15:
            logger.warning("Skipping short period '%s' (%d rows)", period, len(prices))
            continue

        base_rows, base_hists = evaluate_baselines_on_period(
            prices, features, cfg, period, in_sample=in_sample
        )
        rows.extend(base_rows)
        for name, h in base_hists.items():
            curves[(name, -1, period)] = h

        for meta in trained:
            algo, seed = meta["algo"], meta["seed"]
            model = load_model(algo, meta["model_path"], device=cfg.train.device)
            row, hist = evaluate_model_on_period(
                algo, model, prices, features, cfg, period, seed,
                in_sample=in_sample,
            )
            rows.append(row)
            curves[(algo, seed, period)] = hist

    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mean_ci(x, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based CI half-width, appropriate for small seed counts."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0.0
    if a.size == 1:
        return float(a[0]), float(a[0]), float(a[0])
    m = a.mean()
    se = stats.sem(a)
    h = se * stats.t.ppf(0.5 + confidence / 2.0, a.size - 1)
    return float(m), float(m - h), float(m + h)


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def bootstrap_diff_ci(
    a, b, n_boot: int = 10_000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    diffs = rng.choice(a, (n_boot, a.size)).mean(1) - rng.choice(b, (n_boot, b.size)).mean(1)
    lo = np.quantile(diffs, (1 - confidence) / 2)
    hi = np.quantile(diffs, 1 - (1 - confidence) / 2)
    return float(lo), float(hi)


def significance_tests(
    results: pd.DataFrame,
    metric: str = "sharpe",
    algo_a: str = "DQN",
    algo_b: str = "PPO",
) -> pd.DataFrame:
    """Welch t-test + Mann-Whitney + bootstrap CI, per evaluation period."""
    rows = []
    for period, grp in results[results.kind == "rl"].groupby("period", observed=True):
        a = grp.loc[grp.strategy == algo_a, metric].to_numpy(float)
        b = grp.loc[grp.strategy == algo_b, metric].to_numpy(float)
        if a.size < 2 or b.size < 2:
            continue
        t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
        try:
            u_stat, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u_stat, p_u = np.nan, np.nan
        lo, hi = bootstrap_diff_ci(a, b)
        ma, ma_lo, ma_hi = mean_ci(a)
        mb, mb_lo, mb_hi = mean_ci(b)
        rows.append(
            {
                "period": period,
                "in_sample": bool(grp["in_sample"].any()) if "in_sample" in grp else False,
                "metric": metric,
                f"{algo_a}_mean": ma,
                f"{algo_a}_ci_lo": ma_lo,
                f"{algo_a}_ci_hi": ma_hi,
                f"{algo_b}_mean": mb,
                f"{algo_b}_ci_lo": mb_lo,
                f"{algo_b}_ci_hi": mb_hi,
                "diff": ma - mb,
                "boot_ci_lo": lo,
                "boot_ci_hi": hi,
                "welch_t": float(t_stat),
                "p_welch": float(p_t),
                "mannwhitney_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                "p_mannwhitney": float(p_u) if np.isfinite(p_u) else np.nan,
                "cohens_d": cohens_d(a, b),
                "significant_05": bool(p_t < 0.05),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_strategy(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean ± CI per (strategy, period) across seeds."""
    metrics = metrics or [
        "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
        "calmar", "cvar_95", "ann_turnover",
    ]
    rows = []
    for (strategy, period), grp in results.groupby(["strategy", "period"], observed=True):
        row = {"strategy": strategy, "period": period, "n_seeds": len(grp),
               "in_sample": bool(grp["in_sample"].any()) if "in_sample" in grp else False}
        for m in metrics:
            if m not in grp:
                continue
            mean, lo, hi = mean_ci(grp[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def beat_baseline_rate(
    results: pd.DataFrame, baseline: str = "static_60_30_10", metric: str = "sharpe"
) -> pd.DataFrame:
    """Fraction of seeds whose out-of-sample metric beats a chosen baseline."""
    rows = []
    for period, grp in results.groupby("period", observed=True):
        ref = grp.loc[grp.strategy == baseline, metric]
        if ref.empty:
            continue
        ref_val = float(ref.iloc[0])
        for algo, sub in grp[grp.kind == "rl"].groupby("strategy", observed=True):
            vals = sub[metric].to_numpy(float)
            rows.append(
                {
                    "period": period,
                    "strategy": algo,
                    "metric": metric,
                    "baseline": baseline,
                    "baseline_value": ref_val,
                    "n_seeds": vals.size,
                    "win_rate": float((vals > ref_val).mean()) if vals.size else 0.0,
                }
            )
    return pd.DataFrame(rows)
