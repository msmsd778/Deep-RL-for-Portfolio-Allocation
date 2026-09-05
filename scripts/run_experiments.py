#!/usr/bin/env python
"""End-to-end experiment driver.

    python scripts/run_experiments.py --config configs/default.yaml

Stages (each can be skipped once it has run, see --skip-training):
  1. load or generate prices, build causal features
  2. split by time, standardise using TRAIN moments only
  3. train every (algorithm, seed) pair
  4. roll every model and every baseline through every held-out period
  5. compute significance tests and write all tables + figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_rl.config import Config                    # noqa: E402
from portfolio_rl.data import (                           # noqa: E402
    build_features, load_prices, looks_synthetic, make_splits, standardise,
)
from portfolio_rl.evaluate import (                       # noqa: E402
    aggregate_by_strategy, beat_baseline_rate, evaluate_all, significance_tests,
)
from portfolio_rl.plots import make_all_figures           # noqa: E402
from portfolio_rl.train import train_all                  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--results-dir", default=None,
                   help="override the results directory from the config")
    p.add_argument("--reward", default=None,
                   help="override the reward function (log_return | "
                        "differential_sharpe | cvar_penalised)")
    p.add_argument("--timesteps", type=int, default=None,
                   help="override total training timesteps per run")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="force the offline synthetic price generator")
    p.add_argument("--strict-splits", action="store_true",
                   help="abort if any evaluation period overlaps the training "
                        "window (default: warn loudly and tag the period)")
    p.add_argument("--require-real-data", action="store_true",
                   help="abort instead of falling back to synthetic data. Use "
                        "this for the run you intend to present.")
    p.add_argument("--download-retries", type=int, default=None,
                   help="how many times to retry a failed price download "
                        "before falling back to synthetic data")
    p.add_argument("--prices-csv", default=None,
                   help="load prices from your own CSV (Date,SPY,IEF,GLD) "
                        "instead of yfinance. Bypasses the network entirely.")
    p.add_argument("--skip-training", action="store_true",
                   help="reuse models already on disk and only re-evaluate")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke-test run: 2 seeds, 8k steps")
    return p.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.results_dir:
        cfg.results_dir = args.results_dir
        cfg.figures_dir = str(Path(args.results_dir) / "figures")
    if args.reward:
        cfg.env.reward = args.reward
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps
    if args.seeds:
        cfg.train.seeds = args.seeds
    if args.algos:
        cfg.train.algos = args.algos
    if args.synthetic:
        cfg.data.synthetic = True
    if args.download_retries is not None:
        cfg.data.download_retries = args.download_retries
    if args.quick:
        cfg.train.seeds = cfg.train.seeds[:2]
        cfg.train.total_timesteps = 8_000
    return cfg


def main() -> int:
    args = parse_args()
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg = apply_overrides(cfg, args)

    results_dir = Path(cfg.results_dir)
    models_dir = results_dir / "models"
    tables_dir = results_dir / "tables"
    for d in (results_dir, models_dir, tables_dir, Path(cfg.figures_dir)):
        d.mkdir(parents=True, exist_ok=True)
    cfg.save(results_dir / "config_used.yaml")

    t_start = time.perf_counter()

    # ---- 1. data --------------------------------------------------------- #
    log.info("Loading prices ...")
    prices = load_prices(
        cfg.data.tickers, cfg.data.start, cfg.data.end,
        cache_dir=cfg.data.cache_dir, synthetic=cfg.data.synthetic,
        synthetic_seed=cfg.data.synthetic_seed, csv_path=args.prices_csv,
        retries=cfg.data.download_retries, backoff=cfg.data.download_backoff,
        require_real=args.require_real_data,
    )
    log.info("Prices: %d rows, %s .. %s", len(prices),
             prices.index[0].date(), prices.index[-1].date())
    if looks_synthetic(prices):
        log.warning("-" * 70)
        log.warning("DATA PROVENANCE: this series covers every business day with "
                    "no exchange holidays, i.e. it is SIMULATED, not market data.")
        log.warning("Regime labels below are date ranges only. Do not present "
                    "these numbers as real results.")
        log.warning("-" * 70)

    features = build_features(prices, window=cfg.env.window)
    log.info("Features: %d rows x %d columns", *features.shape)

    # ---- 2. split, then standardise using TRAIN statistics only ---------- #
    splits = make_splits(prices, features, cfg.splits, strict=args.strict_splits)
    log.info("Periods: %s", ", ".join(
        f"{k}({len(v['prices'])}){'[IN-SAMPLE]' if v.get('in_sample') else ''}"
        for k, v in splits.items()))

    train_feat = splits["train"]["features"]
    other_feat = {k: v["features"] for k, v in splits.items() if k != "train"}
    scaled_train, scaled_other, moments = standardise(train_feat, other_feat)
    splits["train"]["features"] = scaled_train
    for k, v in scaled_other.items():
        splits[k]["features"] = v
    pd.DataFrame(moments).to_csv(tables_dir / "feature_scaling_moments.csv")

    # ---- 3. train -------------------------------------------------------- #
    meta_path = results_dir / "training_meta.json"
    if args.skip_training and meta_path.exists():
        log.info("Reusing existing models from %s", models_dir)
        trained = json.loads(meta_path.read_text())
    else:
        trained = train_all(
            splits["train"]["prices"], splits["train"]["features"], cfg, models_dir
        )
        meta_path.write_text(json.dumps(trained, indent=2))

    # ---- 4. evaluate ----------------------------------------------------- #
    log.info("Evaluating on %d held-out periods ...", len(splits) - 1)
    results, curves = evaluate_all(trained, splits, cfg)
    results.to_csv(tables_dir / "raw_results.csv", index=False)

    agg = aggregate_by_strategy(results)
    agg.to_csv(tables_dir / "aggregated_results.csv", index=False)

    # ---- 5. statistics + figures ----------------------------------------- #
    sig_frames = []
    for metric in ("sharpe", "cagr", "max_drawdown", "cvar_95"):
        s = significance_tests(results, metric=metric)
        if not s.empty:
            sig_frames.append(s)
    sig = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    if not sig.empty:
        sig.to_csv(tables_dir / "significance_tests.csv", index=False)

    wins = beat_baseline_rate(results, baseline="static_60_30_10", metric="sharpe")
    if not wins.empty:
        wins.to_csv(tables_dir / "baseline_win_rates.csv", index=False)

    sharpe_sig = sig[sig.metric == "sharpe"] if not sig.empty else pd.DataFrame()
    make_all_figures(prices, trained, results, curves, sharpe_sig, cfg.figures_dir)

    curves_dir = results_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for (strategy, seed, period), hist in curves.items():
        if seed in (-1, 0) and not hist.empty:
            hist.to_csv(curves_dir / f"{strategy}_seed{seed}_{period}.csv")

    # ---- summary --------------------------------------------------------- #
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"DONE in {elapsed/60:.1f} min — results in {results_dir.resolve()}")
    print("=" * 78)
    cols = ["strategy", "period", "sharpe_mean", "max_drawdown_mean", "cagr_mean"]
    oos = agg[~agg.get("in_sample", False)]
    ins = agg[agg.get("in_sample", False)] if "in_sample" in agg else agg.iloc[0:0]
    print("OUT-OF-SAMPLE (this is the result):")
    print(oos[cols].sort_values(["period", "sharpe_mean"], ascending=[True, False])
          .to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    if not ins.empty:
        print("\nIN-SAMPLE / OVERLAPS TRAINING (memorisation check only, NOT a result):")
        print(ins[cols].sort_values(["period", "sharpe_mean"], ascending=[True, False])
              .to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    if not sharpe_sig.empty:
        print("\nDQN vs PPO (Sharpe):")
        print(
            sharpe_sig[["period", "DQN_mean", "PPO_mean", "diff", "p_welch",
                        "cohens_d", "significant_05"]]
            .to_string(index=False, float_format=lambda v: f"{v: .3f}")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
