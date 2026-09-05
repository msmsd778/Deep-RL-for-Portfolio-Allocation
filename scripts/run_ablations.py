#!/usr/bin/env python
"""Run the three ablations the report needs, and tabulate them.

    # turnover penalty sweep -- the headline ablation
    python scripts/run_ablations.py --which turnover

    # reward-function comparison
    python scripts/run_ablations.py --which rewards

    # transaction-cost sensitivity
    python scripts/run_ablations.py --which costs

    # everything (slow)
    python scripts/run_ablations.py --which all

Each sub-run is a full experiment written to `results_ablation/<tag>/`. The
script then collects out-of-sample means into one CSV and one figure, which is
what actually goes on the slide.

Cut the cost with `--algos DQN --seeds 0 1 2 --timesteps 60000` while
developing; use the full grid for the final numbers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

TURNOVER_LAMBDAS = [0.0, 0.002, 0.005, 0.02, 0.05, 0.1]
REWARDS = ["log_return", "differential_sharpe", "cvar_penalised", "turnover_penalised"]
COSTS = [0.0, 5.0, 20.0, 50.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", default="turnover",
                   choices=["turnover", "rewards", "costs", "all"])
    p.add_argument("--out", default="results_ablation")
    p.add_argument("--algos", nargs="+", default=["DQN", "PPO"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--prices-csv", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="don't re-run a variant whose results already exist")
    return p.parse_args()


def build_variants(which: str) -> list[tuple[str, dict]]:
    """(tag, config-overrides) pairs."""
    out: list[tuple[str, dict]] = []
    if which in ("turnover", "all"):
        for lam in TURNOVER_LAMBDAS:
            if lam == 0.0:
                out.append(("turnover_lam0.000", {"env": {"reward": "log_return"}}))
            else:
                out.append((f"turnover_lam{lam:.3f}", {"env": {
                    "reward": "turnover_penalised", "reward_kwargs": {"lam": lam}}}))
    if which in ("rewards", "all"):
        for r in REWARDS:
            out.append((f"reward_{r}", {"env": {"reward": r}}))
    if which in ("costs", "all"):
        for c in COSTS:
            out.append((f"cost_{c:g}bps", {"env": {"transaction_cost_bps": c}}))
    return out


def run_variant(tag: str, overrides: dict, args) -> Path:
    out_dir = Path(args.out) / tag
    results_csv = out_dir / "tables" / "raw_results.csv"
    if args.skip_existing and results_csv.exists():
        print(f"[skip] {tag} (already has results)")
        return out_dir

    cfg_path = Path(args.out) / f"_cfg_{tag}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(overrides, sort_keys=False))

    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_experiments.py"),
        "--config", str(cfg_path),
        "--results-dir", str(out_dir),
        "--algos", *args.algos,
        "--seeds", *[str(s) for s in args.seeds],
        "--timesteps", str(args.timesteps),
    ]
    if args.synthetic:
        cmd.append("--synthetic")
    if args.prices_csv:
        cmd += ["--prices-csv", args.prices_csv]

    print(f"\n=== {tag} ===")
    subprocess.run(cmd, check=True)
    return out_dir


def collect(tags: list[str], args) -> pd.DataFrame:
    rows = []
    for tag in tags:
        f = Path(args.out) / tag / "tables" / "raw_results.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "in_sample" in d:
            d = d[~d.in_sample.astype(bool)]     # out-of-sample only
        rl = d[d.kind == "rl"]
        for algo, g in rl.groupby("strategy"):
            rows.append({
                "variant": tag,
                "algo": algo,
                "sharpe": g.sharpe.mean(),
                "sharpe_sd": g.sharpe.std(),
                "ann_turnover": g.ann_turnover.mean(),
                "max_drawdown": g.max_drawdown.mean(),
                "cagr": g.cagr.mean(),
                "n": len(g),
            })
        base = d[d.kind == "baseline"]
        if not base.empty:
            b = base[base.strategy == "static_60_30_10"]
            if not b.empty:
                rows.append({
                    "variant": tag, "algo": "static_60_30_10",
                    "sharpe": b.sharpe.mean(), "sharpe_sd": 0.0,
                    "ann_turnover": b.ann_turnover.mean(),
                    "max_drawdown": b.max_drawdown.mean(),
                    "cagr": b.cagr.mean(), "n": len(b),
                })
    return pd.DataFrame(rows)


def plot_turnover_sweep(df: pd.DataFrame, out_png: Path) -> None:
    sub = df[df.variant.str.startswith("turnover_lam") & (df.algo != "static_60_30_10")]
    if sub.empty:
        return
    sub = sub.assign(lam=sub.variant.str.replace("turnover_lam", "").astype(float))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for algo, g in sub.groupby("algo"):
        g = g.sort_values("lam")
        ax1.plot(g.lam, g.ann_turnover, "o-", label=algo)
        ax2.errorbar(g.lam, g.sharpe, yerr=g.sharpe_sd, fmt="o-", capsize=3, label=algo)
    ax1.set_xlabel("turnover penalty  λ"); ax1.set_ylabel("annual turnover")
    ax1.set_title("Churning falls as λ rises"); ax1.set_yscale("log"); ax1.legend()
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("turnover penalty  λ"); ax2.set_ylabel("out-of-sample Sharpe")
    ax2.set_title("Sharpe is single-peaked in λ"); ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main() -> int:
    args = parse_args()
    variants = build_variants(args.which)
    tags = [t for t, _ in variants]

    for tag, ov in variants:
        run_variant(tag, ov, args)

    df = collect(tags, args)
    if df.empty:
        print("No results collected.")
        return 1

    out_csv = Path(args.out) / f"summary_{args.which}.csv"
    df.to_csv(out_csv, index=False)
    plot_turnover_sweep(df, Path(args.out) / "figures" / "10_turnover_sweep.png")

    print("\n" + "=" * 78)
    print(f"ABLATION SUMMARY ({args.which}) — out-of-sample means")
    print("=" * 78)
    print(df.sort_values(["variant", "algo"])
            .to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    print(f"\nWritten to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
