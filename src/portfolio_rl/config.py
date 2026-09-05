"""Configuration objects for the portfolio-RL experiments.

Everything that a reviewer might ask "why this number?" about lives here, in one
place, so that the answer is never "it was hard-coded somewhere in the training
loop".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    tickers: list[str] = field(default_factory=lambda: ["SPY", "IEF", "GLD"])
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data"
    # If True, never touch the network: build a regime-switching synthetic series
    # instead. Useful for CI, for offline work, and for reproducibility checks.
    synthetic: bool = False
    synthetic_seed: int = 20260420
    # Transient download failures (rate limits, locked sqlite cache, empty
    # responses) are common. Retry before giving up and silently switching
    # the whole experiment to simulated data.
    download_retries: int = 4
    download_backoff: float = 2.0


@dataclass
class SplitConfig:
    """Time-ordered splits. No shuffling, ever: that would leak the future."""

    train: list[str] = field(default_factory=lambda: ["2009-07-01", "2015-12-31"])
    val: list[str] = field(default_factory=lambda: ["2016-01-01", "2019-12-31"])
    # Named out-of-sample regimes. Each is evaluated separately so that we can
    # talk about generalisation instead of a single averaged number.
    test_regimes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "covid_crash": ["2020-01-01", "2020-06-30"],
            "recovery_2020_21": ["2020-07-01", "2021-12-31"],
            "inflation_2022": ["2022-01-01", "2022-12-31"],
            "calm_2023_24": ["2023-01-01", "2024-12-31"],
        }
    )
    # A crisis window that sits *before* the training set, used as a
    # "train-on-calm / test-on-crisis" stress check.
    stress_regime: dict[str, list[str]] = field(
        default_factory=lambda: {"gfc": ["2007-07-01", "2009-06-30"]}
    )


@dataclass
class EnvConfig:
    window: int = 30                 # look-back length in trading days
    episode_length: int = 252        # ~1 trading year per episode
    transaction_cost_bps: float = 5.0
    initial_cash: float = 1.0
    reward: str = "log_return"       # see rewards.REWARD_REGISTRY
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    # Discrete action space granularity: weights live on a simplex grid with
    # step 1/grid_steps. 3 assets, grid_steps=4  ->  15 actions.
    grid_steps: int = 4
    max_weight: float = 1.0
    allow_cash: bool = False
    reward_scale: float = 100.0      # log-returns are ~1e-3; scale for stable nets


@dataclass
class TrainConfig:
    algos: list[str] = field(default_factory=lambda: ["DQN", "PPO"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    total_timesteps: int = 150_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 3
    device: str = "cpu"              # tiny MLPs: CPU beats GPU here
    dqn: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 5_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {"net_arch": [128, 128]},
        }
    )
    ppo: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 3e-4,
            "n_steps": 1_024,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        }
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    results_dir: str = "results"
    figures_dir: str = "results/figures"

    # ---- (de)serialisation -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return cls(
            data=DataConfig(**raw.get("data", {})),
            splits=SplitConfig(**raw.get("splits", {})),
            env=EnvConfig(**raw.get("env", {})),
            train=TrainConfig(**raw.get("train", {})),
            results_dir=raw.get("results_dir", "results"),
            figures_dir=raw.get("figures_dir", "results/figures"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
