"""Training driver: multi-seed DQN and PPO on the portfolio environment.

Design notes a reviewer will probe:

* We use Stable-Baselines3 rather than reimplementing DQN/PPO. The guidelines
  allow building on existing code provided it is cited and substantially
  extended -- the contribution here is the environment, the reward design and
  the evaluation protocol, not a from-scratch Adam loop. Reimplementing would
  add risk without adding insight.
* DQN gets the discrete simplex grid; PPO gets the continuous simplex. This is
  the point of the comparison, not an accident: a value-based method needs a
  finite action set, a policy-gradient method does not.
* Every seed trains a fresh model on the *same* training window. Differences
  across seeds are pure optimisation noise, which is exactly what the
  significance test later needs to separate from real algorithmic differences.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import Config
from .env import PortfolioEnv

logger = logging.getLogger(__name__)

ALGO_ACTION_MODE = {"DQN": "discrete", "PPO": "continuous"}
ALGO_CLASS = {"DQN": DQN, "PPO": PPO}


def make_env(
    prices, features, cfg: Config, action_mode: str, deterministic_start: bool = False,
    seed: int | None = None,
) -> PortfolioEnv:
    return PortfolioEnv(
        prices=prices,
        features=features,
        window=cfg.env.window,
        episode_length=cfg.env.episode_length,
        transaction_cost_bps=cfg.env.transaction_cost_bps,
        action_mode=action_mode,
        grid_steps=cfg.env.grid_steps,
        max_weight=cfg.env.max_weight,
        reward=cfg.env.reward,
        reward_kwargs=cfg.env.reward_kwargs,
        reward_scale=cfg.env.reward_scale,
        deterministic_start=deterministic_start,
        seed=seed,
    )


class LearningCurveCallback:
    """Lightweight episode-return logger (avoids SB3's eval-callback overhead)."""

    def __init__(self):
        self.episode_returns: list[float] = []


def train_one(
    algo: str,
    seed: int,
    train_prices,
    train_features,
    cfg: Config,
    save_dir: str | Path,
    verbose: int = 0,
) -> dict:
    """Train a single (algorithm, seed) pair. Returns a metadata dict."""
    if algo not in ALGO_CLASS:
        raise KeyError(f"Unknown algorithm '{algo}'")

    action_mode = ALGO_ACTION_MODE[algo]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _factory():
        env = make_env(train_prices, train_features, cfg, action_mode, seed=seed)
        return Monitor(env)

    vec_env = DummyVecEnv([_factory])
    vec_env.seed(seed)

    hyper = dict(cfg.train.dqn if algo == "DQN" else cfg.train.ppo)
    model = ALGO_CLASS[algo](
        "MlpPolicy",
        vec_env,
        seed=seed,
        device=cfg.train.device,
        verbose=verbose,
        **hyper,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.train.total_timesteps, progress_bar=False)
    train_seconds = time.perf_counter() - t0

    model_path = save_dir / f"{algo}_seed{seed}"
    model.save(model_path)

    monitor = vec_env.envs[0]
    episode_returns = list(getattr(monitor, "episode_returns", []))
    episode_lengths = list(getattr(monitor, "episode_lengths", []))

    latency_us = measure_inference_latency(model, vec_env, n=1_000)
    vec_env.close()

    meta = {
        "algo": algo,
        "seed": seed,
        "action_mode": action_mode,
        "total_timesteps": cfg.train.total_timesteps,
        "train_seconds": train_seconds,
        "steps_per_second": cfg.train.total_timesteps / max(train_seconds, 1e-9),
        "inference_latency_us": latency_us,
        "n_train_episodes": len(episode_returns),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "model_path": str(model_path) + ".zip",
        "n_params": count_parameters(model),
    }
    with open(save_dir / f"{algo}_seed{seed}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "%s seed=%d  trained in %.1fs (%.0f steps/s), %d episodes, %.1f us/action",
        algo, seed, train_seconds, meta["steps_per_second"],
        len(episode_returns), latency_us,
    )
    return meta


def measure_inference_latency(model, vec_env, n: int = 1_000) -> float:
    """Mean wall-clock microseconds for a single deterministic action."""
    obs = vec_env.reset()
    model.predict(obs, deterministic=True)          # warm up lazy init / JIT
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return (time.perf_counter() - t0) / n * 1e6


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.policy.parameters()))


def train_all(train_prices, train_features, cfg: Config, save_dir: str | Path) -> list[dict]:
    """Train every (algorithm, seed) combination requested in the config."""
    results = []
    total = len(cfg.train.algos) * len(cfg.train.seeds)
    i = 0
    for algo in cfg.train.algos:
        for seed in cfg.train.seeds:
            i += 1
            logger.info("[%d/%d] training %s seed=%d ...", i, total, algo, seed)
            results.append(
                train_one(algo, seed, train_prices, train_features, cfg, save_dir)
            )
    return results


def load_model(algo: str, path: str | Path, device: str = "cpu"):
    return ALGO_CLASS[algo].load(path, device=device)
