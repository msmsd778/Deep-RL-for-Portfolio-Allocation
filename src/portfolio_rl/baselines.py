"""Non-learning baselines.

"Compared to what?" is the first question anyone will ask about an RL trading
result, and it is the question that sinks most student projects. These
baselines run through the *same* environment, pay the *same* transaction costs
and are measured on the *same* days, so the comparison is apples to apples.

  BuyAndHold      : set weights once at t0, never trade again (weights drift).
  FixedWeight     : rebalance back to a target every step (60/30/10 by default).
  EqualWeight     : the 1/N portfolio, which is a famously hard benchmark.
  InverseVol      : risk-parity-lite; weight inversely to trailing volatility.
  RandomAgent     : uniform random actions. If RL cannot beat this, nothing in
                    the learning pipeline is working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .env import PortfolioEnv


class BasePolicy:
    name = "base"

    def reset(self, env: PortfolioEnv) -> None:
        pass

    def target_weights(self, env: PortfolioEnv) -> np.ndarray:
        raise NotImplementedError


class BuyAndHold(BasePolicy):
    name = "buy_and_hold"

    def __init__(self, weights: np.ndarray | None = None):
        self.init_weights = weights
        self._done_initial = False

    def reset(self, env):
        self._done_initial = False

    def target_weights(self, env):
        if not self._done_initial:
            self._done_initial = True
            if self.init_weights is not None:
                w = np.asarray(self.init_weights, dtype=float)
                return w / w.sum()
            return np.ones(env.n_assets) / env.n_assets
        # Returning current (already drifted) weights => zero turnover, zero cost
        return env.weights.copy()


class FixedWeight(BasePolicy):
    def __init__(self, weights, name: str = "fixed_weight"):
        w = np.asarray(weights, dtype=float)
        self.w = w / w.sum()
        self.name = name

    def target_weights(self, env):
        if len(self.w) != env.n_assets:
            raise ValueError("Fixed weights do not match the asset universe size")
        return self.w.copy()


class EqualWeight(FixedWeight):
    def __init__(self, n_assets: int = 3):
        super().__init__(np.ones(n_assets), name="equal_weight")


class InverseVol(BasePolicy):
    """Weight ∝ 1/σ, using a trailing realised-volatility estimate."""

    name = "inverse_vol"

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def target_weights(self, env):
        lo = max(0, env.t - self.lookback)
        window = env.asset_log_returns[lo : env.t + 1]
        if window.shape[0] < 5:
            return np.ones(env.n_assets) / env.n_assets
        vol = window.std(axis=0, ddof=1)
        vol = np.where(vol < 1e-8, 1e-8, vol)
        inv = 1.0 / vol
        return inv / inv.sum()


class RandomAgent(BasePolicy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def target_weights(self, env):
        if env.action_mode == "discrete":
            return env.action_grid[self.rng.integers(len(env.action_grid))].copy()
        w = self.rng.dirichlet(np.ones(env.n_assets))
        return w


def run_policy(env: PortfolioEnv, policy: BasePolicy) -> pd.DataFrame:
    """Roll a rule-based policy through the environment once, deterministically."""
    env.deterministic_start = True
    env.reset()
    policy.reset(env)
    done = False
    while not done:
        target = policy.target_weights(env)
        # Exact weights, not an action-space index: see `step_with_weights`.
        _, _, terminated, truncated, _ = env.step_with_weights(target)
        done = terminated or truncated
    return env.history_frame()


def default_baselines(n_assets: int, asset_names: list[str] | None = None) -> list[BasePolicy]:
    policies: list[BasePolicy] = [
        EqualWeight(n_assets),
        BuyAndHold(),
        InverseVol(),
        RandomAgent(seed=12345),
    ]
    # The classic 60/30/10 only makes sense for a 3-asset equity/bond/gold sleeve.
    if n_assets == 3:
        policies.insert(0, FixedWeight([0.6, 0.3, 0.1], name="static_60_30_10"))
    return policies
