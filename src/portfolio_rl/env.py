"""A custom Gymnasium environment for daily portfolio allocation.

MDP specification
-----------------
State  s_t : concatenation of
             (a) standardised causal features for every asset over the last
                 `window` days, flattened;
             (b) the weights the agent is currently holding (post-drift).
             All of (a) is computed from data up to and including t-1.

Action a_t : the *target* weight vector for the next day.
             - discrete   : an index into a fixed simplex grid (step 1/k).
                            This is what DQN needs.
             - continuous : a real vector mapped to the simplex by softmax.
                            This is what PPO can exploit.

Transition : weights are set to the target (paying transaction costs on the
             turnover), the market moves, and the weights drift with the
             realised asset returns.

Reward  r_t: one of the three functions in `rewards.py`, scaled by
             `reward_scale` because raw daily log-returns (~1e-3) are far too
             small for stable value-function learning.

Termination: the episode ends after `episode_length` steps or when the price
             series runs out. There is no early bankruptcy termination -- with
             long-only weights and no leverage, the portfolio cannot go to zero.

Stochasticity
-------------
The price path itself is deterministic (it is history). Stochasticity enters
through the random choice of episode start index at every `reset()`, which
turns a single price series into a distribution over episodes. This is the
right notion of randomness for backtest-style RL: it prevents the agent from
memorising one particular trajectory, without pretending we can resample the
market. During evaluation we switch to `deterministic_start=True` and walk the
whole period once, which is what a backtest actually is.
"""

from __future__ import annotations

from itertools import product

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .rewards import BaseReward, make_reward


def simplex_grid(n_assets: int, steps: int, max_weight: float = 1.0) -> np.ndarray:
    """All weight vectors on the n-simplex with resolution 1/steps.

    Size is C(steps + n - 1, n - 1): for (3 assets, steps=4) that is 15 actions,
    for (3, 10) it is 66. Keep this small -- DQN's output layer is one unit per
    action and Q-learning's sample complexity grows with |A|.
    """
    combos = []
    for c in product(range(steps + 1), repeat=n_assets):
        if sum(c) == steps:
            w = np.asarray(c, dtype=np.float64) / steps
            if w.max() <= max_weight + 1e-9:
                combos.append(w)
    return np.asarray(combos)


class PortfolioEnv(gym.Env):
    """Long-only, fully-invested, daily-rebalanced portfolio environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        window: int = 30,
        episode_length: int = 252,
        transaction_cost_bps: float = 5.0,
        action_mode: str = "discrete",
        grid_steps: int = 4,
        max_weight: float = 1.0,
        reward: str | BaseReward = "log_return",
        reward_kwargs: dict | None = None,
        reward_scale: float = 100.0,
        logit_scale: float = 5.0,
        deterministic_start: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        common = prices.index.intersection(features.index)
        self.prices = prices.loc[common].astype(np.float64)
        self.features = features.loc[common].astype(np.float64)
        self.dates = self.prices.index
        self.asset_names = list(self.prices.columns)
        self.n_assets = len(self.asset_names)

        self.window = int(window)
        self.transaction_cost = float(transaction_cost_bps) / 1e4
        self.action_mode = action_mode
        self.reward_scale = float(reward_scale)
        # SB3 expects continuous actions in [-1, 1]. A raw softmax over that
        # range can never produce a concentrated portfolio (max weight ~0.7 for
        # 3 assets), which would silently hand DQN an unfair advantage. Scaling
        # the logits restores the full simplex to PPO's reachable set.
        self.logit_scale = float(logit_scale)
        self.deterministic_start = deterministic_start

        self.asset_log_returns = (
            np.log(self.prices / self.prices.shift(1)).fillna(0.0).to_numpy()
        )
        self.feature_matrix = self.features.to_numpy()
        self.n_features = self.feature_matrix.shape[1]

        n_steps_available = len(self.prices) - self.window - 1
        if n_steps_available < 10:
            raise ValueError(
                f"Period too short: {len(self.prices)} rows with window={window} "
                f"leaves {n_steps_available} usable steps."
            )
        self.episode_length = int(min(episode_length, n_steps_available))

        self.reward_fn: BaseReward = (
            reward if isinstance(reward, BaseReward)
            else make_reward(reward, **(reward_kwargs or {}))
        )

        # ---- spaces --------------------------------------------------------
        if action_mode == "discrete":
            self.action_grid = simplex_grid(self.n_assets, grid_steps, max_weight)
            self.action_space = spaces.Discrete(len(self.action_grid))
        elif action_mode == "continuous":
            self.action_grid = None
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
            )
        else:
            raise ValueError(f"action_mode must be 'discrete' or 'continuous'")

        obs_dim = self.window * self.n_features + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._np_random_seed = seed
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        max_start = len(self.prices) - self.episode_length - 1
        low = self.window
        if self.deterministic_start or max_start <= low:
            self.start_idx = low
            self.episode_length = min(
                self.episode_length, len(self.prices) - low - 1
            )
        else:
            self.start_idx = int(self.np_random.integers(low, max_start))

        self.t = self.start_idx
        self.step_count = 0
        # Start from equal weights: a neutral prior that does not pre-bake a
        # view into the agent's starting point.
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.reward_fn.reset()

        self.history: dict[str, list] = {
            "date": [], "value": [], "weights": [], "reward": [],
            "port_return": [], "cost": [], "turnover": [],
        }
        return self._observation(), {}

    def step(self, action):
        return self._advance(self._action_to_weights(action))

    def step_with_weights(self, target_weights):
        """Advance using an exact target weight vector, bypassing the action space.

        Rule-based baselines use this. They are not learning agents, so forcing
        them through DQN's coarse simplex grid would distort them -- projecting
        buy-and-hold's drifted weights onto a 1/4-resolution grid silently turns
        it into a rebalancing strategy, which is a different (and weaker)
        benchmark. Giving baselines exact weights makes them *harder* to beat,
        which is the conservative choice when the RL agent is the thing under
        test.
        """
        w = np.asarray(target_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} weights, got {w.shape[0]}")
        w = np.clip(w, 0.0, None)
        total = w.sum()
        w = w / total if total > 1e-12 else np.ones(self.n_assets) / self.n_assets
        return self._advance(w)

    def _advance(self, target: np.ndarray):
        # 1) rebalance, paying costs proportional to turnover
        turnover = float(np.abs(target - self.weights).sum())
        cost = self.transaction_cost * turnover
        self.weights = target

        # 2) the market moves: returns at t+1 are unknown at decision time t
        self.t += 1
        asset_log_ret = self.asset_log_returns[self.t]
        asset_simple_ret = np.exp(asset_log_ret) - 1.0

        gross = float(self.weights @ asset_simple_ret)
        net = gross - cost
        port_log_return = float(np.log1p(max(net, -0.99)))

        # 3) weights drift with realised performance
        grown = self.weights * (1.0 + asset_simple_ret)
        total = grown.sum()
        self.weights = grown / total if total > 1e-12 else self.weights
        self.portfolio_value *= 1.0 + net

        raw_reward = self.reward_fn(port_log_return, 0.0, self.weights, turnover)
        reward = float(raw_reward * self.reward_scale)

        self.step_count += 1
        terminated = False
        truncated = (
            self.step_count >= self.episode_length or self.t >= len(self.prices) - 1
        )

        self.history["date"].append(self.dates[self.t])
        self.history["value"].append(self.portfolio_value)
        self.history["weights"].append(self.weights.copy())
        self.history["reward"].append(reward)
        self.history["port_return"].append(net)
        self.history["cost"].append(cost)
        self.history["turnover"].append(turnover)

        info = {
            "portfolio_value": self.portfolio_value,
            "port_return": net,
            "turnover": turnover,
            "cost": cost,
            "weights": self.weights.copy(),
            "date": self.dates[self.t],
        }
        return self._observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _action_to_weights(self, action) -> np.ndarray:
        if self.action_mode == "discrete":
            return self.action_grid[int(action)].copy()
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a, -1.0, 1.0) * self.logit_scale
        e = np.exp(a - a.max())          # softmax -> long-only, sums to 1
        return e / e.sum()

    def _observation(self) -> np.ndarray:
        lo = self.t - self.window + 1
        win = self.feature_matrix[lo : self.t + 1]
        if win.shape[0] < self.window:    # pad at the very start of the series
            pad = np.repeat(win[:1], self.window - win.shape[0], axis=0)
            win = np.vstack([pad, win])
        obs = np.concatenate([win.reshape(-1), self.weights])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def history_frame(self) -> pd.DataFrame:
        """Per-step record of the last episode, for backtest analytics."""
        if not self.history["date"]:
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "value": self.history["value"],
                "port_return": self.history["port_return"],
                "reward": self.history["reward"],
                "cost": self.history["cost"],
                "turnover": self.history["turnover"],
            },
            index=pd.DatetimeIndex(self.history["date"], name="Date"),
        )
        w = np.vstack(self.history["weights"])
        for i, name in enumerate(self.asset_names):
            df[f"w_{name}"] = w[:, i]
        return df
