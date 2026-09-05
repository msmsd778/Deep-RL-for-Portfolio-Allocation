"""Reward functions for the portfolio environment.

Three variants, deliberately ordered by how much risk they encode:

  log_return           -- pure P&L. Fast to learn, but risk-blind.
  differential_sharpe  -- online Sharpe increment (Moody & Saffell, 1998).
  cvar_penalised       -- P&L minus a penalty on the rolling left tail.

Each is a small stateful object because two of the three need running
statistics that must be reset at episode boundaries.

Reward-hacking notes (the exam will ask):
  * `log_return` has no volatility term, so the return-maximising policy is a
    levered bet on whichever asset had the highest realised drift in the
    training window. It will look excellent in-sample and fail out-of-sample.
  * `differential_sharpe` divides by a running standard deviation. Early in an
    episode that denominator is tiny, so the first few steps produce enormous
    rewards; we warm it up and clip to stop the agent from learning "make the
    first trade look good, then coast".
  * `cvar_penalised` can be gamed in the opposite direction: if lambda is large
    the optimal policy is 100% bonds forever, which scores well on the reward
    but is not a portfolio-management strategy. We report the realised weights
    precisely so this degenerate solution is visible rather than hidden behind
    a good-looking reward curve.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class BaseReward:
    """Interface: `reset()` at episode start, `__call__` once per step."""

    name = "base"

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def __call__(self, port_log_return: float, cost: float, weights: np.ndarray,
                 turnover: float = 0.0) -> float:
        raise NotImplementedError


class LogReturnReward(BaseReward):
    """r_t = log(1 + R_t) - cost_t. Risk-neutral baseline."""

    name = "log_return"

    def __call__(self, port_log_return: float, cost: float, weights: np.ndarray,
                 turnover: float = 0.0) -> float:
        return float(port_log_return - cost)


class DifferentialSharpeReward(BaseReward):
    """Moody & Saffell's differential Sharpe ratio.

    Maintains exponentially weighted first and second moments (A, B) of the
    per-step return and returns the instantaneous increment of the Sharpe ratio
    with respect to the decay rate eta.
    """

    name = "differential_sharpe"

    def __init__(self, eta: float = 0.02, warmup: int = 20, clip: float = 5.0):
        self.eta, self.warmup, self.clip = eta, warmup, clip
        self.reset()

    def reset(self) -> None:
        self.A = 0.0
        self.B = 1e-6
        self.t = 0

    def __call__(self, port_log_return: float, cost: float, weights: np.ndarray,
                 turnover: float = 0.0) -> float:
        r = float(port_log_return - cost)
        self.t += 1
        dA, dB = r - self.A, r**2 - self.B

        denom = (self.B - self.A**2) ** 1.5
        if self.t <= self.warmup or denom <= 1e-12:
            # During warm-up we fall back to raw P&L: the ratio is numerically
            # meaningless until the moments have seen some data.
            dsr = r
        else:
            dsr = (self.B * dA - 0.5 * self.A * dB) / denom

        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(dsr, -self.clip, self.clip))


class CVaRPenalisedReward(BaseReward):
    """r_t = log(1+R_t) - cost_t - lambda * CVaR_alpha(recent returns).

    The CVaR term is computed on a trailing window of realised portfolio
    returns, so it is a *backward*-looking penalty. That is intentional: it is
    the same quantity a risk desk would monitor intraday, and it keeps the MDP
    causal.
    """

    name = "cvar_penalised"

    def __init__(self, lam: float = 2.0, alpha: float = 0.05, window: int = 60):
        self.lam, self.alpha, self.window = lam, alpha, window
        self.reset()

    def reset(self) -> None:
        self.hist: deque[float] = deque(maxlen=self.window)

    def __call__(self, port_log_return: float, cost: float, weights: np.ndarray,
                 turnover: float = 0.0) -> float:
        r = float(port_log_return - cost)
        self.hist.append(r)
        if len(self.hist) < 20:
            return r
        arr = np.asarray(self.hist)
        var = np.quantile(arr, self.alpha)
        tail = arr[arr <= var]
        cvar = float(tail.mean()) if tail.size else float(var)
        # cvar is negative in the loss tail; subtracting a negative would reward
        # losses, so we penalise its magnitude.
        return r - self.lam * abs(min(cvar, 0.0))


class TurnoverPenalisedReward(BaseReward):
    """r_t = log(1+R_t) - lambda * turnover_t.

    Motivation: with only the 5 bps execution cost in the loop, both DQN and
    PPO converge on "bang-bang" policies that flip between simplex corners
    almost every day (150-180x annual turnover, i.e. the whole portfolio
    reshuffled most days). Nothing in the reward makes holding a position
    worth anything, so churning is free up to the spread.

    `lam` is an ARTIFICIAL extra cost per unit of turnover, expressed in return
    units. It is deliberately far larger than the real 5 bps spread: the agent
    must be paid to hold, not merely charged to trade.

    Measured dose-response (DQN, 1 seed, 30k steps, synthetic data):

        lam      annual turnover     mean Sharpe
        0.000        173.8              -0.29
        0.002        148.4              -0.42
        0.005         79.6              -0.69
        0.020         22.7              +0.13     <- default
        0.050         11.4              -0.14
        0.100          3.4              -0.33

    Turnover falls monotonically; Sharpe is single-peaked. Too little penalty
    and the agent churns; too much and it freezes into a near-static portfolio
    and stops allocating at all. Re-fit this on YOUR data before quoting it --
    the optimum depends on the cost assumption and the asset universe.

    Interpreting the ablation: sweep lam and plot realised annual turnover and
    out-of-sample Sharpe against it. If Sharpe rises as turnover falls, the
    original reward -- not the algorithm -- was the binding constraint. That is
    a result about reward design, which is exactly what the failure-analysis
    section is for.

    Reward-hacking note: at large lam the optimal policy is to trade once and
    then never again, which scores well and is not portfolio management. Report
    realised turnover alongside Sharpe so this shows up rather than hiding.
    """

    name = "turnover_penalised"

    def __init__(self, lam: float = 0.02):
        self.lam = float(lam)

    def __call__(self, port_log_return: float, cost: float, weights: np.ndarray,
                 turnover: float = 0.0) -> float:
        return float(port_log_return - cost - self.lam * abs(turnover))


REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    LogReturnReward.name: LogReturnReward,
    DifferentialSharpeReward.name: DifferentialSharpeReward,
    CVaRPenalisedReward.name: CVaRPenalisedReward,
    TurnoverPenalisedReward.name: TurnoverPenalisedReward,
}


def make_reward(name: str, **kwargs) -> BaseReward:
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward '{name}'. Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name](**kwargs)
