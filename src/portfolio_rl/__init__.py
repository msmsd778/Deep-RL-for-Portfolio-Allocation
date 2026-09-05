"""Deep RL for portfolio allocation — SEAI Standard RL Project.

A custom Gymnasium MDP for long-only daily portfolio allocation, plus a
controlled DQN (discrete action space) vs PPO (continuous action space)
comparison with multi-seed statistics, static baselines and regime-wise
out-of-sample evaluation.
"""

from .config import Config, DataConfig, EnvConfig, SplitConfig, TrainConfig
from .env import PortfolioEnv, simplex_grid
from .rewards import REWARD_REGISTRY, make_reward

__version__ = "1.2.0"

__all__ = [
    "Config",
    "DataConfig",
    "EnvConfig",
    "SplitConfig",
    "TrainConfig",
    "PortfolioEnv",
    "simplex_grid",
    "make_reward",
    "REWARD_REGISTRY",
    "__version__",
]
