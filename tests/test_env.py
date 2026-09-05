"""Correctness tests. Run with: pytest -q

These are not decoration. Three of them (`test_no_lookahead_in_features`,
`test_transaction_costs_reduce_return`, `test_scaler_uses_train_moments_only`)
directly guard the failure modes that most often invalidate a financial RL
result, and they are the ones worth showing on a slide if anyone asks how you
know the backtest is honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_rl.config import Config                                  # noqa: E402
from portfolio_rl.data import build_features, make_synthetic_prices, standardise  # noqa: E402
from portfolio_rl.env import PortfolioEnv, simplex_grid                 # noqa: E402
from portfolio_rl.metrics import max_drawdown, sharpe_ratio             # noqa: E402
from portfolio_rl.rewards import make_reward                            # noqa: E402


@pytest.fixture(scope="module")
def data():
    prices = make_synthetic_prices(["A", "B", "C"], "2010-01-01", "2016-12-31", seed=7)
    features = build_features(prices, window=30)
    common = prices.index.intersection(features.index)
    return prices.loc[common], features.loc[common]


@pytest.fixture
def env(data):
    prices, features = data
    return PortfolioEnv(prices, features, window=30, episode_length=120, seed=0)


# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #
def test_simplex_grid_sums_to_one():
    grid = simplex_grid(3, 4)
    assert grid.shape == (15, 3)                     # C(4+2, 2) = 15
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()


def test_simplex_grid_respects_max_weight():
    grid = simplex_grid(3, 4, max_weight=0.5)
    assert (grid.max(axis=1) <= 0.5 + 1e-9).all()


def test_continuous_action_maps_to_simplex(data):
    prices, features = data
    e = PortfolioEnv(prices, features, action_mode="continuous", seed=0)
    for a in [np.array([1.0, -1.0, 0.0]), np.zeros(3), np.array([5.0, 5.0, -5.0])]:
        w = e._action_to_weights(a)
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


def test_logit_scale_allows_concentration(data):
    prices, features = data
    e = PortfolioEnv(prices, features, action_mode="continuous", logit_scale=5.0, seed=0)
    w = e._action_to_weights(np.array([1.0, -1.0, -1.0]))
    # With scale=5 the softmax can reach a genuinely concentrated allocation.
    assert w.max() > 0.9


# --------------------------------------------------------------------------- #
# Environment mechanics
# --------------------------------------------------------------------------- #
def test_observation_shape_and_finiteness(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)


def test_weights_stay_on_simplex_through_episode(env):
    env.reset()
    for _ in range(100):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        w = info["weights"]
        assert np.isclose(w.sum(), 1.0, atol=1e-8)
        assert (w >= -1e-12).all()
        if term or trunc:
            break


def test_episode_terminates(env):
    env.reset()
    steps = 0
    done = False
    while not done and steps < 10_000:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
    assert done
    assert steps <= env.episode_length


def test_deterministic_start_is_reproducible(data):
    prices, features = data
    e1 = PortfolioEnv(prices, features, deterministic_start=True, seed=0)
    e2 = PortfolioEnv(prices, features, deterministic_start=True, seed=999)
    o1, _ = e1.reset()
    o2, _ = e2.reset()
    assert e1.start_idx == e2.start_idx
    assert np.allclose(o1, o2)


def test_random_start_varies(data):
    prices, features = data
    e = PortfolioEnv(prices, features, deterministic_start=False, seed=0)
    starts = set()
    for _ in range(20):
        e.reset()
        starts.add(e.start_idx)
    assert len(starts) > 1


# --------------------------------------------------------------------------- #
# Economic correctness
# --------------------------------------------------------------------------- #
def test_transaction_costs_reduce_return(data):
    """Same action sequence, higher costs => strictly worse terminal value."""
    prices, features = data
    actions = None
    finals = []
    for bps in (0.0, 50.0):
        e = PortfolioEnv(prices, features, transaction_cost_bps=bps,
                         deterministic_start=True, episode_length=200, seed=3)
        e.reset()
        rng = np.random.default_rng(42)          # identical action stream
        done = False
        while not done:
            a = int(rng.integers(e.action_space.n))
            _, _, term, trunc, _ = e.step(a)
            done = term or trunc
        finals.append(e.portfolio_value)
    assert finals[1] < finals[0]


def test_zero_turnover_costs_nothing(data):
    """Holding a constant target weight after the first trade is free."""
    prices, features = data
    e = PortfolioEnv(prices, features, transaction_cost_bps=100.0,
                     deterministic_start=True, episode_length=50, seed=0)
    e.reset()
    costs = []
    for _ in range(30):
        target = e.weights.copy()                       # ask for what we hold
        idx = int(np.argmin(np.abs(e.action_grid - target).sum(axis=1)))
        _, _, term, trunc, info = e.step(idx)
        costs.append(info["cost"])
        if term or trunc:
            break
    assert np.mean(costs[5:]) < np.max(costs) * 0.5     # costs decay to ~0


def test_buy_and_hold_matches_manual_computation(data):
    """A 100%-single-asset, zero-cost policy must reproduce that asset's return."""
    prices, features = data
    e = PortfolioEnv(prices, features, transaction_cost_bps=0.0,
                     deterministic_start=True, episode_length=100, seed=0)
    e.reset()
    asset_idx = 0
    target = np.zeros(e.n_assets)
    target[asset_idx] = 1.0
    idx = int(np.argmin(np.abs(e.action_grid - target).sum(axis=1)))
    start_t = e.t
    done = False
    while not done:
        _, _, term, trunc, _ = e.step(idx)
        done = term or trunc
    manual = prices.iloc[e.t, asset_idx] / prices.iloc[start_t, asset_idx]
    assert np.isclose(e.portfolio_value, manual, rtol=1e-6)


def test_step_with_weights_is_exact(data):
    """Baselines must get the weights they ask for, not a grid approximation."""
    prices, features = data
    e = PortfolioEnv(prices, features, grid_steps=4, deterministic_start=True, seed=0)
    e.reset()
    target = np.array([0.6, 0.3, 0.1])
    e.step_with_weights(target)
    # After one step the weights have drifted, but only by the day's returns.
    drift = np.abs(e.weights - target).max()
    assert drift < 0.05, "weights should differ from the target only by daily drift"
    # The nearest grid point would have been 0.5/0.25/0.25 -> far coarser.
    nearest = e.action_grid[np.argmin(np.abs(e.action_grid - target).sum(axis=1))]
    assert np.abs(nearest - target).max() > 0.05


def test_buy_and_hold_differs_from_equal_weight(data):
    """The two must not collapse onto each other (regression test)."""
    from portfolio_rl.baselines import BuyAndHold, EqualWeight, run_policy

    prices, features = data
    results = {}
    for policy in (BuyAndHold(), EqualWeight(3)):
        e = PortfolioEnv(prices, features, transaction_cost_bps=20.0,
                         deterministic_start=True, episode_length=400, seed=0)
        results[policy.name] = run_policy(e, policy)

    bh, ew = results["buy_and_hold"], results["equal_weight"]
    # Buy-and-hold trades once; equal-weight rebalances every single day.
    assert bh["turnover"].sum() < ew["turnover"].sum()
    assert not np.isclose(bh["value"].iloc[-1], ew["value"].iloc[-1], rtol=1e-6)


def test_buy_and_hold_turnover_is_near_zero_after_first_trade(data):
    from portfolio_rl.baselines import BuyAndHold, run_policy

    prices, features = data
    e = PortfolioEnv(prices, features, deterministic_start=True,
                     episode_length=200, seed=0)
    hist = run_policy(e, BuyAndHold())
    assert hist["turnover"].iloc[1:].max() < 1e-9


# --------------------------------------------------------------------------- #
# Split hygiene (regression: an in-sample period must never look out-of-sample)
# --------------------------------------------------------------------------- #
def _splits_cfg(train, stress):
    from portfolio_rl.config import SplitConfig

    return SplitConfig(
        train=train,
        val=["2016-01-01", "2016-12-31"],
        test_regimes={},
        stress_regime=stress,
    )


def test_overlapping_eval_period_is_flagged(data, caplog):
    from portfolio_rl.data import make_splits

    prices, features = data
    cfg = _splits_cfg(["2010-01-01", "2015-12-31"],
                      {"inside_train": ["2011-01-01", "2011-12-31"]})
    with caplog.at_level("ERROR"):
        splits = make_splits(prices, features, cfg)
    assert splits["inside_train"]["in_sample"] is True
    assert "IN-SAMPLE EVALUATION PERIOD DETECTED" in caplog.text


def test_non_overlapping_period_is_not_flagged(data):
    from portfolio_rl.data import make_splits

    prices, features = data
    cfg = _splits_cfg(["2012-01-01", "2015-12-31"],
                      {"before_train": ["2010-01-01", "2010-12-31"]})
    splits = make_splits(prices, features, cfg)
    assert splits["before_train"]["in_sample"] is False


def test_strict_splits_raises_on_overlap(data):
    from portfolio_rl.data import make_splits

    prices, features = data
    cfg = _splits_cfg(["2010-01-01", "2015-12-31"],
                      {"inside_train": ["2011-01-01", "2011-12-31"]})
    with pytest.raises(ValueError, match="overlap the training window"):
        make_splits(prices, features, cfg, strict=True)


def test_shipped_default_config_has_no_overlap():
    """The bundled default must not evaluate on its own training data."""
    from portfolio_rl.config import Config
    from portfolio_rl.data import _overlaps

    cfg = Config.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    t0, t1 = cfg.splits.train
    for name, (a, b) in {**cfg.splits.test_regimes, **cfg.splits.stress_regime}.items():
        assert not _overlaps(a, b, t0, t1), f"period '{name}' overlaps train"
    assert not _overlaps(*cfg.splits.val, t0, t1), "val overlaps train"


# --------------------------------------------------------------------------- #
# Cache provenance (regression: a fallback must never masquerade as real data)
# --------------------------------------------------------------------------- #
def test_synthetic_fallback_does_not_write_the_yahoo_cache(tmp_path, monkeypatch):
    from portfolio_rl import data as D

    monkeypatch.setattr(D, "_download_yahoo", lambda *a, **k: None)
    df = D.load_prices(["A", "B", "C"], "2010-01-01", "2014-12-31",
                       cache_dir=tmp_path, synthetic=False)
    assert len(df) > 300
    names = {p.name for p in tmp_path.glob("prices_*.csv")}
    assert any("synthetic" in n for n in names)
    assert not any("yahoo" in n for n in names), (
        "a failed download wrote the synthetic series under the yahoo cache name"
    )


def test_require_real_data_raises_instead_of_falling_back(tmp_path, monkeypatch):
    from portfolio_rl import data as D

    monkeypatch.setattr(D, "_download_yahoo", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="require_real"):
        D.load_prices(["A", "B", "C"], "2010-01-01", "2014-12-31",
                      cache_dir=tmp_path, synthetic=False, require_real=True)


def test_poisoned_legacy_cache_is_rejected(tmp_path, caplog):
    """A yahoo-tagged file with no sidecar and a synthetic index must be refused."""
    from portfolio_rl import data as D

    synth = D.make_synthetic_prices(["A", "B", "C"], "2010-01-01", "2014-12-31", seed=1)
    csv, meta = D._cache_paths(tmp_path, "yahoo", ["A", "B", "C"],
                               "2010-01-01", "2014-12-31")
    synth.to_csv(csv)                      # no sidecar: simulates the old bug
    assert not meta.exists()
    with caplog.at_level("ERROR"):
        got = D._read_cache(csv, meta, "yahoo", ["A", "B", "C"])
    assert got is None
    assert "POISONED CACHE" in caplog.text


def test_cache_sidecar_records_provenance(tmp_path):
    from portfolio_rl import data as D

    D.load_prices(["A", "B", "C"], "2010-01-01", "2014-12-31",
                  cache_dir=tmp_path, synthetic=True)
    metas = list(tmp_path.glob("prices_*.meta.json"))
    assert len(metas) == 1
    payload = json.loads(metas[0].read_text())
    assert payload["source"] == "synthetic"
    assert payload["rows"] > 300


def test_looks_synthetic_flags_business_day_indices():
    from portfolio_rl import data as D

    synth = D.make_synthetic_prices(["A", "B", "C"], "2010-01-01", "2014-12-31", seed=2)
    assert D.looks_synthetic(synth)
    # Drop ~10 days a year to imitate an exchange holiday calendar.
    holidays = synth.index[::26]
    realish = synth.drop(index=holidays)
    assert not D.looks_synthetic(realish)


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def test_no_lookahead_in_features(data):
    """Truncating the price series must not change earlier feature rows.

    If any feature peeked at the future, the values computed on the full series
    would differ from those computed on a truncated one.
    """
    prices, _ = data
    cut = len(prices) // 2
    full = build_features(prices, window=30)
    partial = build_features(prices.iloc[:cut], window=30)
    common = full.index.intersection(partial.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        full.loc[common], partial.loc[common], check_exact=False, atol=1e-10
    )


def test_observation_excludes_current_return(data):
    """The obs at time t must not contain the return the agent is about to earn."""
    prices, features = data
    e = PortfolioEnv(prices, features, window=5, deterministic_start=True, seed=0)
    obs, _ = e.reset()
    next_ret = e.asset_log_returns[e.t + 1]
    # No element of the observation may equal the unseen next-step return.
    for r in next_ret:
        assert not np.any(np.isclose(obs, r, atol=1e-12))


def test_scaler_uses_train_moments_only(data):
    _, features = data
    train = features.iloc[:800]
    test = features.iloc[800:]
    scaled_train, scaled_other, moments = standardise(train, {"test": test})
    assert np.allclose(scaled_train.mean().to_numpy(), 0.0, atol=1e-8)
    # The test block is NOT re-centred: that would be leakage.
    assert not np.allclose(scaled_other["test"].mean().to_numpy(), 0.0, atol=1e-3)
    pd.testing.assert_series_equal(moments["mean"], train.mean())


# --------------------------------------------------------------------------- #
# Rewards & metrics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["log_return", "differential_sharpe", "cvar_penalised"])
def test_rewards_are_finite_and_resettable(name):
    fn = make_reward(name)
    rng = np.random.default_rng(0)
    w = np.array([0.5, 0.3, 0.2])
    for _ in range(200):
        r = fn(float(rng.normal(0, 0.01)), 0.0, w)
        assert np.isfinite(r)
    fn.reset()
    assert np.isfinite(fn(0.001, 0.0, w))


def test_cvar_penalty_punishes_volatile_streams():
    calm = make_reward("cvar_penalised", lam=5.0)
    wild = make_reward("cvar_penalised", lam=5.0)
    rng = np.random.default_rng(1)
    w = np.ones(3) / 3
    calm_total = sum(calm(float(rng.normal(0, 0.001)), 0.0, w) for _ in range(120))
    rng = np.random.default_rng(1)
    wild_total = sum(wild(float(rng.normal(0, 0.02)), 0.0, w) for _ in range(120))
    assert wild_total < calm_total


def test_metrics_on_known_series():
    r = np.full(252, 0.001)                       # constant positive return
    assert sharpe_ratio(r) > 1e6 or np.isinf(sharpe_ratio(r)) or sharpe_ratio(r) == 0.0
    assert np.isclose(max_drawdown(r), 0.0)
    down = np.array([0.1, -0.5, 0.1])
    assert max_drawdown(down) < -0.4


def test_config_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "c.yaml"
    cfg.save(p)
    again = Config.from_yaml(p)
    assert again.env.window == cfg.env.window
    assert again.train.seeds == cfg.train.seeds


# --------------------------------------------------------------------------- #
# Turnover penalty
# --------------------------------------------------------------------------- #
def test_turnover_penalty_punishes_churning():
    fn = make_reward("turnover_penalised", lam=0.001)
    w = np.ones(3) / 3
    held = fn(0.002, 0.0, w, 0.0)
    churned = fn(0.002, 0.0, w, 2.0)
    assert held > churned
    assert np.isclose(held - churned, 0.002)


def test_turnover_reaches_the_reward_function(data):
    """Regression: env must pass realised turnover, not a hard-coded zero."""
    prices, features = data
    seen = []

    class Probe(make_reward("log_return").__class__):
        def __call__(self, r, c, w, turnover=0.0):
            seen.append(turnover)
            return r

    e = PortfolioEnv(prices, features, deterministic_start=True,
                     episode_length=20, reward=Probe(), seed=0)
    e.reset()
    # Force a large rebalance, then request the same weights again.
    far = int(np.argmax(np.abs(e.action_grid - e.weights).sum(axis=1)))
    e.step(far)
    assert seen[0] > 0.5, "first rebalance should register real turnover"


def test_all_registered_rewards_accept_turnover():
    from portfolio_rl.rewards import REWARD_REGISTRY

    w = np.ones(3) / 3
    for name in REWARD_REGISTRY:
        fn = make_reward(name)
        assert np.isfinite(fn(0.001, 0.0, w, 0.4)), f"{name} rejected turnover"
