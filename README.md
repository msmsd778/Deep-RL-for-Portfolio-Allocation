# Deep RL for Portfolio Allocation — DQN vs PPO

**Symbolic and Evolutionary Artificial Intelligence — Standard RL Project (SRLP)**
University of Pisa, MSc Artificial Intelligence and Data Engineering

---

> **New to finance or RL?** Read `GUIDE.md` first. It explains every concept in
> this project from zero — portfolios, Sharpe ratios, MDPs, DQN, PPO — with
> worked numeric examples, plus a section on how to read your own results.

## 1. What this project is, in one paragraph

Portfolio allocation is a sequential decision problem under uncertainty: every
day you choose how to split capital across assets, you pay to change your mind,
and you only learn whether you were right afterwards. That is an MDP. This
project builds a custom Gymnasium environment for long-only daily allocation
across an equity / bond / gold sleeve, and runs a **controlled comparison of two
structurally different RL algorithms** on it:

| | **DQN** | **PPO** |
|---|---|---|
| Family | Value-based (off-policy) | Policy-gradient (on-policy) |
| Action space | **Discrete** — 15 points on a weight simplex grid | **Continuous** — softmax over the full simplex |
| Learns | Q(s,a) | π(a\|s) and V(s) |
| Sample reuse | Replay buffer | On-policy rollouts only |

The discrete-vs-continuous split is the *point* of the comparison, not an
accident: a value-based method structurally requires a finite action set, and
the cost of that discretisation is exactly what we measure.

Everything is evaluated out-of-sample across **named market regimes** (GFC,
COVID crash, 2020–21 recovery, 2022 inflation shock, calm 2023–24) against five
non-learning baselines, with results averaged over multiple seeds and reported
with confidence intervals and significance tests.

> **A word on honesty of results.** RL does not reliably beat a static 60/30/10
> portfolio on daily data with realistic transaction costs. This project is not
> built to manufacture a win. It is built to measure the comparison correctly
> and to report what actually happens — including the cases where both agents
> lose to a fixed-weight benchmark. A rigorous negative result is a much
> stronger submission than an impressive-looking one with look-ahead bias in it,
> and it is far easier to defend in the oral.

---

## 2. Quick start

**Requires Python 3.10–3.13.** Not 3.8 or 3.9: `gymnasium` 1.x, recent
`stable-baselines3` and this codebase all use PEP 585 builtin generics
(`list[str]`, `type[Thread]`). On 3.8 pip will install *something* for every
requirement and then things fail at import time with
`TypeError: 'type' object is not subscriptable`. Check with `python --version`
before anything else.

```bash
# 1. clone / unzip, then create an environment
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. verify the installation (should take <5 seconds)
pytest -q

# 3. smoke test: 2 seeds, 8k steps, offline synthetic data (~1 minute)
python scripts/run_experiments.py --quick --synthetic --results-dir results_smoke

# 4. the real run (~35 minutes on a laptop CPU)
python scripts/run_experiments.py
```

Results land in `results/`:

```
results/
├── config_used.yaml          exact configuration of this run (reproducibility)
├── training_meta.json        timings, latencies, learning curves per run
├── models/                   saved DQN_seed0.zip, PPO_seed0.zip, ...
├── tables/
│   ├── raw_results.csv       one row per (strategy, seed, period)
│   ├── aggregated_results.csv mean ± 95% CI across seeds
│   ├── significance_tests.csv Welch t-test, Mann-Whitney, bootstrap CI
│   └── baseline_win_rates.csv fraction of seeds beating 60/30/10
├── curves/                   per-day equity/weights for plotting in LaTeX
└── figures/                  PNG + PDF, ready to drop into Beamer
```

### Useful flags

| Flag | Effect |
|---|---|
| `--quick` | 2 seeds, 8k steps. Use while developing. |
| `--synthetic` | Never touch the network; use the built-in regime-switching generator. |
| `--reward cvar_penalised` | Swap the reward function (see §5). |
| `--timesteps 300000` | Longer training. |
| `--seeds 0 1 2 3 4 5 6 7` | More seeds → tighter confidence intervals. |
| `--algos PPO` | Train only one algorithm. |
| `--skip-training` | Re-evaluate and re-plot existing models without retraining. |

### Ablations

`scripts/run_ablations.py` runs a grid of variants and collects the
out-of-sample means into one CSV plus one figure.

```bash
# the headline ablation: does penalising turnover fix the chattering?
python scripts/run_ablations.py --which turnover

# reward-function comparison, and transaction-cost sensitivity
python scripts/run_ablations.py --which rewards
python scripts/run_ablations.py --which costs
```

Add `--algos DQN --seeds 0 1 2 --timesteps 60000` to cut the cost while
developing. Results land in `results_ablation/`, with
`figures/10_turnover_sweep.png` ready for the slide.

---

## 3. Data

Default universe: **SPY** (US equity), **IEF** (7–10y Treasuries), **GLD** (gold)
— a minimal sleeve that still contains a real diversification story, since the
equity/bond correlation flips sign in a crisis and gold behaves as a safe haven.

Prices are downloaded once via `yfinance` and cached to `data/`, so every later
run is byte-identical. **If `yfinance` is unavailable or the network is blocked,
the pipeline automatically falls back to a regime-switching synthetic
generator** (`data.make_synthetic_prices`) — a hidden 3-state Markov chain
(calm / crisis / inflation) driving a multivariate GBM with a *different
correlation matrix per regime*. This exists so the whole project is runnable and
reviewable offline, and so a reviewer can reproduce results without depending on
a third-party API. Synthetic runs are clearly labelled in the cache filename.

> ⚠️ If you present synthetic results, **say so explicitly on the slide.**
> Presenting simulated data as market data is the one mistake that would sink
> the project.
>
> ⚠️ **Regime labels are meaningless under `--synthetic`.** The split names
> (`gfc`, `covid_crash`, ...) are just *date ranges*. On real data those ranges
> contain the events they are named after; on synthetic data the hidden Markov
> chain puts its crises wherever it likes, so the window labelled `gfc` may be a
> perfectly calm stretch. This is why the bundled example run in
> `results_example/` shows an absurd Sharpe of ~6.8 on `gfc` — the agent was not
> surviving a crisis, it was riding a quiet synthetic bull market. Treat any
> implausibly good number as a bug hypothesis first. Only the real-data run
> supports regime-wise conclusions.

### Bring your own data (no yfinance needed)

If `yfinance` will not install, export daily adjusted closes to a CSV shaped
like this and skip the network entirely:

```
Date,SPY,IEF,GLD
2005-01-03,86.35,52.10,42.74
...
```

```bash
python scripts/run_experiments.py --prices-csv data/my_prices.csv
```

Stooq, Nasdaq Data Link, Investing.com and most broker exports are one `pandas`
call away from this format. This is the **recommended** path for real-data runs
on a locked-down machine.

### Data provenance — read this before trusting any number

Every cached price file now carries a `.meta.json` sidecar recording its source
(`yahoo` or `synthetic`), row count, date range and creation time. Three
guarantees follow:

1. **A failed download can never poison the real-data cache.** Simulated series
   are written under their own `prices_synthetic_*.csv` filename. Earlier
   versions wrote the fallback under the `yahoo` name, so one transient network
   failure would silently turn every subsequent run into a simulation.
2. **Legacy caches with no sidecar are checked heuristically.** The synthetic
   generator uses `pd.bdate_range`, which includes market holidays; a real
   exchange calendar does not. A `yahoo`-tagged file whose index has no holiday
   gaps is refused with a `POISONED CACHE DETECTED` error naming the file.
3. **`--require-real-data` aborts rather than falling back.** Use it for the run
   whose numbers you intend to present:

   ```bash
   python scripts/run_experiments.py --require-real-data
   ```

The run header also prints a `DATA PROVENANCE` warning whenever the loaded
series looks generated, regardless of where it came from.

**Quick manual check:** 2005–2024 has ~5030 US trading days. If the log says
`Prices: 5217 rows`, that is the business-day count including holidays — the
data is synthetic.

### Features (all strictly causal)

Per asset: yesterday's log return, momentum over 5/21/63 days, 21-day realised
volatility (annualised), and drawdown from a rolling high. Every column is
`.shift(1)`-ed so the observation at time *t* contains only information
available at *t−1*. The environment then hands the agent a `window`-day stack of
these features plus its current weights.

---

## 4. The MDP

| Component | Definition |
|---|---|
| **State** `s_t` | `window × n_features` standardised causal features (flattened) **+** current post-drift weights. Default: 30 × 18 + 3 = **543 dims**. |
| **Action** `a_t` | *Discrete*: index into `simplex_grid(3, 4)` → **15 actions**, each a weight vector on a 1/4 grid. *Continuous*: `Box(-1,1)^3` → scaled softmax → any point on the simplex. |
| **Transition** | Rebalance to target (pay `5 bps × turnover`), market moves, weights drift with realised returns. |
| **Reward** `r_t` | `100 × f(portfolio log-return, weights)`, `f` ∈ {log-return, differential Sharpe, CVaR-penalised}. |
| **Termination** | After `episode_length` (252) steps or at the end of the series. No bankruptcy state: long-only, unlevered portfolios cannot reach zero. |
| **Discount** `γ` | 0.99 — an effective horizon of ~100 trading days, which matches the medium-term horizon a tactical allocator actually cares about. |

### On stochasticity (expect this question)

The price path is *deterministic* — it is history. Randomness enters through the
**random episode start index** at every `reset()`, which turns one price series
into a distribution over overlapping 1-year episodes. This prevents the agent
from memorising a single trajectory without pretending we can resample the
market. At evaluation time we set `deterministic_start=True` and walk each
period once, start to finish: that is what a backtest is.

### Why 15 actions?

`|A| = C(steps + n − 1, n − 1)`. With 3 assets and `grid_steps=4` that is 15.
DQN's output layer has one unit per action and its sample complexity grows with
`|A|`, so this is a deliberate trade-off: `grid_steps=10` would give 66 actions
and a much finer allocation grid, at the cost of far slower learning. **This is
precisely the discretisation cost the DQN-vs-PPO comparison is designed to
expose** — try `--grid-steps` variations as an ablation if you have time.

---

## 5. Reward design (and how each one can be gamed)

| Reward | Formula | Failure mode |
|---|---|---|
| `log_return` | `log(1+R_t) − cost_t` | **Risk-blind.** The optimiser's answer is a concentrated bet on whatever drifted up in-sample. Looks great in training, breaks out-of-sample. |
| `turnover_penalised` | `log(1+R_t) − λ·turnover_t` | At large λ the optimal policy is to trade once and never again — a static portfolio that scores well and is not allocation. Report realised turnover alongside Sharpe. |
| `differential_sharpe` | Online Sharpe increment (Moody & Saffell, 1998) | Divides by a running σ that is ~0 early in an episode → enormous early rewards. Mitigated with a warm-up and clipping. |
| `cvar_penalised` | `log(1+R_t) − λ·\|CVaR₅%(recent)\|` | If λ is large, the optimal policy is **100% bonds forever** — a degenerate solution that scores well but is not portfolio management. |

We report realised average weights per asset in every results table specifically
so degenerate solutions are *visible* rather than hidden behind a nice-looking
reward curve. This is the reward-hacking analysis the guidelines ask for, and
demonstrating you found the failure mode is worth more than avoiding it.

---

## 6. Baselines — "compared to what?"

| Baseline | Why it is here |
|---|---|
| `static_60_30_10` | The industry-standard multi-asset benchmark. |
| `equal_weight` | The 1/N portfolio: famously hard to beat (DeMiguel et al., 2009). |
| `buy_and_hold` | Trades once, then never again. Isolates the value of *trading at all*. |
| `inverse_vol` | Risk-parity-lite. A non-trivial rule-based competitor. |
| `random` | **Sanity floor.** If RL cannot beat uniform-random actions, the learning pipeline is broken, not the market. |

Baselines run through the *same* environment, pay the *same* transaction costs,
on the *same* days. They step via `env.step_with_weights()` (exact weights)
rather than the coarse action grid — this makes them *harder* to beat, which is
the conservative choice when the RL agent is the thing under test.

---

## 7. Evaluation protocol

1. **Time-ordered splits only.** Train 2009-07–2015, validate 2016–2019, test
   on five named out-of-sample regimes. Training deliberately starts *after* the
   GFC so the 2007–2009 window is a true "never seen a crisis like this" test.
   **Any evaluation period overlapping the training window is detected, logged
   as `IN-SAMPLE EVALUATION PERIOD DETECTED`, tagged `in_sample=True` in every
   results table and printed in a separate block.** Use `--strict-splits` to
   make it an error. `configs/long_train.yaml` offers the opposite trade-off:
   training from 2005 for more data, with the GFC regime removed rather than
   left in to be misread.
2. **Feature standardisation uses training moments only.** Fitting a scaler on
   the full sample is one of the most common silent leaks in financial ML.
3. **Multiple seeds.** 5 by default. Metrics reported as mean ± 95% t-based CI.
4. **Significance testing.** Welch's t-test (not Student's — no reason to assume
   equal variance; DQN's spread across seeds is usually much wider), plus
   Mann-Whitney U as a non-parametric cross-check, plus a bootstrap CI on the
   difference in means, plus Cohen's *d* for effect size.
5. **Win rate over baseline.** The fraction of seeds beating 60/30/10 — a more
   honest headline than a mean that one lucky seed can carry.
6. **Efficiency.** Wall-clock training time, throughput, and per-action
   inference latency (µs), because "AI & Data Engineers must go beyond
   qualitative observation".

### Leakage guards are tested, not assumed

Three tests in `tests/test_env.py` exist purely to prove the backtest is honest:

- `test_no_lookahead_in_features` — truncating the price series must not change
  earlier feature rows. If any feature peeked forward, it would.
- `test_observation_excludes_current_return` — the observation at *t* must not
  contain the return the agent is about to earn.
- `test_scaler_uses_train_moments_only` — the test block must *not* come out
  re-centred at zero.

If anyone asks "how do you know there's no look-ahead bias?", the answer is a
slide with these three tests on it.

---

## 8. Project layout

```
├── README.md                    setup, protocol, assessment mapping
├── GUIDE.md                     ★ concepts from zero: finance + RL explained
├── requirements.txt
├── configs/default.yaml         every tunable number, in one auditable place
├── src/portfolio_rl/
│   ├── config.py                dataclass config + YAML (de)serialisation
│   ├── data.py                  loading, synthetic generator, features, splits
│   ├── env.py                   ★ PortfolioEnv — the MDP
│   ├── rewards.py               three reward functions + hacking notes
│   ├── baselines.py             five non-learning policies
│   ├── metrics.py               Sharpe, Sortino, MDD, Calmar, VaR, CVaR
│   ├── train.py                 multi-seed SB3 training + timing/latency
│   ├── evaluate.py              rollouts, aggregation, significance tests
│   └── plots.py                 nine slide-ready figures
├── scripts/run_experiments.py   the one command that does everything
├── tests/test_env.py            24 tests, including three leakage guards
└── results_example/             a small bundled demo run (3 seeds, 30k steps,
                                 SYNTHETIC data) so you can see what the
                                 outputs look like before running anything.
                                 Not a result — see the warning in §3.
```

**Start reading at `env.py`.** Everything else is scaffolding around that MDP.

---

## 9. Mapping to the assessment criteria

| Guideline requirement | Where it lives |
|---|---|
| ≥ 2 distinct RL techniques | DQN vs PPO — `train.py` |
| State space: dimensionality, discrete/continuous, normalisation | §4 above; `data.standardise`, `env._observation` |
| Action space: impact of type on algorithm choice | §4; `env.simplex_grid` vs scaled softmax |
| Reward engineering: sparse/dense, reward hacking | §5; `rewards.py` docstring |
| Stochasticity | §4 "On stochasticity"; `env.reset` |
| Algorithm rationale (value vs policy-gradient) | §1 table; `train.py` header |
| Episodes / convergence criteria | `configs/default.yaml`, learning-curve figure |
| Exploration–exploitation | DQN ε-schedule; PPO `ent_coef` — both in the config |
| Replay buffer: sizing, sampling, sample efficiency | `train.dqn.buffer_size`, discussed vs PPO's on-policy rollouts |
| Architecture: topology, activations, target networks | `policy_kwargs.net_arch`, `target_update_interval`, `tau: 1.0` |
| Statistical significance, multiple seeds, CIs, t-test | `evaluate.significance_tests` |
| Training time, sample complexity, inference latency | `train.measure_inference_latency`, figure 08 |
| Generalisation to environment variations | Five out-of-sample regimes + the pre-training GFC window |
| Failure analysis | §10 below |
| Cite + substantially extend any forked code | §11 |

---

## 10. Failure analysis — what to actually say

Do not skip this section in the presentation; it is explicitly graded, and it is
where a strong project separates itself.

- **Non-stationarity.** The training window (2005–2015) contains one deflationary
  crisis and a long QE bull market. 2022 was an inflationary regime where stocks
  *and* bonds fell together — a correlation structure the agent never saw. Expect
  both algorithms to degrade badly there. This is the concrete instance of the
  distribution-shift problem.
- **Turnover sensitivity.** Re-run with `transaction_cost_bps: 20` and watch
  performance collapse. Strategies that look profitable at 0 bps and die at
  20 bps were never real. Show both.
- **Bang-bang chattering (expect to observe this).** Under `log_return` the
  reward contains no turnover penalty beyond the 5 bps cost, so both agents tend
  to converge on flipping between simplex *corners* almost daily -- ~250-280x
  annual turnover, i.e. the whole portfolio reshuffled every day. Look at
  `05_weights_*`: near-vertical colour stripes rather than smooth blocks. That
  is ~1.3%/yr of pure cost drag at 5 bps, and it is a genuine finding worth a
  slide: the reward was under-specified, not the algorithm broken. The natural
  ablation is to add an explicit turnover penalty and show turnover fall while
  Sharpe rises.
- **DQN's discretisation ceiling.** With 15 actions the finest expressible tilt
  is 25%. DQN cannot make a 5% adjustment even when that is optimal.
- **PPO's variance.** On-policy learning with one price path is sample-hungry;
  expect wide seed spread and check whether the CI actually excludes zero.
- **Seed spread is the headline.** If the between-seed CI is wider than the
  DQN-vs-PPO gap, the honest conclusion is *"we cannot distinguish them at this
  sample size"* — say that, don't bury it.
- **Overlapping episodes.** Random starts on one series create heavily
  correlated episodes, so effective sample size is far below the nominal step
  count. This is a real limitation of backtest-style RL, worth one slide.

---

## 11. Attribution

- **Stable-Baselines3** (Raffin et al., JMLR 2021) provides the DQN and PPO
  implementations — https://github.com/DLR-RM/stable-baselines3
- **Gymnasium** (Farama Foundation) provides the environment API.

Per the open-source policy in the guidelines, these are cited and the
contribution here is everything *around* them: the custom MDP, the reward
designs and their failure analysis, the baseline suite, the leakage-guard tests,
the regime-wise evaluation protocol and the multi-seed statistics.
Reimplementing DQN's optimiser loop from scratch would have added risk without
adding insight; the added value is in the environment and the experimental
design, which is where the interesting choices are.

**Key references for the report:** Mnih et al. (2015) *DQN*; Schulman et al.
(2017) *PPO*; Moody & Saffell (1998) *differential Sharpe*; DeMiguel, Garlappi &
Uppal (2009) *1/N*; Jiang, Xu & Liang (2017) *RL portfolio management*; Rockafellar
& Uryasev (2000) *CVaR*.

---

## 12. Suggested 20-slide structure

| # | Slide |
|---|---|
| 1 | Title, group members |
| 2 | Motivation: allocation as sequential decision-making |
| 3 | Problem statement & research question |
| 4 | Data: universe, period, regime timeline (fig 01) |
| 5 | Feature engineering & causality |
| 6 | MDP: state space |
| 7 | MDP: action space — discrete grid vs continuous simplex |
| 8 | MDP: reward design |
| 9 | Reward hacking: three failure modes we found |
| 10 | Transition dynamics, costs, stochasticity |
| 11 | Algorithms: why DQN and why PPO |
| 12 | Architectures & hyperparameters |
| 13 | Training dynamics / learning curves (fig 02) |
| 14 | Experimental protocol: splits, seeds, leakage guards |
| 15 | Baselines |
| 16 | Results: out-of-sample equity curves (fig 03) |
| 17 | Results: metrics with CIs (fig 04) + significance (fig 09) |
| 18 | Generalisation across regimes (fig 06) |
| 19 | Efficiency: training time & inference latency (fig 08) |
| 20 | Failure analysis, limitations, conclusions |

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `Axes.boxplot() got an unexpected keyword argument 'labels'` | Fixed in v1.0.1. If you are on an older copy, edit `plots.py` to drop `labels=algos` from the `boxplot` call and set tick labels afterwards. The kwarg was renamed in matplotlib 3.9 and removed in 3.11. |
| `OperationalError('database is locked')` from yfinance | Its sqlite timezone cache is contended — common on Windows and fatal under OneDrive/synced folders. The pipeline now relocates that cache into `data/yf_cache/`, downloads serially, and retries with backoff. Raise it with `--download-retries 8` if your connection is flaky. |
| `POISONED CACHE DETECTED` | A `prices_yahoo_*.csv` written by the pre-1.0.2 fallback contains simulated data. Delete the named file and re-run. |
| Log says `DATA PROVENANCE: ... SIMULATED` | You are training on generated prices. Delete `data/prices_*.csv`, fix the download (or use `--prices-csv`), and re-run with `--require-real-data`. |
| `IN-SAMPLE EVALUATION PERIOD DETECTED` | An evaluation window overlaps training. Those rows are memorisation, not results — fix the split or use `configs/long_train.yaml`. A Sharpe above ~3 there is the classic symptom. |
| Sharpe > 3 anywhere | Treat as a bug hypothesis first: check `in_sample`, then costs, then the data source. Real long-only multi-asset Sharpes live in 0–2. |
| `Prices: 5217 rows` for 2005-2024 | That is the business-day count including holidays. Real markets have ~5030 trading days. The series is synthetic. |
| `TypeError: 'type' object is not subscriptable` on import | You are on Python ≤3.9. Recreate the venv with 3.10–3.12. Quick unblock on 3.8: `pip install "multitasking==0.0.11"`, but other deps will bite you later — upgrade properly. |
| `yfinance` download fails / empty / won't import | The pipeline auto-falls back to synthetic and logs a warning. For **real** data without yfinance, export a CSV and use `--prices-csv mydata.csv` (see §3). |
| `yfinance` download fails / empty | Pipeline auto-falls back to synthetic. Force it with `--synthetic`, or download once on a different network — it caches. |
| Training feels slow | `device: cpu` is correct here; these MLPs are too small for GPU transfer overhead to pay off. Reduce `--timesteps` while developing. |
| Flat learning curves | Check `reward_scale`. Raw daily log-returns are ~1e-3 and the value function will not learn from them. |
| DQN collapses to one action | Increase `exploration_fraction`, or reduce `grid_steps` — 15 actions on a 543-dim state is already demanding. |
| `Period too short` | A test window is shorter than `window + 15` days. Widen the date range in the config. |
| Results differ between runs | Confirm the seed list, and check whether you are on cached vs freshly downloaded prices. `results/config_used.yaml` records exactly what ran. |
