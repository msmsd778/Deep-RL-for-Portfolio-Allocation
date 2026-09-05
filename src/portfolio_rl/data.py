"""Price loading, feature construction and time-ordered splitting.

Two loading paths:

1. `yfinance` (default when available and `synthetic=False`). Prices are cached
   to CSV on first download so that every later run is byte-identical.
2. A regime-switching synthetic generator. This exists so the whole pipeline is
   runnable with no network access, and so that a reviewer can verify the code
   without depending on a third-party API that may rate-limit or change schema.

Everything downstream consumes the same tidy `DataFrame` of adjusted closes
indexed by date, so the two paths are interchangeable.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _cache_paths(cache_dir: Path, source: str, tickers: list[str], start: str, end: str):
    key = f"{'-'.join(tickers)}_{start}_{end}"
    csv = cache_dir / f"prices_{source}_{key}.csv"
    return csv, csv.with_suffix(".meta.json")


def _write_cache(df: pd.DataFrame, source: str, cache_dir: Path,
                 tickers: list[str], start: str, end: str) -> None:
    """Write the price cache plus a provenance sidecar.

    The sidecar is what makes the cache auditable. Without it there is no way,
    days later, to tell a real series from a fallback-generated one -- and a
    silently simulated experiment is far worse than a failed one.
    """
    csv, meta = _cache_paths(cache_dir, source, tickers, start, end)
    df.to_csv(csv)
    meta.write_text(json.dumps({
        "source": source,
        "tickers": tickers,
        "start": start,
        "end": end,
        "rows": int(len(df)),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    logger.info("Cached %s prices to %s (%d rows)", source, csv.name, len(df))


def looks_synthetic(df: pd.DataFrame) -> bool:
    """Heuristic: does this index cover *every* business day?

    The synthetic generator uses `pd.bdate_range`, which includes market
    holidays. A real exchange calendar does not -- US markets close roughly
    9-10 days a year beyond weekends. An index with no holiday gaps at all is
    therefore almost certainly generated.
    """
    if len(df) < 300:
        return False
    full = pd.bdate_range(df.index[0], df.index[-1])
    return len(df) == len(full) and df.index.equals(pd.DatetimeIndex(full))


def _read_cache(csv: Path, meta: Path, expected_source: str,
                tickers: list[str]) -> pd.DataFrame | None:
    """Load a cached series, refusing it if provenance does not check out."""
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        logger.warning("Cache %s missing %s; ignoring it.", csv.name, missing)
        return None
    df = df[tickers]

    if meta.exists():
        recorded = json.loads(meta.read_text()).get("source")
        if recorded != expected_source:
            logger.error(
                "Cache %s says source=%r but %r was requested. Refusing it.",
                csv.name, recorded, expected_source,
            )
            return None
        logger.info("Loaded %d rows from cache %s (source=%s)",
                    len(df), csv.name, recorded)
        return df

    # No sidecar: written by an older version, provenance unknown.
    if expected_source == "yahoo" and looks_synthetic(df):
        logger.error("=" * 70)
        logger.error("POISONED CACHE DETECTED: %s", csv.resolve())
        logger.error(
            "This file is tagged as real market data but its index covers every "
            "business day with no exchange holidays, which means it was written "
            "by the synthetic fallback in an earlier version of this code."
        )
        logger.error("DELETE IT and re-run:  del \"%s\"", csv)
        logger.error("=" * 70)
        return None

    logger.warning(
        "Cache %s has no provenance sidecar (written by an older version). "
        "Assuming source=%s. Delete it if you are unsure.", csv.name, expected_source
    )
    return df


# --------------------------------------------------------------------------- #
def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data",
    synthetic: bool = False,
    synthetic_seed: int = 20260420,
    csv_path: str | Path | None = None,
    retries: int = 4,
    backoff: float = 2.0,
    require_real: bool = False,
) -> pd.DataFrame:
    """Return a (dates x tickers) frame of adjusted close prices.

    Resolution order:
      1. `csv_path`, if given -- your own file, no network needed.
      2. the on-disk cache **for the requested source only**.
      3. yfinance (unless `synthetic=True`).
      4. the synthetic generator -- cached under its OWN filename, never under
         the real-data one. A fallback must not be able to masquerade as a
         download on the next run.

    Set `require_real=True` to make a failed download raise instead of falling
    back. Use that for the run whose numbers you intend to present.
    """
    if csv_path is not None:
        return load_prices_from_csv(csv_path, tickers, start, end)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = "synthetic" if synthetic else "yahoo"

    csv, meta = _cache_paths(cache_dir, source, tickers, start, end)
    if csv.exists():
        cached = _read_cache(csv, meta, source, tickers)
        if cached is not None:
            return cached

    if synthetic:
        df = make_synthetic_prices(tickers, start, end, seed=synthetic_seed)
        _write_cache(df, "synthetic", cache_dir, tickers, start, end)
        return df

    df = _download_yahoo(
        tickers, start, end, retries=retries, backoff=backoff, cache_dir=cache_dir
    )
    if df is not None:
        _write_cache(df, "yahoo", cache_dir, tickers, start, end)
        return df

    if require_real:
        raise RuntimeError(
            f"Could not download real prices after {retries} attempts and "
            "require_real=True. Re-run without --require-real-data to use "
            "simulated data, or supply --prices-csv <file>."
        )

    logger.warning("=" * 70)
    logger.warning("FALLING BACK TO SYNTHETIC DATA after %d failed attempts.", retries)
    logger.warning(
        "These results are SIMULATED. Regime labels (gfc, covid_crash, ...) are "
        "meaningless here -- they are just date ranges. Do not present them."
    )
    logger.warning(
        "The simulated series is cached under its own 'synthetic' filename, so "
        "it will NOT be mistaken for real data on the next run."
    )
    logger.warning("For real data without yfinance: --prices-csv <file>. README section 3.")
    logger.warning("=" * 70)

    synth_csv, synth_meta = _cache_paths(cache_dir, "synthetic", tickers, start, end)
    if synth_csv.exists():
        cached = _read_cache(synth_csv, synth_meta, "synthetic", tickers)
        if cached is not None:
            return cached
    df = make_synthetic_prices(tickers, start, end, seed=synthetic_seed)
    _write_cache(df, "synthetic", cache_dir, tickers, start, end)
    return df


def load_prices_from_csv(
    path: str | Path, tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Load adjusted closes from a user-supplied CSV.

    Expected shape: a date column first (any parseable format), then one column
    per ticker of adjusted close prices.

        Date,SPY,IEF,GLD
        2005-01-03,86.35,52.10,42.74
        ...

    This is the escape hatch when yfinance will not install. Stooq, Nasdaq Data
    Link, Investing.com and your broker's export all produce something one
    `pandas` call away from this.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path.resolve()}")

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path.name} is missing column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[tickers].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if len(df) < 300:
        raise ValueError(
            f"Only {len(df)} usable rows in {path.name} between {start} and {end}. "
            "Check the date column parsed correctly and the range overlaps."
        )
    df.index.name = "Date"
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _download_yahoo(
    tickers: list[str],
    start: str,
    end: str,
    retries: int = 4,
    backoff: float = 2.0,
    cache_dir: str | Path = "data",
) -> pd.DataFrame | None:
    """Download adjusted closes, retrying transient failures with backoff.

    Retries matter here because the common failures are *transient*: rate
    limiting, a locked sqlite cache, or an empty response from an endpoint that
    works fine seconds later. Falling straight through to synthetic data on the
    first hiccup silently swaps your real experiment for a simulated one, which
    is the worst possible failure mode -- it looks like success.
    """
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad. yfinance is an OPTIONAL dependency, and a broken
        # install must degrade to the synthetic fallback rather than kill the
        # run. Catching only ImportError is not enough: on Python 3.8 the
        # `multitasking` transitive dependency raises TypeError at import time
        # ("'type' object is not subscriptable", PEP 585 generics need 3.9+).
        logger.warning("yfinance could not be imported (%s: %s)", type(exc).__name__, exc)
        return None

    # yfinance keeps a small sqlite database for timezone metadata. Its default
    # location is shared across processes and, on Windows (especially under
    # OneDrive or a synced folder), concurrent access produces
    # `OperationalError('database is locked')`. Pointing it at our own cache
    # directory removes the contention.
    try:
        loc = Path(cache_dir) / "yf_cache"
        loc.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(loc))
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        logger.debug("Could not relocate the yfinance tz cache: %s", exc)

    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                # Serial downloads: for a handful of tickers the speed cost is
                # negligible and it avoids concurrent writes to the sqlite
                # cache, which is the usual source of 'database is locked'.
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001 - network/API failures are equivalent
            raw, last_error = None, f"{type(exc).__name__}: {exc}"
        else:
            if raw is None or len(raw) == 0:
                # yf.download often does NOT raise: it logs and returns an empty
                # frame. That is still a failure and still worth retrying.
                last_error = "empty response"

        if raw is not None and len(raw) > 0:
            close = (
                raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            )
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            missing = [t for t in tickers if t not in close.columns]
            if missing:
                last_error = f"missing tickers in response: {missing}"
            else:
                close = close[tickers].dropna(how="any")
                if len(close) > 0:
                    if attempt > 1:
                        logger.info("Download succeeded on attempt %d", attempt)
                    return close
                last_error = "no rows survived NaN filtering"

        if attempt < retries:
            wait = backoff ** (attempt - 1)
            logger.warning(
                "Price download attempt %d/%d failed (%s); retrying in %.1fs ...",
                attempt, retries, last_error, wait,
            )
            time.sleep(wait)

    logger.warning(
        "All %d download attempts failed. Last error: %s", retries, last_error
    )
    return None


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
# Annualised (drift, vol) per regime, ordered as [equity, bond, gold]. These are
# not fitted parameters, they are deliberately stylised: equities carry the risk
# premium and the crash risk, bonds are the low-vol diversifier, gold is the
# safe-haven asset whose correlation with equities flips sign in a crisis.
_REGIME_PARAMS = {
    "calm": {
        "mu": np.array([0.10, 0.03, 0.05]),
        "sigma": np.array([0.12, 0.05, 0.14]),
        "corr": np.array(
            [[1.00, -0.25, 0.05], [-0.25, 1.00, 0.15], [0.05, 0.15, 1.00]]
        ),
    },
    "crisis": {
        "mu": np.array([-0.45, 0.06, 0.12]),
        "sigma": np.array([0.42, 0.09, 0.26]),
        "corr": np.array(
            [[1.00, -0.55, -0.30], [-0.55, 1.00, 0.30], [-0.30, 0.30, 1.00]]
        ),
    },
    "inflation": {
        "mu": np.array([-0.12, -0.08, 0.02]),
        "sigma": np.array([0.22, 0.11, 0.18]),
        "corr": np.array([[1.00, 0.45, 0.10], [0.45, 1.00, 0.05], [0.10, 0.05, 1.00]]),
    },
}
# Daily transition matrix over (calm, crisis, inflation): crises are rare, short
# and sticky once entered.
_TRANSITION = np.array(
    [
        [0.9955, 0.0030, 0.0015],
        [0.0180, 0.9820, 0.0000],
        [0.0090, 0.0010, 0.9900],
    ]
)


def make_synthetic_prices(
    tickers: list[str], start: str, end: str, seed: int = 20260420
) -> pd.DataFrame:
    """Regime-switching multivariate GBM on business days.

    A hidden 3-state Markov chain selects the drift/vol/correlation block for
    each day. Correlations differ across regimes, which is the whole point: a
    static-covariance simulator would make diversification look far too reliable
    and the RL agent would learn a lesson that does not survive a real crisis.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days, n_assets = len(dates), len(tickers)
    if n_assets != 3:
        # Generalise by tiling the 3-asset blocks; keeps the generator usable if
        # a group swaps in a different universe.
        _widen_regime_params(n_assets)

    states = np.empty(n_days, dtype=int)
    states[0] = 0
    for t in range(1, n_days):
        states[t] = rng.choice(3, p=_TRANSITION[states[t - 1]])

    names = ["calm", "crisis", "inflation"]
    chols = {}
    for i, name in enumerate(names):
        p = _REGIME_PARAMS[name]
        cov = np.outer(p["sigma"], p["sigma"]) * p["corr"]
        chols[i] = (p["mu"], np.linalg.cholesky(cov[:n_assets, :n_assets]))

    dt = 1.0 / TRADING_DAYS
    log_rets = np.empty((n_days, n_assets))
    for t in range(n_days):
        mu, chol = chols[states[t]]
        mu = mu[:n_assets]
        z = rng.standard_normal(n_assets)
        diffusion = chol @ z * np.sqrt(dt)
        drift = (mu - 0.5 * np.diag(chol @ chol.T)) * dt
        log_rets[t] = drift + diffusion

    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"
    return df


def _widen_regime_params(n_assets: int) -> None:
    """Tile the 3-asset parameter blocks up to `n_assets` (best-effort)."""
    for params in _REGIME_PARAMS.values():
        reps = int(np.ceil(n_assets / 3))
        params["mu"] = np.tile(params["mu"], reps)[:n_assets]
        params["sigma"] = np.tile(params["sigma"], reps)[:n_assets]
        corr = np.eye(n_assets) * 0.0
        base = params["corr"]
        for i in range(n_assets):
            for j in range(n_assets):
                corr[i, j] = base[i % 3, j % 3] if i != j else 1.0
        params["corr"] = corr


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="any")


def build_features(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Per-asset features that are all *causal* (no future information).

    For each asset:
      - `ret_1d`   : yesterday's log return
      - `mom_{k}`  : cumulative log return over the last k days (k = 5, 21, 63)
      - `vol_21`   : 21-day realised volatility, annualised
      - `dd`       : drawdown from the rolling `window`-day high

    Every rolling statistic ends at t-1 by construction because the environment
    only ever hands the agent rows strictly before the return it is about to
    earn. The `.shift(1)` here is the second line of defence.
    """
    rets = compute_log_returns(prices)
    feats: dict[str, pd.Series] = {}

    for col in prices.columns:
        r = rets[col]
        feats[f"{col}__ret_1d"] = r
        for k in (5, 21, 63):
            feats[f"{col}__mom_{k}"] = r.rolling(k).sum()
        feats[f"{col}__vol_21"] = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
        roll_max = prices[col].rolling(window, min_periods=1).max()
        feats[f"{col}__dd"] = (prices[col] / roll_max - 1.0).reindex(r.index)

    df = pd.DataFrame(feats).shift(1)          # strictly information up to t-1
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def standardise(
    train_df: pd.DataFrame, other: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Z-score using **training-set** moments only.

    Fitting the scaler on the full sample is one of the most common silent
    look-ahead leaks in financial ML: the test-period mean and standard
    deviation are not knowable at training time.
    """
    mu = train_df.mean()
    sd = train_df.std().replace(0.0, 1.0)
    scaled_train = (train_df - mu) / sd
    scaled_other = {k: (v - mu) / sd for k, v in other.items()}
    return scaled_train, scaled_other, {"mean": mu, "std": sd}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return max(pd.Timestamp(a_start), pd.Timestamp(b_start)) <= min(
        pd.Timestamp(a_end), pd.Timestamp(b_end)
    )


def make_splits(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    splits_cfg,
    strict: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build a dict of named periods, each holding aligned prices+features.

    Any evaluation period that overlaps the training window is flagged
    `in_sample=True`. That flag then travels all the way into the results
    tables, because an in-sample period is a *memorisation check*, not a
    generalisation test, and the two must never be read off the same row. With
    `strict=True` an overlap raises instead of warning.
    """
    common = features.index.intersection(prices.index)
    prices, features = prices.loc[common], features.loc[common]

    train_start, train_end = splits_cfg.train

    periods: dict[str, list[str]] = {
        "train": splits_cfg.train,
        "val": splits_cfg.val,
        **splits_cfg.test_regimes,
        **splits_cfg.stress_regime,
    }

    out: dict[str, dict[str, pd.DataFrame]] = {}
    overlapping: list[str] = []
    for name, (start, end) in periods.items():
        p = slice_period(prices, start, end)
        f = slice_period(features, start, end)
        if len(p) < 30:
            logger.warning("Period '%s' has only %d rows - skipping", name, len(p))
            continue
        in_sample = name != "train" and _overlaps(start, end, train_start, train_end)
        if in_sample:
            overlapping.append(f"{name} ({start}..{end})")
        out[name] = {"prices": p, "features": f, "in_sample": in_sample}

    if overlapping:
        msg = (
            "Evaluation period(s) overlap the training window "
            f"({train_start}..{train_end}): {', '.join(overlapping)}. "
            "Results there measure memorisation, not generalisation, and will "
            "look implausibly good."
        )
        if strict:
            raise ValueError(msg)
        logger.error("=" * 70)
        logger.error("IN-SAMPLE EVALUATION PERIOD DETECTED")
        logger.error("%s", msg)
        logger.error(
            "These periods are tagged in_sample=True in the results tables. "
            "Do NOT present them as out-of-sample performance."
        )
        logger.error("=" * 70)

    return out
