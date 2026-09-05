"""Backtest performance and risk metrics.

All functions take a series of *simple* periodic returns. Annualisation assumes
252 trading days. Sharpe is computed on excess returns over a constant
risk-free rate (default 0, which is the honest choice unless you also model the
short-rate path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_array(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    return r[np.isfinite(r)]


def total_return(returns) -> float:
    r = _as_array(returns)
    return float(np.prod(1.0 + r) - 1.0) if r.size else 0.0


def cagr(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_array(returns)
    if r.size == 0:
        return 0.0
    growth = np.prod(1.0 + r)
    years = r.size / periods_per_year
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_array(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0


def sharpe_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_array(returns) - rf / periods_per_year
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_array(returns) - rf / periods_per_year
    downside = r[r < 0]
    if r.size < 2 or downside.size < 2 or downside.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    r = _as_array(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def calmar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, periods_per_year) / mdd) if mdd > 1e-12 else 0.0


def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical VaR, reported as a positive loss magnitude."""
    r = _as_array(returns)
    return float(-np.quantile(r, alpha)) if r.size else 0.0


def conditional_var(returns, alpha: float = 0.05) -> float:
    """Historical CVaR / expected shortfall, positive loss magnitude."""
    r = _as_array(returns)
    if r.size == 0:
        return 0.0
    cutoff = np.quantile(r, alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if tail.size else float(-cutoff)


def summarise(returns, turnover=None, prefix: str = "") -> dict[str, float]:
    """One row of backtest statistics."""
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_vol": annual_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "var_95": value_at_risk(returns, 0.05),
        "cvar_95": conditional_var(returns, 0.05),
        "n_days": int(_as_array(returns).size),
    }
    if turnover is not None:
        t = _as_array(turnover)
        out["avg_daily_turnover"] = float(t.mean()) if t.size else 0.0
        out["ann_turnover"] = float(t.mean() * TRADING_DAYS) if t.size else 0.0
    return {f"{prefix}{k}": v for k, v in out.items()}


def equity_curve(returns, index=None) -> pd.Series:
    r = _as_array(returns)
    eq = np.cumprod(1.0 + r)
    return pd.Series(eq, index=index[: len(eq)] if index is not None else None)
