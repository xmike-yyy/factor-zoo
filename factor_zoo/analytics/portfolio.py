"""Portfolio construction from factor return series."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from factor_zoo.analytics.stats import (
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
    t_statistic,
    compute_all_stats,
)


@dataclass
class PortfolioResult:
    returns: pd.Series
    weights: dict[str, float]
    stats: dict
    factor_stats: pd.DataFrame


def construct_portfolio(
    wide_df: pd.DataFrame,
    method: str = "equal",
    weights: Optional[list[float]] = None,
) -> PortfolioResult:
    """Construct a multi-factor portfolio.

    Parameters
    ----------
    wide_df : pd.DataFrame — wide monthly returns (date index, factor columns)
    method : str — "equal", "custom", "max_sharpe", or "risk_parity"
    weights : list[float] — required when method="custom", must sum to 1.0
    """
    df = wide_df.dropna(how="any")
    if len(df) < 60:
        warnings.warn(
            f"Portfolio intersection is {len(df)} months (< 60 recommended)",
            UserWarning,
            stacklevel=2,
        )

    factors = list(df.columns)
    n = len(factors)

    if method == "equal":
        w = np.ones(n) / n
    elif method == "custom":
        if weights is None or len(weights) != n:
            raise ValueError(
                f"weights must have same length as factors ({n}), got "
                f"{len(weights) if weights is not None else None}"
            )
        w = np.array(weights, dtype=float)
        if abs(w.sum() - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {w.sum():.6f}")
    elif method == "max_sharpe":
        w = _max_sharpe_weights(df)
    elif method == "risk_parity":
        w = _risk_parity_weights(df)
    else:
        raise ValueError(f"Unknown portfolio method: {method!r}")

    weight_dict = dict(zip(factors, w.tolist()))
    portfolio_returns = (df * w).sum(axis=1)
    portfolio_returns.name = "portfolio"

    stats = compute_all_stats(portfolio_returns)

    rows = []
    for fid in factors:
        s = df[fid]
        rows.append({
            "factor_id": fid,
            "weight": weight_dict[fid],
            "ann_return": annualized_return(s),
            "ann_vol": annualized_vol(s),
            "sharpe": sharpe_ratio(s),
            "max_drawdown": max_drawdown(s),
            "t_stat": t_statistic(s),
        })
    factor_stats = pd.DataFrame(rows).set_index("factor_id")

    return PortfolioResult(
        returns=portfolio_returns,
        weights=weight_dict,
        stats=stats,
        factor_stats=factor_stats,
    )


def _max_sharpe_weights(df: pd.DataFrame) -> np.ndarray:
    n = df.shape[1]

    def neg_sharpe(w: np.ndarray) -> float:
        port = (df * w).sum(axis=1)
        mean = float(port.mean()) * 12
        vol = float(port.std()) * np.sqrt(12)
        if vol < 1e-10:
            return 0.0
        return -mean / vol

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    return result.x


def _risk_parity_weights(df: pd.DataFrame) -> np.ndarray:
    vols = df.std() * np.sqrt(12)
    inv_vol = 1.0 / vols.replace(0, np.nan)
    w = inv_vol / inv_vol.sum()
    return w.fillna(0.0).values
