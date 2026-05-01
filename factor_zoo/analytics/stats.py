"""Factor performance statistics. All returns must be decimals (0.01 = 1%)."""
import math
from typing import Optional

import numpy as np
import pandas as pd


def annualized_return(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    if len(clean) == 0:
        return float("nan")
    return float(clean.mean() * 12)


def annualized_vol(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std() * math.sqrt(12))


def sharpe_ratio(monthly_returns: pd.Series) -> float:
    ret = annualized_return(monthly_returns)
    vol = annualized_vol(monthly_returns)
    if math.isnan(vol) or abs(vol) < 1e-10:
        return float("nan")
    return ret / vol


def max_drawdown(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    if len(clean) == 0:
        return float("nan")
    cum = (1 + clean).cumprod()
    # Prepend 1.0 so the very first down-move is captured as a drawdown
    cum = pd.concat([pd.Series([1.0], index=[cum.index[0] - pd.tseries.frequencies.to_offset("1D")]), cum])
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def t_statistic(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    n = len(clean)
    if n < 2:
        return float("nan")
    return float(clean.mean() / (clean.std() / math.sqrt(n)))


def pre_post_sharpe(
    monthly_returns: pd.Series,
    pub_year: Optional[int],
) -> tuple[float, float]:
    """Returns (pre_pub_sharpe, post_pub_sharpe).
    Split is on the publication year: pre = strictly before pub_year,
    post = pub_year onward."""
    if pub_year is None or not isinstance(monthly_returns.index, pd.DatetimeIndex):
        nan = float("nan")
        return nan, nan
    pre = monthly_returns[monthly_returns.index.year < pub_year]
    post = monthly_returns[monthly_returns.index.year >= pub_year]
    return sharpe_ratio(pre), sharpe_ratio(post)


def compute_all_stats(
    monthly_returns: pd.Series,
    pub_year: Optional[int] = None,
) -> dict:
    """Compute all stats for a single factor. Returns dict ready to merge into factors table."""
    ret = monthly_returns.dropna()
    pre, post = pre_post_sharpe(monthly_returns, pub_year)
    return {
        "ann_return": annualized_return(ret),
        "ann_vol": annualized_vol(ret),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "t_stat": t_statistic(ret),
        "pre_pub_sharpe": pre,
        "post_pub_sharpe": post,
        "has_short_history": len(ret) < 60,
    }
