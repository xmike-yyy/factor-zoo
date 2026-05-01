"""Factor decay analysis — rolling Sharpe, half-life, and publication decay curves."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

from factor_zoo.analytics.stats import sharpe_ratio


@dataclass
class DecayResult:
    factor_id: str
    rolling_sharpe: pd.Series
    half_life: Optional[float]
    decay_rate: Optional[float]
    pre_pub_sharpe: float
    post_pub_sharpe: float
    publication_year: Optional[int]

    def plot(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.rolling_sharpe.index,
            y=self.rolling_sharpe.values,
            name="Rolling Sharpe (36m)",
            line=dict(color="#2563eb"),
        ))
        if self.publication_year is not None:
            fig.add_vline(
                x=pd.Timestamp(f"{self.publication_year}-01-01").timestamp() * 1000,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Published {self.publication_year}",
            )
        title = f"{self.factor_id} — Factor Decay"
        if self.half_life is not None:
            title += f"  (half-life ≈ {self.half_life:.0f} months)"
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Rolling Sharpe Ratio",
            template="plotly_white",
        )
        return fig


def compute_decay(
    returns: pd.Series,
    factor_id: str,
    publication_year: Optional[int],
    window: int = 36,
) -> DecayResult:
    """Compute decay metrics for a single factor's return series."""
    rs = _rolling_sharpe(returns, window)
    half_life, decay_rate = _fit_decay(rs, publication_year)

    if publication_year is not None:
        pre = returns[returns.index.year < publication_year]
        post = returns[returns.index.year >= publication_year]
        pre_sharpe = sharpe_ratio(pre)
        post_sharpe = sharpe_ratio(post)
    else:
        pre_sharpe = sharpe_ratio(returns)
        post_sharpe = float("nan")

    return DecayResult(
        factor_id=factor_id,
        rolling_sharpe=rs,
        half_life=half_life,
        decay_rate=decay_rate,
        pre_pub_sharpe=pre_sharpe,
        post_pub_sharpe=post_sharpe,
        publication_year=publication_year,
    )


def _rolling_sharpe(returns: pd.Series, window: int = 36) -> pd.Series:
    """Annualized rolling Sharpe ratio, NaN rows dropped."""
    mean = returns.rolling(window).mean() * 12
    vol = returns.rolling(window).std() * np.sqrt(12)
    sharpe = mean / vol.replace(0, np.nan)
    return sharpe.dropna()


def _fit_decay(
    rolling_sharpe: pd.Series,
    publication_year: Optional[int],
) -> tuple[Optional[float], Optional[float]]:
    """Fit exponential decay to post-publication rolling Sharpe. Returns (half_life, lambda)."""
    if publication_year is None:
        return None, None

    post = rolling_sharpe[rolling_sharpe.index.year >= publication_year]
    if len(post) < 24:
        return None, None

    pub_date = pd.Timestamp(f"{publication_year}-01-01")
    t = np.array([(d - pub_date).days / 30.0 for d in post.index], dtype=float)
    s = post.values.astype(float)

    mask = np.isfinite(s)
    if mask.sum() < 24:
        return None, None
    t, s = t[mask], s[mask]

    try:
        S0_init = float(s[0]) if abs(float(s[0])) > 1e-10 else 0.5
        popt, _ = curve_fit(
            lambda t, S0, lam: S0 * np.exp(-lam * t),
            t, s,
            p0=[S0_init, 0.01],
            maxfev=5000,
        )
        lam = float(popt[1])
        if lam <= 0:
            return None, lam
        return float(np.log(2) / lam), lam
    except (RuntimeError, ValueError):
        return None, None
