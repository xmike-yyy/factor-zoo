"""Drawdown analysis — underwater equity curve, max drawdown, and duration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import pandas as pd
import plotly.graph_objects as go


@dataclass
class DrawdownResult:
    factor_id: str
    drawdown_series: pd.Series       # underwater equity curve (0 to -1, DatetimeIndex)
    max_drawdown: float              # worst peak-to-trough (negative, or 0 if none)
    max_drawdown_start: Optional[pd.Timestamp] # date of the peak before max drawdown
    max_drawdown_end: Optional[pd.Timestamp]   # date of the trough of max drawdown
    max_drawdown_duration: int       # months from peak to trough
    current_drawdown: float          # drawdown as of last observation

    def plot(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.drawdown_series.index,
            y=self.drawdown_series.values,
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.2)",
            line=dict(color="#EF4444", width=1.5),
            name="Drawdown",
            hovertemplate="%{x|%b %Y}: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{self.factor_id} — Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            template="plotly_white",
            margin=dict(t=40, b=40, l=60, r=20),
            height=320,
        )
        return fig


def compute_drawdown(returns: pd.Series, factor_id: str) -> DrawdownResult:
    """Compute underwater equity curve and drawdown statistics.

    Parameters
    ----------
    returns : pd.Series
        Decimal monthly returns with DatetimeIndex.
    factor_id : str
        Stored on the result for display purposes.
    """
    clean = returns.dropna()
    if clean.empty:
        empty = pd.Series(dtype=float, name=factor_id)
        return DrawdownResult(
            factor_id=factor_id,
            drawdown_series=empty,
            max_drawdown=float("nan"),
            max_drawdown_start=None,
            max_drawdown_end=None,
            max_drawdown_duration=0,
            current_drawdown=float("nan"),
        )

    cum = (1 + clean).cumprod()
    running_max = cum.cummax()
    dd_series = (cum / running_max - 1).rename(factor_id)

    max_dd = float(dd_series.min())

    if max_dd >= 0:
        first = cast(pd.Timestamp, clean.index[0])
        return DrawdownResult(
            factor_id=factor_id,
            drawdown_series=dd_series,
            max_drawdown=0.0,
            max_drawdown_start=first,
            max_drawdown_end=first,
            max_drawdown_duration=0,
            current_drawdown=0.0,
        )

    trough_idx = cast(pd.Timestamp, dd_series.idxmin())
    peak_idx = cast(pd.Timestamp, cum.loc[:trough_idx].idxmax())
    duration = len(dd_series.loc[peak_idx:trough_idx]) - 1

    return DrawdownResult(
        factor_id=factor_id,
        drawdown_series=dd_series,
        max_drawdown=max_dd,
        max_drawdown_start=peak_idx,
        max_drawdown_end=trough_idx,
        max_drawdown_duration=duration,
        current_drawdown=float(dd_series.iloc[-1]),
    )
