"""Quintile portfolio analysis — spread, monotonicity, and cumulative return visualization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from factor_zoo.analytics.stats import sharpe_ratio


@dataclass
class QuintileResult:
    factor_id: str
    quintiles: pd.DataFrame   # columns q1–q5, DatetimeIndex
    spread: pd.Series         # q5 - q1, decimal, DatetimeIndex
    monotonicity_score: float # fraction of periods where q1<q2<q3<q4<q5
    spread_sharpe: float

    def plot(self) -> go.Figure:
        fig = go.Figure()
        colors = ["#d32f2f", "#f57c00", "#388e3c", "#1976d2", "#7b1fa2"]
        labels = ["Q1 (Bottom)", "Q2", "Q3", "Q4", "Q5 (Top)"]
        for q, color, label in zip(["q1", "q2", "q3", "q4", "q5"], colors, labels):
            s = self.quintiles[q].dropna()
            cum = (1 + s).cumprod()
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                name=label, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            ))
        fig.update_layout(
            title=f"{self.factor_id} — Quintile Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (base = 1)",
            hovermode="x unified",
            template="plotly_white",
            height=420,
        )
        return fig


def compute_quintile_analysis(factor_id: str, quintiles: pd.DataFrame) -> QuintileResult:
    """Compute spread, monotonicity score, and Sharpe from quintile return DataFrame."""
    spread = (quintiles["q5"] - quintiles["q1"]).rename(factor_id)

    df = quintiles[["q1", "q2", "q3", "q4", "q5"]].dropna()
    if len(df) == 0:
        mono_score = float("nan")
    else:
        strictly_increasing = (
            (df["q1"] < df["q2"]) & (df["q2"] < df["q3"]) &
            (df["q3"] < df["q4"]) & (df["q4"] < df["q5"])
        )
        mono_score = float(strictly_increasing.mean())

    return QuintileResult(
        factor_id=factor_id,
        quintiles=quintiles,
        spread=spread,
        monotonicity_score=mono_score,
        spread_sharpe=sharpe_ratio(spread.dropna()),
    )
