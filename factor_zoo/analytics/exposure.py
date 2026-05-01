"""OLS factor exposure analysis for user-provided return series."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go


@dataclass
class ExposureResult:
    loadings: dict[str, float]
    t_stats: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    alpha: float
    alpha_t_stat: float
    overlap_months: int

    def plot(self) -> go.Figure:
        factors = list(self.loadings.keys())
        vals = [self.loadings[f] for f in factors]
        errors = []
        for f in factors:
            t = self.t_stats.get(f, 0)
            se = abs(self.loadings[f] / t) if abs(t) > 1e-10 else 0.0
            errors.append(se)
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
        fig = go.Figure(data=go.Bar(
            x=factors,
            y=vals,
            error_y=dict(type="data", array=errors, visible=True),
            marker_color=colors,
        ))
        fig.update_layout(
            title=f"Factor Loadings  (R²={self.r_squared:.2f}, α={self.alpha*100:.2f}%/yr)",
            xaxis_title="Factor",
            yaxis_title="Loading",
            template="plotly_white",
        )
        return fig


def compute_exposure(
    user_returns: pd.Series,
    factor_wide: pd.DataFrame,
) -> ExposureResult:
    """Run OLS regression of user_returns on factor_wide."""
    combined = pd.concat([user_returns.rename("__y__"), factor_wide], axis=1).dropna()

    if len(combined) < 36:
        warnings.warn(
            f"Overlap between user returns and factor data is {len(combined)} months "
            f"(< 36 recommended for reliable estimates)",
            UserWarning,
            stacklevel=2,
        )

    y = combined["__y__"]
    X = combined.drop(columns=["__y__"])
    X_const = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X_const).fit()

    factor_names = list(X.columns)
    loadings = {f: float(model.params[f]) for f in factor_names}
    t_stats = {f: float(model.tvalues[f]) for f in factor_names}
    p_values = {f: float(model.pvalues[f]) for f in factor_names}

    alpha_monthly = float(model.params["const"])
    alpha_annualized = alpha_monthly * 12
    alpha_t = float(model.tvalues["const"])

    return ExposureResult(
        loadings=loadings,
        t_stats=t_stats,
        p_values=p_values,
        r_squared=float(model.rsquared),
        alpha=alpha_annualized,
        alpha_t_stat=alpha_t,
        overlap_months=len(combined),
    )
