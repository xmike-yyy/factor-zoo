"""Tests for analytics/quintiles.py."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytest
from factor_zoo.analytics.quintiles import QuintileResult, compute_quintile_analysis


def _make_df(n=24, q1=-0.02, q2=-0.01, q3=0.00, q4=0.01, q5=0.02):
    """Create n months of constant quintile returns."""
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.DataFrame({"q1": [q1]*n, "q2": [q2]*n, "q3": [q3]*n, "q4": [q4]*n, "q5": [q5]*n}, index=dates)


def test_returns_quintile_result():
    result = compute_quintile_analysis("TestFactor", _make_df())
    assert isinstance(result, QuintileResult)
    assert result.factor_id == "TestFactor"


def test_quintiles_preserved():
    df = _make_df()
    result = compute_quintile_analysis("TestFactor", df)
    pd.testing.assert_frame_equal(result.quintiles, df)


def test_spread_is_q5_minus_q1():
    df = _make_df()
    result = compute_quintile_analysis("TestFactor", df)
    expected = df["q5"] - df["q1"]
    assert abs((result.spread - expected).abs().max()) < 1e-10


def test_monotonicity_score_perfect():
    result = compute_quintile_analysis("TestFactor", _make_df())
    assert result.monotonicity_score == 1.0


def test_monotonicity_score_never():
    # reversed order — q1 > q2 > q3 > q4 > q5
    df = _make_df(q1=0.02, q2=0.01, q3=0.00, q4=-0.01, q5=-0.02)
    result = compute_quintile_analysis("TestFactor", df)
    assert result.monotonicity_score == 0.0


def test_spread_sharpe_positive_for_positive_spread():
    # Create returns where spread varies in magnitude
    dates = pd.date_range("2010-01-31", periods=24, freq="ME")
    noise = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.01  # varying spread
    df = pd.DataFrame({
        "q1": -0.02 + noise * 0.5,
        "q2": -0.01 + noise * 0.5,
        "q3": 0.00 + noise * 0.5,
        "q4": 0.01 + noise * 0.5,
        "q5": 0.02 + noise,  # larger swing in q5 creates variable spread
    }, index=dates)
    result = compute_quintile_analysis("TestFactor", df)
    assert result.spread_sharpe > 0


def test_plot_returns_five_traces():
    result = compute_quintile_analysis("TestFactor", _make_df())
    fig = result.plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5
