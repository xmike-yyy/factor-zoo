"""Tests for analytics/drawdown.py."""
import pytest
import pandas as pd
import plotly.graph_objects as go

from factor_zoo.analytics.drawdown import compute_drawdown, DrawdownResult


def _series(vals, start="2010-01-31"):
    dates = pd.date_range(start, periods=len(vals), freq="ME")
    return pd.Series(vals, index=dates, name="TEST")


class TestComputeDrawdown:
    def test_returns_drawdown_result(self):
        result = compute_drawdown(_series([0.01] * 60), "X")
        assert isinstance(result, DrawdownResult)
        assert result.factor_id == "X"

    def test_all_positive_no_drawdown(self):
        # All positive returns → underwater curve is always 0
        result = compute_drawdown(_series([0.01] * 60), "X")
        assert result.max_drawdown == pytest.approx(0.0, abs=1e-10)
        assert result.max_drawdown_duration == 0

    def test_known_drawdown(self):
        # +100% then -50%: cum = [2.0, 1.0], running_max = [2.0, 2.0]
        # drawdown series = [0.0, -0.5], so max_drawdown = -0.5
        result = compute_drawdown(_series([1.0, -0.5]), "X")
        assert result.max_drawdown == pytest.approx(-0.5, abs=1e-6)

    def test_drawdown_series_nonpositive(self):
        # The underwater curve must be <= 0 at every point
        result = compute_drawdown(_series([0.01, -0.05, 0.02, -0.03] * 12), "X")
        assert (result.drawdown_series <= 1e-10).all()

    def test_drawdown_series_length(self):
        result = compute_drawdown(_series([0.01] * 36), "X")
        assert len(result.drawdown_series) == 36

    def test_current_drawdown_after_loss(self):
        # Sequence ends with a loss, so current drawdown should be negative
        result = compute_drawdown(_series([0.1, 0.1, -0.3]), "X")
        assert result.current_drawdown < 0

    def test_max_drawdown_start_before_end(self):
        # Peak date must not be after trough date
        result = compute_drawdown(_series([0.2, -0.1, -0.15, 0.05, -0.2]), "X")
        if result.max_drawdown < 0:
            assert result.max_drawdown_start <= result.max_drawdown_end

    def test_plot_returns_figure(self):
        result = compute_drawdown(_series([0.01] * 36), "X")
        assert isinstance(result.plot(), go.Figure)
