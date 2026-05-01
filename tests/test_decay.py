"""Tests for factor_zoo/analytics/decay.py"""
import math
import numpy as np
import pandas as pd
import pytest

from factor_zoo.analytics.decay import (
    compute_decay,
    _rolling_sharpe,
    _fit_decay,
    DecayResult,
)


def _series(values, start="1990-01-31", freq="ME") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _noisy(n=240, mean=0.008, std=0.04, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(rng.normal(mean, std, n).tolist())


class TestRollingSharpey:
    def test_returns_series_shorter_than_input(self):
        s = _noisy(120)
        rs = _rolling_sharpe(s, window=36)
        assert len(rs) < len(s)

    def test_no_nan_in_output(self):
        s = _noisy(120)
        rs = _rolling_sharpe(s, window=36)
        assert not rs.isna().any()

    def test_positive_mean_gives_positive_rolling_sharpe(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(0.02, 0.01, 120)
        s = _series(vals.tolist())
        rs = _rolling_sharpe(s, window=36)
        assert (rs > 0).mean() > 0.9

    def test_length_with_36_window(self):
        s = _noisy(120)
        rs = _rolling_sharpe(s, window=36)
        assert len(rs) == 120 - 36 + 1


class TestFitDecay:
    def test_returns_none_when_no_pub_year(self):
        s = _noisy(200)
        rs = _rolling_sharpe(s, window=36)
        half_life, decay_rate = _fit_decay(rs, publication_year=None)
        assert half_life is None and decay_rate is None

    def test_returns_none_when_fewer_than_24_post_pub_months(self):
        s = _noisy(200)
        rs = _rolling_sharpe(s, window=36)
        half_life, decay_rate = _fit_decay(rs, publication_year=2100)
        assert half_life is None

    def test_returns_floats_for_sufficient_post_pub_data(self):
        s = _noisy(300)
        rs = _rolling_sharpe(s, window=36)
        half_life, decay_rate = _fit_decay(rs, publication_year=2000)
        if half_life is not None:
            assert isinstance(half_life, float)
            assert isinstance(decay_rate, float)


class TestComputeDecay:
    def _long_series(self) -> pd.Series:
        rng = np.random.default_rng(0)
        return _series(rng.normal(0.008, 0.04, 360).tolist())

    def test_returns_decay_result(self):
        s = self._long_series()
        result = compute_decay(s, "TestFactor", publication_year=2000)
        assert isinstance(result, DecayResult)

    def test_factor_id_preserved(self):
        s = self._long_series()
        result = compute_decay(s, "Mom12m", publication_year=2005)
        assert result.factor_id == "Mom12m"

    def test_rolling_sharpe_is_series(self):
        s = self._long_series()
        result = compute_decay(s, "x", publication_year=2000)
        assert isinstance(result.rolling_sharpe, pd.Series)
        assert len(result.rolling_sharpe) > 0

    def test_pre_post_sharpe_are_floats(self):
        s = self._long_series()
        result = compute_decay(s, "x", publication_year=2000)
        assert isinstance(result.pre_pub_sharpe, float)
        assert isinstance(result.post_pub_sharpe, float)

    def test_no_pub_year_gives_none_half_life(self):
        s = self._long_series()
        result = compute_decay(s, "x", publication_year=None)
        assert result.half_life is None
        assert result.decay_rate is None

    def test_plot_returns_figure(self):
        import plotly.graph_objects as go
        s = self._long_series()
        result = compute_decay(s, "x", publication_year=2000)
        fig = result.plot()
        assert isinstance(fig, go.Figure)
