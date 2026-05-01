"""Tests for factor_zoo/analytics/exposure.py"""
import math
import numpy as np
import pandas as pd
import pytest

from factor_zoo.analytics.exposure import compute_exposure, ExposureResult


def _synthetic_exposure_data(n=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    F0 = pd.Series(rng.normal(0.008, 0.04, n), index=idx, name="F0")
    F1 = pd.Series(rng.normal(0.005, 0.03, n), index=idx, name="F1")
    noise = rng.normal(0, 0.01, n)
    user_returns = 0.5 * F0 + 0.3 * F1 + noise
    user_returns.name = "portfolio"
    wide = pd.concat([F0, F1], axis=1)
    return user_returns, wide


class TestComputeExposure:
    def test_returns_exposure_result(self):
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        assert isinstance(result, ExposureResult)

    def test_loadings_keys_match_factors(self):
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        assert set(result.loadings.keys()) == {"F0", "F1"}

    def test_loadings_close_to_true_values(self):
        user, wide = _synthetic_exposure_data(n=240)
        result = compute_exposure(user, wide)
        assert abs(result.loadings["F0"] - 0.5) < 0.15
        assert abs(result.loadings["F1"] - 0.3) < 0.15

    def test_r_squared_between_0_and_1(self):
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        assert 0.0 <= result.r_squared <= 1.0

    def test_alpha_is_float(self):
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        assert isinstance(result.alpha, float)

    def test_t_stats_and_p_values_keys_match_loadings(self):
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        assert set(result.t_stats.keys()) == set(result.loadings.keys())
        assert set(result.p_values.keys()) == set(result.loadings.keys())

    def test_overlap_months_is_correct(self):
        user, wide = _synthetic_exposure_data(n=120)
        result = compute_exposure(user, wide)
        assert result.overlap_months == 120

    def test_warns_when_overlap_below_36_months(self):
        rng = np.random.default_rng(9)
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        user = pd.Series(rng.normal(0.008, 0.04, 24), index=idx)
        wide = pd.DataFrame({"F0": rng.normal(0.005, 0.03, 24)}, index=idx)
        with pytest.warns(UserWarning, match="36"):
            compute_exposure(user, wide)

    def test_plot_returns_figure(self):
        import plotly.graph_objects as go
        user, wide = _synthetic_exposure_data()
        result = compute_exposure(user, wide)
        fig = result.plot()
        assert isinstance(fig, go.Figure)
