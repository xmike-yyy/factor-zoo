"""Tests for rolling correlation logic used in the Compare page."""
import pandas as pd
import numpy as np
import pytest
from factor_zoo.analytics.correlation import rolling_correlation


def make_series(n=60, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="ME")
    return pd.Series(rng.standard_normal(n), index=idx)


def test_rolling_corr_returns_series():
    s1, s2 = make_series(60), make_series(60, seed=99)
    result = rolling_correlation(s1, s2, window=12)
    assert isinstance(result, pd.Series)
    assert len(result) == 60


def test_rolling_corr_perfect_correlation():
    s = make_series(60)
    result = rolling_correlation(s, s, window=12)
    assert result.dropna().between(-1.001, 1.001).all()
    # Perfect self-correlation should give all 1.0 after warmup
    assert (result.dropna().round(10) == 1.0).all()


def test_rolling_corr_window_too_large_returns_all_nan():
    s1, s2 = make_series(10), make_series(10, seed=99)
    result = rolling_correlation(s1, s2, window=36)
    assert result.dropna().empty  # window > series length → all NaN


def test_rolling_corr_empty_overlap():
    idx1 = pd.date_range("2000-01-01", periods=20, freq="ME")
    idx2 = pd.date_range("2010-01-01", periods=20, freq="ME")
    s1 = pd.Series(np.ones(20), index=idx1)
    s2 = pd.Series(np.ones(20), index=idx2)
    result = rolling_correlation(s1, s2, window=12)
    assert result.empty or result.dropna().empty
