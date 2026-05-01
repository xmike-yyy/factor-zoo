"""Tests for factor_zoo/analytics/stats.py.

All return series use decimals (0.01 = 1%) to match the storage convention.
"""
import math
import numpy as np
import pandas as pd
import pytest

from factor_zoo.analytics.stats import (
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
    t_statistic,
    pre_post_sharpe,
    compute_all_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _series(values, start="2000-01-31", freq="ME") -> pd.Series:
    """Build a monthly pd.Series with a DatetimeIndex."""
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _flat(monthly_ret: float, n: int = 60) -> pd.Series:
    """All-constant monthly return series."""
    return _series([monthly_ret] * n)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:
    def test_flat_one_pct(self):
        # 0.01/month → 0.12/year
        result = annualized_return(_flat(0.01))
        assert abs(result - 0.12) < 1e-10

    def test_flat_zero(self):
        assert annualized_return(_flat(0.0)) == pytest.approx(0.0)

    def test_negative(self):
        result = annualized_return(_flat(-0.005))
        assert result == pytest.approx(-0.06)

    def test_empty_series(self):
        assert math.isnan(annualized_return(pd.Series(dtype=float)))

    def test_all_nan(self):
        assert math.isnan(annualized_return(_series([float("nan")] * 12)))

    def test_mixed_nan_ignored(self):
        # 11 NaN + 12 clean months at 0.01 → mean is still 0.01
        vals = [float("nan")] * 11 + [0.01] * 12
        result = annualized_return(_series(vals))
        assert result == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# annualized_vol
# ---------------------------------------------------------------------------

class TestAnnualizedVol:
    def test_zero_variance(self):
        # Constant series has zero std
        assert annualized_vol(_flat(0.01)) == pytest.approx(0.0, abs=1e-10)

    def test_known_vol(self):
        # std of 0.02 monthly → 0.02 * sqrt(12) annualized
        rng = np.random.default_rng(42)
        vals = rng.normal(0.01, 0.02, 120)
        result = annualized_vol(_series(vals.tolist()))
        expected = float(np.std(vals, ddof=1) * math.sqrt(12))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_element(self):
        assert math.isnan(annualized_vol(_series([0.01])))

    def test_empty_series(self):
        assert math.isnan(annualized_vol(pd.Series(dtype=float)))


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_sharpe(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0.01, 0.04, 120)
        s = _series(vals.tolist())
        result = sharpe_ratio(s)
        expected = annualized_return(s) / annualized_vol(s)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_vol_returns_nan(self):
        # Non-zero mean but zero variance → Sharpe is undefined (not inf)
        assert math.isnan(sharpe_ratio(_flat(0.01)))

    def test_empty_series(self):
        assert math.isnan(sharpe_ratio(pd.Series(dtype=float)))


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotone_up_no_drawdown(self):
        # Always positive returns → drawdown is 0
        result = max_drawdown(_flat(0.01))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # +100%, -50%: peak = 2.0, trough = 1.0 → drawdown = -0.50
        s = _series([1.0, -0.5])
        result = max_drawdown(s)
        assert result == pytest.approx(-0.5, rel=1e-6)

    def test_single_down(self):
        s = _series([-0.3])
        result = max_drawdown(s)
        assert result == pytest.approx(-0.3, rel=1e-6)

    def test_empty_series(self):
        assert math.isnan(max_drawdown(pd.Series(dtype=float)))

    def test_negative_is_negative(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(0.005, 0.05, 120).tolist()
        result = max_drawdown(_series(vals))
        assert result <= 0.0


# ---------------------------------------------------------------------------
# t_statistic
# ---------------------------------------------------------------------------

class TestTStatistic:
    def test_known_values(self):
        # mean=0.01, std=0.04, n=100 → t = 0.01 / (0.04/10) = 2.5
        vals = [0.01] * 100
        # Perturb to get exact std; easier to just check formula algebraically
        # Use a series where we control mean/std exactly
        base = [0.01 + (0.04 if i % 2 == 0 else -0.04) for i in range(100)]
        s = pd.Series(base)
        result = t_statistic(s)
        expected = s.mean() / (s.std() / math.sqrt(len(s)))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_element(self):
        assert math.isnan(t_statistic(_series([0.01])))

    def test_empty_series(self):
        assert math.isnan(t_statistic(pd.Series(dtype=float)))

    def test_sign(self):
        # Positive mean → positive t-stat
        assert t_statistic(_flat(0.01, 60)) > 0
        # Negative mean → negative t-stat
        assert t_statistic(_flat(-0.01, 60)) < 0


# ---------------------------------------------------------------------------
# pre_post_sharpe
# ---------------------------------------------------------------------------

class TestPrePostSharpe:
    def _spanning_series(self) -> pd.Series:
        """20 years of data (1990–2009) for splitting at 2000."""
        idx = pd.date_range("1990-01-31", periods=240, freq="ME")
        rng = np.random.default_rng(99)
        vals = rng.normal(0.01, 0.04, 240)
        return pd.Series(vals, index=idx)

    def test_split_is_correct(self):
        s = self._spanning_series()
        pre, post = pre_post_sharpe(s, pub_year=2000)
        # Verify pre uses only pre-2000 data
        pre_manual = sharpe_ratio(s[s.index.year < 2000])
        post_manual = sharpe_ratio(s[s.index.year >= 2000])
        assert pre == pytest.approx(pre_manual, rel=1e-6)
        assert post == pytest.approx(post_manual, rel=1e-6)

    def test_no_pub_year_returns_nan(self):
        s = self._spanning_series()
        pre, post = pre_post_sharpe(s, pub_year=None)
        assert math.isnan(pre) and math.isnan(post)

    def test_non_datetime_index_returns_nan(self):
        s = pd.Series([0.01] * 60)  # integer index
        pre, post = pre_post_sharpe(s, pub_year=2000)
        assert math.isnan(pre) and math.isnan(post)

    def test_all_pre_no_post(self):
        # pub_year is after all data → post should be NaN (empty series)
        # Use a varied series so pre-Sharpe is well-defined (non-zero vol)
        rng = np.random.default_rng(7)
        idx = pd.date_range("1990-01-31", periods=60, freq="ME")
        s = pd.Series(rng.normal(0.01, 0.04, 60), index=idx)
        pre, post = pre_post_sharpe(s, pub_year=2050)
        assert not math.isnan(pre)
        assert math.isnan(post)


# ---------------------------------------------------------------------------
# compute_all_stats
# ---------------------------------------------------------------------------

class TestComputeAllStats:
    def _good_series(self) -> pd.Series:
        rng = np.random.default_rng(42)
        idx = pd.date_range("1990-01-31", periods=360, freq="ME")
        return pd.Series(rng.normal(0.008, 0.04, 360), index=idx)

    def test_returns_all_keys(self):
        s = self._good_series()
        result = compute_all_stats(s, pub_year=2000)
        expected_keys = {
            "ann_return", "ann_vol", "sharpe", "max_drawdown",
            "t_stat", "pre_pub_sharpe", "post_pub_sharpe", "has_short_history",
        }
        assert set(result.keys()) == expected_keys

    def test_finite_values_on_good_series(self):
        s = self._good_series()
        result = compute_all_stats(s, pub_year=2000)
        for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat"):
            assert math.isfinite(result[key]), f"{key} should be finite"

    def test_has_short_history_false_for_long_series(self):
        s = self._good_series()  # 360 months >> 60
        result = compute_all_stats(s)
        assert result["has_short_history"] is False

    def test_has_short_history_true_for_short_series(self):
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        s = pd.Series([0.01] * 24, index=idx)
        result = compute_all_stats(s)
        assert result["has_short_history"] is True

    def test_no_pub_year_pre_post_nan(self):
        s = self._good_series()
        result = compute_all_stats(s, pub_year=None)
        assert math.isnan(result["pre_pub_sharpe"])
        assert math.isnan(result["post_pub_sharpe"])

    def test_empty_series_graceful(self):
        result = compute_all_stats(pd.Series(dtype=float))
        for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat"):
            assert math.isnan(result[key]), f"{key} should be NaN for empty input"
