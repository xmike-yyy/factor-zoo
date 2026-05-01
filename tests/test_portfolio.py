"""Tests for factor_zoo/analytics/portfolio.py"""
import warnings
import numpy as np
import pandas as pd
import pytest

from factor_zoo.analytics.portfolio import construct_portfolio, PortfolioResult


def _wide(n_months=120, n_factors=3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    data = rng.normal(0.008, 0.04, (n_months, n_factors))
    cols = [f"F{i}" for i in range(n_factors)]
    return pd.DataFrame(data, index=idx, columns=cols)


class TestEqualWeight:
    def test_returns_portfolio_result(self):
        df = _wide()
        result = construct_portfolio(df, method="equal")
        assert isinstance(result, PortfolioResult)

    def test_equal_weights_sum_to_one(self):
        df = _wide()
        result = construct_portfolio(df, method="equal")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    def test_equal_weights_are_equal(self):
        df = _wide(n_factors=4)
        result = construct_portfolio(df, method="equal")
        weights = list(result.weights.values())
        assert all(abs(w - 0.25) < 1e-9 for w in weights)

    def test_returns_has_correct_length(self):
        df = _wide(n_months=120)
        result = construct_portfolio(df, method="equal")
        assert len(result.returns) == 120

    def test_factor_stats_has_all_factors(self):
        df = _wide(n_factors=3)
        result = construct_portfolio(df, method="equal")
        assert set(result.factor_stats.index) == {"F0", "F1", "F2"}

    def test_stats_dict_has_expected_keys(self):
        df = _wide()
        result = construct_portfolio(df, method="equal")
        assert "sharpe" in result.stats
        assert "ann_return" in result.stats


class TestCustomWeights:
    def test_custom_weights_preserved(self):
        df = _wide(n_factors=2)
        result = construct_portfolio(df, method="custom", weights=[0.3, 0.7])
        assert abs(result.weights["F0"] - 0.3) < 1e-9
        assert abs(result.weights["F1"] - 0.7) < 1e-9

    def test_weights_not_summing_to_one_raises(self):
        df = _wide(n_factors=2)
        with pytest.raises(ValueError, match="sum to 1"):
            construct_portfolio(df, method="custom", weights=[0.3, 0.5])

    def test_wrong_length_weights_raises(self):
        df = _wide(n_factors=3)
        with pytest.raises(ValueError):
            construct_portfolio(df, method="custom", weights=[0.5, 0.5])


class TestMaxSharpe:
    def test_returns_portfolio_result(self):
        df = _wide()
        result = construct_portfolio(df, method="max_sharpe")
        assert isinstance(result, PortfolioResult)

    def test_weights_sum_to_one(self):
        df = _wide()
        result = construct_portfolio(df, method="max_sharpe")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_weights_non_negative(self):
        df = _wide()
        result = construct_portfolio(df, method="max_sharpe")
        assert all(w >= -1e-9 for w in result.weights.values())


class TestRiskParity:
    def test_returns_portfolio_result(self):
        df = _wide()
        result = construct_portfolio(df, method="risk_parity")
        assert isinstance(result, PortfolioResult)

    def test_weights_sum_to_one(self):
        df = _wide()
        result = construct_portfolio(df, method="risk_parity")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    def test_lower_vol_gets_higher_weight(self):
        idx = pd.date_range("2000-01-31", periods=120, freq="ME")
        rng = np.random.default_rng(1)
        data = np.column_stack([
            rng.normal(0.008, 0.01, 120),
            rng.normal(0.008, 0.05, 120),
        ])
        df = pd.DataFrame(data, index=idx, columns=["F0", "F1"])
        result = construct_portfolio(df, method="risk_parity")
        assert result.weights["F0"] > result.weights["F1"]


class TestIntersection:
    def test_warns_if_fewer_than_60_months_of_overlap(self):
        idx = pd.date_range("2000-01-31", periods=30, freq="ME")
        rng = np.random.default_rng(2)
        df = pd.DataFrame(rng.normal(0.008, 0.04, (30, 2)), index=idx, columns=["F0", "F1"])
        with pytest.warns(UserWarning, match="60"):
            construct_portfolio(df, method="equal")
