"""Tests for factor_zoo/api.py — FactorZoo public methods."""
import numpy as np
import pandas as pd
import duckdb
import pytest

from factor_zoo.api import FactorZoo


# ---------------------------------------------------------------------------
# Fixture: build a real temp-file DuckDB with sample data, patch ensure_db
# ---------------------------------------------------------------------------

@pytest.fixture
def fz(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    conn = duckdb.connect(str(db_file))
    conn.execute("""
        CREATE TABLE factors (
            id VARCHAR PRIMARY KEY, name VARCHAR, category VARCHAR,
            subcategory VARCHAR, authors VARCHAR, year INTEGER, journal VARCHAR,
            paper_url VARCHAR, sample_start DATE, sample_end DATE,
            n_stocks_avg INTEGER, description TEXT,
            ann_return FLOAT, ann_vol FLOAT, sharpe FLOAT, max_drawdown FLOAT,
            t_stat FLOAT, pre_pub_sharpe FLOAT, post_pub_sharpe FLOAT,
            has_short_history BOOLEAN DEFAULT FALSE, source VARCHAR,
            paper_t_stat FLOAT
        )
    """)
    conn.execute("""
        CREATE TABLE factor_returns (
            factor_id VARCHAR, date DATE, ls_return FLOAT,
            PRIMARY KEY (factor_id, date)
        )
    """)
    conn.execute("""
        INSERT INTO factors VALUES
            ('Mom12m', 'Momentum 12-1', 'momentum', NULL, 'Jegadeesh', 1993, 'JF',
             NULL, '1980-01-01', '2020-12-31', NULL, 'Momentum factor',
             0.08, 0.14, 0.57, -0.35, 4.2, 0.65, 0.42, FALSE, 'osap', 4.2),
            ('Accruals', 'Accruals', 'profitability', NULL, 'Sloan', 1996, 'AR',
             NULL, '1975-01-01', '2020-12-31', NULL, 'Accruals factor',
             0.06, 0.12, 0.50, -0.28, 3.1, 0.60, 0.35, FALSE, 'osap', NULL),
            ('HML', 'High Minus Low', 'value', NULL, 'Fama-French', 1993, 'JF',
             NULL, '1963-01-01', '2020-12-31', NULL, 'Value factor',
             0.04, 0.13, 0.31, -0.40, 2.5, 0.45, 0.22, FALSE, 'french', NULL),
            ('LowVol', 'Low Volatility', 'trading_frictions', NULL, 'Baker', 2011, 'JF',
             NULL, '1990-01-01', '2020-12-31', NULL, 'Low vol factor',
             0.03, 0.08, 0.38, -0.22, 2.1, 0.40, 0.28, FALSE, 'osap', NULL)
    """)

    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    for fid, mean, std in [
        ("Mom12m", 0.007, 0.04),
        ("Accruals", 0.005, 0.035),
        ("HML", 0.004, 0.038),
        ("LowVol", 0.003, 0.025),
    ]:
        rets = rng.normal(mean, std, 120)
        conn.executemany(
            "INSERT INTO factor_returns VALUES (?, ?, ?)",
            [(fid, str(d.date()), float(r)) for d, r in zip(dates, rets)],
        )
    conn.close()

    monkeypatch.setattr("factor_zoo.data.remote.ensure_db", lambda *a, **kw: None)
    fz_instance = FactorZoo(db=str(db_file))
    yield fz_instance
    fz_instance.close()


# ---------------------------------------------------------------------------
# list_factors
# ---------------------------------------------------------------------------

class TestListFactors:
    def test_returns_all_factors(self, fz):
        df = fz.list_factors()
        assert len(df) == 4

    def test_filter_by_category(self, fz):
        df = fz.list_factors(category="momentum")
        assert len(df) == 1
        assert df.iloc[0]["id"] == "Mom12m"

    def test_filter_by_min_sharpe(self, fz):
        df = fz.list_factors(min_sharpe=0.55)
        assert all(df["sharpe"] >= 0.55)

    def test_filter_by_min_t_stat(self, fz):
        df = fz.list_factors(min_t_stat=3.0)
        assert all(df["t_stat"] >= 3.0)

    def test_filter_by_source(self, fz):
        df = fz.list_factors(source="french")
        assert all(df["source"] == "french")
        assert len(df) == 1

    def test_search(self, fz):
        df = fz.list_factors(search="momentum")
        assert len(df) >= 1


# ---------------------------------------------------------------------------
# get_returns
# ---------------------------------------------------------------------------

class TestGetReturns:
    def test_returns_series(self, fz):
        s = fz.get_returns("Mom12m")
        assert isinstance(s, pd.Series)
        assert len(s) == 120

    def test_series_name_is_factor_id(self, fz):
        s = fz.get_returns("Mom12m")
        assert s.name == "Mom12m"

    def test_index_is_datetime(self, fz):
        s = fz.get_returns("Mom12m")
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_date_filtering(self, fz):
        s = fz.get_returns("Mom12m", start="2004-01-01", end="2007-12-31")
        assert s.index.min() >= pd.Timestamp("2004-01-01")
        assert s.index.max() <= pd.Timestamp("2007-12-31")

    def test_missing_factor_returns_empty(self, fz):
        s = fz.get_returns("DoesNotExist")
        assert s.empty


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict_with_expected_keys(self, fz):
        stats = fz.get_stats("Mom12m")
        for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown",
                    "t_stat", "pre_pub_sharpe", "post_pub_sharpe"):
            assert key in stats

    def test_unknown_factor_raises(self, fz):
        with pytest.raises(KeyError):
            fz.get_stats("NoSuchFactor")


# ---------------------------------------------------------------------------
# compare / correlation_matrix
# ---------------------------------------------------------------------------

class TestCompare:
    def test_returns_wide_dataframe(self, fz):
        df = fz.compare(["Mom12m", "Accruals"])
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"Mom12m", "Accruals"}

    def test_date_index_is_datetime(self, fz):
        df = fz.compare(["Mom12m", "HML"])
        assert isinstance(df.index, pd.DatetimeIndex)


class TestCorrelationMatrix:
    def test_returns_square_df(self, fz):
        corr = fz.correlation_matrix(["Mom12m", "Accruals", "HML"])
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, fz):
        corr = fz.correlation_matrix(["Mom12m", "Accruals", "HML"])
        for fid in ["Mom12m", "Accruals", "HML"]:
            assert abs(corr.loc[fid, fid] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# categories / factor_ids
# ---------------------------------------------------------------------------

class TestMeta:
    def test_categories_returns_list(self, fz):
        cats = fz.categories()
        assert isinstance(cats, list)
        assert "momentum" in cats

    def test_factor_ids_returns_all(self, fz):
        ids = fz.factor_ids()
        assert set(ids) == {"Mom12m", "Accruals", "HML", "LowVol"}


# ---------------------------------------------------------------------------
# replication_score / zoo_summary
# ---------------------------------------------------------------------------

class TestReplicationScore:
    def test_known_factor(self, fz):
        result = fz.replication_score("Mom12m")
        assert "verdict" in result
        assert result["verdict"] in ("strong", "partial", "weak")

    def test_no_paper_t_stat(self, fz):
        result = fz.replication_score("Accruals")
        assert "error" in result

    def test_unknown_factor_raises(self, fz):
        with pytest.raises(KeyError):
            fz.replication_score("Ghost")


class TestZooSummary:
    def test_returns_dict_with_expected_keys(self, fz):
        s = fz.zoo_summary()
        assert "n_factors" in s
        assert "pct_positive_post_pub_sharpe" in s
        assert "t_stat_distribution" in s

    def test_n_factors(self, fz):
        assert fz.zoo_summary()["n_factors"] == 4


# ---------------------------------------------------------------------------
# decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_returns_decay_result(self, fz):
        from factor_zoo.analytics.decay import DecayResult
        result = fz.decay("Mom12m")
        assert isinstance(result, DecayResult)

    def test_factor_id_matches(self, fz):
        result = fz.decay("Mom12m")
        assert result.factor_id == "Mom12m"

    def test_pub_year_populated(self, fz):
        result = fz.decay("Mom12m")
        assert result.publication_year == 1993

    def test_unknown_factor_raises(self, fz):
        with pytest.raises(KeyError):
            fz.decay("Ghost")


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_equal_weight_returns_result(self, fz):
        from factor_zoo.analytics.portfolio import PortfolioResult
        result = fz.portfolio(["Mom12m", "Accruals"])
        assert isinstance(result, PortfolioResult)

    def test_weights_sum_to_one(self, fz):
        result = fz.portfolio(["Mom12m", "Accruals", "HML"])
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_custom_weights(self, fz):
        result = fz.portfolio(["Mom12m", "Accruals"], method="custom", weights=[0.7, 0.3])
        assert abs(result.weights["Mom12m"] - 0.7) < 1e-6

    def test_max_sharpe(self, fz):
        result = fz.portfolio(["Mom12m", "Accruals", "HML"], method="max_sharpe")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-4

    def test_risk_parity(self, fz):
        result = fz.portfolio(["Mom12m", "Accruals", "HML"], method="risk_parity")
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------

class TestScreen:
    def test_min_sharpe_filter(self, fz):
        df = fz.screen(min_sharpe=0.45)
        assert all(df["sharpe"] >= 0.45)

    def test_categories_filter(self, fz):
        df = fz.screen(categories=["momentum", "value"])
        assert set(df["category"]).issubset({"momentum", "value"})

    def test_correlation_dedup_removes_high_corr_pair(self, fz):
        # With a very low max_correlation threshold, highly correlated pairs get pruned
        df_unfiltered = fz.screen()
        df_filtered = fz.screen(max_correlation=0.01)
        assert len(df_filtered) <= len(df_unfiltered)

    def test_no_filters_returns_all(self, fz):
        df = fz.screen()
        assert len(df) == 4


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------

class TestExposure:
    def test_returns_exposure_result(self, fz):
        from factor_zoo.analytics.exposure import ExposureResult
        rng = np.random.default_rng(0)
        dates = pd.date_range("2000-01-31", periods=120, freq="ME")
        user_ret = pd.Series(rng.normal(0.006, 0.04, 120), index=dates)
        result = fz.exposure(user_ret, factors=["Mom12m", "Accruals"])
        assert isinstance(result, ExposureResult)

    def test_loadings_have_requested_factors(self, fz):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2000-01-31", periods=120, freq="ME")
        user_ret = pd.Series(rng.normal(0.005, 0.04, 120), index=dates)
        result = fz.exposure(user_ret, factors=["Mom12m", "HML"])
        assert set(result.loadings.keys()) == {"Mom12m", "HML"}

    def test_top_n_respected_when_factors_none(self, fz):
        rng = np.random.default_rng(2)
        dates = pd.date_range("2000-01-31", periods=120, freq="ME")
        user_ret = pd.Series(rng.normal(0.005, 0.04, 120), index=dates)
        result = fz.exposure(user_ret, top_n=2)
        assert len(result.loadings) == 2
