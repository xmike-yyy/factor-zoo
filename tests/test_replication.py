"""Tests for factor_zoo/analytics/replication.py"""
import math
import pytest

from factor_zoo.analytics.replication import replication_score, zoo_summary


class TestReplicationScore:
    def test_known_factor_returns_dict(self, mem_conn):
        result = replication_score(mem_conn, "Mom12m")
        assert isinstance(result, dict)
        assert "paper_t_stat" in result
        assert "computed_t_stat" in result
        assert "replication_score" in result
        assert "verdict" in result

    def test_verdict_is_valid(self, mem_conn):
        result = replication_score(mem_conn, "Mom12m")
        assert result["verdict"] in ("strong", "partial", "weak")

    def test_score_is_ratio_of_t_stats(self, mem_conn):
        result = replication_score(mem_conn, "Mom12m")
        expected_score = result["computed_t_stat"] / result["paper_t_stat"]
        assert abs(result["replication_score"] - expected_score) < 0.001

    def test_verdict_strong_when_score_ge_08(self, mem_conn):
        result = replication_score(mem_conn, "Mom12m")
        if result["replication_score"] >= 0.8:
            assert result["verdict"] == "strong"

    def test_no_paper_t_stat_returns_error(self, mem_conn):
        result = replication_score(mem_conn, "Accruals")
        assert "error" in result

    def test_unknown_factor_raises(self, mem_conn):
        with pytest.raises(KeyError):
            replication_score(mem_conn, "DoesNotExist")


class TestZooSummary:
    def test_returns_dict(self, mem_conn):
        result = zoo_summary(mem_conn)
        assert isinstance(result, dict)

    def test_n_factors_is_int(self, mem_conn):
        result = zoo_summary(mem_conn)
        assert isinstance(result["n_factors"], int)
        assert result["n_factors"] == 3

    def test_expected_keys_present(self, mem_conn):
        result = zoo_summary(mem_conn)
        assert "pct_positive_post_pub_sharpe" in result
        assert "t_stat_distribution" in result
        assert "mean" in result["t_stat_distribution"]
        assert "median" in result["t_stat_distribution"]
        assert "pct_above_3" in result["t_stat_distribution"]

    def test_pct_positive_is_between_0_and_1(self, mem_conn):
        result = zoo_summary(mem_conn)
        pct = result["pct_positive_post_pub_sharpe"]
        assert 0.0 <= pct <= 1.0
