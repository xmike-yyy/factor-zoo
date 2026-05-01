"""Tests for factor_zoo/analytics/cluster.py"""
import numpy as np
import pandas as pd
import pytest

from factor_zoo.analytics.cluster import cluster_factors, ClusterResult


def _wide_for_clustering(n_months=120, n_factors=9, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    data = rng.normal(0.008, 0.04, (n_months, n_factors))
    cols = [f"F{i}" for i in range(n_factors)]
    return pd.DataFrame(data, index=idx, columns=cols)


class TestClusterFactors:
    def test_returns_cluster_result(self):
        df = _wide_for_clustering()
        result = cluster_factors(df, n_clusters=3)
        assert isinstance(result, ClusterResult)

    def test_assignments_covers_all_factors(self):
        df = _wide_for_clustering(n_factors=9)
        result = cluster_factors(df, n_clusters=3)
        assert set(result.assignments.keys()) == set(df.columns)

    def test_n_clusters_respected(self):
        df = _wide_for_clustering(n_factors=9)
        result = cluster_factors(df, n_clusters=3)
        n_unique = len(set(result.assignments.values()))
        assert n_unique == 3

    def test_cluster_dict_keys_present(self):
        df = _wide_for_clustering(n_factors=6)
        result = cluster_factors(df, n_clusters=2)
        for key, val in result.clusters.items():
            assert "label" in val
            assert "factors" in val
            assert isinstance(val["factors"], list)

    def test_every_factor_in_some_cluster(self):
        df = _wide_for_clustering(n_factors=6)
        result = cluster_factors(df, n_clusters=2)
        all_in_clusters = [f for v in result.clusters.values() for f in v["factors"]]
        assert set(all_in_clusters) == set(df.columns)

    def test_plot_returns_figure(self):
        import plotly.graph_objects as go
        df = _wide_for_clustering()
        result = cluster_factors(df, n_clusters=3)
        fig = result.plot()
        assert isinstance(fig, go.Figure)
