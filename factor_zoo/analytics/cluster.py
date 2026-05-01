"""Correlation-based hierarchical clustering of factors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go

from factor_zoo.analytics.correlation import correlation_matrix


@dataclass
class ClusterResult:
    assignments: dict[str, int]
    clusters: dict[str, dict]
    corr_matrix: pd.DataFrame

    def plot(self) -> go.Figure:
        order = sorted(self.assignments.keys(), key=lambda f: self.assignments[f])
        mat = self.corr_matrix.loc[order, order]
        fig = go.Figure(data=go.Heatmap(
            z=mat.values,
            x=list(mat.columns),
            y=list(mat.index),
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
        ))
        fig.update_layout(
            title="Factor Correlation — Sorted by Cluster",
            height=600,
            template="plotly_white",
        )
        return fig


def cluster_factors(
    wide_df: pd.DataFrame,
    n_clusters: int = 10,
    category_map: Optional[dict[str, str]] = None,
) -> ClusterResult:
    """Group factors by pairwise correlation using agglomerative clustering."""
    df = wide_df.dropna(how="all")
    factors = list(df.columns)

    corr = correlation_matrix(df)
    distance = 1.0 - corr.abs().values
    np.fill_diagonal(distance, 0.0)

    n_clusters_actual = min(n_clusters, len(factors))
    model = AgglomerativeClustering(
        n_clusters=n_clusters_actual,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distance)

    assignments = dict(zip(factors, labels.tolist()))

    clusters: dict[str, dict] = {}
    for cluster_id in range(n_clusters_actual):
        members = [f for f, c in assignments.items() if c == cluster_id]
        label = _auto_label(members, cluster_id, category_map)
        clusters[f"cluster_{cluster_id}"] = {"label": label, "factors": members}

    return ClusterResult(
        assignments=assignments,
        clusters=clusters,
        corr_matrix=corr,
    )


def _auto_label(
    members: list[str],
    cluster_id: int,
    category_map: Optional[dict[str, str]],
) -> str:
    if not category_map or not members:
        return f"cluster_{cluster_id}"
    cats = [category_map.get(f) for f in members if category_map.get(f)]
    if not cats:
        return f"cluster_{cluster_id}"
    from collections import Counter
    most_common, count = Counter(cats).most_common(1)[0]
    if count / len(members) >= 0.4:
        return most_common.replace("_", " ").title()
    return f"cluster_{cluster_id}"
