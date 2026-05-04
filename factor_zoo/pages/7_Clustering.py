"""Clustering page — factor grouping by correlation structure."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from factor_zoo.app import get_conn, load_all_factors, load_returns_wide, cumulative_returns_chart, CATEGORY_COLORS


@st.cache_data(ttl=3600)
def _cached_cluster(n_clusters: int):
    from factor_zoo import FactorZoo
    from factor_zoo.data.store import db_path
    fz = FactorZoo(db=str(db_path()))
    result = fz.cluster_factors(n_clusters=n_clusters)
    fz.close()
    return result


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    st.title("Factor Clustering")
    st.caption("Group factors by correlation structure using agglomerative clustering")

    n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=10)

    if st.button("Run Clustering", type="primary"):
        st.session_state["cluster_n"] = n_clusters

    n = st.session_state.get("cluster_n", n_clusters)

    with st.spinner(f"Clustering {len(load_all_factors())} factors into {n} groups..."):
        result = _cached_cluster(n)

    st.subheader("Correlation Heatmap (sorted by cluster)")
    st.plotly_chart(result.plot(), use_container_width=True)

    st.subheader("Cluster Assignments")
    cluster_rows = []
    for cluster_id, info in sorted(result.clusters.items()):
        members = info.get("factors", [])
        cluster_rows.append({
            "Cluster": cluster_id,
            "Size": len(members),
            "Factors": ", ".join(members[:10]) + ("…" if len(members) > 10 else ""),
        })
    st.dataframe(pd.DataFrame(cluster_rows), hide_index=True, use_container_width=True)

    st.subheader("Drill Into Cluster")
    cluster_ids = sorted(result.clusters.keys())
    selected_cluster = st.selectbox("Select cluster to explore", options=cluster_ids)
    if selected_cluster:
        members = result.clusters[selected_cluster].get("factors", [])
        st.caption(f"{len(members)} factors in this cluster")
        wide = load_returns_wide(tuple(members[:8]))
        if not wide.empty:
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = list(CATEGORY_COLORS.values())
            for i, col in enumerate(wide.columns):
                s = wide[col].dropna()
                if s.empty:
                    continue
                cum = (1 + s).cumprod()
                cum = cum / cum.iloc[0]
                fig.add_trace(go.Scatter(
                    x=cum.index, y=cum.values,
                    name=col, mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig.update_layout(
                title=f"Cluster {selected_cluster} — Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Normalized (1.0 at start)",
                hovermode="x unified",
                height=400,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)


main()
