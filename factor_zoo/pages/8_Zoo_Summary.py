"""Zoo Summary page — aggregate statistics across all factors."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from factor_zoo.app import get_conn, load_all_factors, fmt_f, fmt_pct


@st.cache_data(ttl=3600)
def _cached_zoo_summary():
    from factor_zoo import FactorZoo
    from factor_zoo.data.store import db_path
    fz = FactorZoo(db=str(db_path()))
    summary = fz.zoo_summary()
    fz.close()
    return summary


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    st.title("Zoo Summary")
    st.caption("Aggregate statistics across all factors in the database")

    with st.spinner("Loading zoo summary (this may take 20–30s on first run)..."):
        summary = _cached_zoo_summary()
        factors_df = load_all_factors()

    t_dist = summary.get("t_stat_distribution", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Factors", summary.get("n_factors", "—"))
    col2.metric("% Positive Post-pub Sharpe", fmt_pct(summary.get("pct_positive_post_pub_sharpe")))
    col3.metric("Median Replication Score", fmt_f(summary.get("median_replication_score")))
    col4.metric("Median t-stat", fmt_f(t_dist.get("median")))

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("t-Statistic Distribution")
        t_data = factors_df["t_stat"].dropna()
        fig = px.histogram(t_data, nbins=40, labels={"value": "t-stat"}, template="plotly_white")
        fig.add_vline(x=1.96, line_dash="dash", line_color="orange", annotation_text="1.96")
        fig.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="3.0")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Sharpe Ratio Distribution")
        sharpe_data = factors_df["sharpe"].dropna()
        fig2 = px.histogram(sharpe_data, nbins=40, labels={"value": "Sharpe"}, template="plotly_white")
        fig2.add_vline(x=0, line_dash="dash", line_color="gray")
        fig2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pre vs Post-Publication Sharpe")
    scatter_df = factors_df[["id", "pre_pub_sharpe", "post_pub_sharpe", "category"]].dropna()
    if not scatter_df.empty:
        fig3 = px.scatter(
            scatter_df, x="pre_pub_sharpe", y="post_pub_sharpe",
            hover_name="id", color="category",
            labels={"pre_pub_sharpe": "Pre-publication Sharpe", "post_pub_sharpe": "Post-publication Sharpe"},
            template="plotly_white", height=400,
        )
        fig3.add_shape(type="line", x0=-2, y0=-2, x1=3, y1=3, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig3, use_container_width=True)

    pairs = summary.get("most_correlated_pairs", [])
    if pairs:
        st.subheader("Most Correlated Factor Pairs (Top 20)")
        pairs_df = pd.DataFrame(pairs)
        st.dataframe(pairs_df, hide_index=True, use_container_width=True)

    independent = summary.get("most_independent_factors", [])
    if independent:
        st.subheader("Most Independent Factors (Top 10)")
        st.write("Lowest average pairwise absolute correlation:")
        cols = st.columns(min(5, len(independent)))
        for i, fid in enumerate(independent):
            cols[i % 5].metric(fid, "")


main()
