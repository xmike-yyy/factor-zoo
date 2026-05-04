"""Detail page — single factor deep-dive."""
import sys
import math
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from factor_zoo.app import (
    get_conn, load_all_factors, load_returns,
    CATEGORY_COLORS, fmt_pct, fmt_f, category_badge, short_history_badge,
    cumulative_returns_chart, code_snippet,
)


def _decay_bar_chart(row):
    import plotly.graph_objects as go
    pre = row.get("pre_pub_sharpe")
    post = row.get("post_pub_sharpe")

    def clean(v):
        if v is None:
            return None
        try:
            return None if math.isnan(float(v)) else float(v)
        except (TypeError, ValueError):
            return None

    pre, post = clean(pre), clean(post)
    if pre is None and post is None:
        return None

    pub_year = row.get("year")
    labels, values, colors = [], [], []
    if pre is not None:
        labels.append(f"Pre-{pub_year}" if pub_year else "Pre-publication")
        values.append(pre)
        colors.append("#4CAF50" if pre > 0 else "#F44336")
    if post is not None:
        labels.append(f"Post-{pub_year}" if pub_year else "Post-publication")
        values.append(post)
        colors.append("#4CAF50" if post > 0 else "#F44336")

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Publication Decay: Sharpe Ratio",
        yaxis_title="Sharpe Ratio",
        margin=dict(t=40, b=40, l=50, r=20),
        height=280,
        showlegend=False,
    )
    return fig


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()

    factor_id = st.session_state.get("detail_id", "")
    if not factor_id:
        st.info("No factor selected. Go to Browse and click 'View details'.")
        if st.button("Go to Browse"):
            st.switch_page("pages/1_Browse.py")
        return

    row = factors_df[factors_df["id"] == factor_id]
    if row.empty:
        st.error(f"Factor '{factor_id}' not found.")
        return
    row = row.iloc[0]

    if st.button("← Back to Browse"):
        st.switch_page("pages/1_Browse.py")

    cat = row.get("category", "other")
    col_title, col_meta = st.columns([2, 1])
    with col_title:
        st.markdown(f"<h2 style='margin-bottom:4px'>{row['id']}</h2>", unsafe_allow_html=True)
        st.markdown(
            f"{category_badge(cat)} {short_history_badge() if row.get('has_short_history') else ''}",
            unsafe_allow_html=True,
        )
        if row.get("authors"):
            st.caption(f"{row['authors']} ({row.get('year', '')})")
        if row.get("journal"):
            st.caption(f"Published in: **{row['journal']}**")
        if row.get("paper_url"):
            st.markdown(f"[View paper]({row['paper_url']})")
    with col_meta:
        st.metric("Sample Period", f"{str(row.get('sample_start', ''))[:7]} → {str(row.get('sample_end', ''))[:7]}")
        st.metric("Source", str(row.get("source", "")).upper())

    st.divider()
    returns = load_returns(factor_id)
    left, right = st.columns([3, 2])
    with left:
        log_scale = st.checkbox("Log scale", key="detail_log")
        if not returns.empty:
            st.plotly_chart(
                cumulative_returns_chart(returns, log_scale=log_scale, color=CATEGORY_COLORS.get(cat, "#2196F3")),
                use_container_width=True,
            )
        else:
            st.info("No return data available.")
    with right:
        st.subheader("Performance Stats")
        stats_df = pd.DataFrame([
            ("Ann. Return", fmt_pct(row.get("ann_return"))),
            ("Ann. Volatility", fmt_pct(row.get("ann_vol"))),
            ("Sharpe Ratio", fmt_f(row.get("sharpe"))),
            ("Max Drawdown", fmt_pct(row.get("max_drawdown"))),
            ("t-Statistic", fmt_f(row.get("t_stat"))),
        ], columns=["Metric", "Value"])
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.subheader("Publication Decay")
    fig = _decay_bar_chart(row)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Pre/post publication Sharpe not available.")

    if row.get("description"):
        st.subheader("Description")
        st.write(row["description"])

    st.subheader("Use this factor")
    st.markdown(code_snippet(factor_id))


main()
