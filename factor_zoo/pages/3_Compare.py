"""Compare page — multi-factor side-by-side analysis."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factor_zoo.app import (
    get_conn, load_all_factors, load_returns_wide,
    ALL_CATEGORIES, CATEGORY_LABELS, MAX_COMPARE,
    fmt_pct, fmt_f, multi_cumulative_chart, correlation_heatmap,
)
from factor_zoo.analytics.correlation import correlation_matrix


@st.cache_data(ttl=3600)
def _cached_rolling_corr(factor_ids: tuple[str, ...], window: int) -> pd.DataFrame:
    from factor_zoo.app import load_returns_wide
    from factor_zoo.analytics.correlation import rolling_correlation
    from itertools import combinations

    wide = load_returns_wide(factor_ids)
    result = {}
    for a, b in combinations(factor_ids, 2):
        if a in wide.columns and b in wide.columns:
            result[f"{a} vs {b}"] = rolling_correlation(
                pd.Series(wide[a]), pd.Series(wide[b]), window=window
            )
    return pd.DataFrame(result)


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()
    st.title("Compare Factors")
    st.caption(f"Select up to {MAX_COMPARE} factors to compare side by side")

    all_ids = sorted(factors_df["id"].dropna().tolist())
    default = [x for x in ["BM", "Mom12m"] if x in all_ids][:2] or all_ids[:2]
    selected = st.multiselect(
        "Select factors", options=all_ids, default=default, max_selections=MAX_COMPARE
    )

    if not selected:
        st.info("Select at least one factor above.")
        return

    wide = load_returns_wide(tuple(selected))

    st.subheader("Cumulative Returns (normalized to 1.0)")
    if not wide.empty:
        st.plotly_chart(multi_cumulative_chart(wide), use_container_width=True)
    else:
        st.warning("No return data available for the selected factors.")

    if len(selected) > 1:
        st.subheader("Correlation Matrix")
        corr = correlation_matrix(wide.dropna(how="all"))
        if not corr.empty:
            st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    st.subheader("Rolling Correlation")
    if len(selected) < 2:
        st.info("Select at least 2 factors to see rolling correlations.")
    else:
        window = st.radio(
            "Window",
            [12, 24, 36],
            index=2,
            horizontal=True,
            format_func=lambda w: f"{w}m",
        )

        roll_factors = list(selected)
        if len(roll_factors) > 6:
            st.warning(
                "Rolling correlation is shown for up to 6 factors (15 pairs max). "
                "Showing first 6."
            )
            roll_factors = roll_factors[:6]

        rolling_df = _cached_rolling_corr(tuple(sorted(roll_factors)), window)

        if not rolling_df.empty:
            fig = go.Figure()
            for col in rolling_df.columns:
                fig.add_trace(go.Scatter(
                    x=rolling_df.index,
                    y=rolling_df[col],
                    mode="lines",
                    name=col,
                    hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig.update_layout(
                title=f"Rolling {window}-Month Correlation",
                xaxis_title="Date",
                yaxis_title="Correlation",
                yaxis=dict(range=[-1, 1]),
                template="plotly_white",
                margin=dict(t=40, b=40, l=60, r=20),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Pearson correlation, {window}-month rolling window")

    st.subheader("Summary Statistics")
    subset = factors_df[factors_df["id"].isin(selected)][
        ["id", "category", "ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat",
         "pre_pub_sharpe", "post_pub_sharpe", "sample_start", "sample_end"]
    ].copy()
    for col in ["ann_return", "ann_vol", "max_drawdown"]:
        subset[col] = subset[col].map(fmt_pct)
    for col in ["sharpe", "t_stat", "pre_pub_sharpe", "post_pub_sharpe"]:
        subset[col] = subset[col].map(fmt_f)
    subset.columns = ["ID", "Category", "Ann. Return", "Ann. Vol", "Sharpe",
                      "Max DD", "t-stat", "Pre Sharpe", "Post Sharpe", "Start", "End"]
    st.dataframe(subset, hide_index=True, use_container_width=True)


main()
