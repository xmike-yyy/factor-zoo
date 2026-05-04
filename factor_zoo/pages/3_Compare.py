"""Compare page — multi-factor side-by-side analysis."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from factor_zoo.app import (
    get_conn, load_all_factors, load_returns_wide,
    ALL_CATEGORIES, CATEGORY_LABELS, MAX_COMPARE,
    fmt_pct, fmt_f, multi_cumulative_chart, correlation_heatmap,
)
from factor_zoo.analytics.correlation import correlation_matrix


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
