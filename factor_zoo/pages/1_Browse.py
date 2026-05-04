"""Browse page — searchable factor grid."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from factor_zoo.app import (
    get_conn, load_all_factors, load_returns,
    CATEGORY_COLORS, CATEGORY_LABELS, ALL_CATEGORIES,
    fmt_pct, fmt_f, category_badge, short_history_badge,
)


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found. Run `uv run python scripts/build_db.py` first.")
        return

    factors_df = load_all_factors()
    st.title("Factor Zoo Browser")
    st.caption("200+ published equity factors from academic asset pricing research")

    all_ids = sorted(factors_df["id"].dropna().tolist())

    with st.sidebar:
        st.header("Filters")
        pinned_ids = st.multiselect(
            "Factor (type to search)", options=all_ids, placeholder="e.g. BM, Mom12m, GP …"
        )
        keyword = st.text_input("Keyword search", placeholder="e.g. momentum, fama, accruals")
        selected_cats = st.multiselect(
            "Category", options=ALL_CATEGORIES,
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
        )
        min_sharpe = st.slider("Min Sharpe ratio", -1.0, 3.0, -1.0, step=0.1)
        min_years = st.slider("Min sample years", 0, 40, 0, step=5)
        source_filter = st.selectbox("Source", ["All", "osap", "french"])

    df = factors_df.copy()
    if pinned_ids:
        df = df[df["id"].isin(pinned_ids)]
    elif keyword:
        kw = keyword.lower()
        mask = (
            df["id"].str.lower().str.contains(kw, na=False)
            | df["category"].str.lower().str.contains(kw, na=False)
            | df["subcategory"].str.lower().str.contains(kw, na=False)
            | df["authors"].str.lower().str.contains(kw, na=False)
            | df["description"].str.lower().str.contains(kw, na=False)
        )
        df = df[mask]
    if selected_cats:
        df = df[df["category"].isin(selected_cats)]
    if min_sharpe > -1.0:
        df = df[df["sharpe"].fillna(-999) >= min_sharpe]
    if min_years > 0:
        df = df[
            (pd.to_datetime(df["sample_end"]) - pd.to_datetime(df["sample_start"]))
            .dt.days / 365.25 >= min_years
        ]
    if source_filter != "All":
        df = df[df["source"] == source_filter]

    st.caption(f"{len(df)} factor(s) matching filters")
    if df.empty:
        st.info("No factors match the current filters.")
        return

    cols = st.columns(3)
    for i, (_, row) in enumerate(df.sort_values("sharpe", ascending=False).iterrows()):
        with cols[i % 3]:
            warning = short_history_badge() if row.get("has_short_history") else ""
            badge = category_badge(row.get("category", "other"))
            st.markdown(
                f"""
                <div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;
                            margin-bottom:12px;background:#fafafa;color:#1a1a1a">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    {badge} {warning}
                  </div>
                  <div style="font-size:1rem;font-weight:700;margin:6px 0 2px;color:#1a1a1a">{row['id']}</div>
                  <div style="font-size:0.8rem;color:#444;margin-bottom:8px">
                    Sharpe: <b>{fmt_f(row.get('sharpe'))}</b> &nbsp;|&nbsp;
                    t-stat: <b>{fmt_f(row.get('t_stat'))}</b>
                  </div>
                  <div style="font-size:0.75rem;color:#666">
                    {str(row.get('sample_start', ''))[:4]}–{str(row.get('sample_end', ''))[:4]}
                    &nbsp;·&nbsp; {row.get('source', '').upper()}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View details", key=f"detail_{row['id']}"):
                st.session_state["detail_id"] = row["id"]
                st.switch_page("pages/2_Detail.py")


main()
