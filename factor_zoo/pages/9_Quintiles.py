"""Quintile Analysis page — per-quintile returns and spread analysis."""
import sys
import math
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factor_zoo.app import get_conn, load_all_factors
from factor_zoo import FactorZoo


@st.cache_data(ttl=3600)
def _cached_quintile(factor_id: str):
    conn = get_conn()
    if conn is None:
        return None
    from factor_zoo.data.store import db_path
    fz = FactorZoo(db=str(db_path()))
    try:
        result = fz.quintile_analysis(factor_id)
    except KeyError:
        result = None  # French factor or missing data
    fz.close()
    return result


@st.cache_data(ttl=3600)
def _cached_spread(factor_id: str) -> "pd.Series | None":
    conn = get_conn()
    if conn is None:
        return None
    from factor_zoo.data.store import db_path
    fz = FactorZoo(db=str(db_path()))
    try:
        spread = fz.quintile_spread(factor_id)
    except KeyError:
        spread = None
    fz.close()
    return spread


def _quintile_stats(quintiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        s = quintiles[q].dropna()
        ann_ret = (1 + s).prod() ** (12 / len(s)) - 1 if len(s) > 0 else float("nan")
        ann_vol = s.std() * math.sqrt(12) if len(s) > 1 else float("nan")
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        rows.append({
            "Quintile": q.upper(),
            "Ann. Return": f"{ann_ret:.1%}" if not math.isnan(ann_ret) else "N/A",
            "Ann. Vol": f"{ann_vol:.1%}" if not math.isnan(ann_vol) else "N/A",
            "Sharpe": f"{sharpe:.2f}" if not math.isnan(sharpe) else "N/A",
        })
    return pd.DataFrame(rows)


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()
    st.title("Quintile Analysis")
    st.caption("Per-quintile returns and long-short spread for OSAP factors")

    all_ids = sorted(factors_df["id"].dropna().tolist())
    factor_id = st.selectbox(
        "Select factor",
        options=all_ids,
        index=all_ids.index("Mom12m") if "Mom12m" in all_ids else 0,
    )

    if st.button("Calculate", type="primary"):
        st.session_state["quintile_factor"] = factor_id

    target = st.session_state.get("quintile_factor")
    if target is None:
        st.stop()

    result = _cached_quintile(target)
    if result is None:
        st.info("Quintile data not available for this factor.")
        st.stop()

    # Cumulative return chart
    st.plotly_chart(result.plot(), use_container_width=True)

    # Key metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Monotonicity Score",
        f"{result.monotonicity_score:.1%}",
        delta_color="off",
    )
    col2.metric(
        "Spread Sharpe",
        f"{result.spread_sharpe:.2f}",
        delta_color="off",
    )
    # L/S Sharpe from factors table
    factors_row = factors_df[factors_df["id"] == target]
    ls_sharpe = factors_row.iloc[0]["sharpe"] if not factors_row.empty else float("nan")
    ls_sharpe_val = float(ls_sharpe) if ls_sharpe is not None else float("nan")
    col3.metric(
        "L/S Sharpe",
        f"{ls_sharpe_val:.2f}" if not math.isnan(ls_sharpe_val) else "N/A",
        delta_color="off",
    )

    # Quintile stats table
    st.subheader("Quintile Statistics")
    stats_df = _quintile_stats(result.quintiles)
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # Compare spreads expander
    with st.expander("Compare quintile spreads"):
        compare_ids = st.multiselect(
            "Add factors to compare",
            options=[x for x in all_ids if x != target],
            max_selections=3,
            key="quintile_compare",
        )
        if compare_ids:
            fig = go.Figure()
            # Add current factor's spread
            cum_main = (1 + result.spread.dropna()).cumprod()
            fig.add_trace(go.Scatter(
                x=cum_main.index,
                y=cum_main.values,
                name=target,
                mode="lines",
            ))
            # Add comparison factors
            for cid in compare_ids:
                sp = _cached_spread(cid)
                if sp is not None and not sp.empty:
                    cum = (1 + sp.dropna()).cumprod()
                    fig.add_trace(go.Scatter(
                        x=cum.index,
                        y=cum.values,
                        name=cid,
                        mode="lines",
                    ))
            fig.update_layout(
                title="Q5–Q1 Spread Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (base=1)",
                template="plotly_white",
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Select factors above to compare their quintile spreads.")


main()
