"""Decay page — rolling Sharpe and publication decay for individual factors."""
import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from factor_zoo.app import get_conn, load_all_factors, fmt_f
from factor_zoo import FactorZoo


@st.cache_data(ttl=3600)
def _cached_decay(factor_id: str):
    conn = get_conn()
    if conn is None:
        return None
    from factor_zoo.data.store import db_path
    fz = FactorZoo(db=str(db_path()))
    result = fz.decay(factor_id)
    fz.close()
    return result


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()
    st.title("Factor Decay Analysis")
    st.caption("Rolling Sharpe ratio and publication decay for individual factors")

    all_ids = sorted(factors_df["id"].dropna().tolist())
    factor_id = st.selectbox("Select factor", options=all_ids, index=all_ids.index("Mom12m") if "Mom12m" in all_ids else 0)

    compare_ids = st.multiselect(
        "Compare with (up to 3 additional factors)",
        options=[x for x in all_ids if x != factor_id],
        max_selections=3,
    )

    if st.button("Run Decay Analysis", type="primary"):
        st.session_state["decay_factor"] = factor_id
        st.session_state["decay_compare"] = compare_ids

    target = st.session_state.get("decay_factor", factor_id)
    compare = st.session_state.get("decay_compare", [])

    result = _cached_decay(target)
    if result is None:
        return

    st.subheader(f"Rolling Sharpe — {target}")
    if result.half_life is None:
        st.info("Insufficient post-publication data to estimate half-life (need ≥ 24 months).")

    fig = result.plot()
    if compare:
        import plotly.graph_objects as go
        compare_colors = ["#FF9800", "#4CAF50", "#9C27B0"]
        for comp_id, color in zip(compare, compare_colors):
            r = _cached_decay(comp_id)
            if r is not None:
                fig.add_trace(go.Scatter(
                    x=r.rolling_sharpe.index,
                    y=r.rolling_sharpe.values,
                    name=f"{comp_id} (36m Sharpe)",
                    line=dict(color=color, dash="dash"),
                ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decay Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pre-pub Sharpe", fmt_f(result.pre_pub_sharpe))
    col2.metric("Post-pub Sharpe", fmt_f(result.post_pub_sharpe))
    col3.metric("Half-life (months)", fmt_f(result.half_life) if result.half_life else "N/A")
    col4.metric("Decay Rate λ", fmt_f(result.decay_rate, 4) if result.decay_rate else "N/A")


main()
