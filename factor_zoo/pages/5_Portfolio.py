"""Portfolio Builder — multi-factor portfolio construction and backtesting."""
import sys
import warnings
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factor_zoo.app import get_conn, load_all_factors, load_returns_wide, fmt_pct, fmt_f, MAX_COMPARE
from factor_zoo import FactorZoo
from factor_zoo.data.store import db_path


def _build_portfolio(factor_ids: list[str], method: str, custom_weights: list[float] | None):
    from factor_zoo.data.store import read_returns_wide
    conn = get_conn()
    wide = read_returns_wide(conn, factor_ids)
    wide.index = pd.to_datetime(wide.index)
    from factor_zoo.analytics.portfolio import construct_portfolio
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = construct_portfolio(wide, method=method, weights=custom_weights)
    short_warning = any("< 60" in str(w.message) for w in caught)
    return result, short_warning


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()
    st.title("Portfolio Builder")
    st.caption("Construct and backtest multi-factor long-short portfolios")

    all_ids = sorted(factors_df["id"].dropna().tolist())
    selected = st.multiselect(
        "Select factors (2–10)",
        options=all_ids,
        default=[x for x in ["BM", "Mom12m", "GrossProfit"] if x in all_ids],
        max_selections=10,
    )

    method = st.radio(
        "Weighting method",
        ["Equal", "Risk Parity", "Max Sharpe", "Custom"],
        horizontal=True,
    )
    method_map = {"Equal": "equal", "Risk Parity": "risk_parity", "Max Sharpe": "max_sharpe", "Custom": "custom"}

    custom_weights = None
    if method == "Custom" and selected:
        st.markdown("**Set weights (must sum to 1.0):**")
        cols = st.columns(len(selected))
        raw_weights = [
            cols[i].number_input(fid, min_value=0.0, max_value=1.0, value=round(1.0 / len(selected), 3), step=0.01, key=f"w_{fid}")
            for i, fid in enumerate(selected)
        ]
        total = sum(raw_weights)
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.3f} — must equal 1.0.")
        else:
            custom_weights = raw_weights

    if len(selected) < 2:
        st.info("Select at least 2 factors to build a portfolio.")
        return

    if st.button("Construct Portfolio", type="primary"):
        with st.spinner("Building portfolio..."):
            result, short_warning = _build_portfolio(selected, method_map[method], custom_weights)

        if short_warning:
            st.warning("Portfolio intersection is < 60 months — statistics may be unreliable.")

        st.subheader("Portfolio Cumulative Return")
        port_cum = (1 + result.returns.dropna()).cumprod()
        eq_wide = load_returns_wide(tuple(selected))
        eq_benchmark = eq_wide.mean(axis=1).dropna()
        eq_cum = (1 + eq_benchmark).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, name=f"{method} Portfolio", line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=eq_cum.index, y=eq_cum.values, name="Equal-weight Benchmark", line=dict(color="#94a3b8", width=1, dash="dash")))
        fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (base=1)", hovermode="x unified", height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Statistics")
        stats = result.stats
        stat_df = pd.DataFrame([
            ("Ann. Return", fmt_pct(stats.get("ann_return"))),
            ("Ann. Volatility", fmt_pct(stats.get("ann_vol"))),
            ("Sharpe Ratio", fmt_f(stats.get("sharpe"))),
            ("Max Drawdown", fmt_pct(stats.get("max_drawdown"))),
            ("t-Statistic", fmt_f(stats.get("t_stat"))),
        ], columns=["Metric", "Portfolio"])
        st.dataframe(stat_df, hide_index=True, use_container_width=True)

        st.subheader("Factor Weights")
        wt_fig = go.Figure(go.Bar(
            x=list(result.weights.keys()),
            y=[round(v * 100, 1) for v in result.weights.values()],
            marker_color="#2563eb",
        ))
        wt_fig.update_layout(yaxis_title="Weight (%)", height=300, template="plotly_white")
        st.plotly_chart(wt_fig, use_container_width=True)

        st.subheader("Per-Factor Statistics")
        st.dataframe(result.factor_stats, use_container_width=True)

        csv = result.returns.to_frame(name="portfolio_return").to_csv()
        st.download_button("Download Returns CSV", csv, file_name="portfolio_returns.csv", mime="text/csv")


main()
