"""Exposure Analyzer — OLS factor loading for user-provided return series."""
import sys
import io
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import streamlit as st

from factor_zoo.app import get_conn, load_all_factors, fmt_f


def _parse_returns_csv(uploaded) -> pd.Series | None:
    """Parse uploaded CSV to a monthly return Series. Auto-detects percent vs decimal."""
    try:
        df = pd.read_csv(uploaded, parse_dates=["date"])
        df = df.rename(columns={"date": "date", "return": "return", "ret": "return", "r": "return"})
        if "date" not in df.columns or "return" not in df.columns:
            st.error("CSV must have columns 'date' and 'return'.")
            return None
        s = df.set_index("date")["return"].sort_index()
        s.index = pd.to_datetime(s.index)
        if s.dropna().abs().mean() > 0.1:
            s = s / 100.0
        return s
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return None


def main():
    conn = get_conn()
    if conn is None:
        st.error("Database not found.")
        return

    factors_df = load_all_factors()
    st.title("Exposure Analyzer")
    st.caption("Decompose any return series into academic factor loadings via OLS regression")

    st.subheader("Upload Return Series")
    uploaded = st.file_uploader(
        "CSV with columns 'date' (YYYY-MM-DD) and 'return' (decimal or percent)",
        type=["csv"],
    )

    returns = None
    if uploaded:
        returns = _parse_returns_csv(uploaded)
        if returns is not None:
            st.success(f"Loaded {len(returns)} monthly returns ({returns.index.min().date()} → {returns.index.max().date()})")
    else:
        st.info("Upload a CSV, or paste returns in the text box below.")
        pasted = st.text_area("Paste CSV (date,return)", placeholder="2010-01-31,0.012\n2010-02-28,-0.003\n...")
        if pasted.strip():
            try:
                returns = _parse_returns_csv(io.StringIO(pasted))
            except Exception:
                pass

    if returns is None or returns.empty:
        return

    all_ids = sorted(factors_df["id"].dropna().tolist())
    selected_factors = st.multiselect(
        "Restrict to specific factors (leave empty for automatic top-10 by |t-stat|)",
        options=all_ids,
    )
    top_n = st.slider("Auto-select top N factors", 5, 30, 10, disabled=bool(selected_factors))

    if st.button("Run Exposure Analysis", type="primary"):
        with st.spinner("Running OLS regression..."):
            from factor_zoo import FactorZoo
            from factor_zoo.data.store import db_path
            fz = FactorZoo(db=str(db_path()))
            result = fz.exposure(
                returns,
                factors=selected_factors if selected_factors else None,
                top_n=top_n,
            )
            fz.close()

        overlap = result.overlap_months
        st.info(f"Using {overlap} months of overlapping data.")
        if overlap < 36:
            st.warning("Fewer than 36 months of overlap — results may be unreliable.")

        st.subheader("Factor Loadings")
        st.plotly_chart(result.plot(), use_container_width=True)

        st.subheader("Regression Details")
        table = pd.DataFrame({
            "Factor": list(result.loadings.keys()),
            "Loading": [round(v, 4) for v in result.loadings.values()],
            "t-stat": [round(result.t_stats[f], 2) for f in result.loadings],
            "p-value": [round(result.p_values[f], 4) for f in result.loadings],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("R²", f"{result.r_squared:.3f}")
        col2.metric("Alpha (ann.)", fmt_f(result.alpha * 12 * 100) + "% / yr")


main()
