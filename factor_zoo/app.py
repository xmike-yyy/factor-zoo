"""Factor Zoo Browser — Streamlit UI.

Launch with:
    uv run streamlit run factor_zoo/app.py
"""
import sys
import math
from pathlib import Path
from typing import Optional

# Make sure the repo root is on sys.path so `factor_zoo` is importable
# regardless of how Streamlit was launched (uv run vs direct streamlit call).
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from factor_zoo.data.store import connect, db_path, read_factors, read_returns, read_returns_wide

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_COLORS: dict[str, str] = {
    "value": "#2196F3",
    "momentum": "#FF9800",
    "profitability": "#4CAF50",
    "investment": "#9C27B0",
    "intangibles": "#00BCD4",
    "trading_frictions": "#F44336",
    "other": "#9E9E9E",
}

CATEGORY_LABELS: dict[str, str] = {
    "value": "Value",
    "momentum": "Momentum",
    "profitability": "Profitability",
    "investment": "Investment",
    "intangibles": "Intangibles",
    "trading_frictions": "Trading Frictions",
    "other": "Other",
}

ALL_CATEGORIES = list(CATEGORY_LABELS.keys())
MAX_COMPARE = 6

# ---------------------------------------------------------------------------
# Cached DB connection + data helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_conn():
    path = db_path()
    if not path.exists():
        return None
    return connect(path, read_only=True)


@st.cache_data(ttl=3600)
def load_all_factors() -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    return read_factors(conn)


@st.cache_data(ttl=3600)
def load_returns(factor_id: str) -> pd.Series:
    conn = get_conn()
    df = read_returns(conn, factor_id)
    if df.empty:
        return pd.Series(dtype=float, name=factor_id)
    s = df.set_index("date")["ls_return"]
    s.index = pd.to_datetime(s.index)
    s.name = factor_id
    return s


@st.cache_data(ttl=3600)
def load_returns_wide(factor_ids: tuple[str, ...]) -> pd.DataFrame:
    conn = get_conn()
    df = read_returns_wide(conn, list(factor_ids))
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val * 100:.1f}%"


def _fmt_f(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def _category_badge(cat: str) -> str:
    color = CATEGORY_COLORS.get(cat, "#9E9E9E")
    label = CATEGORY_LABELS.get(cat, cat)
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600">{label}</span>'
    )


def _short_history_badge() -> str:
    return (
        '<span style="background:#FFC107;color:#333;padding:2px 6px;'
        'border-radius:4px;font-size:0.7rem;font-weight:600">⚠ &lt;60mo</span>'
    )


def _cumulative_returns_chart(
    returns: pd.Series,
    title: str = "",
    log_scale: bool = False,
    color: str = "#2196F3",
) -> go.Figure:
    cum = (1 + returns.dropna()).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        mode="lines", line=dict(color=color, width=2),
        name=returns.name or "Factor",
        hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (base = 1)",
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
        margin=dict(t=40, b=40, l=50, r=20),
        height=380,
    )
    return fig


def _multi_cumulative_chart(wide_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = list(CATEGORY_COLORS.values())
    factors_df = load_all_factors()
    cat_map = dict(zip(factors_df["id"], factors_df["category"]))

    for i, col in enumerate(wide_df.columns):
        s = wide_df[col].dropna()
        if s.empty:
            continue
        # Normalize to 1.0 at first valid date
        cum = (1 + s).cumprod()
        cum = cum / cum.iloc[0]
        cat = cat_map.get(col, "other")
        color = CATEGORY_COLORS.get(cat, colors[i % len(colors)])
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=col,
            line=dict(color=color, width=2),
            hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Return (1.0 at start)",
        hovermode="x unified",
        margin=dict(t=20, b=40, l=50, r=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _decay_chart(factor_row: pd.Series) -> go.Figure:
    pre = factor_row.get("pre_pub_sharpe")
    post = factor_row.get("post_pub_sharpe")

    def _clean(v):
        if v is None:
            return None
        try:
            return None if math.isnan(float(v)) else float(v)
        except (TypeError, ValueError):
            return None

    pre, post = _clean(pre), _clean(post)
    if pre is None and post is None:
        return None

    labels = []
    values = []
    bar_colors = []
    pub_year = factor_row.get("year")
    pre_label = f"Pre-{pub_year}" if pub_year else "Pre-publication"
    post_label = f"Post-{pub_year}" if pub_year else "Post-publication"

    if pre is not None:
        labels.append(pre_label)
        values.append(pre)
        bar_colors.append("#4CAF50" if pre > 0 else "#F44336")
    if post is not None:
        labels.append(post_label)
        values.append(post)
        bar_colors.append("#4CAF50" if post > 0 else "#F44336")

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Publication Decay: Sharpe Ratio",
        yaxis_title="Sharpe Ratio",
        margin=dict(t=40, b=40, l=50, r=20),
        height=280,
        showlegend=False,
    )
    return fig


def _correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=400,
        coloraxis_colorbar=dict(title="ρ"),
    )
    return fig


def _code_snippet(factor_id: str) -> str:
    return f"""```python
from factor_zoo import FactorZoo

fz = FactorZoo()
returns = fz.get_returns("{factor_id}")
stats   = fz.get_stats("{factor_id}")
```"""


# ---------------------------------------------------------------------------
# Page: Browse
# ---------------------------------------------------------------------------


def page_browse(factors_df: pd.DataFrame) -> None:
    st.title("Factor Zoo Browser")
    st.caption("200+ published equity factors from academic asset pricing research")

    all_ids = sorted(factors_df["id"].dropna().tolist())

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Multiselect gives real typeahead over factor IDs
        pinned_ids = st.multiselect(
            "Factor (type to search)",
            options=all_ids,
            placeholder="e.g. BM, Mom12m, GP …",
        )

        # Free-text search over authors, description, category keywords
        keyword = st.text_input(
            "Keyword search",
            placeholder="e.g. momentum, fama, accruals",
        )

        selected_cats = st.multiselect(
            "Category",
            options=ALL_CATEGORIES,
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
        )
        min_sharpe = st.slider("Min Sharpe ratio", -1.0, 3.0, -1.0, step=0.1)
        min_years = st.slider("Min sample years", 0, 40, 0, step=5)
        source_filter = st.selectbox("Source", ["All", "osap", "french"])

    df = factors_df.copy()

    # Pin-by-ID filter (multiselect) takes precedence over keyword
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
            .dt.days / 365.25
            >= min_years
        ]
    if source_filter != "All":
        df = df[df["source"] == source_filter]

    st.caption(f"{len(df)} factor(s) matching filters")

    if df.empty:
        st.info("No factors match the current filters.")
        return

    # Card grid — 3 columns
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.sort_values("sharpe", ascending=False).iterrows()):
        col = cols[i % 3]
        with col:
            warning = _short_history_badge() if row.get("has_short_history") else ""
            badge = _category_badge(row.get("category", "other"))
            st.markdown(
                f"""
                <div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;
                            margin-bottom:12px;background:#fafafa;color:#1a1a1a">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    {badge} {warning}
                  </div>
                  <div style="font-size:1rem;font-weight:700;margin:6px 0 2px;color:#1a1a1a">{row['id']}</div>
                  <div style="font-size:0.8rem;color:#444;margin-bottom:8px">
                    Sharpe: <b>{_fmt_f(row.get('sharpe'))}</b> &nbsp;|&nbsp;
                    t-stat: <b>{_fmt_f(row.get('t_stat'))}</b>
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
                st.session_state["page"] = "detail"
                st.session_state["detail_id"] = row["id"]
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Factor Detail
# ---------------------------------------------------------------------------


def page_detail(factors_df: pd.DataFrame, factor_id: str) -> None:
    row = factors_df[factors_df["id"] == factor_id]
    if row.empty:
        st.error(f"Factor '{factor_id}' not found.")
        return
    row = row.iloc[0]

    if st.button("← Back to Browse"):
        st.session_state["page"] = "browse"
        st.rerun()

    cat = row.get("category", "other")
    badge_html = _category_badge(cat)
    warning_html = _short_history_badge() if row.get("has_short_history") else ""

    # Header
    col_title, col_meta = st.columns([2, 1])
    with col_title:
        st.markdown(
            f"<h2 style='margin-bottom:4px'>{row['id']}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"{badge_html} {warning_html}", unsafe_allow_html=True)
        if row.get("authors"):
            st.caption(f"{row['authors']} ({row.get('year', '')})")
        if row.get("journal"):
            st.caption(f"Published in: **{row['journal']}**")
        if row.get("paper_url"):
            st.markdown(f"[View paper]({row['paper_url']})")

    with col_meta:
        sample_start = str(row.get("sample_start", ""))[:7]
        sample_end = str(row.get("sample_end", ""))[:7]
        st.metric("Sample Period", f"{sample_start} → {sample_end}")
        st.metric("Source", str(row.get("source", "")).upper())

    st.divider()

    # Returns + stats
    returns = load_returns(factor_id)
    left, right = st.columns([3, 2])

    with left:
        log_scale = st.checkbox("Log scale", key="detail_log")
        color = CATEGORY_COLORS.get(cat, "#2196F3")
        if not returns.empty:
            fig = _cumulative_returns_chart(returns, log_scale=log_scale, color=color)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No return data available.")

    with right:
        st.subheader("Performance Stats")
        stats_data = {
            "Ann. Return": _fmt_pct(row.get("ann_return")),
            "Ann. Volatility": _fmt_pct(row.get("ann_vol")),
            "Sharpe Ratio": _fmt_f(row.get("sharpe")),
            "Max Drawdown": _fmt_pct(row.get("max_drawdown")),
            "t-Statistic": _fmt_f(row.get("t_stat")),
        }
        stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric", "Value"])
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # Decay chart
    st.subheader("Publication Decay")
    decay_fig = _decay_chart(row)
    if decay_fig:
        st.plotly_chart(decay_fig, use_container_width=True)
    else:
        st.caption("Pre/post publication Sharpe not available for this factor.")

    # Description
    if row.get("description"):
        st.subheader("Description")
        st.write(row["description"])

    # Code snippet
    st.subheader("Use this factor")
    st.markdown(_code_snippet(factor_id))


# ---------------------------------------------------------------------------
# Page: Compare
# ---------------------------------------------------------------------------


def page_compare(factors_df: pd.DataFrame) -> None:
    st.title("Compare Factors")
    st.caption(f"Select up to {MAX_COMPARE} factors to compare side by side")

    all_ids = sorted(factors_df["id"].dropna().tolist())
    selected = st.multiselect(
        "Select factors",
        options=all_ids,
        default=["BM", "Mom12m"] if "BM" in all_ids and "Mom12m" in all_ids else all_ids[:2],
        max_selections=MAX_COMPARE,
    )

    if not selected:
        st.info("Select at least one factor above.")
        return

    wide = load_returns_wide(tuple(selected))

    # Overlaid cumulative return chart
    st.subheader("Cumulative Returns (normalized to 1.0)")
    if not wide.empty:
        fig = _multi_cumulative_chart(wide)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No return data available for the selected factors.")

    # Correlation heatmap
    if len(selected) > 1:
        st.subheader("Correlation Matrix")
        from factor_zoo.analytics.correlation import correlation_matrix
        corr = correlation_matrix(wide.dropna(how="all"))
        if not corr.empty:
            fig2 = _correlation_heatmap(corr)
            st.plotly_chart(fig2, use_container_width=True)

    # Summary stats table
    st.subheader("Summary Statistics")
    subset = factors_df[factors_df["id"].isin(selected)][
        ["id", "category", "ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat",
         "pre_pub_sharpe", "post_pub_sharpe", "sample_start", "sample_end"]
    ].copy()
    for col in ["ann_return", "ann_vol", "max_drawdown"]:
        subset[col] = subset[col].map(lambda v: _fmt_pct(v))
    for col in ["sharpe", "t_stat", "pre_pub_sharpe", "post_pub_sharpe"]:
        subset[col] = subset[col].map(lambda v: _fmt_f(v))
    subset.columns = [
        "ID", "Category", "Ann. Return", "Ann. Vol", "Sharpe",
        "Max DD", "t-stat", "Pre Sharpe", "Post Sharpe", "Start", "End"
    ]
    st.dataframe(subset, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Factor Zoo",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Check DB
    conn = get_conn()
    if conn is None:
        st.error(
            f"Database not found at `{db_path()}`.\n\n"
            "Run the build script first:\n```\nuv run python scripts/build_db.py\n```"
        )
        return

    factors_df = load_all_factors()
    if factors_df.empty:
        st.warning("Database is empty. Run `python scripts/build_db.py` to load data.")
        return

    # Navigation
    with st.sidebar:
        st.markdown("## Factor Zoo 🦁")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Browse", "Compare"],
            key="nav_radio",
        )
        st.markdown("---")
        total = len(factors_df)
        st.caption(f"{total} factors loaded")

    if st.session_state.get("page") == "detail":
        page_detail(factors_df, st.session_state.get("detail_id", ""))
    elif page == "Compare":
        page_compare(factors_df)
    else:
        st.session_state["page"] = "browse"
        page_browse(factors_df)


if __name__ == "__main__":
    main()
