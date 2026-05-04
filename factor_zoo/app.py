"""Factor Zoo Browser — Streamlit entrypoint and shared helpers.

Launch with:
    uv run streamlit run factor_zoo/app.py
"""
import sys
import math
from pathlib import Path
from typing import Optional

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
# Cached DB + data helpers (imported by all pages)
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
# UI helpers (imported by all pages)
# ---------------------------------------------------------------------------


def fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val * 100:.1f}%"


def fmt_f(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def category_badge(cat: str) -> str:
    color = CATEGORY_COLORS.get(cat, "#9E9E9E")
    label = CATEGORY_LABELS.get(cat, cat)
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600">{label}</span>'
    )


def short_history_badge() -> str:
    return (
        '<span style="background:#FFC107;color:#333;padding:2px 6px;'
        'border-radius:4px;font-size:0.7rem;font-weight:600">⚠ &lt;60mo</span>'
    )


def cumulative_returns_chart(
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


def multi_cumulative_chart(wide_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = list(CATEGORY_COLORS.values())
    factors_df = load_all_factors()
    cat_map = dict(zip(factors_df["id"], factors_df["category"]))
    for i, col in enumerate(wide_df.columns):
        s = wide_df[col].dropna()
        if s.empty:
            continue
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


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
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


def code_snippet(factor_id: str) -> str:
    return f"""```python
from factor_zoo import FactorZoo

fz = FactorZoo()
returns = fz.get_returns("{factor_id}")
stats   = fz.get_stats("{factor_id}")
```"""


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Factor Zoo",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    conn = get_conn()
    if conn is None:
        st.error(
            f"Database not found at `{db_path()}`.\n\n"
            "Run the build script first:\n```\nuv run python scripts/build_db.py\n```"
        )
        return

    factors_df = load_all_factors()
    n = len(factors_df)

    st.title("Factor Zoo 🦁")
    st.markdown(
        f"**{n} published equity factors** from academic asset pricing research. "
        "Use the sidebar to navigate."
    )
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Factors", n)
    col2.metric("Data Sources", "OSAP + Ken French")
    col3.metric("API Version", "v0.3.0")


if __name__ == "__main__":
    main()
