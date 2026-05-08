"""Tests for the quintile stats helper used in the Quintile Analysis page."""
import math
import pandas as pd
import numpy as np


def quintile_stats(quintiles: pd.DataFrame) -> pd.DataFrame:
    """Copy of the helper function for testing."""
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


def make_quintile_df(n=60, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="ME")
    data = {f"q{i}": rng.normal(0.005 * i, 0.04, n) for i in range(1, 6)}
    return pd.DataFrame(data, index=idx)


def test_quintile_stats_returns_5_rows():
    df = make_quintile_df()
    result = quintile_stats(df)
    assert len(result) == 5
    assert list(result["Quintile"]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_quintile_stats_ann_return_format():
    df = make_quintile_df()
    result = quintile_stats(df)
    # Ann. Return should be a percentage string
    for v in result["Ann. Return"]:
        assert "%" in v


def test_quintile_stats_handles_nan_data():
    df = make_quintile_df()
    df.loc[:, "q3"] = float("nan")
    result = quintile_stats(df)
    q3_row = result[result["Quintile"] == "Q3"].iloc[0]
    assert q3_row["Ann. Return"] == "N/A"
    assert q3_row["Ann. Vol"] == "N/A"


def test_quintile_stats_positive_returns_give_positive_sharpe():
    idx = pd.date_range("2000-01-01", periods=60, freq="ME")
    df = pd.DataFrame({f"q{i}": [0.01 * i] * 60 for i in range(1, 6)}, index=idx)
    result = quintile_stats(df)
    # All positive returns → all positive Sharpes
    for v in result["Sharpe"]:
        assert float(v) > 0
