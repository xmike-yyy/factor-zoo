"""Replication scoring and zoo-level aggregate statistics."""
from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd


def replication_score(conn: duckdb.DuckDBPyConnection, factor_id: str) -> dict[str, Any]:
    """Compare paper-reported t-stat to computed t-stat.

    Returns dict with keys: paper_t_stat, computed_t_stat, replication_score, verdict.
    Returns {"error": ...} if paper_t_stat is unavailable.
    Raises KeyError if factor not found.
    """
    row = conn.execute(
        "SELECT paper_t_stat, t_stat FROM factors WHERE id = ?", [factor_id]
    ).df()
    if row.empty:
        raise KeyError(f"Factor '{factor_id}' not found in database.")

    paper_t = row.iloc[0]["paper_t_stat"]
    computed_t = row.iloc[0]["t_stat"]

    if pd.isna(paper_t):
        return {"error": "paper_t_stat unavailable — run factorzoo build to refresh"}
    if pd.isna(computed_t):
        return {"error": "computed t-stat unavailable — factor has no return data"}

    paper_t = float(paper_t)
    computed_t = float(computed_t)

    if paper_t == 0:
        return {"error": "paper_t_stat is zero — cannot compute ratio"}

    score = computed_t / paper_t
    if score >= 0.8:
        verdict = "strong"
    elif score >= 0.5:
        verdict = "partial"
    else:
        verdict = "weak"

    return {
        "paper_t_stat": round(paper_t, 4),
        "computed_t_stat": round(computed_t, 4),
        "replication_score": round(score, 4),
        "verdict": verdict,
    }


def zoo_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Aggregate statistics about the entire factor zoo."""
    row = conn.execute("""
        SELECT
            COUNT(*) AS n_factors,
            AVG(CASE WHEN post_pub_sharpe > 0 THEN 1.0 ELSE 0.0 END)
                AS pct_positive_post_pub_sharpe,
            MEDIAN(CASE WHEN paper_t_stat > 0 AND t_stat IS NOT NULL
                        THEN t_stat / paper_t_stat END)
                AS median_replication_score,
            AVG(t_stat) AS mean_t_stat,
            MEDIAN(t_stat) AS median_t_stat,
            AVG(CASE WHEN t_stat >= 3.0 THEN 1.0 ELSE 0.0 END) AS pct_above_3
        FROM factors
        WHERE t_stat IS NOT NULL
    """).df().iloc[0]

    med_rep = row["median_replication_score"]

    return {
        "n_factors": int(conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]),
        "pct_positive_post_pub_sharpe": round(float(row["pct_positive_post_pub_sharpe"] or 0.0), 4),
        "median_replication_score": round(float(med_rep), 4) if pd.notna(med_rep) else None,
        "t_stat_distribution": {
            "mean": round(float(row["mean_t_stat"] or 0.0), 4),
            "median": round(float(row["median_t_stat"] or 0.0), 4),
            "pct_above_3": round(float(row["pct_above_3"] or 0.0), 4),
        },
    }
