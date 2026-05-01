"""DuckDB read/write layer. All returns stored as decimals (0.01 = 1%)."""
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import polars as pl

DEFAULT_DB = Path.home() / ".factor_zoo" / "factors.db"
_SCHEMA = Path(__file__).parent / "schema.sql"


def db_path() -> Path:
    raw = os.environ.get("FACTOR_ZOO_DB")
    return Path(raw) if raw else DEFAULT_DB


def connect(path: Optional[Path] = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(p), read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_SCHEMA.read_text())
    # Migration: add paper_t_stat to any existing DB that predates this column
    conn.execute(
        "ALTER TABLE factors ADD COLUMN IF NOT EXISTS paper_t_stat FLOAT"
    )


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

_FACTORS_COLS = [
    "id", "name", "category", "subcategory", "authors", "year", "journal",
    "paper_url", "sample_start", "sample_end", "n_stocks_avg", "description",
    "ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat",
    "pre_pub_sharpe", "post_pub_sharpe", "has_short_history", "source",
    "paper_t_stat",
]

def upsert_factors(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    """Insert or replace rows in the factors table."""
    conn.register("_factors_staging", df.to_pandas())
    cols = ", ".join(_FACTORS_COLS)
    conn.execute(f"""
        INSERT OR REPLACE INTO factors ({cols})
        SELECT {cols} FROM _factors_staging
    """)
    conn.unregister("_factors_staging")


def upsert_returns(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    """Insert or replace rows in factor_returns. df must have columns:
    factor_id (str), date (date), ls_return (float)."""
    conn.register("_returns_staging", df.to_pandas())
    conn.execute("""
        INSERT OR REPLACE INTO factor_returns (factor_id, date, ls_return)
        SELECT factor_id, date, ls_return FROM _returns_staging
    """)
    conn.unregister("_returns_staging")


# ---------------------------------------------------------------------------
# Read helpers (return pandas for public API compatibility)
# ---------------------------------------------------------------------------

def read_factors(conn: duckdb.DuckDBPyConnection, where: str = "") -> pd.DataFrame:
    sql = "SELECT * FROM factors"
    if where:
        sql += f" WHERE {where}"
    return conn.execute(sql).df()


def read_returns(
    conn: duckdb.DuckDBPyConnection,
    factor_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    conditions = ["factor_id = ?"]
    params: list = [factor_id]
    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)
    sql = f"SELECT date, ls_return FROM factor_returns WHERE {' AND '.join(conditions)} ORDER BY date"
    return conn.execute(sql, params).df()


def read_returns_wide(
    conn: duckdb.DuckDBPyConnection,
    factor_ids: list[str],
) -> pd.DataFrame:
    """Return a wide DataFrame (date index, one column per factor_id).
    Uses pandas pivot to avoid DuckDB's restriction on parameterized PIVOT."""
    placeholders = ", ".join(["?"] * len(factor_ids))
    sql = f"""
        SELECT factor_id, date, ls_return
        FROM factor_returns
        WHERE factor_id IN ({placeholders})
        ORDER BY date
    """
    long = conn.execute(sql, factor_ids).df()
    if long.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=factor_ids)
    wide = long.pivot(index="date", columns="factor_id", values="ls_return")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    # Preserve the requested column order
    existing = [f for f in factor_ids if f in wide.columns]
    return wide[existing]


def read_returns_wide_all(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return a wide DataFrame of all factor returns (date index, one column per factor_id)."""
    long = conn.execute(
        "SELECT factor_id, date, ls_return FROM factor_returns ORDER BY date"
    ).df()
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="date", columns="factor_id", values="ls_return")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide
