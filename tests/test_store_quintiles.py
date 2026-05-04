"""Tests for factor_quintiles DB layer."""
import datetime
import pandas as pd
import polars as pl
import duckdb
import pytest
from factor_zoo.data.store import init_schema, upsert_quintiles, read_quintiles


@pytest.fixture
def mem_conn():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


def _sample_df():
    return pl.DataFrame({
        "factor_id": ["Mom12m", "Mom12m", "BM"],
        "date": [
            datetime.date(2010, 1, 31),
            datetime.date(2010, 2, 28),
            datetime.date(2010, 1, 31),
        ],
        "q1": [-0.02, -0.01, -0.015],
        "q2": [0.01,  0.02,  0.01],
        "q3": [0.03,  0.04,  0.025],
        "q4": [0.05,  0.06,  0.04],
        "q5": [0.08,  0.09,  0.06],
    })


def test_upsert_and_read_roundtrip(mem_conn):
    upsert_quintiles(mem_conn, _sample_df())
    result = read_quintiles(mem_conn, "Mom12m")
    assert len(result) == 2
    assert list(result.columns) == ["q1", "q2", "q3", "q4", "q5"]
    assert abs(result.iloc[0]["q5"] - 0.08) < 1e-5


def test_read_empty_for_missing_factor(mem_conn):
    upsert_quintiles(mem_conn, _sample_df())
    result = read_quintiles(mem_conn, "NonExistent")
    assert result.empty
    assert list(result.columns) == ["q1", "q2", "q3", "q4", "q5"]


def test_upsert_is_idempotent(mem_conn):
    df = _sample_df()
    upsert_quintiles(mem_conn, df)
    upsert_quintiles(mem_conn, df)
    result = read_quintiles(mem_conn, "Mom12m")
    assert len(result) == 2


def test_upsert_updates_existing_row(mem_conn):
    upsert_quintiles(mem_conn, _sample_df())
    updated = pl.DataFrame({
        "factor_id": ["Mom12m"],
        "date": [datetime.date(2010, 1, 31)],
        "q1": [-0.02], "q2": [0.01], "q3": [0.03], "q4": [0.05], "q5": [0.99],
    })
    upsert_quintiles(mem_conn, updated)
    result = read_quintiles(mem_conn, "Mom12m")
    row = result.loc[pd.Timestamp("2010-01-31")]
    assert abs(row["q5"] - 0.99) < 1e-5


def test_read_quintiles_has_datetime_index(mem_conn):
    upsert_quintiles(mem_conn, _sample_df())
    result = read_quintiles(mem_conn, "Mom12m")
    assert isinstance(result.index, pd.DatetimeIndex)
