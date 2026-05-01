"""Shared test fixtures for factor_zoo tests."""
import datetime
import duckdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mem_conn():
    """In-memory DuckDB connection with schema + sample data."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE factors (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            subcategory VARCHAR,
            authors VARCHAR,
            year INTEGER,
            journal VARCHAR,
            paper_url VARCHAR,
            sample_start DATE,
            sample_end DATE,
            n_stocks_avg INTEGER,
            description TEXT,
            ann_return FLOAT,
            ann_vol FLOAT,
            sharpe FLOAT,
            max_drawdown FLOAT,
            t_stat FLOAT,
            pre_pub_sharpe FLOAT,
            post_pub_sharpe FLOAT,
            has_short_history BOOLEAN DEFAULT FALSE,
            source VARCHAR,
            paper_t_stat FLOAT
        )
    """)
    conn.execute("""
        CREATE TABLE factor_returns (
            factor_id VARCHAR,
            date DATE,
            ls_return FLOAT,
            PRIMARY KEY (factor_id, date)
        )
    """)

    conn.execute("""
        INSERT INTO factors VALUES
            ('Mom12m', 'Momentum 12-1', 'momentum', NULL, 'Jegadeesh', 1993, 'JF',
             NULL, '1980-01-01', '2020-12-31', NULL, 'Momentum factor',
             0.08, 0.14, 0.57, -0.35, 4.2, 0.65, 0.42, FALSE, 'osap', 4.2),
            ('Accruals', 'Accruals', 'profitability', NULL, 'Sloan', 1996, 'AR',
             NULL, '1975-01-01', '2020-12-31', NULL, 'Accruals factor',
             0.06, 0.12, 0.50, -0.28, 3.1, 0.60, 0.35, FALSE, 'osap', NULL),
            ('HML', 'High Minus Low', 'value', NULL, 'Fama-French', 1993, 'JF',
             NULL, '1963-01-01', '2020-12-31', NULL, 'Value factor',
             0.04, 0.13, 0.31, -0.40, 2.5, 0.45, 0.22, FALSE, 'french', NULL)
    """)

    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    returns = rng.normal(0.007, 0.04, 120)
    rows = [(str(d.date()), float(r)) for d, r in zip(dates, returns)]
    conn.executemany(
        "INSERT INTO factor_returns VALUES ('Mom12m', ?, ?)", rows
    )
    returns2 = rng.normal(0.005, 0.035, 120)
    rows2 = [(str(d.date()), float(r)) for d, r in zip(dates, returns2)]
    conn.executemany(
        "INSERT INTO factor_returns VALUES ('Accruals', ?, ?)", rows2
    )

    yield conn
    conn.close()
