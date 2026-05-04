"""Tests for _pivot_quintiles transformation in loader.py."""
import datetime
import polars as pl
import pytest
from factor_zoo.data.loader import _pivot_quintiles


def _raw_long(signalnames, dates, ports, rets):
    return pl.DataFrame({
        "signalname": signalnames,
        "date": dates,
        "port": ports,
        "ret": rets,
    })


def test_pivot_quintiles_produces_wide_row():
    raw = _raw_long(
        signalnames=["Mom12m"] * 5,
        dates=[datetime.date(2010, 1, 31)] * 5,
        ports=["Q1", "Q2", "Q3", "Q4", "Q5"],
        rets=[-2.0, 1.0, 3.0, 5.0, 8.0],
    )
    result = _pivot_quintiles(raw)
    assert len(result) == 1
    assert "factor_id" in result.columns
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        assert q in result.columns


def test_pivot_quintiles_converts_percent_to_decimal():
    raw = _raw_long(["Mom12m"] * 5, [datetime.date(2010, 1, 31)] * 5,
                    ["Q1", "Q2", "Q3", "Q4", "Q5"], [-2.0, 1.0, 3.0, 5.0, 8.0])
    result = _pivot_quintiles(raw)
    assert abs(result["q1"][0] - (-0.02)) < 1e-6
    assert abs(result["q5"][0] - 0.08) < 1e-6


def test_pivot_quintiles_multiple_factor_date_combos():
    raw = _raw_long(
        signalnames=["Mom12m"] * 5 + ["BM"] * 5,
        dates=[datetime.date(2010, 1, 31)] * 5 + [datetime.date(2010, 2, 28)] * 5,
        ports=["Q1", "Q2", "Q3", "Q4", "Q5"] * 2,
        rets=[-2.0, 1.0, 3.0, 5.0, 8.0, -1.5, 1.0, 2.5, 4.0, 6.0],
    )
    result = _pivot_quintiles(raw)
    assert len(result) == 2
    assert set(result["factor_id"].to_list()) == {"Mom12m", "BM"}


def test_pivot_quintiles_ignores_non_quintile_ports():
    raw = _raw_long(
        signalnames=["Mom12m"] * 6,
        dates=[datetime.date(2010, 1, 31)] * 6,
        ports=["Q1", "Q2", "Q3", "Q4", "Q5", "LS"],
        rets=[-2.0, 1.0, 3.0, 5.0, 8.0, 10.0],
    )
    result = _pivot_quintiles(raw)
    assert len(result) == 1
    assert "LS" not in result.columns
