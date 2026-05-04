"""Tests for OSAP download cache helpers in loader.py."""
import datetime
import os
import polars as pl
import pytest
from pathlib import Path
from factor_zoo.data.loader import _cache_dir, _find_cache, _save_cache, _clean_old_cache


def test_cache_dir_uses_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    assert _cache_dir() == tmp_path


def test_cache_dir_default():
    result = _cache_dir()
    assert result == Path.home() / ".factor_zoo" / "cache"


def test_find_cache_miss_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    assert _find_cache("osap_returns", max_age_days=7) is None


def test_find_cache_hit_recent(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    today = datetime.date.today().strftime("%Y%m%d")
    path = tmp_path / f"osap_returns_{today}.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    assert _find_cache("osap_returns", max_age_days=7) == path


def test_find_cache_miss_too_old(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    old = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y%m%d")
    path = tmp_path / f"osap_returns_{old}.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    assert _find_cache("osap_returns", max_age_days=7) is None


def test_find_cache_max_age_zero_always_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    today = datetime.date.today().strftime("%Y%m%d")
    path = tmp_path / f"osap_returns_{today}.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    assert _find_cache("osap_returns", max_age_days=0) is None


def test_save_cache_creates_dated_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    df = pl.DataFrame({"factor_id": ["BM"], "ret": [0.01]})
    _save_cache("osap_returns", df)
    today = datetime.date.today().strftime("%Y%m%d")
    expected = tmp_path / f"osap_returns_{today}.parquet"
    assert expected.exists()
    assert pl.read_parquet(expected).equals(df)


def test_clean_old_cache_removes_old_keeps_recent(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_ZOO_CACHE_DIR", str(tmp_path))
    old = (datetime.date.today() - datetime.timedelta(days=35)).strftime("%Y%m%d")
    recent = datetime.date.today().strftime("%Y%m%d")
    old_file = tmp_path / f"osap_returns_{old}.parquet"
    recent_file = tmp_path / f"osap_returns_{recent}.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(old_file)
    pl.DataFrame({"a": [1]}).write_parquet(recent_file)
    _clean_old_cache(max_age_days=30)
    assert not old_file.exists()
    assert recent_file.exists()
