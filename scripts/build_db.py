#!/usr/bin/env python
"""One-time (idempotent) script to seed the Factor Zoo DuckDB database.

Usage:
    uv run python scripts/build_db.py

Re-running is safe: it upserts, never duplicates.
"""
import sys
import datetime
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import polars as pl

from factor_zoo.data.store import connect, init_schema, upsert_factors, upsert_returns, db_path
from factor_zoo.data.loader import (
    load_osap_signal_doc,
    load_osap_returns,
    load_french_returns,
    build_french_metadata,
)
from factor_zoo.analytics.stats import compute_all_stats


def _year_to_date(year: int | None, last_day: bool = False) -> datetime.date | None:
    if year is None:
        return None
    if last_day:
        return datetime.date(year, 12, 31)
    return datetime.date(year, 1, 1)


def main(no_cache: bool = False) -> None:
    db = db_path()
    print(f"Building database at: {db}")

    from factor_zoo.data.loader import _clean_old_cache
    _clean_old_cache(max_age_days=30)

    conn = connect()
    init_schema(conn)

    max_age = 0 if no_cache else 7

    # ------------------------------------------------------------------
    # 1. OSAP signal documentation
    # ------------------------------------------------------------------
    print("\n[1/4] Loading OSAP signal documentation...")
    doc = load_osap_signal_doc(max_age_days=max_age)
    print(f"      {len(doc)} predictors found in OSAP documentation")

    # Convert year columns → date columns for factors table
    factors_meta = doc.with_columns([
        pl.col("sample_start_year").map_elements(
            lambda y: _year_to_date(y, last_day=False), return_dtype=pl.Date
        ).alias("sample_start"),
        pl.col("sample_end_year").map_elements(
            lambda y: _year_to_date(y, last_day=True), return_dtype=pl.Date
        ).alias("sample_end"),
        pl.lit(None).cast(pl.Int32).alias("n_stocks_avg"),
        pl.lit(None).cast(pl.Float32).alias("ann_return"),
        pl.lit(None).cast(pl.Float32).alias("ann_vol"),
        pl.lit(None).cast(pl.Float32).alias("sharpe"),
        pl.lit(None).cast(pl.Float32).alias("max_drawdown"),
        pl.lit(None).cast(pl.Float32).alias("t_stat"),
        pl.lit(None).cast(pl.Float32).alias("pre_pub_sharpe"),
        pl.lit(None).cast(pl.Float32).alias("post_pub_sharpe"),
        pl.lit(False).alias("has_short_history"),
        pl.col("t_stat_paper").alias("paper_t_stat"),
    ]).select([
        "id", "name", "category", "subcategory", "authors", "year", "journal",
        "paper_url", "sample_start", "sample_end", "n_stocks_avg", "description",
        "ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat",
        "pre_pub_sharpe", "post_pub_sharpe", "has_short_history", "source",
        "paper_t_stat",
    ])
    upsert_factors(conn, factors_meta)

    # ------------------------------------------------------------------
    # 2. OSAP portfolio returns
    # ------------------------------------------------------------------
    print("\n[2/4] Loading OSAP long-short returns...")
    osap_returns = load_osap_returns(max_age_days=max_age)
    # Only keep factors that are in the doc
    valid_ids = set(doc["id"].to_list())
    osap_returns = osap_returns.filter(pl.col("factor_id").is_in(valid_ids))
    print(f"      {osap_returns['factor_id'].n_unique()} factors, "
          f"{len(osap_returns):,} monthly return observations")
    upsert_returns(conn, osap_returns)

    # ------------------------------------------------------------------
    # 3. Ken French factors
    # ------------------------------------------------------------------
    print("\n[3/4] Loading Ken French factors...")
    french_returns, french_ids = load_french_returns()
    french_meta = build_french_metadata(french_ids)

    # Derive sample_start / sample_end from the actual return dates
    date_ranges = (
        french_returns
        .group_by("factor_id")
        .agg([
            pl.col("date").min().alias("sample_start"),
            pl.col("date").max().alias("sample_end"),
        ])
    )
    french_meta = (
        french_meta
        .drop(["sample_start_year", "sample_end_year"])
        .join(date_ranges, left_on="id", right_on="factor_id", how="left")
        .with_columns([
            pl.lit(None).cast(pl.Int32).alias("n_stocks_avg"),
            pl.lit(None).cast(pl.Float32).alias("ann_return"),
            pl.lit(None).cast(pl.Float32).alias("ann_vol"),
            pl.lit(None).cast(pl.Float32).alias("sharpe"),
            pl.lit(None).cast(pl.Float32).alias("max_drawdown"),
            pl.lit(None).cast(pl.Float32).alias("t_stat"),
            pl.lit(None).cast(pl.Float32).alias("pre_pub_sharpe"),
            pl.lit(None).cast(pl.Float32).alias("post_pub_sharpe"),
            pl.lit(False).alias("has_short_history"),
            pl.lit(None).cast(pl.Float32).alias("paper_t_stat"),
        ])
        .select([
            "id", "name", "category", "subcategory", "authors", "year", "journal",
            "paper_url", "sample_start", "sample_end", "n_stocks_avg", "description",
            "ann_return", "ann_vol", "sharpe", "max_drawdown", "t_stat",
            "pre_pub_sharpe", "post_pub_sharpe", "has_short_history", "source",
            "paper_t_stat",
        ])
    )
    upsert_factors(conn, french_meta)
    upsert_returns(conn, french_returns)
    print(f"      {len(french_ids)} French factors loaded")

    # ------------------------------------------------------------------
    # 4. Compute and store all stats
    # ------------------------------------------------------------------
    print("\n[4/4] Computing performance statistics...")
    all_ids = conn.execute("SELECT id, year, source FROM factors").df()
    failures = []
    updated = 0

    for _, row in all_ids.iterrows():
        fid = row["id"]
        pub_year = int(row["year"]) if pd.notna(row["year"]) else None

        try:
            ret_df = conn.execute(
                "SELECT date, ls_return FROM factor_returns WHERE factor_id = ? ORDER BY date",
                [fid],
            ).df()

            if ret_df.empty:
                continue

            ret_series = pd.Series(
                ret_df["ls_return"].values,
                index=pd.to_datetime(ret_df["date"]),
            )
            stats = compute_all_stats(ret_series, pub_year)

            conn.execute("""
                UPDATE factors SET
                    ann_return = ?,
                    ann_vol = ?,
                    sharpe = ?,
                    max_drawdown = ?,
                    t_stat = ?,
                    pre_pub_sharpe = ?,
                    post_pub_sharpe = ?,
                    has_short_history = ?
                WHERE id = ?
            """, [
                stats["ann_return"], stats["ann_vol"], stats["sharpe"],
                stats["max_drawdown"], stats["t_stat"],
                stats["pre_pub_sharpe"], stats["post_pub_sharpe"],
                stats["has_short_history"],
                fid,
            ])
            updated += 1
        except Exception as e:
            failures.append((fid, str(e)))

    print(f"      Stats computed for {updated} factors")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM factors) as n_factors,
            (SELECT COUNT(*) FROM factors WHERE source = 'osap') as osap_factors,
            (SELECT COUNT(*) FROM factors WHERE source = 'french') as french_factors,
            (SELECT MIN(date) FROM factor_returns) as earliest_date,
            (SELECT MAX(date) FROM factor_returns) as latest_date,
            (SELECT COUNT(DISTINCT factor_id) FROM factor_returns) as factors_with_returns
    """).df()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    row = summary.iloc[0]
    print(f"  Total factors:        {int(row['n_factors'])}")
    print(f"  OSAP factors:         {int(row['osap_factors'])}")
    print(f"  French factors:       {int(row['french_factors'])}")
    print(f"  Factors with returns: {int(row['factors_with_returns'])}")
    print(f"  Date range:           {row['earliest_date']} → {row['latest_date']}")
    print(f"  Database:             {db}")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for fid, err in failures[:10]:
            print(f"    {fid}: {err}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the Factor Zoo DuckDB database.")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-download from OSAP, ignoring any cached parquet files."
    )
    args = parser.parse_args()
    main(no_cache=args.no_cache)
