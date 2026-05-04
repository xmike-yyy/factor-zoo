"""Public Python API for Factor Zoo.

Example usage:
    from factor_zoo import FactorZoo

    fz = FactorZoo()
    factors = fz.list_factors(category="momentum", min_sharpe=0.4)
    returns = fz.get_returns("Mom12m", start="2000-01-01")
    stats   = fz.get_stats("BM")
    corr    = fz.correlation_matrix(["BM", "Mom12m", "GrossProfit"])
    df      = fz.compare(["BM", "Mom12m", "GrossProfit"])

All returns are in decimal form (0.01 = 1%).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from factor_zoo.data.store import (
    connect,
    db_path,
    read_factors,
    read_returns,
    read_returns_wide,
    read_quintiles,
)
from factor_zoo.analytics.correlation import correlation_matrix as _corr_matrix
from factor_zoo.analytics.decay import DecayResult, compute_decay
from factor_zoo.analytics.portfolio import PortfolioResult, construct_portfolio
from factor_zoo.analytics.cluster import ClusterResult, cluster_factors as _cluster_factors
from factor_zoo.analytics.exposure import ExposureResult, compute_exposure
from factor_zoo.analytics.quintiles import QuintileResult, compute_quintile_analysis
from factor_zoo.analytics.replication import replication_score as _replication_score
from factor_zoo.analytics.replication import zoo_summary as _zoo_summary


class FactorZoo:
    """Browse, query, and compare equity factors from academic asset pricing research."""

    def __init__(self, db: Optional[str] = None) -> None:
        from pathlib import Path
        from factor_zoo.data.remote import ensure_db

        path = Path(db) if db else db_path()
        ensure_db(path)
        self._conn = connect(path, read_only=True)

    # ------------------------------------------------------------------
    # Listing / filtering
    # ------------------------------------------------------------------

    def list_factors(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        max_sharpe: Optional[float] = None,
        min_t_stat: Optional[float] = None,
        min_sample_years: Optional[int] = None,
        search: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of factors matching the given filters.

        Parameters
        ----------
        category : str, optional
            One of: value, momentum, quality, investment, profitability,
            intangibles, trading_frictions, other.
        source : str, optional
            'osap' or 'french'.
        min_sharpe : float, optional
            Minimum annualized Sharpe ratio.
        max_sharpe : float, optional
            Maximum annualized Sharpe ratio.
        min_t_stat : float, optional
            Minimum t-statistic.
        min_sample_years : int, optional
            Minimum years of return history.
        search : str, optional
            Case-insensitive search in id, name, and description.

        Returns
        -------
        pd.DataFrame with one row per matching factor.
        """
        conditions = []
        if category:
            conditions.append(f"category = '{category}'")
        if source:
            conditions.append(f"source = '{source}'")
        if min_sharpe is not None:
            conditions.append(f"sharpe >= {min_sharpe}")
        if max_sharpe is not None:
            conditions.append(f"sharpe <= {max_sharpe}")
        if min_t_stat is not None:
            conditions.append(f"t_stat >= {min_t_stat}")
        if min_sample_years is not None:
            months = min_sample_years * 12
            conditions.append(
                f"id IN (SELECT factor_id FROM factor_returns GROUP BY factor_id "
                f"HAVING COUNT(*) >= {months})"
            )
        if search:
            safe = search.replace("'", "''")
            conditions.append(
                f"(LOWER(id) LIKE '%{safe.lower()}%' OR "
                f"LOWER(name) LIKE '%{safe.lower()}%' OR "
                f"LOWER(description) LIKE '%{safe.lower()}%')"
            )
        where = " AND ".join(conditions)
        return read_factors(self._conn, where=where)

    # ------------------------------------------------------------------
    # Return series
    # ------------------------------------------------------------------

    def get_returns(
        self,
        factor_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Monthly long-short returns as a pandas Series (decimal).

        Parameters
        ----------
        factor_id : str
            Factor identifier, e.g. 'Mom12m', 'BM'.
        start : str, optional
            ISO date string, e.g. '2000-01-01'.
        end : str, optional
            ISO date string.

        Returns
        -------
        pd.Series with DatetimeIndex.
        """
        df = read_returns(self._conn, factor_id, start=start, end=end)
        if df.empty:
            return pd.Series(dtype=float, name=factor_id)
        s = df.set_index("date")["ls_return"]
        s.index = pd.to_datetime(s.index)
        s.name = factor_id
        return s

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, factor_id: str) -> dict:
        """Return a dict of precomputed performance stats for one factor.

        Keys: ann_return, ann_vol, sharpe, max_drawdown, t_stat,
              pre_pub_sharpe, post_pub_sharpe, has_short_history.
        """
        row = self._conn.execute(
            """SELECT ann_return, ann_vol, sharpe, max_drawdown, t_stat,
                      pre_pub_sharpe, post_pub_sharpe, has_short_history
               FROM factors WHERE id = ?""",
            [factor_id],
        ).df()
        if row.empty:
            raise KeyError(f"Factor '{factor_id}' not found in database.")
        return row.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Multi-factor
    # ------------------------------------------------------------------

    def compare(
        self,
        factor_ids: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Wide DataFrame of aligned monthly returns for multiple factors.

        Columns are factor IDs, index is date. Rows with all NaN are dropped.

        Parameters
        ----------
        factor_ids : list[str]
            Up to any number of factor IDs.
        start, end : str, optional
            ISO date strings to filter the date range.

        Returns
        -------
        pd.DataFrame with DatetimeIndex.
        """
        df = read_returns_wide(self._conn, factor_ids)
        df.index = pd.to_datetime(df.index)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df.dropna(how="all")

    def correlation_matrix(
        self,
        factor_ids: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pairwise Pearson correlation matrix for the given factors.

        Parameters
        ----------
        factor_ids : list[str]
        start, end : str, optional

        Returns
        -------
        pd.DataFrame (square, factor_ids × factor_ids).
        """
        wide = self.compare(factor_ids, start=start, end=end)
        return _corr_matrix(wide)

    # ------------------------------------------------------------------
    # New analytics methods
    # ------------------------------------------------------------------

    def decay(self, factor_id: str) -> DecayResult:
        """Rolling Sharpe, half-life, and pre/post-publication decay for one factor."""
        pub_year_row = self._conn.execute(
            "SELECT year FROM factors WHERE id = ?", [factor_id]
        ).df()
        if pub_year_row.empty:
            raise KeyError(f"Factor '{factor_id}' not found.")
        pub_year = pub_year_row.iloc[0]["year"]
        pub_year = int(pub_year) if pd.notna(pub_year) else None

        returns = self.get_returns(factor_id)
        return compute_decay(returns, factor_id, pub_year)

    def portfolio(
        self,
        factor_ids: list[str],
        method: str = "equal",
        weights: Optional[list[float]] = None,
    ) -> PortfolioResult:
        """Construct a multi-factor portfolio from the given factors.

        Parameters
        ----------
        factor_ids : list[str]
        method : str — "equal", "custom", "max_sharpe", or "risk_parity"
        weights : list[float] — required when method="custom", must sum to 1.0
        """
        wide = self.compare(factor_ids)
        return construct_portfolio(wide, method=method, weights=weights)

    def replication_score(self, factor_id: str) -> dict:
        """Compare paper-reported t-stat to computed t-stat for one factor."""
        return _replication_score(self._conn, factor_id)

    def zoo_summary(self) -> dict:
        """Aggregate statistics about the entire factor zoo."""
        return _zoo_summary(self._conn)

    def cluster_factors(self, n_clusters: int = 10) -> ClusterResult:
        """Group all factors by correlation structure using agglomerative clustering."""
        all_ids = self.factor_ids()
        wide = read_returns_wide(self._conn, all_ids)
        wide.index = pd.to_datetime(wide.index)

        cat_rows = self._conn.execute("SELECT id, category FROM factors").df()
        category_map = dict(zip(cat_rows["id"], cat_rows["category"]))

        return _cluster_factors(wide, n_clusters=n_clusters, category_map=category_map)

    def get_quintiles(self, factor_id: str) -> pd.DataFrame:
        """Quintile value-weighted returns for one factor. Columns q1–q5, DatetimeIndex.

        Raises KeyError if the factor has no quintile data (French factors won't have it;
        run build_db.py to populate OSAP quintile data).
        """
        df = read_quintiles(self._conn, factor_id)
        if df.empty:
            raise KeyError(
                f"Factor '{factor_id}' has no quintile data. "
                "French factors do not have quintiles; OSAP factors require a DB rebuild "
                "with load_osap_quintiles() wired in build_db.py."
            )
        return df

    def quintile_spread(self, factor_id: str) -> pd.Series:
        """Monthly q5 - q1 return (decimal) for one factor, as a pandas Series."""
        df = self.get_quintiles(factor_id)
        spread = df["q5"] - df["q1"]
        spread.name = factor_id
        return spread

    def quintile_analysis(self, factor_id: str) -> QuintileResult:
        """Full quintile analysis: spread, monotonicity score, Sharpe, and plot."""
        df = self.get_quintiles(factor_id)
        return compute_quintile_analysis(factor_id, df)

    def exposure(
        self,
        returns: "pd.Series",
        factors: Optional[list[str]] = None,
        top_n: int = 10,
    ) -> ExposureResult:
        """OLS regression of user returns on factor returns.

        Parameters
        ----------
        returns : pd.Series — monthly returns with DatetimeIndex
        factors : list[str], optional — factor IDs to regress on.
            If None, runs against all factors and keeps top ``top_n`` by |t-stat|.
        top_n : int — how many factors to keep when factors=None (default 10)
        """
        ids = factors if factors is not None else self.factor_ids()
        wide = read_returns_wide(self._conn, ids)
        wide.index = pd.to_datetime(wide.index)

        result = compute_exposure(returns, wide)

        if factors is None:
            top_keys = sorted(result.t_stats, key=lambda f: abs(result.t_stats[f]), reverse=True)[:top_n]
            result = ExposureResult(
                loadings={f: result.loadings[f] for f in top_keys},
                t_stats={f: result.t_stats[f] for f in top_keys},
                p_values={f: result.p_values[f] for f in top_keys},
                r_squared=result.r_squared,
                alpha=result.alpha,
                alpha_t_stat=result.alpha_t_stat,
                overlap_months=result.overlap_months,
            )

        return result

    def screen(
        self,
        min_sharpe: Optional[float] = None,
        min_t_stat: Optional[float] = None,
        max_correlation: Optional[float] = None,
        categories: Optional[list[str]] = None,
        min_years: Optional[int] = None,
    ) -> pd.DataFrame:
        """Filter factors by thresholds, then deduplicate by correlation.

        Parameters
        ----------
        min_sharpe, min_t_stat : float, optional
        max_correlation : float, optional — iteratively remove the lower-Sharpe
            factor from any pair whose correlation exceeds this threshold.
        categories : list[str], optional — restrict to these categories.
        min_years : int, optional — minimum years of return history.
        """
        df = self.list_factors(
            min_sharpe=min_sharpe,
            min_t_stat=min_t_stat,
            min_sample_years=min_years,
        )
        if categories:
            df = df[df["category"].isin(categories)]

        if max_correlation is not None and len(df) > 1:
            ids = df["id"].tolist()
            wide = read_returns_wide(self._conn, ids)
            corr = _corr_matrix(wide).abs()

            remaining = set(ids)
            sharpe_map = df.set_index("id")["sharpe"].to_dict()
            for i, fi in enumerate(ids):
                if fi not in remaining:
                    continue
                for fj in ids[i + 1:]:
                    if fj not in remaining:
                        continue
                    if fi in corr.index and fj in corr.columns:
                        if corr.loc[fi, fj] > max_correlation:
                            drop = fi if sharpe_map.get(fi, 0) < sharpe_map.get(fj, 0) else fj
                            remaining.discard(drop)

            df = df[df["id"].isin(remaining)].reset_index(drop=True)

        return df

    def update(self, check_only: bool = False) -> dict:
        """Check for new factors in OSAP; optionally rebuild the database.

        Parameters
        ----------
        check_only : bool — if True, return status dict without modifying anything.

        Returns
        -------
        dict with keys: new_signals (list), update_available (bool)
        """
        from factor_zoo.data.remote import check_for_update

        result = check_for_update()
        if check_only or not result.get("update_available"):
            return result

        import sys
        import os
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
        sys.path.insert(0, os.path.abspath(scripts_dir))
        import build_db
        self._conn.close()
        build_db.main()
        self._conn = connect(db_path(), read_only=True)
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return sorted list of all categories present in the database."""
        rows = self._conn.execute(
            "SELECT DISTINCT category FROM factors WHERE category IS NOT NULL ORDER BY category"
        ).df()
        return rows["category"].tolist()

    def factor_ids(self) -> list[str]:
        """Return sorted list of all factor IDs."""
        rows = self._conn.execute("SELECT id FROM factors ORDER BY id").df()
        return rows["id"].tolist()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FactorZoo":
        return self

    def __exit__(self, *_) -> None:
        self.close()
