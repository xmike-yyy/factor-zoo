# Factor Zoo

A research toolkit for exploring, comparing, and downloading **200+ published equity factors** from the academic asset pricing literature. Built for quants, researchers, and students who want a single, queryable database of factor returns rather than hunting down individual datasets.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Build the local database (~5 min, downloads OSAP + Ken French data)
uv run python scripts/build_db.py

# 3. Launch the browser UI
uv run streamlit run factor_zoo/app.py
```

The database is stored at `~/.factor_zoo/factors.db`. Override with the `FACTOR_ZOO_DB` environment variable.

## Python API

```python
from factor_zoo import FactorZoo

fz = FactorZoo()

# List all momentum factors with Sharpe ≥ 0.4
factors = fz.list_factors(category="momentum", min_sharpe=0.4)

# Monthly long-short returns as a pandas Series (decimal, 0.01 = 1%)
returns = fz.get_returns("Mom12m", start="2000-01-01")

# Precomputed performance stats
stats = fz.get_stats("BM")
# {"ann_return": 0.073, "ann_vol": 0.112, "sharpe": 0.65, ...}

# Wide DataFrame of aligned returns for multiple factors
df = fz.compare(["BM", "Mom12m", "GrossProfit"])

# Pairwise Pearson correlation matrix
corr = fz.correlation_matrix(["BM", "Mom12m", "GrossProfit"])
```

Use as a context manager to ensure the connection is closed:

```python
with FactorZoo() as fz:
    factors = fz.list_factors(source="osap", min_t_stat=3.0)
```

## Data Sources

**Open Source Asset Pricing (OSAP)**
Chen, Andrew Y. and Zimmermann, Tom, "Open Source Cross-Sectional Asset Pricing" (2022), *Critical Finance Review*.
~212 published equity factors with long-short portfolio returns, paper t-statistics, and full metadata. Data via the [`openassetpricing`](https://github.com/OpenSourceAP/CrossSection) Python package.

**Ken French Data Library**
Fama, Eugene F. and French, Kenneth R. Classic benchmark factors downloaded directly from Dartmouth:
- FF3 (1993): MktRF, SMB, HML
- FF5 (2015): RMW, CMA
- Momentum / UMD (Carhart 1997): Mom

## Factor Categories

| Category | Description |
|---|---|
| `value` | Book-to-market, earnings yield, cash flow multiples |
| `momentum` | Price momentum, reversal, lead-lag effects |
| `profitability` | ROE, gross profit, accruals, earnings quality |
| `investment` | Asset growth, capex, external financing |
| `intangibles` | R&D, analyst recommendations, information proxies |
| `trading_frictions` | Liquidity, volatility, short-sale constraints, turnover |
| `other` | Size, leverage, default risk, payout, ownership |

## API Reference

### `FactorZoo(db=None)`
Opens a read-only connection to the database. `db` is an optional path string; defaults to `~/.factor_zoo/factors.db`.

### `list_factors(...) → pd.DataFrame`
| Parameter | Type | Description |
|---|---|---|
| `category` | `str` | Filter by category slug |
| `source` | `str` | `'osap'` or `'french'` |
| `min_sharpe` / `max_sharpe` | `float` | Annualized Sharpe ratio bounds |
| `min_t_stat` | `float` | Minimum t-statistic |
| `min_sample_years` | `int` | Minimum years of return history |
| `search` | `str` | Case-insensitive substring search on id/name/description |

Returns one row per matching factor with all metadata and precomputed stats.

### `get_returns(factor_id, start=None, end=None) → pd.Series`
Monthly long-short returns with a `DatetimeIndex`. Values are decimals (`0.01 = 1%`). `start`/`end` are ISO date strings.

### `get_stats(factor_id) → dict`
Precomputed stats dict. Keys: `ann_return`, `ann_vol`, `sharpe`, `max_drawdown`, `t_stat`, `pre_pub_sharpe`, `post_pub_sharpe`, `has_short_history`.

### `compare(factor_ids, start=None, end=None) → pd.DataFrame`
Wide DataFrame with one column per factor, aligned on the date index. Rows with all NaN dropped.

### `correlation_matrix(factor_ids, start=None, end=None) → pd.DataFrame`
Square pairwise Pearson correlation matrix.

### `categories() → list[str]`
All category slugs present in the database.

### `factor_ids() → list[str]`
All factor IDs, sorted.

## Contributing: Adding a New Factor

1. **Add the loader** — in `factor_zoo/data/loader.py`, add a download function that returns a `pl.DataFrame` with columns `factor_id` (str), `date` (Date), `ls_return` (Float64, decimal).

2. **Add metadata** — return a `pl.DataFrame` with the columns expected by `upsert_factors` (see `_FACTORS_COLS` in `store.py`): `id`, `name`, `category`, `source`, etc.

3. **Wire it into `build_db.py`** — call your loader, then `upsert_factors(conn, meta)` and `upsert_returns(conn, returns)`.

4. **Rebuild the database**:
   ```bash
   uv run python scripts/build_db.py
   ```
   The script is fully idempotent — safe to re-run.

5. **Run the tests** to verify stats compute cleanly:
   ```bash
   uv run pytest
   ```
