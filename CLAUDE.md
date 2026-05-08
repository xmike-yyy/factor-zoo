# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Build the database (~5 min, downloads OSAP + Ken French data)
uv run python scripts/build_db.py

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_api.py
uv run pytest tests/test_decay.py::TestDecayResult::test_half_life_none_when_insufficient_data

# Run with coverage
uv run pytest --cov=factor_zoo

# Launch the Streamlit browser UI
uv run streamlit run factor_zoo/app.py
# or
./run.sh

# CLI (requires built DB)
uv run factorzoo list --category momentum --min-sharpe 0.4
uv run factorzoo detail Mom12m
uv run factorzoo zoo-summary
uv run factorzoo update --check-only
uv run factorzoo build
```

## Architecture

### Data flow

```
scripts/build_db.py
  └─ loader.py          downloads OSAP + French → polars DataFrames (percent → decimal)
  └─ store.py           upserts into DuckDB at ~/.factor_zoo/factors.db
  └─ analytics/stats.py computes all perf stats per factor → stored back in factors table

FactorZoo (api.py)
  └─ data/remote.py     ensure_db() auto-downloads pre-built DB on first use
  └─ data/store.py      read-only DuckDB queries → pandas
  └─ analytics/*        called by API wrapper methods (decay, portfolio, exposure, cluster…)
```

### Key invariants

- **All returns are decimal** — OSAP delivers percent, `loader.py` divides by 100 before storing. Never re-divide.
- **DB is read-only at API time** — `FactorZoo.__init__` opens `read_only=True`. Only `build_db.py` and `update()` write.
- **Stats are precomputed** — `get_stats()` reads from the `factors` table, not from `factor_returns`. Stats are recomputed only when `build_db.py` runs.
- **DB path** — defaults to `~/.factor_zoo/factors.db`, overridable via `FACTOR_ZOO_DB` env var.
- **Schema migrations don't run at API time** — `init_schema()` only runs on writable connections (`build_db.py`). SQL in analytics modules that references new columns must wrap queries in try/except with a NULL fallback for backward compatibility with older DBs.
- **OSAP download cache** — raw parquet files cached at `~/.factor_zoo/cache/{prefix}_{YYYYMMDD}.parquet`. Pass `max_age_days=0` to force re-download.

### Two DuckDB tables

`factors` — one row per factor with metadata + precomputed stats (`ann_return`, `sharpe`, `t_stat`, `pre_pub_sharpe`, `post_pub_sharpe`, `paper_t_stat`, etc.)

`factor_returns` — long-format monthly returns (`factor_id`, `date`, `ls_return`). `read_returns_wide()` pivots using pandas (not DuckDB PIVOT, for version compatibility).

`factor_quintiles` — quintile portfolio returns (`factor_id`, `date`, `q1`–`q5`, decimal). Populated by `build_db.py` from OSAP. Read via `store.read_quintiles()`.

### OSAP data specifics

- `Cat.Signal == 'Predictor'` filters to ~212 published factors (out of 331 total signals)
- `port == 'LS'` selects the long-short portfolio from `dl_port('op', ...)`
- Publication year comes from OSAP's signal doc `year` column → stored as `factors.year`
- `paper_t_stat` is OSAP's `t.stat` column; NULL for French factors

### Analytics modules

Each module is a pure function + dataclass — no DB access. The `FactorZoo` API methods in `api.py` fetch data from the DB, then call the analytics functions:

| Module | Key function | Returns |
|---|---|---|
| `analytics/decay.py` | `compute_decay(returns, factor_id, pub_year)` | `DecayResult` |
| `analytics/portfolio.py` | `construct_portfolio(wide_df, method, weights)` | `PortfolioResult` |
| `analytics/replication.py` | `replication_score(conn, factor_id)` | `dict` |
| `analytics/cluster.py` | `cluster_factors(wide_df, n_clusters, category_map)` | `ClusterResult` |
| `analytics/exposure.py` | `compute_exposure(user_returns, factor_wide)` | `ExposureResult` |
| `analytics/stats.py` | `compute_all_stats(returns, pub_year)` | `dict` |
| `analytics/correlation.py` | `correlation_matrix(wide_df)`, `rolling_correlation(s1, s2, window)` | `pd.DataFrame` / `pd.Series` |
| `analytics/quintiles.py` | `compute_quintile_analysis(factor_id, quintiles_df)` | `QuintileResult` |
| `analytics/drawdown.py` | `compute_drawdown(returns, factor_id)` | `DrawdownResult` |

### Tests

Tests in `tests/` use in-memory DuckDB fixtures from `conftest.py` (`mem_conn` fixture). The `test_api.py` fixture creates a real temp-file DB and monkeypatches `ensure_db` to avoid network calls. All analytics module tests are pure (no DB needed).

### Pyright

Pyright false positives from `polars`, `openassetpricing`, and `streamlit` stubs are expected and safe to ignore. Correctness is verified by the test suite (`uv run pytest`).

### Roadmap

- **v0.2.0** — complete. Analytics API (decay, portfolio, exposure, clustering, replication), CLI, auto-download DB.
- **v0.3.0** — complete. Multipage Streamlit app (8 pages in `factor_zoo/pages/`), quintile returns, OSAP download cache. See `docs/superpowers/specs/2026-05-01-factorzoo-v030-design.md`.
- **v0.4.0** — complete. Drawdown analytics, rolling correlation panel, Quintile Analysis page (`9_Quintiles.py`), Streamlit Cloud deployment config. See `docs/superpowers/specs/2026-05-07-factorzoo-v040-design.md`.
- **v0.5.0** — planned. AQR data source, conditional factor performance/macro regimes, factor timing signals, CSV/PDF export, research quality badge system, international factors.

### Adding a new data source

1. Add loader function to `loader.py` → returns `(meta_df, returns_df)` as polars DataFrames
2. Ensure `returns_df` has columns `factor_id` (Utf8), `date` (Date), `ls_return` (Float64, decimal)
3. Ensure `meta_df` columns match `_FACTORS_COLS` in `store.py`
4. Call `upsert_factors` + `upsert_returns` from `build_db.py`
5. `build_db.py` is fully idempotent — safe to re-run

## Architecture map (graphify)

`graphify-out/GRAPH_REPORT.md` contains a pre-built knowledge graph of this repo. Read it before any codebase exploration, architecture questions, or "where does X live" lookups — it's faster than grepping.

**God nodes** (highest centrality — start here):
- `FactorZoo` — central hub bridging API, CLI, tests, and data layers
- `PortfolioResult`, `DecayResult`, `QuintileResult`, `ExposureResult`, `DrawdownResult` — result dataclasses connecting analytics to tests and UI
- `construct_portfolio()` — bridge between Portfolio, Analytics, and DB layers
- `get_conn()` — Streamlit's shared DB entry point

**Community map** (major clusters):
- Analytics Results & Dataclasses · DuckDB Store & Remote Access · Performance Statistics
- FactorZoo API Layer · OSAP Data Loading & Caching · Streamlit App & Build Pipeline
- Portfolio Construction · Factor Decay Analysis · Replication & Zoo Summary
- Factor Clustering & Correlation · Quintile Returns API · Factor Exposure Analysis
- Command Line Interface · French Data Loader · Quintiles DB Layer
- v0.4.0 Design & Planning Docs · Rolling Correlation · Publication Decay Concepts

To query the graph: `/graphify query "<question>"` or `/graphify explain "<node>"`.
To rebuild after major changes: `/graphify --update`.

## Releasing a new version

1. Bump the version in `pyproject.toml` (e.g., `0.4.0`)
2. Update `factor_zoo/app.py` — change the "API Version" metric on the landing page
3. Build the DB: `uv run python scripts/build_db.py`
4. Build the package: `uv build`
5. Publish to PyPI: `uv publish`
6. Tag the release on GitHub: `git tag v0.4.0 && git push --tags`
7. Upload the built DB file (`~/.factor_zoo/factors.db`) as a release asset on the GitHub Release tagged `v0.4.0`
   - This is a manual step — the file is ~200 MB
   - `ensure_db()` in `data/remote.py` must point to this new release URL
8. Update the `ensure_db()` download URL in `factor_zoo/data/remote.py` to the new release asset URL
