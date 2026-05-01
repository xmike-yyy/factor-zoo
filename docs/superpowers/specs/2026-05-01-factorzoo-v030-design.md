# factorzoo v0.3.0 Design Spec

**Date:** 2026-05-01  
**Status:** Draft  

## Vision

v0.3.0 completes the feedback loop between the Python library and the Streamlit UI. v0.2.0 built a deep analytics API (decay, portfolio, exposure, clustering) that the UI never exposes. v0.3.0 wires them together, adds quintile returns for richer analysis, and makes the project deployable as a hosted community tool.

**Theme: from research library → research platform.**

## Not In Scope

- AQR data source integration (v0.4.0)
- Stock-level signals / WRDS integration (v0.4.0)
- International factors (v0.4.0)
- User accounts, saved portfolios, or any persistent server-side state
- Real-time market data

---

## Section 1: Streamlit App — Analytics Pages

The current app has three pages: Browse, Detail, Compare. None of the v0.2.0 analytics (decay, portfolio, exposure, clustering) appear in the UI. This section adds four new pages.

### 1.1 Migrate to Streamlit `pages/` structure

Replace the `session_state`-based routing in `app.py` with Streamlit's built-in multipage structure. This gives each page a real URL, back-button support, and cleaner code.

**New file layout:**
```
factor_zoo/
  app.py                  # entrypoint — sidebar nav only, no page logic
  pages/
    1_Browse.py           # existing page_browse() → moved here
    2_Detail.py           # existing page_detail() → moved here
    3_Compare.py          # existing page_compare() → moved here
    4_Decay.py            # new
    5_Portfolio.py        # new
    6_Exposure.py         # new
    7_Clustering.py       # new
    8_Zoo_Summary.py      # new
```

Shared state (DB connection, cached DataFrames) stays in `app.py` via `st.session_state`. Each page imports helpers from `app.py` — `get_conn()`, `load_all_factors()`, `load_returns()`, etc.

### 1.2 Decay Page (`pages/4_Decay.py`)

```
Factor selector (searchable dropdown)
  ↓
Rolling Sharpe chart (plotly, from DecayResult.plot())
  ↓
Stats panel: pre/post Sharpe, half-life, decay rate
  ↓
"Compare decay" — add up to 4 factors, overlay rolling Sharpe series
```

Implementation:
- Call `fz.decay(factor_id)` on selection
- `DecayResult.plot()` already returns a plotly Figure → `st.plotly_chart()`
- If `half_life` is None (< 24 months post-pub data), show an info banner: "Insufficient post-publication data to estimate half-life."
- Cache per factor with `@st.cache_data`

### 1.3 Portfolio Builder Page (`pages/5_Portfolio.py`)

```
Multi-factor selector (checkbox list or multiselect)
Method radio: Equal | Risk Parity | Max Sharpe | Custom weights
  ↓ (for Custom: weight sliders that sum-to-1, validated live)
Construct button
  ↓
Cumulative return chart vs. equal-weighted benchmark
Stats table: portfolio + per-factor side-by-side (from PortfolioResult.factor_stats)
Weights bar chart
Download CSV button (portfolio returns + weights)
```

Implementation:
- Call `fz.portfolio(factor_ids, method=method, weights=weights)`
- Warn inline (via `st.warning`) when intersection < 60 months — surfaced from the UserWarning `construct_portfolio` already emits, caught at the page level
- `st.download_button` for CSV export of `PortfolioResult.returns`

### 1.4 Exposure Analyzer Page (`pages/6_Exposure.py`)

```
Upload CSV (date, return) or paste returns as text
  ↓ (parse and validate)
Optional: restrict to specific factors (multiselect, default = top 10 auto)
Run Exposure button
  ↓
Bar chart: loadings with confidence intervals (ExposureResult.plot())
Table: factor, loading, t-stat, p-value
Alpha / R² summary panel
```

Implementation:
- Accept CSV with columns `date` (YYYY-MM-DD) and `return` (decimal or percent — auto-detect by checking if abs(mean) > 0.1)
- Parse to `pd.Series` with `DatetimeIndex`, call `fz.exposure(returns, factors=selected or None)`
- Show overlap period prominently: "Using 48 months of overlapping data (2010–2014)"
- If overlap < 36 months, show `st.warning`

### 1.5 Clustering Page (`pages/7_Clustering.py`)

```
n_clusters slider (2–20, default 10)
Cluster button
  ↓
Correlation heatmap sorted by cluster (ClusterResult.plot())
Cluster table: cluster label, factors, count
"Drill into cluster": click a cluster → show its factors' cumulative return overlay
```

Implementation:
- `fz.cluster_factors(n_clusters=n)` — runs on all ~218 factors; cache with `@st.cache_data(hash_funcs=...)`
- Heatmap already available from `ClusterResult.plot()`
- Cluster table is `ClusterResult.clusters` dict → `st.dataframe`

### 1.6 Zoo Summary Page (`pages/8_Zoo_Summary.py`)

```
Auto-loads on page visit (no inputs needed)
  ↓
Key metrics: n_factors, % positive post-pub Sharpe, median replication score, median t-stat
Distribution charts:
  - t-stat histogram
  - Sharpe histogram
  - Replication score histogram (factors with paper_t_stat only)
  - Pre vs. post-publication Sharpe scatter
Most correlated pairs table (top 20)
Most independent factors table (top 10 by lowest avg pairwise correlation)
```

Implementation:
- `fz.zoo_summary()` for the headline metrics
- Histograms built from `fz.list_factors()` which returns all stats columns
- Most-correlated pairs: compute from full correlation matrix — expensive, cache aggressively
- `zoo_summary()` needs two new fields: `most_correlated_pairs` and `most_independent_factors`

**`zoo_summary()` additions** (extend `replication.py`):

```python
fz.zoo_summary()
# adds to existing dict:
# "most_correlated_pairs": [{"factor_a": "Mom12m", "factor_b": "Mom6m", "correlation": 0.91}, ...]  # top 20
# "most_independent_factors": ["Accruals", "ROE", "IVol", ...]  # 10 lowest avg pairwise corr
```

---

## Section 2: Quintile Portfolio Returns

OSAP provides quintile value-weighted returns via `dl_port('quintiles_vw', 'polars')`. Adding these enables long-only backtesting and quintile spread analysis — a standard academic tool.

### New table: `factor_quintiles`

```sql
CREATE TABLE IF NOT EXISTS factor_quintiles (
    factor_id VARCHAR,
    date DATE,
    q1 FLOAT,   -- bottom quintile (value-weighted monthly return, decimal)
    q2 FLOAT,
    q3 FLOAT,
    q4 FLOAT,
    q5 FLOAT,   -- top quintile
    PRIMARY KEY (factor_id, date)
);
```

### Loader change (`loader.py`)

Add `load_osap_quintiles() -> pl.DataFrame` alongside `load_osap_returns()`. Same OSAP call, different `port` argument:

```python
quintiles = openap.dl_port("quintiles_vw", "polars")
# Filter to Predictor signals, pivot q1–q5 into columns, divide by 100
```

### Store change (`store.py`)

Add `upsert_quintiles(conn, df)` and `read_quintiles(conn, factor_id) -> pd.DataFrame`.

`read_quintiles` returns a DataFrame with columns `date`, `q1`, `q2`, `q3`, `q4`, `q5`, indexed by date.

### API additions (`api.py`)

```python
fz.get_quintiles("Mom12m")
# → pd.DataFrame — columns q1–q5, DatetimeIndex

fz.quintile_spread("Mom12m")
# → pd.Series — q5 - q1 monthly return (decimal), DatetimeIndex
# This is a richer signal than pure LS returns: shows monotonicity of the factor
```

### `QuintileResult` dataclass (`analytics/quintiles.py`, new)

```python
@dataclass
class QuintileResult:
    factor_id: str
    quintiles: pd.DataFrame          # q1–q5 columns
    spread: pd.Series                # q5 - q1
    monotonicity_score: float        # fraction of periods where q1<q2<q3<q4<q5 (0–1)
    spread_sharpe: float
    def plot(self) -> go.Figure: ... # 5-line cumulative return chart, colored q1→q5
```

```python
fz.quintile_analysis("Mom12m")
# → QuintileResult
```

---

## Section 3: Streamlit Cloud Deployment

Make the app one-click deployable to Streamlit Community Cloud (free tier).

### 3.1 `secrets.toml` support

Streamlit Cloud injects secrets via `st.secrets`. Add to `app.py`:

```python
import os
if hasattr(st, "secrets") and "FACTOR_ZOO_DB" in st.secrets:
    os.environ["FACTOR_ZOO_DB"] = st.secrets["FACTOR_ZOO_DB"]
```

For hosted deployment, the DB path points to a mounted volume or a pre-downloaded file in the repo. Document both options in README.

### 3.2 Bootstrap script for hosted use

New file: `scripts/bootstrap_for_cloud.py`

Downloads the pre-built DB (via `ensure_db()`) to `~/.factor_zoo/factors.db` before the app starts. Called from a `packages.txt` or Streamlit Cloud startup hook.

Add to `README.md`:
```
## Deploying to Streamlit Cloud
1. Fork this repo
2. Connect to share.streamlit.io
3. Set main file: factor_zoo/app.py
4. Add secret FACTOR_ZOO_DB=~/.factor_zoo/factors.db (optional, default used if absent)
5. The app auto-downloads the pre-built database on first start
```

### 3.3 `.streamlit/config.toml`

Add to the repo:
```toml
[theme]
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#1e293b"

[server]
maxUploadSize = 10   # MB — sufficient for return series CSV uploads on Exposure page
```

---

## Section 4: Build Performance

`build_db.py` re-downloads OSAP data (~200MB) every time it runs. Add a cache layer so repeat builds skip the download if the data is recent.

### Cache location: `~/.factor_zoo/cache/`

```
~/.factor_zoo/cache/
  osap_returns_YYYYMMDD.parquet   # OSAP long-short returns (dated by download day)
  osap_signals_YYYYMMDD.parquet   # OSAP signal doc metadata
  osap_quintiles_YYYYMMDD.parquet # OSAP quintile returns (new in v0.3.0)
```

### Cache logic in `loader.py`

```python
def load_osap_returns(max_age_days: int = 7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load OSAP returns, using cache if < max_age_days old."""
    cache = _find_cache("osap_returns", max_age_days)
    if cache:
        return pl.read_parquet(cache["returns"]), pl.read_parquet(cache["signals"])
    returns, signals = _download_osap()
    _save_cache("osap_returns", returns, signals)
    return returns, signals
```

`build_db.py` passes `--no-cache` flag to force a fresh download:
```
uv run python scripts/build_db.py --no-cache
```

### Cache invalidation

Files older than `max_age_days` are ignored. A new download replaces the old file (not accumulates). Old cache files (> 30 days) are deleted automatically on each run.

---

## New Files Summary

```
factor_zoo/
  pages/
    1_Browse.py
    2_Detail.py
    3_Compare.py
    4_Decay.py
    5_Portfolio.py
    6_Exposure.py
    7_Clustering.py
    8_Zoo_Summary.py
  analytics/
    quintiles.py          # QuintileResult, quintile_analysis()
.streamlit/
  config.toml
scripts/
  bootstrap_for_cloud.py
```

## Modified Files Summary

```
factor_zoo/app.py              # strip page logic → entrypoint only
factor_zoo/api.py              # add get_quintiles(), quintile_spread(), quintile_analysis()
factor_zoo/data/schema.sql     # add factor_quintiles table
factor_zoo/data/store.py       # add upsert_quintiles(), read_quintiles()
factor_zoo/data/loader.py      # add load_osap_quintiles(), cache layer
factor_zoo/analytics/replication.py  # zoo_summary() adds correlated pairs + independent factors
scripts/build_db.py            # wire quintile loading, --no-cache flag, cache cleanup
README.md                      # add Streamlit Cloud deployment section
```

## Dependencies Added

None — all required packages (`plotly`, `streamlit`, `pandas`, `polars`) already present.

---

## Schema Migration

One new table: `factor_quintiles`. Existing DBs missing this table will get it added automatically via `init_schema()` (uses `CREATE TABLE IF NOT EXISTS`). `get_quintiles()` raises `KeyError` with a helpful message if the factor has no quintile data (French factors won't have it).

---

## Metrics for "Done"

- All 8 Streamlit pages render without error on the local DB
- `fz.get_quintiles("Mom12m")` returns a non-empty DataFrame
- `fz.quintile_analysis("Mom12m")` returns a `QuintileResult` with a valid `monotonicity_score`
- `build_db.py` second run uses cache (verify via timing: should be <10s vs ~5min cold)
- App deploys to Streamlit Cloud with only README instructions (no manual steps beyond secrets)
- 150+ tests passing
