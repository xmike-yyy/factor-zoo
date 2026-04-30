# factorzoo v0.2.0 Design Spec

**Date:** 2026-04-30  
**Status:** Approved  

## Vision

factorzoo v0.2.0 becomes two things at once:

1. **A factor zoo research toolkit** — the best tool for evaluating published academic equity factors: replication scores, decay curves, correlation clustering, and aggregate zoo-level statistics.
2. **A portfolio factor exposure analyzer** — users bring their own return series and measure which factors explain their portfolio's performance.

This positions the library for an audience of academics, students, and quant practitioners doing research — not yet live trading (that's v0.3.0+, requiring stock-level signals).

## Not In Scope

- Hosted web app (v0.3.0)
- Stock-level factor signals (requires WRDS / Compustat)
- AQR data source integration
- Quintile portfolio returns
- International factors

---

## Section 1: Distribution & Packaging

### PyPI packaging

`pyproject.toml` already uses hatchling. Changes needed:
- Add `[project.scripts]` entry point: `factorzoo = "factor_zoo.cli:app"`
- Add PyPI metadata: `description`, `keywords`, `classifiers`, `homepage`
- Pin new dependencies: `scipy`, `statsmodels`, `scikit-learn`

### Pre-built DB auto-download

New file: `factor_zoo/data/remote.py`

- `DB_VERSION` constant — matches the installed package version, used to construct the GitHub Releases download URL (e.g. `https://github.com/<owner>/factorzoo/releases/download/v0.2.0/factors-v0.2.0.db`)
- `ensure_db()` — called by `FactorZoo.__init__()` if `~/.factor_zoo/factors.db` does not exist. Downloads the versioned `.db` file with a `tqdm` progress bar. Writes atomically (temp file → rename) to avoid corrupt partial downloads.
- `check_for_update()` — hits the OSAP signal doc API, compares signal count against local DB. Returns `{"new_signals": [...], "update_available": bool}`.

`FactorZoo.__init__()` updated to call `ensure_db()` on construction. Raises a clear error with instructions if download fails.

### Update mechanism

- `FactorZoo.update(check_only=False)` — calls `check_for_update()`. If `check_only=True`, returns the result dict without modifying anything. If new signals found, runs the relevant parts of `build_db.main` and upserts them.
- `factorzoo update` CLI wraps `FactorZoo().update()`.
- `factorzoo build` CLI forces a full local rebuild from OSAP + French (existing `scripts/build_db.py` behavior, promoted to first-class command).

### DB hosting

Pre-built DB published as a GitHub Release asset on each version tag. File naming: `factors-v{version}.db`. Size estimate: ~50–100MB. URL pinned in `remote.py` via `DB_VERSION`.

---

## Section 2: Factor Decay Analysis

New file: `factor_zoo/analytics/decay.py`

### API

```python
decay = fz.decay("momentum_12_1")

decay.rolling_sharpe    # pd.Series — 36-month rolling Sharpe over time
decay.half_life         # float — months until post-pub Sharpe drops 50%
decay.pre_pub_sharpe    # float
decay.post_pub_sharpe   # float
decay.decay_rate        # float — slope of Sharpe decline post-publication
decay.plot()            # plotly Figure: rolling Sharpe + publication date marker
```

### Implementation

- Rolling Sharpe computed using existing `rolling_correlation` window logic pattern from `correlation.py`, applied to a 36-month window.
- Half-life estimated by fitting an exponential decay curve to the rolling Sharpe series post-publication using `scipy.optimize.curve_fit`. Formula: `S(t) = S0 * exp(-lambda * t)`. Half-life = `ln(2) / lambda`.
- If fewer than 24 months of post-publication data exist, `half_life` is `None` and a warning is surfaced.
- Returns `DecayResult` dataclass.

### DecayResult dataclass

```python
@dataclass
class DecayResult:
    factor_id: str
    rolling_sharpe: pd.Series
    half_life: float | None
    decay_rate: float | None
    pre_pub_sharpe: float
    post_pub_sharpe: float
    publication_year: int | None
    def plot(self) -> go.Figure: ...
```

---

## Section 3: Portfolio Construction

New file: `factor_zoo/analytics/portfolio.py`

### API

```python
port = fz.portfolio(["accruals", "momentum_12_1", "roe"], method="equal")
port = fz.portfolio(["accruals", "momentum_12_1"], weights=[0.3, 0.7])
port = fz.portfolio(["accruals", "momentum_12_1", "roe"], method="max_sharpe")
port = fz.portfolio(["accruals", "momentum_12_1", "roe"], method="risk_parity")
```

### Implementation

- **equal** — equal weight across all factors.
- **custom** — user-supplied weights list, must sum to 1.0.
- **max_sharpe** — `scipy.optimize.minimize` with negative Sharpe as objective, constraints: weights sum to 1, all weights >= 0.
- **risk_parity** — inverse-volatility weighting: `w_i = (1 / vol_i) / sum(1 / vol_j)`.
- Returns are trimmed to the intersection of available dates across all factors. Warns if intersection < 60 months.

### PortfolioResult dataclass

```python
@dataclass
class PortfolioResult:
    returns: pd.Series
    weights: dict[str, float]
    stats: dict          # same format as get_stats()
    factor_stats: pd.DataFrame   # per-factor stats for comparison
```

---

## Section 4: Factor Screening

Added to `factor_zoo/api.py` as `FactorZoo.screen()`.

### API

```python
df = fz.screen(
    min_sharpe=0.5,
    min_t_stat=2.0,
    max_correlation=0.6,
    categories=["value", "momentum"],
    min_years=10,
)
```

### Implementation

- Filters `list_factors()` result by scalar thresholds first.
- Then applies correlation-based deduplication: iteratively removes the factor with lower Sharpe from any pair exceeding `max_correlation`. Uses `read_returns_wide()` + `correlation_matrix()` — already in the codebase.
- Returns a `pd.DataFrame` with a `cluster_label` column added if `cluster_factors()` has been run.

---

## Section 5: Replication Analysis (Option 1)

New file: `factor_zoo/analytics/replication.py`

### fz.replication_score(factor_id)

OSAP's signal doc includes the paper's reported t-stat (`t.stat` column). `build_db.py` will be updated to store this as `paper_t_stat` in the `factors` table (requires schema migration — add nullable `paper_t_stat FLOAT` column).

```python
fz.replication_score("accruals")
# → {
#     "paper_t_stat": 4.2,
#     "computed_t_stat": 3.1,
#     "replication_score": 0.74,
#     "verdict": "partial"   # "strong" >= 0.8, "partial" 0.5-0.8, "weak" < 0.5
# }
```

Verdict thresholds: `strong` ≥ 0.8, `partial` 0.5–0.8, `weak` < 0.5.

### fz.zoo_summary()

Returns aggregate stats about the factor zoo — the entry point for meta-research:

```python
fz.zoo_summary()
# → {
#     "n_factors": 218,
#     "pct_positive_post_pub_sharpe": 0.61,
#     "median_replication_score": 0.71,
#     "median_decay_rate": -0.42,
#     "t_stat_distribution": {"mean": 3.8, "median": 3.2, "pct_above_3": 0.54},
#     "most_correlated_pairs": [("mom12m", "mom6m", 0.91), ...],
#     "most_independent_factors": ["accruals", "roe", "ivol", ...],  # lowest avg pairwise correlation
# }
```

---

## Section 6: Correlation Clustering (Option 1)

New file: `factor_zoo/analytics/cluster.py`

### fz.cluster_factors(n_clusters=10)

Groups factors by correlation structure using agglomerative hierarchical clustering (Ward linkage) via `sklearn.cluster.AgglomerativeClustering`. Input is the pairwise correlation matrix from `correlation_matrix()`.

```python
clusters = fz.cluster_factors(n_clusters=8)
# → {
#     "cluster_0": {"label": "Short-term Reversal", "factors": ["ret_1_0", ...]},
#     "cluster_1": {"label": "Earnings Quality", "factors": ["accruals", "noa", ...]},
#     ...
# }
```

Cluster labels are auto-generated by inspecting the OSAP category distribution within each cluster and picking the plurality category name. Falls back to `"cluster_N"` if no single category has plurality. Users can override labels manually.

Returns a `ClusterResult` with `.assignments` (factor → cluster id dict), `.dendrogram_data` (for plotting), and `.plot()` (plotly heatmap sorted by cluster).

---

## Section 7: Portfolio Exposure Analysis (Option 3)

New file: `factor_zoo/analytics/exposure.py`

### fz.exposure(returns, factors=None)

User provides their own monthly portfolio return series as a `pd.Series` with a `DatetimeIndex`. We run OLS regression of portfolio returns on factor returns.

```python
my_returns = pd.Series(...)   # user's own monthly returns

exposure = fz.exposure(my_returns, factors=["mom12m", "bm", "roe", "accruals"])
# → ExposureResult:
#     .loadings       # {"mom12m": 0.43, "bm": -0.12, ...}
#     .t_stats        # significance of each loading
#     .p_values       # p-values per factor
#     .r_squared      # float
#     .alpha          # annualized unexplained return
#     .alpha_t_stat   # significance of alpha
#     .plot()         # plotly bar chart of loadings with confidence intervals
```

If `factors=None`, runs against all factors in the DB and returns top 10 by absolute t-stat. Overlap period between `returns` and factor data is used (warn if < 36 months).

### Implementation

- Uses `statsmodels.OLS` with `add_constant` for the alpha term.
- Factor returns fetched via `read_returns_wide()`, trimmed to overlap with `returns.index`.
- `ExposureResult` is a dataclass.

### ExposureResult dataclass

```python
@dataclass
class ExposureResult:
    loadings: dict[str, float]
    t_stats: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    alpha: float
    alpha_t_stat: float
    overlap_months: int
    def plot(self) -> go.Figure: ...
```

---

## Section 8: CLI

New file: `factor_zoo/cli.py` using `click` (added as dependency).

```
factorzoo list [--category value] [--min-sharpe 0.5] [--limit 20]
factorzoo detail <factor_id>
factorzoo update [--check-only]
factorzoo build
factorzoo zoo-summary
```

- `list` — tabular output of `fz.list_factors()` with key stats columns.
- `detail` — prints `fz.get_stats()` + `fz.replication_score()` for a single factor.
- `update` — runs `fz.update()`.
- `build` — full local rebuild from OSAP + French.
- `zoo-summary` — prints `fz.zoo_summary()`.

---

## Section 9: README

Covers:
1. One-paragraph intro ("what is the factor zoo")
2. Install: `pip install factorzoo`
3. Quick start: `FactorZoo()`, `list_factors()`, `get_returns()`, `decay()` in 10 lines
4. Portfolio exposure example
5. Zoo summary example
6. API reference link
7. Data attribution (OSAP, Ken French Data Library)
8. Contributing guide stub

---

## New Files Summary

```
factor_zoo/
  data/
    remote.py              # DB download + update checking
  analytics/
    decay.py               # DecayResult, rolling Sharpe, half-life
    portfolio.py           # PortfolioResult, construction methods
    replication.py         # replication_score(), zoo_summary()
    cluster.py             # ClusterResult, hierarchical clustering
    exposure.py            # ExposureResult, OLS factor exposure
  cli.py                   # click CLI entry points
docs/
  superpowers/specs/
    2026-04-30-factorzoo-v020-design.md   # this file
```

## Modified Files Summary

```
factor_zoo/api.py          # add portfolio(), decay(), screen(), exposure(),
                           #     replication_score(), zoo_summary(), cluster_factors(),
                           #     update()
factor_zoo/data/store.py   # add paper_t_stat reads; wide-format reads for exposure
factor_zoo/data/schema.sql # add paper_t_stat FLOAT column to factors table
scripts/build_db.py        # store paper t-stat from OSAP signal doc
pyproject.toml             # add scipy, statsmodels, scikit-learn, click; CLI entry point
README.md                  # full quickstart
```

## Dependencies Added

| Package | Use |
|---|---|
| `scipy` | Max Sharpe optimization, decay curve fitting |
| `statsmodels` | OLS regression for exposure analysis |
| `scikit-learn` | Agglomerative clustering for cluster_factors() |
| `click` | CLI framework |

---

## Schema Migration

One schema change: add `paper_t_stat FLOAT` (nullable) to the `factors` table. `build_db.py` will populate this from the `t.stat` column in the OSAP signal doc. Existing DBs missing this column will have `NULL` values; `replication_score()` returns `{"error": "paper_t_stat unavailable — run factorzoo build to refresh"}` in that case.
