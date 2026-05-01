"""Download and normalize OSAP + Ken French data into polars DataFrames."""
import io
import re
import zipfile
from typing import Optional

import polars as pl
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Category mapping from OSAP's Cat.Economic → our schema categories
# ---------------------------------------------------------------------------

_ECON_TO_CATEGORY: dict[str, str] = {
    "valuation": "value",
    "earnings growth": "value",
    "momentum": "momentum",
    "short-term reversal": "momentum",
    "long term reversal": "momentum",
    "lead lag": "momentum",
    "profitability": "profitability",
    "profitability alt": "profitability",
    "accruals": "profitability",
    "composite accounting": "profitability",
    "cash flow risk": "profitability",
    "earnings event": "profitability",
    "earnings forecast": "profitability",
    "investment": "investment",
    "investment alt": "investment",
    "investment growth": "investment",
    "external financing": "investment",
    "asset composition": "investment",
    "R&D": "intangibles",
    "info proxy": "intangibles",
    "recommendation": "intangibles",
    "liquidity": "trading_frictions",
    "turnover": "trading_frictions",
    "volume": "trading_frictions",
    "short sale constraints": "trading_frictions",
    "volatility": "trading_frictions",
    "risk": "trading_frictions",
    "market risk": "trading_frictions",
    "optionrisk": "trading_frictions",
    "leverage": "other",
    "default risk": "other",
    "ownership": "other",
    "payout indicator": "other",
    "sales growth": "other",
    "size": "other",
    "informed trading": "trading_frictions",
    "other": "other",
}

# Known publication years for Ken French factors
_FRENCH_PUB_YEARS: dict[str, int] = {
    "MktRF": 1993,
    "SMB": 1993,
    "HML": 1993,
    "RMW": 2015,
    "CMA": 2015,
    "Mom": 1997,
    "UMD": 1997,
}

_FRENCH_DESCRIPTIONS: dict[str, str] = {
    "MktRF": "Market excess return (Rm - Rf). From Fama-French (1993).",
    "SMB": "Small Minus Big: return of small-cap stocks minus large-cap. Fama-French (1993).",
    "HML": "High Minus Low: value (high B/M) minus growth (low B/M). Fama-French (1993).",
    "RMW": "Robust Minus Weak: profitability factor. Fama-French (2015).",
    "CMA": "Conservative Minus Aggressive: investment factor. Fama-French (2015).",
    "Mom": "Momentum (UMD): Up Minus Down, 12-1 month prior returns. Carhart (1997).",
    "UMD": "Momentum (UMD): Up Minus Down, 12-1 month prior returns. Carhart (1997).",
}


def _download_bytes(url: str, desc: str = "") -> bytes:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc or url[:60]) as pbar:
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# OSAP
# ---------------------------------------------------------------------------

def load_osap_signal_doc() -> pl.DataFrame:
    """Download OSAP signal documentation and return normalized DataFrame."""
    import openassetpricing as oap  # only needed during DB build
    openap = oap.OpenAP()
    raw = openap.dl_signal_doc("polars")
    # Filter to published predictors only
    raw = raw.filter(pl.col("Cat.Signal") == "Predictor")

    def map_cat(econ: Optional[str]) -> str:
        if econ is None:
            return "other"
        return _ECON_TO_CATEGORY.get(str(econ).strip(), "other")

    doc = raw.with_columns([
        pl.col("Acronym").alias("id"),
        pl.col("Acronym").alias("name"),
        pl.col("Cat.Economic").map_elements(map_cat, return_dtype=pl.Utf8).alias("category"),
        pl.col("Cat.Economic").alias("subcategory"),
        pl.col("Authors").alias("authors"),
        pl.col("Year").cast(pl.Int32, strict=False).alias("year"),
        pl.col("Journal").alias("journal"),
        pl.col("SampleStartYear").cast(pl.Int32, strict=False).alias("sample_start_year"),
        pl.col("SampleEndYear").cast(pl.Int32, strict=False).alias("sample_end_year"),
        pl.col("LongDescription").alias("description"),
        pl.lit("osap").alias("source"),
        pl.lit(None).cast(pl.Utf8).alias("paper_url"),
        pl.col("t.stat").cast(pl.Float32, strict=False).alias("t_stat_paper"),
    ]).select([
        "id", "name", "category", "subcategory", "authors", "year", "journal",
        "paper_url", "sample_start_year", "sample_end_year", "description", "source",
        "t_stat_paper",
    ])
    return doc


def load_osap_returns() -> pl.DataFrame:
    """Download OSAP long-short portfolio returns for all predictors.
    Returns DataFrame with columns: factor_id, date, ls_return (decimal)."""
    import openassetpricing as oap
    openap = oap.OpenAP()
    print("Downloading OSAP portfolio returns (this may take a minute)...")
    raw = openap.dl_port("op", "polars")
    ls = raw.filter(pl.col("port") == "LS")
    result = ls.select([
        pl.col("signalname").alias("factor_id"),
        pl.col("date"),
        (pl.col("ret") / 100.0).alias("ls_return"),  # percent → decimal
    ])
    return result


# ---------------------------------------------------------------------------
# Ken French
# ---------------------------------------------------------------------------

_FRENCH_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
_FRENCH_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
_FRENCH_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


def _parse_french_csv(raw_bytes: bytes, zip_name: str) -> list:
    """Extract and parse a Ken French CSV from a zip file.

    French CSVs are comma-separated with rows like:
        192607,   2.89,  -2.55,  -2.39,   0.22
    A header row (e.g. ',Mkt-RF,SMB,HML,RF') precedes the data.
    Monthly data ends at the first blank line; annual data follows (4-digit year, ignored).

    Returns list of (year, month, [float, ...]) tuples with returns as decimals.
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        fname = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(fname).decode("latin-1")

    lines = text.splitlines()
    rows = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_data:
                break   # blank line ends the monthly section
            continue
        # Monthly data rows start with a 6-digit YYYYMM followed by a comma
        if re.match(r"^\d{6},", stripped):
            in_data = True
            parts = [p.strip() for p in stripped.split(",")]
            yyyymm = parts[0]
            year = int(yyyymm[:4])
            month = int(yyyymm[4:])
            values = [float(v) / 100.0 for v in parts[1:] if v]  # percent → decimal
            rows.append((year, month, values))

    if not rows:
        raise ValueError(f"No monthly data rows found in {zip_name}")
    return rows


def _french_rows_to_long(rows: list, factor_names: list[str]) -> pl.DataFrame:
    import calendar
    import datetime
    records = []
    for year, month, values in rows:
        # End-of-month date
        last_day = calendar.monthrange(year, month)[1]
        d = datetime.date(year, month, last_day)
        for fname, val in zip(factor_names, values):
            records.append({"date": d, "factor_id": fname, "ls_return": val})
    return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))


def load_french_returns() -> tuple[pl.DataFrame, list[str]]:
    """Download FF3, FF5, and Momentum factors from Ken French library.
    Returns (returns_df, factor_ids_list)."""
    print("Downloading Fama-French 3-factor data...")
    ff3_bytes = _download_bytes(_FRENCH_FF3_URL, "FF3")
    print("Downloading Fama-French 5-factor data...")
    ff5_bytes = _download_bytes(_FRENCH_FF5_URL, "FF5")
    print("Downloading Momentum factor data...")
    mom_bytes = _download_bytes(_FRENCH_MOM_URL, "Mom")

    ff3_rows = _parse_french_csv(ff3_bytes, "FF3")
    ff5_rows = _parse_french_csv(ff5_bytes, "FF5")
    mom_rows = _parse_french_csv(mom_bytes, "Mom")

    # FF3 columns: Mkt-RF, SMB, HML, RF — we rename Mkt-RF → MktRF, drop RF
    ff3_df = _french_rows_to_long(ff3_rows, ["MktRF", "SMB", "HML", "RF"])
    ff3_df = ff3_df.filter(pl.col("factor_id") != "RF")

    # FF5 adds RMW, CMA on top of FF3 columns — we only want the new ones
    ff5_df = _french_rows_to_long(ff5_rows, ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"])
    ff5_df = ff5_df.filter(pl.col("factor_id").is_in(["RMW", "CMA"]))

    # Mom CSV: Mom, RF — drop RF, rename to "Mom"
    mom_df = _french_rows_to_long(mom_rows, ["Mom", "RF"])
    mom_df = mom_df.filter(pl.col("factor_id") == "Mom")

    combined = pl.concat([ff3_df, ff5_df, mom_df])
    factor_ids = ["MktRF", "SMB", "HML", "RMW", "CMA", "Mom"]
    return combined, factor_ids


def build_french_metadata(factor_ids: list[str]) -> pl.DataFrame:
    """Build the factors table rows for Ken French factors."""
    rows = []
    for fid in factor_ids:
        pub_year = _FRENCH_PUB_YEARS.get(fid)
        rows.append({
            "id": fid,
            "name": fid,
            "category": _get_french_category(fid),
            "subcategory": None,
            "authors": _get_french_authors(fid),
            "year": pub_year,
            "journal": _get_french_journal(fid),
            "paper_url": None,
            "sample_start_year": None,
            "sample_end_year": None,
            "description": _FRENCH_DESCRIPTIONS.get(fid, ""),
            "source": "french",
        })
    return pl.DataFrame(rows)


def _get_french_category(fid: str) -> str:
    cats = {
        "MktRF": "other",
        "SMB": "other",
        "HML": "value",
        "RMW": "profitability",
        "CMA": "investment",
        "Mom": "momentum",
        "UMD": "momentum",
    }
    return cats.get(fid, "other")


def _get_french_authors(fid: str) -> str:
    if fid in ("MktRF", "SMB", "HML"):
        return "Fama, Eugene F. and French, Kenneth R."
    if fid in ("RMW", "CMA"):
        return "Fama, Eugene F. and French, Kenneth R."
    if fid in ("Mom", "UMD"):
        return "Carhart, Mark M."
    return ""


def _get_french_journal(fid: str) -> str:
    if fid in ("MktRF", "SMB", "HML"):
        return "JF"
    if fid in ("RMW", "CMA"):
        return "JFE"
    if fid in ("Mom", "UMD"):
        return "JF"
    return ""
