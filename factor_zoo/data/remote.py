"""Remote DB download and update checking."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from factor_zoo.data.store import db_path

DB_VERSION = "0.2.0"
_REPO = "michaelsun/factorzoo"
_DB_URL = f"https://github.com/{_REPO}/releases/download/v{DB_VERSION}/factors-v{DB_VERSION}.db"

_MISSING_MSG = """
Factor Zoo database not found at: {path}

Attempting to download the pre-built database...
If the download fails, build it locally:
    uv run python scripts/build_db.py
"""


def ensure_db(path: Path | None = None) -> None:
    """Download the pre-built DB if it does not already exist."""
    target = Path(path) if path else db_path()
    if target.exists():
        return

    print(_MISSING_MSG.format(path=target))
    target.parent.mkdir(parents=True, exist_ok=True)
    _download_db(target)


def _download_db(target: Path) -> None:
    """Download versioned .db file to target path. Writes atomically."""
    try:
        resp = requests.get(_DB_URL, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download Factor Zoo database from {_DB_URL}.\n"
            f"Error: {exc}\n\n"
            "Build it locally instead:\n"
            "    uv run python scripts/build_db.py"
        ) from exc

    total = int(resp.headers.get("content-length", 0))
    tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=target.parent, suffix=".tmp"
    )
    try:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading factors DB") as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
                pbar.update(len(chunk))
        tmp.close()
        Path(tmp.name).rename(target)
        print(f"Database saved to {target}")
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise


def check_for_update() -> dict[str, Any]:
    """Check OSAP signal doc for new signals not in the local DB.

    Returns dict with keys:
        new_signals: list[str]  — signal Acronyms not yet in local DB
        update_available: bool
    """
    try:
        import openassetpricing as oap
        import duckdb as _duckdb

        openap = oap.OpenAP()
        remote_doc = openap.dl_signal_doc("polars")
        import polars as pl
        remote_doc = remote_doc.filter(pl.col("Cat.Signal") == "Predictor")
        remote_ids = set(remote_doc["Acronym"].to_list())

        conn = _duckdb.connect(str(db_path()), read_only=True)
        local_ids = set(
            r[0] for r in conn.execute(
                "SELECT id FROM factors WHERE source = 'osap'"
            ).fetchall()
        )
        conn.close()

        new = sorted(remote_ids - local_ids)
        return {"new_signals": new, "update_available": len(new) > 0}
    except Exception as exc:
        return {"new_signals": [], "update_available": False, "error": str(exc)}
