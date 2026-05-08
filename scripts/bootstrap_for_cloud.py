"""Bootstrap script for Streamlit Cloud: downloads pre-built DB if not present.

Configure as a startup command in Streamlit Cloud Advanced Settings:
    python scripts/bootstrap_for_cloud.py
"""
from factor_zoo.data.remote import ensure_db
from factor_zoo.data.store import db_path

if __name__ == "__main__":
    path = db_path()
    print(f"Ensuring database at {path} ...")
    ensure_db(path)
    print("Bootstrap complete.")
