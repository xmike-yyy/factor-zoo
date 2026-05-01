#!/bin/bash
# Launch the Factor Zoo Browser.
# Run from the repo root: ./run.sh
set -e
cd "$(dirname "$0")"
uv run streamlit run factor_zoo/app.py "$@"
