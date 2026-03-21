#!/usr/bin/env bash
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv (--clear replaces any cached venv from previous builds)
uv venv .venv --python 3.12 --clear

# Activate venv — sets VIRTUAL_ENV so uv pip install targets it correctly
source .venv/bin/activate

# Install project dependencies from pyproject.toml
uv pip install .

# Download spaCy model into the active venv
python -m spacy download en_core_web_md
