#!/usr/bin/env bash
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with Python 3.12 (--clear in case it already exists from cache)
uv venv /opt/render/project/src/.venv --python 3.12 --clear
export PATH="/opt/render/project/src/.venv/bin:$PATH"

# Install project dependencies from pyproject.toml
uv pip install .

# Download spaCy model
python -m spacy download en_core_web_md
