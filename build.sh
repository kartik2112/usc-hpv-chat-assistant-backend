#!/usr/bin/env bash
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install dependencies
uv venv /opt/render/project/src/.venv
export PATH="/opt/render/project/src/.venv/bin:$PATH"

uv pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md
