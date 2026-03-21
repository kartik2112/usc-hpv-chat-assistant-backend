#!/usr/bin/env bash
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies with uv
uv pip install -r requirements.txt --system

# Download spaCy model
python -m spacy download en_core_web_md
