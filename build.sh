#!/usr/bin/env bash
set -o errexit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install project dependencies from pyproject.toml
uv sync

# Download spaCy medium model (~40MB) for Presidio NER
uv run python -m spacy download en_core_web_md
uv run python -m spacy download es_core_news_md