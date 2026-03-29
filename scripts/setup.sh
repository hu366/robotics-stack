#!/usr/bin/env bash
# Setup script: create venv and install dependencies

set -e

echo "Setting up robotics-stack environment with uv..."

uv sync --extra dev

echo "Setup complete. Run tests with: uv run pytest"
