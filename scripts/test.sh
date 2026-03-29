#!/usr/bin/env bash
# Test script: run pytest and linting

set -e

echo "Running tests..."
uv run pytest "$@"

echo ""
echo "Running ruff lint..."
uv run ruff check .

echo ""
echo "All checks passed!"
