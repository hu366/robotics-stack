#!/usr/bin/env bash
# Demo script for T2: Maintain contact normal force
# M0 stub - to be implemented in M2

set -e

echo "Demo T2: Maintain contact normal force"
echo "======================================="
echo "M0 stub - Running smoke test..."

uv run python -c "
import mujoco
import numpy as np
print('MuJoCo version:', mujoco.__version__)
print('NumPy version:', np.__version__)
print('T2 demo: Force control not yet implemented (M2)')
"

echo ""
echo "Demo complete (M0 stub)"
