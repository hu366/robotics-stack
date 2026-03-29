#!/usr/bin/env bash
# Demo script for T1: Free-space reach & place
# M0 stub - to be implemented in M1

set -e

echo "Demo T1: Free-space reach & place"
echo "=================================="
echo "M0 stub - Running smoke test..."

uv run python -c "
import mujoco
import numpy as np
print('MuJoCo version:', mujoco.__version__)
print('NumPy version:', np.__version__)
print('T1 demo: Environment not yet implemented (M1)')
"

echo ""
echo "Demo complete (M0 stub)"
