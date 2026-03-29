#!/usr/bin/env bash
# Demo script for T3: Wipe rectangle with force control
# M0 stub - to be implemented in M3

set -e

echo "Demo T3: Wipe rectangle with force control"
echo "==========================================="
echo "M0 stub - Running smoke test..."

uv run python -c "
import mujoco
import numpy as np
print('MuJoCo version:', mujoco.__version__)
print('NumPy version:', np.__version__)
print('T3 demo: Wipe behavior not yet implemented (M3)')
"

echo ""
echo "Demo complete (M0 stub)"
