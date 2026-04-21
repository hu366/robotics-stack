# Control

Trajectory tracking, servo loops, and force or impedance control live here.

## Backends

- `SymbolicControlBackend`: deterministic baseline for pipeline regression.
- `MjctrlMPCBackend`: closed-loop receding-horizon controller inspired by
  `third_party/mjctrl` differential IK and operational-space formulations.

## CLI

- `uv run python apps/run_task.py "place the bottle on the tray" --control-backend mjctrl_mpc`
- `uv run python apps/run_benchmark.py --control-backend mjctrl_mpc`
