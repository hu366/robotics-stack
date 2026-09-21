# Control

Trajectory tracking, servo loops, and force or impedance control live here.

Absolute 6-DoF EE trajectories from ego2dex (`T_ee + g`) are converted with
the vendored Cheng Chi / UMI helpers in `modules/umi_pose/` — not reimplemented
here. Training and eval must both call `convert_pose_mat_rep`.

## Backends

- `SymbolicControlBackend`: deterministic baseline for pipeline regression.
- `MjctrlMPCBackend`: closed-loop receding-horizon controller inspired by
  `third_party/mjctrl` differential IK and operational-space formulations.

## CLI

- `uv run python apps/run_task.py "place the bottle on the tray" --control-backend mjctrl_mpc`
- `uv run python apps/run_benchmark.py --control-backend mjctrl_mpc`
