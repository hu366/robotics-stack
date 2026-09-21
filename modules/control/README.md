# Control

Trajectory tracking, servo loops, and force or impedance control live here.

Absolute 6-DoF EE trajectories from ego2dex (`T_ee + g`) are converted with
the vendored Cheng Chi / UMI helpers in `modules/umi_pose/` — not reimplemented
here. Training and eval must both call `convert_pose_mat_rep`.
