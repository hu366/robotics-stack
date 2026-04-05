# Toolchain

This document defines the recommended baseline toolchain for the modular robotics stack in this repository.

## Implementation Status

Status labels used in this document:

- `[done]`: implemented in the current repository baseline
- `[partial]`: partially implemented or only configured
- `[todo]`: not implemented yet

Current repository status summary:

- `[done]` Python baseline with `uv`, `Ruff`, `mypy`, `pytest`
- `[done]` minimal end-to-end Python execution loop across parsing, grounding, planning, control, and tracing
- `[partial]` benchmark runner with local JSON cases
- `[partial]` Docker Compose workflow for Python app, benchmark, and tests
- `[todo]` ROS 2 Jazzy workspace
- `[todo]` MoveIt 2 / OMPL / Pinocchio integration
- `[todo]` ros2_control / ros2_controllers integration
- `[todo]` BehaviorTree.CPP task execution
- `[todo]` Gazebo Harmonic integration
- `[todo]` rosbag2 / launch_testing / ros2_tracing / DVC integration

## Goals

The toolchain should support:

- interpretable task execution instead of end-to-end action generation
- modular integration across parsing, grounding, planning, control, and evaluation
- reproducible simulation and benchmarking
- practical deployment on a single development machine with constrained GPU memory

## Recommended Baseline

### Operating Environment

- `[todo]` Primary development runtime: `WSL2 Ubuntu 24.04`
- `[done]` Host environment: `Windows`
- `[partial]` Recommendation: run robotics middleware and backend services in WSL2, keep heavy GUI and local LLM usage coordinated to avoid GPU contention

Rationale:

- Ubuntu 24.04 is the most practical base for ROS 2 Jazzy
- WSL2 provides a workable Linux robotics environment without requiring a separate workstation
- The host machine has limited GPU headroom, so backend-first deployment is safer than full concurrent GUI and model workloads

### Robotics Middleware

- `[todo]` `ROS 2 Jazzy`
- `[todo]` `colcon`
- `[todo]` `rosdep`

Rationale:

- ROS 2 Jazzy is the stable integration backbone for planning, transforms, control, launch, bagging, and testing
- `colcon` and `rosdep` are the standard build and dependency tools for a ROS 2 workspace

### Planning and Geometry

- `[todo]` `MoveIt 2`
- `[todo]` `OMPL`
- `[todo]` `Pinocchio`

Recommended responsibilities:

- `MoveIt 2`: planning scene, collision checking, constrained motion planning integration
- `OMPL`: sampling-based motion planning
- `Pinocchio`: kinematics, dynamics, and explicit SE(3) reasoning

Rationale:

- This combination aligns with a geometry-first, interpretable robotics stack
- It avoids collapsing task reasoning and motion generation into a single opaque policy

### Control

- `[todo]` `ros2_control`
- `[todo]` `ros2_controllers`

Rationale:

- provides a standard hardware abstraction and controller management layer
- supports simulation and future transfer to real hardware

### Task Execution

- `[todo]` `BehaviorTree.CPP`

Rationale:

- behavior trees are a good fit for explicit task decomposition, monitoring, retry, and fallback
- they are easier to inspect and debug than an opaque sequential policy

### Simulation

- `[todo]` Primary simulator: `Gazebo Harmonic`
- `[todo]` Secondary simulator for focused physics experiments: `MuJoCo`

Rationale for choosing Gazebo first:

- better fit for the current project stage
- closer integration with ROS 2 workflows
- lower practical resource pressure than Isaac Sim for this machine profile
- sufficient for system integration, control validation, and benchmark execution

Rationale for not choosing Isaac Sim as the baseline:

- higher platform complexity
- stronger GPU pressure
- more valuable later when the project needs high-fidelity synthetic data, RTX sensors, or Omniverse workflows

### Logging, Testing, and Evaluation

- `[todo]` `rosbag2`
- `[done]` `pytest`
- `[todo]` `launch_testing`
- `[todo]` `ros2_tracing`
- `[todo]` `DVC`

Recommended responsibilities:

- `rosbag2`: record execution inputs, observations, and outputs
- `pytest`: unit tests for schemas, task parsing, and Python tools
- `launch_testing`: ROS integration tests
- `ros2_tracing`: timing and execution analysis
- `DVC`: version benchmark datasets, logs, reports, and model artifacts

Rationale:

- the project should optimize for replayability and reproducibility, not only demo success

### Python Tooling

- `[done]` `uv`
- `[done]` `Ruff`
- `[done]` `mypy`

Rationale:

- lightweight Python dependency management and execution
- fast linting and formatting
- type checking for shared schemas and tooling code

### Containerization

- `[partial]` `Docker Compose`

Rationale:

- useful for consistent auxiliary services and reproducible environment setup
- should be used selectively, not as a replacement for the ROS 2 development workflow inside WSL2

Current repository status:

- the checked-in container setup is intentionally limited to the Python application, benchmark, and test workflow
- ROS 2, MoveIt 2, Gazebo, and hardware integration are not containerized in this phase

## Deployment Guidance

### What should run in WSL2

- `[todo]` ROS 2 core services
- `[todo]` MoveIt 2
- `[todo]` planning services
- `[partial]` world model
- `[partial]` evaluation runners
- `[partial]` benchmark automation
- `[todo]` headless simulation when possible

### What should not be assumed to run concurrently

- a large local LLM with high VRAM residency
- simulator GUI
- RViz
- segmentation or perception models

Rationale:

- the development machine has finite GPU memory
- concurrent LLM, simulation GUI, and vision models can easily exceed safe GPU headroom

## Toolchain by Repository Area

### `apps/`

- `[done]` Python entry points
- `[done]` `uv`
- `[done]` `pytest`

### `modules/task_parser/`

- `[done]` Python-first
- `[done]` `uv`
- `[done]` `pytest`
- `[done]` `mypy`

### `modules/grounding/`

- `[partial]` Python and C++
- `[todo]` `ROS 2`
- `[todo]` `Pinocchio`

### `modules/world_model/`

- `[partial]` C++-leaning
- `[todo]` `ROS 2`
- `[todo]` `tf2`
- `[todo]` `MoveIt 2`

### `modules/planner/`

- `[partial]` Python and C++
- `[todo]` `MoveIt 2`
- `[todo]` `OMPL`
- `[todo]` `BehaviorTree.CPP`

### `modules/skills/`

- `[partial]` C++-leaning
- `[todo]` `MoveIt 2`
- `[todo]` `ros2_control`

### `modules/control/`

- `[partial]` C++
- `[todo]` `ros2_control`

### `eval/`

- `[done]` Python-first
- `[todo]` `rosbag2`
- `[done]` `pytest`
- `[todo]` `launch_testing`
- `[todo]` `DVC`

## Non-Baseline Tools

These are not rejected outright, but they are not recommended as the initial baseline.

### Isaac Sim

Use later if the project needs:

- high-fidelity synthetic data generation
- RTX sensor simulation
- Omniverse-based asset workflows
- larger-scale GPU simulation capacity

Not recommended as the first simulator for this repository because:

- it increases setup and debugging complexity
- it is less compatible with the current machine budget
- it is not necessary to validate the project architecture in the first phase

### Drake

Use selectively for research-heavy planning or dynamics validation.

Not recommended as the central runtime stack because:

- it overlaps with ROS 2, MoveIt 2, and control responsibilities
- it raises integration cost for the current repository phase

## Initial Implementation Order

1. `[partial]` Establish the ROS 2 Jazzy workspace and Python tooling baseline
   Current state: Python tooling baseline is done; ROS 2 Jazzy workspace is not done.
2. `[todo]` Integrate MoveIt 2, OMPL, and ros2_control
3. `[todo]` Stand up Gazebo Harmonic with a small tabletop benchmark environment
4. `[todo]` Add BehaviorTree.CPP-based task execution
5. `[partial]` Add rosbag2, pytest, and launch_testing coverage
   Current state: `pytest` coverage exists; `rosbag2` and `launch_testing` do not.
6. `[todo]` Add DVC-based benchmark and trace versioning
7. `[todo]` Add optional secondary simulation or perception services only after the baseline loop is stable

## Summary

The baseline toolchain is:

`WSL2 Ubuntu 24.04 + ROS 2 Jazzy + MoveIt 2 + OMPL + Pinocchio + ros2_control + BehaviorTree.CPP + Gazebo Harmonic + rosbag2 + pytest + launch_testing + ros2_tracing + DVC + uv + Ruff + mypy`

This stack is chosen to maximize:

- modularity
- geometric interpretability
- reproducibility
- practical fit for the current machine constraints
