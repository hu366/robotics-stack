# RGBD Mapping And Semantic Grounding Architecture

This document defines the stage-1 architecture for simulation-first RGBD reconstruction
and semantic-to-geometric grounding.

The goal is to give the implementation window a stable contract surface without forcing
heavy algorithms, hidden coupling, or planner-visible raw geometry.

If the workspace already contains experimental files such as `interfaces/mapping.py`
or a first draft of a MuJoCo mapping CLI, treat them as transitional scaffolding.
The intended contract boundary for future work is the split interface layout defined
in this document: `common`, `perception`, `scene_graph`, `grounding`, and `world_state`.

## Scope

Stage 1 assumptions:

- environment is MuJoCo-first
- wrist camera is `wrist_rgbd`
- multi-frame RGBD fusion uses simulator ground-truth camera poses
- no SLAM in stage 1
- no neural fields, Gaussian splats, or SE(3) models in the primary state
- world state stores summaries and artifact references, not dense raw clouds

Out of scope for stage 1:

- online loop-closure
- global bundle adjustment
- learned implicit map as the authoritative state
- planner access to raw point clouds or neural field internals

## Design Goals

- Keep the execution pipeline modular and inspectable.
- Separate perception ingestion, world-state storage, task-conditioned grounding, and planning.
- Preserve explicit provenance for every field that may come from simulation truth, fused geometry, rules, or future learned models.
- Make every material stage produce auditable trace payloads and file artifacts.

## Two Complementary Flows

The repository-level task pipeline remains:

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

Stage-1 RGBD reconstruction introduces an additional perception ingestion flow that
feeds the world model outside the task parser path:

`MuJoCo RGBD capture -> frame manifests -> point-cloud fusion -> geometry summary -> scene graph -> world model snapshot`

Interpretation:

- `modules/world_model` owns persistent scene memory and references to mapping artifacts.
- `modules/grounding` owns task-conditioned queries over that scene memory.
- `modules/planner` still consumes explicit candidates and constraints, not raw perception tensors.

## Module Ownership

### `interfaces/`

These files own all cross-module shapes.

- `interfaces/common.py`
  - `ArtifactReference`
  - `SourceAttribution`
- `interfaces/perception.py`
  - `CameraIntrinsics`
  - `CameraPose`
  - `RGBDFrame`
  - `PointCloudFrame`
  - `GeometryMapSummary`
  - `MapArtifactSummary`
  - `MappingTracePayload`
- `interfaces/scene_graph.py`
  - `SceneNode`
  - `SceneEdge`
  - `SceneGraph`
  - `ConstraintState`
- `interfaces/grounding.py`
  - `SceneGraphQuery`
  - `GroundingRequest`
  - `ObjectCandidate`
  - `SurfaceCandidate`
  - `PoseCandidate`
  - `GroundingResult`
- `interfaces/world_state.py`
  - extend `WorldState` to carry summaries and artifact references only

### `modules/world_model/`

Responsibilities:

- ingest mapping outputs into a `WorldState`
- track current scene revision
- store `GeometryMapSummary`, `SceneGraph`, `ConstraintState`, and `MapArtifactSummary`
- expose snapshot reads for grounding and planning

Must not do:

- task-conditioned semantic matching
- dense mapping algorithms
- planner-specific heuristics
- storing large point arrays inline inside `WorldState`

Suggested stage-1 API shape:

```python
class WorldModelStore:
    def update_mapping_snapshot(
        self,
        scene_id: str,
        geometry_map: GeometryMapSummary,
        scene_graph: SceneGraph,
        constraints: list[ConstraintState],
        artifacts: MapArtifactSummary,
    ) -> WorldState: ...

    def current(self) -> WorldState: ...
```

### `modules/grounding/`

Responsibilities:

- translate `TaskSpec` into structured `GroundingRequest`
- query `WorldState.scene_graph`
- combine scene-graph matches with map summaries and constraints
- emit planner-facing `GroundingResult`

Must not do:

- RGBD fusion
- direct file IO for dense map artifacts in planner-facing paths
- opaque end-to-end grounding that skips structured candidates

Suggested stage-1 API shape:

```python
class SceneGrounder:
    def build_request(self, task: TaskSpec, world_state: WorldState) -> GroundingRequest: ...

    def query_scene_graph(
        self,
        query: SceneGraphQuery,
        scene_graph: SceneGraph,
    ) -> list[str]: ...

    def ground(self, task: TaskSpec, world_state: WorldState) -> GroundingResult: ...
```

`query_scene_graph()` should return matching node ids or typed candidate objects, not raw cloud slices.

### `apps/`

Stage-1 app ownership:

- keep `apps/capture_mujoco_rgbd.py` as the single-frame RGBD capture tool
- optionally add a map-building CLI such as `apps/build_mujoco_map.py`
- optionally add a scene export CLI such as `apps/export_scene_graph.py`
- extend `apps/run_task.py` later to load an existing mapped scene snapshot before grounding

Suggested map-building CLI phases:

1. scan or load RGBD frames
2. read simulator camera poses
3. back-project each frame to a lightweight point-cloud artifact
4. fuse into a stage-1 geometry map summary
5. derive scene graph and constraints
6. write artifacts plus a mapping trace

## Interface Contracts

### `CameraIntrinsics`

Purpose:

- stable camera model contract for MuJoCo capture and future real sensors

Required fields:

- `camera_name`
- `width_px`, `height_px`
- `fx`, `fy`, `cx`, `cy`
- `near_m`, `far_m`
- `depth_unit`

### `CameraPose`

Purpose:

- explicit pose source for each RGBD frame

Required fields:

- `frame_id`
- `parent_frame_id`
- `position_xyz`
- `orientation_xyzw`
- `timestamp_s`
- `pose_source`

Stage 1 rule:

- `pose_source` is normally `sim_ground_truth`

### `RGBDFrame`

Purpose:

- frame manifest entry for one RGB image and one depth image

Must contain:

- frame identity and sequence index
- capture timestamp
- intrinsics and pose
- artifact references to `rgb.png` and `depth.npy`
- optional valid-depth statistics
- provenance

### `PointCloudFrame`

Purpose:

- explicit back-projected geometry product for one RGBD frame

Must contain:

- source RGBD frame id
- point count
- metric bounds
- artifact reference to an `.npz`, `.ply`, or similarly simple file
- provenance

### `GeometryMapSummary`

Purpose:

- planner-safe and world-model-safe summary of fused geometry

Must contain:

- `map_id`
- `scene_id`
- fused frame count and point count
- metric bounds
- voxel size if voxelization is used
- optional occupied voxel count
- support-surface ids that can be referred to from the scene graph
- provenance

Must not contain:

- dense point arrays
- TSDF tensors
- neural field parameters

### `SceneGraph`

Purpose:

- explicit bridge between semantic labels and geometric anchors

Node types in stage 1 should at least support:

- `object`
- `surface`
- `region`
- `agent`
- `frame`

Recommended node fields:

- `label`
- `node_type`
- `center_pose`
- `bbox_extent_m`
- `geometry_anchor_id`
- `semantic_tags`
- `state_tags`
- `properties`
- `provenance`

Recommended edge relation families:

- `spatial`
- `support`
- `contact`
- `reachability`
- `semantic`

### `ConstraintState`

Purpose:

- explicit task-relevant constraints separate from raw geometry

Stage-1 examples:

- support relation between a tray surface and a bottle
- contact state between gripper and object
- reachability estimate from current robot base or wrist frame
- clearance threshold near a target pose
- simplified force or stability flags from rule-based checks

### `MappingTracePayload`

Purpose:

- standard payload for trace events emitted by the mapping pipeline

Must contain:

- `mapping_run_id`
- `scene_id`
- `camera_name`
- `pose_source`
- ordered `frame_ids`
- `fused_frame_count`
- dropped frame ids
- timing breakdown
- `GeometryMapSummary`
- `MapArtifactSummary`
- stage notes

## Artifact Protocol

Recommended directory layout:

```text
artifacts/maps/<mapping_run_id>/
  manifest.json
  frames/
    frame_000000/
      frame.json
      rgb.png
      depth.npy
      pointcloud.npz
  fused/
    geometry_map.json
    fused_pointcloud.npz
    scene_graph.json
    constraints.json
  traces/
    mapping_trace.json
```

### Files To Save

Required in stage 1:

- `manifest.json`
  - top-level run metadata
- `frames/*/frame.json`
  - one `RGBDFrame` manifest per captured frame
- `frames/*/rgb.png`
  - raw RGB observation
- `frames/*/depth.npy`
  - raw depth in meters
- `frames/*/pointcloud.npz`
  - back-projected per-frame point cloud
- `fused/geometry_map.json`
  - serialized `GeometryMapSummary`
- `fused/fused_pointcloud.npz`
  - optional but recommended stage-1 fused cloud artifact
- `fused/scene_graph.json`
  - serialized `SceneGraph`
- `fused/constraints.json`
  - serialized list of `ConstraintState`
- `traces/mapping_trace.json`
  - trace with `MappingTracePayload`

Optional in stage 1:

- `frames_manifest.json`
  - denormalized list of all frame manifests
- `coverage_debug.json`
  - debugging summary for frame overlap or missing coverage

### Approximate JSON Shapes

`manifest.json`

```json
{
  "mapping_run_id": "map-demo-0001",
  "scene_id": "tabletop_pick_place_v1",
  "camera_name": "wrist_rgbd",
  "pose_source": "sim_ground_truth",
  "frame_count": 24,
  "root_dir": "artifacts/maps/map-demo-0001",
  "artifacts": {
    "geometry_map": "fused/geometry_map.json",
    "scene_graph": "fused/scene_graph.json",
    "constraints": "fused/constraints.json",
    "trace": "traces/mapping_trace.json"
  }
}
```

`frames/<id>/frame.json`

```json
{
  "frame_id": "frame_000012",
  "sequence_index": 12,
  "timestamp_s": 1.2,
  "camera_name": "wrist_rgbd",
  "intrinsics": {
    "width_px": 640,
    "height_px": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5
  },
  "pose": {
    "frame_id": "wrist_rgbd",
    "parent_frame_id": "world",
    "position_xyz": [0.55, 0.02, 0.68],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "pose_source": "sim_ground_truth"
  },
  "rgb_artifact": {"path": "frames/frame_000012/rgb.png"},
  "depth_artifact": {"path": "frames/frame_000012/depth.npy"}
}
```

`fused/geometry_map.json`

```json
{
  "map_id": "map-demo-0001",
  "scene_id": "tabletop_pick_place_v1",
  "frame_count": 24,
  "fused_point_count": 182340,
  "voxel_size_m": 0.01,
  "bounds_min_m": [-0.3, -0.4, 0.0],
  "bounds_max_m": [0.9, 0.5, 0.8],
  "support_surface_ids": ["table_top", "tray_inner_surface"],
  "provenance": [
    {"source_type": "rgbd_fused_geometry", "source_id": "fusion_v0"}
  ]
}
```

`fused/scene_graph.json`

```json
{
  "scene_id": "tabletop_pick_place_v1",
  "root_frame_id": "world",
  "revision": 3,
  "nodes": [
    {
      "node_id": "bottle_1",
      "label": "bottle",
      "node_type": "object",
      "center_pose": [0.42, 0.06, 0.11, 0.0, 0.0, 0.0, 1.0],
      "geometry_anchor_id": "cluster_17",
      "provenance": [
        {"source_type": "sim_ground_truth", "source_id": "model/body/bottle"}
      ]
    }
  ],
  "edges": [
    {
      "edge_id": "bottle_on_table",
      "source_node_id": "bottle_1",
      "target_node_id": "table_top",
      "relation_type": "on",
      "relation_family": "support"
    }
  ]
}
```

### Trace Fields

The mapping trace should include events such as:

- `scan_started`
- `frame_captured`
- `frame_backprojected`
- `frame_rejected`
- `fusion_completed`
- `scene_graph_built`
- `constraints_inferred`
- `world_state_snapshot_written`

Each event payload should include only compact structured data:

- identifiers: `mapping_run_id`, `scene_id`, `frame_id`, `node_id`
- stage stats: counts, timings, bounds, voxel size
- provenance: pose source, fusion source, rule source
- artifact references written at that stage
- validation flags such as NaN ratio, empty depth, dropped frames, missing labels

Do not embed large arrays in traces.

## Semantic To Geometric Grounding Contract

### Input

The grounding module should consume:

- `TaskSpec`
- current `WorldState`
- optionally a prior execution context or planner feedback later

For stage 1, the `WorldState` should already expose:

- `geometry_map`
- `scene_graph`
- `constraint_states`
- `map_artifacts`

### Querying The Scene Graph

Grounding should not scan raw point-cloud files during planning.

Instead it should build one or more `SceneGraphQuery` objects such as:

```python
SceneGraphQuery(
    query_id="target_object",
    text="bottle",
    target_label="bottle",
    node_types=["object"],
    relation_filters=["visible", "reachable"],
    required_tags=["movable"],
)
```

Recommended query matching stages:

1. label and alias match
2. node-type filtering
3. state-tag filtering
4. relation filtering over `SceneEdge`
5. provenance preference, for example prefer simulator truth over weak learned output when both exist

### Planner-Facing Output

The grounding output must be a `GroundingResult` that carries:

- `object_candidates`
- `surface_candidates`
- `pose_candidates`
- `constraint_states`
- unresolved semantic slots
- short trace notes for why candidates were kept or rejected

Planner-visible candidate semantics:

- `ObjectCandidate`
  - object id, scene-graph node id, pose, bounding box, support parent, evidence
- `SurfaceCandidate`
  - surface id, pose, normal, parent object, evidence
- `PoseCandidate`
  - candidate pose to grasp, place, inspect, or approach

The planner should consume these candidates directly rather than touching dense map files.

## Provenance Rules

Every field that can change source over time should be attributable.

### From simulator ground truth

Typical fields:

- `CameraPose.position_xyz`
- `CameraPose.orientation_xyzw`
- node ids aligned to MuJoCo bodies, geoms, or sites
- exact object pose when bootstrapping scene graph in stage 1

### From RGBD fused geometry

Typical fields:

- `GeometryMapSummary.frame_count`
- `GeometryMapSummary.fused_point_count`
- support-surface extents derived from fused observations
- object or surface bounds estimated from frame fusion
- coverage or visibility statistics

### From inferred rules

Typical fields:

- `ConstraintState.status`
- support or contact edges inferred from pose plus height thresholds
- reachability or clearance flags from deterministic checks
- simplified force or stability annotations

### From future learned models

Typical fields:

- semantic labels proposed by open-vocabulary detectors
- shape priors or category priors
- pose refinement from SE(3)-equivariant models
- occupancy completion from implicit or neural fields

Rule:

- when future learned models add information, they enrich `SceneNode.provenance`,
  `SceneEdge.provenance`, or candidate `source_types`; they do not replace the
  explicit `WorldState` contract.

## Evolution Path

### Stage 1 to Stage 2

Stage 2 may add:

- visual odometry or SLAM instead of simulator-truth poses
- implicit occupancy or neural field artifacts
- learned scene-graph proposal models
- SE(3) or equivariant pose refinement

Compatibility rule:

- keep `WorldState`, `SceneGraph`, `ConstraintState`, and `GroundingResult`
  as the planner-facing contract
- add new artifact references and provenance entries instead of changing the
  planner contract to consume opaque tensors

Examples:

- SLAM only changes how `CameraPose.pose_source` and frame poses are produced
- neural fields only add artifact references under `MapArtifactSummary`
- SE(3) refiners only add candidate provenance or refined poses with confidence

## Testing Guidance

Implementation-side tests should cover:

- serialization of every interface dataclass
- world-state snapshots that carry summaries but not dense arrays
- scene-graph query behavior on small deterministic scenes
- trace payload validation for dropped frames and provenance
- planner integration tests that confirm only candidates and constraints are consumed

## Recommended First Implementation Milestone

1. Capture a scripted MuJoCo wrist scan with truth poses.
2. Serialize `RGBDFrame` manifests and per-frame point-cloud artifacts.
3. Produce a lightweight fused cloud and `GeometryMapSummary`.
4. Build a minimal `SceneGraph` with objects and support surfaces.
5. Build rule-based `ConstraintState` entries.
6. Update `WorldModelStore` with a snapshot referencing those artifacts.
7. Ground a parsed instruction into planner-facing candidates.
