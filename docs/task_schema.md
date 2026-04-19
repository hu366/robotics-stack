# Task Schema

V1 defines `TaskSpec` as the stable interface between natural-language parsing and downstream grounding/planning.

## Core Structure

`TaskSpec` contains:

- `task_id`: stable task identifier for traces and plans
- `instruction`: normalized natural-language instruction
- `goal`: planner-facing high-level goal
- `action`: parser-facing action family
- `arguments`: structured task slots
- `spatial_relation`: requested relation between object and location when present
- `preconditions`: symbolic checks expected before execution
- `postconditions`: symbolic outcomes expected after execution
- `constraints`: task-level execution constraints
- `recovery_policy`: failure handling policy
- `substeps`: parser-produced symbolic substep outline

`TaskArgument` contains:

- `role`
- `text`
- `entity_type`
- `is_required`

`TaskStepSpec` contains:

- `step_id`
- `action`
- `description`
- `required_arguments`
- `success_criteria`

## V1 Value Set

`goal` is limited to:

- `place_object`
- `open_object`
- `close_object`
- `insert_object`
- `inspect_scene`

`action` is limited to:

- `place`
- `move`
- `open`
- `close`
- `insert`
- `inspect`

`arguments` currently support only:

- `target_object`
- `target_location`

`spatial_relation` is limited to:

- `on`
- `in`
- `to`
- `inside`
- `null`

Default constraints:

- `maintain_collision_safety`
- `keep_traceability`
- `respect_target_relation` when `target_location` exists

## Example

Input instruction:

```text
把瓶子放到托盘上
```

Representative serialized task:

```json
{
  "task_id": "task-1234abcd",
  "instruction": "把瓶子放到托盘上",
  "goal": "place_object",
  "action": "place",
  "arguments": [
    {
      "role": "target_object",
      "text": "瓶子",
      "entity_type": "object",
      "is_required": true
    },
    {
      "role": "target_location",
      "text": "托盘",
      "entity_type": "object",
      "is_required": true
    }
  ],
  "spatial_relation": "on",
  "preconditions": [
    "target_object_identified",
    "target_location_identified"
  ],
  "postconditions": [
    "object_transferred",
    "target_relation_satisfied"
  ],
  "constraints": [
    "maintain_collision_safety",
    "keep_traceability",
    "respect_target_relation"
  ],
  "recovery_policy": "replan",
  "substeps": [
    {
      "step_id": "step_1",
      "action": "locate",
      "description": "Resolve task entities in the scene.",
      "required_arguments": ["target_object"],
      "success_criteria": ["object_pose_resolved"]
    },
    {
      "step_id": "step_2",
      "action": "grasp",
      "description": "Establish a stable grasp on the target object.",
      "required_arguments": ["target_object"],
      "success_criteria": ["stable_grasp"]
    },
    {
      "step_id": "step_3",
      "action": "place",
      "description": "Place the target object at the requested location.",
      "required_arguments": ["target_object", "target_location"],
      "success_criteria": ["target_relation_satisfied"]
    }
  ]
}
```

## V1 Limits

V1 intentionally does not support:

- multi-step natural-language commands such as `先...再...`
- pronoun resolution
- disambiguation of multiple matching entities
- structured attribute decomposition such as color or relative-position modifiers
- multiple objects or multiple target locations in one instruction
