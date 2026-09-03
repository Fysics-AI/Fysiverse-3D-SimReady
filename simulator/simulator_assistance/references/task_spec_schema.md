# Task Spec Schema

`task_spec.json` is the task planner's compact description of user intent. It is not the executable physics plan; it is the source of task roles and success criteria.

```json
{
  "schema": "simulator_task_spec.v1",
  "goal": "矿泉水倒下，并砸到了旁边的金属铭牌",
  "task_type": "fall_and_hit",
  "active_object": "mask_003_object",
  "target_object": "mask_002_object",
  "passive_static_objects": ["mask_001_object", "mask_004_object"],
  "goal_interpretation": "The water bottle should fall toward and contact the metal nameplate.",
  "direction_hint": [1.0, 0.0, 0.0],
  "success_conditions": [
    "active object becomes fallen or strongly tilted",
    "active object contacts target object"
  ],
  "assumptions": [
    "semantic labels are inferred from crops"
  ]
}
```

Keep object names exact. Use `final_3d_object_name` values from `scene_digest.json`.
