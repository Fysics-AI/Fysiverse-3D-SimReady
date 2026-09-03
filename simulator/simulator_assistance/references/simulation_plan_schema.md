# Simulation Plan Schema

`simulation_plan.json` is the contract between Codex planning and deterministic SAPIEN execution.

```json
{
  "schema": "simulator_simulation_plan.v1",
  "goal": "water bottle falls down and hits the metal nameplate",
  "scene_description": "scene_description.json",
  "ground_plane": {"type": "fixed_z0", "point": [0, 0, 0], "normal": [0, 0, 1], "z": 0.0},
  "assumptions": ["water bottle is mask_003_object"],
  "success_conditions": ["bottle moves toward and contacts mask_002_object"],
  "engine": {
    "name": "sapien",
    "steps": 360,
    "timestep": 0.005,
    "sample_every": 2,
    "gravity_z": -9.81,
    "ground_z": 0.0,
    "physx_gpu": false
  },
  "defaults": {
    "unmentioned_body_type": "static",
    "static_friction": 0.8,
    "dynamic_friction": 0.6,
    "restitution": 0.05,
    "density": 700.0,
    "linear_damping": 0.02,
    "angular_damping": 0.02
  },
  "objects": [
    {
      "name": "mask_003_object",
      "mask_id": 3,
      "semantic_label": "water bottle",
      "role": "active",
      "body_type": "dynamic",
      "density": 400.0,
      "linear_damping": 0.03,
      "angular_damping": 0.03,
      "initial_pose_offset": {
        "translation": [0.0, 0.0, 0.02],
        "rotation_euler_deg": [0.0, 20.0, 0.0],
        "rotation_order": "xyz"
      },
      "initial_linear_velocity": [0.6, 0.0, 0.0],
      "initial_angular_velocity": [0.0, 8.0, 0.0]
    },
    {
      "name": "mask_002_object",
      "mask_id": 2,
      "semantic_label": "metal nameplate",
      "role": "target",
      "body_type": "dynamic",
      "density": 2500.0
    }
  ]
}
```

Supported `body_type` values:

- `dynamic`: affected by forces and collisions.
- `kinematic`: moved by the script or fixed pose; can interact when controlled.
- `static`: collision obstacle, not affected by forces.

Supported object fields:

- `name` or `mask_id`: object identity. `name` must match the SAPIEN export manifest.
- `density`, `linear_damping`, `angular_damping`.
- `initial_pose_offset.translation`: world-space meters added to actor pose.
- `initial_pose_offset.rotation_euler_deg`: actor-frame initial rotation in degrees.
- `initial_linear_velocity`: world-space velocity.
- `initial_angular_velocity`: world-space angular velocity.
- `events`: optional per-object events with `step`, `linear_velocity`, `angular_velocity`, `force`, `torque`, and `duration_steps`.

Keep plans small and explicit. Do not hide important initial conditions in comments.
The runner enforces `ground_z = 0.0` even if a generated plan contains another value.
