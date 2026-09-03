# Rigid-Body Simulation Checklist

Fill these fields before running SAPIEN:

- Scene source: session path, `final_scene_manifest.json`, selected input blend.
- Coordinate system: Blender/SAPIEN Z-up, units assumed meters unless noted otherwise.
- Ground: fixed `z=0`, normal `+Z`; gravity vector is negative Z.
- Objects: semantic label, object name, mask id, role, body type.
- Collision: convex collision mesh path, whether object-object collisions are enabled.
- Material/contact: static friction, dynamic friction, restitution.
- Inertia proxy: density, damping, max linear/angular velocity if needed.
- Initial condition: pose offset, tilt, linear velocity, angular velocity.
- Control/events: any force/torque/velocity applied at a step.
- Sampling: `steps`, `timestep`, `sample_every`.
- Success condition: expected contact or qualitative outcome.
- Assumptions: anything inferred from crop images or missing upstream metadata.

Recommended defaults for first pass:

```json
{
  "gravity_z": -9.81,
  "ground_z": 0.0,
  "timestep": 0.005,
  "steps": 360,
  "sample_every": 2,
  "static_friction": 0.8,
  "dynamic_friction": 0.6,
  "restitution": 0.05,
  "density": 700.0,
  "linear_damping": 0.02,
  "angular_damping": 0.02
}
```
