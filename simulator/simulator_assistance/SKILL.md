---
name: simulator_assistance
description: Convert a completed Fysiverse-3D-SimReady reconstruction session and a natural-language physical goal into a SAPIEN rigid-body simulation plan, then optionally run the deterministic SAPIEN/Blender executor. Reuses the session's final Blend scene, camera, object identities, and collision assets when available.
---

# Simulator Assistance

Use this skill when a reconstructed Fysiverse-3D-SimReady scene needs to become
an executable physics rollout.

## Assumptions

- Workflow root is the repository checkout containing this file.
- Runtime paths are supplied by CLI arguments, `config/simulator_assistance.yaml`,
  or environment variables such as `BLENDER_BIN`, `CONDA_BIN`, and
  `FYSIVERSE_CONDA_ENV`.
- The ground plane is fixed at `z=0` with normal `+Z`.
- The coordinate frame is Blender/SAPIEN Z-up, with gravity along negative Z.
- SAPIEN rendering is disabled; SAPIEN writes pose sequences and Blender renders
  the final animation from the reconstructed original camera.
- Object identity must preserve `mask_id`, `mask_name`, and
  `final_3d_object_name` from `final_scene_manifest.json`.
- Do not create new physical bodies for unsegmented background objects.

## Normal Entry Point

Prefer the wrapper from the repository root:

```bash
python3 simulator/run_simulator_assistance.py \
  --session sessions/<session_id> \
  --goal "<natural-language physical goal>" \
  --mode auto
```

Plan only:

```bash
python3 simulator/run_simulator_assistance.py \
  --session sessions/<session_id> \
  --goal "<natural-language physical goal>" \
  --mode auto \
  --no-run
```

## Planning Contract

The planner must produce JSON that respects:

- `references/scene_description_schema.md`
- `references/task_spec_schema.md`
- `references/simulation_plan_schema.md`
- `references/physics_checklist.md`

The agentic decomposition is:

- Planner: choose the task roles, causal phases, success conditions, and
  high-level interaction plan.
- Reasoner: infer semantic labels, rigid body types, material priors, density,
  damping, friction, and task-relevant initial conditions.
- Builder: validate object references, body types, collision assets, parameter
  ranges, and required initial states before preparing the simulator command.

## Required Outputs

For plan-only mode, write:

```text
scene_description.json
simulation_plan.json
codex_summary.md
```

For executed runs, the deterministic runner creates:

```text
run/export/manifest.json
run/animation_sequence.json
run/final_poses.json
run/sapien_sequence_report.json
run/run_report.json
run/simulation_animation.blend
run/simulation_animation_original_view.mp4
```

## Object And Physics Rules

- Use exact object names from the scene digest in `simulation_plan.objects`.
- Keep support/background objects static unless the goal requires motion.
- Prefer conservative mass, friction, and damping values when the material is
  uncertain.
- Keep the active object and target object dynamic when they must move or react.
- Use fixed/static bodies for large support surfaces and irrelevant background
  obstacles.
- Store assumptions explicitly in both the simulation plan and summary.

## Deterministic Execution

The deterministic runner should:

1. Load `results/final_scene_manifest.json`.
2. Reuse upstream SAPIEN/CoACD assets when `reuse_upstream_sapien_export` is
   `always` or `auto` and the assets exist.
3. Replace each reused actor's initial pose with the upstream gravity-settled
   final pose when available.
4. Run the SAPIEN physics sequence with `ground_z=0`.
5. Write sampled actor poses as row-major 4x4 matrices.
6. Apply the pose sequence to the final Blend scene as Blender keyframes.
7. Render the animation from the reconstructed original camera unless rendering
   is explicitly skipped.
