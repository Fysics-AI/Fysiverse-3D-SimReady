# Simulator Assistance

`simulator_assistance` converts a completed Fysiverse-3D-SimReady session and a
natural-language physical goal into a rigid-body simulation plan. It can then run
the deterministic SAPIEN/Blender executor, write a Blender animation, and render
the result from the reconstructed original camera.

The default policy reuses the geometry produced by the reconstruction pipeline:
the final Blend scene, optimized camera, object poses, and exported SAPIEN/CoACD
collision assets are read from `results/final_scene_manifest.json`. The agentic
stage adds task-level reasoning, physical parameters, initial conditions, and a
deterministic execution plan.

## Inputs

A runnable session must contain the full local reconstruction results, not only
the lightweight web release package:

```text
sessions/<session_id>/
  input/
    image.png
    mask_label.png
  segmentation/
    overlay.png
    crops/
  results/
    final_scene_manifest.json
```

With `reuse_upstream_sapien_export: always`, the manifest must also point to an
existing `intermediate.sapien_export_manifest`, and each reused object must have
valid collision asset paths.

## Configuration

Edit `config/simulator_assistance.yaml`, or override the same values with
environment variables. Keep real API keys out of git.

```yaml
llm_backend: dashscope_qwen  # dashscope_qwen | openai_compatible | codex

api:
  base_url: ""
  model: ""
  key: ""

deterministic_runner:
  blender_bin: blender
  conda_bin: conda
  conda_env: fysiverse-scene
  reuse_upstream_sapien_export: always
  render_width: 480
  render_height: 480
  render_samples: 8
  render_max_frames: 50
```

For the recommended Qwen API path, create an API key in the DashScope console,
enable the target vision-language model service, and set:

```bash
export SIMULATOR_ASSISTANCE_LLM_BACKEND="dashscope_qwen"
export SIMULATOR_ASSISTANCE_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export SIMULATOR_ASSISTANCE_API_MODEL="<your-qwen-vision-model>"
export SIMULATOR_ASSISTANCE_API_KEY="<your-dashscope-api-key>"
```

This API path does not require an agent CLI. The `codex` backend is optional and
uses a local Codex CLI installation, so only configure it if you explicitly want
that backend. Advanced users can connect another vision-language API with the
generic `openai_compatible` backend by setting its base URL, model name, and key.

## Run

From the repository root:

```bash
python3 simulator/run_simulator_assistance.py \
  --session <session_id> \
  --goal "the trash bin bounces toward the microwave" \
  --mode single
```

`--session` accepts a bare session id, `sessions/<session_id>`, or an absolute
session directory.

Plan only, without running SAPIEN or Blender:

```bash
python3 simulator/run_simulator_assistance.py \
  --session <session_id> \
  --goal "the trash bin bounces toward the microwave" \
  --mode single \
  --no-run
```

If a `simulation_plan.json` already exists, run only the deterministic executor:

```bash
python3 simulator/simulator_assistance/scripts/run_simulation_plan.py \
  --final-manifest sessions/<session_id>/results/final_scene_manifest.json \
  --plan sessions/<session_id>/results/simulator_assistance/<run_id>/simulation_plan.json \
  --work-dir sessions/<session_id>/results/simulator_assistance/<run_id>/run
```

## Planning Modes

```text
auto:
  Uses parallel mode when the scene has at least three objects; otherwise single.

parallel:
  Planner -> task_spec.json
  Semantics -> object_semantics.json
  Physics -> physics_params.json
  Initial conditions -> initial_conditions.json
  Merge -> scene_description.json + simulation_plan.json

single:
  One planner call writes scene_description.json and simulation_plan.json.
```

Use `single` for API backends unless you have verified that the selected model
returns strict JSON for every parallel subtask. `parallel` and `auto` are best
suited to the `codex` backend or to thoroughly tested custom API deployments.

## Outputs

Each invocation writes:

```text
sessions/<session_id>/results/simulator_assistance/<run_id>/
  scene_digest.json
  scene_description.json
  simulation_plan.json
  codex_summary.md
  invocation.json
  agents/
```

When execution is enabled, the runner also writes:

```text
run/
  export/manifest.json
  animation_sequence.json
  final_poses.json
  sapien_sequence_report.json
  run_report.json
  simulation_animation.blend
  simulation_animation_original_view.mp4
```

## Conventions

- Blender/SAPIEN use a Z-up frame.
- Gravity points along negative Z.
- The ground plane is fixed at `z=0`.
- SAPIEN is used for physics; Blender is used for keyframing and original-view rendering.
- Object identity is taken from `mask_id`, `mask_name`, and `final_3d_object_name`.
- By default, camera and object pose optimization from the reconstruction stage are reused.
