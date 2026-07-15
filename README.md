# Fysiverse-3D-SimReady

This open-source subset provides the interactive image-to-3D scene generation workflow. Use the Gradio page to upload an image, annotate objects, run segmentation, wait for the scene graph, and launch 3D scene generation.

To allow the community to experience our technology as soon as possible, the version we are currently releasing is built on top of open-source models. We will subsequently integrate our self-developed models into this system.

## Environment

Create a new conda environment for this package. The installer does not modify existing conda environments, system CUDA, or NVIDIA drivers, and it does not download model weights.

```bash
cd open_source_wobg
ENV_NAME=fysiverse-3d CONDA_BIN=conda scripts/setup_conda_env.sh
conda activate fysiverse-3d
```

PyTorch is installed from the official CUDA wheel index. Other Python packages are installed with `uv pip` from the Tsinghua mirror by default.

Install these tools outside conda:

- Blender 4.5.4. Set `pipeline.blender_bin` in `config/main.yaml` to `blender` or the absolute Blender binary path.
- `gltfpack` for compressed browser GLBs. If unavailable, the exporter can fall back to `gltf-transform` when configured.

Provide local third-party code and weights before launching:

- Grounded-SAM-2 source tree and checkpoints.
- SAM3D source tree and runtime config.
- MoGe source tree and checkpoint.
- Scene graph API credentials, either in a local config copy or environment variables.

## Configuration

Edit `config/main.yaml` first:

- `pipeline.workflow_root`: absolute path to this `open_source_wobg` directory.
- `pipeline.conda_bin`: conda executable, for example `conda` or `/absolute/path/to/conda`.
- `pipeline.conda_env`: the new environment name created above.
- `pipeline.blender_bin`: Blender 4.5.4 executable.
- `server.name`, `server.port`: Gradio bind address and preferred port.
- `paths.*`: local roots and checkpoint paths for GSAM2, SAM3D, and MoGe.
- `features.scene_graph`: keep `true` for the normal interactive flow.

Then review the module configs in `config/`. Each field has a short comment describing what it changes and when to adjust it.

Keep real API keys out of git. The setup script should not contain API keys.

## Launch

Start the interactive page:

```bash
cd open_source_wobg
./launch_gradio_app.sh
```

For a custom config file:

```bash
PIPELINE_CONFIG=/absolute/path/to/main.yaml ./launch_gradio_app.sh
```

Open the local URL printed by Gradio.

## UI Workflow

We provide a video for usage at `./assets/usage.mp4`

1. Upload an RGB image in `Input image`.
2. Annotate objects.
   - In `box` mode, click two opposite corners for each object box. **(recommended)**
   - In `point` mode, select foreground/background and click object prompt points.
   - Use `Clear boxes` or `Clear points` to reset prompts.
3. Optionally add a ground prompt in `Ground prompt`. These points help estimate the ground plane.
5. Click `Run Segmentation`.  (around 1 min)
6. Wait for `Scene graph status` to finish. The `Run 3D Inference` button is enabled when the scene graph is ready. (around 3 min)
7. Click `Run 3D Inference`.  (around 5- 10 min)
8. Download results from `Output files` and the optional `Pipeline diagnostic animation (.blend)`.

The UI writes each run under `sessions/<session_id>/`. The clean publishable package is in:

```text
sessions/<session_id>/results/release_package/
  final_scene_manifest.json
  blends/
    final_state.blend
    pipeline_animation.blend
  glb/
    final_state.glb
    raw_objects/*.glb
    web_objects/*.glb
  web/
    animation_manifest.json
```

`final_state.glb` is uncompressed and preserves independent object nodes. `raw_objects/*.glb` keeps the original per-object assets, and `web_objects/*.glb` is prepared for browser viewing.

## Web View

Serve the `open_source_wobg` directory with a static file server:

```bash
cd open_source_wobg

python -m http.server 8080
```

Open the final scene viewer:

```text
http://localhost:8080/project_page/index.html?manifest=../sessions/<session_id>/results/release_package/final_scene_manifest.json&autoload=1
```

Open the lightweight pipeline animation viewer:

```text
http://localhost:8080/web_preview/pipeline_animation/index.html?manifest=../sessions/<session_id>/results/release_package/web/animation_manifest.json
```



Replace `<session_id>` with the session directory shown in the Gradio page. The final scene viewer loads packaged GLB assets and browser-side rigid-body controls.

## TODO

- [ ] Feature:
  - [ ] 3DGS background support
  - [ ] Automatically simulation