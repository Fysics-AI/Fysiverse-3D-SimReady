## Environment

Create a new conda environment for this package. The installer does not modify existing conda environments, system CUDA, or NVIDIA drivers, and it does not download model weights.

```bash
cd Fysiverse-3D-SimReady
CONDA_BIN=conda scripts/setup_conda_env.sh
conda activate fysiverse-scene
```

Install PyTorch according to the CUDA version on your machine by following the
official PyTorch instructions:
[Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/).
The setup script provides a default CUDA 12.1 wheel selection; adjust
`TORCH_VERSION`, `TORCHVISION_VERSION`, `TORCHAUDIO_VERSION`, and
`PYTORCH_INDEX_URL` before running it if your driver or CUDA stack requires a
different build. Other Python packages are installed with `uv pip`.

Install these external executables outside conda:

- Blender 4.5 LTS is recommended for writing `.blend` files and rendering.
  Download it from the official Blender website:
  [blender.org/download](https://www.blender.org/download/). After installation,
  make sure `blender --version` works, or set `pipeline.blender_bin` in
  `config/main.yaml` and `deterministic_runner.blender_bin` in
  `config/simulator_assistance.yaml` to the absolute Blender executable path.
- `gltfpack` is recommended for compressed browser GLBs. Install it from the
  official meshoptimizer project:
  [meshoptimizer/gltf](https://github.com/zeux/meshoptimizer/tree/master/gltf),
  then check that `gltfpack -h` works. If the binary is not in `PATH`, set
  `gltfpack_bin` in `config/web_assets.yaml` to the absolute path. If
  `gltfpack` is unavailable, the exporter can use `gltf-transform` when
  `web_compressor`, `preview_compressor`, and `gltf_transform_bin` are
  configured in `config/web_assets.yaml`.

The setup script installs the Python packages used by reconstruction,
postprocessing, SAPIEN simulation, CoACD collision preparation, the Gradio UI,
and API clients. It does not download third party repositories or model
weights.

## Third Party Code and Weights

Place the third party repositories and weights under the default paths below,
or point `config/main.yaml` and `config/background_3dgs.yaml` to your own local
locations. The default layout is:

```text
Fysiverse-3D-SimReady/
  third_party/
    Grounded-SAM-2/
    SAM3D/
    MoGe/
    3dgrut/                 # optional, only for 3DGS background
    GaussianSplats3D/       # optional, only for 3DGS background export
  checkpoints/
    moge-2-vitl-normal/
      model.pt
```

| Component | Used for | Default location | Required files | Config or environment override |
| --- | --- | --- | --- | --- |
| Grounded-SAM-2 | Object and ground segmentation | `third_party/Grounded-SAM-2` | `checkpoints/sam2.1_hiera_large.pt`, `gdino_checkpoints/groundingdino_swint_ogc.pth`, and the bundled `sam2/` and `grounding_dino/` packages | `paths.gsam2_root` or `GSAM2_ROOT` |
| SAM3D | Object level 3D asset reconstruction | `third_party/SAM3D` | SAM3D source tree and `runtime_configs/pipeline_hf_cache_moge_modelpt.yaml` | `paths.sam3d_root`, `paths.sam3d_config`, `SAM3D_ROOT`, or `SAM3D_CONFIG` |
| SAM3D Python env | Optional isolated SAM3D runtime | active conda env | `bin/python` if a separate env is used | `paths.sam3d_env_dir`, `SAM3D_ENV_DIR`, or `SAM3D_PYTHON` |
| MoGe | Monocular geometry, gravity alignment, and pose refinement | `third_party/MoGe` | MoGe source tree | `paths.moge_root` or `MOGE_ROOT` |
| MoGe checkpoint | MoGe inference | `checkpoints/moge-2-vitl-normal/model.pt` | local checkpoint file | `paths.moge_ckpt` or `MOGE_CKPT` |
| 3DGRUT | Optional single image 3DGS background training | `third_party/3dgrut` | `train_diy.py` and a working `.venv` inside the 3DGRUT checkout | `dependencies.threedgrut_root` or `THREEDGRUT_ROOT` |
| GaussianSplats3D | Optional KSplat export for browser viewing | `third_party/GaussianSplats3D` | `util/create-ksplat.js` and Node.js dependencies required by that project | `dependencies.gaussian_splats_root` or `GAUSSIAN_SPLATS_ROOT` |

Grounded-SAM-2 should be installed according to its upstream instructions so
that both SAM2 and GroundingDINO can be imported from the selected
`GSAM2_ROOT`. SAM3D and MoGe should likewise be installed according to their
upstream requirements. If you want `scripts/setup_conda_env.sh` to try editable
installs for repositories already placed in `third_party/`, run:

```bash
INSTALL_EDITABLE_THIRD_PARTY=1 CONDA_BIN=conda scripts/setup_conda_env.sh
```

Keep `third_party/`, `checkpoints/`, local sessions, logs, and `.env.local` out
of git.

## Simulation and Rendering Runtime

SAPIEN and CoACD Python packages are installed by `scripts/setup_conda_env.sh`.
The reconstruction pipeline uses SAPIEN for gravity settling and simulation,
CoACD or convex hulls for collision proxies, and Blender for applying poses,
writing `.blend` files, and rendering videos from the original view.

Before launching, check:

- `blender` is available in `PATH`, or set `pipeline.blender_bin` in `config/main.yaml`.
- `deterministic_runner.blender_bin` in `config/simulator_assistance.yaml` points to the same Blender executable when running agentic simulation.
- `pipeline.conda_env`, `moge.nvdiffrast.env`, `collision.coacd.env`, `sapien.env`, and `simulator_assistance.deterministic_runner.conda_env` refer to environments that exist on your machine. The default is `fysiverse-scene`.
- `gltfpack` or a configured `gltf-transform` fallback is available if you want compressed browser assets.

## API Keys

No API key is required for local segmentation, 3D asset reconstruction,
geometry alignment, collision correction, or SAPIEN settling. API keys are only
needed for language or image editing stages:

| Stage | Required when | Variables |
| --- | --- | --- |
| Scene graph VLM | `features.scene_graph: true` | `SCENE_GRAPH_BACKEND`, `SCENE_GRAPH_API_BASE_URL`, `SCENE_GRAPH_API_MODEL`, `SCENE_GRAPH_API_KEY` |
| 3DGS background image completion with Seedream | Running `Run 3DGS Background` with `background_image.backend: seeddream` | `ARK_API_KEY` or `BACKGROUND_IMAGE_SEEDDREAM_API_KEY` |
| 3DGS background image completion with Qwen Image | Running `Run 3DGS Background` with `background_image.backend: qwen` | `QWEN_IMAGE_API_KEY` or `DASHSCOPE_API_KEY` |
| Agentic physical simulation | Running `simulator/run_simulator_assistance.py` with an API backend | `SIMULATOR_ASSISTANCE_LLM_BACKEND`, `SIMULATOR_ASSISTANCE_API_BASE_URL`, `SIMULATOR_ASSISTANCE_API_MODEL`, `SIMULATOR_ASSISTANCE_API_KEY` |

You can export keys in the shell before launching, or create an untracked
`.env.local` file in the repository root. `launch_gradio_app.sh` loads
`.env.local` automatically when it exists.

Example DashScope setup for scene graph and simulator assistance:

```bash
export SCENE_GRAPH_BACKEND="dashscope_qwen"
export SCENE_GRAPH_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export SCENE_GRAPH_API_MODEL="<your-qwen-vision-model>"
export SCENE_GRAPH_API_KEY="<your-dashscope-api-key>"

export SIMULATOR_ASSISTANCE_LLM_BACKEND="dashscope_qwen"
export SIMULATOR_ASSISTANCE_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export SIMULATOR_ASSISTANCE_API_MODEL="<your-qwen-vision-model>"
export SIMULATOR_ASSISTANCE_API_KEY="<your-dashscope-api-key>"
```

Example Seedream setup for the optional 3DGS background image completion stage:

```bash
export BACKGROUND_IMAGE_BACKEND="seeddream"
export BACKGROUND_IMAGE_SEEDDREAM_API_KEY="<your-seeddream-or-ark-api-key>"
```

## Configuration

Edit `config/main.yaml` first:

- `pipeline.workflow_root`: leave blank to use this checkout directory, or set an absolute path when launching from another location.
- `pipeline.conda_bin`: conda executable, for example `conda` or `/absolute/path/to/conda`.
- `pipeline.conda_env`: the new environment name created above.
- `pipeline.blender_bin`: Blender 4.5.4 executable.
- `server.name`, `server.port`: Gradio bind address and preferred port.
- `paths.*`: local roots and checkpoint paths for GSAM2, SAM3D, and MoGe.
- `features.scene_graph`: keep `true` for the normal interactive flow.
- `module_configs.background_3dgs`: optional 3DGS background reconstruction after the foreground scene has been aligned and settled.

Then review the module configs in `config/`. Each field has a short comment describing what it changes and when to adjust it.

Keep real API keys out of git. The setup script should not contain API keys.

For a quick configuration sanity check:

```bash
python3 scripts/load_pipeline_config.py --config config/main.yaml
```

This command prints the environment variables that will be exported by the
launcher. It should not print any real API key unless you intentionally put one
in your local shell or untracked `.env.local`.
