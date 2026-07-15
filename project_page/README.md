# Project Page Viewer

Static Three.js viewer for generated scene packages.

Serve the open-source directory:

```bash
python -m http.server 8080
```

Open a packaged scene:

```text
http://localhost:8080/project_page/index.html?manifest=../sessions/<session_id>/results/release_package/final_scene_manifest.json&autoload=1
```

The viewer supports:

- compressed per-object GLB loading;
- browser-side rigid-body interaction via Rapier.

Keep manifests and asset paths relative when publishing packages.
