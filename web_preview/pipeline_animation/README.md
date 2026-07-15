# Pipeline Animation Web Preview

This static viewer plays a packaged `web/animation_manifest.json` and the
compressed per-object GLBs referenced by that manifest.

Run from the open-source project root:

```bash
python3 -m http.server 8075 --bind 0.0.0.0
```

Open:

```text
http://<host>:8075/web_preview/pipeline_animation/index.html?manifest=../sessions/<session_id>/results/release_package/web/animation_manifest.json
```

The viewer loads a manifest from the `manifest` query parameter. Use the
packaged release manifest path when sharing a scene:

```text
web/animation_manifest.json
```

You can also use `?session=<session_id>` for the standard packaged path under
`sessions/<session_id>/results/release_package/web/animation_manifest.json`.

For local debugging before packaging, point directly to the intermediate export:

```text
../sessions/<session_id>/results/web_pipeline_animation/animation_manifest.json
```
