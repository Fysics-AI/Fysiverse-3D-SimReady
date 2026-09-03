# Web Alignment Guide for Background 3DGS

This note describes how the web viewer should align the background 3DGS with
the foreground objects for session `20260703_231333_72ed5ffc`.

## Goal

Only the initial loaded scene needs to align:

- foreground GLB objects
- background 3DGS splat

It is acceptable if later Rapier simulation moves foreground objects away from
the static 3DGS background.

## Coordinate-System Contract

The background 3DGS is already exported in the OOD workflow final foreground
world:

```text
foreground_world_blender_z_up
```

The background generator uses the differentiable-rendering optimized
`OriginalInputCamera`:

```text
sessions/20260703_231333_72ed5ffc/results/sam3d_moge_separated_original_input_camera.json
```

The transform used during background export is:

```text
MoGe OpenCV camera point [x, y, z]
  -> Blender camera local [x, -y, -z]
  -> optimized camera_to_world
  -> foreground_world_blender_z_up
```

Therefore the 3DGS splat is already in the same world frame as foreground
objects loaded from:

```text
sessions/20260703_231333_72ed5ffc/results/final_scene_manifest.json
```

using each object's:

```text
sapien_final_pose.final_pose
```

## Do Not Add the Old Grounding Translation

Do not add the early grounded-stage global translation to the splat:

```text
[0, 0, 1.0923254489898682]
```

That value exists in:

```text
sessions/20260703_231333_72ed5ffc/results/sam3d_moge_grounded_report.json
```

but the current background 3DGS uses the later separated/final world. Adding
that translation again would double-count a stage transform and misalign the
background.

The current splat transform should remain identity:

```json
{
  "position": [0, 0, 0],
  "quaternion_xyzw": [0, 0, 0, 1],
  "scale": [1, 1, 1]
}
```

## Required Web-Side Rule

If the web viewer applies any initial scene-level transform to foreground
objects, the same transform must also be applied to the background splat.

Examples of scene-level transforms:

- global translation
- global rotation
- global scale
- centering or normalization transform
- conversion to another viewer coordinate frame

The rule is:

```text
final_splat_transform = web_scene_transform * scene_json.splat_transform
final_object_transform = web_scene_transform * object_final_pose
```

If no web-side scene transform is applied, use identity for both.

## Recommended Schema Handling

`results/3dgs_bg/scene.json` may contain:

```json
{
  "coordinate_system": "foreground_world_blender_z_up",
  "world_transform": {
    "position": [0, 0, 0],
    "quaternion_xyzw": [0, 0, 0, 1],
    "scale": [1, 1, 1]
  },
  "splat": {
    "position": [0, 0, 0],
    "quaternion_xyzw": [0, 0, 0, 1],
    "scale": [1, 1, 1]
  }
}
```

The web viewer should treat missing `world_transform` as identity.

When the viewer has its own initial foreground scene transform, it should either:

1. overwrite `world_transform` with that transform before loading both layers, or
2. multiply that transform into both foreground objects and the splat at load time.

## GLB Visual Coordinate Conversion

Foreground GLB visual content may need this conversion:

```js
const GLTF_VISUAL_TO_BLENDER_Z_UP = new THREE.Matrix4().makeRotationX(Math.PI / 2);
```

This transform is GLB-visual-only. It should be applied to the GLB visual child
before applying the object pose.

It must not be applied to the 3DGS splat.

Recommended structure:

```text
objectRoot
  - carries object final pose in foreground_world_blender_z_up
  - child visualRoot
      - carries GLTF_VISUAL_TO_BLENDER_Z_UP
```

## Minimal Frontend Implementation

If the frontend currently does this:

```js
foregroundRoot.position.copy(sceneOffset);
foregroundRoot.quaternion.copy(sceneRotation);
foregroundRoot.scale.setScalar(sceneScale);
```

then it must also apply the same transform to the splat:

```js
const splatPosition = manifest.splat.position ?? [0, 0, 0];
const splatRotation = manifest.splat.quaternion_xyzw ?? [0, 0, 0, 1];
const splatScale = manifest.splat.scale ?? [1, 1, 1];

// Conceptually:
// finalSplat = sceneTransform * splatTransform
```

For translation-only scene offsets:

```js
const finalSplatPosition = [
  sceneOffset[0] + splatPosition[0],
  sceneOffset[1] + splatPosition[1],
  sceneOffset[2] + splatPosition[2],
];
```

For rotation or nonuniform scale, use `THREE.Matrix4` composition/decomposition
rather than adding vectors manually.

## Practical Recommendation

For this session, the safest initial-alignment behavior is:

1. Load foreground objects using `sapien_final_pose.final_pose`.
2. Load background splat using identity transform from `scene.json`.
3. Do not add `[0, 0, 1.0923254489898682]`.
4. If the frontend applies any extra initial transform to foreground objects,
   apply the exact same extra transform to the splat.
5. Ignore later physics drift for this alignment check.

