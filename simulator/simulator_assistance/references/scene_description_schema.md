# Scene Description Schema

`scene_description.json` is a compact, editable summary of a Fysiverse-3D-SimReady workflow session.

```json
{
  "schema": "simulator_scene_description.v1",
  "source_final_scene_manifest": "path/to/final_scene_manifest.json",
  "session_id": "20260603_190957_431a2860",
  "coordinate_system": "Blender/SAPIEN Z-up",
  "ground_plane": {
    "type": "fixed_z0",
    "point": [0.0, 0.0, 0.0],
    "normal": [0.0, 0.0, 1.0],
    "z": 0.0
  },
  "inputs": {
    "image": "path/to/image.png",
    "overlay": "path/to/segmentation/overlay.png",
    "final_blend": "path/to/final.blend"
  },
  "objects": [
    {
      "mask_id": 3,
      "mask_name": "mask_003",
      "object_name": "mask_003_object",
      "semantic_label": "water bottle",
      "visual_description": "green plastic bottled water",
      "crop_rgb": "path/to/crop.png",
      "sam3d_input_mask": "path/to/2.png",
      "bbox_2d_xyxy": [0, 0, 10, 10],
      "bbox_3d_min": [0.0, 0.0, 0.0],
      "bbox_3d_max": [1.0, 1.0, 1.0],
      "bbox_3d_center": [0.5, 0.5, 0.5],
      "bbox_3d_extent": [1.0, 1.0, 1.0],
      "pose_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "role_candidates": ["active", "target", "passive"],
      "physical_hints": {
        "likely_material": "plastic",
        "rigid": true,
        "hollow_or_light": true
      }
    }
  ],
  "relations": [
    {
      "a": "mask_003_object",
      "b": "mask_002_object",
      "center_delta": [0.39, 0.01, -0.02],
      "center_distance": 0.39,
      "xy_distance": 0.39
    }
  ],
  "missing_or_uncertain": [
    "semantic labels were inferred from crops",
    "real metric scale is approximate"
  ]
}
```

Keep semantic labels short and concrete. Preserve `mask_id` and `object_name` exactly.
The ground plane is not uncertain: simulator_assistance fixes it to `z=0` in a Z-up world.
