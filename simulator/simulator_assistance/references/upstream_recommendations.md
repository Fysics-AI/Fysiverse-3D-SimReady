# Upstream Recommendations

The current Fysiverse-3D-SimReady workflow already provides enough identity and geometry for first-pass rigid-body simulation.

Already useful upstream fields:

- `semantic_label`, `description`, `vlm_confidence`, `vlm_visibility`, and `vlm_support_status` per object in `final_scene_manifest.json`.
- `results/scene_graph.json` with mask-scoped object labels, relations, and `support_relations`.
- `support.support_relations` in `final_scene_manifest.json`, already mapped to `mask_XXX_object` names when available.
- `outputs.final_blend`, `outputs.gravity_settled_blend`, and fallback paths.
- `sapien_export` bbox/asset paths and `sapien_final_pose` for each final object.

Fields that would still improve automation quality if added upstream:

- `material_hint`: plastic, metal, paper, fabric, glass, wood, unknown.
- `physical_hint`: rigid, deformable, hollow, fragile, liquid container, articulated.
- `scale_confidence`: confidence that Blender/SAPIEN units are metric.
- `upright_axis` and object orientation hints: top, bottom, front, long axis.
- `contact_graph`: initial touching/nearby object pairs.
- `collision_quality`: convex collision status, mesh size, known failures.
- `task_role_candidate`: active/target/background if generated from a user task.
Do not add a ground-plane estimator upstream for this workflow; simulator_assistance fixes ground to `z=0` in a Z-up world.
