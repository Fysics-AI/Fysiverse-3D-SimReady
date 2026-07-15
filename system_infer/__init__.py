"""Lightweight inference wrapper package.

The submodules intentionally avoid heavy imports from here. Workers should
import the concrete wrapper they need, for example
``from system_infer.sam3d_infer import SAM3DInfer``.
"""

__all__ = [
    "sam3d_infer",
    "trellis_infer",
    "flux_infer",
    "gsam2_infer",
    "scenegen_infer",
    "g2vlm_infer",
]
