# Evaluation: HumanSplat (3DGS) vs current GLB path

Use this template when you have exports from both branches.

## Setup

- **GLB path**: `generated/task_*/garment.glb` from `tshirt_to_glb_pose_ready.py` + AR try-on page.
- **3DGS path**: HumanSplat (or other) `.ply` / `.splat` loaded in `pages/splat-viewer.html`.

## Metrics (qualitative / quantitative)

| Criterion | GLB + Pix2Surf | 3DGS human |
|-----------|----------------|------------|
| Silhouette fidelity (torso, sleeves) | Template mesh; good if OBJ matches product | View-dependent; may halo or thin geometry |
| Texture sharpness (print readability) | Strong when front/back photos align | Depends on training/views |
| Real-time AR on phone | Designed for this (single GLB + landmarks) | Heavier viewer; may need LOD or streaming |
| Garment swap / flat-lay → body | Pix2Surf UV on fixed topology | Requires fusion work (not in repo yet) |
| Time to first result | Minutes (CPU/GPU + Blender) | Upstream HumanSplat install + inference |

## Notes

- Record hardware, image resolution, and whether `rembg` was used (`garment_preprocess.json`).
- If comparing to Tencent Hunyuan, see `docs/HUNYUAN_EVAL.md` (mesh-centric baseline).
