# Pipeline architecture (prototype)

## Live path (what `POST /api/generate` runs)

1. **Uploads** — front/back garment photos saved under `generated/task_<id>/`.
2. **Garment preprocess** — `pipeline/garment_preprocess.py`: `rembg` → OpenCV GrabCut → passthrough. Writes `garment_preprocess.json` with `fg_bbox` (xywh) per view and a sidecar `<name>.mask.png` that carries the true alpha silhouette downstream (keeps white / light-coloured garments from bleeding a studio backdrop onto the model).
3. **Mesh + texture** — `tshirt_to_glb_pose_ready.py`: body-anchored tile placement (landmarks from the sidecar mask) → Pix2Surf / direct / hybrid texture → SMPL skinning → Blender → `garment.glb` + `garment.json`.
4. **Try-on** — `apps/ar-tryon/` (Vite + TypeScript) loads the GLB and drives bones with MediaPipe pose. `server-local.py` serves the built bundle at `/dist/apps/ar-tryon/` and redirects `/` there.

## Offline / research scripts (not auto-chained into the live path)

| Stage | Script | Role |
|--------|--------|------|
| Segmentation | `pipeline/sam_segment.py` | Person / torso masks; optional SAM 1.x via `SAM_CHECKPOINT`. |
| Human 3DGS | `pipeline/run_humansplat.py` | Thin wrapper when `HUMANSPLAT_REPO` points at upstream HumanSplat. Exports land on disk; there is no UI viewer wired into the try-on. |

## Gaps vs full Gaussian fusion

- No optimisation loop that blends garment appearance onto a splat avatar or enforces multi-view consistency losses.
- The try-on uses in-browser MediaPipe pose; masks produced by `sam_segment.py` are for offline analysis only.
