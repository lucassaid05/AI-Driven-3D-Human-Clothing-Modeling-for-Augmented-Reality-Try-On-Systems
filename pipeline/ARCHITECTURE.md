# Pipeline architecture (prototype)

## Live path (what `POST /api/generate` runs)

1. **Uploads** — front/back garment photos saved under `generated/task_<id>/`.
2. **Garment preprocess** — `pipeline/garment_preprocess.py`: `rembg` → OpenCV GrabCut → passthrough. Writes `garment_preprocess.json` with `fg_bbox` (xywh) per view for print-region heuristics and dissertation logging.
3. **Mesh + texture** — `tshirt_to_glb_pose_ready.py`: Pix2Surf (neural UV texture), fixed shirt OBJ, SMPL skinning, Blender → `garment.glb` + `garment.json`.
4. **Try-on** — `legacy/ar-tryon-v13.html` (static) or `apps/ar-tryon/` (Vite + TypeScript) loads GLB and drives bones with MediaPipe pose.

## Offline / research paths (not auto-chained yet)

| Stage | Script | Role |
|--------|--------|------|
| Segmentation | `pipeline/sam_segment.py` | Person / torso masks; optional SAM 1.x via `SAM_CHECKPOINT` |
| Human 3DGS | `pipeline/run_humansplat.py` | Thin wrapper when `HUMANSPLAT_REPO` points at upstream HumanSplat |
| 3DGS preview | `pages/splat-viewer.html` | Load `.ply` / `.splat` / `.ksplat` exports |

## Tencent Hunyuan 3D (optional)

- **REST:** `POST /api/hunyuan/submit`, `GET /api/hunyuan/query` — see `docs/HUNYUAN_EVAL.md`.
- **UI:** `pages/hunyuan-studio.html` (same origin as `server-local.py`).
- Outputs are **not** auto-imported into the SMPL garment pipeline.

## Gaps vs full Gaussian fusion

- No optimisation loop that blends garment appearance onto a splat avatar or enforces multi-view consistency losses.
