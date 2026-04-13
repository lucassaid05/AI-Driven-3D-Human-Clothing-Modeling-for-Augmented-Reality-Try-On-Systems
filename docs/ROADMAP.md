# Roadmap (vs `GET /api/pipeline`)

The server exposes the canonical stage list as JSON: **`GET /api/pipeline`** (see also `pipeline/ARCHITECTURE.md`).

| Order | Stage | Target capability | In this repo today |
|------:|--------|-------------------|---------------------|
| 1 | Segmentation | Person + torso/shirt masks | **Partial** — `pipeline/sam_segment.py` (offline); live try-on uses MediaPipe in the browser |
| 2 | Human reconstruction | HumanSplat → 3DGS | **Partial** — `pipeline/run_humansplat.py` wrapper + `pages/splat-viewer.html`; not called from `POST /api/generate` |
| 3 | Garment processing | BG removal, print ROI, UV | **Partial** — `pipeline/garment_preprocess.py` + Pix2Surf in `tshirt_to_glb_pose_ready.py`; inpainting beyond front/back is not neural |
| 4 | Appearance fusion | Torso / view consistency | **Partial** — SMPL weight transfer to garment mesh; no splat-native fusion |
| 5 | Export | GLB + optional 3DGS | **Live** — GLB/JSON under `/generated/`; splats via manual load in the viewer |

**External / hybrid:** [HumanSplat](https://humansplat.github.io/) (offline wrapper). **Tencent Hunyuan 3D** is integrated via `POST /api/hunyuan/submit` when `.env` credentials are set — see `docs/HUNYUAN_EVAL.md` and `pages/hunyuan-studio.html`.
