# Hunyuan3D in this project

**Product:** [Hunyuan3D](https://3d.hunyuanglobal.com/) · **Tencent Cloud API:** [Quick start](https://www.tencentcloud.com/document/product/1284/75287)

## Integrated (this repo)

1. Install the international SDK: `pip install tencentcloud-sdk-python-intl-en` (also listed in `requirements-pipeline.txt`).
2. In `.env`, set either:
   - `TENCENTCLOUD_SECRET_ID` and `TENCENTCLOUD_SECRET_KEY`, or  
   - `HUNYUAN_SECRET_ID` and `HUNYUAN_SECRET_KEY` (aliases).
3. Optional: `HUNYUAN_3D_REGION` — `ap-singapore` (default) or `na-siliconvalley` per Tencent docs.
4. Activate **Tencent HY** in the [console](https://console.tencentcloud.com/hunyuan) and ensure the sub-account has policy **QcloudAI3DFullAccess**.
5. Open **`pages/hunyuan-studio.html`** (with `python server-local.py`) or call:
   - `POST /api/hunyuan/submit` — JSON body: `edition` (`pro` | `rapid`), and exactly one of `prompt`, `image` (data URI), or `image_url`.
   - `GET /api/hunyuan/query?job_id=…&edition=pro|rapid` — poll until `Status` is `DONE` or `FAIL`; download links appear in `ResultFile3Ds[].Url`.

Generated meshes are **not** auto-rigged into the AR try-on GLB path; use them for comparison, concept art, or manual retopo (see below).

---

## When Hunyuan is a good fit (vs Pix2Surf GLB)

- Rapid concept meshes from text or a single image without a fixed shirt template.
- Exploring stylised or loose clothing where exact retail fit is secondary.

## When the Pix2Surf / GLB path is a better fit

- **Controlled retail silhouette** — same `shirts.obj` topology for repeatable measurements and AR alignment.
- **Flat-lay + back photo** — Pix2Surf is trained for upper-body texture from front/back views.
- **On-device try-on** — one small GLB + `garment.json` landmark mapping vs large splat/mesh assets.

## Retargeting Hunyuan output into AR try-on (manual / research)

1. Download the model from `ResultFile3Ds` when the job is `DONE`.
2. Retopology or map to your fixed `shirts.obj` (UV transfer / cage projection in Blender).
3. Optionally bake textures and re-run weight transfer / export similar to `tshirt_to_glb_pose_ready.py`.

## Comparison log (fill in)

| Asset | Generator | Polycount | Texture res | AR try-on load OK? | Notes |
|-------|-----------|-----------|-------------|-------------------|-------|
| | | | | | |
