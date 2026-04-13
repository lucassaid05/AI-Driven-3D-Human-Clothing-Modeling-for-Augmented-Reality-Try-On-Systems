# Pix2Surf — Web AR try-on

A **dissertation prototype** that turns **front and back flat-lay photos of a shirt** into a **rigged GLB** (fixed shirt topology + Pix2Surf neural UV texture), then previews it in the browser with **MediaPipe pose–driven AR try-on**.

The project is deliberately scoped to **one app**: the browser AR try-on. The Python server's only job is to serve that built app and run the photo-to-GLB generation pipeline for it.

## Start here

- **If you want the demo working quickly**: jump to **[Quick start](#quick-start)**.
- **If your phone camera won’t open over Wi‑Fi**: read **[`docs/TUNNELS.md`](docs/TUNNELS.md)**.
- **If you’re changing pipeline behaviour**: read **[`pipeline/ARCHITECTURE.md`](pipeline/ARCHITECTURE.md)**.
- **If you’re looking for outputs**: check **`generated/task_<id>/`** (gitignored by design).

## Table of contents

- [Goals](#goals)
- [How it fits together](#how-it-fits-together)
  - [Live path (`POST /api/generate`)](#live-path-post-apigenerate)
  - [The one front-end app](#the-one-front-end-app)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
  - [1. Install and build the try-on](#1-install-and-build-the-try-on)
  - [2. Start the local server](#2-start-the-local-server)
  - [3. Develop the try-on with live reload (optional)](#3-develop-the-try-on-with-live-reload-optional)
- [Configuration (`.env`)](#configuration-env)
- [HTTP API](#http-api)
- [Repository layout](#repository-layout)
- [What is *not* committed to git](#what-is-not-committed-to-git)
- [Known limitations (honest scope)](#known-limitations-honest-scope)
- [License and academic use](#license-and-academic-use)

---

## Goals

1. **Controlled retail silhouette** — One canonical shirt mesh (`shirts.obj` topology) so measurements, UVs, and AR alignment stay comparable across runs and dissertation experiments.
2. **Photo-real print from two views** — Pix2Surf estimates texture from **front + back** images. Preprocessing persists the true garment silhouette (rembg/GrabCut alpha sidecar) and a body-anchored tile alignment guarantees that prints on the front and back land at matching body coordinates on the model, independently of how each photo was framed.
3. **On-device try-on** — A compact **GLB + `garment.json`** landmark mapping, driven by **MediaPipe** in the browser (no native app required).

---

## How it fits together

### Live path (`POST /api/generate`)

When you run `python server-local.py` and upload front/back photos from the AR try-on panel, the server will:

1. Save uploads under `generated/task_<id>/`.
2. Run **`pipeline/garment_preprocess.py`** (`rembg` → OpenCV GrabCut → passthrough) and write `garment_preprocess.json` plus a companion **alpha-mask PNG** (`<name>.mask.png`) next to each processed image.
3. Run **`tshirt_to_glb_pose_ready.py`**: body-anchored tile placement → Pix2Surf / direct / hybrid texture → SMPL skinning → Blender export → **`garment.glb`** + **`garment.json`**.
4. The **AR try-on** front-end loads that GLB and animates it with pose from the device camera.

Implementation details: [`pipeline/ARCHITECTURE.md`](pipeline/ARCHITECTURE.md).

### The one front-end app

- [`apps/ar-tryon/`](apps/ar-tryon/) — **Vite + TypeScript + vanilla CSS** AR try-on. Build with `npm run build`; the Python server then serves it at `/dist/apps/ar-tryon/index.html` and redirects `/` to that page. During development run `npm run dev` for Vite's HTTPS dev server (needed on phones so `getUserMedia` works over a LAN IP).

---

## Prerequisites

- **Node.js** (for Vite / TypeScript try-on and builds).
- **Python 3** with packages from [`requirements-pipeline.txt`](requirements-pipeline.txt), plus your Pix2Surf / PyTorch environment for `tshirt_to_glb_pose_ready.py`.
- **Blender** on `PATH` or configured via `.env` (`BLENDER_BIN`) — used for export/skinning in the GLB script.
- **Pix2Surf** checkout and **SMPL** model paths — the server defaults expect a layout relative to this repo (see **Configuration** below).

---

## Quick start

### 1. Install and build the try-on

```bash
npm install
npm run build
```

This writes `dist/apps/ar-tryon/index.html` (plus the bundled JS/CSS) — that is the only page this project exposes.

### 2. Start the local server

```bash
pip install -r requirements-pipeline.txt
python server-local.py
```

Open **`http://localhost:8000/`** (or the port set in `.env`). The server redirects `/` straight to the built try-on — no hub, no other pages. For phones on the same Wi‑Fi, use the **LAN URL** printed at startup. Camera access may require **HTTPS**; see [`docs/TUNNELS.md`](docs/TUNNELS.md).

### 3. Develop the try-on with live reload (optional)

```bash
npm run dev
```

Vite serves the TypeScript source over HTTPS at `https://<LAN-IP>:5173/apps/ar-tryon/` and proxies `/api/*` and `/generated/*` to the Python server on port 8000. Keep `python server-local.py` running in parallel so `/api/generate` still works.

#### Which URL should I open?

| You are… | Run | Open |
|---|---|---|
| Just trying the built app | `npm run build` then `python server-local.py` | `http://localhost:8000/` |
| Editing front-end code | `python server-local.py` + `npm run dev` | `https://<LAN-IP>:5173/apps/ar-tryon/` |

---

## Configuration (`.env`)

Create a **`.env`** file next to `server-local.py`. It is **gitignored** — do not commit secrets. The full list of keys lives in `CONFIG` inside [`server-local.py`](server-local.py); the most common ones are documented in the README table in this repo.

Use `.env.example` as a starting point.

---

## HTTP API

All endpoints live on the Python server (`server-local.py`). The AR try-on is the only client that calls them.

| Route | Purpose |
|-------|---------|
| `GET /` | 302 → `/dist/apps/ar-tryon/index.html` (the single page). |
| `GET /api/health` | Liveness ping used by the try-on's status dot. |
| `GET /api/config` | Reports whether Pix2Surf, SMPL, Blender and the garment script are all reachable. |
| `GET /api/server-info` | LAN URLs, scheme, Vite port, and camera/HTTPS hints for QR codes / bookmarks. |
| `GET /api/garments` | Library of previously generated GLBs under `OUTPUT_DIR`. |
| `POST /api/generate` | Queue a GLB generation job from two uploaded photos. |
| `GET /api/status/<task_id>` | Poll a generation task. |
| `GET /generated/<task>/…` | Static access to a task's outputs (GLB, JSON, processed images). |

Exact payloads and behaviour are implemented in `server-local.py`.

---

## Repository layout

```
apps/ar-tryon/                # Vite + TS AR try-on (the only front-end app)
src/apps/ar-tryon/            # TypeScript sources for the try-on
public/quick-generation/      # Reference photos for "Quick Generation"
public/sam3/                  # Canonical shirt GLB reference
pipeline/                     # Python preprocess + offline research helpers
docs/                         # Mobile/TLS tunnels + notes
server-local.py               # Static file server + /api/generate
tshirt_to_glb_pose_ready.py   # Pix2Surf → SMPL → Blender → GLB
index.html                    # Minimal redirect to the built try-on
package.json                  # npm scripts (dev, build, preview)
```

**`generated/`** holds per-task outputs (images, GLB, JSON). It is listed in **`.gitignore`** to avoid committing large binaries.

---

## What is *not* committed to git

- `node_modules/` (installed dependencies)
- `dist/` (build output; rebuild with `npm run build`)
- `.env` (local paths and secrets; use `.env.example` as a template)
- `generated/` (GLBs, intermediate images, per-run artifacts)

---

## Known limitations (honest scope)

- **Template topology only** — every GLB shares the same shirt mesh, so irregular necklines / open-fronts won’t match geometry exactly; textures and prints still appear.
- **Phone HTTPS** — `getUserMedia` over a LAN IP often requires HTTPS; see [`docs/TUNNELS.md`](docs/TUNNELS.md).

---

## License and academic use

This repository supports a **Level 6 dissertation** (Pix2Surf / virtual try-on). Third-party components (Pix2Surf, SMPL, Three.js, MediaPipe, etc.) remain under **their respective licenses**.
