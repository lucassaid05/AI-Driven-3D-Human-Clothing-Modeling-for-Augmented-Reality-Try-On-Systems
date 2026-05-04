r"""
tshirt_to_glb_pose_ready.py

Enhanced version of tshirt_to_glb_skinned.py that:
1. Exports a GLB with SMPL armature for real-time skinning
2. Exports a JSON metadata file mapping SMPL joints to MediaPipe Pose landmarks
3. Includes bind pose joint positions for calibration

This enables a web frontend to:
- Load the GLB and apply real-time pose from MediaPipe
- Map MediaPipe's 33 landmarks to SMPL's 24 joints
- Correctly scale and position the garment on detected humans

Usage:
python tshirt_to_glb_pose_ready.py \
  --pix2surf_root ./pix2surf \
  --front_img ./mu_front.png \
  --back_img ./mu_back.png \
  --smpl_model SMPL_NEUTRAL.pkl \
  --garment_obj ./pix2surf/test_data/meshes/shorts/upper_0.obj \
  --blender_bin "blender" \
  --out_glb ./out/tshirt_pose_ready.glb \
  --device cuda:0 \
  --smooth_weights
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

# Optional dependencies
try:
    import cv2
except Exception:
    cv2 = None

# PyTorch is only needed for pix2surf/hybrid — lazy-import to keep direct+rigid runs fast on CPU.
torch = None
nn = None
F = None
transforms = None


def _ensure_torch() -> None:
    global torch, nn, F, transforms
    if torch is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        from torchvision import transforms as _transforms

        torch = _torch
        nn = _nn
        F = _F
        transforms = _transforms
    except Exception as e:
        _die(f"PyTorch/torchvision required for this texture mode: {e}")

try:
    import trimesh
except Exception:
    trimesh = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# -------------------------
# MediaPipe to SMPL Joint Mapping
# -------------------------
# MediaPipe Pose has 33 landmarks, SMPL has 24 joints
# This mapping allows us to drive SMPL bones from MediaPipe detections

MEDIAPIPE_TO_SMPL_MAPPING = {
    # MediaPipe idx -> SMPL joint name
    # Core body
    0: None,  # nose - no direct SMPL equivalent
    11: "left_shoulder",   # left shoulder
    12: "right_shoulder",  # right shoulder
    13: "left_elbow",      # left elbow
    14: "right_elbow",     # right elbow
    15: "left_wrist",      # left wrist
    16: "right_wrist",     # right wrist
    23: "left_hip",        # left hip
    24: "right_hip",       # right hip
    25: "left_knee",       # left knee
    26: "right_knee",      # right knee
    27: "left_ankle",      # left ankle
    28: "right_ankle",     # right ankle
}

# SMPL joint indices (matching _SMPL_24_NAMES)
SMPL_JOINT_INDICES = {
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2,
    "spine1": 3,
    "left_knee": 4, "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7, "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10, "right_foot": 11,
    "neck": 12,
    "left_collar": 13, "right_collar": 14,
    "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}

RELAXED_BIND_ROTATION_DEGREES = {
    "left_collar": -15.0,
    "right_collar": 15.0,
    "left_shoulder": -52.0,
    "right_shoulder": 52.0,
    "left_elbow": -10.0,
    "right_elbow": 10.0,
}

# Upper body joints relevant for t-shirt (we mainly animate these)
UPPER_BODY_JOINTS = [
    "pelvis", "spine1", "spine2", "spine3", "neck",
    "left_collar", "right_collar",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

_SMPL_24_NAMES = [
    "pelvis",
    "left_hip", "right_hip",
    "spine1",
    "left_knee", "right_knee",
    "spine2",
    "left_ankle", "right_ankle",
    "spine3",
    "left_foot", "right_foot",
    "neck",
    "left_collar", "right_collar",
    "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]


def _die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def _rot_z(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _smpl_parents(smpl: SMPLModel) -> np.ndarray:
    kt = np.asarray(smpl.kintree_table)
    if kt.shape[0] != 2:
        _die(f"Unexpected kintree_table shape: {kt.shape}")
    parents = kt[0].astype(np.int32).copy()
    parents[0] = -1
    return parents


def _fk_bind_pose(smpl: SMPLModel, bind_pose: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward kinematics for SMPL bind pose.
    Returns (parents, world_joints, world_rots).
    Weight transfer uses shirts.obj in T-pose space; bind_pose 'relaxed' rotates arms down.
    """
    if bind_pose not in {"t_pose", "relaxed"}:
        _die(f"Unsupported bind pose '{bind_pose}' (expected t_pose or relaxed)")

    joints_ref = np.asarray(smpl.joints, dtype=np.float32)
    parents = _smpl_parents(smpl)
    J = joints_ref.shape[0]

    local_offsets = np.zeros((J, 3), dtype=np.float32)
    for i in range(J):
        p = int(parents[i])
        local_offsets[i] = joints_ref[i] if p < 0 else (joints_ref[i] - joints_ref[p])

    local_rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], J, axis=0)
    if bind_pose == "relaxed":
        for joint_name, deg in RELAXED_BIND_ROTATION_DEGREES.items():
            idx = SMPL_JOINT_INDICES.get(joint_name, None)
            if idx is not None and idx < J:
                local_rots[idx] = _rot_z(deg) @ local_rots[idx]

    world_rots = np.zeros((J, 3, 3), dtype=np.float32)
    world_joints = np.zeros((J, 3), dtype=np.float32)
    for i in range(J):
        p = int(parents[i])
        if p < 0:
            world_rots[i] = local_rots[i]
            world_joints[i] = local_offsets[i]
        else:
            world_rots[i] = world_rots[p] @ local_rots[i]
            world_joints[i] = world_joints[p] + (world_rots[p] @ local_offsets[i])
    return parents, world_joints, world_rots


def _compute_bind_pose_joints(smpl: SMPLModel, bind_pose: str) -> np.ndarray:
    """World joint positions for the chosen bind pose (see _fk_bind_pose)."""
    _, world_joints, _ = _fk_bind_pose(smpl, bind_pose)
    return world_joints


def _joint_global_matrices(world_rots: np.ndarray, world_joints: np.ndarray) -> np.ndarray:
    """(J, 4, 4) global rigid transforms from FK world rotation + joint position."""
    J = world_joints.shape[0]
    mats = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], J, axis=0)
    mats[:, :3, :3] = world_rots
    mats[:, :3, 3] = world_joints
    return mats


def _pose_garment_vertices_to_bind(
    garment_V: np.ndarray,
    garment_weights: np.ndarray,
    smpl: SMPLModel,
    bind_pose: str,
) -> np.ndarray:
    """
    Deform garment mesh from T-pose (weight-transfer rest) into the target bind pose.
    Without this, BIND_POSE only moves bones while shirts.obj keeps arms-out geometry.
    """
    if bind_pose == "t_pose":
        return garment_V

    _, rest_j, rest_r = _fk_bind_pose(smpl, "t_pose")
    _, bind_j, bind_r = _fk_bind_pose(smpl, bind_pose)
    G_rest = _joint_global_matrices(rest_r, rest_j)
    G_bind = _joint_global_matrices(bind_r, bind_j)
    M = np.matmul(G_bind, np.linalg.inv(G_rest)).astype(np.float32)

    W = np.asarray(garment_weights, dtype=np.float32)
    V = np.asarray(garment_V, dtype=np.float32)
    if W.shape[0] != V.shape[0]:
        _die("garment weights row count must match vertex count")
    J = min(W.shape[1], M.shape[0])
    W = W[:, :J]
    M = M[:J]

    V_h = np.concatenate([V, np.ones((V.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.zeros_like(V)
    for j in range(J):
        w = W[:, j]
        if not np.any(w > 1e-6):
            continue
        transformed = (M[j] @ V_h.T).T[:, :3]
        posed += w[:, None] * transformed

    span_before = float(V[:, 0].max() - V[:, 0].min())
    span_after = float(posed[:, 0].max() - posed[:, 0].min())
    print(
        f"[OK] Posed garment mesh to bind_pose={bind_pose}: "
        f"shoulder span X {span_before:.4f} → {span_after:.4f}",
        flush=True,
    )
    return posed.astype(np.float32)


def _ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        _die(f"Missing {what}: {p}")


def _limit_skin_influences_to_four(W: np.ndarray) -> np.ndarray:
    """
    Keep the 4 strongest influences per vertex and renormalize rows to sum 1.
    Matches core glTF skinning (JOINTS_0 + WEIGHTS_0, VEC4) and reference GLBs.
    """
    W = np.maximum(W.astype(np.float32), 0.0)
    n, J = W.shape
    k = min(4, J)
    out = np.zeros_like(W, dtype=np.float32)
    idx_part = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    for i in range(n):
        idx = idx_part[i]
        vals = W[i, idx]
        order = np.argsort(-vals)
        idx = idx[order]
        vals = vals[order]
        s = float(vals.sum())
        if s < 1e-8:
            out[i, 0] = 1.0
        else:
            out[i, idx] = vals / s
    return out


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _to_jpg_rgb(src: Path, dst: Path, quality: int = 95) -> None:
    img = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=quality)
    _copy_mask_sidecar(src, dst)


def _sidecar_mask_path(image_path: Path) -> Path:
    """Canonical companion path for the garment alpha sidecar (see pipeline/garment_preprocess.py)."""
    p = Path(image_path)
    return p.with_name(p.stem + ".mask.png")


def _load_sidecar_mask(image_path: Path) -> Optional[np.ndarray]:
    """Return a boolean HxW mask (True = garment) if a sidecar exists, else None."""
    mask_path = _sidecar_mask_path(image_path)
    if not mask_path.exists():
        return None
    try:
        m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        return m >= 128
    except Exception as e:
        print(f"[WARN] Failed to read sidecar mask {mask_path.name}: {e}")
        return None


def _copy_mask_sidecar(src_image: Path, dst_image: Path) -> Optional[Path]:
    """
    If `src_image` has a sidecar mask, persist a matching one next to `dst_image`.
    Resizes to the destination image size when it differs from the mask.
    """
    src_mask = _sidecar_mask_path(src_image)
    if not src_mask.exists():
        return None
    try:
        with Image.open(dst_image) as im:
            target_size = im.size
        m_img = Image.open(src_mask).convert("L")
        if m_img.size != target_size:
            m_img = m_img.resize(target_size, Image.NEAREST)
        arr = np.asarray(m_img, dtype=np.uint8)
        arr = np.where(arr >= 128, 255, 0).astype(np.uint8)
        dst_mask = _sidecar_mask_path(dst_image)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="L").save(dst_mask, "PNG", optimize=True)
        return dst_mask
    except Exception as e:
        print(f"[WARN] Failed to propagate mask sidecar {src_mask.name}: {e}")
        return None


def _border_strip_samples_rgb(arr: np.ndarray, strip: int) -> np.ndarray:
    """RGB samples along the image edges (studio / rembg backgrounds are uniform there)."""
    h, w, _ = arr.shape
    s = max(1, min(strip, h // 2, w // 2))
    parts = [
        arr[:s, :, :].reshape(-1, 3),
        arr[-s:, :, :].reshape(-1, 3),
        arr[:, :s, :].reshape(-1, 3),
        arr[:, -s:, :].reshape(-1, 3),
    ]
    return np.vstack(parts)


def _foreground_mask_product_shot(
    arr: np.ndarray,
    *,
    white_thresh: int,
    mask_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Foreground mask for a flat-lay garment.

    If a ``mask_hint`` is provided (e.g. the rembg alpha sidecar from
    `pipeline/garment_preprocess.py`), it is used verbatim — this is essential
    for white / near-white garments (cardigans, cream shirts, etc.) where the
    brightness-based fallback cannot distinguish garment from a white backdrop
    and would otherwise let the backdrop leak into the model texture.

    Fallback heuristics (no mask hint):
    - White / near-white backdrops (post-rembg JPEGs)
    - Light grey studio backgrounds (#bdbdbd–#e8e8e8) where a fixed white threshold fails
    """
    if mask_hint is not None:
        mh = np.asarray(mask_hint)
        if mh.shape[:2] == arr.shape[:2]:
            return mh.astype(bool)
        print(
            f"[WARN] sidecar mask shape {mh.shape[:2]} != image {arr.shape[:2]} — ignoring hint"
        )

    h, w, _ = arr.shape
    strip = max(3, min(h, w) // 40)
    samples = _border_strip_samples_rgb(arr, strip).astype(np.float32)
    bg_med = np.median(samples, axis=0)
    dev = np.abs(samples - bg_med.reshape(1, 3))
    per_sample = np.max(dev, axis=1)
    tol = float(np.percentile(per_sample, 90)) + 10.0
    tol = max(14.0, min(tol, 52.0))

    fg = np.max(np.abs(arr.astype(np.float32) - bg_med.reshape(1, 1, 3)), axis=-1) > tol

    if int(fg.sum()) < 500:
        bg = (arr[..., 0] >= white_thresh) & (arr[..., 1] >= white_thresh) & (arr[..., 2] >= white_thresh)
        fg = ~bg

    return fg


def _normalize_white_bg_product_shot(
    src: Path,
    dst: Path,
    *,
    out_size: int = 1024,
    white_thresh: int = 245,
    pad_frac: float = 0.08,
) -> None:
    """
    Crop garment from flat-lay; composite onto square white canvas for Pix2Surf.

    If a companion mask sidecar exists next to ``src`` (produced by
    `pipeline/garment_preprocess.py`), it is used as the authoritative garment
    silhouette and a matching, co-transformed mask is written next to ``dst``.
    This keeps the true garment region known downstream so no white backdrop
    ever leaks into the final texture — even for white / cream garments.
    """
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img)
    hint = _load_sidecar_mask(src)
    fg = _foreground_mask_product_shot(arr, white_thresh=white_thresh, mask_hint=hint)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if fg.sum() < 500:
        canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
        tmp = img.copy()
        tmp.thumbnail((out_size, out_size), Image.BICUBIC)
        ox = (out_size - tmp.size[0]) // 2
        oy = (out_size - tmp.size[1]) // 2
        canvas.paste(tmp, (ox, oy))
        canvas.save(dst, "JPEG", quality=95)
        # Co-transform the mask so the saved JPEG and sidecar stay aligned.
        mask_full = (fg.astype(np.uint8) * 255) if fg.any() else np.full(arr.shape[:2], 255, np.uint8)
        m_img = Image.fromarray(mask_full, mode="L").resize(tmp.size, Image.NEAREST)
        mask_canvas = Image.new("L", (out_size, out_size), 0)
        mask_canvas.paste(m_img, (ox, oy))
        mask_canvas.save(_sidecar_mask_path(dst), "PNG", optimize=True)
        return

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h = y1 - y0 + 1
    w = x1 - x0 + 1
    pad = int(max(h, w) * pad_frac)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)

    crop = img.crop((x0, y0, x1 + 1, y1 + 1))
    canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
    crop.thumbnail((out_size, out_size), Image.BICUBIC)
    ox = (out_size - crop.size[0]) // 2
    oy = (out_size - crop.size[1]) // 2
    canvas.paste(crop, (ox, oy))
    canvas.save(dst, "JPEG", quality=95)

    # Propagate the mask through the identical crop/resize/paste transform so the
    # foreground region stays pixel-aligned with the saved JPEG.
    mask_full = (fg.astype(np.uint8) * 255)
    m_img = Image.fromarray(mask_full, mode="L").crop((x0, y0, x1 + 1, y1 + 1))
    m_img = m_img.resize(crop.size, Image.NEAREST)
    mask_canvas = Image.new("L", (out_size, out_size), 0)
    mask_canvas.paste(m_img, (ox, oy))
    mask_canvas.save(_sidecar_mask_path(dst), "PNG", optimize=True)


def _upscale_min_edge(img_path: Path, min_edge: int) -> None:
    """
    If the longest image side is below min_edge, upscale in place with LANCZOS.
    Pix2Surf resizes to 256² internally; starting from a sharper source often
    improves garment texture detail. Use min_edge=0 to skip.
    """
    if min_edge <= 0:
        return
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m >= min_edge:
        return
    scale = min_edge / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    out = img.resize((nw, nh), resample)
    out.save(img_path, "JPEG", quality=95)
    print(
        f"[INFO] Prep upscale for Pix2Surf: {img_path.name} {w}x{h} → {nw}x{nh} (min_edge={min_edge})",
        flush=True,
    )
    # Keep the sidecar garment mask aligned with the upscaled JPEG.
    mask_path = _sidecar_mask_path(img_path)
    if mask_path.exists():
        try:
            m_img = Image.open(mask_path).convert("L").resize((nw, nh), Image.NEAREST)
            m_img.save(mask_path, "PNG", optimize=True)
        except Exception as e:
            print(f"[WARN] Could not upscale mask {mask_path.name}: {e}")


# -------------------------
# Pix2Surf texture inference
# -------------------------
@dataclass(frozen=True)
class Pix2SurfAssets:
    seg_shirts_front: Path
    seg_shirts_back: Path
    map_shirts_front: Path
    map_shirts_back: Path


def _dominant_garment_color(
    tile_rgb: np.ndarray, fg_mask: np.ndarray
) -> Optional[np.ndarray]:
    """
    Robust estimate of the garment's main body colour, used as a safety-net
    fill for any non-foreground region the atlas leaves uncovered.

    Ignores near-white / near-black pixels (logos, prints, sleeve trim) and
    returns the median of the remaining foreground pixels.  None is returned
    when the mask is empty.
    """
    if not np.any(fg_mask):
        return None
    rgb_fg = tile_rgb[fg_mask]
    if rgb_fg.size == 0:
        return None
    lum = rgb_fg.astype(np.int32).sum(axis=1)
    keep = (lum > 60) & (lum < 720)
    if int(keep.sum()) >= 50:
        rgb_fg = rgb_fg[keep]
    return np.median(rgb_fg.astype(np.float32), axis=0).astype(np.uint8)


def _stabilize_tile_background(tile_rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Fill non-garment pixels with nearby garment colours so UV edge samples never
    fall back to white background (a common cause of sleeve/side white strips).

    Strategy:
      1. Repaint the entire tile with the dominant garment colour as a safe
         base (prevents pure-white areas when the garment only covers a
         fraction of the tile, e.g. Arsenal front: red body in centre, large
         white margins on all sides).
      2. Overlay the true foreground pixels so logos/prints stay intact.
      3. Extend the garment edges horizontally and vertically into any
         remaining uncovered rows for a natural-looking UV-safe tile.
    """
    h, w = fg_mask.shape
    dom = _dominant_garment_color(tile_rgb, fg_mask)
    if dom is not None:
        out = np.broadcast_to(dom.reshape(1, 1, 3), (h, w, 3)).astype(np.uint8).copy()
        # Preserve original pixels where the mask says garment.
        out[fg_mask] = tile_rgb[fg_mask]
    else:
        out = tile_rgb.copy()

    def _near_garment(px: np.ndarray) -> bool:
        if dom is None:
            return True
        # Reject edge samples that look like bright background leakage so the
        # extension/fill never turns a coloured garment into white borders.
        lum = int(px.astype(np.int32).sum())
        if lum > 720:
            return False
        return True

    # Horizontal extension per row.
    for y in range(h):
        xs = np.where(fg_mask[y])[0]
        if xs.size == 0:
            continue
        lx, rx = int(xs.min()), int(xs.max())
        left_px = out[y, lx]
        right_px = out[y, rx]
        if dom is not None and not _near_garment(left_px):
            left_px = dom
        if dom is not None and not _near_garment(right_px):
            right_px = dom
        out[y, :lx] = left_px
        out[y, rx + 1 :] = right_px

    # Vertical extension for rows with no foreground.  Guard each row we copy
    # from against bright background leakage so a single white hem corner does
    # not flood the entire lower half of the tile with white.
    fg_rows = np.where(np.any(fg_mask, axis=1))[0]
    if fg_rows.size > 0:
        top_idx = int(fg_rows.min())
        bot_idx = int(fg_rows.max())
        top_row = out[top_idx]
        bot_row = out[bot_idx]
        if dom is not None:
            top_lum = top_row.astype(np.int32).sum(axis=1)
            bot_lum = bot_row.astype(np.int32).sum(axis=1)
            # Replace stray near-white samples with the dominant garment colour.
            top_row = np.where(top_lum[:, None] > 720, dom.reshape(1, 3), top_row).astype(np.uint8)
            bot_row = np.where(bot_lum[:, None] > 720, dom.reshape(1, 3), bot_row).astype(np.uint8)
        for y in range(0, top_idx):
            out[y] = top_row
        for y in range(bot_idx + 1, h):
            out[y] = bot_row
        for y in range(h):
            if np.any(fg_mask[y]):
                continue
            nearest = int(fg_rows[np.argmin(np.abs(fg_rows - y))])
            src_row = out[nearest]
            if dom is not None:
                lum = src_row.astype(np.int32).sum(axis=1)
                src_row = np.where(lum[:, None] > 720, dom.reshape(1, 3), src_row).astype(np.uint8)
            out[y] = src_row

    # Optional inpaint pass for any remaining tiny holes inside the garment.
    # Restrict inpainting to small holes contained within the mask bounding
    # box so it does not diffuse bright background colours across the large
    # empty region below a centrally-framed garment photo.
    if cv2 is not None and fg_rows.size > 0:
        fg_cols = np.where(np.any(fg_mask, axis=0))[0]
        if fg_cols.size > 0:
            y0 = int(fg_rows.min())
            y1 = int(fg_rows.max()) + 1
            x0 = int(fg_cols.min())
            x1 = int(fg_cols.max()) + 1
            holes_full = (~fg_mask).astype(np.uint8) * 255
            holes_crop = np.zeros_like(holes_full)
            holes_crop[y0:y1, x0:x1] = holes_full[y0:y1, x0:x1]
            if np.any(holes_crop):
                num, labels, stats, _ = cv2.connectedComponentsWithStats(holes_crop, 8)
                small = np.zeros_like(holes_crop)
                max_area = max(64, int(0.02 * (y1 - y0) * (x1 - x0)))
                for i in range(1, num):
                    if int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
                        small[labels == i] = 255
                if np.any(small):
                    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    out_bgr = cv2.inpaint(out_bgr, small, 3, cv2.INPAINT_TELEA)
                    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out


def _body_landmarks_from_mask(mask: np.ndarray) -> dict:
    """
    Robust flat-lay garment landmarks derived from pixel mass:
      - y_shoulder : 5th-percentile row of cumulative foreground mass
      - y_hem      : 95th-percentile row
      - y_mid      : 50th-percentile row
      - x_center   : median of per-row bbox centres between shoulder and hem
    Pixel-mass percentiles are used instead of the top/bottom of the bounding
    box because they are insensitive to thin sleeves poking above the
    shoulder or below the hem — the torso dominates the mass. This makes
    placement consistent across front/back photos regardless of how the
    photographer arranged the sleeves.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2:
        return {}
    row_area = m.sum(axis=1).astype(np.float64)
    total = float(row_area.sum())
    if total <= 0.0:
        return {}
    cum = np.cumsum(row_area) / total

    def _y_at(q: float) -> int:
        idx = int(np.searchsorted(cum, q))
        return int(min(len(cum) - 1, max(0, idx)))

    y_top = _y_at(0.05)
    y_mid = _y_at(0.50)
    y_bot = _y_at(0.95)

    centres: list[float] = []
    for y in range(y_top, y_bot + 1):
        xs = np.where(m[y])[0]
        if xs.size > 0:
            centres.append(0.5 * (float(xs.min()) + float(xs.max())))
    x_center = int(np.median(centres)) if centres else m.shape[1] // 2

    return {
        "y_shoulder": y_top,
        "y_hem": y_bot,
        "y_mid": y_mid,
        "x_center": x_center,
    }


def _align_garment_to_tile(
    arr: np.ndarray,
    fg: np.ndarray,
    tw: int,
    th: int,
    *,
    shoulder_y_frac: float = 0.12,
    hem_y_frac: float = 0.92,
    side_margin_frac: float = 0.06,
    garment_fill_frac: float = 1.0,
    mode: str = "shoulder",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Place the input garment onto a front/back tile.

    Two placement modes are supported:

    ``mode="shoulder"`` *(default)* — landmark-anchored alignment:
        - The image's detected **shoulder line** is mapped to
          ``shoulder_y_frac * th`` in tile space (matches the mesh's shoulder
          UV row ≈ 12% from tile top).
        - The image's **hem line** is mapped to ``hem_y_frac * th``
          (≈ 92% from tile top; matches the mesh's hem UV row).
        - The image's **torso centre x** is mapped to the tile's horizontal
          centre.
        - This ensures the shoulders/collar/sleeves of the printed graphic
          land on the shoulder vertices of the 3D mesh, and the hem of the
          graphic lands on the hem of the mesh — no dominant-colour fill
          leaks into the shoulder/neck region of the garment.
        - ``garment_fill_frac`` (default 1.0) optionally scales the garment
          down uniformly below the landmark fit (e.g. 0.85 → 85% of the
          shoulder-to-hem landmark span, re-centred around the shoulder/hem
          midline).

    ``mode="center"`` — geometric centre-anchored alignment (legacy):
        - The image's geometric centre is mapped to the tile centre.
        - ``garment_fill_frac`` controls how much of the tile the garment
          bbox fills on both axes (0.70 = 70% of limiting axis).

    In both modes ``side_margin_frac`` enforces a hard minimum horizontal
    margin so the garment never touches the tile seam.

    Returns ``(rgb, fg)`` at the tile resolution, or ``None`` if the mask is
    empty / too small (caller should fall back).
    """
    ys_m, xs_m = np.where(fg)
    if ys_m.size == 0 or xs_m.size == 0:
        return None
    bbox_y0 = int(ys_m.min())
    bbox_y1 = int(ys_m.max())
    bbox_x0 = int(xs_m.min())
    bbox_x1 = int(xs_m.max())
    mh = bbox_y1 - bbox_y0 + 1
    mw = bbox_x1 - bbox_x0 + 1
    if mh < 8 or mw < 8:
        return None

    h_img, w_img = arr.shape[:2]

    # --- Choose scale and target anchor points based on mode. ---
    target_x = tw / 2.0  # horizontal centre for all modes

    if mode == "shoulder":
        # Mass-percentile landmarks are robust to sleeves poking above the
        # shoulder line or asymmetric framing.
        lm = _body_landmarks_from_mask(fg)
        if not lm:
            return None
        y_sh = float(lm["y_shoulder"])
        y_hem = float(lm["y_hem"])
        x_mid = float(lm["x_center"])
        if y_hem - y_sh < 8.0:
            return None

        target_y_sh = shoulder_y_frac * th
        target_y_hem = hem_y_frac * th
        span_img = y_hem - y_sh
        span_tile = target_y_hem - target_y_sh
        scale_v = (span_tile / span_img) * float(garment_fill_frac)
        scale_w_cap = (tw * (1.0 - 2.0 * side_margin_frac)) / float(mw)
        scale = max(1e-6, min(scale_v, scale_w_cap))

        # After scaling, place so that the midpoint between shoulder and hem
        # lands at the midpoint between their tile targets. This keeps the
        # shoulder exactly on ``shoulder_y_frac`` whenever width is not the
        # limiter; otherwise it splits the vertical slack evenly between
        # above-shoulder and below-hem so the graphic stays centred on the
        # torso band.
        y_mid_img = 0.5 * (y_sh + y_hem)
        target_y_mid = 0.5 * (target_y_sh + target_y_hem)
        src_cx, src_cy = x_mid, y_mid_img
        dst_cx, dst_cy = target_x, target_y_mid
    else:
        # Legacy geometric-centre mode.
        frac_w = min(garment_fill_frac, 1.0 - 2.0 * side_margin_frac)
        scale_h = (th * garment_fill_frac) / float(mh)
        scale_w = (tw * frac_w) / float(mw)
        scale = max(1e-6, min(scale_h, scale_w))
        src_cx, src_cy = w_img / 2.0, h_img / 2.0
        dst_cx, dst_cy = target_x, th / 2.0

    nw = max(1, int(round(w_img * scale)))
    nh = max(1, int(round(h_img * scale)))

    if cv2 is not None:
        rgb_resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        mask_u8 = (fg.astype(np.uint8) * 255)
        mask_resized = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST) >= 128
    else:
        rgb_resized = np.asarray(
            Image.fromarray(arr).resize((nw, nh), Image.LANCZOS), dtype=np.uint8
        )
        mask_resized = np.asarray(
            Image.fromarray((fg.astype(np.uint8) * 255), mode="L").resize((nw, nh), Image.NEAREST),
            dtype=np.uint8,
        ) >= 128

    # Place so (src_cx, src_cy) in the scaled image lands at (dst_cx, dst_cy)
    # in the tile.
    ox = int(round(dst_cx - src_cx * scale))
    oy = int(round(dst_cy - src_cy * scale))

    canvas_rgb = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas_fg = np.zeros((th, tw), dtype=bool)

    src_y0 = max(0, -oy)
    src_y1 = min(nh, th - oy)
    src_x0 = max(0, -ox)
    src_x1 = min(nw, tw - ox)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return None

    dst_y0 = oy + src_y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = ox + src_x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = rgb_resized[src_y0:src_y1, src_x0:src_x1]
    canvas_fg[dst_y0:dst_y1, dst_x0:dst_x1] = mask_resized[src_y0:src_y1, src_x0:src_x1]
    return canvas_rgb, canvas_fg


def _use_texture_full_bleed() -> bool:
    """
    When true, scale each photo to cover the full front/back tile and map mesh UVs
    across the entire tile. Avoids gray/white shoulder–hem margin bands on the model.
    Set GARMENT_TEXTURE_FULL_BLEED=0 to restore landmark tile alignment.
    """
    v = os.environ.get("GARMENT_TEXTURE_FULL_BLEED", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _crop_tile_cover_texture(path: Path, tw: int, th: int, thresh: int = 238) -> Tuple[np.ndarray, np.ndarray]:
    """Place the garment photo so it covers the entire tile (object-fit: cover)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    hint = _load_sidecar_mask(path)
    fg = _foreground_mask_product_shot(arr, white_thresh=thresh, mask_hint=hint)
    if int(fg.sum()) < 500:
        rgb = np.array(img.resize((tw, th), Image.LANCZOS))
        return rgb, np.ones((th, tw), dtype=bool)

    ys, xs = np.where(fg)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    pad = int(max(y1 - y0, x1 - x0) * 0.04)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)

    crop_rgb = arr[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_fg = fg[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop_rgb.shape[0], crop_rgb.shape[1]
    scale = max(float(tw) / max(cw, 1), float(th) / max(ch, 1))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))

    if cv2 is not None:
        rgb_big = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        mask_big = cv2.resize(
            (crop_fg.astype(np.uint8) * 255), (nw, nh), interpolation=cv2.INTER_NEAREST
        ) >= 128
    else:
        rgb_big = np.asarray(Image.fromarray(crop_rgb).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
        mask_big = np.asarray(
            Image.fromarray((crop_fg.astype(np.uint8) * 255), mode="L").resize((nw, nh), Image.NEAREST),
            dtype=np.uint8,
        ) >= 128

    ox = max(0, (nw - tw) // 2)
    oy = max(0, (nh - th) // 2)
    canvas = rgb_big[oy : oy + th, ox : ox + tw].copy()
    fg_canvas = mask_big[oy : oy + th, ox : ox + tw].copy()
    if canvas.shape[0] != th or canvas.shape[1] != tw:
        canvas = np.asarray(Image.fromarray(canvas).resize((tw, th), Image.LANCZOS), dtype=np.uint8)
        fg_canvas = np.asarray(
            Image.fromarray((fg_canvas.astype(np.uint8) * 255), mode="L").resize((tw, th), Image.NEAREST),
            dtype=np.uint8,
        ) >= 128

    print(f"  [tile-cover] {path.name}: full-bleed {tw}x{th}, fg={int(fg_canvas.sum())}/{th * tw}", flush=True)
    filled = _stabilize_tile_background(canvas, fg_canvas) if np.any(fg_canvas) else canvas
    return filled, fg_canvas


def _crop_tile_for_texture(path: Path, tw: int, th: int, thresh: int = 238) -> Tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    hint = _load_sidecar_mask(path)
    fg = _foreground_mask_product_shot(arr, white_thresh=thresh, mask_hint=hint)
    if int(fg.sum()) < 500:
        rgb = np.array(img.resize((tw, th), Image.LANCZOS))
        return rgb, np.ones((th, tw), dtype=bool)

    if _use_texture_full_bleed():
        return _crop_tile_cover_texture(path, tw, th, thresh=thresh)

    # Landmark-anchored placement by default (mode="shoulder"): the shirt's
    # detected shoulder line lands on the mesh's shoulder UV row and its hem
    # lands on the mesh's hem UV row, so the shoulders/sleeves of the printed
    # graphic align with the shoulders/sleeves of the 3D mesh (no dominant-
    # colour fill leaks into the collar/neck region). Set
    # GARMENT_TILE_ALIGN_MODE=center for legacy image-centre placement, or
    # GARMENT_TILE_ALIGN_TEMPLATE=0 to disable the overlay entirely.
    if os.environ.get("GARMENT_TILE_ALIGN_TEMPLATE", "1") != "0":
        mode = (os.environ.get("GARMENT_TILE_ALIGN_MODE", "shoulder") or "shoulder").strip().lower()
        if mode not in ("shoulder", "center"):
            mode = "shoulder"
        default_fill = "1.0" if mode == "shoulder" else "0.70"
        try:
            fill_f = float(os.environ.get("GARMENT_TILE_FILL_FRAC", default_fill) or default_fill)
        except ValueError:
            fill_f = 1.0 if mode == "shoulder" else 0.70
        try:
            sh_y = float(os.environ.get("GARMENT_TILE_SHOULDER_FRAC", "0.12") or "0.12")
        except ValueError:
            sh_y = 0.12
        try:
            hem_y = float(os.environ.get("GARMENT_TILE_HEM_FRAC", "0.92") or "0.92")
        except ValueError:
            hem_y = 0.92
        aligned = _align_garment_to_tile(
            arr, fg, tw, th,
            shoulder_y_frac=sh_y,
            hem_y_frac=hem_y,
            garment_fill_frac=fill_f,
            mode=mode,
        )
        if aligned is not None:
            rgb_canvas, fg_canvas = aligned
            if mode == "shoulder":
                desc = (
                    f"shoulder→{sh_y:.2f}·th, hem→{hem_y:.2f}·th, x→tile-centre, "
                    f"fill={fill_f:.2f}"
                )
            else:
                desc = f"image-centre→tile-centre, fill={fill_f:.2f}"
            print(
                f"  [tile-align] {path.name}: {desc}, "
                f"fg={int(fg_canvas.sum())}/{th * tw}",
                flush=True,
            )
            filled = _stabilize_tile_background(rgb_canvas, fg_canvas) if np.any(fg_canvas) else rgb_canvas
            return filled, fg_canvas
        print(f"  [tile-align] {path.name}: alignment skipped — falling back to bbox-center")

    ys, xs = np.where(fg)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    pad = int(max(h, w) * 0.05)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)

    crop_rgb = arr[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_fg = fg[y0 : y1 + 1, x0 : x1 + 1]

    # Resize crop + mask together so detail stays aligned.
    crop_img = Image.fromarray(crop_rgb)
    mask_img = Image.fromarray((crop_fg.astype(np.uint8) * 255), mode="L")
    scale = min(float(tw) / max(crop_rgb.shape[1], 1), float(th) / max(crop_rgb.shape[0], 1))
    nw = max(1, int(round(crop_rgb.shape[1] * scale)))
    nh = max(1, int(round(crop_rgb.shape[0] * scale)))
    crop_resized = np.asarray(crop_img.resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    mask_resized = np.asarray(mask_img.resize((nw, nh), Image.NEAREST), dtype=np.uint8) >= 128

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    fg_canvas = np.zeros((th, tw), dtype=bool)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = crop_resized
    fg_canvas[oy : oy + nh, ox : ox + nw] = mask_resized

    filled = _stabilize_tile_background(canvas, fg_canvas) if np.any(fg_canvas) else canvas
    return filled, fg_canvas


def _build_direct_texture_atlas(front_jpg: Path, back_jpg: Path) -> np.ndarray:
    if cv2 is None:
        _die("OpenCV not available. Install opencv-python.")
    f_direct, f_mask = _crop_tile_for_texture(front_jpg, 1000, 2000)
    b_direct_raw, b_mask_raw = _crop_tile_for_texture(back_jpg, 1000, 2000)
    b_direct = cv2.flip(b_direct_raw, 1)
    b_mask = np.fliplr(b_mask_raw)
    atlas = np.zeros((2000, 2000, 3), dtype=np.uint8)
    atlas[0:2000, 0:1000] = f_direct
    atlas[0:2000, 1000:2000] = b_direct

    # Feather blend around the front/back split to reduce a hard seam at U=0.5.
    seam_band = 64
    for i in range(seam_band):
        lf = f_direct[:, 1000 - seam_band + i].astype(np.float32)
        rb = b_direct[:, i].astype(np.float32)
        # Prefer whichever side has stronger foreground support near the seam.
        lf_m = f_mask[:, 1000 - seam_band + i].astype(np.float32)
        rb_m = b_mask[:, i].astype(np.float32)
        t = float(i + 1) / float(seam_band + 1)
        t = np.where((lf_m + rb_m) > 0.0, (rb_m + t) / (lf_m + rb_m + 1.0), t).astype(np.float32)[:, None]
        mix = np.clip((1.0 - t) * lf + t * rb, 0.0, 255.0).astype(np.uint8)
        atlas[:, 1000 - seam_band + i] = mix
        atlas[:, 1000 + i] = mix
    return atlas


def _resolve_pix2surf_assets(pix2surf_root: Path) -> Pix2SurfAssets:
    pretrained = pix2surf_root / "pretrained"
    seg_shirts_front = pretrained / "seg_shirts_front.pt"
    seg_shirts_back = pretrained / "seg_shirts_back.pt"
    map_shirts_front = pretrained / "map_shirts_front.pt"
    map_shirts_back = pretrained / "map_shirts_back.pt"

    for p, name in [
        (seg_shirts_front, "seg_shirts_front.pt"),
        (seg_shirts_back, "seg_shirts_back.pt"),
        (map_shirts_front, "map_shirts_front.pt"),
        (map_shirts_back, "map_shirts_back.pt"),
    ]:
        _ensure_exists(p, name)

    return Pix2SurfAssets(
        seg_shirts_front=seg_shirts_front,
        seg_shirts_back=seg_shirts_back,
        map_shirts_front=map_shirts_front,
        map_shirts_back=map_shirts_back,
    )


def _infer_upper_texture_with_pix2surf(
    *,
    pix2surf_root: Path,
    assets: Optional[Pix2SurfAssets],
    front_jpg: Path,
    back_jpg: Path,
    device: str,
    texture_mode: str,
    out_tex_jpg: Path,
) -> None:
    if texture_mode not in {"pix2surf", "hybrid", "direct"}:
        _die(f"Unsupported texture_mode '{texture_mode}' (expected pix2surf, hybrid, or direct)")
    if cv2 is None:
        _die("OpenCV not available. Install opencv-python.")

    def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """
        SSIM on RGB images (higher is better).
        Uses skimage if available, else falls back to a global SSIM approximation.
        """
        try:
            from skimage.metrics import structural_similarity as _ssim  # type: ignore
            return float(_ssim(a, b, channel_axis=-1, data_range=255))
        except Exception:
            pass

        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        mu_a = a_f.mean()
        mu_b = b_f.mean()
        var_a = a_f.var()
        var_b = b_f.var()
        cov = ((a_f - mu_a) * (b_f - mu_b)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        if den <= 1e-12:
            return None
        return float(num / den)

    def _lpips_rgb(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """
        LPIPS (lower is better). Requires torch + lpips.
        Computes on a downscaled copy for speed.
        """
        try:
            import torch as _torch  # type: ignore
            import lpips as _lpips  # type: ignore
        except Exception:
            return None

        try:
            side = 256
            aa = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
            bb = cv2.resize(b, (side, side), interpolation=cv2.INTER_AREA)

            def to_t(x: np.ndarray) -> Any:
                t = _torch.from_numpy(x.astype(np.float32) / 255.0)  # HWC 0..1
                t = t.permute(2, 0, 1).unsqueeze(0)  # 1CHW
                return (t * 2.0) - 1.0  # [-1, 1]

            ta = to_t(aa)
            tb = to_t(bb)
            dev = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
            ta = ta.to(dev)
            tb = tb.to(dev)
            net = _lpips.LPIPS(net="alex").to(dev)
            with _torch.no_grad():
                v = net(ta, tb)
            return float(v.item())
        except Exception:
            return None

    def _write_metrics(workdir: Path, metrics: dict) -> None:
        try:
            p = workdir / "metrics.json"
            p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except Exception:
            pass

    if texture_mode == "direct":
        print("[INFO] Building garment texture (mode=direct)…", flush=True)
        base = _build_direct_texture_atlas(front_jpg, back_jpg)
        print(f"  Direct atlas brightness: mean={float(base.mean()):.1f}/255")
        out_tex_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_tex_jpg), cv2.cvtColor(base, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[OK] Wrote upper texture: {out_tex_jpg}")

        metrics = {"ssim": 1.0, "lpips": 0.0, "mode": "direct", "note": "direct atlas (reference==output)"}
        print("SSIM ↑LPIPS ↓", flush=True)
        print(f"[METRICS] SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}", flush=True)
        _write_metrics(out_tex_jpg.parent, metrics)
        return
    if assets is None:
        _die("Pix2Surf assets are required for texture_mode pix2surf/hybrid.")
    _ensure_torch()

    sys.path.insert(0, str(pix2surf_root))
    try:
        from train import network
        from train.utils import binarizeimage, tensor2image
    except Exception as e:
        _die(f"Could not import pix2surf modules from {pix2surf_root}: {e}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    def read_img(p: Path) -> Any:
        img = transform(Image.open(p).convert("RGB"))
        return img.unsqueeze(0).to(device)

    print("[INFO] Pix2Surf: loading weights and running 4 neural passes (GPU recommended)…", flush=True)

    def get_img_rep(seg_out: Any) -> Any:
        m = torch.nn.Softmax2d()
        out = m(seg_out)
        out = out.squeeze(0)[1, :, :]
        out_fg_binary = binarizeimage(out)

        x = torch.from_numpy(np.linspace(-1, 1, 256))
        y = torch.from_numpy(np.linspace(-1, 1, 256))
        xx = x.view(-1, 1).repeat(1, 256)
        yy = y.repeat(256, 1)
        meshed = torch.cat([yy.unsqueeze_(2), xx.unsqueeze_(2)], 2)
        meshed = meshed.permute(2, 0, 1)

        out_fg_binary = out_fg_binary.unsqueeze(0)
        mask2 = torch.cat((out_fg_binary, out_fg_binary), dim=0)
        rend_rep = mask2.float() * meshed.float()
        return rend_rep.unsqueeze(0).to(device)

    net_map = nn.Module()  # Placeholder
    net_seg = nn.Module()  # Placeholder

    try:
        from train import network as net_module
        net_map = net_module.ResnetGenerator(2, 2, 0, 64, n_blocks=6, norm_layer=nn.InstanceNorm2d)
        net_seg = net_module.UnetGenerator(input_nc=3, output_nc=2, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d)
    except Exception as e:
        _die(f"Could not create Pix2Surf networks: {e}")

    def run_one(img_path: Path, seg_pth: Path, map_pth: Path) -> np.ndarray:
        try:
            net_seg.load_state_dict(torch.load(seg_pth, map_location=device, weights_only=True))
            net_map.load_state_dict(torch.load(map_pth, map_location=device, weights_only=True))
        except TypeError:
            net_seg.load_state_dict(torch.load(seg_pth, map_location=device))
            net_map.load_state_dict(torch.load(map_pth, map_location=device))

        net_seg.to(device).eval()
        net_map.to(device).eval()

        image = read_img(img_path)
        seg_out = net_seg(image)
        map_in = get_img_rep(seg_out)
        grid = net_map(map_in)
        grid = grid.permute(0, 2, 3, 1)
        uv_out = F.grid_sample(image, grid, align_corners=False)
        return tensor2image(uv_out[0, :, :, :])

    print("[INFO] Pix2Surf: front image…", flush=True)
    up_front = run_one(front_jpg, assets.seg_shirts_front, assets.map_shirts_front)
    print("[INFO] Pix2Surf: back image…", flush=True)
    up_back = run_one(back_jpg, assets.seg_shirts_back, assets.map_shirts_back)

    # tensor2image returns BGR (cv2/torch convention) — convert to RGB immediately.
    cut1 = cv2.cvtColor(cv2.resize(up_front, (1000, 2000)), cv2.COLOR_BGR2RGB)
    cut2 = cv2.cvtColor(cv2.resize(up_back,  (1000, 2000)), cv2.COLOR_BGR2RGB)
    cut2 = cv2.flip(cut2, 1)

    p2s_base = np.zeros((2000, 2000, 3), dtype=np.uint8)
    p2s_base[0:2000, 0:1000] = cut1
    p2s_base[0:2000, 1000:2000] = cut2

    mean_brightness = float(p2s_base.mean())
    print(f"  Texture brightness (Pix2Surf): mean={mean_brightness:.1f}/255")
    if texture_mode == "pix2surf":
        base = p2s_base
        print(f"  Using pix2surf output (mode=pix2surf, brightness={mean_brightness:.0f}/255)")
    else:
        direct = _build_direct_texture_atlas(front_jpg, back_jpg)
        direct_mean = float(direct.mean())
        # Hybrid mode prioritizes direct-photo detail (logos/text) and only keeps a
        # light amount of Pix2Surf warping for plausible UV correspondence.
        w_p2s = max(0.0, min(0.35, (mean_brightness - 40.0) / 140.0))
        base = np.clip(
            p2s_base.astype(np.float32) * w_p2s + direct.astype(np.float32) * (1.0 - w_p2s),
            0,
            255,
        ).astype(np.uint8)
        print(
            f"  Hybrid blend: p2s_w={w_p2s:.2f} direct_mean={direct_mean:.0f} final_mean={base.mean():.1f}/255"
        )

    # Save as JPEG — Blender embeds it directly, correct colours guaranteed
    out_tex_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_tex_jpg), cv2.cvtColor(base, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Wrote upper texture: {out_tex_jpg}")

    # Metrics: compare direct-photo atlas vs Pix2Surf atlas (diagnostic)
    try:
        direct_ref = _build_direct_texture_atlas(front_jpg, back_jpg)
        ssim_v = _ssim_rgb(direct_ref, p2s_base)
        lpips_v = _lpips_rgb(direct_ref, p2s_base)
        metrics = {"ssim": ssim_v, "lpips": lpips_v, "mode": str(texture_mode), "note": "direct atlas vs pix2surf atlas"}
        print("SSIM ↑LPIPS ↓", flush=True)
        ssim_s = f"{ssim_v:.4f}" if isinstance(ssim_v, (int, float)) else "n/a"
        lpips_s = f"{lpips_v:.4f}" if isinstance(lpips_v, (int, float)) else "n/a"
        print(f"[METRICS] SSIM={ssim_s}  LPIPS={lpips_s}", flush=True)
        _write_metrics(out_tex_jpg.parent, metrics)
    except Exception:
        pass


# -------------------------
# OBJ loader
# -------------------------
def _load_obj_vertices_uv_faces(obj_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    vs: list[list[float]] = []
    vts: list[list[float]] = []
    faces: list[list[int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt" and len(parts) >= 3:
                vts.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for p in parts[1:]:
                    a = p.split("/")[0]
                    if a:
                        idx.append(int(a) - 1)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])

    V = np.asarray(vs, dtype=np.float32)
    VT = np.asarray(vts, dtype=np.float32) if vts else None
    F = np.asarray(faces, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        _die(f"OBJ appears empty or unsupported: {obj_path}")
    return V, VT, F


def _tile_uv_params() -> Tuple[float, float, float]:
    """Shoulder/hem tile fractions (from top) and horizontal margin — must match _align_garment_to_tile."""
    try:
        sh = float(os.environ.get("GARMENT_TILE_SHOULDER_FRAC", "0.12") or "0.12")
    except ValueError:
        sh = 0.12
    try:
        hem = float(os.environ.get("GARMENT_TILE_HEM_FRAC", "0.92") or "0.92")
    except ValueError:
        hem = 0.92
    return sh, hem, 0.06


def _mesh_torso_y_extent(V: np.ndarray) -> Tuple[float, float]:
    """Robust shoulder (high Y) and hem (low Y) on the garment mesh."""
    y = np.asarray(V[:, 1], dtype=np.float64)
    y_sh = float(np.percentile(y, 93))
    y_hem = float(np.percentile(y, 7))
    if y_sh - y_hem < 1e-5:
        y_sh, y_hem = float(y.max()), float(y.min())
    return y_sh, y_hem


def _use_rigid_export() -> bool:
    """Export a static Mesh GLB (SAM3-style) instead of SMPL skinning."""
    v = os.environ.get("GARMENT_EXPORT_RIGID", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _resolve_canonical_glb_path(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        p = explicit.resolve()
        if p.is_file():
            return p
    env = os.environ.get("GARMENT_CANONICAL_GLB", "").strip()
    if not env:
        return None
    p = Path(env)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p if p.is_file() else None


def _load_mesh_from_glb(glb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if trimesh is None:
        _die("trimesh is required to load GARMENT_CANONICAL_GLB")
    loaded = trimesh.load(str(glb_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            _die(f"No mesh geometry in GLB: {glb_path}")
        loaded = trimesh.util.concatenate(parts)
    V = np.asarray(loaded.vertices, dtype=np.float32)
    F = np.asarray(loaded.faces, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        _die(f"Empty mesh in GLB: {glb_path}")
    return V, F


def _geometric_torso_arm_mask(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Torso vs sleeve split without skin weights (canonical SAM3 mesh)."""
    x = np.asarray(V[:, 0], dtype=np.float64)
    y = np.asarray(V[:, 1], dtype=np.float64)
    cx = float(np.median(x))
    x_thr = float(np.percentile(np.abs(x - cx), 42))
    x_thr = max(x_thr, 1e-4)
    y_lo = float(np.percentile(y, 12))
    y_hi = float(np.percentile(y, 88))
    is_torso = (np.abs(x - cx) <= x_thr) & (y >= y_lo) & (y <= y_hi)
    is_arm = (np.abs(x - cx) > x_thr * 0.85) & ~is_torso
    if int(is_torso.sum()) < 32:
        is_torso = np.ones(V.shape[0], dtype=bool)
        is_arm = np.zeros(V.shape[0], dtype=bool)
    return is_torso, is_arm


def _export_garment_json_rigid(garment_V: np.ndarray, out_json: Path, scale: float = 1.0) -> None:
    """Metadata for rigid try-on (bbox calibration, no SMPL bones)."""
    g_min = garment_V.min(axis=0) * scale
    g_max = garment_V.max(axis=0) * scale
    g_size = g_max - g_min
    y_sh = float(np.percentile(garment_V[:, 1], 93))
    y_hem = float(np.percentile(garment_V[:, 1], 7))
    band = garment_V[(garment_V[:, 1] >= y_hem) & (garment_V[:, 1] <= y_sh)]
    if band.shape[0] >= 8:
        shoulder_w = float(band[:, 0].max() - band[:, 0].min()) * scale
    else:
        shoulder_w = float(g_size[0])
    torso_h = max(y_sh - y_hem, 1e-6) * scale
    metadata = {
        "version": "1.0",
        "export_mode": "rigid",
        "garment_type": "tshirt",
        "garment_bounds": {
            "min": g_min.tolist(),
            "max": g_max.tolist(),
            "size": g_size.tolist(),
        },
        "calibration": {
            "mesh_shoulder_width": shoulder_w,
            "mesh_torso_height": torso_h,
            "smpl_shoulder_width": shoulder_w,
            "smpl_torso_height": torso_h,
        },
        "mediapipe_mapping": {},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Wrote rigid garment metadata: {out_json}", flush=True)


def _garment_torso_arm_mask(Wg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Classify vertices as torso vs arm from transferred SMPL weights."""
    torso_j = [0, 3, 6, 9, 12, 13, 14, 15]
    arm_j = [16, 17, 18, 19, 20, 21, 22, 23]
    j = min(Wg.shape[1], 24)
    tw = Wg[:, :j][:, [i for i in torso_j if i < j]].sum(axis=1)
    aw = Wg[:, :j][:, [i for i in arm_j if i < j]].sum(axis=1)
    is_torso = tw >= np.maximum(aw, 1e-6)
    is_arm = (aw > tw * 1.08) & (aw > 0.12)
    return is_torso, is_arm


def _compute_garment_front_back_uv(
    V: np.ndarray,
    F: np.ndarray,
    Wg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Front/back planar UVs (front U 0..0.5, back U 0.5..1).

    Chest logos require the front photo centre to land on torso vertices, not sleeves.
    When skin weights are available, UV bbox and U placement use torso verts; arm verts
    are pushed to the lateral edges of the front/back tile (sleeve regions in flat photos).
    """
    Vd = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    side_margin = 0.06 if not _use_texture_full_bleed() else 0.02

    vn = np.zeros((Vd.shape[0], 3), dtype=np.float64)
    v0, v1, v2 = Vd[F[:, 0]], Vd[F[:, 1]], Vd[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)

    front = vn[:, 2] > 0.2
    back = vn[:, 2] < -0.2
    # Ambiguous side normals: treat as front only if near the torso column.
    side = ~(front | back)

    if Wg is not None and Wg.shape[0] == Vd.shape[0]:
        is_torso, is_arm = _garment_torso_arm_mask(Wg)
    else:
        is_torso, is_arm = _geometric_torso_arm_mask(Vd)

    ref_torso = Vd[is_torso]
    if ref_torso.shape[0] < 8:
        ref_torso = Vd
    min_x = float(ref_torso[:, 0].min())
    max_x = float(ref_torso[:, 0].max())
    min_y = float(ref_torso[:, 1].min())
    max_y = float(ref_torso[:, 1].max())
    cx = float(ref_torso[:, 0].mean())
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    torso_half_w = max(dx * 0.5, 1e-6)

    u01 = np.zeros(Vd.shape[0], dtype=np.float64)
    torso_u = np.clip((Vd[:, 0] - min_x) / dx, 0.0, 1.0)
    torso_u = side_margin + torso_u * (1.0 - 2.0 * side_margin)
    u01[is_torso] = torso_u[is_torso]

    arm_front = is_arm & (front | side)
    if np.any(arm_front):
        rel = (Vd[:, 0] - cx) / torso_half_w
        rel = np.clip(rel, -1.8, 1.8)
        # Centre of front tile (u01=0.5) = chest; arms sample left/right sleeve areas.
        u_arm = 0.5 + rel * 0.38
        u01[arm_front] = np.clip(u_arm[arm_front], side_margin, 1.0 - side_margin)

    if _use_texture_full_bleed():
        v_coord = np.clip((Vd[:, 1] - min_y) / dy, 0.0, 1.0).astype(np.float64)
    else:
        shoulder_frac, hem_frac, _ = _tile_uv_params()
        v_shoulder = 1.0 - shoulder_frac
        v_hem = 1.0 - hem_frac
        y_sh, y_hem = _mesh_torso_y_extent(ref_torso)
        span_y = max(y_sh - y_hem, 1e-6)
        v01 = np.clip((Vd[:, 1] - y_hem) / span_y, 0.0, 1.0)
        v_coord = (v01 * (v_shoulder - v_hem) + v_hem).astype(np.float64)

    # Side panels that are not arms: keep torso U so prints stay on the chest column.
    side_torso = side & is_torso
    if np.any(side_torso):
        u01[side_torso] = torso_u[side_torso]

    uv = np.zeros((Vd.shape[0], 2), dtype=np.float32)
    use_front = front | side_torso | (side & is_arm & (np.abs(Vd[:, 0] - cx) < torso_half_w * 1.1))
    use_back = back | (side & ~use_front)

    uv[use_front, 0] = (u01[use_front] * 0.5).astype(np.float32)
    uv[use_front, 1] = v_coord[use_front].astype(np.float32)
    uv[use_back, 0] = (0.5 + (1.0 - u01[use_back]) * 0.5).astype(np.float32)
    uv[use_back, 1] = v_coord[use_back].astype(np.float32)
    return uv


def _write_obj_triangles(
    path: Path,
    V: np.ndarray,
    F: np.ndarray,
    uv: Optional[np.ndarray] = None,
) -> None:
    """Triangle OBJ (1-based). Optional per-vertex vt so Blender keeps atlas UVs on posed meshes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# garment export\n")
        for i in range(V.shape[0]):
            x, y, z = float(V[i, 0]), float(V[i, 1]), float(V[i, 2])
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        if uv is not None:
            for i in range(uv.shape[0]):
                f.write(f"vt {float(uv[i, 0]):.8f} {float(uv[i, 1]):.8f}\n")
        for i in range(F.shape[0]):
            a, b, c = int(F[i, 0]) + 1, int(F[i, 1]) + 1, int(F[i, 2]) + 1
            if uv is not None:
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")


def _subdivide_garment_mesh(
    garment_V: np.ndarray,
    garment_F: np.ndarray,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loop subdivision (trimesh) before weight transfer — more vertices / triangles for
    smoother skinning and closer match to SMPL near shoulders and sleeves.
    """
    if iterations <= 0:
        return garment_V, garment_F
    if trimesh is None:
        print("[WARN] trimesh missing — skip subdivision.", flush=True)
        return garment_V, garment_F
    try:
        m = trimesh.Trimesh(vertices=np.asarray(garment_V, dtype=np.float64),
                            faces=np.asarray(garment_F, dtype=np.int64),
                            process=False)
        for it in range(iterations):
            m = m.subdivide()
            print(
                f"  [subdivide] iter {it + 1}/{iterations}: V={len(m.vertices)}, F={len(m.faces)}",
                flush=True,
            )
        V = np.asarray(m.vertices, dtype=np.float32)
        F = np.asarray(m.faces, dtype=np.int32)
        return V, F
    except Exception as e:
        print(f"[WARN] Subdivision failed ({e}) — using base mesh.", flush=True)
        return garment_V, garment_F


# -------------------------
# SMPL model loader
# -------------------------
@dataclass(frozen=True)
class SMPLModel:
    v_template: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    J_regressor: np.ndarray
    kintree_table: np.ndarray
    joints: np.ndarray
    joint_names: list[str]


def _to_dense_J_regressor(J_reg: Any) -> np.ndarray:
    # Handle the _SparseMat stub (created when scipy was unavailable at
    # pickle-load time but IS now available for the actual conversion).
    if hasattr(J_reg, "toarray"):
        return np.asarray(J_reg.toarray(), dtype=np.float32)
    try:
        import scipy.sparse
        if scipy.sparse.issparse(J_reg):
            return J_reg.toarray().astype(np.float32)
    except Exception:
        pass
    return np.asarray(J_reg, dtype=np.float32)


def _load_smpl_pkl(pkl_path: Path) -> SMPLModel:
    _ensure_exists(pkl_path, "SMPL model .pkl")

    # scipy.sparse stub so the pkl loads even when scipy is not installed.
    # _to_dense_J_regressor() will convert the sparse matrix to ndarray later.
    # Detect whether scipy is importable right now.
    try:
        import scipy.sparse as _sp
        _scipy_available = True
    except ImportError:
        _sp = None
        _scipy_available = False

    class _SparseMat:
        """Stub used ONLY when scipy is absent at pickle time."""
        def __init__(self, *a, **kw):
            self._args = a
        def toarray(self):
            raise RuntimeError(
                "scipy is required to load the SMPL model.\n"
                "Run:  pip install scipy trimesh\n"
                "Then restart the server."
            )

    _SPARSE_NAMES = frozenset({
        "csr_matrix", "csc_matrix", "coo_matrix",
        "bsr_matrix", "dia_matrix", "dok_matrix", "lil_matrix",
        "csr_array", "csc_array", "coo_array",
    })

    class _ChumpyStubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return type(name, (), {})
            # When scipy IS present, let pickle reconstruct the real
            # sparse matrix object (handles all pickle protocols correctly).
            # Only fall back to the stub when scipy is genuinely absent.
            if module.startswith("scipy.sparse") and name in _SPARSE_NAMES:
                if _scipy_available:
                    return super().find_class(module, name)
                return _SparseMat
            return super().find_class(module, name)

    with pkl_path.open("rb") as f:
        data = _ChumpyStubUnpickler(f, encoding="latin1", errors="ignore").load()

    required = ["v_template", "f", "weights", "J_regressor", "kintree_table"]
    for k in required:
        if k not in data:
            _die(f"SMPL pkl missing key '{k}'. Found keys: {list(data.keys())[:50]}")

    v_template = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    weights = np.asarray(data["weights"], dtype=np.float32)
    J_regressor = _to_dense_J_regressor(data["J_regressor"])
    kintree_table = np.asarray(data["kintree_table"], dtype=np.int64)
    joints = (J_regressor @ v_template).astype(np.float32)

    J = joints.shape[0]
    joint_names = _SMPL_24_NAMES if J == 24 else [f"joint_{i:02d}" for i in range(J)]

    return SMPLModel(
        v_template=v_template,
        faces=faces,
        weights=weights,
        J_regressor=J_regressor,
        kintree_table=kintree_table,
        joints=joints,
        joint_names=joint_names,
    )


# -------------------------
# Weight transfer
# -------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_region_kdtrees(smpl: SMPLModel) -> dict[str, Any]:
    if cKDTree is None:
        return {}

    J = smpl.weights.shape[1]
    if J != 24:
        return {}

    left_arm_j = [13, 16, 18, 20, 22]
    right_arm_j = [14, 17, 19, 21, 23]
    torso_j = [0, 3, 6, 9, 12, 15]

    W = smpl.weights
    left_arm_score = W[:, left_arm_j].sum(axis=1)
    right_arm_score = W[:, right_arm_j].sum(axis=1)
    torso_score = W[:, torso_j].sum(axis=1)

    arm_thr = 0.55
    torso_thr = 0.60

    left_arm_idx = np.where(left_arm_score > arm_thr)[0]
    right_arm_idx = np.where(right_arm_score > arm_thr)[0]
    torso_idx = np.where(torso_score > torso_thr)[0]

    if left_arm_idx.size < 200:
        left_arm_idx = np.where(left_arm_score > 0.40)[0]
    if right_arm_idx.size < 200:
        right_arm_idx = np.where(right_arm_score > 0.40)[0]
    if torso_idx.size < 500:
        torso_idx = np.where(torso_score > 0.45)[0]

    out: dict[str, Any] = {}

    def make_tree(name: str, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        V = smpl.v_template[idx]
        out[name] = {
            "idx": idx,
            "V": V,
            "W": smpl.weights[idx],
            "tree": cKDTree(V),
        }

    make_tree("left_arm", left_arm_idx)
    make_tree("right_arm", right_arm_idx)
    make_tree("torso", torso_idx)

    out["left_arm_j"] = left_arm_j
    out["right_arm_j"] = right_arm_j
    out["torso_j"] = torso_j
    return out


def _closest_point_on_triangle(P: np.ndarray,
                               A: np.ndarray,
                               B: np.ndarray,
                               C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised closest-point-on-triangle for N query points against N triangles.
    Ericson "Real-Time Collision Detection" §5.1.5 — fully vectorised.

    All inputs shape (N, 3) float64.
    Returns:
        closest : (N, 3)  closest point on triangle surface
        uvw     : (N, 3)  barycentric (u=wA, v=wB, w=wC), >=0, sum=1
    """
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = np.einsum("ij,ij->i", AB, AP)   # dot(AB, AP)
    d2 = np.einsum("ij,ij->i", AC, AP)   # dot(AC, AP)

    BP = P - B
    d3 = np.einsum("ij,ij->i", AB, BP)   # dot(AB, BP)
    d4 = np.einsum("ij,ij->i", AC, BP)   # dot(AC, BP)

    CP = P - C
    d5 = np.einsum("ij,ij->i", AB, CP)   # dot(AB, CP)
    d6 = np.einsum("ij,ij->i", AC, CP)   # dot(AC, CP)

    # Voronoi region classifiers
    vc = d1 * d4 - d3 * d2   # "signed area" numerators
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    mask_A  = (d1 <= 0.0) & (d2 <= 0.0)
    mask_B  = (d3 >= 0.0) & (d4 <= d3)
    mask_C  = (d6 >= 0.0) & (d5 <= d6)
    mask_AB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    mask_AC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    mask_BC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    N  = P.shape[0]
    uu = np.zeros(N, dtype=np.float64)
    vv = np.zeros(N, dtype=np.float64)
    ww = np.zeros(N, dtype=np.float64)

    # ---- Vertex regions ------------------------------------------------
    uu[mask_A] = 1.0
    vv[mask_B] = 1.0
    ww[mask_C] = 1.0

    # ---- Edge AB  (t = d1 / (d1 - d3)) --------------------------------
    dAB = (d1 - d3)
    dAB_safe = np.where(np.abs(dAB) < 1e-12, 1.0, dAB)
    t_AB = np.clip(d1 / dAB_safe, 0.0, 1.0)
    uu[mask_AB] = 1.0 - t_AB[mask_AB]
    vv[mask_AB] = t_AB[mask_AB]

    # ---- Edge AC  (t = d2 / (d2 - d6)) --------------------------------
    dAC = (d2 - d6)
    dAC_safe = np.where(np.abs(dAC) < 1e-12, 1.0, dAC)
    t_AC = np.clip(d2 / dAC_safe, 0.0, 1.0)
    uu[mask_AC] = 1.0 - t_AC[mask_AC]
    ww[mask_AC] = t_AC[mask_AC]

    # ---- Edge BC  (t = (d4-d3) / ((d4-d3) + (d5-d6))) ----------------
    snom = d4 - d3
    sdenom = d5 - d6
    dBC = snom + sdenom
    dBC_safe = np.where(np.abs(dBC) < 1e-12, 1.0, dBC)
    t_BC = np.clip(snom / dBC_safe, 0.0, 1.0)
    vv[mask_BC] = 1.0 - t_BC[mask_BC]
    ww[mask_BC] = t_BC[mask_BC]

    # ---- Interior ------------------------------------------------------
    interior = ~(mask_A | mask_B | mask_C | mask_AB | mask_AC | mask_BC)
    dint = va + vb + vc
    dint_safe = np.where(np.abs(dint) < 1e-12, 1.0, dint)
    vv[interior] = vb[interior] / dint_safe[interior]
    ww[interior] = vc[interior] / dint_safe[interior]
    uu[interior] = 1.0 - vv[interior] - ww[interior]

    uvw = np.stack([uu, vv, ww], axis=1).astype(np.float32)
    uvw = np.maximum(uvw, 0.0)
    rs  = uvw.sum(axis=1, keepdims=True)
    rs  = np.where(rs < 1e-12, 1.0, rs)
    uvw /= rs

    closest = A * uvw[:, 0:1] + B * uvw[:, 1:2] + C * uvw[:, 2:3]
    return closest, uvw


def _closest_surface_weights(smpl: SMPLModel, garment_V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each garment vertex find the closest point on the SMPL body surface and
    return the interpolated skinning weights at that point.

    RTREE-FREE IMPLEMENTATION — uses only scipy.spatial.cKDTree:
      1. Build a KD-tree on SMPL vertices.
      2. For each garment vertex, query the K=8 nearest SMPL vertices.
      3. Collect all faces that share any of those K vertices (~50-150 candidates).
      4. Evaluate the exact closest point on each candidate triangle.
      5. Keep the nearest result and compute barycentric weights.

    This replicates trimesh.proximity.closest_point without requiring the
    'rtree' or 'pyembree' packages that trimesh uses internally.
    """
    if cKDTree is None:
        _die("scipy is required for weight transfer. Run: pip install scipy")

    V = smpl.v_template.astype(np.float64)
    F = smpl.faces
    Gv = garment_V.astype(np.float64)
    Ng = Gv.shape[0]

    print(f"  Building KD-tree on {V.shape[0]} SMPL vertices...", flush=True)
    vtx_tree = cKDTree(V)

    # K nearest SMPL vertices per garment vertex
    K = 8
    _, nn_idx = vtx_tree.query(Gv, k=K)   # (Ng, K)

    # Build vertex → incident faces index
    print(f"  Building vertex→face adjacency...", flush=True)
    vtx_to_faces: list = [[] for _ in range(V.shape[0])]
    for fi, (a, b, c) in enumerate(F):
        vtx_to_faces[a].append(fi)
        vtx_to_faces[b].append(fi)
        vtx_to_faces[c].append(fi)

    print(f"  Computing closest surface point for {Ng} garment vertices...", flush=True)

    best_dist2  = np.full(Ng, np.inf, dtype=np.float64)
    best_face   = np.zeros(Ng, dtype=np.int32)
    best_uvw    = np.zeros((Ng, 3), dtype=np.float32)

    # Process in batches so we don't blow out memory
    BATCH = 512
    for start in range(0, Ng, BATCH):
        end = min(start + BATCH, Ng)
        if start == 0 or end == Ng or (start // BATCH) % 8 == 0:
            pct = min(99, int(100.0 * end / max(Ng, 1)))
            print(f"  Weight transfer closest-point: {end}/{Ng} verts ({pct}%)", flush=True)
        P_batch = Gv[start:end]                 # (B, 3)
        nn_batch = nn_idx[start:end]             # (B, K)
        B_size = end - start

        # Collect unique candidate faces for this batch
        cand_set = set()
        for i in range(B_size):
            for vi in nn_batch[i]:
                cand_set.update(vtx_to_faces[vi])
        cand_faces = np.array(sorted(cand_set), dtype=np.int32)

        if cand_faces.size == 0:
            continue

        # Triangle vertices for all candidates  (Nc, 3)
        A = V[F[cand_faces, 0]]
        Bv = V[F[cand_faces, 1]]
        C = V[F[cand_faces, 2]]
        Nc = cand_faces.shape[0]

        for i in range(B_size):
            # Broadcast single query point against all candidate triangles
            P_rep = np.tile(P_batch[i], (Nc, 1))   # (Nc, 3)
            cp, uvw = _closest_point_on_triangle(P_rep, A, Bv, C)
            d2 = np.einsum("ij,ij->i", cp - P_rep, cp - P_rep)
            best_j = int(np.argmin(d2))
            gi = start + i
            if d2[best_j] < best_dist2[gi]:
                best_dist2[gi] = d2[best_j]
                best_face[gi]  = cand_faces[best_j]
                best_uvw[gi]   = uvw[best_j]

    distances = np.sqrt(np.maximum(best_dist2, 0.0)).astype(np.float32)

    # Interpolate skinning weights using barycentric coords
    tri = F[best_face]                    # (Ng, 3)
    Wa  = smpl.weights[tri[:, 0]]        # (Ng, J)
    Wb  = smpl.weights[tri[:, 1]]
    Wc  = smpl.weights[tri[:, 2]]
    W_surface = (Wa * best_uvw[:, 0:1]
               + Wb * best_uvw[:, 1:2]
               + Wc * best_uvw[:, 2:3])

    print(f"[OK] Surface weight transfer complete. "
          f"Mean dist={distances.mean():.4f}, Max={distances.max():.4f}")
    return W_surface.astype(np.float32), distances


def _transfer_weights_barycentric_fixed(smpl: SMPLModel, garment_V: np.ndarray) -> np.ndarray:
    if garment_V.ndim != 2 or garment_V.shape[1] != 3:
        _die("garment_V must be (N,3)")

    J = smpl.weights.shape[1]
    W_surface, d_surface = _closest_surface_weights(smpl, garment_V)

    kd = _build_region_kdtrees(smpl)
    if not kd:
        print("[WARN] Using surface transfer only.")
        Wg = W_surface
        Wg = np.maximum(Wg, 0.0)
        Wg /= np.maximum(Wg.sum(axis=1, keepdims=True), 1e-8)
        return Wg.astype(np.float32)

    x = garment_V[:, 0]
    y = garment_V[:, 1]
    x_abs = np.abs(x)

    x_q85 = np.quantile(x_abs, 0.85)
    y_q45 = np.quantile(y, 0.45)
    y_q65 = np.quantile(y, 0.65)

    alpha_side = _sigmoid((x_abs - x_q85) / max(x_q85 * 0.12, 1e-6))
    alpha_height = _sigmoid((y - y_q45) / max((y_q65 - y_q45) * 0.50, 1e-6))
    alpha_geom = (alpha_side * alpha_height).astype(np.float32)

    def query(tree_key: str, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        block = kd.get(tree_key, None)
        if block is None:
            return np.full((points.shape[0],), np.inf, dtype=np.float32), np.full((points.shape[0],), -1, dtype=np.int32)
        dist, nn = block["tree"].query(points, k=1)
        return dist.astype(np.float32), nn.astype(np.int32)

    d_torso, nn_torso = query("torso", garment_V)

    left_mask = x > 0
    right_mask = ~left_mask

    d_arm = np.full_like(d_torso, np.inf, dtype=np.float32)
    nn_arm = np.full_like(nn_torso, -1, dtype=np.int32)
    arm_block_key = np.empty((garment_V.shape[0],), dtype=object)

    if left_mask.any():
        dl, nrl = query("left_arm", garment_V[left_mask])
        d_arm[left_mask] = dl
        nn_arm[left_mask] = nrl
        arm_block_key[left_mask] = "left_arm"

    if right_mask.any():
        dr, nrr = query("right_arm", garment_V[right_mask])
        d_arm[right_mask] = dr
        nn_arm[right_mask] = nrr
        arm_block_key[right_mask] = "right_arm"

    prox = (d_torso - d_arm)
    alpha_prox = _sigmoid(prox / 0.015).astype(np.float32)

    alpha = np.clip(alpha_geom * alpha_prox, 0.0, 1.0)

    W_arm = np.zeros_like(W_surface, dtype=np.float32)
    for key in ("left_arm", "right_arm"):
        block = kd.get(key, None)
        if block is None:
            continue
        sel = (arm_block_key == key)
        if not np.any(sel):
            continue
        nn = nn_arm[sel]
        nn = np.clip(nn, 0, block["W"].shape[0] - 1)
        W_arm[sel] = block["W"][nn].astype(np.float32)

    Wg = (1.0 - alpha[:, None]) * W_surface + alpha[:, None] * W_arm

    Wg = np.maximum(Wg, 0.0)
    Wg /= np.maximum(Wg.sum(axis=1, keepdims=True), 1e-8)

    print(f"[OK] Weight transfer (fixed): alpha mean={alpha.mean():.3f}, alpha>0.5={(alpha>0.5).sum()} / {alpha.size}")
    return Wg.astype(np.float32)


def _smooth_weights_laplacian(faces: np.ndarray, W: np.ndarray, iters: int = 10, lam: float = 0.5) -> np.ndarray:
    """
    SEV-3 FIX: Vectorized Laplacian smoothing via scipy sparse matrix multiply.

    The original pure-Python O(V*J*iters) loop took 60-120 s on CPU with
    10k vertices and 10 iterations.  This version builds a row-stochastic
    sparse adjacency matrix once and applies it with scipy, reducing the
    same workload to under 1 s regardless of iteration count.

    Falls back to the original loop if scipy is unavailable.
    """
    try:
        from scipy.sparse import csr_matrix, diags as sp_diags
    except ImportError:
        print("[WARN] scipy.sparse not available — falling back to slow loop smoother")
        # --- original pure-Python fallback ---
        Vn = W.shape[0]
        adj = [[] for _ in range(Vn)]
        for (a, b, c) in faces:
            adj[a].extend([b, c])
            adj[b].extend([a, c])
            adj[c].extend([a, b])
        adj = [list(set(nbrs)) for nbrs in adj]
        Wn = W.copy()
        for _ in range(iters):
            Wnew = Wn.copy()
            for i in range(Vn):
                nbrs = adj[i]
                if not nbrs:
                    continue
                mean = Wn[nbrs].mean(axis=0)
                Wnew[i] = (1 - lam) * Wn[i] + lam * mean
            Wnew = np.maximum(Wnew, 0.0)
            s = Wnew.sum(axis=1, keepdims=True)
            s = np.where(s < 1e-8, 1.0, s)
            Wn = (Wnew / s).astype(np.float32)
        return Wn

    Vn = W.shape[0]

    # Build symmetric adjacency from triangle faces
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                           faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                           faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(Vn, Vn))

    # Row-normalise → row-stochastic matrix L
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    L = sp_diags(1.0 / deg) @ A

    # One-step blending matrix:  M = (1-lam)*I + lam*L
    I = sp_diags(np.ones(Vn, dtype=np.float32))
    M = (1.0 - lam) * I + lam * L

    Wn = W.astype(np.float32).copy()
    for _ in range(iters):
        Wn = (M @ Wn).toarray() if hasattr((M @ Wn), "toarray") else np.asarray(M @ Wn)
        Wn = np.maximum(Wn, 0.0)
        s = Wn.sum(axis=1, keepdims=True)
        s[s < 1e-8] = 1.0
        Wn /= s

    print(f"[OK] Smoothed garment weights (vectorized, {iters} iters).")
    return Wn.astype(np.float32)


# -------------------------
# Pose metadata export
# -------------------------
def _export_pose_metadata(
    smpl: SMPLModel,
    joint_positions: np.ndarray,
    garment_V: np.ndarray,
    out_json: Path,
    scale: float = 1.0
) -> None:
    """
    Export metadata JSON for the web frontend containing:
    - SMPL joint positions (bind pose)
    - Joint hierarchy (parent indices)
    - MediaPipe to SMPL mapping
    - Garment bounding box for scaling
    """
    
    # Calculate garment bounding box
    g_min = garment_V.min(axis=0) * scale
    g_max = garment_V.max(axis=0) * scale
    g_center = ((g_min + g_max) / 2).tolist()
    g_size = (g_max - g_min).tolist()
    
    # Joint positions scaled
    joints_scaled = (joint_positions * scale).tolist()
    
    # Parent indices
    kt = smpl.kintree_table
    parents = kt[0].astype(int).tolist()
    parents[0] = -1
    
    # Build MediaPipe mapping with SMPL joint indices
    mp_to_smpl = {}
    for mp_idx, smpl_name in MEDIAPIPE_TO_SMPL_MAPPING.items():
        if smpl_name is not None and smpl_name in SMPL_JOINT_INDICES:
            mp_to_smpl[str(mp_idx)] = {
                "smpl_joint": smpl_name,
                "smpl_index": SMPL_JOINT_INDICES[smpl_name]
            }
    
    # Upper body joint indices for t-shirt
    upper_body_indices = [SMPL_JOINT_INDICES[j] for j in UPPER_BODY_JOINTS if j in SMPL_JOINT_INDICES]
    
    metadata = {
        "version": "1.0",
        "garment_type": "tshirt",
        "armature": {
            "joint_names": smpl.joint_names,
            "joint_positions": joints_scaled,
            "parent_indices": parents,
            "num_joints": len(smpl.joint_names)
        },
        "garment_bounds": {
            "min": g_min.tolist(),
            "max": g_max.tolist(),
            "center": g_center,
            "size": g_size
        },
        "mediapipe_mapping": mp_to_smpl,
        "upper_body_joints": upper_body_indices,
        "coordinate_system": {
            "up": "Y",
            "forward": "-Z",
            "handedness": "right"
        },
        "calibration": {
            "smpl_shoulder_width": float(np.linalg.norm(
                joint_positions[SMPL_JOINT_INDICES["left_shoulder"]] -
                joint_positions[SMPL_JOINT_INDICES["right_shoulder"]]
            ) * scale),
            "smpl_torso_height": float(np.linalg.norm(
                joint_positions[SMPL_JOINT_INDICES["neck"]] -
                joint_positions[SMPL_JOINT_INDICES["pelvis"]]
            ) * scale)
        }
    }
    
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Wrote pose metadata: {out_json}")


# -------------------------
# Blender script generator
# -------------------------
def _write_blender_skin_export_script(dst: Path) -> None:
    script = r'''import bpy
import os
import sys
import argparse
import numpy as np

def extract_args(argv=None):
    if argv is None:
        argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1:]

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path, forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(filepath=path, axis_forward='-Z', axis_up='Y')
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    for m in meshes:
        n_uv = len(m.data.uv_layers)
        print(f"[UV] {m.name}: {n_uv} UV layer(s), {len(m.data.vertices)} verts")
        if n_uv == 0:
            # shirts.obj often has no vt UV lines — generate a simple front/back
            # projection atlas that matches the texture layout:
            # - front  -> left half  (U 0..0.5)
            # - back   -> right half (U 0.5..1.0), mirrored horizontally
            print("[UV] No UV data — generating front/back projection UVs (tile shoulder/hem aligned)")
            uv_layer = m.data.uv_layers.new(name="UVMap")
            mesh = m.data

            all_x = [v.co.x for v in mesh.vertices]
            all_y = [v.co.y for v in mesh.vertices]
            min_x, max_x = min(all_x), max(all_x)
            dx = max_x - min_x if max_x > min_x else 1.0
            y_sh = float(np.percentile(all_y, 93))
            y_hem = float(np.percentile(all_y, 7))
            if y_sh - y_hem < 1e-5:
                y_sh, y_hem = max(all_y), min(all_y)
            span_y = max(y_sh - y_hem, 1e-6)
            shoulder_frac = float(os.environ.get("GARMENT_TILE_SHOULDER_FRAC", "0.12") or "0.12")
            hem_frac = float(os.environ.get("GARMENT_TILE_HEM_FRAC", "0.92") or "0.92")
            v_shoulder = 1.0 - shoulder_frac
            v_hem = 1.0 - hem_frac
            side_margin = 0.06

            for poly in mesh.polygons:
                is_front = (poly.normal.z >= 0)
                for li in poly.loop_indices:
                    co = mesh.vertices[mesh.loops[li].vertex_index].co
                    u01 = (co.x - min_x) / dx
                    u01 = side_margin + u01 * (1.0 - 2.0 * side_margin)
                    v01 = (co.y - y_hem) / span_y
                    v01 = max(0.0, min(1.0, v01))
                    v_coord = v01 * (v_shoulder - v_hem) + v_hem
                    if is_front:
                        uv_layer.data[li].uv = (u01 * 0.5, v_coord)
                    else:
                        uv_layer.data[li].uv = (0.5 + (1.0 - u01) * 0.5, v_coord)
            mesh.update()
            print(f"[UV] Projection done — {len(m.data.uv_layers)} UV layer(s)")
    return meshes

def join_meshes(meshes):
    if not meshes:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def make_pbr_material(name, tex_path):
    # Match reference manu_tshirt_smpl_ready.glb: upper_mat — opaque PBR, base color
    # texture only, double-sided, metallic 0, roughness 0.5, no alpha clip.
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    try:
        mat.blend_method = "OPAQUE"
    except Exception:
        pass
    try:
        mat.use_backface_culling = False
    except Exception:
        pass

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.5
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.0
    elif "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = (0.0, 0.0, 0.0, 1.0)

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-400, 0)
    img = bpy.data.images.load(tex_path, check_existing=True)
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    tex.image = img
    try:
        img.pack()
    except Exception:
        pass

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def center_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

def create_armature(name, joints, parents, joint_names):
    arm_data = bpy.data.armatures.new(name + "_data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_obj.data.edit_bones
    bones = []

    for i, jn in enumerate(joint_names):
        b = edit_bones.new(jn)
        head = joints[i]

        child_idxs = np.where(parents == i)[0]
        if len(child_idxs) > 0:
            child_mean = joints[child_idxs].mean(axis=0)
            tail = head + 0.5 * (child_mean - head)
            if np.linalg.norm(tail - head) < 1e-4:
                tail = head + np.array([0.0, 0.05, 0.0], dtype=np.float32)
        else:
            tail = head + np.array([0.0, 0.05, 0.0], dtype=np.float32)

        b.head = head.tolist()
        b.tail = tail.tolist()
        bones.append(b)

    for i in range(len(joint_names)):
        p = int(parents[i])
        if p >= 0:
            bones[i].parent = bones[p]

    bpy.ops.armature.select_all(action='SELECT')
    try:
        bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Y')
    except Exception:
        pass

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def assign_vertex_group_weights(mesh_obj, joint_names, weights):
    vgs = {}
    for jn in joint_names:
        vgs[jn] = mesh_obj.vertex_groups.new(name=jn)

    J = weights.shape[1]
    for vi, w in enumerate(weights):
        for j in range(J):
            val = float(w[j])
            if val > 1e-5:
                vgs[joint_names[j]].add([vi], val, 'REPLACE')

def apply_scale(obj, s):
    obj.scale = (s, s, s)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def bind_to_armature(mesh_obj, arm_obj):
    mesh_obj.parent = arm_obj

    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj
    mod.use_deform_preserve_volume = True

def export_glb(out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Core glTF skinning: 4 influences per vertex (JOINTS_0 + WEIGHTS_0), like reference GLB
    base = dict(
        filepath=out_path,
        export_format="GLB",
        export_apply=False,
        export_skins=True,
        export_all_influences=False,
        use_active_scene=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    # Prefer embedded JPEG (reference ~2000², ~300KB in buffer)
    extended = dict(
        base,
        export_unused_images=False,
    )
    for k, v in (
        ("export_image_format", "JPEG"),
        ("export_jpeg_quality", 95),
    ):
        extended[k] = v

    try:
        bpy.ops.export_scene.gltf(**extended)
        return
    except TypeError as e:
        print(f"[WARN] GLTF extended options not all supported: {e}")

    try:
        bpy.ops.export_scene.gltf(**base)
    except TypeError as e:
        print(f"[WARN] GLTF export param error: {e} — retrying with minimal kwargs")
        safe = dict(
            filepath=out_path,
            export_format="GLB",
            export_apply=False,
            export_skins=True,
            use_active_scene=True,
            export_yup=True,
        )
        bpy.ops.export_scene.gltf(**safe)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment_obj", required=True)
    ap.add_argument("--garment_tex", required=True)
    ap.add_argument("--weights_npz", required=True)
    ap.add_argument("--out_glb", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--center", action="store_true")
    args = ap.parse_args(extract_args())

    reset_scene()

    npz = np.load(args.weights_npz, allow_pickle=True)
    joints = npz["joints"].astype(np.float32)
    parents = npz["parents"].astype(np.int32)
    joint_names = [str(x) for x in npz["joint_names"]]
    # Match reference GLB (e.g. manu_tshirt_smpl_ready.glb): collar bones drive the upper sleeve.
    _arm_name_remap = {"left_shoulder": "left_collar_1", "right_shoulder": "right_collar_1"}
    joint_names = [_arm_name_remap.get(n, n) for n in joint_names]
    weights = npz["garment_weights"].astype(np.float32)

    meshes = import_obj(args.garment_obj)
    if not meshes:
        raise RuntimeError("No mesh objects imported from OBJ")
    garment = join_meshes(meshes)
    garment.name = "upper_3"
    if garment.data:
        garment.data.name = "upper_3"

    arm = create_armature("SMPL_Armature", joints, parents, joint_names)

    apply_scale(garment, args.scale)
    apply_scale(arm, args.scale)

    mat = make_pbr_material("upper_mat", args.garment_tex)
    assign_material(garment, mat)

    assign_vertex_group_weights(garment, joint_names, weights)
    bind_to_armature(garment, arm)

    if args.center:
        center_object(garment)
        arm.location = (0.0, 0.0, 0.0)

    try:
        bpy.context.scene.name = "Scene"
    except Exception:
        pass

    export_glb(args.out_glb)
    print("[OK] Exported skinned GLB:", args.out_glb)

if __name__ == "__main__":
    main()
'''
    dst.write_text(script, encoding="utf-8")


def _write_blender_template_retexture_script(dst: Path) -> None:
    script = r'''import bpy
import os
import sys
import argparse

def extract_args(argv=None):
    if argv is None:
        argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1:]

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects imported from GLB template")
    return meshes

def make_pbr_material(name, tex_path):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    try:
        mat.blend_method = "OPAQUE"
    except Exception:
        pass
    try:
        mat.use_backface_culling = False
    except Exception:
        pass

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.5

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-400, 0)
    img = bpy.data.images.load(os.path.abspath(tex_path), check_existing=True)
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    tex.image = img
    try:
        img.pack()
    except Exception:
        pass

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def assign_material_to_meshes(meshes, mat):
    for m in meshes:
        if m.data.materials:
            m.data.materials[0] = mat
        else:
            m.data.materials.append(mat)

def apply_uniform_scale_all_objects(s):
    if abs(float(s) - 1.0) < 1e-8:
        return
    for obj in bpy.context.scene.objects:
        obj.scale = (s, s, s)
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.context.scene.objects[0] if bpy.context.scene.objects else None
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

def export_glb(out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    base = dict(
        filepath=out_path,
        export_format="GLB",
        export_apply=False,
        export_skins=True,
        export_all_influences=False,
        use_active_scene=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    extended = dict(base, export_unused_images=False, export_image_format="JPEG", export_jpeg_quality=95)
    try:
        bpy.ops.export_scene.gltf(**extended)
        return
    except TypeError:
        bpy.ops.export_scene.gltf(**base)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_glb", required=True)
    ap.add_argument("--garment_tex", required=True)
    ap.add_argument("--out_glb", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(extract_args())

    reset_scene()
    meshes = import_glb(args.template_glb)
    mat = make_pbr_material("upper_mat", args.garment_tex)
    assign_material_to_meshes(meshes, mat)
    apply_uniform_scale_all_objects(args.scale)

    export_glb(args.out_glb)
    print("[OK] Exported template-based GLB:", args.out_glb)

if __name__ == "__main__":
    main()
'''
    dst.write_text(script, encoding="utf-8")


def _write_blender_rigid_textured_export_script(dst: Path) -> None:
    """Static mesh + PBR texture (SAM3-style GLB for try-on)."""
    script = r'''import bpy
import os
import sys
import argparse

def extract_args(argv=None):
    if argv is None:
        argv = sys.argv
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1:]

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path, forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(filepath=path, axis_forward='-Z', axis_up='Y')
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    return meshes

def join_meshes(meshes):
    if not meshes:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def make_pbr_material(name, tex_path):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    try:
        mat.blend_method = "OPAQUE"
    except Exception:
        pass
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.5
    tex = nodes.new(type="ShaderNodeTexImage")
    img = bpy.data.images.load(tex_path, check_existing=True)
    try:
        img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    tex.image = img
    try:
        img.pack()
    except Exception:
        pass
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def center_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

def apply_scale(obj, s):
    obj.scale = (s, s, s)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def export_glb(out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # Draco compression is critical: the raw OBJ → GLB export of the SAM3
    # canonical mesh balloons from ~14 MB to ~60 MB because Blender deindexes
    # vertices at every UV seam. Draco brings it back down to ~3-6 MB and is
    # natively supported by Three.js (DRACOLoader registered on the client).
    kwargs = dict(
        filepath=out_path,
        export_format="GLB",
        export_apply=False,
        export_skins=False,
        use_active_scene=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=False,
        export_materials="EXPORT",
        export_image_format="JPEG",
        export_jpeg_quality=88,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=14,
        export_draco_normal_quantization=10,
        export_draco_texcoord_quantization=12,
    )
    fallback_keys = (
        "export_draco_mesh_compression_enable",
        "export_draco_mesh_compression_level",
        "export_draco_position_quantization",
        "export_draco_normal_quantization",
        "export_draco_texcoord_quantization",
        "export_image_format",
        "export_jpeg_quality",
    )
    try:
        bpy.ops.export_scene.gltf(**kwargs)
        return
    except TypeError:
        pass
    except Exception as e:
        print("[WARN] Draco GLB export failed, retrying without Draco:", e)
    for key in fallback_keys:
        kwargs.pop(key, None)
    try:
        bpy.ops.export_scene.gltf(**kwargs)
    except TypeError:
        kwargs.pop("export_image_format", None)
        kwargs.pop("export_jpeg_quality", None)
        bpy.ops.export_scene.gltf(**kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment_obj", required=True)
    ap.add_argument("--garment_tex", required=True)
    ap.add_argument("--out_glb", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--center", action="store_true")
    args = ap.parse_args(extract_args())

    reset_scene()
    meshes = import_obj(args.garment_obj)
    if not meshes:
        raise RuntimeError("No mesh imported")
    garment = join_meshes(meshes)
    garment.name = "shirt_canonical"
    apply_scale(garment, args.scale)
    mat = make_pbr_material("upper_mat", args.garment_tex)
    assign_material(garment, mat)
    if args.center:
        center_object(garment)
    export_glb(args.out_glb)
    print("[OK] Exported rigid textured GLB:", args.out_glb)

if __name__ == "__main__":
    main()
'''
    dst.write_text(script, encoding="utf-8")


def _blender_retexture_glb(
    blender_bin: Path,
    workdir: Path,
    template_glb: Path,
    garment_tex: Path,
    out_glb: Path,
    scale: float,
) -> None:
    """
    Import an existing GLB, swap the base-color texture, export GLB.
    Preserves topology (no OBJ round-trip that deindexes UVs to millions of verts).
    """
    blender_script = workdir / "blender_retexture.py"
    _write_blender_template_retexture_script(blender_script)
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_script.resolve()),
        "--",
        "--template_glb",
        str(template_glb.resolve()),
        "--garment_tex",
        str(garment_tex.resolve()),
        "--out_glb",
        str(out_glb.resolve()),
        "--scale",
        str(scale),
    ]
    _run(cmd)


def _blender_export_rigid_glb(
    blender_bin: Path,
    workdir: Path,
    garment_obj: Path,
    garment_tex: Path,
    out_glb: Path,
    scale: float,
    center: bool,
) -> None:
    blender_script = workdir / "blender_rigid_export.py"
    _write_blender_rigid_textured_export_script(blender_script)
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--garment_obj",
        str(garment_obj),
        "--garment_tex",
        str(garment_tex),
        "--out_glb",
        str(out_glb),
        "--scale",
        str(scale),
    ]
    if center:
        cmd.append("--center")
    _run(cmd)


# -------------------------
# Main
# -------------------------
def _check_dependencies(*, require_torch: bool = False, require_scipy: bool = True) -> None:
    """
    Fail fast with an actionable install command if key deps are missing.
    NOTE: 'rtree' is NO LONGER required — closest-point search is implemented
    using scipy.spatial.cKDTree so trimesh.proximity is not called.

    direct + SAM3 canonical rigid export does not need PyTorch or scipy — skipping
    those imports saves 1–4+ minutes of CPU startup time.
    """
    missing = []
    if require_scipy:
        try:
            import scipy
        except ImportError:
            missing.append("scipy")
    try:
        import trimesh
    except ImportError:
        missing.append("trimesh")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    if require_torch:
        try:
            import torch
        except ImportError:
            missing.append("torch torchvision")
    if missing:
        print("[ERROR] Missing required packages:", ", ".join(missing), file=sys.stderr)
        print("[ERROR] Run:  pip install", " ".join(missing), file=sys.stderr)
        sys.exit(1)
    if require_scipy:
        try:
            import trimesh as _tm
            _ = _tm.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
        except Exception as e:
            print(f"[WARN] trimesh sanity check failed: {e}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pix2surf_root", type=Path, default=None)
    ap.add_argument("--front_img", type=Path, default=None)
    ap.add_argument("--back_img", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--smpl_model", type=Path, required=True)
    ap.add_argument("--garment_obj", type=Path, required=True)
    ap.add_argument(
        "--garment_template_glb",
        type=Path,
        default=None,
        help="Optional pre-rigged garment GLB template. If set, keeps template rig/pose and retextures it.",
    )
    ap.add_argument(
        "--garment_canonical_glb",
        type=Path,
        default=None,
        help="SAM3-style canonical shirt GLB (rigid mesh). When set with GARMENT_EXPORT_RIGID=1, uses this shape.",
    )
    ap.add_argument("--garment_tex", type=Path, default=None)

    ap.add_argument("--blender_bin", type=Path, required=True)
    ap.add_argument("--out_glb", type=Path, required=True)

    ap.add_argument("--workdir", type=Path, default=Path("./_skinned_work"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--skip_normalize", action="store_true")
    ap.add_argument(
        "--texture_mode",
        type=str,
        choices=["pix2surf", "hybrid", "direct"],
        default="hybrid",
        help="Texture generation mode: pix2surf (warp), hybrid (detail-preserving), direct (photo atlas).",
    )
    ap.add_argument(
        "--bind_pose",
        type=str,
        choices=["t_pose", "relaxed"],
        default="relaxed",
        help="SMPL bind/rest pose used for export (default: relaxed).",
    )

    ap.add_argument("--smooth_weights", action="store_true")
    ap.add_argument("--smooth_iters", type=int, default=4)
    ap.add_argument("--smooth_lam", type=float, default=0.5)

    ap.add_argument(
        "--subdivide_iters",
        type=int,
        default=0,
        help="Garment mesh Loop subdivision before weight transfer (0=fast, 1+= denser & slower).",
    )
    ap.add_argument(
        "--prep_min_edge",
        type=int,
        default=0,
        help="Before texture build, upscale front/back JPEGs so max(w,h) is at least this (0=off, fastest).",
    )

    args = ap.parse_args()

    canonical_fast = (
        args.garment_canonical_glb is not None
        and _use_rigid_export()
        and str(args.texture_mode).strip().lower() == "direct"
    )
    _check_dependencies(
        require_torch=args.texture_mode in {"pix2surf", "hybrid"},
        require_scipy=not canonical_fast,
    )

    print(
        f"[INFO] Starting garment pipeline (texture_mode={args.texture_mode}, "
        f"prep_min_edge={args.prep_min_edge}, skip_normalize={args.skip_normalize})…",
        flush=True,
    )

    device = args.device
    if args.texture_mode in {"pix2surf", "hybrid"}:
        _ensure_torch()
        try:
            if isinstance(device, str) and device.lower().startswith("cuda"):
                if not torch.cuda.is_available():
                    print(f"[WARN] CUDA requested ({device}) but not available; falling back to CPU.")
                    device = "cpu"
        except Exception:
            device = "cpu"

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Texture handling
    if args.garment_tex is not None:
        garment_tex = args.garment_tex.resolve()
        _ensure_exists(garment_tex, "garment_tex")
    else:
        if args.front_img is None or args.back_img is None:
            _die("Either provide --garment_tex, or provide --front_img --back_img")
        if args.texture_mode in {"pix2surf", "hybrid"} and args.pix2surf_root is None:
            _die("texture_mode requires --pix2surf_root for pix2surf/hybrid modes")

        pix2surf_root = args.pix2surf_root.resolve() if args.pix2surf_root is not None else None
        if pix2surf_root is not None:
            _ensure_exists(pix2surf_root, "pix2surf_root directory")
        _ensure_exists(args.front_img, "front_img")
        _ensure_exists(args.back_img, "back_img")

        front_jpg = workdir / "shirt_front.jpg"
        back_jpg = workdir / "shirt_back.jpg"
        if args.skip_normalize:
            _to_jpg_rgb(args.front_img, front_jpg)
            _to_jpg_rgb(args.back_img, back_jpg)
        else:
            _normalize_white_bg_product_shot(args.front_img, front_jpg)
            _normalize_white_bg_product_shot(args.back_img, back_jpg)

        if args.prep_min_edge > 0:
            _upscale_min_edge(front_jpg, args.prep_min_edge)
            _upscale_min_edge(back_jpg, args.prep_min_edge)

        assets = None
        if args.texture_mode in {"pix2surf", "hybrid"}:
            assets = _resolve_pix2surf_assets(pix2surf_root) if pix2surf_root is not None else None
        garment_tex = workdir / "upper_tex.jpg"
        _infer_upper_texture_with_pix2surf(
            pix2surf_root=pix2surf_root if pix2surf_root is not None else Path("."),
            assets=assets,
            front_jpg=front_jpg,
            back_jpg=back_jpg,
            device=device,
            texture_mode=args.texture_mode,
            out_tex_jpg=garment_tex,
        )

    out_glb = args.out_glb.resolve()
    out_json = out_glb.with_suffix(".json")
    blender_bin = args.blender_bin.resolve()
    _ensure_exists(blender_bin, "blender executable")
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    canonical_glb = _resolve_canonical_glb_path(
        args.garment_canonical_glb.resolve() if args.garment_canonical_glb else None
    )
    if canonical_glb is not None and _use_rigid_export():
        garment_V, garment_F = _load_mesh_from_glb(canonical_glb)
        print(
            f"[OK] SAM3-style canonical mesh: {canonical_glb.name} "
            f"(V={garment_V.shape[0]}, F={garment_F.shape[0]})",
            flush=True,
        )
        _export_garment_json_rigid(garment_V, out_json, scale=args.scale)
        print(
            "[INFO] Retexturing canonical GLB in Blender (keeps mesh size; avoids OBJ bloat)…",
            flush=True,
        )
        _blender_retexture_glb(
            blender_bin,
            workdir,
            canonical_glb,
            garment_tex.resolve(),
            out_glb,
            args.scale,
        )
        try:
            sz_mb = out_glb.stat().st_size / (1024 * 1024)
            print(f"[OK] Exported rigid textured GLB: {out_glb} ({sz_mb:.1f} MB)", flush=True)
            if sz_mb > 28:
                print(
                    "[WARN] GLB is unusually large for try-on — browsers may fail to display it. "
                    "Check Blender export or regenerate.",
                    flush=True,
                )
        except OSError:
            print(f"[OK] Exported rigid textured GLB: {out_glb}", flush=True)
        print(f"[DONE] Rigid GLB (canonical mesh + your photos): {out_glb}", flush=True)
        print(f"[DONE] Metadata: {out_json}", flush=True)
        return

    # Load SMPL and garment
    smpl = _load_smpl_pkl(args.smpl_model.resolve())
    print(
        f"[OK] Loaded SMPL: V={smpl.v_template.shape[0]}, F={smpl.faces.shape[0]}, J={smpl.weights.shape[1]}",
        flush=True,
    )

    garment_src = args.garment_obj.resolve()
    _ensure_exists(garment_src, "garment_obj")
    garment_V, _, garment_F = _load_obj_vertices_uv_faces(garment_src)
    print(f"[OK] Loaded garment OBJ: V={garment_V.shape[0]}, F={garment_F.shape[0]}", flush=True)
    template_glb = args.garment_template_glb.resolve() if args.garment_template_glb is not None else None
    if template_glb is not None:
        _ensure_exists(template_glb, "garment_template_glb")

    garment_V, garment_F = _subdivide_garment_mesh(garment_V, garment_F, args.subdivide_iters)

    garment_for_blender = garment_src
    if args.subdivide_iters > 0:
        garment_for_blender = workdir / "garment_subdivided.obj"
        _write_obj_triangles(garment_for_blender, garment_V, garment_F)
        print(
            f"[OK] Subdivided garment for export: V={garment_V.shape[0]}, F={garment_F.shape[0]} → {garment_for_blender.name}",
            flush=True,
        )

    Wg: Optional[np.ndarray] = None
    if template_glb is None:
        # Weight transfer (on subdivided vertices)
        Wg = _transfer_weights_barycentric_fixed(smpl, garment_V)

        if args.smooth_weights:
            Wg = _smooth_weights_laplacian(garment_F, Wg, iters=args.smooth_iters, lam=args.smooth_lam)
            print("[OK] Smoothed garment weights.")

        Wg = _limit_skin_influences_to_four(Wg)
        print("[OK] Skin weights limited to top-4 influences per vertex (renormalized).")
    else:
        print(f"[INFO] Using template GLB rig/weights: {template_glb}")

    parents = _smpl_parents(smpl)
    bind_joints = _compute_bind_pose_joints(smpl, args.bind_pose)

    if template_glb is None and Wg is not None:
        garment_V = _pose_garment_vertices_to_bind(garment_V, Wg, smpl, args.bind_pose)
        garment_uv = _compute_garment_front_back_uv(garment_V, garment_F, Wg=Wg)
        garment_for_blender = workdir / "garment_posed.obj"
        _write_obj_triangles(garment_for_blender, garment_V, garment_F, uv=garment_uv)
        bleed = "full-bleed" if _use_texture_full_bleed() else "landmark-aligned"
        print(f"[OK] Wrote posed garment with {bleed} UVs", flush=True)

    if _use_rigid_export() and template_glb is None and Wg is not None:
        if not (workdir / "garment_posed.obj").is_file():
            garment_uv = _compute_garment_front_back_uv(garment_V, garment_F, Wg=Wg)
            garment_for_blender = workdir / "garment_posed.obj"
            _write_obj_triangles(garment_for_blender, garment_V, garment_F, uv=garment_uv)
        _export_garment_json_rigid(garment_V, out_json, scale=args.scale)
        _blender_export_rigid_glb(
            blender_bin,
            workdir,
            garment_for_blender,
            garment_tex.resolve(),
            out_glb,
            args.scale,
            args.center,
        )
        print(f"[DONE] Rigid GLB written to: {out_glb}", flush=True)
        print(f"[DONE] Pose metadata written to: {out_json}", flush=True)
        return

    _export_pose_metadata(smpl, bind_joints, garment_V, out_json, scale=args.scale)

    if template_glb is not None:
        blender_script = workdir / "blender_template_retexture_export.py"
        _write_blender_template_retexture_script(blender_script)
        cmd = [
            str(blender_bin),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--template_glb", str(template_glb),
            "--garment_tex", str(garment_tex),
            "--out_glb", str(out_glb),
            "--scale", str(args.scale),
        ]
    else:
        # Write NPZ for Blender
        if Wg is None:
            _die("Internal error: garment weights missing for non-template export path")
        weights_npz = workdir / "garment_skin.npz"
        np.savez(
            weights_npz,
            joints=bind_joints,
            parents=parents,
            joint_names=np.asarray(smpl.joint_names, dtype=object),
            garment_weights=Wg,
        )
        print(f"[OK] Wrote skin package: {weights_npz}")

        blender_script = workdir / "blender_smpl_skin_export.py"
        _write_blender_skin_export_script(blender_script)
        cmd = [
            str(blender_bin),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--garment_obj", str(garment_for_blender),
            "--garment_tex", str(garment_tex),
            "--weights_npz", str(weights_npz),
            "--out_glb", str(out_glb),
            "--scale", str(args.scale),
        ]
        if args.center:
            cmd.append("--center")

    _run(cmd)
    print(f"[DONE] Skinned GLB written to: {out_glb}")
    print(f"[DONE] Pose metadata written to: {out_json}")


if __name__ == "__main__":
    main()
