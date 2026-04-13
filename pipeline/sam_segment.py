#!/usr/bin/env python3
"""
Segmentation helpers for the dissertation pipeline (SAM family + practical fallbacks).

SAM 3 is not bundled here; when Meta's segment-anything checkpoint is available,
set SAM_CHECKPOINT and use --engine sam (SAM 1.x). Otherwise we use MediaPipe
selfie segmentation and/or rembg for robust person masks without extra weights.

Examples:
  python pipeline/sam_segment.py person --input photo.jpg --out-dir ./masks/
  python pipeline/sam_segment.py garment --input shirt.png --out-dir ./masks/
  python pipeline/sam_segment.py person --input photo.jpg --out-dir ./masks/ --engine sam
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _read_bgr(path: Path):
    import cv2

    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return im


def _save_mask_png(path: Path, mask_u8: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_u8)


def segment_person_mediapipe(bgr: np.ndarray) -> np.ndarray:
    """Binary uint8 mask 0/255 via MediaPipe selfie segmentation."""
    import cv2

    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError("Install mediapipe: pip install mediapipe") from e

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_seg = mp.solutions.selfie_segmentation
    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(rgb)
    if res.segmentation_mask is None:
        raise RuntimeError("MediaPipe returned no segmentation_mask")
    m = (res.segmentation_mask * 255.0).astype(np.uint8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def segment_person_rembg_rgba(bgr: np.ndarray) -> np.ndarray:
    from PIL import Image
    import cv2

    try:
        from rembg import remove as rembg_remove
    except ImportError as e:
        raise RuntimeError("Install rembg: pip install rembg") from e

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    out = rembg_remove(pil.convert("RGBA"))
    a = np.array(out.split()[3])
    _, m = cv2.threshold(a, 8, 255, cv2.THRESH_BINARY)
    return m


def segment_person_sam(bgr: np.ndarray, checkpoint: str, model_type: str) -> np.ndarray:
    """Pick largest SAM mask covering image center (single-subject heuristic)."""
    import cv2

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
    except ImportError as e:
        raise RuntimeError("Install segment-anything and torch; pip install git+https://github.com/facebookresearch/segment-anything.git") from e

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    gen = SamAutomaticMaskGenerator(sam)
    masks = gen.generate(rgb)
    if not masks:
        raise RuntimeError("SAM returned no masks")
    h, w = bgr.shape[:2]
    cy, cx = h // 2, w // 2
    best = None
    best_score = -1
    for m in masks:
        seg = m["segmentation"].astype(np.uint8) * 255
        if seg[cy, cx] < 128:
            continue
        area = int(m["area"])
        if area > best_score:
            best_score = area
            best = seg
    if best is None:
        best = max(masks, key=lambda x: int(x["area"]))["segmentation"].astype(np.uint8) * 255
    return best


def torso_mask_from_pose(bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Shirt/torso region from MediaPipe pose landmarks (falls back to bbox on mask)."""
    import cv2

    h, w = bgr.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    try:
        import mediapipe as mp

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            res = pose.process(rgb)
        if not res.pose_landmarks:
            raise RuntimeError("no pose")
        lm = res.pose_landmarks.landmark

        def pt(idx):
            p = lm[idx]
            return int(p.x * w), int(p.y * h)

        # Shoulders and hips (shirt block)
        pts = np.array(
            [pt(11), pt(12), pt(24), pt(23)], dtype=np.int32
        )  # L shoulder, R shoulder, R hip, L hip
        cv2.fillConvexPoly(out, pts, 255)
        # Dilate slightly into sleeves
        k = max(5, min(h, w) // 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2, k * 2))
        out = cv2.dilate(out, kernel)
        out = cv2.bitwise_and(out, person_mask)
        return out
    except Exception:
        # Central rectangle on foreground bbox
        ys, xs = np.where(person_mask > 127)
        if xs.size == 0:
            return person_mask
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw, bh = x1 - x0, y1 - y0
        tx0 = int(x0 + 0.15 * bw)
        tx1 = int(x0 + 0.85 * bw)
        ty0 = int(y0 + 0.12 * bh)
        ty1 = int(y0 + 0.62 * bh)
        out[ty0:ty1, tx0:tx1] = 255
        return cv2.bitwise_and(out, person_mask)


def run_person(args: argparse.Namespace) -> dict[str, Any]:
    bgr = _read_bgr(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = args.engine
    if engine == "auto":
        try:
            pm = segment_person_mediapipe(bgr)
            engine_used = "mediapipe_selfie"
        except Exception:
            pm = segment_person_rembg_rgba(bgr)
            engine_used = "rembg"
    elif engine == "mediapipe":
        pm = segment_person_mediapipe(bgr)
        engine_used = "mediapipe_selfie"
    elif engine == "rembg":
        pm = segment_person_rembg_rgba(bgr)
        engine_used = "rembg"
    elif engine == "sam":
        ckpt = os.environ.get("SAM_CHECKPOINT", "").strip() or args.sam_checkpoint
        if not ckpt:
            raise SystemExit("Set SAM_CHECKPOINT or pass --sam-checkpoint for --engine sam")
        mtype = os.environ.get("SAM_MODEL_TYPE", "vit_b").strip() or args.sam_model_type
        pm = segment_person_sam(bgr, ckpt, mtype)
        engine_used = f"sam_{mtype}"
    else:
        raise SystemExit(f"Unknown engine {engine}")

    _save_mask_png(out_dir / "person_mask.png", pm)
    tm = torso_mask_from_pose(bgr, pm)
    _save_mask_png(out_dir / "torso_shirt_mask.png", tm)

    meta = {
        "mode": "person",
        "engine": engine_used,
        "person_mask": str((out_dir / "person_mask.png").resolve()),
        "torso_shirt_mask": str((out_dir / "torso_shirt_mask.png").resolve()),
    }
    (out_dir / "segmentation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return meta


def run_garment(args: argparse.Namespace) -> dict[str, Any]:
    from pipeline.garment_preprocess import preprocess_garment_product_shot

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = out_dir / "garment_white_bg.jpg"
    meta = preprocess_garment_product_shot(inp, proc)

    import cv2

    # Derive binary mask from processed vs white background (inverse of near-white)
    im = cv2.imread(str(proc))
    if im is None:
        raise RuntimeError(f"failed to read {proc}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _save_mask_png(out_dir / "garment_fg_mask.png", mask)

    bbox = meta.get("fg_bbox")
    garment_meta = {
        "mode": "garment",
        "preprocess": meta,
        "foreground_mask": str((out_dir / "garment_fg_mask.png").resolve()),
        "print_region_bbox_xywh": bbox,
        "processed_image": str(proc.resolve()),
    }
    (out_dir / "garment_segmentation.json").write_text(
        json.dumps(garment_meta, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(garment_meta, indent=2, default=str))
    return garment_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("person", help="Person + torso/shirt masks from a photo")
    pp.add_argument("--input", "-i", required=True, type=Path)
    pp.add_argument("--out-dir", "-o", required=True, type=Path)
    pp.add_argument(
        "--engine",
        choices=("auto", "mediapipe", "rembg", "sam"),
        default="auto",
        help="auto tries MediaPipe then rembg",
    )
    pp.add_argument("--sam-checkpoint", default="", help="SAM .pth (or set SAM_CHECKPOINT)")
    pp.add_argument("--sam-model-type", default="vit_b", help="SAM registry key, e.g. vit_b, vit_h")
    pp.set_defaults(_run=run_person)

    pg = sub.add_parser("garment", help="Flat garment: BG removal + fg mask + bbox JSON")
    pg.add_argument("--input", "-i", required=True, type=Path)
    pg.add_argument("--out-dir", "-o", required=True, type=Path)
    pg.set_defaults(_run=run_garment)

    args = p.parse_args()
    args._run(args)


if __name__ == "__main__":
    # Allow `python pipeline/sam_segment.py` from repo root (parent on sys.path)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()
