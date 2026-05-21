"""
Garment flat-lay preprocessing: background removal + foreground / print-region bbox.

Used by server-local.py and pipeline/sam_segment.py so behaviour stays in sync.

Handles difficult product shots: near-white jerseys, light-grey studio backdrops,
and small holes inside the alpha mask after segmentation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

_rembg_session = None


def _get_rembg_session():
    """Reuse a lightweight rembg model (u2netp) — much faster than default u2net on CPU."""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    from rembg import new_session

    model = (os.environ.get("REMBG_MODEL") or "u2netp").strip() or "u2netp"
    _rembg_session = new_session(model)
    return _rembg_session


def _maybe_downscale_for_rembg(img_pil, max_edge: int):
    """Segmentation does not need 4K; cap longest side to keep CPU runs under ~2 min."""
    from PIL import Image

    if max_edge <= 0:
        return img_pil, 1.0
    w, h = img_pil.size
    m = max(w, h)
    if m <= max_edge:
        return img_pil, 1.0
    scale = max_edge / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return img_pil.resize((nw, nh), resample), scale


def _bbox_from_mask(mask: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) in pixel coords for mask > thresh, or None if empty."""
    ys, xs = np.where(mask > thresh)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def sidecar_mask_path(image_path: Path) -> Path:
    """Canonical location of the garment alpha sidecar for a processed image."""
    p = Path(image_path)
    return p.with_name(p.stem + ".mask.png")


def _write_mask_sidecar(dst_image: Path, alpha: np.ndarray, *, binarize_thresh: int = 16) -> Path:
    """
    Persist the garment silhouette next to the processed image so downstream
    stages can skip brittle brightness-based foreground detection
    (critical for white / light-coloured garments like cardigans).
    """
    from PIL import Image

    mask_path = sidecar_mask_path(dst_image)
    a = np.asarray(alpha).astype(np.uint8)
    a = np.where(a > int(binarize_thresh), 255, 0).astype(np.uint8)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode="L").save(mask_path, "PNG", optimize=True)
    return mask_path


def _defringe_alpha_matte(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Replace rembg halos on dark garments with the median garment colour.

    Fringe pixels that look like backdrop bleed are repainted instead of punched
    out of the mask so the flat-lay silhouette stays solid (no clear lines
    between torso and sleeves after UV warp).
    """
    try:
        import cv2
    except ImportError:
        return alpha

    a = np.asarray(alpha, dtype=np.uint8)
    if not np.any(a > 16):
        return a

    rgb = np.asarray(rgba[:, :, :3], dtype=np.int32)
    fg = a > 16
    rgb_fg = rgb[fg]
    if rgb_fg.size == 0:
        return a

    dom_lum = int(np.median(rgb_fg.astype(np.int32).sum(axis=1)))
    # Only defringe when the garment body is clearly darker than the studio backdrop.
    if dom_lum >= 520:
        return a

    near_white = (rgb[..., 0] >= 236) & (rgb[..., 1] >= 236) & (rgb[..., 2] >= 236)
    bright = rgb.sum(axis=-1) > 690
    fg_u8 = (fg.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(fg_u8, k, iterations=1) > 0
    fringe = fg & ~eroded
    repaint = fringe & (near_white | bright)
    if not np.any(repaint):
        return a

    dom = np.median(rgb_fg, axis=0).astype(np.uint8)
    rgba[repaint, :3] = dom
    return a


def _repair_alpha_holes(alpha: np.ndarray, *, kernel: int, iterations: int) -> np.ndarray:
    """
    Close small gaps inside the garment mask (common on cream/white fabric with rembg).
    kernel/iterations 0 disables.
    """
    if kernel <= 0 or iterations <= 0:
        return alpha
    try:
        import cv2
    except ImportError:
        return alpha
    k = max(3, int(kernel))
    if k % 2 == 0:
        k += 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    a = alpha.astype(np.uint8)
    _, a = cv2.threshold(a, 16, 255, cv2.THRESH_BINARY)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, el, iterations=int(iterations))
    # Remove tiny background speckles classified as foreground
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, el, iterations=1)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def preprocess_garment_product_shot(src_path: Path, dst_path: Path) -> dict[str, Any]:
    """
    Remove background from a garment product shot; composite onto white for Pix2Surf.

    Returns metadata: method, fg_bbox (xywh or null), paths.
    """
    from PIL import Image

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "src": str(src_path.resolve()),
        "dst": str(dst_path.resolve()),
        "mask": None,
        "method": None,
        "fg_bbox": None,
    }

    fast = os.environ.get("GARMENT_PREPROCESS_FAST", "").strip().lower() in ("1", "true", "yes", "on")

    if not fast:
        try:
            from rembg import remove as rembg_remove

            max_edge = int(os.environ.get("GARMENT_PREPROCESS_MAX_EDGE", "1280") or "1280")
            img_pil = Image.open(src_path).convert("RGBA")
            seg_pil, seg_scale = _maybe_downscale_for_rembg(img_pil, max_edge)
            session = _get_rembg_session()
            result = rembg_remove(seg_pil, session=session)
            if seg_scale != 1.0:
                result = result.resize(img_pil.size, Image.LANCZOS)
            alpha = np.array(result.split()[3])
            rk = int(os.environ.get("GARMENT_PREPROCESS_ALPHA_CLOSE_KERNEL", "5") or "5")
            ri = int(os.environ.get("GARMENT_PREPROCESS_ALPHA_CLOSE_ITERS", "2") or "2")
            alpha = _repair_alpha_holes(alpha, kernel=rk, iterations=ri)
            rgba = np.array(result)
            rgba[:, :, 3] = alpha
            alpha = _defringe_alpha_matte(rgba, alpha)
            rgba[:, :, 3] = alpha
            meta["fg_bbox"] = _bbox_from_mask(alpha)
            result = Image.fromarray(rgba, mode="RGBA")
            white = Image.new("RGBA", result.size, (255, 255, 255, 255))
            white.paste(result, mask=result.split()[3])
            white.convert("RGB").save(dst_path, "JPEG", quality=95)
            mask_path = _write_mask_sidecar(dst_path, alpha)
            meta["method"] = "rembg"
            meta["mask"] = str(mask_path.resolve())
            print(f"  [rembg] Background removed: {src_path.name} → {dst_path.name} (+ {mask_path.name})")
            return meta
        except ImportError:
            print("  [rembg] not installed — falling back to GrabCut")
        except Exception as e:
            print(f"  [rembg] failed ({e}) — falling back to GrabCut")

    try:
        import cv2

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            raise RuntimeError(f"cv2.imread failed for {src_path}")
        h, w = img_bgr.shape[:2]
        border = max(4, int(min(h, w) * 0.05))
        mask_gc = np.zeros((h, w), np.uint8)
        mask_gc[:] = cv2.GC_PR_FGD
        mask_gc[:border, :] = cv2.GC_BGD
        mask_gc[-border:, :] = cv2.GC_BGD
        mask_gc[:, :border] = cv2.GC_BGD
        mask_gc[:, -border:] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (border, border, w - 2 * border, h - 2 * border)
        cv2.grabCut(img_bgr, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        fg_mask = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        meta["fg_bbox"] = _bbox_from_mask(fg_mask)
        alpha3 = fg_mask[:, :, None].astype(np.float32) / 255.0
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        composited = (img_bgr.astype(np.float32) * alpha3 + white.astype(np.float32) * (1 - alpha3)).astype(
            np.uint8
        )
        cv2.imwrite(str(dst_path), composited, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mask_path = _write_mask_sidecar(dst_path, fg_mask)
        meta["method"] = "grabcut"
        meta["mask"] = str(mask_path.resolve())
        print(f"  [GrabCut] Background removed: {src_path.name} → {dst_path.name} (+ {mask_path.name})")
        return meta
    except Exception as e:
        print(f"  [GrabCut] failed ({e}) — using raw image")

    shutil.copy2(src_path, dst_path)
    try:
        from PIL import Image

        with Image.open(dst_path) as im:
            w_px, h_px = im.size
        full_mask = np.full((h_px, w_px), 255, dtype=np.uint8)
        mask_path = _write_mask_sidecar(dst_path, full_mask)
        meta["mask"] = str(mask_path.resolve())
    except Exception as e:
        print(f"  [passthrough] Could not write mask sidecar: {e}")
    meta["method"] = "passthrough"
    print(f"  [passthrough] No BG removal applied: {src_path.name}")
    return meta
