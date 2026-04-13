"""
Garment flat-lay preprocessing: background removal + foreground / print-region bbox.

Used by server-local.py and pipeline/sam_segment.py so behaviour stays in sync.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _bbox_from_mask(mask: np.ndarray, thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) in pixel coords for mask > thresh, or None if empty."""
    ys, xs = np.where(mask > thresh)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


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
        "method": None,
        "fg_bbox": None,
    }

    try:
        from rembg import remove as rembg_remove

        img_pil = Image.open(src_path).convert("RGBA")
        result = rembg_remove(img_pil)
        alpha = np.array(result.split()[3])
        meta["fg_bbox"] = _bbox_from_mask(alpha)
        white = Image.new("RGBA", result.size, (255, 255, 255, 255))
        white.paste(result, mask=result.split()[3])
        white.convert("RGB").save(dst_path, "JPEG", quality=95)
        meta["method"] = "rembg"
        print(f"  [rembg] Background removed: {src_path.name} → {dst_path.name}")
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
        meta["method"] = "grabcut"
        print(f"  [GrabCut] Background removed: {src_path.name} → {dst_path.name}")
        return meta
    except Exception as e:
        print(f"  [GrabCut] failed ({e}) — using raw image")

    shutil.copy2(src_path, dst_path)
    meta["method"] = "passthrough"
    print(f"  [passthrough] No BG removal applied: {src_path.name}")
    return meta
