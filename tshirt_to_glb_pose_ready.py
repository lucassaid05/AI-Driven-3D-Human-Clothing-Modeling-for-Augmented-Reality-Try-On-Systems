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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
except Exception:
    torch = None

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


def _normalize_white_bg_product_shot(
    src: Path,
    dst: Path,
    *,
    out_size: int = 1024,
    white_thresh: int = 245,
    pad_frac: float = 0.08,
) -> None:
    """Conservative white-background normalizer for product shots."""
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img)
    bg = (arr[..., 0] >= white_thresh) & (arr[..., 1] >= white_thresh) & (arr[..., 2] >= white_thresh)
    fg = ~bg

    if fg.sum() < 500:
        canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
        tmp = img.copy()
        tmp.thumbnail((out_size, out_size), Image.BICUBIC)
        ox = (out_size - tmp.size[0]) // 2
        oy = (out_size - tmp.size[1]) // 2
        canvas.paste(tmp, (ox, oy))
        dst.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst, "JPEG", quality=95)
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
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "JPEG", quality=95)


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


# -------------------------
# Pix2Surf texture inference
# -------------------------
@dataclass(frozen=True)
class Pix2SurfAssets:
    seg_shirts_front: Path
    seg_shirts_back: Path
    map_shirts_front: Path
    map_shirts_back: Path


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
    assets: Pix2SurfAssets,
    front_jpg: Path,
    back_jpg: Path,
    device: str,
    out_tex_jpg: Path,
) -> None:
    if torch is None:
        _die("PyTorch/torchvision not available. Install torch + torchvision.")
    if cv2 is None:
        _die("OpenCV not available. Install opencv-python.")

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

    base = np.zeros((2000, 2000, 3), dtype=np.uint8)
    base[0:2000, 0:1000]    = cut1
    base[0:2000, 1000:2000] = cut2

    mean_brightness = float(base.mean())
    print(f"  Texture brightness (Pix2Surf): mean={mean_brightness:.1f}/255")

    # If pix2surf output is dim, blend with direct source images for correct colours
    if mean_brightness < 100:
        def _crop_tile(path, tw, th, thresh=238):
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
            bg = (arr[...,0]>=thresh)&(arr[...,1]>=thresh)&(arr[...,2]>=thresh)
            fg = ~bg
            if fg.sum() < 500:
                return np.array(img.resize((tw,th), Image.LANCZOS))
            ys,xs = np.where(fg)
            y0,y1,x0,x1 = int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max())
            h=y1-y0+1; w=x1-x0+1; pad=int(max(h,w)*0.04)
            y0=max(0,y0-pad); x0=max(0,x0-pad)
            y1=min(arr.shape[0]-1,y1+pad); x1=min(arr.shape[1]-1,x1+pad)
            crop=img.crop((x0,y0,x1+1,y1+1))
            canvas=Image.new("RGB",(tw,th),(255,255,255))
            crop.thumbnail((tw,th),Image.LANCZOS)
            ox=(tw-crop.size[0])//2; oy=(th-crop.size[1])//2
            canvas.paste(crop,(ox,oy))
            return np.array(canvas)
        f_direct = _crop_tile(front_jpg, 1000, 2000)
        b_direct = cv2.flip(_crop_tile(back_jpg, 1000, 2000), 1)
        direct = np.zeros((2000,2000,3), dtype=np.uint8)
        direct[0:2000, 0:1000]    = f_direct
        direct[0:2000, 1000:2000] = b_direct
        direct_mean = float(direct.mean())
        # Blend: weight toward direct when pix2surf is dim
        w_p2s = max(0.0, min(1.0, (mean_brightness - 20) / 80.0))
        base = np.clip(
            base.astype(np.float32)*w_p2s + direct.astype(np.float32)*(1-w_p2s),
            0, 255).astype(np.uint8)
        print(f"  Blended: p2s_w={w_p2s:.2f}  direct_mean={direct_mean:.0f}  final_mean={base.mean():.1f}/255")
    else:
        print(f"  Using pix2surf output (brightness OK: {mean_brightness:.0f}/255)")

    # Save as JPEG — Blender embeds it directly, correct colours guaranteed
    out_tex_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_tex_jpg), cv2.cvtColor(base, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Wrote upper texture: {out_tex_jpg}")


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


def _write_obj_triangles(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    """Minimal triangle OBJ (1-based faces, no vt — Blender import script projects UVs if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# garment (possibly subdivided)\n")
        for i in range(V.shape[0]):
            x, y, z = float(V[i, 0]), float(V[i, 1]), float(V[i, 2])
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for i in range(F.shape[0]):
            a, b, c = int(F[i, 0]) + 1, int(F[i, 1]) + 1, int(F[i, 2]) + 1
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
    joints_scaled = (smpl.joints * scale).tolist()
    
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
                smpl.joints[SMPL_JOINT_INDICES["left_shoulder"]] - 
                smpl.joints[SMPL_JOINT_INDICES["right_shoulder"]]
            ) * scale),
            "smpl_torso_height": float(np.linalg.norm(
                smpl.joints[SMPL_JOINT_INDICES["neck"]] - 
                smpl.joints[SMPL_JOINT_INDICES["pelvis"]]
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
            print("[UV] No UV data — generating front/back projection UVs")
            uv_layer = m.data.uv_layers.new(name="UVMap")
            mesh = m.data

            all_x = [v.co.x for v in mesh.vertices]
            all_y = [v.co.y for v in mesh.vertices]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            dx = max_x - min_x if max_x > min_x else 1.0
            dy = max_y - min_y if max_y > min_y else 1.0

            for poly in mesh.polygons:
                is_front = (poly.normal.z >= 0)
                for li in poly.loop_indices:
                    co = mesh.vertices[mesh.loops[li].vertex_index].co
                    u01 = (co.x - min_x) / dx        # 0..1 left-to-right
                    v01 = (co.y - min_y) / dy        # 0..1 bottom-to-top
                    if is_front:
                        uv_layer.data[li].uv = (u01 * 0.5, v01)
                    else:
                        uv_layer.data[li].uv = (0.5 + (1.0 - u01) * 0.5, v01)
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


# -------------------------
# Main
# -------------------------
def _check_dependencies() -> None:
    """
    Fail fast with an actionable install command if key deps are missing.
    NOTE: 'rtree' is NO LONGER required — closest-point search is implemented
    using scipy.spatial.cKDTree so trimesh.proximity is not called.
    """
    missing = []
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
    try:
        import torch
    except ImportError:
        missing.append("torch torchvision")
    if missing:
        print("[ERROR] Missing required packages:", ", ".join(missing), file=sys.stderr)
        print("[ERROR] Run:  pip install", " ".join(missing), file=sys.stderr)
        sys.exit(1)
    # rtree is optional — we implement our own spatial search without it.
    # But warn if trimesh would fail for other reasons.
    try:
        import trimesh as _tm
        _ = _tm.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
    except Exception as e:
        print(f"[WARN] trimesh sanity check failed: {e}", file=sys.stderr)


def main() -> None:
    _check_dependencies()
    ap = argparse.ArgumentParser()

    ap.add_argument("--pix2surf_root", type=Path, default=None)
    ap.add_argument("--front_img", type=Path, default=None)
    ap.add_argument("--back_img", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--smpl_model", type=Path, required=True)
    ap.add_argument("--garment_obj", type=Path, required=True)
    ap.add_argument("--garment_tex", type=Path, default=None)

    ap.add_argument("--blender_bin", type=Path, required=True)
    ap.add_argument("--out_glb", type=Path, required=True)

    ap.add_argument("--workdir", type=Path, default=Path("./_skinned_work"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--skip_normalize", action="store_true")

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
        default=768,
        help="Before Pix2Surf, upscale front/back JPEGs so max(w,h) is at least this (0=off). Improves texture; slower I/O.",
    )

    args = ap.parse_args()

    device = args.device
    try:
        if isinstance(device, str) and device.lower().startswith("cuda") and torch is not None:
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
        if args.pix2surf_root is None or args.front_img is None or args.back_img is None:
            _die("Either provide --garment_tex, or provide --pix2surf_root --front_img --back_img")

        pix2surf_root = args.pix2surf_root.resolve()
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

        assets = _resolve_pix2surf_assets(pix2surf_root)
        garment_tex = workdir / "upper_tex.jpg"
        _infer_upper_texture_with_pix2surf(
            pix2surf_root=pix2surf_root,
            assets=assets,
            front_jpg=front_jpg,
            back_jpg=back_jpg,
            device=device,
            out_tex_jpg=garment_tex,
        )

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

    garment_V, garment_F = _subdivide_garment_mesh(garment_V, garment_F, args.subdivide_iters)

    garment_for_blender = garment_src
    if args.subdivide_iters > 0:
        garment_for_blender = workdir / "garment_subdivided.obj"
        _write_obj_triangles(garment_for_blender, garment_V, garment_F)
        print(
            f"[OK] Subdivided garment for export: V={garment_V.shape[0]}, F={garment_F.shape[0]} → {garment_for_blender.name}",
            flush=True,
        )

    # Weight transfer (on subdivided vertices)
    Wg = _transfer_weights_barycentric_fixed(smpl, garment_V)

    if args.smooth_weights:
        Wg = _smooth_weights_laplacian(garment_F, Wg, iters=args.smooth_iters, lam=args.smooth_lam)
        print("[OK] Smoothed garment weights.")

    Wg = _limit_skin_influences_to_four(Wg)
    print("[OK] Skin weights limited to top-4 influences per vertex (renormalized).")

    # Parent list
    kt = smpl.kintree_table
    if kt.shape[0] != 2:
        _die(f"Unexpected kintree_table shape: {kt.shape}")
    parents = kt[0].astype(np.int32).copy()
    parents[0] = -1

    # Write NPZ for Blender
    weights_npz = workdir / "garment_skin.npz"
    np.savez(
        weights_npz,
        joints=smpl.joints,
        parents=parents,
        joint_names=np.asarray(smpl.joint_names, dtype=object),
        garment_weights=Wg,
    )
    print(f"[OK] Wrote skin package: {weights_npz}")

    # Export pose metadata JSON
    out_glb = args.out_glb.resolve()
    out_json = out_glb.with_suffix(".json")
    _export_pose_metadata(smpl, garment_V, out_json, scale=args.scale)

    # Write Blender exporter script
    blender_script = workdir / "blender_smpl_skin_export.py"
    _write_blender_skin_export_script(blender_script)

    blender_bin = args.blender_bin.resolve()
    _ensure_exists(blender_bin, "blender executable")

    out_glb.parent.mkdir(parents=True, exist_ok=True)

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
