"""
AR Try-On Server with Local GLB Generation

This server:
1. Serves the hub page (index.html) at /, the prototype landing page at
   /prototype.html, and the built AR try-on bundle under /dist/apps/ar-tryon/
2. Provides API endpoints for GLB generation using tshirt_to_glb_pose_ready.py
3. Handles CORS and Permissions-Policy so browsers on the LAN can use the site
4. Binds to 0.0.0.0 so phones/tablets on the same Wi‑Fi can open the page

Usage:
    python server-local.py
    Open http://localhost:PORT/  — hub with Google 2D, WANNA, and Prototype cards

Mobile:
    Startup prints a LAN URL (e.g. http://192.168.x.x:PORT/). Open that on your
    phone. If the camera is blocked (insecure context), use HTTPS via optional
    cert files (HTTPS_CERT / HTTPS_KEY in .env) or a tunnel — see docs/TUNNELS.md.
    Or run `npm run dev` for Vite's HTTPS dev server on :5173.

    JSON helper: GET /api/server-info — origins and deep links for QR / bookmarks.

Configuration:
    Use .env in the same folder as this script (see CONFIG keys below).
"""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import json
import subprocess
import shutil
import base64
import urllib.parse
import posixpath
import uuid
import threading
import traceback
import sys
import time
import re
import ssl as ssl_module
import urllib.error
import urllib.request

# Ensure Windows consoles don't crash on Unicode banners/emojis.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================================
# CONFIGURATION — loaded automatically from .env (same folder as this script)
# Edit .env to change paths. Do NOT hardcode values here.
# ============================================================================

def _load_dotenv(env_file: Path) -> dict:
    """Parse KEY=VALUE lines; ignore # comments and blank lines."""
    env: dict = {}
    if not env_file.exists():
        print(f"  WARNING: .env not found at {env_file} — using built-in defaults")
        return env
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


# Resolve the server's own directory once — every path calculation uses this.
_SERVER_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_ENV = _load_dotenv(_SERVER_DIR / ".env")


def _e(key: str, default: str) -> str:
    """Return value from .env or fall back to default."""
    return _ENV.get(key, default)


CONFIG = {
    "PIX2SURF_ROOT": _e("PIX2SURF_ROOT", "../../pix2surf"),
    "SMPL_MODEL":    _e("SMPL_PKL",      "../../smpl/SMPL_NEUTRAL.pkl"),
    # Same OBJ as your reference GLB → same silhouette; only texture comes from photos.
    "GARMENT_OBJ":   _e("GARMENT_OBJ",   "../../pix2surf/assets/init_meshes/shirts.obj"),
    # Optional pre-rigged template GLB (keeps original armature/weights, only swaps texture).
    "GARMENT_TEMPLATE_GLB": _e("GARMENT_TEMPLATE_GLB", "../../manu_tshirt_smpl_ready.glb"),
    # SAM3 canonical shirt shape + your photos → rigid GLB (best try-on parity with SAM3 demo)
    "GARMENT_CANONICAL_GLB": _e("GARMENT_CANONICAL_GLB", "public/sam3/shirt_canonical.glb"),
    "GARMENT_EXPORT_RIGID": _e("GARMENT_EXPORT_RIGID", "1"),
    "BLENDER_BIN":   _e("BLENDER_BIN",   "blender"),
    "DEVICE":        _e("DEVICE",        "cpu"),
    "OUTPUT_DIR":    _e("OUTPUT_DIR",    "./generated"),
    # Script sits in the same folder as server-local.py
    "SCRIPT_PATH":   _e("SCRIPT_PATH",   "./tshirt_to_glb_pose_ready.py"),
    # "." means serve HTML from the same folder as this script
    "STATIC_DIR":    _e("STATIC_DIR",    "."),
    # 0 = fastest (recommended for <~5 min on CPU); 1+ = denser mesh, much slower weight transfer
    "SUBDIVIDE_ITERS": _e("SUBDIVIDE_ITERS", "0"),
    # Weight smoothing iterations in tshirt_to_glb_pose_ready.py (lower = faster)
    "SMOOTH_ITERS": _e("SMOOTH_ITERS", "4"),
    # Kill subprocess if still running after this many seconds (default 5 min)
    "GEN_TIMEOUT_SEC": int(_e("GEN_TIMEOUT_SEC", "300")),
    # direct=photo-preserving (best for logos/text), hybrid=photo+Pix2Surf, pix2surf=neural warp only
    "TEXTURE_MODE": _e("TEXTURE_MODE", "direct"),
    # Min longest side (px) for front/back before texture; 0 = skip upscale (fastest). 896 only for hybrid/pix2surf.
    "PREP_MIN_EDGE": int(
        _e(
            "PREP_MIN_EDGE",
            "0" if str(_e("TEXTURE_MODE", "direct")).strip().lower() == "direct" else "512",
        )
    ),
    # Bind/rest armature pose exported by tshirt_to_glb_pose_ready.py
    "BIND_POSE": _e("BIND_POSE", "relaxed"),
    # Landmark-anchored tile placement. The detected shoulder line of the
    # uploaded photo is mapped to the mesh's shoulder UV row and the hem to
    # the mesh's hem UV row, so the shoulders/sleeves of the printed graphic
    # align with the 3D mesh's shoulders/sleeves — the shirt no longer sits
    # above the user's shoulders.
    #   TILE_ALIGN_TEMPLATE : "1" enabled (default), "0" legacy bbox-fit.
    #   TILE_ALIGN_MODE     : "shoulder" (default) or "center" (legacy
    #                         geometric-centre placement).
    #   TILE_FILL_FRAC      : in shoulder mode, uniform scale on top of the
    #                         landmark fit (1.0 = exact landmark span).
    #                         In center mode, fraction of the tile the bbox
    #                         fills on each axis (0.70 ≈ comfortable size).
    #   TILE_SHOULDER_FRAC  : mesh shoulder UV row, default 0.12 (12% of tile
    #                         height from the top).
    #   TILE_HEM_FRAC       : mesh hem UV row, default 0.92.
    "TILE_ALIGN_TEMPLATE": _e("TILE_ALIGN_TEMPLATE", "1"),
    "TILE_ALIGN_MODE": _e("TILE_ALIGN_MODE", "shoulder"),
    "TILE_FILL_FRAC": _e("TILE_FILL_FRAC", "1.0"),
    "TILE_SHOULDER_FRAC": _e("TILE_SHOULDER_FRAC", "0.12"),
    "TILE_HEM_FRAC": _e("TILE_HEM_FRAC", "0.92"),
    # 1 = photo covers full tile + mesh UV uses full bbox (default; avoids margin bands)
    "TEXTURE_FULL_BLEED": _e("GARMENT_TEXTURE_FULL_BLEED", "1"),
    # Optional HTTPS (paths relative to this script’s folder). Create with mkcert or openssl.
    "HTTPS_CERT": _e("HTTPS_CERT", "").strip(),
    "HTTPS_KEY": _e("HTTPS_KEY", "").strip(),
}

HTTP_PORT = int(_e("PORT", "8000"))
# Shown in logs /api/server-info so phones know which URL to use for `npm run dev` (Vite defaults to 5173).
TRYON_VITE_PORT = int(_e("TRYON_VITE_PORT", "5173"))
# Inline base64 in /api/status when GLB is smaller than this (ngrok-friendly).
GLB_INLINE_MAX_BYTES = int(_e("GLB_INLINE_MAX_BYTES", str(6 * 1024 * 1024)))
# Bumped when API behaviour changes (client reads /api/health).
API_VERSION = 2


# Track generation tasks
tasks = {}
tasks_lock = threading.Lock()

# Resolved once at startup — avoids slow/repeated .resolve() on OneDrive paths every generation.
_GENERATION_PATHS: dict | None = None


def get_generation_paths() -> dict:
    """Cache absolute paths for the pipeline subprocess (pay resolve cost once)."""
    global _GENERATION_PATHS
    if _GENERATION_PATHS is not None:
        return _GENERATION_PATHS
    root = _SERVER_DIR
    template = root / CONFIG["GARMENT_TEMPLATE_GLB"]
    canonical = root / CONFIG["GARMENT_CANONICAL_GLB"]
    _GENERATION_PATHS = {
        "script_path": (root / CONFIG["SCRIPT_PATH"]).resolve(),
        "pix2surf_root": (root / CONFIG["PIX2SURF_ROOT"]).resolve(),
        "smpl_model": (root / CONFIG["SMPL_MODEL"]).resolve(),
        "garment_obj": (root / CONFIG["GARMENT_OBJ"]).resolve(),
        "garment_template_glb": template.resolve(),
        "garment_canonical_glb": canonical.resolve(),
        "has_template_glb": template.is_file(),
        "has_canonical_glb": canonical.is_file(),
        "export_rigid": str(CONFIG["GARMENT_EXPORT_RIGID"]).strip().lower() not in ("0", "false", "no", "off"),
    }
    return _GENERATION_PATHS


def _garment_glb_path(task_id: str) -> Path:
    return (_SERVER_DIR / CONFIG["OUTPUT_DIR"] / f"task_{task_id}" / "garment.glb").resolve()


def _wait_for_garment_file(path: Path, timeout_sec: float = 30.0) -> int:
    """Wait until GLB exists and has bytes (OneDrive / slow disks). Returns file size."""
    deadline = time.time() + timeout_sec
    last_size = 0
    stable = 0
    while time.time() < deadline:
        try:
            if path.is_file():
                sz = int(path.stat().st_size)
                if sz >= 64:
                    if sz == last_size:
                        stable += 1
                        if stable >= 2:
                            return sz
                    else:
                        stable = 0
                        last_size = sz
        except OSError:
            pass
        time.sleep(0.35)
    raise FileNotFoundError(f"Garment GLB not ready at {path}")


def _error_recovery_hint(message: str) -> str:
    """Short user-facing hint for common generation failures."""
    m = (message or "").lower()
    if "cuda" in m or "cudnn" in m:
        return "Try DEVICE=cpu in .env, or install a CUDA-enabled PyTorch wheel."
    if "pix2surf" in m or "checkpoint" in m or ("model" in m and "load" in m):
        return "Verify PIX2SURF_ROOT in .env and that Pix2Surf weights are present."
    if "smpl" in m or "pkl" in m:
        return "Add SMPL_NEUTRAL.pkl and set SMPL_PKL in .env."
    if "blender" in m or "exit code" in m:
        return "Install Blender, set BLENDER_BIN in .env to the executable path."
    if "timeout" in m or "time limit" in m:
        return (
            "Exceeded GEN_TIMEOUT_SEC; set DEVICE=cuda in .env if you have a GPU, "
            "or SUBDIVIDE_ITERS=0, PREP_MIN_EDGE=0, and use smaller source images."
        )
    return "Check the server terminal log and .env paths; ensure photos are clear, front/back, good lighting."


def _lan_ipv4():
    """
    Best-effort LAN IPv4 for URLs shown to phones. UDP 'connect' to 8.8.8.8 often works but fails offline or
    on locked-down networks — fall back to resolving the hostname's non-loopback addresses.
    """
    import socket

    def _is_private_lan(a: str) -> bool:
        if a.startswith("127.") or a == "0.0.0.0":
            return False
        # RFC1918 + link-local (common for some USB tethering)
        if a.startswith("10.") or a.startswith("192.168."):
            return True
        if a.startswith("169.254."):
            return True
        if a.startswith("172."):
            try:
                second = int(a.split(".")[1])
                return 16 <= second <= 31
            except (ValueError, IndexError):
                return False
        return False

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.35)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and _is_private_lan(ip):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        candidates = []
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = res[4][0]
            if _is_private_lan(ip):
                candidates.append(ip)
        if candidates:
            # Prefer 192.168.x.x then 10.x then 172.16–31 then 169.254
            for pref in ("192.168.", "10.", "172.", "169.254."):
                for c in candidates:
                    if c.startswith(pref):
                        return c
            return candidates[0]
    except Exception:
        pass

    return None


def _resolve_https_cert_paths():
    """Return (cert_path, key_path) if both configured and exist, else (None, None)."""
    c = CONFIG.get("HTTPS_CERT") or ""
    k = CONFIG.get("HTTPS_KEY") or ""
    if not c or not k:
        return None, None
    cert = (_SERVER_DIR / c).resolve()
    key = (_SERVER_DIR / k).resolve()
    if cert.is_file() and key.is_file():
        return cert, key
    return None, None


def _server_public_urls(scheme: str):
    """Build shareable URLs for desktop and LAN (for /api/server-info and startup logs)."""
    host = "localhost"
    port = HTTP_PORT
    base_local = f"{scheme}://{host}:{port}"
    lan = _lan_ipv4()
    base_lan = f"{scheme}://{lan}:{port}" if lan else None
    # Surface the hub, the prototype landing page, and the built AR try-on.
    # "/" now serves the hub (index.html); the prototype page links onward to
    # the built app under /dist/apps/ar-tryon/.
    pages = {
        "home": "/",
        "prototype": "/prototype.html",
        "tryon": "/dist/apps/ar-tryon/index.html",
    }
    out = {"localhost": {k: base_local + v for k, v in pages.items()}}
    if base_lan:
        out["lan"] = {k: base_lan + v for k, v in pages.items()}
    return out


def _static_extensions_map():
    """Windows http.server often serves .js as text/plain — browsers reject ES modules. Merge safe types."""
    base = dict(getattr(SimpleHTTPRequestHandler, "extensions_map", {}))
    base.update(
        {
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".cjs": "application/javascript",
            ".ts": "application/javascript",
            ".tsx": "application/javascript",
            ".json": "application/json",
            ".css": "text/css; charset=utf-8",
            ".wasm": "application/wasm",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".woff2": "font/woff2",
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
        }
    )
    return base


class APIHandler(SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints and CORS support"""

    extensions_map = _static_extensions_map()

    def guess_type(self, path):
        """Windows mimetypes often label .ts/.js as text/plain; ES modules require application/javascript."""
        _base, ext = posixpath.splitext(path)
        if ext.lower() in (".ts", ".tsx"):
            return "application/javascript", None
        return super().guess_type(path)

    def end_headers(self):
        """Add CORS + Permissions-Policy so mobile browsers may request camera/mic on this origin."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '86400')
        self.send_header(
            'Permissions-Policy',
            'camera=(self), microphone=(self), fullscreen=(self), display-capture=(self)',
        )
        p = (getattr(self, "path", None) or "").lower()
        # Do not let phones/ngrok sit on an old dist/ bundle — run npm run build after UI changes.
        if "/dist/" in p and (p.endswith(".js") or p.endswith(".html") or p.endswith(".css") or p.endswith(".map")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urllib.parse.urlparse(self.path)
        path_lc = parsed.path.lower()
        # Static quick-generation presets are kept in public/ for Vite.
        # Mirror /quick-generation/* to /public/quick-generation/* for python server-local.py.
        # Static presets + SAM3 GLBs: Vite copies public/ → dist/ at build time. Serve from either folder so
        # /sam3/* works whether the asset lives under public/ or only under dist/ after npm run build.
        if parsed.path.startswith("/quick-generation/"):
            for base in ("/public", "/dist"):
                mapped = base + parsed.path
                candidate = (_SERVER_DIR / mapped.lstrip("/")).resolve()
                if candidate.is_file():
                    self.path = mapped
                    return super().do_GET()
        if parsed.path.startswith("/sam3/"):
            for base in ("/public", "/dist"):
                mapped = base + parsed.path
                candidate = (_SERVER_DIR / mapped.lstrip("/")).resolve()
                if candidate.is_file():
                    self.path = mapped
                    return super().do_GET()
        # "/" serves index.html (the hub page) via SimpleHTTPRequestHandler's
        # directory-index logic. No redirect — each card on the hub navigates
        # to its own destination (prototype.html, external URLs, etc.).

        # Source HTML loads /src/*.ts — only Vite can execute that. Prefer the built bundle on this server.
        if path_lc.rstrip("/") in ("/apps/ar-tryon", "/apps/ar-tryon/index.html"):
            self.send_response(302)
            self.send_header("Location", "/dist/apps/ar-tryon/index.html")
            self.end_headers()
            return
        # If the dev HTML was opened here anyway, map entry assets to the Rollup bundle (requires `npm run build`).
        if path_lc == "/src/apps/ar-tryon/main.ts":
            self.send_response(302)
            self.send_header("Location", "/dist/assets/ar-tryon.js")
            self.end_headers()
            return
        if path_lc == "/src/apps/ar-tryon/ar-tryon.css":
            self.send_response(302)
            self.send_header("Location", "/dist/assets/ar-tryon.css")
            self.end_headers()
            return

        if parsed.path == '/api/config':
            self.handle_config()
        elif parsed.path == '/api/health':
            self.handle_health()
        elif parsed.path == '/api/garments':
            self.handle_garment_list()
        elif parsed.path == '/api/server-info':
            self.handle_server_info()
        elif parsed.path.startswith('/api/status/'):
            task_id = parsed.path.split('/')[-1]
            self.handle_status(task_id)
        elif parsed.path.startswith('/api/garment-file/'):
            task_id = parsed.path.rstrip('/').split('/')[-1]
            self.handle_garment_file(task_id)
        elif parsed.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        elif parsed.path.startswith('/generated/'):
            self.serve_generated_file(parsed.path)
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == '/api/generate':
            self.handle_generate()
        else:
            self.send_error(404)
    
    def handle_config(self):
        """Return server configuration status"""
        server_dir = _SERVER_DIR
        
        canonical_path = (server_dir / CONFIG["GARMENT_CANONICAL_GLB"]).resolve()
        config_status = {
            "pix2surf_available": (server_dir / CONFIG["PIX2SURF_ROOT"]).is_dir(),
            "smpl_available": (server_dir / CONFIG["SMPL_MODEL"]).is_file(),
            "garment_obj_available": (server_dir / CONFIG["GARMENT_OBJ"]).is_file(),
            "garment_template_glb_available": (server_dir / CONFIG["GARMENT_TEMPLATE_GLB"]).is_file(),
            "garment_canonical_glb_available": canonical_path.is_file(),
            "export_rigid": str(CONFIG["GARMENT_EXPORT_RIGID"]) not in ("0", "false", "no", "off"),
            "texture_full_bleed": str(CONFIG["TEXTURE_FULL_BLEED"]) not in ("0", "false", "no", "off"),
            "blender_available": shutil.which(CONFIG["BLENDER_BIN"]) is not None or Path(CONFIG["BLENDER_BIN"]).is_file(),
            "script_available": (server_dir / CONFIG["SCRIPT_PATH"]).is_file(),
            "device": CONFIG["DEVICE"],
            "subdivide_iters": int(CONFIG["SUBDIVIDE_ITERS"]),
            "gen_timeout_sec": int(CONFIG["GEN_TIMEOUT_SEC"]),
            "prep_min_edge": int(CONFIG["PREP_MIN_EDGE"]),
            "texture_mode": str(CONFIG["TEXTURE_MODE"]),
            "bind_pose": str(CONFIG["BIND_POSE"]),
            "generation_mode": (
                "sam3_canonical_rigid"
                if canonical_path.is_file() and str(CONFIG["GARMENT_EXPORT_RIGID"]) not in ("0", "false", "no", "off")
                else "smpl_skinned"
            ),
            "paths": {
                "pix2surf": str((server_dir / CONFIG["PIX2SURF_ROOT"]).resolve()),
                "smpl": str((server_dir / CONFIG["SMPL_MODEL"]).resolve()),
                "garment": str((server_dir / CONFIG["GARMENT_OBJ"]).resolve()),
                "garment_template_glb": str((server_dir / CONFIG["GARMENT_TEMPLATE_GLB"]).resolve()),
                "garment_canonical_glb": str(canonical_path),
                "blender": CONFIG["BLENDER_BIN"],
                "script": str((server_dir / CONFIG["SCRIPT_PATH"]).resolve()),
            }
        }
        config_status["ready"] = all([
            config_status["pix2surf_available"],
            config_status["smpl_available"],
            config_status["garment_obj_available"],
            config_status["garment_canonical_glb_available"],
            config_status["blender_available"],
            config_status["script_available"],
        ])
        self.send_json(config_status)

    def handle_health(self):
        """Lightweight liveness check for the SPA (no path resolution work)."""
        self.send_json({
            "status": "ok",
            "time": time.time(),
            "api_version": API_VERSION,
            "garment_file_api": True,
            "glb_inline_max_bytes": GLB_INLINE_MAX_BYTES,
        })

    def handle_server_info(self):
        """Origins and page URLs for opening the site from phones on the LAN (and tunnel docs)."""
        cert_path, _ = _resolve_https_cert_paths()
        scheme = "https" if cert_path else "http"
        lan = _lan_ipv4()
        urls = _server_public_urls(scheme)
        # Vite uses @vitejs/plugin-basic-ssl — HTTPS on dev + preview so phone cameras work on LAN.
        lan_vite = f"https://{lan}:{TRYON_VITE_PORT}/apps/ar-tryon/" if lan else None
        self.send_json(
            {
                "scheme": scheme,
                "port": HTTP_PORT,
                "tryon_vite_port": TRYON_VITE_PORT,
                "lan_ipv4": lan,
                "https_enabled": bool(cert_path),
                "urls": {
                    "localhost_home": urls["localhost"]["home"],
                    "localhost_tryon": urls["localhost"]["tryon"],
                    "lan_home": urls.get("lan", {}).get("home"),
                    "lan_tryon": urls.get("lan", {}).get("tryon"),
                    "lan_tryon_vite": lan_vite,
                },
                "all": urls,
                "camera_hint": (
                    "`npm run dev` serves HTTPS (self-signed) on port "
                    f"{TRYON_VITE_PORT} so the phone camera works on your LAN IP. Tap Advanced / Continue if the "
                    "browser warns about the certificate. For static pages on this Python server only, use "
                    "HTTPS_CERT and HTTPS_KEY in .env or a tunnel — see docs/TUNNELS.md."
                ),
                "vite_hint": (
                    "For the TypeScript AR try-on, run `npm run dev` (HTTPS on all interfaces) and on your phone open "
                    f"https://<this-PC-LAN-IP>:{TRYON_VITE_PORT}/apps/ar-tryon/ — accept the certificate warning once. "
                    "Keep python server-local.py running on this PC for /api/generate. Windows: allow Node through the firewall."
                ),
            }
        )

    def handle_garment_list(self):
        """List completed GLBs under OUTPUT_DIR (for demo carousel / quick reload)."""
        server_dir = _SERVER_DIR
        output_dir = (server_dir / CONFIG["OUTPUT_DIR"]).resolve()
        items = []
        if output_dir.is_dir():
            for d in sorted(
                (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("task_")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                glb = d / "garment.glb"
                js = d / "garment.json"
                if glb.is_file():
                    items.append(
                        {
                            "id": d.name,
                            "glb_url": f"/generated/{d.name}/garment.glb",
                            "json_url": f"/generated/{d.name}/garment.json" if js.is_file() else None,
                            "mtime": int(glb.stat().st_mtime),
                        }
                    )
        self.send_json({"garments": items[:40]})

    def handle_generate(self):
        """Handle GLB generation request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            front_image = data.get('front_image')  # base64 data URI
            back_image = data.get('back_image')    # base64 data URI
            
            if not front_image or not back_image:
                self.send_json({'error': 'Both front_image and back_image are required'}, 400)
                return
            
            # Create task
            task_id = str(uuid.uuid4())[:8]
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'pending',
                    'progress': 0,
                    'message': 'Starting generation...',
                    'glb_path': None,
                    'json_path': None,
                    'error': None,
                    'hint': None,
                    'logs': [],
                    'started_at': time.time(),
                    'duration_sec': None,
                    'metrics': None,
                }
            
            # Start generation in background thread
            thread = threading.Thread(
                target=self.run_generation,
                args=(task_id, front_image, back_image)
            )
            thread.daemon = True
            thread.start()
            
            self.send_json({'task_id': task_id, 'status': 'started'})
            
        except Exception as e:
            traceback.print_exc()
            self.send_json({'error': str(e)}, 500)
    
    def run_generation(self, task_id, front_image, back_image):
        """Run the GLB generation pipeline in background"""
        def update_task(**kwargs):
            with tasks_lock:
                tasks[task_id].update(kwargs)

        def push_log(line: str) -> None:
            line = (line or "").rstrip("\n")
            if not line:
                return
            with tasks_lock:
                t = tasks.get(task_id)
                if not t:
                    return
                logs = t.setdefault("logs", [])
                logs.append(line)
                if len(logs) > 250:
                    del logs[: len(logs) - 250]

        def set_progress(pct: int, message: str | None = None) -> None:
            pct = int(max(0, min(100, pct)))
            with tasks_lock:
                cur = int(tasks[task_id].get("progress", 0))
            if pct < cur:
                pct = cur
            if message is None:
                update_task(progress=pct)
            else:
                update_task(progress=pct, message=message)

        def update_from_output(line: str) -> None:
            s = (line or "").strip()
            if not s:
                return
            print(f"[TASK {task_id}] {s}")
            push_log(s)

            if "[INFO] Pix2Surf:" in s:
                set_progress(34, "Pix2Surf: neural inference…")
                return
            wm = re.search(r"Weight transfer closest-point:\s*(\d+)/(\d+)", s, re.I)
            if wm:
                a, b = int(wm.group(1)), int(wm.group(2))
                subp = 60 + int(17 * min(1.0, a / max(b, 1)))
                tail = s.split("closest-point:", 1)[-1].strip()[:88]
                set_progress(subp, f"Skinning weights… {tail}")
                return

            stages: list[tuple[re.Pattern, int, str]] = [
                (re.compile(r"\[INFO\]\s+Starting garment pipeline", re.I), 32, "Pipeline started…"),
                (re.compile(r"\[INFO\]\s+Building garment texture", re.I), 36, "Building photo texture…"),
                (re.compile(r"\[rembg\].*Background removed", re.I), 18, "Preprocessing images…"),
                (re.compile(r"\[GrabCut\].*Background removed", re.I), 18, "Preprocessing images…"),
                (re.compile(r"Direct atlas brightness", re.I), 42, "Mapping photos to UV atlas…"),
                (re.compile(r"\bTexture brightness \(Pix2Surf\)", re.I), 38, "Pix2Surf: generating texture…"),
                (re.compile(r"\[OK\] Wrote upper texture:", re.I), 45, "Texture ready."),
                (re.compile(r"\[OK\] SAM3-style canonical mesh", re.I), 52, "Loaded shirt mesh…"),
                (re.compile(r"\[OK\] Wrote rigid garment metadata", re.I), 78, "Writing try-on metadata…"),
                (re.compile(r"Retexturing canonical GLB", re.I), 84, "Applying photo texture to shirt mesh…"),
                (re.compile(r"\[OK\] Exported rigid textured GLB", re.I), 92, "GLB exported. Finalizing…"),
                (re.compile(r"\[OK\] Exported template-based GLB", re.I), 92, "GLB exported. Finalizing…"),
                (re.compile(r"\[DONE\] Rigid GLB", re.I), 95, "Finalizing output…"),
                (re.compile(r"\[OK\] Loaded SMPL:", re.I), 55, "Loading SMPL model…"),
                (re.compile(r"\[OK\] Loaded garment OBJ:", re.I), 60, "Loading garment mesh…"),
                (re.compile(r"Building KD-tree", re.I), 65, "Transferring skinning weights…"),
                (re.compile(r"\[OK\] Surface weight transfer complete", re.I), 72, "Skinning weights transferred."),
                (re.compile(r"\[OK\] Smoothed garment weights", re.I), 76, "Smoothing weights…"),
                (re.compile(r"\[OK\] Wrote skin package:", re.I), 80, "Preparing Blender export…"),
                (re.compile(r"\[OK\] Wrote pose metadata:", re.I), 84, "Writing pose metadata…"),
                (re.compile(r"^\[CMD\]\s+.*\bblender\b", re.I), 86, "Running Blender to export GLB…"),
                (re.compile(r"\[OK\] Exported skinned GLB:", re.I), 92, "GLB exported. Finalizing…"),
                (re.compile(r"\[DONE\] Skinned GLB written to:", re.I), 95, "Finalizing output…"),
            ]

            for pat, pct, msg in stages:
                if pat.search(s):
                    set_progress(pct, msg)
                    return

            if s.startswith("[WARN]") or s.startswith("[ERROR]"):
                update_task(message=s[:180])
        
        # Get server directory
        server_dir = _SERVER_DIR
        
        try:
            # Create output directory
            output_dir = (server_dir / CONFIG["OUTPUT_DIR"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create work directory for this task
            work_dir = output_dir / f"task_{task_id}"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            update_task(status='processing', progress=10, message='Saving uploaded images...', logs=[])
            
            # Save images from base64
            front_path = work_dir / "front.png"
            back_path = work_dir / "back.png"
            
            self.save_base64_image(front_image, front_path)
            self.save_base64_image(back_image, back_path)
            
            # ── Smart background removal (preserves white garment areas) ──
            front_proc = work_dir / "front_proc.png"
            back_proc = work_dir / "back_proc.png"
            set_progress(12, 'Removing background (front)…')
            pre_front = self.preprocess_garment_image(front_path, front_proc)
            set_progress(16, 'Removing background (back)…')
            pre_back = self.preprocess_garment_image(back_path, back_proc)
            gp_meta = {"front": pre_front, "back": pre_back, "task_id": task_id}
            (work_dir / "garment_preprocess.json").write_text(
                json.dumps(gp_meta, indent=2, default=str),
                encoding="utf-8",
            )
            push_log(f"garment_preprocess.json → fg_bbox front={pre_front.get('fg_bbox')} back={pre_back.get('fg_bbox')}")

            print(f"[TASK {task_id}] Saved images to {work_dir}")
            push_log(f"Saved images to {work_dir}")
            
            set_progress(20, 'Starting generation pipeline…')

            out_glb = work_dir / "garment.glb"
            out_json = work_dir / "garment.json"

            paths = get_generation_paths()
            rigid_fast = (
                paths["has_canonical_glb"]
                and paths["export_rigid"]
                and str(CONFIG["TEXTURE_MODE"]).strip().lower() == "direct"
            )

            set_progress(25, 'Launching 3D pipeline…')

            cmd = [
                sys.executable,
                "-u",
                str(paths["script_path"]),
                "--pix2surf_root", str(paths["pix2surf_root"]),
                "--front_img", str(front_proc),
                "--back_img", str(back_proc),
                "--smpl_model", str(paths["smpl_model"]),
                "--garment_obj", str(paths["garment_obj"]),
                "--blender_bin", CONFIG["BLENDER_BIN"],
                "--out_glb", str(out_glb),
                "--device", CONFIG["DEVICE"],
                "--center",
                "--subdivide_iters", str(CONFIG["SUBDIVIDE_ITERS"]),
                "--prep_min_edge", str(CONFIG["PREP_MIN_EDGE"]),
                "--texture_mode", str(CONFIG["TEXTURE_MODE"]),
                "--bind_pose", str(CONFIG["BIND_POSE"]),
                "--workdir", str(work_dir / "_work"),
                "--skip_normalize",
            ]
            if not rigid_fast:
                cmd.extend(["--smooth_weights", "--smooth_iters", str(CONFIG["SMOOTH_ITERS"])])
            if paths["has_template_glb"]:
                cmd.extend(["--garment_template_glb", str(paths["garment_template_glb"])])
            if paths["has_canonical_glb"]:
                cmd.extend(["--garment_canonical_glb", str(paths["garment_canonical_glb"])])

            print(f"[TASK {task_id}] Running command:")
            print(f"  {' '.join(cmd)}")

            utf8_env = os.environ.copy()
            utf8_env["PYTHONIOENCODING"] = "utf-8"
            utf8_env["PYTHONUTF8"] = "1"
            utf8_env["PYTHONUNBUFFERED"] = "1"
            utf8_env["TEXTURE_MODE"] = str(CONFIG["TEXTURE_MODE"])
            utf8_env["GARMENT_TILE_ALIGN_TEMPLATE"] = str(CONFIG["TILE_ALIGN_TEMPLATE"])
            utf8_env["GARMENT_TILE_ALIGN_MODE"] = str(CONFIG["TILE_ALIGN_MODE"])
            utf8_env["GARMENT_TILE_FILL_FRAC"] = str(CONFIG["TILE_FILL_FRAC"])
            utf8_env["GARMENT_TILE_SHOULDER_FRAC"] = str(CONFIG["TILE_SHOULDER_FRAC"])
            utf8_env["GARMENT_TILE_HEM_FRAC"] = str(CONFIG["TILE_HEM_FRAC"])
            utf8_env["GARMENT_TEXTURE_FULL_BLEED"] = str(CONFIG["TEXTURE_FULL_BLEED"])
            utf8_env["GARMENT_EXPORT_RIGID"] = str(CONFIG["GARMENT_EXPORT_RIGID"])
            if paths["has_canonical_glb"]:
                utf8_env["GARMENT_CANONICAL_GLB"] = str(paths["garment_canonical_glb"])

            gen_timeout = max(60, int(CONFIG["GEN_TIMEOUT_SEC"]))
            start_time = time.time()
            heartbeat_stop = threading.Event()

            def _heartbeat() -> None:
                while not heartbeat_stop.wait(12.0):
                    with tasks_lock:
                        tk = tasks.get(task_id)
                        if not tk or tk.get("status") not in ("pending", "processing"):
                            return
                        st = tk.get("started_at")
                    if st is None:
                        continue
                    elapsed = int(time.time() - float(st))
                    with tasks_lock:
                        tk2 = tasks.get(task_id)
                        cur_pct = int(tk2.get("progress", 30)) if tk2 else 30
                    # Nudge % while subprocess warms up (imports / Blender) with few log lines.
                    if 20 <= cur_pct < 88:
                        cur_pct = min(87, cur_pct + 1)
                    update_task(
                        progress=cur_pct,
                        message=(
                            f"Running pipeline… {elapsed}s elapsed — "
                            "live lines appear in the server terminal; "
                            f"timeout {gen_timeout}s"
                        ),
                    )

            set_progress(30, 'Running 3D pipeline…')

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()
            proc = None
            rc: int | None = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    cwd=str(server_dir),
                    env=utf8_env,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    update_from_output(line)
                    if time.time() - start_time > gen_timeout:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=gen_timeout)

                rc = proc.wait(timeout=5)
            finally:
                heartbeat_stop.set()

            if proc is None:
                raise RuntimeError("subprocess failed to start")
            if rc is not None and rc != 0:
                raise Exception(f"Generation failed (exit code {rc}). See logs for details.")
            
            set_progress(90, 'Finalizing…')
            
            glb_size = _wait_for_garment_file(out_glb)

            # Optional metrics produced by tshirt_to_glb_pose_ready.py (SSIM/LPIPS).
            metrics = None
            try:
                # The generator runs with --workdir <task_dir>/_work, and writes
                # texture + metrics next to upper_tex.jpg in that work directory.
                mp = (work_dir / "_work" / "metrics.json")
                if mp.is_file():
                    metrics = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
            
            glb_url = f"/generated/task_{task_id}/garment.glb"
            glb_api_url = f"/api/garment-file/{task_id}"
            json_url = f"/generated/task_{task_id}/garment.json" if out_json.exists() else None

            glb_inline = None
            if glb_size <= GLB_INLINE_MAX_BYTES:
                try:
                    glb_inline = base64.b64encode(out_glb.read_bytes()).decode("ascii")
                except Exception as e:
                    print(f"[TASK {task_id}] inline GLB base64 skipped: {e}")

            with tasks_lock:
                t0 = float(tasks[task_id].get('started_at', time.time()))
            duration_sec = round(time.time() - t0, 2)
            update_task(
                progress=100,
                status='completed',
                message='Generation complete!',
                glb_path=str(out_glb),
                json_path=str(out_json) if out_json.exists() else None,
                glb_url=glb_url,
                glb_api_url=glb_api_url,
                glb_bytes=glb_size,
                glb_inline=glb_inline,
                json_url=json_url,
                duration_sec=duration_sec,
                metrics=metrics,
            )
            
            print(f"[TASK {task_id}] ✓ Complete: {out_glb}")
            
        except subprocess.TimeoutExpired:
            err = f'Generation timed out after {max(60, int(CONFIG["GEN_TIMEOUT_SEC"]))} seconds'
            with tasks_lock:
                t0 = float(tasks[task_id].get('started_at', time.time()))
            duration_sec = round(time.time() - t0, 2)
            update_task(
                status='failed',
                error=err,
                message='Error: Timeout',
                hint=_error_recovery_hint(err),
                duration_sec=duration_sec,
            )
            print(f"[TASK {task_id}] ✗ Timeout")
        except Exception as e:
            traceback.print_exc()
            err = str(e)
            with tasks_lock:
                t0 = float(tasks[task_id].get('started_at', time.time()))
            duration_sec = round(time.time() - t0, 2)
            update_task(
                status='failed',
                error=err,
                message=f'Error: {err[:100]}',
                hint=_error_recovery_hint(err),
                duration_sec=duration_sec,
            )
            print(f"[TASK {task_id}] ✗ Error: {e}")
    
    def save_base64_image(self, data_uri, path):
        """Save a base64 data URI to file"""
        # Remove data URI prefix if present
        if ',' in data_uri:
            data_uri = data_uri.split(',')[1]
        
        # Decode and save
        image_data = base64.b64decode(data_uri)
        with open(path, 'wb') as f:
            f.write(image_data)
        
        print(f"  Saved {len(image_data)} bytes to {path}")

    def preprocess_garment_image(self, src_path, dst_path):
        """
        Delegates to pipeline.garment_preprocess (rembg → GrabCut → passthrough).
        Returns metadata dict including fg_bbox (xywh) for print-region heuristics.
        """
        from pathlib import Path

        from pipeline.garment_preprocess import preprocess_garment_product_shot

        return preprocess_garment_product_shot(Path(src_path), Path(dst_path))
    
    def handle_status(self, task_id):
        """Return task status"""
        with tasks_lock:
            if task_id not in tasks:
                self.send_json({'error': 'Task not found'}, 404)
                return
            
            task = tasks[task_id].copy()
        
        st = task.get('started_at')
        elapsed_sec = None
        if st is not None:
            ds = task.get('duration_sec')
            if ds is not None:
                elapsed_sec = float(ds)
            else:
                elapsed_sec = round(time.time() - float(st), 1)

        response = {
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'message': task['message'],
            'logs': task.get('logs', [])[-50:],
            'elapsed_sec': elapsed_sec,
            'metrics': task.get('metrics'),
        }
        
        if task['status'] == 'completed':
            response['glb_url'] = task.get('glb_url')
            response['glb_api_url'] = task.get('glb_api_url')
            response['glb_bytes'] = task.get('glb_bytes')
            response['json_url'] = task.get('json_url')
            if task.get('glb_inline'):
                response['glb_base64'] = task.get('glb_inline')
        elif task['status'] == 'failed':
            response['error'] = task['error']
            response['hint'] = task.get('hint') or _error_recovery_hint(task.get('error') or '')
        
        self.send_json(response)

    def handle_garment_file(self, task_id: str) -> None:
        """
        Chunked GLB download for ngrok / mobile tunnels.
        GET /api/garment-file/<task_id>?offset=0&limit=196608
        """
        path = _garment_glb_path(task_id)
        if not path.is_file():
            self.send_json({'error': 'Garment not found', 'task_id': task_id}, 404)
            return

        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        try:
            offset = int((qs.get('offset') or ['0'])[0])
        except ValueError:
            offset = 0
        try:
            limit = int((qs.get('limit') or ['196608'])[0])
        except ValueError:
            limit = 196608
        limit = max(4096, min(limit, 512 * 1024))
        offset = max(0, offset)

        try:
            total = int(path.stat().st_size)
        except OSError:
            self.send_json({'error': 'Garment unreadable'}, 500)
            return

        if offset >= total:
            chunk = b""
        else:
            with open(path, 'rb') as f:
                f.seek(offset)
                chunk = f.read(min(limit, total - offset))

        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(chunk)))
        self.send_header('X-Garment-Total', str(total))
        self.send_header('X-Garment-Offset', str(offset))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        if chunk:
            self.wfile.write(chunk)
    
    def serve_generated_file(self, path):
        """Serve files from generated directory"""
        # Remove /generated/ prefix
        rel_path = path[11:]  # len('/generated/') = 11
        
        # Use server_dir to find the generated folder
        server_dir = getattr(self.__class__, 'server_dir', os.path.dirname(os.path.abspath(__file__)))
        full_path = Path(server_dir) / CONFIG["OUTPUT_DIR"] / rel_path
        
        # Security check - ensure path is within output directory
        output_base = (Path(server_dir) / CONFIG["OUTPUT_DIR"]).resolve()
        try:
            full_path.resolve().relative_to(output_base)
        except ValueError:
            self.send_error(403, 'Access denied')
            return
        
        if not full_path.exists():
            self.send_error(404, f'File not found: {rel_path}')
            return
        
        # Determine content type
        ext = full_path.suffix.lower()
        content_types = {
            # octet-stream is more reliable through ngrok/HTTP2 tunnels than model/gltf-binary
            '.glb': 'application/octet-stream',
            '.gltf': 'model/gltf+json',
            '.json': 'application/json',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.obj': 'text/plain',
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        
        # Send file (supports Range: bytes= for ngrok-friendly chunked downloads)
        try:
            file_size = int(full_path.stat().st_size)
            range_header = self.headers.get('Range')
            start = 0
            end = file_size - 1
            status = 200
            if range_header:
                m = re.match(r'bytes=(\d*)-(\d*)', range_header.strip())
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                    if m.group(2):
                        end = int(m.group(2))
                    else:
                        end = file_size - 1
                    start = max(0, min(start, file_size - 1))
                    end = max(start, min(end, file_size - 1))
                    status = 206

            length = end - start + 1
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Accept-Ranges', 'bytes')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            # Helps some tunnels/proxies avoid HTTP/2 stream resets on large GLBs.
            if ext == '.glb':
                self.send_header('Connection', 'close')
            self.end_headers()

            with open(full_path, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except Exception as e:
            self.send_error(500, str(e))
    
    def send_json(self, data, status=200):
        """Send JSON response"""
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)
    
    def log_message(self, format, *args):
        """Custom log format - reduce noise"""
        try:
            if args:
                message = format % args
            else:
                message = format
            
            message = str(message)
            
            # Skip noisy status polling logs
            if '/api/status/' in message and '200' in message:
                return
            if 'Bad request' in message or '400' in message:
                return
            
            print(f"[HTTP] {message}")
        except Exception:
            pass


def check_config():
    """Check if configuration is valid and print status"""
    print("\n📋 Checking configuration...")
    print(f"  ℹ .env loaded from: {_SERVER_DIR / '.env'}")

    server_dir = _SERVER_DIR
    issues = []

    try:
        gp = get_generation_paths()
        mode = "direct+rigid (fast)" if (
            gp.get("has_canonical_glb") and gp.get("export_rigid")
            and str(CONFIG["TEXTURE_MODE"]).strip().lower() == "direct"
        ) else "standard"
        print(f"  ✓ Generation paths cached — pipeline mode: {mode}")
    except Exception as e:
        issues.append(f"  ❌ Could not resolve generation paths: {e}")

    pix2surf_path = (server_dir / CONFIG["PIX2SURF_ROOT"]).resolve()
    if pix2surf_path.is_dir():
        print(f"  ✓ PIX2SURF_ROOT: {pix2surf_path}")
        if (pix2surf_path / "pretrained").is_dir():
            print(f"    ✓ pretrained/ found")
        else:
            issues.append(f"    WARNING pretrained/ not found in {pix2surf_path}")
    else:
        issues.append(f"  ❌ PIX2SURF_ROOT not found: {pix2surf_path}")

    smpl_path = (server_dir / CONFIG["SMPL_MODEL"]).resolve()
    if smpl_path.is_file():
        print(f"  ✓ SMPL_MODEL: {smpl_path}")
    else:
        issues.append(f"  ❌ SMPL_MODEL not found: {smpl_path}")

    garment_path = (server_dir / CONFIG["GARMENT_OBJ"]).resolve()
    if garment_path.is_file():
        print(f"  ✓ GARMENT_OBJ: {garment_path}")
    else:
        issues.append(f"  ❌ GARMENT_OBJ not found: {garment_path}")

    garment_template = (server_dir / CONFIG["GARMENT_TEMPLATE_GLB"]).resolve()
    if garment_template.is_file():
        print(f"  ✓ GARMENT_TEMPLATE_GLB: {garment_template}")
    else:
        print(f"  ℹ GARMENT_TEMPLATE_GLB not found (optional): {garment_template}")

    garment_canonical = (server_dir / CONFIG["GARMENT_CANONICAL_GLB"]).resolve()
    if garment_canonical.is_file():
        print(f"  ✓ GARMENT_CANONICAL_GLB (SAM3 shape): {garment_canonical}")
    else:
        issues.append(f"  ❌ GARMENT_CANONICAL_GLB not found: {garment_canonical}")
    print(f"  ℹ GARMENT_EXPORT_RIGID: {CONFIG['GARMENT_EXPORT_RIGID']}")
    print(f"  ℹ TEXTURE_FULL_BLEED: {CONFIG['TEXTURE_FULL_BLEED']}")

    blender_val = CONFIG["BLENDER_BIN"]
    if shutil.which(blender_val) is not None or Path(blender_val).is_file():
        print(f"  ✓ BLENDER_BIN: {blender_val}")
    else:
        issues.append(f"  ❌ BLENDER_BIN not found: {blender_val}")

    script_path = (server_dir / CONFIG["SCRIPT_PATH"]).resolve()
    if script_path.is_file():
        print(f"  ✓ SCRIPT_PATH: {script_path}")
    else:
        issues.append(f"  ❌ SCRIPT_PATH not found: {script_path}")

    print(f"  ℹ DEVICE:     {CONFIG['DEVICE']}")
    print(f"  ℹ PREP_MIN_EDGE: {CONFIG['PREP_MIN_EDGE']}")
    print(f"  ℹ TEXTURE_MODE: {CONFIG['TEXTURE_MODE']}")
    print(f"  ℹ OUTPUT_DIR: {(server_dir / CONFIG['OUTPUT_DIR']).resolve()}")

    if issues:
        print("\n⚠️  Configuration issues found:")
        for issue in issues:
            print(issue)
        print("\n  Fix paths in .env (same folder as server-local.py).")
        return False
    return True


def run_server():
    """Run the HTTP server"""
    static_dir = (_SERVER_DIR / CONFIG["STATIC_DIR"]).resolve()

    if static_dir.is_dir():
        os.chdir(static_dir)
        print(f"📂 Serving static files from: {static_dir}")
    else:
        os.chdir(_SERVER_DIR)
        print(f"📂 Serving from: {_SERVER_DIR}")
        print(f"  (STATIC_DIR={static_dir} not found — falling back to script folder)")

    html_files = sorted(f for f in os.listdir('.') if f.endswith('.html'))
    if html_files:
        print(f"📄 HTML files: {', '.join(html_files)}")
    else:
        print("⚠️  No HTML files found in directory")

    output_dir = _SERVER_DIR / CONFIG["OUTPUT_DIR"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Output directory: {output_dir.resolve()}")

    APIHandler.server_dir = str(_SERVER_DIR)

    httpd = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), APIHandler)
    cert_path, key_path = _resolve_https_cert_paths()
    scheme = "https" if cert_path else "http"
    if cert_path and key_path:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"🔒 HTTPS enabled ({cert_path.name} / {key_path.name})")

    # "/" serves the hub (index.html). The prototype lives at /prototype.html
    # and launches the built AR try-on under /dist/apps/ar-tryon/.
    lan = _lan_ipv4()
    print(f"\n🚀 Server ({scheme}) — listening on 0.0.0.0:{HTTP_PORT} (all interfaces)")
    print(f"   This machine:  {scheme}://localhost:{HTTP_PORT}/   ← hub (Google 2D · WANNA · Prototype)")
    print(f"   Prototype:     {scheme}://localhost:{HTTP_PORT}/prototype.html")
    print(f"   AR try-on:     {scheme}://localhost:{HTTP_PORT}/dist/apps/ar-tryon/index.html")
    if lan:
        print(f"   Phone (LAN):   {scheme}://{lan}:{HTTP_PORT}/   ← same Wi‑Fi")
        print(
            f"   AR try-on dev: https://{lan}:{TRYON_VITE_PORT}/apps/ar-tryon/  ← `npm run dev` (HTTPS for phone camera)"
        )
    print(f"   Server info:   {scheme}://localhost:{HTTP_PORT}/api/server-info")
    print(
        "   API:  GET /api/garments — list GLBs · POST /api/generate — queue a job\n"
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       AR Try-On Server with Local GLB Generation             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📁 Page:  http://localhost:8000/  (AR try-on only)          ║
║  🔧 API Endpoints:                                           ║
║     GET  /api/health      - Liveness ping                    ║
║     GET  /api/config      - Server configuration status      ║
║     POST /api/generate    - Start GLB generation             ║
║     GET  /api/status/X    - Poll a generation task           ║
║     GET  /api/garments    - List saved GLBs (library)        ║
║     GET  /api/garment-file/X - Chunked GLB (ngrok-safe)     ║
║     GET  /api/server-info - LAN URLs + mobile camera hints   ║
║     GET  /generated/*     - Download generated files         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    config_ok = check_config()
    
    if config_ok:
        print("\n✅ Configuration OK - All dependencies found!")
    else:
        print("\n⚠️  Server will start but generation may fail until paths are fixed.")
    
    run_server()
