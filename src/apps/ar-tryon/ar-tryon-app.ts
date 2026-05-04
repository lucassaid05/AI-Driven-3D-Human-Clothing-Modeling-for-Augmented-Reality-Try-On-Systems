// @ts-nocheck
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
    smoothPoseLandmarks,
    lerpAngle,
} from './pose-utils';
import {
    drawHudOverlay,
    drawSkeletonOverlay,
    drawMpMappingLayer
} from './debug-overlay';
import {
    syncLogLandmarkIntervalFromUI,
    initSessionLog,
    downloadSessionLogJson,
    maybeAppendSessionLogEntry
} from './session-log';

/**
 * Global multiplier applied to the auto-computed uniform scale just before it
 * reaches `garmentGroup.scale`. The shoulder-bone distance of SMPL-derived
 * garments is slightly smaller than the visible shoulder geometry, so matching
 * bone-to-bone span against MediaPipe's shoulder span over-sizes the shirt.
 * 0.92 = 8% global trim, which visually lines the silhouette up with the
 * user's real frame without distorting landmark-anchored placement math.
 */
const AR_TRYON_SCALE_TRIM = 0.92;

export class ARTryOn {
    constructor() {
        this.video = document.getElementById('webcam-video');
        this.threeCanvas = document.getElementById('three-canvas');
        this.debugCanvas = document.getElementById('debug-canvas');
        this.debugCtx = this.debugCanvas.getContext('2d');
        
        this.isRunning = false;
        this.garmentModel = null;
        this.garmentGroup = null;
        this.skinnedMesh = null;
        /** Non-skinned meshes (e.g. SAM3 shirt_canonical.glb) — same materials/orientation rules as skinned. */
        this._rigidMeshes = [];
        
        // Garment measurements from GLB
        this.garmentShoulderWidth = null;
        /** Denominator for auto scale: ≥ shoulder-bone distance; widened when bind mesh is T-pose (bbox.x ≫ bones). */
        this.garmentScaleRefWidth = null;
        this.garmentTorsoHeight = null;
        /** Midpoint between shoulder bones in garmentGroup local space (bind pose); for alignment */
        this.garmentBindShoulderMid = null;
        /** Torso center in garmentGroup local space (bind pose): midpoint between shoulder and hip lines */
        this.garmentBindBodyCenter = null;
        /** Bottom-center of garment AABB in garmentGroup local space (hem ↔ MP 23/24) */
        this.garmentBindHemMid = null;
        /** |shoulderMid.y − hemMid.y| in bind local (for torso-length scale) */
        this._garmentShoulderHemLocalDy = 0.35;
        /** garment.json from pipeline (mediapipe_mapping, calibration, …) */
        this.garmentMetadata = null;
        this._lastJsonUrl = null;
        this.sessionLog = null;
        this._lastSessionLogTs = 0;
        this._lastLandmarkLogWall = 0;
        this._logPrevShoulderMid = null;
        this._logJitterEma = null;
        this._lastRenderMs = 0;
        this._sessionGarmentLabel = null;
        this._lastLatencyMs = 0;
        this._poseFrameTick = 0;
        this._lastRawLandmarks = null;
        
        // Skeleton
        this.skeleton = null;
        this.bones = {};

// Tracking state
        this.smoothedPose = null;
        /** Body span ÷ garment shoulder width; updated continuously from pose (smoothed). */
        this.fitStructuralMul = null;
        this.smoothedScale = null;
        this.smoothedPosition = new THREE.Vector3();
        this.smoothedRotation = 0;
        /** Side-profile x anchor lock ('left'|'right') to prevent depth-noise shoulder flips. */
        this._profileAnchorSide = null;
        /** Last stable torso landmarks for one-limb occlusion fallback. */
        this._lastReliableTorso = null;
        /** Hide garment when tracking is unreliable */
        this._garmentTrackingOk = true;
        
        // Debug values
        this.debugInfo = {
            shoulderWidth: 0,
            torsoHeight: 0,
            effectiveWidth: 0,
            viewAngle: 'front',
            computedScale: 0,
            clampedScale: 0,
            fitStructuralMul: null,
            garmentHidden: false,
            roll: 0,
            pitch: 0,
            zLean: 0,
            facingAway: false,
            faceFacingSource: 'legacy',
            occlusionAssist: false
        };
        
        // Settings
        this.settings = {
            offsetY: -0.15,         // UI slider provides the real initial value
            offsetX: 0,
            rotationY: 0,
            /**
             * Auto shoulder anchor drop (fraction of shoulder→hip distance in world units).
             * MediaPipe shoulders tend to sit slightly above the visible seam where a shirt rests.
             */
            autoShoulderDrop: 0.075,
            /** Final multiplier applied to the auto-computed uniform scale. Lower = smaller garment. */
            autoScaleTrim: 0.92,
            /**
             * Non-uniform X squeeze applied to the *final* garment scale.
             * Helps reduce the "shoulder pad / armour" look when the garment's skeleton shoulder width is
             * slightly narrower than the visible mesh at the sleeves.
             */
            autoShoulderSqueezeX: 0.94,
            frontFaceOnly: true,
            /**
             * Pix2Surf / many SMPL shirt GLBs show the back texture when Three.js uses FrontSide toward the camera.
             * When true, we swap FrontSide/BackSide relative to facingAway for FRONT view (single-sided mode).
             */
            garmentCullInvert: true,
            showSkeleton: false,
            /** true = selfie mirror (default): video CSS flip + camera scale so garment tracks you; false = direct cam / HUD text as captured */
            mirrorView: true,
            /** When true, nose/eyes depth vs shoulders can force shirt FRONT; legacy shoulder hysteresis stays as fallback. */
            useFaceFacingHint: true,
            wireframe: false,
            smoothingFactor: 0.4,
            showHud: true,
            logSession: false,
            /** With session log: store throttled full pose (33×[x,y,z,vis]) — larger files. */
            logIncludeLandmarks: false,
            /** Min milliseconds between landmark snapshots in the log. */
            logLandmarkIntervalMs: 2000,
            showMpMapping: false,
            /** Run MediaPipe detect every N video frames (reuse landmarks in between) — big FPS win */
            poseDetectEvery: 2
        };
        
        // MediaPipe indices (33-landmark pose)
        this.MP = {
            NOSE: 0,
            LEFT_EYE: 2,
            RIGHT_EYE: 5,
            LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
            LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
            LEFT_WRIST: 15, RIGHT_WRIST: 16,
            LEFT_HIP: 23, RIGHT_HIP: 24
        };
        
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();
        
        // Stabilise front vs back (hysteresis) — stops material/yaw flicker at the boundary
        this._hystFacingAway = false;
        this._lastMatFacingAway = null;
        this._lastMatState = '';
        /** Smoothed front/back score (negative=front, positive=back). Helps when MediaPipe swaps L/R briefly. */
        this._facingAwayScore = 0;
        /** Scale reference captured from a stable front view to prevent side-view shrink. */
        this._lastFrontScale = null;

        /** JPEG-normalized garment photos (iOS HEIC / library quirks); used for preview + POST body */
        this._genFrontFile = null;
        this._genBackFile = null;
        this._genFrontPreviewUrl = null;
        this._genBackPreviewUrl = null;
        this._mobileHudDefaultApplied = false;

        // Latest generation result (shown to the user; auto-applied once ready)
        this._pendingGeneratedGlbUrl = null;
        this._pendingGeneratedJsonUrl = null;
        this._presetWarmupStarted = false;
        this._generationInProgress = false;
        this._autoGenTimer = null;
        
        this.setupThreeJS();
        this.setupMediaPipe();
        this.setupUI();
        this.applyMirrorViewTransforms();
        window.addEventListener('resize', () => this.onResize());
        void this._probeApiHealth();
        void this._maybeLoadGarmentFromQueryString();
    }

    _scheduleAutoGenerate() {
        // Auto-run generation once both images exist. Debounced to avoid double-trigger on mobile pick events.
        if (this._autoGenTimer) {
            try { clearTimeout(this._autoGenTimer); } catch (_) { /* ignore */ }
            this._autoGenTimer = null;
        }
        const haveFront = this._genFrontFile != null;
        const haveBack = this._genBackFile != null;
        if (!haveFront || !haveBack) return;
        if (this._generationInProgress) return;
        if (this._pendingGeneratedGlbUrl) return; // already have a result staged
        this._autoGenTimer = setTimeout(() => {
            this._autoGenTimer = null;
            if (this._generationInProgress) return;
            if (!this._genFrontFile || !this._genBackFile) return;
            void this.generateGarment();
        }, 250);
    }

    setMobileControlsOpen(open) {
        const appContainer = document.querySelector('.app-container');
        const mobileToggleBtn = document.getElementById('mobile-controls-toggle');
        const isMobileLayout = typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;
        const shouldOpen = Boolean(open && isMobileLayout);
        appContainer?.classList.toggle('mobile-panel-open', shouldOpen);
        if (mobileToggleBtn) {
            mobileToggleBtn.textContent = shouldOpen ? 'Hide controls' : 'Show controls';
            mobileToggleBtn.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
        }
    }

    /**
     * Bookmarkable deep link: ?glb=/generated/task_xxx/garment.glb&json=/generated/.../garment.json
     */
    _maybeLoadGarmentFromQueryString() {
        try {
            const u = new URL(location.href);
            let glb = (u.searchParams.get('glb') || '').trim();
            if (!glb) return;
            const jsonRel = (u.searchParams.get('json') || '').trim();
            const glbAbs = glb.startsWith('http') ? glb : this.getApiOrigin() + (glb.startsWith('/') ? glb : '/' + glb);
            const jsonAbs = jsonRel
                ? (jsonRel.startsWith('http') ? jsonRel : this.getApiOrigin() + (jsonRel.startsWith('/') ? jsonRel : '/' + jsonRel))
                : null;
            this._lastGlbUrl = glb.startsWith('http') ? glb : (glb.startsWith('/') ? glb : '/' + glb);
            this._lastJsonUrl = jsonRel ? (jsonRel.startsWith('http') ? jsonRel : (jsonRel.startsWith('/') ? jsonRel : '/' + jsonRel)) : null;
            this.showToast('Loading garment from link…', 'info');
            void this.loadGarmentFromUrl(glbAbs, jsonAbs);
        } catch (e) {
            console.warn('Query-string garment load:', e);
        }
    }

    /** Same host as this page — works on phone via http://LAN-IP:PORT/ */
    getApiOrigin() {
        return (typeof location !== 'undefined' && location.origin) ? location.origin : '';
    }

    /** Phones / tablets: shrink uploads — huge base64 JSON often makes mobile Safari fail the POST silently. */
    _isCoarseOrMobileDevice() {
        try {
            if (typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches) return true;
            const ua = typeof navigator !== 'undefined' ? (navigator.userAgent || '') : '';
            if (/iPhone|iPad|iPod|Android|Mobile|webOS|BlackBerry/i.test(ua)) return true;
        } catch (_) { /* ignore */ }
        return false;
    }

    /**
     * Cap longest edge + JPEG re-encode so /api/generate JSON stays small enough for mobile browsers.
     */
    async _downscaleForUpload(file, maxEdge, quality = 0.82) {
        if (!file || !file.size) return file;
        try {
            let bmp;
            try {
                bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
            } catch (_) {
                bmp = await createImageBitmap(file);
            }
            try {
                let w = bmp.width;
                let h = bmp.height;
                const scale = Math.min(1, maxEdge / Math.max(w, h, 1));
                w = Math.max(1, Math.round(w * scale));
                h = Math.max(1, Math.round(h * scale));
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                if (!ctx) return file;
                ctx.drawImage(bmp, 0, 0, w, h);
                const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
                if (!blob || blob.size < 64) return file;
                return new File([blob], 'upload.jpg', { type: 'image/jpeg', lastModified: Date.now() });
            } finally {
                if (typeof bmp.close === 'function') bmp.close();
            }
        } catch (_) {
            return file;
        }
    }

    /**
     * Normal view: no CSS flip on canvases → HUD text and mesh textures read correctly.
     * Mirror on: flip video (CSS) + orthographic camera scale.x = -1 so overlay matches selfie feed
     * without negative scale on garmentGroup (which breaks bone quaternions).
     *
     * IMPORTANT: Do NOT also apply 1-x to landmarks in poseToWorld / poseToScreen — that double-flips
     * horizontal motion vs the camera (shirt slides opposite to the user). One flip: camera + raw MP x.
     * Body yaw/roll/pitch from landmarks are sign-corrected in updateGarment when mirror is on (see yawSign / estimatedPitch / estimatedRoll).
     */
    applyMirrorViewTransforms() {
        const a = this.settings.mirrorView;
        if (this.video) this.video.style.transform = a ? 'scaleX(-1)' : 'none';
        if (this.threeCanvas) this.threeCanvas.style.transform = 'none';
        if (this.debugCanvas) this.debugCanvas.style.transform = 'none';
        if (this.camera) {
            this.camera.scale.set(a ? -1 : 1, 1, 1);
            this.camera.updateMatrixWorld(true);
        }
    }

    async _probeApiHealth() {
        try {
            const r = await fetch(this.getApiOrigin() + '/api/health', { method: 'GET' });
            if (!r.ok) throw new Error(String(r.status));
            this._apiServerReachable = true;
        } catch {
            this._apiServerReachable = false;
            this.showToast(
                'Generation server not detected — run server-local.py on this host to use Generate Garment.',
                'warning'
            );
        }
    }

    _formatConfigMissing(cfg) {
        const miss = [];
        if (!cfg.pix2surf_available) miss.push('Pix2Surf root');
        if (!cfg.smpl_available) miss.push('SMPL .pkl');
        if (!cfg.garment_obj_available) miss.push('garment OBJ');
        if (!cfg.blender_available) miss.push('Blender');
        if (!cfg.script_available) miss.push('tshirt_to_glb_pose_ready.py');
        return miss.length ? miss.join(', ') : 'unknown';
    }
    
    setupThreeJS() {
        const container = this.threeCanvas.parentElement;
        const w = container.clientWidth, h = container.clientHeight;
        
        this.scene = new THREE.Scene();
        
        const aspect = w / h;
        const frustumSize = 2;
        this.camera = new THREE.OrthographicCamera(
            -frustumSize * aspect / 2, frustumSize * aspect / 2,
            frustumSize / 2, -frustumSize / 2,
            0.1, 100
        );
        this.camera.position.set(0, 0, 5);
        this.camera.lookAt(0, 0, 0);
        
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.threeCanvas,
            alpha: true,
            antialias: false,
            powerPreference: 'high-performance',
            stencil: false,
            depth: true
        });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        
        // Lighting
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        const front = new THREE.DirectionalLight(0xffffff, 0.5);
        front.position.set(0, 1, 5);
        this.scene.add(front);
        const back = new THREE.DirectionalLight(0xffffff, 0.25);
        back.position.set(0, 0, -5);
        this.scene.add(back);
    }

    renderTryOn() {
        const t0 = performance.now();
        this.renderer.render(this.scene, this.camera);
        this._lastRenderMs = performance.now() - t0;
    }
    
    async setupMediaPipe() {
        try {
            this.showLoading('Initializing...');
            const visionBundle = await import(
                /* @vite-ignore */
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs'
            );
            const PoseLandmarker = visionBundle.PoseLandmarker;
            const FilesetResolver = visionBundle.FilesetResolver;
            if (!PoseLandmarker || !FilesetResolver) throw new Error('PoseLandmarker bundle incomplete');

            const vision = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
            );

            const modelUrl =
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

            const makeLm = (delegate) =>
                PoseLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: modelUrl, delegate },
                    runningMode: 'VIDEO',
                    numPoses: 1,
                    minPoseDetectionConfidence: 0.45,
                    minPosePresenceConfidence: 0.45,
                    minTrackingConfidence: 0.45
                });

            try {
                this.poseLandmarker = await makeLm('GPU');
                console.log('✓ MediaPipe pose (full, GPU)');
            } catch (gpuErr) {
                console.warn('MediaPipe GPU failed, using CPU:', gpuErr);
                this.poseLandmarker = await makeLm('CPU');
                console.log('✓ MediaPipe pose (full, CPU)');
            }

            this.hideLoading();
        } catch (e) {
            console.error('Init failed:', e);
            this.updateStatus('error', 'Init failed');
            this.hideLoading();
        }
    }
    _cameraErrorMessage(e) {
        const name = e && e.name;
        const insecure =
            typeof window !== 'undefined' &&
            window.isSecureContext === false &&
            typeof location !== 'undefined' &&
            location.hostname !== 'localhost' &&
            location.hostname !== '127.0.0.1';
        if (insecure && (name === 'SecurityError' || name === 'NotAllowedError' || !name)) {
            return (
                'Camera on phones needs HTTPS. Open https://<your-PC-IP>:5173/apps/ar-tryon/ after npm run dev ' +
                '(not http://). Tap Advanced → Continue once if the browser warns about the certificate.'
            );
        }
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
            return 'Camera permission denied — allow camera for this site in browser settings.';
        }
        if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
            return 'No usable camera found.';
        }
        if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') {
            return 'Camera is busy or unavailable — close other apps using the camera and try again.';
        }
        if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') {
            return 'Camera could not match the requested mode — try again (a simpler mode will be used).';
        }
        const msg = (e && e.message) || String(e);
        return msg && msg !== 'undefined' ? msg : 'Camera failed';
    }

    async _getFrontCameraStream() {
        if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
            throw new Error(
                typeof window !== 'undefined' && window.isSecureContext === false
                    ? 'Use https:// (not http://) on your phone — npm run dev serves HTTPS on port 5173; accept the certificate warning once.'
                    : 'getUserMedia is not supported in this browser.'
            );
        }
        const candidates = [
            { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
            { video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } },
            { video: { facingMode: { ideal: 'user' }, width: { ideal: 960 }, height: { ideal: 540 } } },
            { video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } },
            { video: { facingMode: 'user' } },
            { video: true },
        ];
        let lastErr;
        for (const c of candidates) {
            try {
                return await navigator.mediaDevices.getUserMedia(c);
            } catch (err) {
                lastErr = err;
            }
        }
        throw lastErr || new Error('Could not open camera');
    }

    async _waitVideoReadyForPlay() {
        const v = this.video;
        await new Promise((resolve, reject) => {
            let settled = false;
            const cleanup = () => {
                if (settled) return;
                settled = true;
                clearTimeout(tid);
                v.removeEventListener('loadedmetadata', onMeta);
                v.removeEventListener('error', onErr);
            };
            const onMeta = () => {
                cleanup();
                resolve();
            };
            const onErr = () => {
                cleanup();
                reject(new Error('Video track failed to display'));
            };
            const tid = setTimeout(() => {
                cleanup();
                reject(new Error('Camera preview timed out'));
            }, 15000);
            if (v.readyState >= 1) {
                cleanup();
                resolve();
                return;
            }
            v.addEventListener('loadedmetadata', onMeta);
            v.addEventListener('error', onErr);
        });
    }

    async startCamera() {
        try {
            this.showLoading('Starting camera...');
            this.video.setAttribute('playsinline', 'true');
            this.video.setAttribute('webkit-playsinline', 'true');
            this.video.muted = true;
            this.video.playsInline = true;

            const stream = await this._getFrontCameraStream();
            this.video.srcObject = stream;
            await this._waitVideoReadyForPlay();
            try {
                await this.video.play();
            } catch (pe) {
                this.video.muted = true;
                try {
                    await this.video.play();
                } catch (pe2) {
                    throw pe2 || pe;
                }
            }
            this.onResize();
            this.isRunning = true;
            document.getElementById('start-btn').textContent = 'Stop Camera';
            document.getElementById('screenshot-btn').disabled = false;
            this.updateStatus('active', 'Running');
            if (this.settings.logSession) {
                initSessionLog(this);
            }
            this.applyMirrorViewTransforms();
            this.processFrame();
            this.hideLoading();
        } catch (e) {
            console.error('Camera failed:', e);
            const human = this._cameraErrorMessage(e);
            this.updateStatus('error', 'Camera failed');
            this.showToast(human, 'error');
            this.hideLoading();
        }
    }
    
    stopCamera() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }
        this.isRunning = false;
        this._poseFrameTick = 0;
        this._lastRawLandmarks = null;
        document.getElementById('start-btn').textContent = 'Start Camera';
        document.getElementById('screenshot-btn').disabled = true;
        this.updateStatus('', 'Standby');
    }
    
    onResize() {
        const container = this.threeCanvas.parentElement;
        const w = container.clientWidth, h = container.clientHeight;
        this._viewPxW = w;
        this._viewPxH = h;
        
        this.debugCanvas.width = w;
        this.debugCanvas.height = h;
        this.renderer.setSize(w, h);
        
        const aspect = w / h;
        const frustumSize = 2;
        this.camera.left = -frustumSize * aspect / 2;
        this.camera.right = frustumSize * aspect / 2;
        this.camera.top = frustumSize / 2;
        this.camera.bottom = -frustumSize / 2;
        this.camera.updateProjectionMatrix();
        this.applyMirrorViewTransforms();
    }

    /**
     * MediaPipe landmarks are normalized in the *source video* frame.
     * Our UI displays the video behind the canvas using CSS cover/crop, so we must
     * remap landmarks into the *visible viewport* (0..1 in the canvas/container).
     */
    _poseToViewportNorm(lm) {
        const v = this.video;
        const vw = Math.max(1, v?.videoWidth || 0);
        const vh = Math.max(1, v?.videoHeight || 0);
        const cw = Math.max(1, this._viewPxW || this.threeCanvas?.clientWidth || 1);
        const ch = Math.max(1, this._viewPxH || this.threeCanvas?.clientHeight || 1);

        // If metadata isn't ready yet, fall back to raw normalized coords.
        if (!vw || !vh || !cw || !ch) {
            return { x: lm.x, y: lm.y };
        }

        // object-fit: cover mapping: scale video to fill container, crop excess equally on both sides.
        const s = Math.max(cw / vw, ch / vh);
        const dispW = vw * s;
        const dispH = vh * s;
        const offX = (cw - dispW) * 0.5;
        const offY = (ch - dispH) * 0.5;

        const px = lm.x * vw * s + offX;
        const py = lm.y * vh * s + offY;

        return {
            x: THREE.MathUtils.clamp(px / cw, 0, 1),
            y: THREE.MathUtils.clamp(py / ch, 0, 1),
        };
    }
    
    /**
     * Collect every drawable the GLTFLoader produced (static Mesh, line/points, InstancedMesh) so SAM3
     * and other exporters still match even if flags differ slightly between Three versions.
     */
    _assignGarmentFromGltf(gltf) {
        this.skinnedMesh = null;
        this.skeleton = null;
        this._rigidMeshes = [];
        const skinnedList = [];
        const rigid = [];
        gltf.scene.traverse((child) => {
            if (child.isSkinnedMesh) {
                skinnedList.push(child);
                return;
            }
            if (!child.geometry) return;
            if (child.isMesh) {
                rigid.push(child);
                return;
            }
            if (child.isLine || child.isLineSegments || child.isLineLoop) {
                rigid.push(child);
                return;
            }
            if (child.isPoints) {
                rigid.push(child);
                return;
            }
            const t = (child as THREE.Object3D & { type?: string }).type || '';
            if (t === 'Mesh' || t === 'InstancedMesh' || t.indexOf('Mesh') >= 0) {
                rigid.push(child as unknown as THREE.Mesh);
            }
        });
        if (!skinnedList[0] && !rigid.length) {
            gltf.scene.traverse((child) => {
                if (child.isSkinnedMesh) return;
                if (child.type === 'Mesh' && (child as THREE.Mesh).geometry) {
                    rigid.push(child as THREE.Mesh);
                }
            });
        }
        if (skinnedList[0]) {
            this.skinnedMesh = skinnedList[0];
            this.skeleton = this.skinnedMesh.skeleton;
        }
        this._rigidMeshes = rigid;
        if (!this.skinnedMesh && !rigid.length) {
            const box = new THREE.Box3().setFromObject(gltf.scene);
            const s = new THREE.Vector3();
            const empty = box.isEmpty() || box.getSize(s).lengthSq() < 1e-20;
            if (empty) {
                throw new Error('This GLB has no visible 3D geometry. Export a mesh, then try again.');
            }
            const types: string[] = [];
            gltf.scene.traverse((c) => types.push(c.type));
            throw new Error(
                'No mesh the app could use (SkinnedMesh / Mesh with geometry). Types in file: ' +
                [...new Set(types)].slice(0, 12).join(', ') +
                (types.length > 12 ? '…' : '') +
                ' — re-export, or run npm run build and hard-refresh if the app is outdated via :8000',
            );
        }
    }

    /**
     * SAM3 / trimesh often ship COLOR_0 only (no images). Three’s glTF default PBR is metallic; even after
     * tweaking metalness, cached/shared materials and mobile drivers can still look flat grey. Replace
     * rigid meshes that carry vertex colors with a simple Lambert material so paint always shows.
     * Skinned / instanced meshes are left alone (Pix2Surf pipeline textures + skinning).
     */
    _promoteVertexColorGarmentMaterials(root) {
        if (!root) return;
        let n = 0;
        root.traverse((child) => {
            if (!child.isMesh || child.isSkinnedMesh || child.isInstancedMesh) return;
            const geo = child.geometry;
            if (!geo || typeof geo.getAttribute !== 'function') return;
            const col = geo.getAttribute('color');
            if (!col || typeof col.count !== 'number' || col.count < 1) return;
            const old = child.material;
            const slots = Array.isArray(old) ? old : old ? [old] : [];
            if (slots.some((m) => m && (m as THREE.Material & { map?: unknown }).map)) return;
            const mk = () =>
                new THREE.MeshLambertMaterial({
                    color: 0xffffff,
                    vertexColors: true,
                    side: THREE.DoubleSide,
                });
            child.material = Array.isArray(old) ? old.map(() => mk()) : mk();
            n += 1;
        });
        if (n > 0) {
            console.log(`[try-on] Vertex-color garment: Lambert materials on ${n} mesh(es) (SAM3 / trimesh path)`);
        }
    }

    /** Rigid meshes still on MeshStandardMaterial (e.g. textured SAM3) — glTF default metalness≈1 reads grey without IBL. */
    _tameRigidGarmentMetallic(root) {
        if (!root) return;
        root.traverse((child) => {
            if (!child.isMesh || child.isSkinnedMesh || child.isInstancedMesh) return;
            const mats = Array.isArray(child.material)
                ? child.material
                : child.material
                    ? [child.material]
                    : [];
            for (const mm of mats) {
                if (!mm || !(mm as THREE.MeshStandardMaterial).isMeshStandardMaterial) continue;
                const m = mm as THREE.MeshStandardMaterial;
                if (typeof m.metalness === 'number' && m.metalness > 0.25) {
                    m.metalness = 0;
                    if (typeof m.roughness === 'number' && m.roughness < 0.35) m.roughness = 0.85;
                    m.needsUpdate = true;
                }
            }
        });
    }

    _garmentDisplayName() {
        if (this.skinnedMesh) return this.skinnedMesh.name || 'skinned';
        if (this._rigidMeshes && this._rigidMeshes[0]) return this._rigidMeshes[0].name || 'mesh';
        return '—';
    }

    async loadGarment(file) {
        try {
            this.showLoading('Loading garment...');
            
            // Cleanup
            if (this.garmentGroup) {
                this.scene.remove(this.garmentGroup);
            }
            this.garmentModel = null;
            this.skinnedMesh = null;
            this._rigidMeshes = [];
            this.skeleton = null;
            this.bones = {};
            this._sessionGarmentLabel = null;
            this.fitStructuralMul = null;
            this.smoothedScale = null;
            this.garmentMetadata = null;
            this.garmentBindBodyCenter = null;
            this._hystFacingAway = false;
            this._facingAwayScore = 0;
            this._lastFrontScale = null;
            this._lastMatFacingAway = null;
            this._lastMatState = '';
            this.smoothedYaw = 0;
            this.smoothedPitch = 0;
            this.smoothedRoll = 0;

            const url = URL.createObjectURL(file);
            const loader = new GLTFLoader();
            const gltf = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
            URL.revokeObjectURL(url);
            
            this._assignGarmentFromGltf(gltf);
            this._promoteVertexColorGarmentMaterials(gltf.scene);
            this._tameRigidGarmentMetallic(gltf.scene);
            
            // Setup
            this.applyMaterialSettings();
            this.cacheSkeletonBones();
            
            // Create group
            this.garmentGroup = new THREE.Group();
            this.garmentModel = gltf.scene;
            this.garmentGroup.add(this.garmentModel);
            this.scene.add(this.garmentGroup);
            
            this.measureGarmentFromBones();
            
            // Update UI
            const modeLabel = this.skinnedMesh
                ? 'Skinned (rigged)'
                : 'Rigid (no bones — e.g. SAM3 mesh)';
            const info = document.getElementById('garment-info');
            info.style.display = 'block';
            info.innerHTML = `
                ${modeLabel}<br>
                Mesh: ${this._garmentDisplayName()}<br>
                Bones: ${Object.keys(this.bones).length}<br>
                Shoulder Width: ${this.garmentShoulderWidth?.toFixed(4) || 'N/A'}<br>
                Scale ref: ${this.garmentScaleRefWidth?.toFixed(4) || 'N/A'}<br>
                Torso Height: ${this.garmentTorsoHeight?.toFixed(4) || 'N/A'}
            `;
            
            console.log(`✓ Garment loaded: shoulders=${this.garmentShoulderWidth?.toFixed(4)}, scaleRef=${this.garmentScaleRefWidth?.toFixed(4)}, torso=${this.garmentTorsoHeight?.toFixed(4)}`);
            this._sessionGarmentLabel = (file && file.name) ? String(file.name) : 'uploaded.glb';
            this.showToast('Garment loaded — stand so both shoulders are visible for automatic fit', 'success');
            this.hideLoading();
        } catch (e) {
            console.error('Load failed:', e);
            this.showToast('Load failed: ' + e.message, 'error');
            this.hideLoading();
        }
    }
    
    measureGarmentFromBones() {
        if (!this.garmentGroup || !this.garmentModel) return;

        const ln = this.resolveUpperArmBone('left');
        const rn = this.resolveUpperArmBone('right');
        const leftShoulder = ln ? this.bones[ln] : null;
        const rightShoulder = rn ? this.bones[rn] : null;
        console.log('Resolved arm bones for measurement:', ln, rn);

        const spine3 = this.bones['spine3'] || this.bones['spine2'];
        const pelvis = this.bones['pelvis'];

        this.garmentGroup.scale.set(1, 1, 1);
        this.garmentGroup.position.set(0, 0, 0);
        this.garmentGroup.rotation.set(0, 0, 0);
        this.garmentModel.updateMatrixWorld(true);

        this.garmentShoulderWidth = null;
        this.garmentTorsoHeight = null;
        this.garmentScaleRefWidth = null;

        if (leftShoulder && rightShoulder) {
            const lp = new THREE.Vector3();
            const rp = new THREE.Vector3();
            leftShoulder.getWorldPosition(lp);
            rightShoulder.getWorldPosition(rp);
            this.garmentShoulderWidth = lp.distanceTo(rp);
            console.log(`Shoulder bones: L(${lp.x.toFixed(3)}, ${lp.y.toFixed(3)}, ${lp.z.toFixed(3)}) R(${rp.x.toFixed(3)}, ${rp.y.toFixed(3)}, ${rp.z.toFixed(3)})`);
        }

        if (spine3 && pelvis) {
            const sp = new THREE.Vector3();
            const pp = new THREE.Vector3();
            spine3.getWorldPosition(sp);
            pelvis.getWorldPosition(pp);
            this.garmentTorsoHeight = sp.distanceTo(pp);
        } else if (this.garmentShoulderWidth) {
            this.garmentTorsoHeight = this.garmentShoulderWidth * 1.5;
        }

        const box = new THREE.Box3().setFromObject(this.garmentModel);
        const size = box.getSize(new THREE.Vector3());
        console.log(`Garment BBox at scale 1: ${size.x.toFixed(4)} × ${size.y.toFixed(4)} × ${size.z.toFixed(4)}`);

        const hemW = new THREE.Vector3(
            (box.min.x + box.max.x) * 0.5,
            box.min.y,
            (box.min.z + box.max.z) * 0.5
        );
        this.garmentBindHemMid = this.garmentGroup.worldToLocal(hemW.clone());

        if (!this.garmentShoulderWidth || this.garmentShoulderWidth < 0.001) {
            this.garmentShoulderWidth = size.x * 0.75;
            if (!this.garmentTorsoHeight || this.garmentTorsoHeight < 0.001) {
                this.garmentTorsoHeight = size.y;
            }
            console.warn('[WARN] Shoulder bones missing — garment span from bbox:', this.garmentShoulderWidth.toFixed(4));
        }

        console.log(`Garment shoulder width: ${this.garmentShoulderWidth?.toFixed(4)}`);
        console.log(`Garment torso height: ${this.garmentTorsoHeight?.toFixed(4)}`);

        const boneW = this.garmentShoulderWidth || 0;
        const sx = size.x;
        this.garmentScaleRefWidth = boneW;
        if (boneW > 1e-6 && sx > boneW * 1.22) {
            const extra = sx - boneW * 1.12;
            this.garmentScaleRefWidth = boneW + Math.max(0, extra) * 0.26;
        }
        this.garmentScaleRefWidth = Math.max(this.garmentScaleRefWidth || 0, boneW * 0.99 || 0.001);
        console.log(`Garment scale ref width (T-pose–aware): ${this.garmentScaleRefWidth.toFixed(4)}`);

        this._cacheGarmentBindShoulderMid();
        this._cacheGarmentBindBodyCenter();
        if (!this.garmentBindShoulderMid) {
            const cx = (box.min.x + box.max.x) * 0.5;
            const cz = (box.min.z + box.max.z) * 0.5;
            const shoulderY = box.max.y - size.y * 0.1;
            this.garmentBindShoulderMid = this.garmentGroup.worldToLocal(new THREE.Vector3(cx, shoulderY, cz));
        }
        if (!this.garmentBindBodyCenter) {
            const cx = (box.min.x + box.max.x) * 0.5;
            const cz = (box.min.z + box.max.z) * 0.5;
            const topY = box.max.y - size.y * 0.1;
            const bodyY = (topY + box.min.y) * 0.5;
            this.garmentBindBodyCenter = this.garmentGroup.worldToLocal(new THREE.Vector3(cx, bodyY, cz));
        }
        this._recomputeGarmentShoulderHemDelta();
    }

    /** Use for body÷garment scale only — not bone midpoints. */
    getGarmentScaleRefWidth() {
        if (this.garmentScaleRefWidth != null && this.garmentScaleRefWidth > 1e-6) return this.garmentScaleRefWidth;
        return Math.max(0.001, this.garmentShoulderWidth || 0.3);
    }

    _recomputeGarmentShoulderHemDelta() {
        if (this.garmentBindShoulderMid && this.garmentBindHemMid) {
            this._garmentShoulderHemLocalDy = Math.abs(
                this.garmentBindShoulderMid.y - this.garmentBindHemMid.y
            );
        } else if (this.garmentTorsoHeight && this.garmentTorsoHeight > 0.001) {
            this._garmentShoulderHemLocalDy = Math.max(0.12, this.garmentTorsoHeight * 0.55);
        }
    }
    
    _cacheGarmentBindShoulderMid() {
        const lbName = this.resolveUpperArmBone('left');
        const rbName = this.resolveUpperArmBone('right');
        const lb = lbName ? this.bones[lbName] : null;
        const rb = rbName ? this.bones[rbName] : null;
        if (!lb || !rb || !this.garmentGroup) {
            this.garmentBindShoulderMid = null;
            this._recomputeGarmentShoulderHemDelta();
            return;
        }
        const l = new THREE.Vector3();
        const r = new THREE.Vector3();
        lb.getWorldPosition(l);
        rb.getWorldPosition(r);
        const mid = l.clone().add(r).multiplyScalar(0.5);
        this.garmentBindShoulderMid = this.garmentGroup.worldToLocal(mid.clone());
    }

    _cacheGarmentBindBodyCenter() {
        if (!this.garmentGroup) {
            this.garmentBindBodyCenter = null;
            return;
        }

        const lbName = this.resolveUpperArmBone('left');
        const rbName = this.resolveUpperArmBone('right');
        const lb = lbName ? this.bones[lbName] : null;
        const rb = rbName ? this.bones[rbName] : null;
        if (!lb || !rb) {
            this.garmentBindBodyCenter = null;
            return;
        }

        const shoulderL = new THREE.Vector3();
        const shoulderR = new THREE.Vector3();
        lb.getWorldPosition(shoulderL);
        rb.getWorldPosition(shoulderR);
        const shoulderMid = shoulderL.clone().add(shoulderR).multiplyScalar(0.5);

        const leftHip = this.bones['left_hip'] || this.bones['LeftUpLeg'] || this.bones['leftUpLeg'] || null;
        const rightHip = this.bones['right_hip'] || this.bones['RightUpLeg'] || this.bones['rightUpLeg'] || null;
        const pelvis = this.bones['pelvis'] || this.bones['hips'] || this.bones['Hips'] || null;

        let hipMid = null;
        if (leftHip && rightHip) {
            const l = new THREE.Vector3();
            const r = new THREE.Vector3();
            leftHip.getWorldPosition(l);
            rightHip.getWorldPosition(r);
            hipMid = l.add(r).multiplyScalar(0.5);
        } else if (pelvis) {
            hipMid = new THREE.Vector3();
            pelvis.getWorldPosition(hipMid);
        }

        if (!hipMid) {
            this.garmentBindBodyCenter = this.garmentGroup.worldToLocal(shoulderMid.clone());
            return;
        }

        const bodyCenter = shoulderMid.clone().add(hipMid).multiplyScalar(0.5);
        this.garmentBindBodyCenter = this.garmentGroup.worldToLocal(bodyCenter.clone());
    }
    
    applyMaterialSettings() {
        const targets = [];
        if (this.skinnedMesh) targets.push(this.skinnedMesh);
        if (this._rigidMeshes && this._rigidMeshes.length) {
            for (const m of this._rigidMeshes) targets.push(m);
        }
        if (!targets.length) return;
        
        // Single-sided front/back is wrong for oblique & profile (whole torso can vanish). DoubleSide there.
        let defaultSide;
        if (this.settings.frontFaceOnly) {
            const va = String(this.debugInfo?.viewAngle || 'FRONT').toUpperCase();
            if (va === 'SIDE' || va === 'ANGLE') {
                defaultSide = THREE.DoubleSide;
            } else {
                let towardCamAsFront = !this.debugInfo?.facingAway;
                if (this.settings.garmentCullInvert) towardCamAsFront = !towardCamAsFront;
                // Ortho cam scale.x = -1 mirrors the render; single-sided shirts otherwise pick the wrong face (logo reads backward).
                if (this.settings.mirrorView) towardCamAsFront = !towardCamAsFront;
                // SMPL / Pix2Surf / many generated GLBs put the chest print on the face Three.js culls as “back”
                // for our pose + mirror setup — use BackSide when the user faces the camera so the front design shows.
                defaultSide = towardCamAsFront ? THREE.BackSide : THREE.FrontSide;
            }
        } else {
            defaultSide = THREE.DoubleSide;
        }
        
        const applyToMat = (mat, matSide) => {
            if (!mat) return;
            if ('side' in mat) mat.side = matSide;
            if ('wireframe' in mat) (mat as THREE.Material & { wireframe: boolean }).wireframe = this.settings.wireframe;
            mat.needsUpdate = true;
        };

        for (const obj of targets) {
            const geom = obj.geometry;
            const vcol = geom && typeof geom.getAttribute === 'function' && geom.getAttribute('color');
            const mats = Array.isArray(obj.material)
                ? obj.material
                : obj.material
                    ? [obj.material]
                    : [];
            // SAM3 / trimesh: COLOR_0 vertex paint, often no glTF material → Three’s default PBR uses metalness 1.
            // Without an env map that kills diffuse and reads as flat grey even with vertexColors: true.
            for (const mm of mats) {
                if (!mm) continue;
                if (vcol && 'vertexColors' in mm) {
                    (mm as THREE.Material & { vertexColors: boolean }).vertexColors = true;
                }
                const hasMap = Boolean((mm as THREE.Material & { map?: unknown }).map);
                if (vcol && !hasMap && 'metalness' in mm) {
                    const std = mm as THREE.MeshStandardMaterial;
                    if (typeof std.metalness === 'number' && std.metalness > 0.25) {
                        std.metalness = 0;
                        if (typeof std.roughness === 'number' && std.roughness < 0.35) {
                            std.roughness = 0.85;
                        }
                    }
                }
            }
            const m0 = mats[0];
            const vcolNoImageMap = Boolean(vcol && m0 && !(m0 as THREE.Material & { map?: unknown }).map);
            const matSide = vcolNoImageMap ? THREE.DoubleSide : defaultSide;

            for (const mm of mats) applyToMat(mm, matSide);
        }
    }
    
    cacheSkeletonBones() {
        if (!this.skeleton) return;
        
        const boneNames = [];
        for (const bone of this.skeleton.bones) {
            this.bones[bone.name] = bone;
            boneNames.push(bone.name);
        }
        
        console.log('=== ALL SKELETON BONES ===');
        console.log(boneNames.join(', '));
        console.log('==========================');

        if (this.garmentGroup) {
            this.measureGarmentFromBones();
        }
    }

    /**
     * Upper-arm / shoulder-line bone for measurements and bind midpoint (garment.glb vs manu_tshirt_smpl_ready.glb).
     */
    resolveUpperArmBone(side) {
        const candidates = side === 'left'
            ? ['left_shoulder', 'left_arm', 'LeftArm', 'leftUpperArm', 'left_upper_arm', 'left_collar_1', 'left_collar']
            : ['right_shoulder', 'right_arm', 'RightArm', 'rightUpperArm', 'right_upper_arm', 'right_collar_1', 'right_collar'];
        return candidates.find(n => this.bones[n]) || null;
    }

    processFrame() {
        if (!this.isRunning) return;
        
        const start = performance.now();
        
        if (this.video.readyState >= 2 && this.poseLandmarker) {
            const every = Math.max(1, Math.min(3, Math.round(this.settings.poseDetectEvery || 1)));
            let pose = null;
            if (every <= 1) {
                const r = this.poseLandmarker.detectForVideo(this.video, start);
                pose = (r.landmarks && r.landmarks[0]) ? r.landmarks[0] : null;
                if (pose) this._lastRawLandmarks = pose;
            } else {
                this._poseFrameTick = (this._poseFrameTick + 1) % every;
                if (this._poseFrameTick === 0) {
                    const r = this.poseLandmarker.detectForVideo(this.video, start);
                    pose = (r.landmarks && r.landmarks[0]) ? r.landmarks[0] : null;
                    if (pose) this._lastRawLandmarks = pose;
                    else this._lastRawLandmarks = null;
                }
                pose = this._lastRawLandmarks;
            }
            this.onPoseResults(pose);
        }
        
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsUpdate >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
            document.getElementById('fps-value').textContent = this.fps;
        }
        document.getElementById('latency-value').textContent = Math.round(now - start) + 'ms';
        this._lastLatencyMs = now - start;
        
        requestAnimationFrame(() => this.processFrame());
    }
    
    onPoseResults(pose) {
        this.debugCtx.clearRect(0, 0, this.debugCanvas.width, this.debugCanvas.height);
        
        if (!pose) {
            this.updateStatus('warning', 'No pose');
            document.getElementById('tracking-confidence').textContent = '--';
            this.renderTryOn();
            return;
        }
        
        const conf = pose.reduce((a, l) => a + (l.visibility || 0), 0) / pose.length;
        document.getElementById('tracking-confidence').textContent = Math.round(conf * 100) + '%';
        
        const ls0 = pose[this.MP.LEFT_SHOULDER];
        const rs0 = pose[this.MP.RIGHT_SHOULDER];
        const lh0 = pose[this.MP.LEFT_HIP];
        const rh0 = pose[this.MP.RIGHT_HIP];
        const hasRecentTorso = this._hasRecentReliableTorso();
        const shoulderVisStrong = (ls0?.visibility || 0) >= 0.42 && (rs0?.visibility || 0) >= 0.42;
        const shoulderVisShow = (ls0?.visibility || 0) >= 0.32 && (rs0?.visibility || 0) >= 0.32;
        const shoulderAnyShow = Math.max((ls0?.visibility || 0), (rs0?.visibility || 0)) >= 0.37;
        const hipVisOk = (lh0?.visibility || 0) >= 0.28 && (rh0?.visibility || 0) >= 0.28;
        const occlusionCarry = !shoulderVisShow && shoulderAnyShow && hasRecentTorso && conf >= 0.24;
        const trackingPoseStrong = conf >= 0.37 && (shoulderVisStrong || occlusionCarry) && (hipVisOk || conf >= 0.44);
        const trackingShowGarment = conf >= 0.26 && (shoulderVisShow || occlusionCarry);
        const shouldersGone = (ls0?.visibility || 0) < 0.22 && (rs0?.visibility || 0) < 0.22;
        const trackingDead = conf < 0.17 || (shouldersGone && !hasRecentTorso);

        this._garmentTrackingOk = trackingShowGarment;
        this.debugInfo.garmentHidden = this.garmentGroup ? !trackingShowGarment : false;
        if (trackingDead) {
            this.updateStatus('warning', conf < 0.17 ? 'Low confidence' : 'Shoulders not visible');
            if (this.garmentGroup) {
                this.garmentGroup.visible = false;
            }
            this.renderTryOn();
            return;
        }

        if (!trackingShowGarment) {
            this.updateStatus('warning', 'Partial pose — need shoulders for garment');
        } else if (occlusionCarry) {
            this.updateStatus('warning', 'Tracking (occlusion assist)');
        } else if (!trackingPoseStrong) {
            this.updateStatus('warning', 'Tracking (partial)');
        } else {
            this.updateStatus('active', 'Tracking');
        }
        
        // Smooth pose
        this.smoothedPose = this.smoothPose(pose);
        
        // Draw skeleton
        if (this.settings.showSkeleton) {
            this.drawSkeleton(this.smoothedPose);
        }
        
        if (this.garmentGroup) {
            this.garmentGroup.visible = trackingShowGarment;
            if (trackingShowGarment) {
                this.updateGarment(this.smoothedPose);
            }
        }

        if (this.settings.showMpMapping && this.garmentMetadata?.mediapipe_mapping && this.smoothedPose) {
            this.drawMpMappingOverlay(this.smoothedPose);
        }

        maybeAppendSessionLogEntry(this, {
            conf,
            smoothedPose: this.smoothedPose,
            trackingPoseStrong,
            trackingShowGarment
        });
        
        // Draw HUD
        if (this.settings.showHud) {
            this.drawHud(conf);
        }

        this.renderTryOn();
    }
    
    smoothPose(pose) {
        return smoothPoseLandmarks(pose, this.smoothedPose, this.settings.smoothingFactor);
    }

    _clonePoseLandmark(lm) {
        if (!lm) return null;
        return {
            x: lm.x,
            y: lm.y,
            z: lm.z ?? 0,
            visibility: lm.visibility ?? 0
        };
    }

    _hasRecentReliableTorso(maxAgeMs = 1100) {
        const t = this._lastReliableTorso;
        return Boolean(t && Number.isFinite(t.ts) && (performance.now() - t.ts) <= maxAgeMs);
    }

    _buildOcclusionRobustTorso(pose) {
        const srcLs = pose[this.MP.LEFT_SHOULDER];
        const srcRs = pose[this.MP.RIGHT_SHOULDER];
        const srcLh = pose[this.MP.LEFT_HIP];
        const srcRh = pose[this.MP.RIGHT_HIP];
        if (!srcLs || !srcRs || !srcLh || !srcRh) return null;

        let ls = this._clonePoseLandmark(srcLs);
        let rs = this._clonePoseLandmark(srcRs);
        let lh = this._clonePoseLandmark(srcLh);
        let rh = this._clonePoseLandmark(srcRh);
        let occlusionAssist = false;

        const prev = this._hasRecentReliableTorso(1200) ? this._lastReliableTorso : null;
        if (prev) {
            const shoulderVecPrev = {
                x: prev.ls.x - prev.rs.x,
                y: prev.ls.y - prev.rs.y,
                z: (prev.ls.z || 0) - (prev.rs.z || 0)
            };
            const lsVis = ls.visibility || 0;
            const rsVis = rs.visibility || 0;
            if (lsVis < 0.24 && rsVis >= 0.34) {
                ls = {
                    x: rs.x + shoulderVecPrev.x,
                    y: rs.y + shoulderVecPrev.y,
                    z: (rs.z || 0) + shoulderVecPrev.z,
                    visibility: Math.max(lsVis, Math.min(0.34, rsVis * 0.72))
                };
                occlusionAssist = true;
            } else if (rsVis < 0.24 && lsVis >= 0.34) {
                rs = {
                    x: ls.x - shoulderVecPrev.x,
                    y: ls.y - shoulderVecPrev.y,
                    z: (ls.z || 0) - shoulderVecPrev.z,
                    visibility: Math.max(rsVis, Math.min(0.34, lsVis * 0.72))
                };
                occlusionAssist = true;
            }

            const hipVecPrev = {
                x: prev.lh.x - prev.rh.x,
                y: prev.lh.y - prev.rh.y,
                z: (prev.lh.z || 0) - (prev.rh.z || 0)
            };
            const lhVis = lh.visibility || 0;
            const rhVis = rh.visibility || 0;
            const currShoulderMid = {
                x: (ls.x + rs.x) * 0.5,
                y: (ls.y + rs.y) * 0.5,
                z: ((ls.z || 0) + (rs.z || 0)) * 0.5
            };
            const prevShoulderMid = {
                x: (prev.ls.x + prev.rs.x) * 0.5,
                y: (prev.ls.y + prev.rs.y) * 0.5,
                z: ((prev.ls.z || 0) + (prev.rs.z || 0)) * 0.5
            };
            const shoulderDrift = {
                x: currShoulderMid.x - prevShoulderMid.x,
                y: currShoulderMid.y - prevShoulderMid.y,
                z: currShoulderMid.z - prevShoulderMid.z
            };
            if (lhVis < 0.2 && rhVis >= 0.28) {
                lh = {
                    x: rh.x + hipVecPrev.x,
                    y: rh.y + hipVecPrev.y,
                    z: (rh.z || 0) + hipVecPrev.z,
                    visibility: Math.max(lhVis, Math.min(0.3, rhVis * 0.68))
                };
                occlusionAssist = true;
            } else if (rhVis < 0.2 && lhVis >= 0.28) {
                rh = {
                    x: lh.x - hipVecPrev.x,
                    y: lh.y - hipVecPrev.y,
                    z: (lh.z || 0) - hipVecPrev.z,
                    visibility: Math.max(rhVis, Math.min(0.3, lhVis * 0.68))
                };
                occlusionAssist = true;
            } else if (lhVis < 0.18 && rhVis < 0.18) {
                lh = {
                    x: prev.lh.x + shoulderDrift.x,
                    y: prev.lh.y + shoulderDrift.y,
                    z: (prev.lh.z || 0) + shoulderDrift.z,
                    visibility: 0.2
                };
                rh = {
                    x: prev.rh.x + shoulderDrift.x,
                    y: prev.rh.y + shoulderDrift.y,
                    z: (prev.rh.z || 0) + shoulderDrift.z,
                    visibility: 0.2
                };
                occlusionAssist = true;
            }
        }

        const shouldersVisible = (ls.visibility || 0) >= 0.2 || (rs.visibility || 0) >= 0.2;
        if (!shouldersVisible) return null;

        const shouldersReliable = (ls.visibility || 0) >= 0.36 && (rs.visibility || 0) >= 0.36;
        const hipsReliable = (lh.visibility || 0) >= 0.24 && (rh.visibility || 0) >= 0.24;
        if (shouldersReliable && hipsReliable) {
            this._lastReliableTorso = {
                ls: this._clonePoseLandmark(ls),
                rs: this._clonePoseLandmark(rs),
                lh: this._clonePoseLandmark(lh),
                rh: this._clonePoseLandmark(rh),
                ts: performance.now()
            };
        }

        return { ls, rs, lh, rh, occlusionAssist };
    }

    /**
     * Convert normalized MediaPipe coordinates to this orthographic camera world space at z=0.
     * This keeps x/y placement aspect-correct, so overlay alignment remains stable when users move.
     */
    poseToWorld(lm) {
        const viewW = (this.camera?.right ?? 1) - (this.camera?.left ?? -1);
        const viewH = (this.camera?.top ?? 1) - (this.camera?.bottom ?? -1);
        const n = this._poseToViewportNorm(lm);
        const x = (n.x - 0.5) * viewW;
        const y = -(n.y - 0.5) * viewH;
        const z = -((lm.z || 0) * viewH);
        return new THREE.Vector3(x, y, z);
    }

    /** Screen-space torso metrics. Null if shoulders unusable. */
    _measureBodyFromPose(pose) {
        const robust = this._buildOcclusionRobustTorso(pose);
        if (!robust) return null;
        const { ls, rs, lh, rh, occlusionAssist } = robust;

        const LS = new THREE.Vector2(this._poseToViewportNorm(ls).x, this._poseToViewportNorm(ls).y);
        const RS = new THREE.Vector2(this._poseToViewportNorm(rs).x, this._poseToViewportNorm(rs).y);
        const LH = new THREE.Vector2(this._poseToViewportNorm(lh).x, this._poseToViewportNorm(lh).y);
        const RH = new THREE.Vector2(this._poseToViewportNorm(rh).x, this._poseToViewportNorm(rh).y);

        const shoulderWidthRaw = Math.abs(LS.x - RS.x);
        const shoulderCenterY = (LS.y + RS.y) / 2;
        const hipCenterY = (LH.y + RH.y) / 2;
        let torsoHeightScreen = Math.abs(hipCenterY - shoulderCenterY);
        // Keep lower floor small so distant users can still shrink naturally.
        torsoHeightScreen = THREE.MathUtils.clamp(torsoHeightScreen, 0.04, 0.62);
        // Cap gross outliers only; avoid compressing shoulder span for far users.
        const shoulderSpanCap = torsoHeightScreen * 1.4;
        const shoulderWidthScreen = THREE.MathUtils.clamp(shoulderWidthRaw, 0.008, shoulderSpanCap);

        const expectedShoulderWidth = Math.max(0.01, torsoHeightScreen * 0.65);
        const viewAngleRatio = shoulderWidthScreen / expectedShoulderWidth;

        let effectiveWidth;
        let viewAngle;
        if (viewAngleRatio < 0.35) {
            effectiveWidth = torsoHeightScreen * 0.45;
            viewAngle = 'SIDE';
        } else if (viewAngleRatio < 0.65) {
            effectiveWidth = Math.max(shoulderWidthScreen, torsoHeightScreen * 0.60);
            viewAngle = 'ANGLE';
        } else {
            const torsoBasedWidth = torsoHeightScreen * 0.65;
            effectiveWidth = Math.max(shoulderWidthScreen, torsoBasedWidth);
            viewAngle = 'FRONT';
        }

        const minEffW = torsoHeightScreen * 0.4;
        if (effectiveWidth < minEffW) effectiveWidth = minEffW;

        const bodyWidth = effectiveWidth * 2;
        const bodyHeight = torsoHeightScreen * 2;
        const bodyWidthFromShoulders = shoulderWidthScreen * 2;
        this.debugInfo.occlusionAssist = occlusionAssist;

        return {
            ls, rs, lh, rh,
            LS, RS, LH, RH,
            shoulderWidthRaw,
            shoulderWidthScreen,
            shoulderCenterY,
            hipCenterY,
            torsoHeightScreen,
            viewAngleRatio,
            effectiveWidth,
            bodyWidth,
            bodyHeight,
            bodyWidthFromShoulders,
            viewAngle,
            occlusionAssist
        };
    }

    /**
     * If nose/eyes are visible and in front of shoulders (MediaPipe z), treat as facing camera → shirt FRONT.
     * Returns 'front' | null. ('back' is not inferred here — legacy hysteresis still handles turned-away.)
     */
    _computeFaceFacingOverride(pose, ls, rs) {
        if (!this.settings.useFaceFacingHint || !pose) return null;
        const nose = pose[this.MP.NOSE];
        if (!nose) return null;
        const le = pose[this.MP.LEFT_EYE];
        const re = pose[this.MP.RIGHT_EYE];
        const shoulderMidZ = ((ls.z || 0) + (rs.z || 0)) * 0.5;
        const nz = nose.z || 0;
        const eyeOk = ((le?.visibility || 0) > 0.28 || (re?.visibility || 0) > 0.28);
        const noseVis = nose.visibility || 0;
        if (noseVis < 0.32 || (!eyeOk && noseVis < 0.45)) return null;
        if (nz >= shoulderMidZ + 0.13) return null;
        const nx = nose.x;
        const lx = ls.x;
        const rx = rs.x;
        const lo = Math.min(lx, rx);
        const hi = Math.max(lx, rx);
        const span = hi - lo;
        if (span >= 0.05) {
            if (nx < lo - span * 0.55 || nx > hi + span * 0.55) return null;
        }
        return 'front';
    }

    /** LEGACY only — same rules as before face override existed (shoulder x-order + nose z + side view). */
    _legacyFacingHysteresisUpdate(pose, ls, rs, isSideView, isAmbiguousSep, rawDiff, nose, noseOk, shoulderMidZ, shoulderSepNorm) {
        const noseClearlyForward = noseOk && (nose.z || 0) < shoulderMidZ + 0.08;
        if (isSideView) {
            if (noseOk) {
                if (nose.z > shoulderMidZ + 0.18) this._hystFacingAway = true;
                else if (nose.z < shoulderMidZ - 0.03) this._hystFacingAway = false;
            }
        } else if (!isAmbiguousSep) {
            const rel = THREE.MathUtils.clamp((shoulderSepNorm - 0.085) / 0.18, 0, 1);
            const signed = THREE.MathUtils.clamp(rawDiff / 0.12, -1, 1);
            const prev = Number.isFinite(this._facingAwayScore) ? this._facingAwayScore : 0;
            // Faster convergence when shoulders are clearly separated.
            this._facingAwayScore = THREE.MathUtils.lerp(prev, signed, 0.10 + 0.18 * rel);
            if (noseClearlyForward) {
                this._hystFacingAway = false;
                // Pull score toward front when face is clearly forward to avoid "sticky back".
                this._facingAwayScore = Math.min(this._facingAwayScore, -0.15);
            } else {
                // Hysteresis via threshold on the smoothed score.
                if (this._facingAwayScore > 0.35) this._hystFacingAway = true;
                else if (this._facingAwayScore < -0.35) this._hystFacingAway = false;
            }
        } else {
            if (noseOk) {
                if (nose.z > shoulderMidZ + 0.14) this._hystFacingAway = true;
                else if (nose.z < shoulderMidZ - 0.05) this._hystFacingAway = false;
            }
        }
    }

    /**
     * Structural uniform-scale target from pose. Uses the same world space as poseToWorld + garment bind
     * (not normalized image coords ÷ metres — that mismatch blew scales to ~1.6×+). T-pose bind meshes use
     * garmentScaleRefWidth so we do not treat shoulder-bone distance as the full shirt silhouette width.
     */
    computeFitStructuralMul(pose) {
        const denom = this.getGarmentScaleRefWidth();
        if (!denom || denom < 0.001) return null;
        const m = this._measureBodyFromPose(pose);
        if (!m) return null;

        const viewW = (this.camera?.right ?? 1) - (this.camera?.left ?? -1);
        const pLS = this.poseToWorld(m.ls);
        const pRS = this.poseToWorld(m.rs);
        // Blend in torso-informed width for SIDE/ANGLE to avoid foreshortened shoulder collapse.
        const userShoulderWorld = Math.max(0.005, Math.abs(pLS.x - pRS.x));
        const userBodyWidthWorld = Math.max(0.005, m.effectiveWidth * viewW);
        let widthForScale = userShoulderWorld;
        if (m.viewAngle === 'SIDE') {
            widthForScale = Math.max(userShoulderWorld, userBodyWidthWorld * 0.94);
        } else if (m.viewAngle === 'ANGLE') {
            widthForScale = THREE.MathUtils.lerp(userShoulderWorld, userBodyWidthWorld, 0.36);
        }
        const shoulderRatio = (widthForScale * 0.998) / denom;

        const hipOk = (m.lh.visibility || 0) >= 0.22 && (m.rh.visibility || 0) >= 0.22;
        const poseShoulderMid = pLS.clone().add(pRS).multiplyScalar(0.5);
        let poseHipMid;
        if (hipOk) {
            poseHipMid = this.poseToWorld(m.lh).clone().add(this.poseToWorld(m.rh)).multiplyScalar(0.5);
        } else {
            poseHipMid = poseShoulderMid.clone();
            const hipDrop = Math.max(Math.max(0.06, m.shoulderWidthScreen * 0.8), m.torsoHeightScreen * 1.9);
            poseHipMid.y = poseShoulderMid.y - hipDrop;
        }
        const dPoseY = Math.abs(poseShoulderMid.y - poseHipMid.y);
        const dGarmentY = Math.max(0.02, this._garmentShoulderHemLocalDy || 0.25);
        const torsoRatio = THREE.MathUtils.clamp(dPoseY / dGarmentY, 0.1, 2.85);

        let raw;
        if (!hipOk) {
            raw = shoulderRatio;
        } else if (m.viewAngle === 'SIDE') {
            // Side view: shoulder span collapses due to foreshortening → would shrink the shirt.
            // Torso length is much more stable in profile, so lean on it more.
            raw = THREE.MathUtils.lerp(shoulderRatio, torsoRatio, 0.72);
        } else if (m.viewAngle === 'ANGLE') {
            raw = THREE.MathUtils.lerp(shoulderRatio, torsoRatio, 0.2);
        } else {
            raw = THREE.MathUtils.lerp(shoulderRatio, torsoRatio, 0.12);
        }

        return THREE.MathUtils.clamp(raw, 0.08, 1.92);
    }

    getUniformGarmentScale() {
        const struct = this.fitStructuralMul != null ? this.fitStructuralMul : 1;
        return THREE.MathUtils.clamp(struct, 0.08, 3.0);
    }

    /**
     * Smoothly tracks computeFitStructuralMul() every frame. Side view freezes the multiplier
     * after the first lock to avoid noisy width.
     */
    _updateContinuousStructuralFit(M, pose) {
        if (!this.garmentGroup || !pose) return;
        if (this.getGarmentScaleRefWidth() < 0.001) return;

        const target = this.computeFitStructuralMul(pose);
        if (target == null) return;

        const prev = this.fitStructuralMul;
        if (prev == null) {
            this.fitStructuralMul = target;
            return;
        }

        const rel = Math.abs(target - prev) / Math.max(1e-6, prev);
        if (M.viewAngle === 'SIDE') {
            // Side view is noisier, but freezing breaks distance response; use slower tracking instead.
            const alpha = rel > 0.18 ? 0.18 : 0.08;
            this.fitStructuralMul = THREE.MathUtils.lerp(prev, target, alpha);
            return;
        }

        if (rel > 0.28) {
            this.fitStructuralMul = THREE.MathUtils.lerp(prev, target, 0.58);
            return;
        }

        const alpha = rel > 0.09 ? 0.26 : 0.16;
        this.fitStructuralMul = THREE.MathUtils.lerp(prev, target, alpha);
    }
    
    updateGarment(pose) {
        const M = this._measureBodyFromPose(pose);
        if (!M) return;

        this._updateContinuousStructuralFit(M, pose);

        const {
            ls, rs, lh, rh, LS, RS, LH, RH,
            shoulderCenterY, hipCenterY, shoulderWidthScreen, torsoHeightScreen, effectiveWidth
        } = M;

        this.debugInfo.viewAngle = M.viewAngle;
        this.debugInfo.shoulderWidth = shoulderWidthScreen;
        this.debugInfo.torsoHeight = torsoHeightScreen;
        this.debugInfo.effectiveWidth = effectiveWidth;
        this.debugInfo.fitStructuralMul = this.fitStructuralMul;

        const pLS = this.poseToWorld(ls);
        const pRS = this.poseToWorld(rs);
        const poseShoulderMid = pLS.clone().add(pRS).multiplyScalar(0.5);
        poseShoulderMid.z = 0;

        const hipVisPose = (lh.visibility || 0) > 0.24 && (rh.visibility || 0) > 0.24;
        let poseHipMid;
        if (hipVisPose) {
            poseHipMid = this.poseToWorld(lh).clone().add(this.poseToWorld(rh)).multiplyScalar(0.5);
        } else {
            poseHipMid = poseShoulderMid.clone();
            const hipDrop = Math.max(Math.max(0.06, shoulderWidthScreen * 0.8), torsoHeightScreen * 1.9);
            poseHipMid.y = poseShoulderMid.y - hipDrop;
        }
        poseHipMid.z = 0;
        const dPoseYWorld = Math.max(1e-6, Math.abs(poseShoulderMid.y - poseHipMid.y));

        // Match computeFitStructuralMul(): shoulder span in image-plane world units.
        const viewW = (this.camera?.right ?? 1) - (this.camera?.left ?? -1);
        const userShoulderWorld = Math.max(0.005, Math.abs(pLS.x - pRS.x));
        const userBodyWidthWorld = Math.max(0.005, effectiveWidth * viewW);
        let widthForScale = userShoulderWorld;
        if (this.debugInfo.viewAngle === 'SIDE') {
            widthForScale = Math.max(userShoulderWorld, userBodyWidthWorld * 0.94);
        } else if (this.debugInfo.viewAngle === 'ANGLE') {
            widthForScale = THREE.MathUtils.lerp(userShoulderWorld, userBodyWidthWorld, 0.36);
        }
        const uBase = this.getUniformGarmentScale();
        const denom = this.getGarmentScaleRefWidth();
        /** ~0.2% slack vs MP shoulder span (≈ sub‑mm feel in ortho space at typical FOV). */
        const shoulderFit = 0.998;
        const uCapShoulder = (widthForScale * shoulderFit) / denom;
        const dPoseY = Math.abs(poseShoulderMid.y - poseHipMid.y);
        const dGarmentY = Math.max(0.02, this._garmentShoulderHemLocalDy || 0.25);
        const uCapTorso = THREE.MathUtils.clamp(dPoseY / dGarmentY, 0.1, 2.85);
        const uCenter = uCapShoulder;
        const uAlt = uCapTorso;
        const lowerGuard = Math.min(uCenter, uAlt) * 0.82;
        const upperGuard = Math.max(uCenter, uAlt) * 1.24;
        // Keep structural scale driven by skeleton fit; only guard obvious one-frame spikes.
        let uRaw = THREE.MathUtils.clamp(uBase, lowerGuard, upperGuard);

        // Prevent profile shrink: capture a stable reference from FRONT and enforce a soft floor in SIDE.
        if (this.debugInfo.viewAngle === 'FRONT') {
            const prevHold = this._lastFrontScale;
            this._lastFrontScale = prevHold == null ? uRaw : THREE.MathUtils.lerp(prevHold, uRaw, 0.12);
        } else if (this.debugInfo.viewAngle === 'SIDE') {
            const hold = this._lastFrontScale;
            if (hold != null && Number.isFinite(hold) && hold > 0.05) {
                uRaw = Math.max(uRaw, hold * 0.96);
            }
        }
        const prevScale = this.smoothedScale;
        const scaleBlend = prevScale != null && uRaw < prevScale ? 0.62 : 0.42;
        this.smoothedScale = prevScale == null
            ? uRaw
            : THREE.MathUtils.lerp(prevScale, uRaw, scaleBlend);
        const u = THREE.MathUtils.clamp(this.smoothedScale, 0.08, 3.0);
        this.smoothedScale = u;

        this.debugInfo.computedScale = this.fitStructuralMul == null ? 1 : this.fitStructuralMul;
        this.debugInfo.clampedScale = u;

        // Baseline trim: the auto fit tends to come out a touch oversized —
        // the mesh's shoulder bones sit a bit inboard of the actual shoulder
        // geometry, so matching bone-distance to MediaPipe shoulder span
        // over-sizes the shirt and it visibly rides above the user's
        // shoulders. An 8% global trim brings the silhouette in line with
        // the user's real frame without affecting the shoulder-anchored
        // position math (which composes with this value).
        const trim = THREE.MathUtils.clamp(Number(this.settings.autoScaleTrim) || AR_TRYON_SCALE_TRIM, 0.65, 1.15);
        const squeezeX = THREE.MathUtils.clamp(Number(this.settings.autoShoulderSqueezeX) || 1, 0.75, 1.05);
        const applied = u * trim;
        this.garmentGroup.scale.set(applied * squeezeX, applied, applied);

        // ========================================
        // ROTATION - YAW, PITCH, and ROLL
        // ========================================
        
        // Image x: person's left shoulder (MP 11) is usually on the camera-right → higher x than MP 12.
        const rawDiff = rs.x - ls.x;
        const shoulderSepNorm = Math.abs(rawDiff);
        const isSideView = this.debugInfo.viewAngle === 'SIDE';
        const isAmbiguousSep = shoulderSepNorm < 0.1;
        const nose = pose[this.MP.NOSE];
        const noseOk = nose && (nose.visibility || 0) > 0.35;
        const shoulderMidZ = (ls.z + rs.z) * 0.5;

        const faceHint = this._computeFaceFacingOverride(pose, ls, rs);
        if (faceHint === 'front') {
            this._hystFacingAway = false;
            this.debugInfo.faceFacingSource = 'face';
        } else {
            this._legacyFacingHysteresisUpdate(
                pose, ls, rs, isSideView, isAmbiguousSep, rawDiff, nose, noseOk, shoulderMidZ, shoulderSepNorm
            );
            this.debugInfo.faceFacingSource = 'legacy';
        }
        const isFacingAway = this._hystFacingAway;
        
        // Body yaw from shoulder depth (raw landmark z). Ortho camera scale.x = -1 mirrors the *rendered* scene,
        // but this yaw is derived in unmirrored landmark space — without a sign fix the shirt twists vs the selfie feed.
        const shoulderDepthDiff = ls.z - rs.z;
        const yawSign = this.settings.mirrorView ? 1 : -1;
        // SIDE: small shoulder separation makes z depth very sensitive — high gain was rotating the mesh
        // heavily and sliding the shirt horizontally (felt “mirrored” on one profile). Use moderate gain + clamp.
        const yawGain = isSideView ? 2.05 : (isAmbiguousSep ? 4.75 : 2);
        let estimatedYaw = yawSign * shoulderDepthDiff * yawGain;
        const shoulderSep = Math.abs(rawDiff);
        const yawReliability = THREE.MathUtils.clamp((shoulderSep - 0.055) / 0.14, 0.2, 1.0);
        estimatedYaw *= yawReliability;
        if (isSideView) {
            estimatedYaw = THREE.MathUtils.clamp(estimatedYaw, -1.12, 1.12);
        }
        // 180° for facing away; hysteresis above prevents boundary flicker (no extra shoulder-width gate)
        // Smoothly blend front->back around profile; binary PI jumps near side cause misalignment.
        const frontalness = THREE.MathUtils.clamp((shoulderSepNorm - 0.075) / 0.16, 0, 1);
        if (isFacingAway) {
            // When shoulders get close (profile/back), frontalness→0. We actually want MORE of the 180° turn there,
            // otherwise yaw hovers near 0 and rapidly flips as MediaPipe swaps left/right for the back-facing pose.
            const backness = 1 - frontalness;
            estimatedYaw += Math.PI * backness;
            // Back-facing + low shoulder separation is the most unstable regime (landmarks often swap).
            // Hold yaw near the current smoothed value so we don't "freak out" around the 180° boundary.
            if (shoulderSepNorm < 0.085 && this.smoothedYaw != null) {
                estimatedYaw = this.smoothedYaw;
            }
        }
        
        const shoulderAvgZ = (ls.z + rs.z) / 2;
        const hipAvgZ = (lh.z + rh.z) / 2;
        const zLean = shoulderAvgZ - hipAvgZ;
        
        let estimatedPitch = zLean * 2.15;
        if (this.settings.mirrorView) estimatedPitch = -estimatedPitch;
        estimatedPitch = THREE.MathUtils.clamp(estimatedPitch, -0.36, 0.36);
        
        const shoulderHeightDiff = RS.y - LS.y;
        const estimatedRoll = shoulderHeightDiff * (this.settings.mirrorView ? -1 : 1);
        
        if (this.smoothedYaw === undefined) this.smoothedYaw = 0;
        if (this.smoothedPitch === undefined) this.smoothedPitch = 0;
        if (this.smoothedRoll === undefined) this.smoothedRoll = 0;
        
        this.smoothedYaw = lerpAngle(this.smoothedYaw, estimatedYaw, 0.15);
        this.smoothedPitch = 0.4 * estimatedPitch + 0.6 * this.smoothedPitch;
        this.smoothedRoll = 0.25 * estimatedRoll + 0.75 * this.smoothedRoll;
        
        this.debugInfo.roll = this.smoothedRoll;
        this.debugInfo.pitch = this.smoothedPitch;
        this.debugInfo.zLean = zLean;
        this.debugInfo.facingAway = isFacingAway;
        
        this.garmentGroup.rotation.set(
            this.smoothedPitch,
            THREE.MathUtils.degToRad(this.settings.rotationY) + this.smoothedYaw,
            this.smoothedRoll
        );

        // ========================================
        // POSITIONING — shoulder bind ↔ MP 11/12; hem bind ↔ hip line (23/24 or estimate)
        // ========================================
        const poseAnchor = poseShoulderMid.clone();
        poseAnchor.z = 0;

        // Automatic vertical correction: lower the anchor slightly so the garment shoulder seam
        // visually matches the user's shoulders (MP landmarks sit a touch high on many feeds).
        const dropFrac = THREE.MathUtils.clamp(Number(this.settings.autoShoulderDrop) || 0, 0, 0.18);
        const viewDropMul = this.debugInfo.viewAngle === 'SIDE' ? 0.55 : (this.debugInfo.viewAngle === 'ANGLE' ? 0.8 : 1.0);
        const autoDrop = THREE.MathUtils.clamp(dropFrac * dPoseYWorld * viewDropMul, 0, 0.18);
        poseAnchor.y -= autoDrop;
        this.debugInfo.shoulderDropWorld = autoDrop;

        const shoulderSepNormPos = Math.abs(rs.x - ls.x);
        const useProfileAnchor = this.debugInfo.viewAngle === 'SIDE' || shoulderSepNormPos < 0.18;
        if (useProfileAnchor) {
            // Profile: 2D midpoint of LS/RS is often biased (occluded shoulder drifts in x). Anchor x to the
            // shoulder closer to the camera (smaller MediaPipe z) so left vs right profile stay consistent
            // with the mirrored selfie camera, instead of sliding the model the wrong way.
            const zl = ls.z || 0;
            const zr = rs.z || 0;
            if (!this._profileAnchorSide) {
                this._profileAnchorSide = zr <= zl ? 'right' : 'left';
            } else {
                const depthSwapMargin = 0.03;
                const side = this._profileAnchorSide;
                const curDepth = side === 'right' ? zr : zl;
                const otherDepth = side === 'right' ? zl : zr;
                const curVis = Math.max(0, side === 'right' ? (rs.visibility || 0) : (ls.visibility || 0));
                const otherVis = Math.max(0, side === 'right' ? (ls.visibility || 0) : (rs.visibility || 0));
                const depthWantsSwap = otherDepth + depthSwapMargin < curDepth;
                const visWantsSwap = curVis < 0.18 && otherVis > curVis + 0.08;
                if (depthWantsSwap || visWantsSwap) {
                    this._profileAnchorSide = side === 'right' ? 'left' : 'right';
                }
            }
            const rightCloser = this._profileAnchorSide === 'right';
            const pFwd = rightCloser ? pRS : pLS;
            const pBack = rightCloser ? pLS : pRS;
            const vFwd = Math.max(0.14, rightCloser ? (rs.visibility || 0) : (ls.visibility || 0));
            const vBack = Math.max(0.14, rightCloser ? (ls.visibility || 0) : (rs.visibility || 0));
            const w = vFwd / (vFwd + vBack);
            // Side gets strong front-shoulder anchoring; angled/back-angled keeps a softer bias.
            const profileStrength = this.debugInfo.viewAngle === 'SIDE'
                ? 1
                : THREE.MathUtils.clamp((0.18 - shoulderSepNormPos) / 0.1, 0, 1);
            const fwdBase = THREE.MathUtils.lerp(0.54, 0.62, profileStrength);
            const fwdGain = THREE.MathUtils.lerp(0.12, 0.28, profileStrength);
            const wFwd = fwdBase + fwdGain * w;
            const wBack = 1 - wFwd;
            poseAnchor.x = pFwd.x * wFwd + pBack.x * wBack;
            if (hipVisPose) {
                const pLH = this.poseToWorld(lh);
                const pRH = this.poseToWorld(rh);
                const vhl = Math.max(0.08, lh.visibility || 0);
                const vhr = Math.max(0.08, rh.visibility || 0);
                const hipMidX = (pLH.x * vhl + pRH.x * vhr) / (vhl + vhr);
                const hipMix = THREE.MathUtils.lerp(0.14, 0.18, profileStrength);
                poseAnchor.x = poseAnchor.x * (1 - hipMix) + hipMidX * hipMix;
            }
        } else {
            this._profileAnchorSide = null;
        }

        const poseBodyCenter = poseAnchor.clone();
        // Use a torso-centered anchor (not just shoulders). Shoulders are reliable but sit high; hips stabilize
        // center-of-mass so the garment doesn't "float" or slide during turns.
        const vAngle = String(this.debugInfo.viewAngle || 'FRONT').toUpperCase();
        const torsoMix =
            vAngle === 'SIDE' ? 0.62 :
                (vAngle === 'ANGLE' ? 0.42 : 0.26);
        poseBodyCenter.x = THREE.MathUtils.lerp(poseAnchor.x, poseHipMid.x, torsoMix);
        poseBodyCenter.y = THREE.MathUtils.lerp(poseAnchor.y, poseHipMid.y, 0.5);
        poseBodyCenter.z = 0;

        let targetPos;
        if (this.garmentBindBodyCenter) {
            const bindOff = (local) => {
                const v = local.clone();
                v.applyQuaternion(this.garmentGroup.quaternion);
                v.multiply(this.garmentGroup.scale);
                return v;
            };
            const bodyOff = bindOff(this.garmentBindBodyCenter);
            targetPos = poseBodyCenter.clone().sub(bodyOff);
            targetPos.z = 0;

            if (this.garmentBindShoulderMid || this.garmentBindHemMid) {
                // Lock the garment's shoulder line to the user's detected
                // shoulder line (shoulders are what the eye reads most; if
                // they drift, the shirt looks like it's floating above the
                // body). Hem gets a much softer nudge so it doesn't fight
                // the shoulder anchor when torso proportions don't match.
                let yCorrection = 0;
                if (this.garmentBindShoulderMid) {
                    const shoulderOff = bindOff(this.garmentBindShoulderMid);
                    const shY = targetPos.y + shoulderOff.y;
                    yCorrection += (poseAnchor.y - shY) * 0.78;
                }
                if (this.garmentBindHemMid) {
                    const hemOff = bindOff(this.garmentBindHemMid);
                    const hemY = targetPos.y + hemOff.y;
                    yCorrection += (poseHipMid.y - hemY) * 0.30;
                }
                targetPos.y += yCorrection;
            }
        } else if (this.garmentBindShoulderMid) {
            const bindOff = (local) => {
                const v = local.clone();
                v.applyQuaternion(this.garmentGroup.quaternion);
                v.multiply(this.garmentGroup.scale);
                return v;
            };
            const shoulderOff = bindOff(this.garmentBindShoulderMid);
            targetPos = poseAnchor.clone().sub(shoulderOff);
            targetPos.z = 0;

            if (this.garmentBindHemMid) {
                const hemOff = bindOff(this.garmentBindHemMid);
                const hemY = targetPos.y + hemOff.y;
                const hemErr = poseHipMid.y - hemY;
                targetPos.y += hemErr * 0.94;
                const shoulderOff2 = bindOff(this.garmentBindShoulderMid);
                const shY = targetPos.y + shoulderOff2.y;
                targetPos.y += (poseAnchor.y - shY) * 0.28;
            }
        } else {
            const shoulderCenterX = (LS.x + RS.x) / 2;
            const nudgeDown = torsoHeightScreen * 0.08;
            const tcx = shoulderCenterX;
            const tcy = shoulderCenterY + nudgeDown;
            targetPos = new THREE.Vector3(
                (tcx - 0.5) * 2 + this.settings.offsetX,
                -(tcy - 0.5) * 2 + this.settings.offsetY,
                0
            );
        }

        // Apply user offsets LAST so the structural anchors don't cancel them every frame.
        // (Particularly Y: shoulder/hem corrections would otherwise override the slider.)
        targetPos.x += Number(this.settings.offsetX) || 0;
        targetPos.y += Number(this.settings.offsetY) || 0;
        targetPos.z = 0;

        this.smoothedPosition.lerp(targetPos, 0.38);
        this.garmentGroup.position.copy(this.smoothedPosition);

        const matKey = `${isFacingAway}|${this.debugInfo.viewAngle}|${this.settings.frontFaceOnly}|${this.settings.wireframe}|${this.settings.garmentCullInvert}`;
        if (matKey !== this._lastMatState) {
            this._lastMatState = matKey;
            this.applyMaterialSettings();
        }
    }
    
    
    drawSkeleton(pose) {
        const w = this.debugCanvas.width;
        const h = this.debugCanvas.height;
        drawSkeletonOverlay(this.debugCtx, w, h, pose, this.settings.mirrorView);
    }

    /** Coloured dots on mapped MediaPipe indices (needs garment.json from pipeline). */
    drawMpMappingOverlay(pose) {
        const w = this.debugCanvas.width;
        const h = this.debugCanvas.height;
        drawMpMappingLayer(this.debugCtx, w, h, pose, this.garmentMetadata, this.settings.mirrorView);
    }

    drawHud(conf) {
        const w = this.debugCanvas.width;
        drawHudOverlay(this.debugCtx, w, conf, {
            debugInfo: this.debugInfo,
            settings: this.settings,
            garmentShoulderWidth: this.garmentShoulderWidth,
            smoothedScale: this.smoothedScale
        });
    }
    
    // ── Generate-from-photos pipeline ─────────────────────────────────────────

    async loadGarmentFromUrl(url, jsonUrl = null) {
        try {
            this.showLoading('Loading generated garment...');
            if (this.garmentGroup) this.scene.remove(this.garmentGroup);
            this.garmentModel = null; this.skinnedMesh = null;
            this._rigidMeshes = [];
            this.skeleton = null; this.bones = {};
            this._sessionGarmentLabel = null;
            this.fitStructuralMul = null;
            this.smoothedScale = null;
            this.garmentMetadata = null;
            this.garmentBindBodyCenter = null;
            this._hystFacingAway = false;
            this._facingAwayScore = 0;
            this._lastFrontScale = null;
            this._lastMatFacingAway = null;
            this._lastMatState = '';
            this.smoothedYaw = 0;
            this.smoothedPitch = 0;
            this.smoothedRoll = 0;
            const loader = new GLTFLoader();
            loader.setCrossOrigin('anonymous');
            const gltf = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
            this._assignGarmentFromGltf(gltf);
            this._promoteVertexColorGarmentMaterials(gltf.scene);
            this._tameRigidGarmentMetallic(gltf.scene);
            this.applyMaterialSettings();
            this.cacheSkeletonBones();
            this.garmentGroup = new THREE.Group();
            this.garmentModel = gltf.scene;
            this.garmentGroup.add(this.garmentModel);
            this.scene.add(this.garmentGroup);
            this.measureGarmentFromBones();
            const modeLabel = this.skinnedMesh
                ? 'Skinned (rigged)'
                : 'Rigid (no bones — e.g. SAM3 mesh)';
            const info = document.getElementById('garment-info');
            info.style.display = 'block';
            info.innerHTML = `${modeLabel}<br>Mesh: ${this._garmentDisplayName()}<br>Bones: ${Object.keys(this.bones).length}<br>Shoulder Width: ${this.garmentShoulderWidth?.toFixed(4) || 'N/A'}<br>Scale ref: ${this.garmentScaleRefWidth?.toFixed(4) || 'N/A'}<br>Torso Height: ${this.garmentTorsoHeight?.toFixed(4) || 'N/A'}`;
            await this._fetchGarmentMetadata(jsonUrl);
            try {
                const u = new URL(url, location.href);
                const seg = u.pathname.split('/').filter(Boolean);
                this._sessionGarmentLabel = seg.length ? seg[seg.length - 1] : url;
            } catch {
                this._sessionGarmentLabel = String(url).slice(-64);
            }
            this.showToast('Garment loaded — automatic body fit when your shoulders are visible', 'success');
            this.hideLoading();
        } catch (e) {
            this.showToast('Load failed: ' + e.message, 'error');
            this.hideLoading();
        }
    }

    async _fetchGarmentMetadata(jsonUrl) {
        this.garmentMetadata = null;
        if (!jsonUrl) return;
        try {
            const u = jsonUrl.startsWith('http') ? jsonUrl : this.getApiOrigin() + jsonUrl;
            const r = await fetch(u);
            if (!r.ok) return;
            this.garmentMetadata = await r.json();
            console.log('[garment.json] mediapipe_mapping keys:', Object.keys(this.garmentMetadata.mediapipe_mapping || {}).length);
            this._applyGarmentCalibrationFromMetadata();
        } catch (e) {
            console.warn('garment.json fetch', e);
        }
    }

    /**
     * Prefer explicit skeleton calibration exported by the pipeline over bbox heuristics.
     * This keeps auto-scale denominator aligned with the SMPL skeleton measurements.
     */
    _applyGarmentCalibrationFromMetadata() {
        const calib = this.garmentMetadata?.calibration;
        if (!calib) return;

        let touched = false;
        const shoulder = Number(calib.smpl_shoulder_width);
        if (Number.isFinite(shoulder) && shoulder > 0.05) {
            this.garmentShoulderWidth = shoulder;
            this.garmentScaleRefWidth = shoulder;
            touched = true;
        }

        const torso = Number(calib.smpl_torso_height);
        if (Number.isFinite(torso) && torso > 0.06) {
            this.garmentTorsoHeight = torso;
            touched = true;
        }

        if (!touched) return;
        this._recomputeGarmentShoulderHemDelta();
        console.log(
            '[garment.json] Applied skeleton calibration:',
            'shoulder=', this.garmentShoulderWidth?.toFixed(4),
            'scaleRef=', this.garmentScaleRefWidth?.toFixed(4),
            'torso=', this.garmentTorsoHeight?.toFixed(4)
        );

        const info = document.getElementById('garment-info');
        if (info && info.style.display !== 'none') {
            const modeLabel = this.skinnedMesh
                ? 'Skinned (rigged)'
                : 'Rigid (no bones — e.g. SAM3 mesh)';
            info.innerHTML = `
                ${modeLabel}<br>
                Mesh: ${this._garmentDisplayName() || 'N/A'}<br>
                Bones: ${Object.keys(this.bones || {}).length}<br>
                Shoulder Width: ${this.garmentShoulderWidth?.toFixed(4) || 'N/A'}<br>
                Scale ref: ${this.garmentScaleRefWidth?.toFixed(4) || 'N/A'}<br>
                Torso Height: ${this.garmentTorsoHeight?.toFixed(4) || 'N/A'}
            `;
        }
    }

    async refreshGarmentLibrarySelect() {
        const sel = document.getElementById('garment-library-select');
        if (!sel) return;
        sel.innerHTML = '<option value="">— none —</option>';
        try {
            const res = await fetch(this.getApiOrigin() + '/api/garments');
            if (!res.ok) throw new Error(String(res.status));
            const data = await res.json();
            const list = data.garments || [];
            for (const g of list) {
                const o = document.createElement('option');
                o.value = JSON.stringify({ glb_url: g.glb_url, json_url: g.json_url });
                o.textContent = g.id + (g.json_url ? ' ✓' : '');
                sel.appendChild(o);
            }
            if (list.length) this.showToast(`Library: ${list.length} garment(s)`, 'success');
        } catch (e) {
            this.showToast('Could not list garments (start server)', 'warning');
        }
    }




    /** Wall-clock duration from server `elapsed_sec` (or client fallback). */
    formatGenElapsed(sec) {
        if (sec == null || !Number.isFinite(Number(sec))) return '';
        const s = Number(sec);
        if (s < 60) {
            const r = Math.round(s * 10) / 10;
            return Number.isInteger(r) ? `${r}s` : `${r.toFixed(1)}s`;
        }
        const m = Math.floor(s / 60);
        const rs = Math.floor(s % 60);
        return `${m}m ${rs}s`;
    }

    _revokeGenPreviewUrl(which) {
        const key = which === 'front' ? '_genFrontPreviewUrl' : '_genBackPreviewUrl';
        const u = this[key];
        if (u) {
            try { URL.revokeObjectURL(u); } catch (_) { /* ignore */ }
            this[key] = null;
        }
    }

    async _jpegFileFromCanvas(canvas, sourceFile) {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
        if (!blob || blob.size < 32) return null;
        const base = (sourceFile.name && sourceFile.name.replace(/\.[^./\\]+$/, '')) || 'photo';
        return new File([blob], `${base}.jpg`, { type: 'image/jpeg', lastModified: Date.now() });
    }

    async _normalizeGarmentViaBitmap(file, withImageOrientation) {
        const maxDim = 4096;
        const bmp = withImageOrientation
            ? await createImageBitmap(file, { imageOrientation: 'from-image' })
            : await createImageBitmap(file);
        try {
            let w = bmp.width;
            let h = bmp.height;
            if (w > maxDim || h > maxDim) {
                const s = maxDim / Math.max(w, h);
                w = Math.round(w * s);
                h = Math.round(h * s);
            }
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('no 2d context');
            ctx.drawImage(bmp, 0, 0, w, h);
            const out = await this._jpegFileFromCanvas(canvas, file);
            if (!out) throw new Error('toBlob');
            return out;
        } finally {
            if (typeof bmp.close === 'function') bmp.close();
        }
    }

    async _normalizeGarmentViaImageElement(file) {
        const maxDim = 4096;
        const url = URL.createObjectURL(file);
        try {
            const img = await new Promise((resolve, reject) => {
                const el = new Image();
                el.onload = () => resolve(el);
                el.onerror = () => reject(new Error('image decode'));
                el.src = url;
            });
            let w = img.naturalWidth;
            let h = img.naturalHeight;
            if (!w || !h) throw new Error('zero dimensions');
            if (w > maxDim || h > maxDim) {
                const s = maxDim / Math.max(w, h);
                w = Math.round(w * s);
                h = Math.round(h * s);
            }
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('no 2d context');
            ctx.drawImage(img, 0, 0, w, h);
            const out = await this._jpegFileFromCanvas(canvas, file);
            if (!out) throw new Error('toBlob');
            return out;
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    /**
     * Raster suitable for POST (JPEG when decoding works — fixes HEIC / mislabeled PNG on server).
     * Bounded time: mobile Safari can stall inside createImageBitmap; fall back to original File.
     */
    async normalizeGarmentPhotoForUpload(file, timeoutMs = 14000) {
        const inner = async () => {
            try {
                return await this._normalizeGarmentViaBitmap(file, true);
            } catch (_) {
                try {
                    return await this._normalizeGarmentViaBitmap(file, false);
                } catch (_) {
                    try {
                        return await this._normalizeGarmentViaImageElement(file);
                    } catch (_) {
                        return file;
                    }
                }
            }
        };
        try {
            return await Promise.race([
                inner(),
                new Promise((_, rej) => setTimeout(() => rej(new Error('normalize-timeout')), timeoutMs)),
            ]);
        } catch (_) {
            return file;
        }
    }

    _setGenerateSlotFromFile(which, raw, placeholder = 'Tap to add photo') {
        if (!raw || !raw.size) return;
        const previewId = which === 'front' ? 'gen-front-preview' : 'gen-back-preview';
        const labelId = which === 'front' ? 'gen-front-label' : 'gen-back-label';
        const p = document.getElementById(previewId);
        const l = document.getElementById(labelId);
        if (!p || !l) return;

        if (which === 'front') this._genFrontFile = raw;
        else this._genBackFile = raw;

        this._revokeGenPreviewUrl(which);
        const url = URL.createObjectURL(raw);
        if (which === 'front') this._genFrontPreviewUrl = url;
        else this._genBackPreviewUrl = url;

        p.onload = () => {
            try { URL.revokeObjectURL(url); } catch (_) { /* ignore */ }
            if (which === 'front') this._genFrontPreviewUrl = null;
            else this._genBackPreviewUrl = null;
        };
        p.onerror = () => {
            try { URL.revokeObjectURL(url); } catch (_) { /* ignore */ }
            if (which === 'front') this._genFrontPreviewUrl = null;
            else this._genBackPreviewUrl = null;
            p.classList.add('hidden');
            l.classList.remove('hidden');
            l.textContent = 'Selected ✓';
            this.updateGenerateBtn();
        };
        p.src = url;
        p.classList.remove('hidden');
        l.classList.add('hidden');
        l.textContent = placeholder;
        this.updateGenerateBtn();
    }

    async _assetImageToFile(url, fallbackName) {
        const candidates = [];
        const pushUnique = (u) => {
            if (!u || candidates.includes(u)) return;
            candidates.push(u);
        };
        pushUnique(url);
        if (url.startsWith('/')) {
            pushUnique(`/public${url}`);
            pushUnique(this.getApiOrigin() + url);
            pushUnique(this.getApiOrigin() + `/public${url}`);
        }

        let lastErr = null;
        for (const candidate of candidates) {
            try {
                const res = await fetch(candidate);
                if (!res.ok) {
                    lastErr = new Error(`HTTP ${res.status} for ${candidate}`);
                    continue;
                }
                const blob = await res.blob();
                if (!blob || blob.size < 64) {
                    lastErr = new Error(`Invalid image payload for ${candidate}`);
                    continue;
                }
                const extFromType = (blob.type || '').includes('png') ? 'png' : 'jpg';
                const fileName = fallbackName.includes('.') ? fallbackName : `${fallbackName}.${extFromType}`;
                return new File([blob], fileName, { type: blob.type || `image/${extFromType}`, lastModified: Date.now() });
            } catch (e) {
                lastErr = e;
            }
        }
        throw new Error(`Could not load ${fallbackName}${lastErr ? ` (${lastErr.message || String(lastErr)})` : ''}`);
    }

    async loadQuickGenerationPreset() {
        const quickBtn = document.getElementById('gen-quick-mu-btn');
        const statusEl = document.getElementById('gen-status');
        if (quickBtn) quickBtn.disabled = true;
        if (statusEl) {
            statusEl.className = 'gen-status busy';
            statusEl.textContent = 'Loading quick photos…';
        }
        try {
            const [front, back] = await Promise.all([
                this._assetImageToFile('/quick-generation/mu-front.png', 'mu-front.png'),
                this._assetImageToFile('/quick-generation/mu-back.png', 'mu-back.png'),
            ]);
            this._setGenerateSlotFromFile('front', front, 'Quick photo ready');
            this._setGenerateSlotFromFile('back', back, 'Quick photo ready');
            if (statusEl) {
                statusEl.className = 'gen-status ok';
                statusEl.textContent = 'Quick photos loaded. Generating…';
            }
            this.showToast('Quick photos loaded', 'success');
        } catch (e) {
            if (statusEl) {
                statusEl.className = 'gen-status fail';
                statusEl.textContent = 'Quick photo load failed';
            }
            this.showToast('Quick photo load failed: ' + ((e && e.message) || String(e)), 'error');
        } finally {
            if (quickBtn) quickBtn.disabled = false;
        }
    }

    async generateGarment() {
        const statusEl   = document.getElementById('gen-status');
        const genBtn     = document.getElementById('gen-generate-btn');
        const frontFile = this._genFrontFile;
        const backFile = this._genBackFile;
        if (!frontFile || !backFile) {
            this.showToast('Select both front and back images first', 'error'); return;
        }
        if (this._generationInProgress) return;
        this._generationInProgress = true;
        genBtn.disabled = true;
        statusEl.className = 'gen-status busy';
        statusEl.textContent = 'Preparing images…';
        let [frontReady, backReady] = await Promise.all([
            this.normalizeGarmentPhotoForUpload(frontFile),
            this.normalizeGarmentPhotoForUpload(backFile),
        ]);
        this._genFrontFile = frontReady;
        this._genBackFile = backReady;

        const mobileLike = this._isCoarseOrMobileDevice();
        const combinedBytes = frontReady.size + backReady.size;
        const maxEdge = mobileLike ? 1792 : 2560;
        if (mobileLike || combinedBytes > 5 * 1024 * 1024) {
            statusEl.textContent = 'Optimizing size for upload…';
            const [f2, b2] = await Promise.all([
                this._downscaleForUpload(frontReady, maxEdge),
                this._downscaleForUpload(backReady, maxEdge),
            ]);
            frontReady = f2;
            backReady = b2;
            this._genFrontFile = frontReady;
            this._genBackFile = backReady;
        }

        const measureImg = (file) => new Promise((resolve) => {
            const img = new Image();
            const u = URL.createObjectURL(file);
            img.onload = () => { URL.revokeObjectURL(u); resolve({ w: img.naturalWidth, h: img.naturalHeight }); };
            img.onerror = () => { URL.revokeObjectURL(u); resolve(null); };
            img.src = u;
        });
        const [fdim, bdim] = await Promise.all([
            measureImg(frontReady),
            measureImg(backReady)
        ]);
        if (fdim && (fdim.w < 640 || fdim.h < 640)) {
            this.showToast(
                'Front photo is small — use ≥900px longest side for clear logos/prints; flat-lay, plain background',
                'warning'
            );
        }
        if (bdim && (bdim.w < 640 || bdim.h < 640)) {
            this.showToast('Back photo is small — use ≥900px longest side; match lighting to front', 'warning');
        }
        const toBase64 = (f) => new Promise((res, rej) => {
            const r = new FileReader();
            r.onload = () => res(r.result);
            r.onerror = () => rej(new Error('Could not read photos on this device — try Export JPEG in Photos'));
            r.readAsDataURL(f);
        });
        this._genWallStart = Date.now();
        statusEl.textContent = 'Uploading images…';
        try {
            try {
                const cr = await fetch(this.getApiOrigin() + '/api/config');
                if (cr.ok) {
                    const cfg = await cr.json();
                    if (!cfg.ready) {
                        this.showToast('Server not ready: missing ' + this._formatConfigMissing(cfg), 'error');
                        genBtn.disabled = false;
                        statusEl.className = 'gen-status fail';
                        statusEl.textContent = 'Fix paths in .env — ' + this._formatConfigMissing(cfg);
                        return;
                    }
                }
            } catch (_) { /* offline: POST may still be tried */ }

            let frontB64;
            let backB64;
            if (mobileLike) {
                frontB64 = await toBase64(frontReady);
                backB64 = await toBase64(backReady);
            } else {
                [frontB64, backB64] = await Promise.all([toBase64(frontReady), toBase64(backReady)]);
            }

            const payload = JSON.stringify({ front_image: frontB64, back_image: backB64 });
            if (payload.length > 28 * 1024 * 1024) {
                throw new Error('Images still too large after resize — pick smaller originals or lower camera resolution');
            }

            const ac = new AbortController();
            const uploadMs = mobileLike ? 240000 : 120000;
            const t = setTimeout(() => ac.abort(), uploadMs);
            let res;
            try {
                res = await fetch(this.getApiOrigin() + '/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload,
                    signal: ac.signal,
                });
            } finally {
                clearTimeout(t);
            }
            if (!res.ok) {
                let detail = '';
                try {
                    const j = await res.json();
                    if (j && j.error) detail = ': ' + j.error;
                } catch (_) { /* ignore */ }
                throw new Error(`Server error ${res.status}${detail}`);
            }
            const { task_id } = await res.json();
            const up = this.formatGenElapsed((Date.now() - this._genWallStart) / 1000);
            statusEl.textContent = up
                ? `Queued… (upload ${up}, task ${task_id.slice(0, 8)}…)`
                : `Queued… (task ${task_id.slice(0, 8)}…)`;
            this.pollGenerationStatus(task_id);
        } catch (e) {
            this._genWallStart = null;
            statusEl.className = 'gen-status fail';
            const name = e && e.name;
            const msg =
                name === 'AbortError'
                    ? 'Upload timed out — use Wi‑Fi or smaller photos (we resize on phone; very large originals can still fail).'
                    : ((e && e.message) || String(e) || 'Upload failed');
            statusEl.textContent = 'Error: ' + msg;
            this.showToast(msg, 'error');
            genBtn.disabled = false;
            this._generationInProgress = false;
        }
    }

    pollGenerationStatus(taskId) {
        const statusEl = document.getElementById('gen-status');
        const genBtn   = document.getElementById('gen-generate-btn');
        const retryBtn = document.getElementById('gen-retry-btn');
        let connErrors = 0;
        const MAX_CONN_ERRORS = 5;
        let intervalId = null;
        const tick = async () => {
            try {
                const res  = await fetch(`${this.getApiOrigin()}/api/status/${taskId}`);
                if (!res.ok) throw new Error(`Server returned ${res.status}`);
                connErrors = 0;
                const data = await res.json();
                if (data.status === 'processing' || data.status === 'pending') {
                    let es = data.elapsed_sec;
                    if (es == null && this._genWallStart != null) {
                        es = (Date.now() - this._genWallStart) / 1000;
                    }
                    const tStr = this.formatGenElapsed(es);
                    const pct = data.progress !== undefined ? ` (${Math.round(data.progress)}%)` : '';
                    const msg = data.message ? ` — ${data.message}` : '';
                    const timeBit = tStr ? ` · ${tStr}` : '';
                    statusEl.textContent = 'Processing…' + timeBit + pct + msg;
                    if (this._lastGenLog !== `${data.progress}|${data.message}|${es}`) {
                        console.log('[GEN]', {
                            taskId, progress: data.progress, message: data.message,
                            elapsed_sec: data.elapsed_sec ?? null,
                            elapsed_display_sec: Number.isFinite(es) ? Math.round(es * 10) / 10 : null,
                            logs_tail: (data.logs && data.logs.length) ? data.logs.slice(-3) : undefined
                        });
                        this._lastGenLog = `${data.progress}|${data.message}|${es}`;
                    }
                } else if (data.status === 'completed') {
                    if (intervalId) clearInterval(intervalId);
                    this._genWallStart = null;
                    const totalStr = this.formatGenElapsed(data.elapsed_sec);
                    statusEl.className = 'gen-status ok';
                    statusEl.textContent = totalStr ? `Done in ${totalStr}. Loading model…` : 'Done! Loading model…';
                    genBtn.disabled = false;
                    this._lastGlbUrl = data.glb_url;
                    this._lastJsonUrl = data.json_url || null;
                    this._setGeneratedGarmentPending(data.glb_url, this._lastJsonUrl, totalStr);
                    this._generationInProgress = false;
                } else if (data.status === 'failed') {
                    if (intervalId) clearInterval(intervalId);
                    this._genWallStart = null;
                    statusEl.className = 'gen-status fail';
                    const failTime = this.formatGenElapsed(data.elapsed_sec);
                    const hint = data.hint ? ` — ${data.hint}` : '';
                    const prefix = failTime ? `Failed after ${failTime}: ` : 'Failed: ';
                    statusEl.textContent = prefix + (data.error || 'unknown') + hint;
                    if (data.hint) this.showToast(data.hint, 'error');
                    genBtn.disabled = false;
                    this._generationInProgress = false;
                }
            } catch (e) {
                connErrors++;
                const isNet = e.message.includes('fetch') || e.message.includes('Failed');
                if (isNet && connErrors < MAX_CONN_ERRORS) {
                    statusEl.textContent = `Server unreachable — retrying… (${connErrors}/${MAX_CONN_ERRORS})`;
                    return;
                }
                if (intervalId) clearInterval(intervalId);
                this._genWallStart = null;
                statusEl.className = 'gen-status fail';
                statusEl.textContent = isNet ? 'Server stopped. Use ↺ Retry if done.' : 'Error: ' + e.message;
                if (isNet) { this._lastPollTaskId = taskId; if (retryBtn) retryBtn.style.display = 'block'; }
                genBtn.disabled = false;
                this._generationInProgress = false;
            }
        };
        void tick();
        intervalId = setInterval(tick, 1000);
    }

    // ─────────────────────────────────────────────────────────────────────

    setupUI() {
        const mobileControlsToggle = document.getElementById('mobile-controls-toggle');
        const mobileControlsClose = document.getElementById('mobile-controls-close');
        const syncMobileUi = () => {
            const isMobileLayout = typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches;
            if (!isMobileLayout) {
                this.setMobileControlsOpen(false);
                return;
            }
            if (!this._mobileHudDefaultApplied) {
                this._mobileHudDefaultApplied = true;
                this.settings.showHud = false;
                document.getElementById('toggle-hud')?.classList.remove('active');
            }
        };
        mobileControlsToggle?.addEventListener('click', () => {
            const appContainer = document.querySelector('.app-container');
            const currentlyOpen = appContainer?.classList.contains('mobile-panel-open');
            this.setMobileControlsOpen(!currentlyOpen);
        });
        mobileControlsClose?.addEventListener('click', () => this.setMobileControlsOpen(false));
        window.addEventListener('resize', syncMobileUi);

        // File upload
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragging'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragging'));
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('dragging');
            if (e.dataTransfer.files.length) this.loadGarment(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) this.loadGarment(fileInput.files[0]);
        });

        this.setupGenerateUI();
        document.getElementById('gen-quick-mu-btn')?.addEventListener('click', () => {
            void this.loadQuickGenerationPreset();
        });

        // Camera
        document.getElementById('start-btn').addEventListener('click', () => {
            if (this.isRunning) this.stopCamera();
            else this.startCamera();
        });
        
        this.setupSlider('offset-y-slider', 'offset-y-value', v => this.settings.offsetY = v);
        this.setupSlider('offset-x-slider', 'offset-x-value', v => this.settings.offsetX = v);
        this.setupSlider('shoulder-drop-slider', 'shoulder-drop-value', v => this.settings.autoShoulderDrop = v);
        this.setupSlider('shoulder-squeeze-slider', 'shoulder-squeeze-value', v => this.settings.autoShoulderSqueezeX = v);
        this.setupSlider('rotation-y-slider', 'rotation-y-value', v => this.settings.rotationY = v, v => v + '°');
        this.setupSlider('smoothing-slider', 'smoothing-value', v => this.settings.smoothingFactor = v);
        this.setupSlider('auto-scale-trim-slider', 'auto-scale-trim-value', v => this.settings.autoScaleTrim = v);

        
        const poseIntervalSel = document.getElementById('pose-detect-interval');
        if (poseIntervalSel) {
            const syncPoseEvery = () => {
                const n = Math.max(1, Math.min(3, Math.round(this.settings.poseDetectEvery || 1)));
                this.settings.poseDetectEvery = n;
                poseIntervalSel.value = String(n);
            };
            syncPoseEvery();
            poseIntervalSel.addEventListener('change', () => {
                const n = Math.max(1, Math.min(3, parseInt(poseIntervalSel.value, 10) || 2));
                this.settings.poseDetectEvery = n;
                this._poseFrameTick = 0;
            });
        }
        
        // Rotation buttons
        document.getElementById('btn-flip-front').addEventListener('click', () => {
            this.settings.rotationY = 0;
            document.getElementById('rotation-y-slider').value = 0;
            document.getElementById('rotation-y-value').textContent = '0°';
        });
        document.getElementById('btn-flip-back').addEventListener('click', () => {
            this.settings.rotationY = 180;
            document.getElementById('rotation-y-slider').value = 180;
            document.getElementById('rotation-y-value').textContent = '180°';
        });
        document.getElementById('btn-reset').addEventListener('click', () => this.resetSettings());
        
        // Toggles
        this.setupToggle('toggle-frontface', a => { this.settings.frontFaceOnly = a; this.applyMaterialSettings(); });
        this.setupToggle('toggle-skeleton', a => this.settings.showSkeleton = a);
        this.setupToggle('toggle-mirror', a => {
            this.settings.mirrorView = a;
            this.applyMirrorViewTransforms();
            this._lastMatState = '';
            this.applyMaterialSettings();
        });
        this.setupToggle('toggle-wireframe', a => { this.settings.wireframe = a; this.applyMaterialSettings(); });
        this.setupToggle('toggle-garment-cull-invert', a => {
            this.settings.garmentCullInvert = a;
            this._lastMatState = '';
            this.applyMaterialSettings();
        });
        this.setupToggle('toggle-hud', a => this.settings.showHud = a);
        this.setupToggle('toggle-face-facing', a => { this.settings.useFaceFacingHint = a; });
        this.setupToggle('toggle-mp-mapping', a => this.settings.showMpMapping = a);
        this.setupToggle('toggle-session-log', a => {
            this.settings.logSession = a;
            if (!a) {
                this.sessionLog = null;
                return;
            }
            if (this.isRunning) initSessionLog(this);
        });
        this.setupToggle('toggle-log-landmarks', a => {
            this.settings.logIncludeLandmarks = a;
            if (this.sessionLog?.logOptions) {
                this.sessionLog.logOptions.includeLandmarks = a;
            }
        });
        const lmSel = document.getElementById('log-landmark-interval');
        if (lmSel) {
            lmSel.addEventListener('change', () => {
                syncLogLandmarkIntervalFromUI(this.settings);
                if (this.sessionLog?.logOptions) {
                    this.sessionLog.logOptions.landmark_sample_min_ms = this.settings.logLandmarkIntervalMs;
                }
            });
        }
        document.getElementById('session-participant-code')?.addEventListener('change', () => {
            if (!this.sessionLog) return;
            const v = (document.getElementById('session-participant-code').value || '').trim() || null;
            this.sessionLog.participantCode = v;
        });
        document.getElementById('btn-download-session-log')?.addEventListener('click', () => downloadSessionLogJson(this));
        document.getElementById('btn-refresh-garments')?.addEventListener('click', () => this.refreshGarmentLibrarySelect());
        document.getElementById('btn-load-library-glb')?.addEventListener('click', async () => {
            const sel = document.getElementById('garment-library-select');
            if (!sel?.value) {
                this.showToast('Pick a garment from the list (Refresh first)', 'warning');
                return;
            }
            try {
                const { glb_url, json_url } = JSON.parse(sel.value);
                this._lastGlbUrl = glb_url;
                this._lastJsonUrl = json_url || null;
                await this.loadGarmentFromUrl(
                    glb_url.startsWith('http') ? glb_url : this.getApiOrigin() + glb_url,
                    json_url
                );
            } catch (e) {
                this.showToast('Load error: ' + e.message, 'error');
            }
        });
        
        document.getElementById('btn-copy-settings').addEventListener('click', () => {
            navigator.clipboard.writeText(JSON.stringify(this.settings, null, 2));
            this.showToast('Settings copied', 'success');
        });
        
        document.getElementById('screenshot-btn').addEventListener('click', () => this.takeScreenshot());
        document.getElementById('reset-link').addEventListener('click', e => { e.preventDefault(); this.resetSettings(); });
        syncMobileUi();
    }
    
    setupGenerateUI() {
        const attachGenSlot = (input, which) => {
            const previewId = which === 'front' ? 'gen-front-preview' : 'gen-back-preview';
            const labelId = which === 'front' ? 'gen-front-label' : 'gen-back-label';
            let lastDedupeKey = '';
            let lastDedupeAt = 0;
            let pickGen = 0;
            const applyPreviewAndStore = (raw) => {
                const p = document.getElementById(previewId);
                const l = document.getElementById(labelId);
                if (which === 'front') this._genFrontFile = raw;
                else this._genBackFile = raw;
                this._revokeGenPreviewUrl(which);
                const url = URL.createObjectURL(raw);
                if (which === 'front') this._genFrontPreviewUrl = url;
                else this._genBackPreviewUrl = url;
                p.onload = () => {
                    try {
                        URL.revokeObjectURL(url);
                    } catch (_) { /* ignore */ }
                    if (which === 'front') this._genFrontPreviewUrl = null;
                    else this._genBackPreviewUrl = null;
                };
                p.onerror = () => {
                    try {
                        URL.revokeObjectURL(url);
                    } catch (_) { /* ignore */ }
                    if (which === 'front') this._genFrontPreviewUrl = null;
                    else this._genBackPreviewUrl = null;
                    p.classList.add('hidden');
                    l.classList.remove('hidden');
                    l.textContent = 'Selected ✓';
                    this.showToast('Photo added (thumbnail hidden here). You can still tap Generate.', 'warning');
                    this.updateGenerateBtn();
                };
                p.src = url;
                p.classList.remove('hidden');
                l.classList.add('hidden');
                l.textContent = 'Tap to add photo';
                this.updateGenerateBtn();
            };
            const processFile = async (raw) => {
                if (!raw || !raw.size) {
                    this.showToast('No image data — try again or pick a different photo', 'warning');
                    return;
                }
                const myGen = ++pickGen;
                applyPreviewAndStore(raw);
                setTimeout(() => {
                    try {
                        input.value = '';
                    } catch (_) { /* ignore */ }
                }, 350);
                try {
                    const normalized = await this.normalizeGarmentPhotoForUpload(raw);
                    if (myGen !== pickGen) return;
                    if (!normalized || normalized.size < 32) return;
                    if (normalized === raw) return;
                    if (which === 'front') this._genFrontFile = normalized;
                    else this._genBackFile = normalized;
                    const p = document.getElementById(previewId);
                    const l = document.getElementById(labelId);
                    const url2 = URL.createObjectURL(normalized);
                    this._revokeGenPreviewUrl(which);
                    if (which === 'front') this._genFrontPreviewUrl = url2;
                    else this._genBackPreviewUrl = url2;
                    p.onload = () => {
                        try {
                            URL.revokeObjectURL(url2);
                        } catch (_) { /* ignore */ }
                        if (which === 'front') this._genFrontPreviewUrl = null;
                        else this._genBackPreviewUrl = null;
                    };
                    p.onerror = () => {
                        try {
                            URL.revokeObjectURL(url2);
                        } catch (_) { /* ignore */ }
                        if (which === 'front') this._genFrontPreviewUrl = null;
                        else this._genBackPreviewUrl = null;
                        p.classList.add('hidden');
                        l.classList.remove('hidden');
                        l.textContent = 'Selected ✓';
                        this.updateGenerateBtn();
                    };
                    p.src = url2;
                    p.classList.remove('hidden');
                    l.classList.add('hidden');
                } catch {
                    /* keep raw in _genFrontFile / _genBackFile */
                }
            };
            const onNativePick = () => {
                const raw = input.files && input.files[0];
                if (!raw) return;
                const key = `${raw.size}\0${raw.lastModified}\0${raw.name || ''}`;
                const t = Date.now();
                if (key === lastDedupeKey && t - lastDedupeAt < 400) return;
                lastDedupeKey = key;
                lastDedupeAt = t;
                void processFile(raw);
            };
            input.addEventListener('change', onNativePick);
            input.addEventListener('input', onNativePick);
        };
        attachGenSlot(document.getElementById('gen-front-input'), 'front');
        attachGenSlot(document.getElementById('gen-back-input'), 'back');
        document.getElementById('gen-generate-btn').addEventListener('click', () => this.generateGarment());
        document.getElementById('gen-retry-btn').addEventListener('click', async () => {
            const retryBtn = document.getElementById('gen-retry-btn');
            const statusEl = document.getElementById('gen-status');
            retryBtn.style.display = 'none';
            if (this._lastGlbUrl) {
                statusEl.className = 'gen-status ok';
                statusEl.textContent = 'Garment ready — click "Use in try-on"';
                this._setGeneratedGarmentPending(this._lastGlbUrl, this._lastJsonUrl || null, null);
            } else if (this._lastPollTaskId) {
                statusEl.className = 'gen-status busy'; statusEl.textContent = 'Checking…';
                try {
                    const res = await fetch(`${this.getApiOrigin()}/api/status/${this._lastPollTaskId}`);
                    const data = await res.json();
                    if (data.status === 'completed' && data.glb_url) {
                        this._lastGlbUrl = data.glb_url;
                        this._lastJsonUrl = data.json_url || null;
                        this._setGeneratedGarmentPending(data.glb_url, this._lastJsonUrl, null);
                        statusEl.className = 'gen-status ok';
                        statusEl.textContent = 'Garment ready — click "Use in try-on"';
                    } else {
                        statusEl.className = 'gen-status fail'; statusEl.textContent = 'Not done yet: ' + data.status;
                        retryBtn.style.display = 'block';
                    }
                } catch(e) { statusEl.className='gen-status fail'; statusEl.textContent='Still unreachable'; retryBtn.style.display='block'; }
            }
        });
        this._wireGeneratedGarmentCardUi();
    }

    _wireGeneratedGarmentCardUi() {
        const useBtn = document.getElementById('btn-use-generated-garment');
        const dismissBtn = document.getElementById('btn-dismiss-generated-garment');
        useBtn?.addEventListener('click', async () => {
            if (!this._pendingGeneratedGlbUrl) return;
            const glb = this._pendingGeneratedGlbUrl;
            const json = this._pendingGeneratedJsonUrl;
            const statusEl = document.getElementById('generated-garment-status');
            if (statusEl) statusEl.textContent = 'Loading…';
            try {
                await this.loadGarmentFromUrl(glb, json || null);
                if (statusEl) statusEl.textContent = 'Applied ✓';
                this.showToast('Generated garment applied', 'success');
            } catch (e) {
                if (statusEl) statusEl.textContent = 'Failed';
                this.showToast('Could not load generated garment: ' + ((e && e.message) || String(e)), 'error');
            }
        });
        dismissBtn?.addEventListener('click', () => this._clearGeneratedGarmentPending());
    }

    _absUrlMaybe(u) {
        if (!u) return null;
        if (u.startsWith('http')) return u;
        if (u.startsWith('/')) return this.getApiOrigin() + u;
        return this.getApiOrigin() + '/' + u;
    }

    _setGeneratedGarmentPending(glbUrl, jsonUrl, totalStr) {
        this._pendingGeneratedGlbUrl = glbUrl;
        this._pendingGeneratedJsonUrl = jsonUrl || null;

        const card = document.getElementById('generated-garment-card');
        const status = document.getElementById('generated-garment-status');
        const glbLink = document.getElementById('generated-glb-link');
        const jsonLink = document.getElementById('generated-json-link');

        if (status) {
            if (totalStr === 'cached') status.textContent = 'Ready (cached)';
            else status.textContent = totalStr ? `Done in ${totalStr}` : 'Ready';
        }
        if (glbLink) glbLink.href = this._absUrlMaybe(glbUrl) || '#';
        if (jsonLink) {
            const absJson = this._absUrlMaybe(jsonUrl || '');
            jsonLink.href = absJson || '#';
            jsonLink.style.display = absJson ? '' : 'none';
        }
        if (card) card.classList.remove('hidden');

        const genStatus = document.getElementById('gen-status');
        if (genStatus) {
            genStatus.className = 'gen-status ok';
            genStatus.textContent = 'Garment ready — loading…';
        }

        // Auto-apply immediately so there is no extra "Use" step.
        void (async () => {
            const statusEl = document.getElementById('generated-garment-status');
            if (statusEl) statusEl.textContent = 'Loading…';
            try {
                await this.loadGarmentFromUrl(glbUrl, jsonUrl || null);
                if (statusEl) statusEl.textContent = 'Applied ✓';
                if (genStatus) genStatus.textContent = 'Applied ✓';
            } catch (e) {
                if (statusEl) statusEl.textContent = 'Failed';
                if (genStatus) genStatus.textContent = 'Auto-load failed (use Open GLB or retry)';
            }
        })();
    }

    _clearGeneratedGarmentPending() {
        this._pendingGeneratedGlbUrl = null;
        this._pendingGeneratedJsonUrl = null;
        const card = document.getElementById('generated-garment-card');
        if (card) card.classList.add('hidden');
    }

    updateGenerateBtn() {
        const h = this._genFrontFile != null;
        const b = this._genBackFile != null;
        document.getElementById('gen-generate-btn').disabled = !(h && b);
        if (h && b) this._scheduleAutoGenerate();
    }

    setupSlider(sliderId, valueId, cb, fmt = v => v.toFixed(2)) {
        const s = document.getElementById(sliderId);
        const v = document.getElementById(valueId);
        if (!s) return;
        const update = () => {
            const raw = (s as HTMLInputElement).value;
            const val = Number.parseFloat(String(raw));
            if (v) v.textContent = fmt(val);
            cb(val);
        };
        s.addEventListener('input', update);
        s.addEventListener('change', update);
        update();
    }
    
    setupToggle(id, cb) {
        const t = document.getElementById(id);
        t.addEventListener('click', () => cb(t.classList.toggle('active')));
        cb(t.classList.contains('active'));
    }
    
    resetSettings() {
        const defaults = {
            'offset-y-slider': -0.15,
            'offset-x-slider': 0,
            'rotation-y-slider': 0,
            'smoothing-slider': 0.4,
            'shoulder-drop-slider': 0.075,
            'auto-scale-trim-slider': 0.92,
            'shoulder-squeeze-slider': 0.94
        };
        
        for (const [id, val] of Object.entries(defaults)) {
            const el = document.getElementById(id);
            if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
        }
        
        this.fitStructuralMul = null;
        this.smoothedScale = null;
        this.smoothedPosition.set(0, 0, 0);
        this.smoothedRotation = 0;
        this.settings.poseDetectEvery = 2;
        this._poseFrameTick = 0;
        const poseIntervalSel = document.getElementById('pose-detect-interval');
        if (poseIntervalSel) poseIntervalSel.value = '2';
        
        this.showToast('Reset complete', 'success');
    }
    
    takeScreenshot() {
        const c = document.createElement('canvas');
        c.width = this.debugCanvas.width;
        c.height = this.debugCanvas.height;
        const ctx = c.getContext('2d');
        
        ctx.save();
        if (this.settings.mirrorView) {
            ctx.translate(c.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(this.video, 0, 0, c.width, c.height);
            ctx.restore();
            ctx.drawImage(this.threeCanvas, 0, 0, c.width, c.height);
        } else {
            ctx.drawImage(this.video, 0, 0, c.width, c.height);
            ctx.drawImage(this.threeCanvas, 0, 0, c.width, c.height);
            ctx.restore();
        }
        
        const a = document.createElement('a');
        a.download = `tryon-${Date.now()}.png`;
        a.href = c.toDataURL('image/png');
        a.click();
    }
    
    showLoading(t) {
        document.getElementById('loading-text').textContent = t;
        document.getElementById('loading-overlay').classList.add('active');
    }
    
    hideLoading() {
        document.getElementById('loading-overlay').classList.remove('active');
    }
    
    updateStatus(type, text) {
        document.getElementById('status-dot').className = 'status-dot ' + type;
        document.getElementById('status-text').textContent = text;
    }
    
    showToast(msg, type = 'success') {
        const container = document.getElementById('toast-container');
        const t = document.createElement('div');
        t.className = `toast ${type}`;
        t.textContent = msg;
        container.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 2500);
    }
}
