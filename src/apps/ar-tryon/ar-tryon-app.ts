// @ts-nocheck
// Regenerated from legacy/ar-tryon-v13.html — run: npm run generate:artyson
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
    smoothPoseLandmarks,
    lerpAngle,
    poseToScreen,
    poseToWorld
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
        
        // Garment measurements from GLB
        this.garmentShoulderWidth = null;
        this.garmentTorsoHeight = null;
        /** Midpoint between shoulder bones in garmentGroup local space (bind pose); for alignment */
        this.garmentBindShoulderMid = null;
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
            faceFacingSource: 'legacy'
        };
        
        // Settings
        this.settings = {
            offsetY: 0.05,         // Match ar-tryon-v12.html
            offsetX: 0,
            rotationY: 0,
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
        
        this.setupThreeJS();
        this.setupMediaPipe();
        this.setupUI();
        this.applyMirrorViewTransforms();
        window.addEventListener('resize', () => this.onResize());
        void this._probeApiHealth();
        void this._maybeLoadGarmentFromQueryString();
    }

    /**
     * Deep link from Hunyuan studio (or bookmarks): ?glb=/generated/task_xxx/garment.glb&json=/generated/.../garment.json
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
    async startCamera() {
        try {
            this.showLoading('Starting camera...');
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } 
            });
            this.video.srcObject = stream;
            await new Promise(r => this.video.onloadedmetadata = r);
            await this.video.play();
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
            this.updateStatus('error', 'Camera failed');
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
    
    async loadGarment(file) {
        try {
            this.showLoading('Loading garment...');
            
            // Cleanup
            if (this.garmentGroup) {
                this.scene.remove(this.garmentGroup);
            }
            this.garmentModel = null;
            this.skinnedMesh = null;
            this.skeleton = null;
            this.bones = {};
            this._sessionGarmentLabel = null;
            this.fitStructuralMul = null;
            this.smoothedScale = null;
            this.garmentMetadata = null;
            this._hystFacingAway = false;
            this._lastMatFacingAway = null;
            this._lastMatState = '';
            this.smoothedYaw = 0;
            this.smoothedPitch = 0;
            this.smoothedRoll = 0;

            const url = URL.createObjectURL(file);
            const loader = new GLTFLoader();
            const gltf = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
            URL.revokeObjectURL(url);
            
            // Find skinned mesh
            gltf.scene.traverse(child => {
                if (child.isSkinnedMesh) {
                    this.skinnedMesh = child;
                    this.skeleton = child.skeleton;
                }
            });
            
            if (!this.skinnedMesh) throw new Error('No skinned mesh found');
            
            // Setup
            this.applyMaterialSettings();
            this.cacheSkeletonBones();
            
            // Create group
            this.garmentGroup = new THREE.Group();
            this.garmentModel = gltf.scene;
            this.garmentGroup.add(this.garmentModel);
            this.scene.add(this.garmentGroup);
            
            // Measure garment from bones
            this.measureGarmentFromBones();
            
            // Update UI
            const info = document.getElementById('garment-info');
            info.style.display = 'block';
            info.innerHTML = `
                Mesh: ${this.skinnedMesh.name}<br>
                Bones: ${Object.keys(this.bones).length}<br>
                Shoulder Width: ${this.garmentShoulderWidth?.toFixed(4) || 'N/A'}<br>
                Torso Height: ${this.garmentTorsoHeight?.toFixed(4) || 'N/A'}
            `;
            
            console.log(`✓ Garment loaded: shoulders=${this.garmentShoulderWidth?.toFixed(4)}, torso=${this.garmentTorsoHeight?.toFixed(4)}`);
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
        this._cacheGarmentBindShoulderMid();
        this._recomputeGarmentShoulderHemDelta();
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
    
    applyMaterialSettings() {
        if (!this.skinnedMesh) return;
        
        // Single-sided front/back is wrong for oblique & profile (whole torso can vanish). DoubleSide there.
        let side;
        if (this.settings.frontFaceOnly) {
            const va = String(this.debugInfo?.viewAngle || 'FRONT').toUpperCase();
            if (va === 'SIDE' || va === 'ANGLE') {
                side = THREE.DoubleSide;
            } else {
                let towardCamAsFront = !this.debugInfo?.facingAway;
                if (this.settings.garmentCullInvert) towardCamAsFront = !towardCamAsFront;
                // Ortho cam scale.x = -1 mirrors the render; single-sided shirts otherwise pick the wrong face (logo reads backward).
                if (this.settings.mirrorView) towardCamAsFront = !towardCamAsFront;
                // SMPL / Pix2Surf / many generated GLBs put the chest print on the face Three.js culls as “back”
                // for our pose + mirror setup — use BackSide when the user faces the camera so the front design shows.
                side = towardCamAsFront ? THREE.BackSide : THREE.FrontSide;
            }
        } else {
            side = THREE.DoubleSide;
        }
        
        const applyToMat = (mat) => {
            mat.side = side;
            mat.wireframe = this.settings.wireframe;
            mat.needsUpdate = true;
        };
        
        if (Array.isArray(this.skinnedMesh.material)) {
            this.skinnedMesh.material.forEach(applyToMat);
        } else if (this.skinnedMesh.material) {
            applyToMat(this.skinnedMesh.material);
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
        const shoulderVisStrong = (ls0?.visibility || 0) >= 0.42 && (rs0?.visibility || 0) >= 0.42;
        const shoulderVisShow = (ls0?.visibility || 0) >= 0.32 && (rs0?.visibility || 0) >= 0.32;
        const hipVisOk = (lh0?.visibility || 0) >= 0.28 && (rh0?.visibility || 0) >= 0.28;
        const trackingPoseStrong = conf >= 0.37 && shoulderVisStrong && (hipVisOk || conf >= 0.44);
        const trackingShowGarment = conf >= 0.26 && shoulderVisShow;
        const trackingDead = conf < 0.17 || ((ls0?.visibility || 0) < 0.22 && (rs0?.visibility || 0) < 0.22);

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

    /** Screen-space torso metrics. Null if shoulders unusable. */
    _measureBodyFromPose(pose) {
        const ls = pose[this.MP.LEFT_SHOULDER];
        const rs = pose[this.MP.RIGHT_SHOULDER];
        const lh = pose[this.MP.LEFT_HIP];
        const rh = pose[this.MP.RIGHT_HIP];
        if (!ls || !rs || (ls.visibility || 0) < 0.3 || (rs.visibility || 0) < 0.3) return null;

        const LS = poseToScreen(ls);
        const RS = poseToScreen(rs);
        const LH = poseToScreen(lh);
        const RH = poseToScreen(rh);

        const shoulderWidthRaw = Math.abs(LS.x - RS.x);
        const shoulderCenterY = (LS.y + RS.y) / 2;
        const hipCenterY = (LH.y + RH.y) / 2;
        let torsoHeightScreen = Math.abs(hipCenterY - shoulderCenterY);
        torsoHeightScreen = THREE.MathUtils.clamp(torsoHeightScreen, 0.11, 0.46);
        const shoulderSpanCap = torsoHeightScreen * 1.05;
        const shoulderWidthScreen = Math.min(shoulderWidthRaw, shoulderSpanCap);

        const expectedShoulderWidth = torsoHeightScreen * 0.65;
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

        const minEffW = torsoHeightScreen * 0.55;
        if (effectiveWidth < minEffW) effectiveWidth = minEffW;

        const bodyWidth = effectiveWidth * 2;
        const bodyHeight = torsoHeightScreen * 2;
        const bodyWidthFromShoulders = shoulderWidthScreen * 2;

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
            viewAngle
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
        const shouldersReliable = shoulderSepNorm >= 0.14;
        if (isSideView) {
            if (noseOk) {
                if (nose.z > shoulderMidZ + 0.18) this._hystFacingAway = true;
                else if (nose.z < shoulderMidZ - 0.03) this._hystFacingAway = false;
            }
        } else if (!isAmbiguousSep) {
            if (rawDiff > 0.045) {
                if (!noseClearlyForward && (noseOk || shouldersReliable)) this._hystFacingAway = true;
            } else if (rawDiff < -0.045) {
                this._hystFacingAway = false;
            }
        } else {
            if (noseOk) {
                if (nose.z > shoulderMidZ + 0.14) this._hystFacingAway = true;
                else if (nose.z < shoulderMidZ - 0.05) this._hystFacingAway = false;
            }
        }
    }

    /** One-shot structural multiplier (no user sliders). Used by Calibrate fit. */
    computeFitStructuralMul(pose) {
        const gsw = this.garmentShoulderWidth;
        if (!gsw || gsw < 0.001) return null;
        const m = this._measureBodyFromPose(pose);
        if (!m) return null;
        const hipOk = (m.lh.visibility || 0) >= 0.22 && (m.rh.visibility || 0) >= 0.22;
        let raw = 1;
        if (!hipOk) {
            const shoulderLine = Math.max(m.shoulderWidthScreen * 2, m.shoulderWidthRaw * 2 * 0.9);
            const torsoLine = m.torsoHeightScreen * 0.52 * 2;
            raw = Math.max(shoulderLine, torsoLine) / gsw;
        } else if (m.viewAngleRatio < 0.35) {
            raw = m.bodyWidth / gsw;
        } else if (m.viewAngleRatio < 0.65) {
            const shoulderLine = Math.max(m.shoulderWidthScreen * 2, m.shoulderWidthRaw * 2 * 0.92);
            const torsoLine = m.torsoHeightScreen * 0.50 * 2;
            raw = Math.max(shoulderLine, torsoLine) / gsw;
        } else {
            raw = m.bodyWidthFromShoulders / gsw;
        }
        return THREE.MathUtils.clamp(raw * 0.998, 0.28, 2.15);
    }

    getUniformGarmentScale() {
        const struct = this.fitStructuralMul != null ? this.fitStructuralMul : 1;
        return THREE.MathUtils.clamp(struct, 0.25, 3.0);
    }

    /**
     * Smoothly tracks computeFitStructuralMul() every frame. Side view freezes the multiplier
     * after the first lock to avoid noisy width.
     */
    _updateContinuousStructuralFit(M, pose) {
        if (!this.garmentGroup || !pose) return;
        const gsw = this.garmentShoulderWidth;
        if (!gsw || gsw < 0.001) return;

        const target = this.computeFitStructuralMul(pose);
        if (target == null) return;

        const sideHold = M.viewAngle === 'SIDE' && this.fitStructuralMul != null;
        if (sideHold) return;

        const prev = this.fitStructuralMul;
        if (prev == null) {
            this.fitStructuralMul = target;
            return;
        }

        const rel = Math.abs(target - prev) / Math.max(1e-6, prev);
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

        const pLS = poseToWorld(ls);
        const pRS = poseToWorld(rs);
        const poseShoulderMid = pLS.clone().add(pRS).multiplyScalar(0.5);
        poseShoulderMid.z = 0;

        const hipVisPose = (lh.visibility || 0) > 0.24 && (rh.visibility || 0) > 0.24;
        let poseHipMid;
        if (hipVisPose) {
            poseHipMid = poseToWorld(lh).clone().add(poseToWorld(rh)).multiplyScalar(0.5);
        } else {
            poseHipMid = poseShoulderMid.clone();
            const hipDrop = Math.max(0.16, torsoHeightScreen * 2.08);
            poseHipMid.y = poseShoulderMid.y - hipDrop;
        }
        poseHipMid.z = 0;

        const userShoulderWorld = Math.max(0.02, pLS.distanceTo(pRS));
        const uBase = this.getUniformGarmentScale();
        const gsw = Math.max(0.001, this.garmentShoulderWidth || 0.3);
        /** ~0.2% slack vs MP shoulder span (≈ sub‑mm feel in ortho space at typical FOV). */
        const shoulderFit = 0.998;
        const uCapShoulder = (userShoulderWorld * shoulderFit) / gsw;
        const dPoseY = Math.abs(poseShoulderMid.y - poseHipMid.y);
        const dGarmentY = Math.max(0.02, this._garmentShoulderHemLocalDy || 0.25);
        const uCapTorso = THREE.MathUtils.clamp(dPoseY / dGarmentY, 0.38, 2.85);
        let uGeom;
        if (this.debugInfo.viewAngle === 'SIDE') {
            uGeom = Math.min(uCapShoulder, uCapTorso);
        } else {
            uGeom = Math.max(uCapShoulder, uCapTorso);
        }
        const uRaw = Math.min(uBase, uGeom);
        const scaleBlend = 0.46;
        this.smoothedScale = this.smoothedScale == null
            ? uRaw
            : THREE.MathUtils.lerp(this.smoothedScale, uRaw, scaleBlend);
        const u = THREE.MathUtils.clamp(this.smoothedScale, 0.25, 3.0);
        this.smoothedScale = u;

        this.debugInfo.computedScale = this.fitStructuralMul == null ? 1 : this.fitStructuralMul;
        this.debugInfo.clampedScale = u;

        this.garmentGroup.scale.set(u, u, u);

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
        if (isSideView) {
            estimatedYaw = THREE.MathUtils.clamp(estimatedYaw, -1.12, 1.12);
        }
        // 180° for facing away; hysteresis above prevents boundary flicker (no extra shoulder-width gate)
        if (isFacingAway && !isSideView) {
            estimatedYaw += Math.PI;
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

        if (this.debugInfo.viewAngle === 'SIDE') {
            // Profile: 2D midpoint of LS/RS is often biased (occluded shoulder drifts in x). Anchor x to the
            // shoulder closer to the camera (smaller MediaPipe z) so left vs right profile stay consistent
            // with the mirrored selfie camera, instead of sliding the model the wrong way.
            const zl = ls.z || 0;
            const zr = rs.z || 0;
            const rightCloser = zr <= zl;
            const pFwd = rightCloser ? pRS : pLS;
            const pBack = rightCloser ? pLS : pRS;
            const vFwd = Math.max(0.14, rightCloser ? (rs.visibility || 0) : (ls.visibility || 0));
            const vBack = Math.max(0.14, rightCloser ? (ls.visibility || 0) : (rs.visibility || 0));
            const w = vFwd / (vFwd + vBack);
            poseAnchor.x = pFwd.x * (0.62 + 0.28 * w) + pBack.x * (0.38 - 0.28 * w);
            if (hipVisPose) {
                const pLH = poseToWorld(lh);
                const pRH = poseToWorld(rh);
                const vhl = Math.max(0.08, lh.visibility || 0);
                const vhr = Math.max(0.08, rh.visibility || 0);
                const hipMidX = (pLH.x * vhl + pRH.x * vhr) / (vhl + vhr);
                poseAnchor.x = poseAnchor.x * 0.82 + hipMidX * 0.18;
            }
        }

        let targetPos;
        if (this.garmentBindShoulderMid) {
            const bindOff = (local) => {
                const v = local.clone();
                v.applyQuaternion(this.garmentGroup.quaternion);
                v.multiply(this.garmentGroup.scale);
                return v;
            };
            const shoulderOff = bindOff(this.garmentBindShoulderMid);
            targetPos = poseAnchor.clone().sub(shoulderOff);
            targetPos.x += this.settings.offsetX;
            targetPos.y += this.settings.offsetY;
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
            this.skeleton = null; this.bones = {};
            this._sessionGarmentLabel = null;
            this.fitStructuralMul = null;
            this.smoothedScale = null;
            this.garmentMetadata = null;
            this._hystFacingAway = false;
            this._lastMatFacingAway = null;
            this._lastMatState = '';
            this.smoothedYaw = 0;
            this.smoothedPitch = 0;
            this.smoothedRoll = 0;
            const loader = new GLTFLoader();
            const gltf = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
            gltf.scene.traverse(child => {
                if (child.isSkinnedMesh) { this.skinnedMesh = child; this.skeleton = child.skeleton; }
            });
            if (!this.skinnedMesh) throw new Error('No skinned mesh found');
            this.applyMaterialSettings();
            this.cacheSkeletonBones();
            this.garmentGroup = new THREE.Group();
            this.garmentModel = gltf.scene;
            this.garmentGroup.add(this.garmentModel);
            this.scene.add(this.garmentGroup);
            this.measureGarmentFromBones();
            const info = document.getElementById('garment-info');
            info.style.display = 'block';
            info.innerHTML = `Mesh: ${this.skinnedMesh.name}<br>Bones: ${Object.keys(this.bones).length}<br>Shoulder Width: ${this.garmentShoulderWidth?.toFixed(4) || 'N/A'}<br>Torso Height: ${this.garmentTorsoHeight?.toFixed(4) || 'N/A'}`;
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
        } catch (e) {
            console.warn('garment.json fetch', e);
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

    async generateGarment() {
        const frontInput = document.getElementById('gen-front-input');
        const backInput  = document.getElementById('gen-back-input');
        const statusEl   = document.getElementById('gen-status');
        const genBtn     = document.getElementById('gen-generate-btn');
        if (!frontInput.files.length || !backInput.files.length) {
            this.showToast('Select both front and back images first', 'error'); return;
        }
        const measureImg = (file) => new Promise((resolve) => {
            const img = new Image();
            const u = URL.createObjectURL(file);
            img.onload = () => { URL.revokeObjectURL(u); resolve({ w: img.naturalWidth, h: img.naturalHeight }); };
            img.onerror = () => { URL.revokeObjectURL(u); resolve(null); };
            img.src = u;
        });
        const [fdim, bdim] = await Promise.all([
            measureImg(frontInput.files[0]),
            measureImg(backInput.files[0])
        ]);
        if (fdim && (fdim.w < 320 || fdim.h < 320)) {
            this.showToast('Front photo is small — use ≥512px, good light, plain background', 'warning');
        }
        if (bdim && (bdim.w < 320 || bdim.h < 320)) {
            this.showToast('Back photo is small — use ≥512px', 'warning');
        }
        const toBase64 = f => new Promise((res, rej) => {
            const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(f);
        });
        genBtn.disabled = true;
        this._genWallStart = Date.now();
        statusEl.className = 'gen-status busy';
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

            const [frontB64, backB64] = await Promise.all([toBase64(frontInput.files[0]), toBase64(backInput.files[0])]);
            const res = await fetch(this.getApiOrigin() + '/api/generate', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ front_image: frontB64, back_image: backB64 })
            });
            if (!res.ok) throw new Error(`Server error ${res.status}`);
            const { task_id } = await res.json();
            const up = this.formatGenElapsed((Date.now() - this._genWallStart) / 1000);
            statusEl.textContent = up
                ? `Queued… (upload ${up}, task ${task_id.slice(0, 8)}…)`
                : `Queued… (task ${task_id.slice(0, 8)}…)`;
            this.pollGenerationStatus(task_id);
        } catch (e) {
            this._genWallStart = null;
            statusEl.className = 'gen-status fail';
            statusEl.textContent = 'Error: ' + e.message;
            genBtn.disabled = false;
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
                    await this.loadGarmentFromUrl(data.glb_url, this._lastJsonUrl);
                    statusEl.textContent = totalStr
                        ? `Garment ready ✓ (${totalStr})`
                        : 'Garment ready ✓';
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
            }
        };
        void tick();
        intervalId = setInterval(tick, 1000);
    }

    // ─────────────────────────────────────────────────────────────────────

    setupUI() {
        // File upload
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');
        
        uploadZone.addEventListener('click', () => fileInput.click());
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

        // Camera
        document.getElementById('start-btn').addEventListener('click', () => {
            if (this.isRunning) this.stopCamera();
            else this.startCamera();
        });
        
        this.setupSlider('offset-y-slider', 'offset-y-value', v => this.settings.offsetY = v);
        this.setupSlider('offset-x-slider', 'offset-x-value', v => this.settings.offsetX = v);
        this.setupSlider('rotation-y-slider', 'rotation-y-value', v => this.settings.rotationY = v, v => v + '°');
        this.setupSlider('smoothing-slider', 'smoothing-value', v => this.settings.smoothingFactor = v);
        
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
    }
    
    setupGenerateUI() {
        const frontBtn   = document.getElementById('gen-front-btn');
        const frontInput = document.getElementById('gen-front-input');
        frontBtn.addEventListener('click', () => frontInput.click());
        frontInput.addEventListener('change', () => {
            if (!frontInput.files.length) return;
            const p = document.getElementById('gen-front-preview');
            const l = document.getElementById('gen-front-label');
            p.src = URL.createObjectURL(frontInput.files[0]);
            p.style.display = 'block'; l.style.display = 'none';
            this.updateGenerateBtn();
        });
        const backBtn   = document.getElementById('gen-back-btn');
        const backInput = document.getElementById('gen-back-input');
        backBtn.addEventListener('click', () => backInput.click());
        backInput.addEventListener('change', () => {
            if (!backInput.files.length) return;
            const p = document.getElementById('gen-back-preview');
            const l = document.getElementById('gen-back-label');
            p.src = URL.createObjectURL(backInput.files[0]);
            p.style.display = 'block'; l.style.display = 'none';
            this.updateGenerateBtn();
        });
        document.getElementById('gen-generate-btn').addEventListener('click', () => this.generateGarment());
        document.getElementById('gen-retry-btn').addEventListener('click', async () => {
            const retryBtn = document.getElementById('gen-retry-btn');
            const statusEl = document.getElementById('gen-status');
            retryBtn.style.display = 'none';
            if (this._lastGlbUrl) {
                statusEl.className = 'gen-status busy'; statusEl.textContent = 'Loading…';
                await this.loadGarmentFromUrl(this._lastGlbUrl, this._lastJsonUrl || null);
                statusEl.className = 'gen-status ok'; statusEl.textContent = 'Garment ready ✓';
            } else if (this._lastPollTaskId) {
                statusEl.className = 'gen-status busy'; statusEl.textContent = 'Checking…';
                try {
                    const res = await fetch(`${this.getApiOrigin()}/api/status/${this._lastPollTaskId}`);
                    const data = await res.json();
                    if (data.status === 'completed' && data.glb_url) {
                        this._lastGlbUrl = data.glb_url;
                        this._lastJsonUrl = data.json_url || null;
                        await this.loadGarmentFromUrl(data.glb_url, this._lastJsonUrl);
                        statusEl.className = 'gen-status ok'; statusEl.textContent = 'Garment ready ✓';
                    } else {
                        statusEl.className = 'gen-status fail'; statusEl.textContent = 'Not done yet: ' + data.status;
                        retryBtn.style.display = 'block';
                    }
                } catch(e) { statusEl.className='gen-status fail'; statusEl.textContent='Still unreachable'; retryBtn.style.display='block'; }
            }
        });
    }

    updateGenerateBtn() {
        const h = document.getElementById('gen-front-input').files.length > 0;
        const b = document.getElementById('gen-back-input').files.length  > 0;
        document.getElementById('gen-generate-btn').disabled = !(h && b);
    }

    setupSlider(sliderId, valueId, cb, fmt = v => v.toFixed(2)) {
        const s = document.getElementById(sliderId);
        const v = document.getElementById(valueId);
        const update = () => { const val = parseFloat(s.value); v.textContent = fmt(val); cb(val); };
        s.addEventListener('input', update);
        update();
    }
    
    setupToggle(id, cb) {
        const t = document.getElementById(id);
        t.addEventListener('click', () => cb(t.classList.toggle('active')));
        cb(t.classList.contains('active'));
    }
    
    resetSettings() {
        const defaults = {
            'offset-y-slider': 0.05,
            'offset-x-slider': 0,
            'rotation-y-slider': 0,
            'smoothing-slider': 0.4
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
    
    hideLoading() { document.getElementById('loading-overlay').classList.remove('active'); }
    
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
