import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const htmlPath = path.join(root, 'legacy', 'ar-tryon-v13.html');
const outPath = path.join(root, 'src', 'apps', 'ar-tryon', 'ar-tryon-app.ts');

const html = fs.readFileSync(htmlPath, 'utf8');
const start = html.indexOf('class ARTryOn {');
const end = html.indexOf('\n    // Defer init until window');
if (start < 0 || end < 0) throw new Error('ARTryOn class markers not found in ar-tryon-v13.html');

let body = html.slice(start, end);
body = body
  .split('\n')
  .map((line) => (line.startsWith('    ') ? line.slice(4) : line))
  .join('\n');

body = body.replace(/^class ARTryOn/m, 'export class ARTryOn');
body = body.replace(/\bnew THREE\.GLTFLoader\(\)/g, 'new GLTFLoader()');

const setupKey = '\n    async setupMediaPipe() {';
const setupStart = body.indexOf(setupKey);
const setupEnd = body.indexOf('\n    async startCamera()');
if (setupStart < 0 || setupEnd < 0) throw new Error('setupMediaPipe / startCamera bounds not found');

const newSetup = `
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
    }`;

body = body.slice(0, setupStart) + newSetup + body.slice(setupEnd);

body = body.replace(
  /\n    lerpAngle\(a, b, t\) \{[\s\S]*?\n    \}\n/,
  '\n'
);
body = body.replace(
  /\n    poseToScreen\(lm\) \{[\s\S]*?\n    \}\n/,
  '\n'
);
body = body.replace(
  /\n    poseToWorld\(lm\) \{[\s\S]*?\n    \}\n/,
  '\n'
);
body = body.replace(
  /\n    smoothPose\(pose\) \{[\s\S]*?\n    \}/,
  `
    smoothPose(pose) {
        return smoothPoseLandmarks(pose, this.smoothedPose, this.settings.smoothingFactor);
    }`
);
body = body.replace(
  /\n    drawSkeleton\(pose\) \{[\s\S]*?\n    \}/,
  `
    drawSkeleton(pose) {
        const w = this.debugCanvas.width;
        const h = this.debugCanvas.height;
        drawSkeletonOverlay(this.debugCtx, w, h, pose, this.settings.mirrorView);
    }`
);
body = body.replace(
  /\n    drawMpMappingOverlay\(pose\) \{[\s\S]*?\n    \}/,
  `
    drawMpMappingOverlay(pose) {
        const w = this.debugCanvas.width;
        const h = this.debugCanvas.height;
        drawMpMappingLayer(this.debugCtx, w, h, pose, this.garmentMetadata, this.settings.mirrorView);
    }`
);
body = body.replace(
  /\n    drawHud\(conf\) \{[\s\S]*?\n    \}/,
  `
    drawHud(conf) {
        const w = this.debugCanvas.width;
        drawHudOverlay(this.debugCtx, w, conf, {
            debugInfo: this.debugInfo,
            settings: this.settings,
            garmentShoulderWidth: this.garmentShoulderWidth,
            smoothedScale: this.smoothedScale
        });
    }`
);
body = body.replace(
  /\n    _syncLogLandmarkIntervalFromUI\(\) \{[\s\S]*?\n    \}\n/,
  '\n'
);
body = body.replace(
  /\n    _initSessionLog\(\) \{[\s\S]*?\n    \}\n/,
  '\n'
);
body = body.replace(
  /\n    downloadSessionLogJson\(\) \{[\s\S]*?\n    \}\n/,
  '\n'
);

const sessionNeedle = 'if (this.settings.logSession && this.sessionLog && conf >= 0.25) {';
const si = body.indexOf(sessionNeedle);
if (si < 0) throw new Error('Session log outer if not found');
const lineStart = body.lastIndexOf('\n', si - 1) + 1;
const openBrace = body.indexOf('{', si);
let depth = 0;
let p = openBrace;
let endBrace = -1;
for (; p < body.length; p++) {
  const c = body[p];
  if (c === '{') depth++;
  else if (c === '}') {
    depth--;
    if (depth === 0) {
      endBrace = p;
      break;
    }
  }
}
if (endBrace < 0) throw new Error('Could not match braces for session log block');

const indent = body.slice(lineStart, si);
const sessionReplace = `${indent}maybeAppendSessionLogEntry(this, {
${indent}    conf,
${indent}    smoothedPose: this.smoothedPose,
${indent}    trackingPoseStrong,
${indent}    trackingShowGarment
${indent}});`;
body = body.slice(0, lineStart) + sessionReplace + body.slice(endBrace + 1);

body = body.replace(/\bthis\.poseToScreen\(/g, 'poseToScreen(');
body = body.replace(/\bthis\.poseToWorld\(/g, 'poseToWorld(');
body = body.replace(/\bthis\.lerpAngle\(/g, 'lerpAngle(');

body = body.replace(/\bthis\._initSessionLog\(\)/g, 'initSessionLog(this)');
body = body.replace(/\bthis\._syncLogLandmarkIntervalFromUI\(\)/g, 'syncLogLandmarkIntervalFromUI(this.settings)');
body = body.replace(
  /\(\) => this\.downloadSessionLogJson\(\)/g,
  '() => downloadSessionLogJson(this)'
);

const header = `// @ts-nocheck
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

`;

fs.writeFileSync(outPath, header + body, 'utf8');
console.log('Wrote', outPath);
