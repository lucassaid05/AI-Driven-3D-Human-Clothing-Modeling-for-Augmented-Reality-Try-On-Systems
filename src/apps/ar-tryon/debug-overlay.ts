import type { PoseLandmark } from './pose-utils';

export type HudDeps = {
  debugInfo: Record<string, unknown>;
  settings: { garmentCullInvert?: boolean };
  garmentShoulderWidth: number | null;
  smoothedScale: number | null;
};

/** Heads-up debug text over the try-on view. */
export function drawHudOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  conf: number,
  deps: HudDeps
): void {
  const { debugInfo, settings, garmentShoulderWidth, smoothedScale } = deps;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
  ctx.fillRect(width - 280, 10, 270, 300);

  ctx.fillStyle = '#00f5d4';
  ctx.font = '11px Space Mono';

  let y = 28;
  const line = (text: string) => {
    ctx.fillText(text, width - 270, y);
    y += 16;
  };

  line(`Confidence: ${(conf * 100).toFixed(0)}%`);
  line(`Garment: ${debugInfo.garmentHidden ? 'HIDDEN (pose)' : 'ON'}`);
  line(`View: ${String(debugInfo.viewAngle)}`);
  if (debugInfo.occlusionAssist) line('Occlusion assist: ON');
  let facing = debugInfo.facingAway ? 'BACK' : 'FRONT';
  if (debugInfo.viewAngle === 'SIDE') facing = 'PROFILE';
  line(`Facing: ${facing}`);
  line(`Shirt side src: ${String(debugInfo.faceFacingSource || 'legacy')}`);
  const pitch = (((debugInfo.pitch as number) || 0) * 180) / Math.PI;
  const roll = (((debugInfo.roll as number) || 0) * 180) / Math.PI;
  const yaw = (((debugInfo.yaw as number) || 0) * 180) / Math.PI;
  const zLean = ((debugInfo.zLean as number) || 0).toFixed(3);
  const pBlend = ((debugInfo.profileBlend as number) || 0).toFixed(2);
  line(`Yaw: ${yaw.toFixed(1)}° | Pitch: ${pitch.toFixed(1)}° | Roll: ${roll.toFixed(1)}°`);
  line(`Profile blend: ${pBlend} | Z-Lean: ${zLean} (neg=fwd)`);
  line('─────────────────────────────');
  line(`Shoulder Width: ${(debugInfo.shoulderWidth as number).toFixed(4)}`);
  line(`Torso Height: ${(debugInfo.torsoHeight as number).toFixed(4)}`);
  line(`Effective Width: ${(debugInfo.effectiveWidth as number).toFixed(4)}`);
  const drop = (debugInfo.shoulderDropWorld as number) || 0;
  line(`Auto shoulder drop: ${drop.toFixed(4)}`);
  line('─────────────────────────────');
  line(`Garment Shoulders: ${garmentShoulderWidth?.toFixed(4) || 'N/A'}`);
  const fb = debugInfo.fitStructuralMul as number | null;
  line(`Struct fit (auto): ${fb != null ? `${fb.toFixed(4)}×` : '—'}`);
  line(`Uniform scale: ${((debugInfo.clampedScale as number) || smoothedScale || 0).toFixed(4)}`);
  line('─────────────────────────────');
  line(`Shirt cull fix: ${settings.garmentCullInvert ? 'ON' : 'OFF'}`);
}

const SKELETON_CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24]
];

/** Landmarks already mapped into visible viewport space (0..1), same as garment poseToWorld. */
export function drawSkeletonOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  pose: PoseLandmark[]
): void {
  ctx.strokeStyle = '#00f5d4';
  ctx.lineWidth = 2;

  for (const [i, j] of SKELETON_CONNECTIONS) {
    const a = pose[i];
    const b = pose[j];
    if (!a || !b) continue;
    if ((a.visibility || 0) > 0.3 && (b.visibility || 0) > 0.3) {
      ctx.beginPath();
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
      ctx.stroke();
    }
  }

  ctx.fillStyle = '#f72585';
  for (const lm of pose) {
    if (!lm || (lm.visibility || 0) <= 0.3) continue;
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawMpMappingLayer(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  pose: PoseLandmark[],
  garmentMetadata: { mediapipe_mapping?: Record<string, { smpl_joint?: string }> } | null
): void {
  const map = garmentMetadata?.mediapipe_mapping;
  if (!map || !pose) return;

  const colorFor = (joint: string | undefined) => {
    const j = (joint || '').toLowerCase();
    if (j.includes('shoulder') || j.includes('collar')) return '#fee440';
    if (j.includes('elbow')) return '#00bbf9';
    if (j.includes('wrist') || j.includes('hand')) return '#f15bb5';
    if (j.includes('hip')) return '#9b5de5';
    if (j.includes('knee') || j.includes('ankle') || j.includes('foot')) return '#00f5a0';
    return '#ffffff';
  };

  ctx.font = 'bold 9px Space Mono, monospace';
  ctx.textBaseline = 'middle';

  for (const [idxStr, info] of Object.entries(map)) {
    const i = parseInt(idxStr, 10);
    if (i < 0 || i >= pose.length) continue;
    const lm = pose[i];
    if (!lm || (lm.visibility || 0) < 0.28) continue;
    const x = lm.x * width;
    const y = lm.y * height;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(info.smpl_joint);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.65)';
    ctx.lineWidth = 2;
    ctx.stroke();
    const tag = String(idxStr);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillText(tag, x - (tag.length > 1 ? 5 : 3), y);
  }
}
