import * as THREE from 'three';

/** Normalised pose landmark from MediaPipe. */
export type PoseLandmark = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
};

export function lerpAngle(a: number, b: number, t: number): number {
  const d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  return a + d * t;
}

export function smoothPoseLandmarks(
  pose: PoseLandmark[],
  smoothedPrev: PoseLandmark[] | null,
  smoothingFactor: number
): PoseLandmark[] {
  const a = smoothingFactor;
  if (!smoothedPrev) return pose.map((lm) => ({ ...lm }));
  return pose.map((lm, i) => ({
    x: a * lm.x + (1 - a) * smoothedPrev[i].x,
    y: a * lm.y + (1 - a) * smoothedPrev[i].y,
    z: a * (lm.z ?? 0) + (1 - a) * (smoothedPrev[i].z ?? 0),
    visibility: lm.visibility
  }));
}

export function poseToScreen(lm: PoseLandmark): THREE.Vector2 {
  return new THREE.Vector2(lm.x, lm.y);
}

export function poseToWorld(lm: PoseLandmark): THREE.Vector3 {
  const x = lm.x;
  const y = lm.y;
  const z = -((lm.z || 0) * 2);
  return new THREE.Vector3((x - 0.5) * 2, -(y - 0.5) * 2, z);
}
