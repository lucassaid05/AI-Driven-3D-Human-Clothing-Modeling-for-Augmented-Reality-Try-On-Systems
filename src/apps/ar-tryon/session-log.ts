import type { PoseLandmark } from './pose-utils';

/** Mutable session log document (written to JSON for evaluation). */
export type SessionLogRecord = {
  version: string;
  app: string;
  purpose: string;
  participantCode: string | null;
  notes: string;
  logOptions: {
    sample_interval_ms: number;
    includeLandmarks: boolean;
    landmark_sample_min_ms: number;
  };
  userAgent: string;
  language: string;
  screen: { w: number; h: number; availW: number; availH: number };
  device: Record<string, unknown>;
  origin: string;
  startedAt: string;
  endedAt?: string;
  activeGarmentAtStart: string | null;
  entries: Record<string, unknown>[];
};

/** Minimal surface of ARTryOn used by session logging. */
export type SessionLogHost = {
  settings: {
    logSession: boolean;
    logIncludeLandmarks: boolean;
    logLandmarkIntervalMs: number;
  };
  sessionLog: SessionLogRecord | null;
  _lastSessionLogTs: number;
  _lastLandmarkLogWall: number;
  _logPrevShoulderMid: { x: number; y: number } | null;
  _logJitterEma: number | null;
  _sessionGarmentLabel: string | null;
  MP: { LEFT_SHOULDER: number; RIGHT_SHOULDER: number };
  fps: number;
  _lastLatencyMs: number;
  _lastRenderMs: number;
  smoothedScale: number | null;
  fitStructuralMul: number | null;
  debugInfo: Record<string, unknown>;
  getApiOrigin(): string;
  showToast(msg: string, type?: string): void;
};

export function syncLogLandmarkIntervalFromUI(settings: { logLandmarkIntervalMs?: number }): void {
  const sel = document.getElementById('log-landmark-interval') as HTMLSelectElement | null;
  if (!sel) return;
  const v = parseInt(sel.value, 10);
  if (Number.isFinite(v) && v >= 500) settings.logLandmarkIntervalMs = v;
}

export function initSessionLog(host: SessionLogHost): void {
  syncLogLandmarkIntervalFromUI(host.settings);
  const codeEl = document.getElementById('session-participant-code') as HTMLInputElement | null;
  const participantCode = (codeEl?.value || '').trim() || null;
  host._lastSessionLogTs = 0;
  host._lastLandmarkLogWall = 0;
  host._logPrevShoulderMid = null;
  host._logJitterEma = null;
  host.sessionLog = {
    version: '1.1',
    app: 'ar-tryon-vite',
    purpose: 'dissertation_evaluation',
    participantCode,
    notes:
      'Pose landmarks are normalised MediaPipe-style coordinates (not images). Proxies: shoulder drift between log samples; EMA ~ tracking jitter; scale delta = |structural_mul - uniform_scale|.',
    logOptions: {
      sample_interval_ms: 500,
      includeLandmarks: !!host.settings.logIncludeLandmarks,
      landmark_sample_min_ms: host.settings.logLandmarkIntervalMs
    },
    userAgent: navigator.userAgent,
    language: navigator.language,
    screen: {
      w: screen.width,
      h: screen.height,
      availW: screen.availWidth,
      availH: screen.availHeight
    },
    device: {
      platform: navigator.platform || '',
      hardwareConcurrency: navigator.hardwareConcurrency ?? null,
      deviceMemoryGB: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null
    },
    origin: host.getApiOrigin(),
    startedAt: new Date().toISOString(),
    activeGarmentAtStart: host._sessionGarmentLabel,
    entries: []
  };
  const uaData = (navigator as Navigator & { userAgentData?: { mobile: boolean; brands: unknown; platform: string; getHighEntropyValues?: (keys: string[]) => Promise<Record<string, unknown>> } }).userAgentData;
  if (uaData) {
    host.sessionLog.device.mobile = uaData.mobile;
    host.sessionLog.device.brands = uaData.brands;
    host.sessionLog.device.uaPlatform = uaData.platform;
    const p = uaData.getHighEntropyValues?.(['model', 'platformVersion', 'fullVersionList']);
    if (p && typeof p.then === 'function') {
      p.then((he) => {
        if (!host.sessionLog) return;
        host.sessionLog.device.model = he.model ?? null;
        host.sessionLog.device.platformVersion = he.platformVersion ?? null;
        host.sessionLog.device.fullVersionList = he.fullVersionList ?? null;
      }).catch(() => {});
    }
  }
  host.showToast('Session logging ON — download JSON when finished', 'info');
}

export function downloadSessionLogJson(host: SessionLogHost): void {
  if (!host.sessionLog || !host.sessionLog.entries.length) {
    host.showToast('Enable “Log session”, run the camera with pose visible, then download', 'warning');
    return;
  }
  host.sessionLog.endedAt = new Date().toISOString();
  const blob = new Blob([JSON.stringify(host.sessionLog, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  const safeCode = (host.sessionLog.participantCode || 'anon').replace(/[^a-zA-Z0-9_-]+/g, '') || 'anon';
  a.download = `tryon-session-${safeCode}-${Date.now()}.json`;
  a.href = URL.createObjectURL(blob);
  a.click();
  URL.revokeObjectURL(a.href);
  host.showToast('Session log downloaded', 'success');
}

export type SessionLogFrameArgs = {
  conf: number;
  smoothedPose: PoseLandmark[] | null;
  trackingPoseStrong: boolean;
  trackingShowGarment: boolean;
};

/** Append a throttled session sample when logging is enabled (call from pose results). */
export function maybeAppendSessionLogEntry(host: SessionLogHost, args: SessionLogFrameArgs): void {
  const { conf, smoothedPose: sp, trackingPoseStrong, trackingShowGarment } = args;
  if (!host.settings.logSession || !host.sessionLog || conf < 0.25) return;
  const t = performance.now();
  if (t - host._lastSessionLogTs <= 500) return;
  host._lastSessionLogTs = t;

  const entry: Record<string, unknown> = {
    t_ms: Math.round(t),
    wall: new Date().toISOString(),
    fps: host.fps,
    latency_ms: Math.round(host._lastLatencyMs || 0),
    render_ms: Math.round(host._lastRenderMs * 1000) / 1000,
    conf: Math.round(conf * 1000) / 1000,
    uniform_scale: host.smoothedScale ?? 0,
    fit_structural: host.fitStructuralMul,
    view: host.debugInfo.viewAngle,
    garment_hidden: !!host.debugInfo.garmentHidden,
    tracking_strong: trackingPoseStrong,
    garment_visible: trackingShowGarment,
    active_garment: host._sessionGarmentLabel
  };
  if (host.fitStructuralMul != null && host.smoothedScale != null) {
    entry.struct_vs_uniform_scale_delta =
      Math.round(Math.abs(host.fitStructuralMul - host.smoothedScale) * 10000) / 10000;
  }
  if (sp && sp[host.MP.LEFT_SHOULDER] && sp[host.MP.RIGHT_SHOULDER]) {
    const l = sp[host.MP.LEFT_SHOULDER];
    const r = sp[host.MP.RIGHT_SHOULDER];
    if ((l.visibility || 0) > 0.2 && (r.visibility || 0) > 0.2) {
      const mx = (l.x + r.x) * 0.5;
      const my = (l.y + r.y) * 0.5;
      if (host._logPrevShoulderMid) {
        const drift = Math.hypot(mx - host._logPrevShoulderMid.x, my - host._logPrevShoulderMid.y);
        entry.shoulder_mid_drift_norm = Math.round(drift * 10000) / 10000;
        const alpha = 0.28;
        host._logJitterEma =
          host._logJitterEma == null ? drift : alpha * drift + (1 - alpha) * host._logJitterEma;
        entry.shoulder_jitter_ema_norm = Math.round((host._logJitterEma as number) * 10000) / 10000;
      }
      host._logPrevShoulderMid = { x: mx, y: my };
    }
  }
  if (host.settings.logIncludeLandmarks && sp && sp.length >= 33) {
    const minMs = Math.max(500, host.settings.logLandmarkIntervalMs || 2000);
    const wall = Date.now();
    if (wall - host._lastLandmarkLogWall >= minMs) {
      host._lastLandmarkLogWall = wall;
      entry.landmarks_33_xyzv = sp.map((lm) => [
        Math.round(lm.x * 10000) / 10000,
        Math.round(lm.y * 10000) / 10000,
        Math.round((lm.z || 0) * 10000) / 10000,
        Math.round((lm.visibility || 0) * 1000) / 1000
      ]);
    }
  }
  host.sessionLog.entries.push(entry);
  if (host.sessionLog.entries.length > 8000) {
    host.sessionLog.entries.splice(0, 2000);
  }
}
