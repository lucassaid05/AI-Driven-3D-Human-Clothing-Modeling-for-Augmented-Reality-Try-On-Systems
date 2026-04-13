/**
 * Shared fetch helpers for static pages served by server-local.py (same origin).
 * @typedef {{ status: string, time?: number }} HealthResponse
 * @typedef {{ ready?: boolean, pix2surf_available?: boolean, [k: string]: unknown }} ConfigResponse
 * @typedef {{ garments: Array<{ id: string, glb_url: string, json_url?: string|null, mtime: number }> }} GarmentsResponse
 * @typedef {{ label: string, path: string }} PipelineAppLink
 * @typedef {{ path: string, role?: string }} PipelineCliScript
 * @typedef {{
 *   order: number,
 *   id: string,
 *   title: string,
 *   implementation: 'planned'|'partial'|'live',
 *   summary?: string,
 *   tasks?: string[],
 *   reference?: string,
 *   app_links?: PipelineAppLink[],
 *   cli_scripts?: PipelineCliScript[],
 * }} PipelineStage
 * @typedef {{
 *   version: number,
 *   live_route?: Record<string, string>,
 *   inputs?: Record<string, string>,
 *   stages: PipelineStage[],
 *   deployment?: unknown,
 *   related_sota?: Record<string, string>,
 *   docs?: Record<string, string>,
 * }} PipelineMetaResponse
 */

const jsonHeaders = { Accept: 'application/json' };

async function parseJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Expected JSON, got ${res.status}: ${text.slice(0, 120)}`);
  }
}

/** @returns {Promise<HealthResponse>} */
export async function fetchHealth() {
  const res = await fetch('/api/health', { headers: jsonHeaders });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return parseJson(res);
}

/** @returns {Promise<ConfigResponse>} */
export async function fetchConfig() {
  const res = await fetch('/api/config', { headers: jsonHeaders });
  if (!res.ok) throw new Error(`config ${res.status}`);
  return parseJson(res);
}

/** @returns {Promise<GarmentsResponse>} */
export async function fetchGarments() {
  const res = await fetch('/api/garments', { headers: jsonHeaders });
  if (!res.ok) throw new Error(`garments ${res.status}`);
  return parseJson(res);
}

/** @returns {Promise<PipelineMetaResponse>} */
export async function fetchPipeline() {
  const res = await fetch('/api/pipeline', { headers: jsonHeaders });
  if (!res.ok) throw new Error(`pipeline ${res.status}`);
  return parseJson(res);
}

/** @returns {Promise<Record<string, unknown>>} LAN URLs and camera hints for phones */
export async function fetchServerInfo() {
  const res = await fetch('/api/server-info', { headers: jsonHeaders });
  if (!res.ok) throw new Error(`server-info ${res.status}`);
  return parseJson(res);
}
