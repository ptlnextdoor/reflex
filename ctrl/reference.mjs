// Reference trajectory: run a control line once through the sim and keep per-ms pose,
// speed, keys, normalized steering state and contact flags for the tracking controller.
// Lines come from Reflex-1's own planner (learn/best.json) or from ctrl/line.mjs (a line this
// module generates from the sim alone). No human runs: see LOOP.md "Isolation rule".
import fs from 'node:fs';
import {runAll, decodeState, STATE_BYTES} from './sim.mjs';

export const WR_FRAMES = 22123; // the human time to beat, used only as a number after the fact
export const STEER_SLEW = 0.01; // normalized steering units per ms toward the key target (measured)

// Load a planner line: {controls:[...]} or {toggles:[[tick,mask],...]} with frames|finishFrame,
// the same format loop/score.mjs accepts. Returns null when the file does not exist.
export function loadLine(path = new URL('../learn/best.json', import.meta.url)) {
  if (!fs.existsSync(path)) return null;
  const c = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (Array.isArray(c.controls)) return Uint8Array.from(c.controls);
  if (Array.isArray(c.toggles)) {
    const n = (c.frames ?? c.finishFrame ?? 30000) + 2000;
    const out = new Uint8Array(n);
    const t = [...c.toggles].sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < t.length; i++) out.fill(t[i][1], t[i][0], i + 1 < t.length ? t[i + 1][0] : n);
    return out;
  }
  throw Error('line needs controls or toggles');
}

// Steering key target: left=+1, right=-1, both or none=0.
export function steerTarget(mask) { return ((mask >> 3) & 1) - ((mask >> 1) & 1); }
export function slew(s, target) {
  if (s < target) return Math.min(target, s + STEER_SLEW);
  if (s > target) return Math.max(target, s - STEER_SLEW);
  return s;
}

// Rotate basis vectors by quaternion.
export function forwardOf(q) { return [2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x), 1 - 2 * (q.x * q.x + q.y * q.y)]; }
export function upOf(q) { return [2 * (q.x * q.y - q.w * q.z), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z + q.w * q.x)]; }
export function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

export function buildReference(controls, trackName = 'summer1') {
  const {finishFrame, raw} = runAll(trackName, controls);
  const n = finishFrame ?? controls.length;
  // Goal: the finish if the line finishes, else the last checkpoint index the line reaches
  // (a segment reference, e.g. ctrl/lines/segment.json ends at checkpoint 1).
  const lastState = decodeState(raw, n - 1);
  const goal = finishFrame ? {finish: true} : {finish: false, checkpoint: lastState.nextCheckpointIndex};
  const pos = new Float64Array(3 * n), fwd = new Float64Array(3 * n), left = new Float64Array(3 * n), up = new Float64Array(3 * n);
  const speed = new Float64Array(n), snorm = new Float64Array(n), airborne = new Uint8Array(n);
  const keys = new Uint8Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const st = decodeState(raw, i);
    pos[3 * i] = st.position.x; pos[3 * i + 1] = st.position.y; pos[3 * i + 2] = st.position.z;
    const f = forwardOf(st.quaternion), u = upOf(st.quaternion), l = cross(u, f);
    for (let k = 0; k < 3; k++) { fwd[3 * i + k] = f[k]; up[3 * i + k] = u[k]; left[3 * i + k] = l[k]; }
    speed[i] = st.speedKmh / 3.6;
    airborne[i] = ((raw[i * STATE_BYTES + 11] >> 3) & 15) === 0 ? 1 : 0;
    keys[i] = controls[i];
    s = slew(s, steerTarget(controls[i]));
    snorm[i] = s;
  }
  return {n, finishFrame, goal, pos, fwd, left, up, speed, snorm, airborne, keys, controls};
}
