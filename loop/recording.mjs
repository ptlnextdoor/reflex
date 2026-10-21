// Toggle-list <-> controls <-> Polytrack recording string. Inverse of sim.decodeRecording.
import {deflateSync, inflateSync} from 'node:zlib';

// controls (Uint8Array of 5-bit masks per ms) -> toggles [[tick, bit], ...] sorted by tick.
export function togglesFromControls(controls) {
  const toggles = [];
  let prev = 0;
  for (let t = 0; t < controls.length; t++) {
    const diff = controls[t] ^ prev;
    if (!diff) continue;
    for (let bit = 0; bit < 5; bit++) if (diff & (1 << bit)) toggles.push([t, bit]);
    prev = controls[t];
  }
  return toggles;
}

// toggles -> controls of length frames. Two toggles of one bit on the same tick cancel (as in decodeRecording).
export function controlsFromToggles(toggles, frames) {
  const controls = new Uint8Array(frames);
  for (const [tick, bit] of toggles) if (tick >= 0 && tick < frames) controls[tick] ^= 1 << bit;
  let mask = 0;
  for (let t = 0; t < frames; t++) controls[t] = mask ^= controls[t];
  return controls;
}

const u24 = n => [n & 255, (n >> 8) & 255, (n >> 16) & 255];

// Raw payload: for each of 5 bits, u24 count then u24 deltas (first absolute, then strictly positive differences).
export function encodeRecordingRaw(controls) {
  const lanes = Array.from({length: 5}, () => []);
  for (const [tick, bit] of togglesFromControls(controls)) lanes[bit].push(tick);
  const bytes = [];
  for (const lane of lanes) {
    bytes.push(...u24(lane.length));
    let last = 0;
    for (const tick of lane) { bytes.push(...u24(tick - last)); last = tick; }
  }
  return Buffer.from(bytes);
}

export function encodeRecording(controls, level = 9) {
  return deflateSync(encodeRecordingRaw(controls), {level}).toString('base64url');
}

export const inflateRecording = recording => inflateSync(Buffer.from(recording, 'base64url'));
