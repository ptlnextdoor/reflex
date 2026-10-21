// Exact labels from the byte-exact Node sim for ANY control sequence. Reads a controls file
// ({track, frames, controls:[mask...]} or {track, frames, toggles:[[tick, mask], ...]}),
// runs it through sim.run, and writes states.bin (all 227-byte states) plus labels.jsonl
// (one row per sampled frame) into an output directory.
// Usage: node data/labels.mjs <controls.json> <outDir> [stepMs=50]
import fs from 'node:fs';
import path from 'node:path';
import {run, decodeState, STATE_BYTES} from '../sim/sim.js';

export const BITS = {up: 1, right: 2, down: 4, left: 8, reset: 16};
export function steer(mask) {
  const left = Boolean(mask & BITS.left), right = Boolean(mask & BITS.right);
  return left === right ? 'straight' : left ? 'left' : 'right';
}
export function throttle(mask) {
  if (mask & BITS.down) return 'brake';
  return mask & BITS.up ? 'gas' : 'coast';
}
export function loadControls(file) {
  const j = JSON.parse(fs.readFileSync(file, 'utf8'));
  const track = j.track ?? 'summer1';
  if (!/^[a-z0-9_-]+$/.test(track)) throw Error('Invalid track name');
  let controls;
  if (Array.isArray(j.controls)) controls = Uint8Array.from(j.controls);
  else if (Array.isArray(j.toggles)) {
    const frames = j.frames ?? j.toggles[j.toggles.length - 1][0] + 1;
    controls = new Uint8Array(frames);
    let mask = 0, next = 0;
    for (let t = 0; t < frames; t++) {
      while (next < j.toggles.length && j.toggles[next][0] <= t) mask = j.toggles[next++][1];
      controls[t] = mask;
    }
  } else throw Error('controls file needs controls[] or toggles[]');
  return {track, controls};
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const [file, outDir = 'runs/frames/drive', stepArg = '50'] = process.argv.slice(2);
  if (!file) throw Error('usage: node data/labels.mjs <controls.json> <outDir> [stepMs]');
  const step = Number(stepArg);
  const {track, controls} = loadControls(file);
  const {finishFrame, states} = run(track, controls);
  // Never feed a finished state to the game (it would trigger the finish/upload path).
  const lastFrame = finishFrame === null ? controls.length : finishFrame - 1;
  fs.mkdirSync(outDir, {recursive: true});
  fs.writeFileSync(path.join(outDir, 'states.bin'), states.raw.subarray(0, lastFrame * STATE_BYTES));
  const rows = [];
  for (let k = step; k <= lastFrame; k += step) {
    // Frame k is the state after k ticks: raw index k-1. mask_held is the control byte the
    // physics saw for that tick; mask_next is the decision to take at frame k.
    const s = decodeState(states.raw, k - 1);
    if (s.frames !== k) throw Error(`frame index mismatch at ${k}`);
    const brakeLight = Boolean(states.raw[(k - 1) * STATE_BYTES + s.numberOfBytes - 1] & 32);
    const maskNext = k < controls.length ? controls[k] : 0;
    rows.push(JSON.stringify({
      k, track, position: s.position, quaternion: s.quaternion, speedKmh: s.speedKmh,
      nextCheckpointIndex: s.nextCheckpointIndex, hasStarted: s.hasStarted,
      mask_held: s.controls, mask_next: maskNext, brakeLight,
      steer: steer(maskNext), throttle: throttle(maskNext),
    }));
  }
  fs.writeFileSync(path.join(outDir, 'labels.jsonl'), rows.join('\n') + '\n');
  fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify({track, source: path.resolve(file), frames: lastFrame, finishFrame, step}));
  console.log(`${rows.length} labels, track=${track} frames ${step}..${lastFrame} step ${step}, finishFrame=${finishFrame}`);
}
