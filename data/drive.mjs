// Random-but-sane drives for the data engine: full throttle with random steering pulses,
// brake pulses, and checkpoint resets when the car falls off or stalls. Closed loop on the
// byte-exact sim (ctrl/sim.mjs), so resets react to the actual state. An episode ends at the
// finish, at the horizon, or when the car is stuck with no checkpoint to respawn at (before
// the first checkpoint the sim's reset bit is a no-op, the game restarts with a fresh car).
// Writes one controls file per episode that data/labels.mjs and data/render.py consume.
// No human input anywhere.
// Usage: node data/drive.mjs <outDir> [track=summer1] [seeds=1-8] [ms=20000]
import fs from 'node:fs';
import path from 'node:path';
import {Car, decodeState} from '../ctrl/sim.mjs';

const outDir = process.argv[2] ?? 'runs/drives';
const track = process.argv[3] ?? 'summer1';
const [seedLo, seedHi] = (process.argv[4] ?? '1-8').split('-').map(Number);
const ms = Number(process.argv[5] ?? 20000);
if (!/^[a-z0-9_-]+$/.test(track)) throw Error('Invalid track name');

function rng(seed) {
  let s = (seed >>> 0) || 1;
  const next = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const pick = table => { let r = next(); for (const [v, p] of table) { if ((r -= p) < 0) return v; } return table[table.length - 1][0]; };
  return {next, pick};
}

export function drive(track, seed, ms) {
  const r = rng(seed);
  const car = new Car(track);
  const controls = new Uint8Array(ms);
  let mask = 1, holdUntil = 0, stallSince = null, resets = 0, startY = null, finish = null, end = ms, stuck = false;
  try {
    for (let t = 0; t < ms; t++) {
      if (t >= holdUntil) {
        // Steering: straight 50%, left 24%, right 24%, both (no-op) 2%. Throttle: gas 86%, coast 6%, brake 8%.
        const steer = r.pick([[0, 0.5], [8, 0.24], [2, 0.24], [10, 0.02]]);
        const throttle = r.pick([[1, 0.86], [0, 0.06], [4, 0.08]]);
        mask = steer | throttle;
        holdUntil = t + 40 + Math.floor(r.next() * 400);
      }
      let m = mask;
      const st = decodeState(car.state);
      startY ??= st.position.y;
      if (st.hasStarted && t > 1500) {
        if (st.speedKmh < 3) stallSince ??= t; else stallSince = null;
        const fell = st.position.y < startY - 25;
        if (fell || (stallSince !== null && t - stallSince > 1200)) {
          if (!st.hasCheckpointToRespawnAt) { stuck = true; end = t; break; }
          m |= 16; resets++; stallSince = null; holdUntil = t + 250; mask = 1;
        }
      }
      controls[t] = m;
      const raw = car.step(m);
      if (raw[11] & 2) { finish = raw[12] | raw[13] << 8 | raw[14] << 16; end = finish; break; }
    }
  } finally {
    car.remove();
  }
  const toggles = [[0, controls[0]]];
  for (let i = 1; i < end; i++) if (controls[i] !== controls[i - 1]) toggles.push([i, controls[i]]);
  return {track, seed, frames: end, finishFrame: finish, resets, stuck, toggles};
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  fs.mkdirSync(outDir, {recursive: true});
  for (let seed = seedLo; seed <= (seedHi ?? seedLo); seed++) {
    const d = drive(track, seed, ms);
    const file = path.join(outDir, `${track}_s${seed}.json`);
    fs.writeFileSync(file, JSON.stringify(d));
    console.log(`${file}: frames=${d.frames} toggles=${d.toggles.length} resets=${d.resets} stuck=${d.stuck} finish=${d.finishFrame}`);
  }
}
