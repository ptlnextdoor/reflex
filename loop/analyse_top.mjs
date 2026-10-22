// Analyse top leaderboard laps in the byte-exact sim: splits, path, speed, airtime, key usage.
//   node loop/analyse_top.mjs            -> runs/top/analysis.json + a printed summary
// Inputs: vendor/polytrack/0.6.3/top/recording-<id>.json (fetched read-only by loop/top_laps.py).
import fs from 'node:fs';
const {decodeRecording} = await import('../sim/sim.js');
const {Car, decodeState} = await import('../ctrl/sim.mjs');

const dir = new URL('../vendor/polytrack/0.6.3/top/', import.meta.url);
const lb = JSON.parse(fs.readFileSync(new URL('summer1_leaderboard.json', dir), 'utf8'));
const out = [];
for (const e of lb.entries) {
  const f = new URL(`recording-${e.id}.json`, dir);
  if (!fs.existsSync(f)) continue;
  const rec = JSON.parse(fs.readFileSync(f, 'utf8'))[0];
  const controls = decodeRecording(rec.recording, e.frames + 50);
  const car = new Car('summer1');
  let last = -1, air = 0, airRun = 0, maxAir = 0, prevMask = -1, toggles = 0;
  const cps = [], path = [], keyMs = {up: 0, right: 0, down: 0, left: 0, reset: 0};
  for (let t = 0; t < controls.length; t++) {
    const m = controls[t];
    if (m !== prevMask) { toggles++; prevMask = m; }
    if (m & 1) keyMs.up++; if (m & 2) keyMs.right++; if (m & 4) keyMs.down++; if (m & 8) keyMs.left++; if (m & 16) keyMs.reset++;
    const raw = car.step(m), s = decodeState(raw);
    const grounded = (raw[11] >> 3) & 15; // wheel contact bits (same field plan.mjs uses)
    if (!grounded) { air++; airRun++; maxAir = Math.max(maxAir, airRun); } else airRun = 0;
    if (s.nextCheckpointIndex !== last) { cps.push([s.nextCheckpointIndex, t]); last = s.nextCheckpointIndex; }
    if (t % 100 === 0) path.push([t, +s.position.x.toFixed(1), +s.position.y.toFixed(1), +s.position.z.toFixed(1), +s.speedKmh.toFixed(0), m, grounded ? 1 : 0]);
    if (car.finished) { cps.push(['finish', car.finishFrame]); break; }
  }
  const fin = cps.find(c => c[0] === 'finish');
  out.push({id: e.id, name: e.nickname, frames: e.frames, simFinish: fin ? fin[1] : null, cps, airMs: air, maxAirMs: maxAir, toggles, keyMs, path});
  const c = Object.fromEntries(cps.map(([k, t]) => [k, t]));
  const split = [c[1], c[2] - c[1], c[3] - c[2], (c.finish ?? NaN) - c[3]].map(v => (v / 1000).toFixed(2));
  console.log(`${e.nickname.slice(0, 14).padEnd(15)} ${e.frames} sim=${fin ? fin[1] : 'DNF'} splits ${split.join(' ')} air ${(air / 1000).toFixed(1)}s maxAir ${(maxAir / 1000).toFixed(2)}s toggles ${toggles} up ${(keyMs.up / 1000).toFixed(1)}s down ${(keyMs.down / 1000).toFixed(1)}s L ${(keyMs.left / 1000).toFixed(1)} R ${(keyMs.right / 1000).toFixed(1)}`);
}
fs.mkdirSync(new URL('../runs/top/', import.meta.url), {recursive: true});
fs.writeFileSync(new URL('../runs/top/analysis.json', import.meta.url), JSON.stringify(out));
