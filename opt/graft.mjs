// Graft an explored segment onto a full lap and measure the finish.
// The explored line reaches checkpoint `cp` at tick T; the base line reaches it at tick B.
// Result = explored controls up to T, then the base line's controls from B on (time-shifted by T-B).
// Different car state at the joint means the tail may not hold; we report the finish (or where it ends).
//   node opt/graft.mjs --explored runs/server/explore_X.json --base runs/server/nickita_22038.json --cp 1 --out runs/server/graft_X.json
import fs from 'node:fs';
const {Car, decodeState} = await import('../ctrl/sim.mjs');
const {decodeRecording} = await import('../sim/sim.js');
const {encodeRecording} = await import('./recording.mjs');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i < 0 ? d : process.argv[i + 1]; };
const load = p => { const j = JSON.parse(fs.readFileSync(p, 'utf8')); return j.recording ? Array.from(decodeRecording(j.recording, j.frames + 200)) : (j.best?.controls ?? j.controls); };
const ex = load(arg('--explored')), base = load(arg('--base')), cp = Number(arg('--cp'));
const reach = c => { const car = new Car('summer1'); for (let t = 0; t < c.length; t++) { const s = decodeState(car.step(c[t])); if (s.nextCheckpointIndex >= cp || car.finished) return t + 1; } return null; };
const T = reach(ex), B = reach(base);
// try joint offsets around the checkpoint: join the base tail a few ms early/late
let best = null;
for (const slip of [-40, -20, -10, -5, 0, 5, 10, 20, 40]) {
  const ctrl = ex.slice(0, T + slip).concat(base.slice(B + slip));
  const car = new Car('summer1'); let s, fin = null;
  for (let t = 0; t < ctrl.length + 50; t++) { s = decodeState(car.step(ctrl[Math.min(t, ctrl.length - 1)])); if (car.finished) { fin = car.finishFrame; break; } }
  const r = {slip, finish: fin, endCp: s.nextCheckpointIndex};
  console.log(JSON.stringify(r));
  if (fin && (!best || fin < best.finish)) best = {...r, ctrl};
}
console.log(`explored cp${cp} at ${T} vs base ${B} (${B - T} ms earlier); best graft finish ${best ? best.finish : 'none'}`);
if (best && arg('--out')) fs.writeFileSync(arg('--out'), JSON.stringify({finishFrame: best.finish, frames: best.finish, recording: encodeRecording(Uint8Array.from(best.ctrl.slice(0, best.finish)))}));
