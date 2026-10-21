// Where an own lap loses time: speed, airtime, and wheel contact per 500 ms segment, and the
// tick each checkpoint is crossed. Own lines only (learn/, ctrl/); never feed it human runs.
//   node learn/profile.mjs learn/best.json
import fs from 'node:fs';
import {Car, decodeState} from '../ctrl/sim.mjs';

const j = JSON.parse(fs.readFileSync(process.argv[2] ?? new URL('best.json', import.meta.url)));
const controls = j.controls ?? (() => {
  const n = j.frames, c = new Uint8Array(n); let k = 0, v = 0;
  for (let t = 0; t < n; t++) { while (k < j.masks.length && j.masks[k][0] <= t) v = j.masks[k++][1]; c[t] = v; }
  return c;
})();
const car = new Car('summer1');
let seg = null, cp = -1;
const rows = [];
for (let t = 0; t < controls.length && !car.finished; t++) {
  const s = car.step(controls[t]);
  const d = decodeState(s);
  if (d.nextCheckpointIndex !== cp) { cp = d.nextCheckpointIndex; rows.push(`  checkpoint -> ${cp} at ${(t / 1000).toFixed(3)} s`); }
  if (t % 500 === 0) {
    if (seg) rows.push(`${(seg.t0 / 1000).toFixed(1).padStart(5)} s  v ${(seg.v / seg.n).toFixed(0).padStart(3)} km/h  air ${String(Math.round(100 * seg.air / seg.n)).padStart(3)}%  mask ${[...seg.masks].join(',')}`);
    seg = {t0: t, v: 0, n: 0, air: 0, masks: new Set()};
  }
  seg.v += d.speedKmh; seg.n++;
  if ((s[11] & 0x78) === 0) seg.air++;
  seg.masks.add(controls[t]);
}
rows.push(`finish ${car.finishFrame}`);
for (const r of rows) console.log(typeof r === 'string' ? r.replace(/\[object Set\]/, '') : r);
car.remove();
