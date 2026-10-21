// Parity: worker snapshot evaluation must agree with sim.run on finish frame for random mutations.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {run, decodeRecording, vendor} from '../sim/sim.js';
import {Pool} from './pool.mjs';
import {togglesFromControls, controlsFromToggles} from './recording.mjs';

const rec = JSON.parse(fs.readFileSync(`${vendor}recording-59372564.json`, 'utf8'))[0];
const wr = decodeRecording(rec.recording, rec.frames);
const pool = new Pool(4);
let t0 = performance.now();
const b = await pool.setBase(wr, 250);
console.log('setBase', JSON.stringify({...b}), 'ms', (performance.now() - t0).toFixed(0));
assert.equal(b.finishFrame, 22123);
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
const toggles = togglesFromControls(wr);
const cands = [];
for (let k = 0; k < 24; k++) {
  const t = toggles.map(x => [...x]);
  const i = Math.floor(rand() * t.length);
  t[i][0] = Math.max(0, t[i][0] + Math.round((rand() * 2 - 1) * 20));
  cands.push(controlsFromToggles(t, 22123 + 500));
}
cands.push(wr);
t0 = performance.now();
const results = await pool.evaluateAll(cands, 22123 + 500);
const wall = performance.now() - t0;
let agree = 0;
for (let k = 0; k < cands.length; k++) {
  const ref = run('summer1', cands[k]).finishFrame;
  if (ref !== results[k].finishFrame) console.log('MISMATCH', k, ref, results[k].finishFrame, results[k]);
  else agree++;
}
console.log(`agreement ${agree}/${cands.length}; ${cands.length} evals in ${wall.toFixed(0)} ms on 4 workers; ticks/eval avg ${(results.reduce((a, r) => a + r.ticks, 0) / results.length).toFixed(0)}`);
console.log('sample result', JSON.stringify(results[0]));
assert.equal(agree, cands.length);
await pool.close();
