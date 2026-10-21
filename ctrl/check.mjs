// Stepper parity: ctrl/sim.mjs Car stepping must be byte-identical to sim.run on the same
// controls, and snapshot/restore must reproduce the same bytes. Uses synthetic controls only
// (a seeded random key script plus a hand-written throttle/steer pattern), no human data.
//   node ctrl/check.mjs
import assert from 'node:assert/strict';
import {run} from '../sim/sim.js';
import {Car, runAll, decodeState, STATE_BYTES} from './sim.mjs';

function sameBytes(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label}: byte length`);
  const mismatch = actual.findIndex((value, index) => value !== expected[index]);
  assert.equal(mismatch, -1, `${label}: first divergent tick=${Math.floor(mismatch / STATE_BYTES) + 1}, offset=${mismatch % STATE_BYTES}`);
}

// Synthetic script: 20 s of random key holds (seeded LCG), 30 to 400 ms per segment,
// throttle mostly on, occasional brake, steer left/right/none, one reset at 12 s.
function script(seed, n = 20000) {
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const controls = new Uint8Array(n);
  let t = 0;
  while (t < n) {
    const len = 30 + Math.floor(rand() * 370);
    const steer = rand() < 0.3 ? 8 : rand() < 0.5 ? 2 : 0;
    const mask = (rand() < 0.9 ? 1 : 0) | (rand() < 0.1 ? 4 : 0) | steer;
    controls.fill(mask, t, Math.min(n, t + len));
    t += len;
  }
  controls[12000] = 16;
  return controls;
}
const scripts = [script(1), script(2)];
const patterned = new Uint8Array(15000).fill(1);
for (let t = 1000; t < 15000; t += 700) patterned.fill(t % 1400 ? 3 : 9, t, t + 250);
scripts.push(patterned);

for (const [k, controls] of scripts.entries()) {
  const reference = run('summer1', controls);
  const stepped = runAll('summer1', controls);
  assert.equal(stepped.finishFrame, reference.finishFrame);
  sameBytes(stepped.raw, reference.states.raw, `stepper vs sim.run (script ${k})`);
  const last = decodeState(reference.states.raw, controls.length - 1);
  console.log(`stepper parity PASS: script ${k}, ${controls.length} ticks byte-identical to sim.run, end pos=(${last.position.x.toFixed(1)},${last.position.y.toFixed(1)},${last.position.z.toFixed(1)}) speed=${last.speedKmh.toFixed(0)} kmh`);
}

// Episode isolation: a second car after the first was removed gives the same bytes.
{
  const controls = scripts[0];
  const reference = run('summer1', controls);
  const again = runAll('summer1', controls);
  sameBytes(again.raw, reference.states.raw, 'second episode');
  console.log('episode isolation PASS');
}

// Snapshot at tick 8000, step 500 ms of a different input, restore, continue the script: identical.
{
  const controls = scripts[0];
  const reference = run('summer1', controls);
  const car = new Car('summer1');
  const raw = new Uint8Array(controls.length * STATE_BYTES);
  for (let t = 0; t < 8000; t++) raw.set(car.step(controls[t]), t * STATE_BYTES);
  const snap = car.snapshot();
  for (let t = 8000; t < 8500; t++) car.step(controls[t] ^ 10);
  const divergedPosition = decodeState(car.state).position;
  car.restore(snap);
  car.tick = 8000;
  for (let t = 8000; t < controls.length; t++) raw.set(car.step(controls[t]), t * STATE_BYTES);
  car.remove();
  sameBytes(raw, reference.states.raw, 'snapshot/restore');
  const refPosition = decodeState(reference.states.raw, 8499).position;
  assert.notDeepEqual(divergedPosition, refPosition, 'the diverged branch really diverged');
  console.log(`snapshot/restore PASS: 500 ms branch diverged by ${Math.hypot(divergedPosition.x - refPosition.x, divergedPosition.z - refPosition.z).toFixed(2)} m, restore reproduced all ${controls.length} ticks`);
}

// Throughput of the stepper alone (controller loops call step per ms).
{
  const controls = scripts[0];
  const car = new Car('summer1');
  const start = performance.now();
  for (let t = 0; t < controls.length; t++) car.step(controls[t]);
  const ms = performance.now() - start;
  car.remove();
  console.log(`stepper throughput: ${(controls.length / ms * 1000 | 0)} ticks/s single thread (${ms.toFixed(1)} ms for ${controls.length} ticks)`);
}
