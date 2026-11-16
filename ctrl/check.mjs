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

// Sparse snapshots under the audit contract (ctrl/sim.mjs initPages): audit before the
// snapshot and after a diverged 3 s branch, then restorePages must give back the whole heap
// byte for byte (every page, not only S), at many ticks of three scripts.
{
  const car = new Car('summer1');
  const t0 = performance.now();
  const calibrated = car.calibratePages();
  const calMs = performance.now() - t0;
  const pristine = car.snapshot();
  let checks = 0, grown = 0, restoreMs = 0;
  const audit = () => { const found = car.auditPages(); grown += found.length; car.addPages(found); };
  for (const controls of scripts) {
    car.restore(pristine);
    for (let t = 0; t < controls.length - 3000; t++) {
      car.step(controls[t]);
      if ((t + 1) % 1500) continue;
      audit();
      const full = car.snapshot(), sparse = car.snapshotPages();
      for (let b = 0; b < 3000; b++) car.step(controls[t + 1 + b] ^ 10);
      audit();
      const r0 = performance.now();
      car.restorePages(sparse);
      restoreMs += performance.now() - r0;
      sameBytes(car.snapshot(), full, `sparse restore at tick ${t + 1}`);
      checks++;
    }
  }
  const a0 = performance.now();
  car.auditPages();
  const auditMs = performance.now() - a0;
  car.remove();
  console.log(`sparse snapshot PASS: ${calibrated} pages calibrated in ${calMs.toFixed(0)} ms, +${grown} by audits, ${checks} whole-heap restores exact, restore ${(restoreMs / checks).toFixed(3)} ms, audit ${auditMs.toFixed(2)} ms`);
}

// Two cars in one physics instance do not interact: car A's payload bytes are identical with a
// second car present (even one spawned at the start while A drives), so a sim twin can run next
// to the real car without a heap copy per frame.
{
  const controls = scripts[2];
  const reference = run('summer1', controls);
  const a = new Car('summer1');
  const raw = new Uint8Array(controls.length * STATE_BYTES);
  for (let t = 0; t < 1000; t++) raw.set(a.step(controls[t]), t * STATE_BYTES);
  const b = new Car('summer1', {carId: a.id + 1});
  a.refreshView();
  for (let t = 1000; t < controls.length; t++) { raw.set(a.step(controls[t]), t * STATE_BYTES); b.step(controls[t] ^ 10); }
  b.remove(); a.remove();
  let payloadMismatch = -1;
  for (let t = 0; t < controls.length && payloadMismatch < 0; t++) {
    const bytes = decodeState(reference.states.raw, t).numberOfBytes;
    for (let o = 0; o < bytes; o++) if (raw[t * STATE_BYTES + o] !== reference.states.raw[t * STATE_BYTES + o]) { payloadMismatch = t; break; }
  }
  assert.equal(payloadMismatch, -1, 'second car changed the first car\'s payload');
  console.log('two-car independence PASS: payload bytes identical with a second car present (padding beyond numberOfBytes may differ)');
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
