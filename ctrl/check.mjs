// Stepper parity: ctrl/sim.mjs Car stepping must be byte-identical to sim.run on the WR
// recording (finish at exactly 22123), and snapshot/restore must reproduce the same bytes.
//   node ctrl/check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {run} from '../sim/sim.js';
import {Car, runAll, decodeRecording, decodeState, vendor, STATE_BYTES} from './sim.mjs';

const recording = JSON.parse(fs.readFileSync(`${vendor}recording-59372564.json`, 'utf8'))[0];
const wr = decodeRecording(recording.recording, recording.frames);
const reference = run('summer1', wr);
assert.equal(reference.finishFrame, 22123);

function sameBytes(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label}: byte length`);
  const mismatch = actual.findIndex((value, index) => value !== expected[index]);
  assert.equal(mismatch, -1, `${label}: first divergent tick=${Math.floor(mismatch / STATE_BYTES) + 1}, offset=${mismatch % STATE_BYTES}`);
}

// 1. Whole run through the stepper.
const stepped = runAll('summer1', wr);
assert.equal(stepped.finishFrame, 22123);
sameBytes(stepped.raw, reference.states.raw, 'stepper vs sim.run (WR)');
console.log(`stepper parity PASS: ${wr.length} ticks byte-identical to sim.run, finishFrame=${stepped.finishFrame}`);

// 2. Second car after the first was removed: still identical (no state leaks between episodes).
const again = runAll('summer1', wr);
sameBytes(again.raw, reference.states.raw, 'second episode');
console.log('episode isolation PASS');

// 3. Snapshot at tick 10000, step 500 ms of a different input, restore, step the WR input: identical to the reference.
{
  const car = new Car('summer1');
  const raw = new Uint8Array(wr.length * STATE_BYTES);
  for (let t = 0; t < 10000; t++) raw.set(car.step(wr[t]), t * STATE_BYTES);
  const snap = car.snapshot();
  for (let t = 10000; t < 10500; t++) car.step(2);
  const divergedPosition = decodeState(car.state).position;
  car.restore(snap);
  car.tick = 10000;
  let finishFrame = null;
  for (let t = 10000; t < wr.length; t++) {
    const s = car.step(wr[t]);
    raw.set(s, t * STATE_BYTES);
    if (finishFrame === null && (s[11] & 2)) finishFrame = s[12] | s[13] << 8 | s[14] << 16;
  }
  car.remove();
  assert.equal(finishFrame, 22123);
  sameBytes(raw, reference.states.raw, 'snapshot/restore');
  const refPosition = decodeState(reference.states.raw, 10499).position;
  assert.notDeepEqual(divergedPosition, refPosition, 'the diverged branch really diverged');
  console.log(`snapshot/restore PASS: 500 ms branch diverged by ${Math.hypot(divergedPosition.x - refPosition.x, divergedPosition.z - refPosition.z).toFixed(2)} m, restore reproduced all ${wr.length} ticks`);
}

// 4. Throughput of the stepper alone (controller loops call step per ms).
{
  const car = new Car('summer1');
  const start = performance.now();
  for (let t = 0; t < wr.length; t++) car.step(wr[t]);
  const ms = performance.now() - start;
  car.remove();
  console.log(`stepper throughput: ${(wr.length / ms * 1000 | 0)} ticks/s single thread (${ms.toFixed(1)} ms for ${wr.length} ticks)`);
}
