import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {deflateSync} from 'node:zlib';
import {run, decodeRecording, decodeState, verifyRecording, vendor, STATE_BYTES} from './sim.js';

// Fresh oracle every check, never a cached pass or a port of the physics.
execFileSync('python3', [new URL('browser.py', import.meta.url).pathname], {stdio: 'inherit', timeout: 120000});
const controls = new Uint8Array(5000).fill(1);
controls.fill(3, 2000);
const first = run('summer1', controls);
const second = run('summer1', controls);
function sameBytes(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label}: byte length`);
  const mismatch = actual.findIndex((value, index) => value !== expected[index]);
  assert.equal(mismatch, -1, `${label}: first divergent tick=${Math.floor(mismatch / STATE_BYTES) + 1}, offset=${mismatch % STATE_BYTES}, Node=${actual[mismatch]}, oracle=${expected[mismatch]}`);
}
sameBytes(first.states.raw, second.states.raw, 'Determinism');
console.log(`determinism PASS: 5000 states, ${first.states.raw.length} bytes, SHA256 ${createHash('sha256').update(first.states.raw).digest('hex')}`);
const oracle = fs.readFileSync(`${vendor}parity.browser.bin`);
sameBytes(first.states.raw, oracle, 'Chromium parity');
for (let index = 0; index < first.states.length; index++) {
  const state = first.states.at(index);
  assert.equal(state.frames, index + 1);
  assert.equal(state.controls, controls[index]);
  assert.ok(state.numberOfBytes <= STATE_BYTES);
}
for (const frame of [1000, 2000, 5000]) {
  const node = first.states.at(frame - 1);
  const browser = decodeState(oracle, frame - 1);
  assert.deepEqual(node.position, browser.position);
  console.log(`parity PASS frame=${frame} Node=Chromium xyz=${JSON.stringify(node.position)} maxError=0`);
}

const recording = JSON.parse(fs.readFileSync(`${vendor}recording-59372564.json`, 'utf8'))[0];
assert.equal(recording.frames, 22123);
const ghost = decodeRecording(recording.recording, recording.frames);
const replay = run('summer1', ghost);
sameBytes(replay.states.raw, fs.readFileSync(`${vendor}recording.browser.bin`), '22123 ms recording parity');
assert.equal(replay.finishFrame, 22123);
assert.equal(replay.states.at(-1).nextCheckpointIndex, 3);
assert.equal(verifyRecording('summer1', recording.recording, recording.frames), true);
assert.equal(verifyRecording('summer1', recording.recording, recording.frames - 1), false);
console.log(`recording parity PASS: id=59372564, ${ghost.length} states (${replay.states.raw.length} bytes), finishFrame=${replay.finishFrame}, original Verify=true, early Verify=false`);

// Malformed inputs, repeated toggles, idle start, braking and checkpoint reset.
assert.throws(() => run('../summer1', controls), /Invalid track name/);
assert.throws(() => run('summer1', new Uint8Array([32])), /5-bit/);
assert.throws(() => run('summer1', []), /Uint8Array/);
assert.throws(() => decodeState(new Uint8Array(1)), /range/);
assert.throws(() => decodeRecording('!'), /encoding/);
assert.throws(() => decodeRecording(deflateSync(new Uint8Array(2)).toString('base64url')), /Truncated/);
const toggles = Buffer.from([2,0,0, 0,0,0, 3,0,0, 1,0,0, 2,0,0, 0,0,0, 0,0,0, 0,0,0]);
assert.deepEqual([...decodeRecording(deflateSync(toggles).toString('base64url'), 6)], [1,1,3,2,2,2]);
const duplicate = Buffer.from(toggles);
duplicate[6] = 0;
assert.throws(() => decodeRecording(deflateSync(duplicate).toString('base64url')), /must increase/);
assert.equal(run('summer1', new Uint8Array()).states.length, 0);
assert.equal(run('summer1', new Uint8Array(10)).states.at(-1).frames, 10);
const mixed = new Uint8Array(6000).fill(1);
mixed.fill(5, 2500, 3000);
mixed.fill(17, 4000, 4001);
sameBytes(run('summer1', mixed).states.raw, run('summer1', mixed).states.raw, 'Brake/reset determinism');
assert.deepEqual(run('summer1', controls).states.raw, first.states.raw);
console.log('edge checks PASS: invalid inputs, recording toggles, empty/idle, brake/reset, post-Verify replay');

const throttle = run('summer1', new Uint8Array(60000).fill(1));
const last = throttle.states.at(-1);
console.log(`Summer 1 full-throttle: finishFrame=${throttle.finishFrame}, horizon=60000 ms, xyz=${JSON.stringify(last.position)}, heightDrop=${(throttle.states.at(0).position.y - last.position.y).toFixed(3)} m, speedKmh=${last.speedKmh}, checkpoint=${last.nextCheckpointIndex} (fell below start, did not finish)`);

// Single Node thread, warmed end-to-end run includes car create/delete and output copy.
for (let i = 0; i < 3; i++) run('summer1', ghost);
const times = [];
for (let i = 0; i < 30; i++) {
  const start = performance.now();
  const result = run('summer1', ghost);
  times.push(performance.now() - start);
  assert.equal(result.finishFrame, 22123);
}
times.sort((a, b) => a - b);
const p50 = times[Math.floor(times.length * 0.5)];
const p99 = times[Math.ceil(times.length * 0.99) - 1];
const total = times.reduce((a, b) => a + b, 0);
const throughput = ghost.length * times.length / total;
console.log(`throughput: ${throughput.toFixed(2)} simulated ms/wall ms (${Math.round(throughput * 1000)} ticks/s), single thread, 30 x ${ghost.length} ticks`);
console.log(`run latency: p50=${p50.toFixed(3)} ms p99=${p99.toFixed(3)} ms, amortized per tick p50=${(p50 / ghost.length).toFixed(6)} ms p99=${(p99 / ghost.length).toFixed(6)} ms; Node ${process.version} ${process.arch}`);
console.log(throughput > 50 ? 'throughput target PASS (>50x realtime)' : 'throughput target MISSED (>50x realtime)');
