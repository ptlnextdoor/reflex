import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
import {inflateSync} from 'node:zlib';

export const vendor = fileURLToPath(new URL('../vendor/polytrack/0.6.3/', import.meta.url));
export const STATE_BYTES = 227;
const maxFrames = 5999999;
const source = fs.readFileSync(`${vendor}simulation_worker.bundle.js`, 'utf8');

// Run the original worker's deterministic Math, track parser and constructors.
// ponytail: one synchronous caller per module instance. Use worker_threads for parallel search.
let physics, outputPointer;
const messages = [];
const context = vm.createContext({
  console, performance, TextDecoder, TextEncoder, WebAssembly,
  atob, btoa, Uint8Array, Float32Array, ArrayBuffer, DataView,
  setInterval() {}, clearInterval() {}, setTimeout, clearTimeout,
  postMessage(message) { messages.push(message); },
  onmessage: null, addEventListener() {}, location: {href: `file://${vendor}simulation_worker.bundle.js`},
});
context.self = context;
context.importScripts = name => {
  vm.runInContext(fs.readFileSync(`${vendor}${name}`, 'utf8'), context, {filename: name});
  const factory = context.PolyTrackPhysics;
  context.PolyTrackPhysics = async () => {
    physics = await factory({wasmBinary: fs.readFileSync(`${vendor}polytrack_physics.wasm`)});
    const ccall = physics.ccall;
    physics.ccall = (...args) => {
      const result = ccall(...args);
      if (args[0] === 'malloc' && args[3]?.[0] === STATE_BYTES && outputPointer === undefined)
        outputPointer = result;
      return result;
    };
    return physics;
  };
};
vm.runInContext(source, context, {filename: 'simulation_worker.bundle.js'});
const initialHandler = context.onmessage;
const startup = performance.now();
while (context.onmessage === initialHandler) {
  if (performance.now() - startup > 10000) throw Error('Worker initialization timed out');
  await new Promise(resolve => setImmediate(resolve));
}
const send = data => context.onmessage({data});
let initialized = false;
const fixtures = new Map();
function fixture(trackName) {
  if (!/^[a-z0-9_-]+$/.test(trackName)) throw Error('Invalid track name');
  if (!fixtures.has(trackName)) {
    const path = `${vendor}${trackName}.fixture.json`;
    if (!fs.existsSync(path)) throw Error(`Missing ${path}. Run python3 sim/browser.py first.`);
    const payload = JSON.parse(fs.readFileSync(path, 'utf8'), (_, value) =>
      value?.float32 ? Float32Array.from(value.float32) : value);
    fixtures.set(trackName, payload);
  }
  const result = fixtures.get(trackName);
  if (!initialized) {
    send({...result.init, isRealtime: false});
    initialized = true;
  }
  return result;
}
const u24 = (bytes, offset) => bytes[offset] | bytes[offset + 1] << 8 | bytes[offset + 2] << 16;

// Raw layout: id:u32 @0, frames:u24 @4, speed:f32 @7, flags:u8 @11.
// Flags: started=1, finished=2, checkpoint=4, wheel contacts=8/16/32/64.
// If finished, finishFrames:u24 @12. Then checkpoint:u16, xyz:3*f32,
// xyzw:4*f32, impulseCount:u8, impulses:count*f32, contacts:24 bytes/set flag,
// suspension length/velocity, delta rotation, skid:4*f32 each, steering:f32,
// control/brake bits:u8. Remaining bytes up to 227 are stale padding, not fields.
export function decodeState(raw, index = 0) {
  if (!(raw instanceof Uint8Array) || !Number.isInteger(index) || index < 0 || (index + 1) * STATE_BYTES > raw.length)
    throw new RangeError('State index out of range');
  const base = index * STATE_BYTES;
  const view = new DataView(raw.buffer, raw.byteOffset + base, STATE_BYTES);
  const flags = raw[base + 11];
  const finished = Boolean(flags & 2);
  let offset = finished ? 15 : 12;
  const nextCheckpointIndex = view.getUint16(offset, true);
  offset += 2;
  const position = {x: view.getFloat32(offset, true), y: view.getFloat32(offset + 4, true), z: view.getFloat32(offset + 8, true)};
  offset += 12;
  const quaternion = {x: view.getFloat32(offset, true), y: view.getFloat32(offset + 4, true), z: view.getFloat32(offset + 8, true), w: view.getFloat32(offset + 12, true)};
  offset += 16;
  const impulses = view.getUint8(offset++);
  if (impulses > 4) throw Error('Invalid collision impulse count');
  offset += 4 * impulses;
  for (let wheel = 0; wheel < 4; wheel++) if (flags & (8 << wheel)) offset += 24;
  offset += 68;
  return {
    carId: view.getUint32(0, true), frames: u24(raw, base + 4), speedKmh: view.getFloat32(7, true),
    hasStarted: Boolean(flags & 1), finished, finishFrames: finished ? u24(raw, base + 12) : null,
    nextCheckpointIndex, hasCheckpointToRespawnAt: Boolean(flags & 4), position, quaternion,
    controls: view.getUint8(offset) & 31, numberOfBytes: offset + 1,
  };
}

export function run(trackName, controls) {
  if (!(controls instanceof Uint8Array) || controls.length > maxFrames || controls.some(mask => mask > 31))
    throw Error('Controls must be Uint8Array of 5-bit masks, at most 5999999 ms');
  const {create} = fixture(trackName);
  // Allocate output before car creation so failed allocation cannot strand a car.
  const raw = new Uint8Array(controls.length * STATE_BYTES);
  send({...create, carRecording: null});
  const id = create.carId;
  let finishFrame = null;
  try {
    physics.HEAPU8.fill(0, outputPointer, outputPointer + STATE_BYTES);
    // No per-tick JS allocation. The physics implementation remains unmodified.
    // Car creation has already provisioned the WASM heap for this track.
    const state = new Uint8Array(physics.HEAPU8.buffer, outputPointer, STATE_BYTES);
    const update = physics._updateCarModel;
    for (let tick = 0; tick < controls.length; tick++) {
      const mask = controls[tick];
      update(id, mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1, (mask >> 4) & 1, outputPointer);
      raw.set(state, tick * STATE_BYTES);
      if (finishFrame === null && (state[11] & 2)) finishFrame = u24(state, 12);
    }
  } finally {
    send({messageType: 4, carId: id});
  }
  // Decode only requested states, avoiding object allocation on the simulation path.
  const states = {
    raw, length: controls.length,
    at(index) { return decodeState(raw, index < 0 ? this.length + index : index); },
    *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this.at(i); },
  };
  return {finishFrame, states};
}

export function decodeRecording(recording, frames) {
  if (typeof recording !== 'string' || !/^[\w-]+$/.test(recording)) throw Error('Invalid recording encoding');
  // ponytail: reject recordings over 9 MB inflated (3 million toggles). Raise for synthetic dense inputs.
  const raw = inflateSync(Buffer.from(recording, 'base64url'), {maxOutputLength: 9000015});
  const events = [];
  let offset = 0, last = 0;
  for (let bit = 0; bit < 5; bit++) {
    if (offset + 3 > raw.length) throw Error('Truncated recording');
    const count = u24(raw, offset);
    offset += 3;
    if (offset + count * 3 > raw.length) throw Error('Truncated recording events');
    let tick = 0;
    for (let i = 0; i < count; i++) {
      const delta = u24(raw, offset);
      if (i > 0 && delta === 0) throw Error('Recording toggle frames must increase');
      tick += delta;
      offset += 3;
      if (tick > maxFrames) throw Error('Recording exceeds maximum duration');
      events.push([tick, 1 << bit]);
      last = Math.max(last, tick);
    }
  }
  if (offset !== raw.length) throw Error('Trailing recording bytes');
  // ponytail: recordings store toggles, not duration. Default ends after last event.
  // Supply metadata.frames to extend final held controls through the finish.
  frames ??= last + 1;
  if (!Number.isInteger(frames) || frames < 0 || frames > maxFrames) throw Error('Invalid recording duration');
  const controls = new Uint8Array(frames);
  for (const [tick, mask] of events) if (tick < frames) controls[tick] ^= mask;
  let mask = 0;
  for (let tick = 0; tick < frames; tick++) controls[tick] = mask ^= controls[tick];
  return controls;
}

export function verifyRecording(trackName, recording, frames) {
  const {create} = fixture(trackName);
  messages.length = 0;
  send({...create, messageType: 1, carRecording: recording, targetFrames: frames});
  const result = messages.find(message => message.messageType === 8);
  return result?.result === true;
}
