// Per-millisecond stepping wrapper around the headless Polytrack physics. Same vm context,
// fixture and CreateCar message as sim/sim.js, but the car persists between calls so a
// controller can decide keys from the current state, advance N ms, and repeat.
// Byte-identical to sim.run on the same controls (checked by `node ctrl/check.mjs`).
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
export {decodeState, STATE_BYTES} from '../sim/sim.js';

export const vendor = fileURLToPath(new URL('../vendor/polytrack/0.6.3/', import.meta.url));
const STATE = 227;
const source = fs.readFileSync(`${vendor}simulation_worker.bundle.js`, 'utf8');

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
      if (args[0] === 'malloc' && args[3]?.[0] === STATE && outputPointer === undefined) outputPointer = result;
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
    fixtures.set(trackName, JSON.parse(fs.readFileSync(path, 'utf8'), (_, value) =>
      value?.float32 ? Float32Array.from(value.float32) : value));
  }
  const result = fixtures.get(trackName);
  if (!initialized) {
    send({...result.init, isRealtime: false});
    initialized = true;
  }
  return result;
}

// One live car. ponytail: one car at a time per module instance (the worker's car list is
// internal); use worker_threads for parallel episodes, as sim.js does.
export class Car {
  constructor(trackName = 'summer1') {
    const {create} = fixture(trackName);
    send({...create, carRecording: null});
    this.id = create.carId;
    this.tick = 0;
    physics.HEAPU8.fill(0, outputPointer, outputPointer + STATE);
    // Live view of the 227-byte output record, valid until the heap grows (it does not during stepping).
    this.state = new Uint8Array(physics.HEAPU8.buffer, outputPointer, STATE);
  }
  // Advance one ms with a 5-bit mask (up=1 right=2 down=4 left=8 reset=16). Returns the live state view.
  step(mask) {
    physics._updateCarModel(this.id, mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1, (mask >> 4) & 1, outputPointer);
    this.tick++;
    return this.state;
  }
  get finished() { return (this.state[11] & 2) !== 0; }
  get finishFrame() { return this.finished ? this.state[12] | this.state[13] << 8 | this.state[14] << 16 : null; }
  // Whole-heap snapshot for model-predictive rollouts. 23 MB, ~0.6 ms copy, ~0.4 ms restore (M4 Pro).
  snapshot(into) {
    const heap = physics.HEAPU8;
    if (!into || into.length !== heap.length) into = new Uint8Array(heap.length);
    into.set(heap);
    return into;
  }
  restore(snap) {
    if (snap.length !== physics.HEAPU8.length) throw Error('Heap size changed since snapshot');
    physics.HEAPU8.set(snap);
  }
  remove() {
    send({messageType: 4, carId: this.id});
    this.id = null;
  }
}

// Convenience: whole-run equivalent of sim.run for parity tests.
export function runAll(trackName, controls) {
  const car = new Car(trackName);
  const raw = new Uint8Array(controls.length * STATE);
  let finishFrame = null;
  try {
    for (let t = 0; t < controls.length; t++) {
      const s = car.step(controls[t]);
      raw.set(s, t * STATE);
      if (finishFrame === null && (s[11] & 2)) finishFrame = s[12] | s[13] << 8 | s[14] << 16;
    }
  } finally {
    car.remove();
  }
  return {finishFrame, raw};
}
