// Stepping engine for search: same loader as sim/sim.js (which keeps physics private), plus
// per-tick stepping, heap snapshot/restore for prefix reuse, and early exit.
// Root of trust stays sim.run / verifyRecording. opt/test_engine.mjs proves byte parity.
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

export const vendor = fileURLToPath(new URL('../vendor/polytrack/0.6.3/', import.meta.url));
export const STATE_BYTES = 227;
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

export class Car {
  constructor(trackName = 'summer1') {
    const {create} = fixture(trackName);
    send({...create, carRecording: null});
    this.id = create.carId;
    physics.HEAPU8.fill(0, outputPointer, outputPointer + STATE_BYTES);
    this.state = new Uint8Array(physics.HEAPU8.buffer, outputPointer, STATE_BYTES);
    this.tick = 0;
  }
  get heap() { return physics.HEAPU8; }
  get heapBuffer() { return physics.HEAPU8.buffer; }
  refresh() { // heap may have been reallocated on growth
    if (this.state.buffer !== physics.HEAPU8.buffer) this.state = new Uint8Array(physics.HEAPU8.buffer, outputPointer, STATE_BYTES);
  }
  step(mask) {
    physics._updateCarModel(this.id, mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1, (mask >> 4) & 1, outputPointer);
    this.tick++;
  }
  finished() { return (this.state[11] & 2) !== 0; }
  finishFrame() { return this.finished() ? u24(this.state, 12) : null; }
  nextCheckpoint() { return this.state[this.finished() ? 15 : 12] | this.state[this.finished() ? 16 : 13] << 8; }
  speedKmh() { return new DataView(this.state.buffer, this.state.byteOffset, STATE_BYTES).getFloat32(7, true); }
  position() {
    const v = new DataView(this.state.buffer, this.state.byteOffset, STATE_BYTES);
    const o = (this.finished() ? 15 : 12) + 2;
    return {x: v.getFloat32(o, true), y: v.getFloat32(o + 4, true), z: v.getFloat32(o + 8, true)};
  }
  destroy() { send({messageType: 4, carId: this.id}); }
}
export const heapSize = () => physics.HEAPU8.length;
