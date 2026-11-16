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
const PAGE = 4096; // WASM page granularity for sparse snapshots
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
  constructor(trackName = 'summer1', {carId} = {}) {
    const {create} = fixture(trackName);
    this.id = carId ?? create.carId;
    send({...create, carId: this.id, carRecording: null});
    this.tick = 0;
    physics.HEAPU8.fill(0, outputPointer, outputPointer + STATE);
    this.refreshView();
  }
  // The heap can grow when a car is created (ArrayBuffer detaches); re-point the live view.
  refreshView() { this.state = new Uint8Array(physics.HEAPU8.buffer, outputPointer, STATE); }
  // Advance one ms with a 5-bit mask (up=1 right=2 down=4 left=8 reset=16). Returns the live state view.
  step(mask) {
    physics._updateCarModel(this.id, mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1, (mask >> 4) & 1, outputPointer);
    this.tick++;
    if (this.state.buffer !== physics.HEAPU8.buffer) this.refreshView();
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
  // Sparse snapshots: copy only the page set S (heap pages driving has written, ~100 of 5952),
  // not 24 MB. Call initPages() on a fresh car (its heap is the pristine one). S grows only
  // through addPages(auditPages()): auditPages lists pages outside S that differ from pristine.
  // A snapshot holds S's first n pages; pages added later restore as pristine. That is exact if
  // the caller audits after every batch of rollouts and, when S grew, adds the pages (same
  // order in every worker) and redoes the batch: then no snapshot ever held a dirty page outside S.
  initPages() {
    this.pristine = Buffer.from(this.snapshot());
    this.pages = [];
    this.inPages = new Uint8Array(this.pristine.length / PAGE);
  }
  // Seeds S from random drives (all 32 masks, ~100 ms holds) so audits rarely grow it later.
  calibratePages({seed = 1, drives = 8, ticks = 12000, every = 100} = {}) {
    if (!this.pristine) this.initPages();
    let s = seed >>> 0 || 1;
    const rand = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 2 ** 32;
    for (let d = 0; d < drives; d++) {
      this.restore(this.pristine);
      let mask = 1;
      for (let t = 1; t <= ticks; t++) {
        if (rand() < 0.01) mask = Math.floor(rand() * 32);
        this.step(mask);
        if (t % every === 0) this.addPages(this.auditPages());
      }
    }
    this.restore(this.pristine);
    return this.pages.length;
  }
  auditPages() {
    const heap = Buffer.from(physics.HEAPU8.buffer, physics.HEAPU8.byteOffset, physics.HEAPU8.length);
    if (heap.length !== this.pristine.length) throw Error('Heap size changed since initPages');
    const found = [], n = this.inPages.length;
    // One memcmp per run of pages outside S; per-page compares only inside a run that differs.
    for (let p = 0; p < n;) {
      if (this.inPages[p]) { p++; continue; }
      let q = p;
      while (q < n && !this.inPages[q]) q++;
      if (heap.compare(this.pristine, p * PAGE, q * PAGE, p * PAGE, q * PAGE) !== 0)
        for (let i = p; i < q; i++) if (heap.compare(this.pristine, i * PAGE, (i + 1) * PAGE, i * PAGE, (i + 1) * PAGE) !== 0) found.push(i);
      p = q;
    }
    return found;
  }
  addPages(list) { for (const i of list) if (!this.inPages[i]) { this.inPages[i] = 1; this.pages.push(i); } }
  // Layout: u32 page count n, then n pages in S order. `into` may be a SharedArrayBuffer view.
  snapshotPages(into) {
    const heap = physics.HEAPU8, n = this.pages.length;
    if (heap.length !== this.pristine.length) throw Error('Heap size changed since initPages');
    into ??= new Uint8Array(4 + n * PAGE);
    if (into.length < 4 + n * PAGE) throw Error(`snapshot buffer holds ${into.length} bytes, needs ${4 + n * PAGE}`);
    new DataView(into.buffer, into.byteOffset, 4).setUint32(0, n, true);
    for (let k = 0; k < n; k++) { const off = this.pages[k] * PAGE; into.set(heap.subarray(off, off + PAGE), 4 + k * PAGE); }
    return into;
  }
  restorePages(snap) {
    const heap = physics.HEAPU8, n = new DataView(snap.buffer, snap.byteOffset, 4).getUint32(0, true);
    if (heap.length !== this.pristine.length) throw Error('Heap size changed since initPages');
    for (let k = 0; k < this.pages.length; k++) {
      const off = this.pages[k] * PAGE;
      heap.set(k < n ? snap.subarray(4 + k * PAGE, 4 + (k + 1) * PAGE) : this.pristine.subarray(off, off + PAGE), off);
    }
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
