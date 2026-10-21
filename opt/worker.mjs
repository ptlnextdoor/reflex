// worker_threads evaluator: one physics engine per worker, base line with sparse heap snapshots
// every `every` ticks, candidate evaluation re-simulates from the last snapshot before the first
// differing tick and stops at the horizon or finish.
import {parentPort, workerData} from 'node:worker_threads';
import {Car, STATE_BYTES} from './engine.mjs';

const PAGE = 4096, CHUNK = 65536;
const car = new Car(workerData?.track ?? 'summer1');
const pristine = car.heap.slice();
const pristineBuf = Buffer.from(pristine.buffer, 0, pristine.length);
let base = null; // {controls, every, snaps: [{tick, pages: [[offset, bytes]]}], finishFrame}

function sparseSnapshot() {
  const heap = car.heap;
  if (heap.length !== pristine.length) throw Error(`heap grew to ${heap.length}; snapshot invalid`);
  const hb = Buffer.from(heap.buffer, 0, heap.length);
  const pages = [];
  for (let c = 0; c < hb.length; c += CHUNK) {
    const cEnd = Math.min(c + CHUNK, hb.length);
    if (hb.compare(pristineBuf, c, cEnd, c, cEnd) === 0) continue;
    for (let p = c; p < cEnd; p += PAGE) {
      const pEnd = Math.min(p + PAGE, cEnd);
      if (hb.compare(pristineBuf, p, pEnd, p, pEnd) !== 0) pages.push([p, heap.slice(p, pEnd)]);
    }
  }
  return pages;
}
function restore(pages) {
  const heap = car.heap;
  heap.set(pristine);
  for (const [offset, bytes] of pages) heap.set(bytes, offset);
  car.refresh();
  car.tick = 0;
}

const view = () => new DataView(car.state.buffer, car.state.byteOffset, STATE_BYTES);
function readPos() {
  const v = view();
  const o = (car.finished() ? 15 : 12) + 2;
  return [v.getFloat32(o, true), v.getFloat32(o + 4, true), v.getFloat32(o + 8, true)];
}

// Step controls[from..horizon) from the current heap (which must represent tick `from`).
// Returns finish info plus per-checkpoint ticks and the terminal position.
function simulate(controls, from, horizon, prevPos, cpTicks) {
  let finishFrame = null, posBefore = prevPos, pos = prevPos;
  let cp = car.nextCheckpoint();
  const last = controls.length ? controls[controls.length - 1] : 0;
  let t = from;
  for (; t < horizon; t++) {
    car.step(t < controls.length ? controls[t] : last);
    const ncp = car.nextCheckpoint();
    if (ncp !== cp) { cpTicks.push([ncp, t + 1]); cp = ncp; }
    if (car.finished()) { finishFrame = t + 1; posBefore = pos; pos = readPos(); break; }
    if (t === horizon - 2) pos = readPos();
  }
  if (finishFrame === null) { posBefore = pos; pos = readPos(); }
  return {finishFrame, pos, posBefore, speedKmh: view().getFloat32(7, true), ticks: t - from, nextCheckpoint: cp};
}

function setBase(controls, every) {
  restore([]);
  const snaps = [{tick: 0, pages: []}];
  const cpTicks = [];
  let cp = 0, finishFrame = null;
  for (let t = 0; t < controls.length; t++) {
    car.step(controls[t]);
    const ncp = car.nextCheckpoint();
    if (ncp !== cp) { cpTicks.push([ncp, t + 1]); cp = ncp; }
    if (finishFrame === null && car.finished()) finishFrame = t + 1;
    if ((t + 1) % every === 0 && finishFrame === null) snaps.push({tick: t + 1, pages: sparseSnapshot()});
  }
  base = {controls: Uint8Array.from(controls), every, snaps, finishFrame, cpTicks};
  return {finishFrame, cpTicks, snapshots: snaps.length, pages: snaps.reduce((a, s) => a + s.pages.length, 0)};
}

function evaluate(controls, horizon) {
  if (!base) throw Error('no base');
  const b = base.controls;
  const n = Math.min(controls.length, b.length);
  let d = 0;
  while (d < n && controls[d] === b[d]) d++;
  if (d === n && controls.length !== b.length) d = n;
  // pick last snapshot with tick <= d
  let s = base.snaps[0];
  for (const snap of base.snaps) if (snap.tick <= d) s = snap; else break;
  restore(s.pages);
  const cpTicks = base.cpTicks.filter(([, tick]) => tick <= s.tick);
  // position at snapshot is unknown without decoding; use sentinel and let simulate overwrite
  const r = simulate(controls, s.tick, horizon, null, cpTicks);
  return {...r, cpTicks, from: s.tick, firstDiff: d};
}

parentPort.on('message', msg => {
  try {
    if (msg.type === 'base') parentPort.postMessage({type: 'base', id: msg.id, ...setBase(msg.controls, msg.every ?? 250)});
    else if (msg.type === 'eval') parentPort.postMessage({type: 'eval', id: msg.id, ...evaluate(msg.controls, msg.horizon)});
    else if (msg.type === 'ping') parentPort.postMessage({type: 'pong', id: msg.id});
  } catch (error) {
    parentPort.postMessage({type: 'error', id: msg.id, error: String(error?.stack ?? error)});
  }
});
parentPort.postMessage({type: 'ready'});
