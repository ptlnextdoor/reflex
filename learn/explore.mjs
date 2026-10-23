// Gate-by-gate shortcut search with Go-Explore: grow a frontier of reached states (cells keyed
// by coarse position + checkpoint), keep the earliest arrival per cell, and expand random
// 300-1200 ms action bursts from them. Finds any physically reachable path, including wall
// clips, ramp skips and drops, because nothing constrains it to the road graph.
//   node learn/explore.mjs --from <line.json> --at <tick> --target-cp <n> --minutes 20 [--seed 1]
// Writes runs/top/explore_<at>_<cp>.json: best line prefix + suffix reaching checkpoint index
// >= target-cp, and the tick it got there. Compare against the WR's tick for that checkpoint.
import fs from 'node:fs';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';
import os from 'node:os';

const {Car, decodeState} = await import('../ctrl/sim.mjs');
const {decodeRecording} = await import('../sim/sim.js');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i < 0 ? d : process.argv[i + 1]; };
const A = [1, 3, 9, 0, 5, 7, 13, 4, 2, 8, 11, 15]; // W WD WA - WS WDS WAS S D A WDA(=W) WDSA

function loadLine(p) {
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  if (j.recording) return Array.from(decodeRecording(j.recording, j.frames));
  return j.controls;
}

if (!isMainThread) {
  const {prefix, target} = workerData;
  const car = new Car('summer1');
  for (const m of prefix) car.step(m);
  const base = car.snapshot();
  let rs = workerData.seed >>> 0;
  const rnd = () => ((rs = (rs * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  parentPort.on('message', ({cells}) => {
    // cells: [{id, ctrl:[...]}]; expand each by a random burst, report new states
    const out = [];
    for (const c of cells) {
      car.restore(base);
      for (const m of c.ctrl) car.step(m);
      const add = [];
      let segs = 1 + Math.floor(rnd() * 4);
      let s;
      for (let k = 0; k < segs; k++) {
        const m = A[Math.floor(rnd() * A.length)], len = 100 + Math.floor(rnd() * 400);
        for (let i = 0; i < len; i++) { s = decodeState(car.step(m)); add.push(m); if (s.nextCheckpointIndex >= target || car.finished) break; }
        if (s.nextCheckpointIndex >= target || car.finished) break;
      }
      out.push({from: c.id, add, cp: s.nextCheckpointIndex, x: s.position.x, y: s.position.y, z: s.position.z, v: s.speedKmh, fin: car.finishFrame});
    }
    parentPort.postMessage(out);
  });
} else {
  const line = loadLine(arg('--from'));
  const AT = Number(arg('--at')), TARGET = Number(arg('--target-cp')), MIN = Number(arg('--minutes', 20));
  const prefix = line.slice(0, AT);
  const threads = Number(arg('--threads', Math.max(1, os.availableParallelism() - 1)));
  const seed = Number(arg('--seed', 1));
  const GOAL = arg('--goal') ? arg('--goal').split(',').map(Number) : null;
  // --via: intermediate gate (cp TARGET-1) to reach first; --tw: metres charged per ms (0.08 = full speed),
  // so a slow rail clip scores worse than a fast one.
  const VIA = arg('--via') ? arg('--via').split(',').map(Number) : null;
  const TW = Number(arg('--tw', 0.02));
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  let bestD = 1e9;
  // WR reference tick for the target checkpoint (to know what "faster" means)
  const ref = (() => { const car = new Car('summer1'); for (let t = 0; t < line.length; t++) { const s = decodeState(car.step(line[t])); if (s.nextCheckpointIndex >= TARGET || car.finished) return t; } return null; })();
  const cells = new Map(); // key -> {id, ctrl, t, v}
  let nid = 0; const byId = new Map();
  const key = (cp, x, y, z, v) => `${cp},${Math.round(x / 6)},${Math.round(y / 4)},${Math.round(z / 6)},${Math.round(v / 60)}`;
  const root = {id: nid++, ctrl: [], t: AT}; byId.set(root.id, root); cells.set('root', root);
  const workers = Array.from({length: threads}, (_, i) => new Worker(new URL(import.meta.url), {workerData: {prefix, target: TARGET, seed: seed * 1000 + i}}));
  let best = null; const t0 = Date.now(); let gen = 0;
  while ((Date.now() - t0) / 60000 < MIN) {
    const pool = [...cells.values()];
    // select: prefer cells with fewer visits and earlier time-per-progress (uniform + recency mix)
    // --goal x,y,z: tournament-pick cells closer to the goal (e.g. the next gate) so the
    // search pushes over guardrails toward it instead of wandering the whole loop.
    const rand = () => pool[Math.floor(Math.random() * pool.length)];
    const pick = GOAL ? () => { let b = rand(); for (let k = 0; k < 3; k++) { const c = rand(); if ((c.d ?? 1e9) < (b.d ?? 1e9)) b = c; } return b; } : rand;
    const res = await Promise.all(workers.map(w => new Promise(r => { w.once('message', r); w.postMessage({cells: Array.from({length: 24}, () => { const c = pick(); return {id: c.id, ctrl: c.ctrl}; })}); })));
    for (const batch of res) for (const o of batch) {
      const parent = byId.get(o.from);
      const ctrl = parent.ctrl.concat(o.add), t = AT + ctrl.length;
      if (o.cp >= TARGET || o.fin) { if (!best || t < best.t) { best = {t, ctrl}; console.log(`REACHED cp${TARGET} at tick ${t} (ref ${ref}, ${ref - t > 0 ? 'saves ' + (ref - t) : 'loses ' + (t - ref)} ms) gen ${gen}`); } continue; }
      if (o.y < -5) continue; // fell off the world
      const k = key(o.cp, o.x, o.y, o.z, o.v);
      const prev = cells.get(k);
      const p = [o.x, o.y, o.z];
      const d = !GOAL ? 0 : (VIA && o.cp < TARGET - 1 ? dist(p, VIA) + dist(VIA, GOAL) : dist(p, GOAL)) + TW * (t - AT);
      if (!prev || t < prev.t) { const c = {id: nid++, ctrl, t, d}; byId.set(c.id, c); cells.set(k, c); if (GOAL && d < bestD) { bestD = d; } }
    }
    gen++;
    if (gen % 20 === 0) console.log(`gen ${gen} cells ${cells.size} best ${best ? best.t : '-'} closest ${bestD.toFixed(0)} ref ${ref} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    // checkpoint the best path so far, so a long run can be grafted while it keeps searching
    if (best && gen % 20 === 0) fs.writeFileSync(`runs/top/explore_${AT}_${TARGET}_${seed}.json`, JSON.stringify({at: AT, target: TARGET, ref, best: {t: best.t, controls: prefix.concat(best.ctrl)}, cells: cells.size}));
  }
  for (const w of workers) w.terminate();
  const out = `runs/top/explore_${AT}_${TARGET}_${seed}.json`;
  fs.writeFileSync(out, JSON.stringify({at: AT, target: TARGET, ref, best: best ? {t: best.t, controls: prefix.concat(best.ctrl)} : null, cells: cells.size}));
  console.log(best ? `DONE best cp${TARGET} at ${best.t} vs ref ${ref}` : `DONE no path to cp${TARGET}`, out);
}
