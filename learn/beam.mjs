// Reflex-1 learns a lap from scratch by time-synchronous beam search on the exact sim. Inputs
// are only the track gates (learn/track.json, via learn/route.mjs) and the physics
// (ctrl/sim.mjs). No human run or external data of any kind (LOOP.md isolation rule).
//
// Every beam node keeps an exact sparse heap snapshot (Car.snapshotPages) in a shared buffer,
// so each generation expands every node with the 9 actions held for DT ms, scores the children
// by route progress, dedupes identical states, keeps the best per position/heading bin, and
// re-simulates the survivors into the next generation's snapshot slots.
//
//   node learn/beam.mjs [--beam 600] [--dt 50] [--threads 6] [--look 1.5] [--bin 0.5] [--yawbin 10]
//                       [--vbin 5] [--per-bin 1] [--max 40000] [--out learn/beam.json] [--log runs/beam/log.csv]
//   node learn/beam.mjs --selftest   # 20 generations, then a fresh replay must hit the node's xyz exactly
import fs from 'node:fs';
import path from 'node:path';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';

const ACTIONS = [1, 9, 3, 0, 8, 2, 4, 12, 6]; // up, up+L, up+R, coast, L, R, brake, brake+L, brake+R
const BIG = 1e4;
const PAGE = 4096, MAXPAGES = 256, SLOT = 4 + MAXPAGES * PAGE;
// Child record fields (Float64): cp, x, y, z, speed, grounded, yaw, finishFrame, tick, hash, score.
const K = 11, CP = 0, X = 1, Y = 2, Z = 3, SPEED = 4, GROUND = 5, YAW = 6, FIN = 7, TICK = 8, HASH = 9, SCORE = 10;

// 53-bit hash (cyrb53) of state bytes [0, 12) and [o, o + 30): header, cp, xyz, quaternion.
function stateHash(s, o) {
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < o + 30; i = i === 11 ? o : i + 1) { h1 = Math.imul(h1 ^ s[i], 2654435761); h2 = Math.imul(h2 ^ s[i], 1597334677); }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

// ---------------------------------------------------------------- worker: expand and commit nodes
if (!isMainThread) {
  const {Car} = await import('../ctrl/sim.mjs');
  const {buildFields} = await import('./route.mjs');
  const {createHash} = await import('node:crypto');
  const {progress} = buildFields();
  const {sab, look, dt, calibrate} = workerData;
  const car = new Car('summer1');
  if (calibrate) car.calibratePages(); else car.initPages();
  const slot = i => new Uint8Array(sab, i * SLOT, SLOT);
  const hold = a => { let s; for (let t = 0; t < dt; t++) s = car.step(ACTIONS[a]); return s; };
  const record = (out, at, s, fin, tick) => {
    const v = new DataView(s.buffer, s.byteOffset, 227), o = fin === null ? 12 : 15;
    const cp = v.getUint16(o, true), x = v.getFloat32(o + 2, true), y = v.getFloat32(o + 6, true), z = v.getFloat32(o + 10, true);
    const qx = v.getFloat32(o + 14, true), qy = v.getFloat32(o + 18, true), qz = v.getFloat32(o + 22, true), qw = v.getFloat32(o + 26, true);
    const speed = v.getFloat32(7, true);
    out[at + CP] = cp; out[at + X] = x; out[at + Y] = y; out[at + Z] = z; out[at + SPEED] = speed;
    out[at + GROUND] = (s[11] >> 3) & 15;
    out[at + YAW] = Math.atan2(2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy));
    out[at + FIN] = fin ?? -1; out[at + TICK] = tick; out[at + HASH] = stateHash(s, o);
    out[at + SCORE] = fin === null ? BIG * cp - progress(cp, x, y, z) + look * speed / 3.6 : Infinity;
  };
  // Cross-worker restores assume pages outside the set are pristine and equal in every worker.
  // (Pristine differs across workers in 2 bytes of page 21, which calibration puts in the set.)
  const outsideDigest = () => {
    const h = createHash('sha1');
    for (let p = 0; p < car.inPages.length; p++) if (!car.inPages[p]) h.update(car.pristine.subarray(p * PAGE, (p + 1) * PAGE));
    return h.digest('hex');
  };
  // A rollout can dirty a page outside the set and crash a later rollout in this worker; the
  // audit then finds the page and the main thread adds it, replaces this worker, and redoes the batch.
  const guarded = work => {
    try { work(); } catch (e) { parentPort.postMessage({redo: true, error: String(e?.stack ?? e), pages: car.auditPages()}); }
  };
  parentPort.on('message', m => {
    if (m.type === 'add') {
      car.addPages(m.pages);
      if (car.pages.length > MAXPAGES) throw Error(`page set grew to ${car.pages.length}, slots hold ${MAXPAGES}`);
      parentPort.postMessage({digest: outsideDigest()});
    } else if (m.type === 'root') {
      car.restore(car.pristine);
      car.snapshotPages(slot(m.slot));
      parentPort.postMessage({pages: car.auditPages()});
    } else if (m.type === 'expand') guarded(() => {
      const out = new Float64Array(m.slots.length * 9 * K);
      for (let i = 0; i < m.slots.length; i++) {
        const snap = slot(m.slots[i]);
        for (let a = 0; a < 9; a++) {
          car.restorePages(snap);
          let s, fin = null, t = 0;
          while (t < dt) { s = car.step(ACTIONS[a]); t++; if (s[11] & 2) { fin = s[12] | s[13] << 8 | s[14] << 16; break; } }
          record(out, (i * 9 + a) * K, s, fin, t);
        }
      }
      parentPort.postMessage({out, pages: car.auditPages()}, [out.buffer]);
    }); else if (m.type === 'commit') guarded(() => {
      // Re-simulate parent+action into the next generation's slot; the state must hash as in expand.
      let bad = 0;
      for (let j = 0; j < m.want.length; j++) {
        car.restorePages(slot(m.jobs[3 * j]));
        const s = hold(m.jobs[3 * j + 1]);
        car.snapshotPages(slot(m.jobs[3 * j + 2]));
        if (stateHash(s, 12) !== m.want[j]) bad++;
      }
      parentPort.postMessage({bad, pages: car.auditPages()});
    });
  });
  parentPort.postMessage({pages: car.pages});
} else if (process.argv[1] === new URL(import.meta.url).pathname) {
  const arg = (name, def) => { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; };
  const selftest = process.argv.includes('--selftest');
  const BEAM = Number(arg('--beam', selftest ? 8 : 600)), DT = Number(arg('--dt', 50));
  const THREADS = Number(arg('--threads', selftest ? 2 : 6)), LOOK = Number(arg('--look', 1.5));
  const BIN = Number(arg('--bin', 0.5)), YAWBIN = Number(arg('--yawbin', 10)), VBIN = Number(arg('--vbin', 5));
  const PERBIN = Number(arg('--per-bin', 1)), MAXMS = Number(arg('--max', 40000));
  const GENS = selftest ? 20 : Infinity;
  const out = arg('--out', new URL('./beam.json', import.meta.url).pathname);
  const logPath = arg('--log', selftest ? '/dev/null' : new URL('../runs/beam/log.csv', import.meta.url).pathname);
  if (logPath !== '/dev/null') fs.mkdirSync(path.dirname(logPath), {recursive: true});
  fs.writeFileSync(logPath, 'gen,sim_ms,wall_s,best_cp,best_score,best_speed,unique,kept,pages\n');

  const t0 = performance.now();
  const sab = new SharedArrayBuffer(2 * BEAM * SLOT); // double-buffered generations; untouched slot tails stay uncommitted
  const workers = [];
  const spawn = (k, calibrate) => {
    const w = new Worker(new URL(import.meta.url), {workerData: {sab, look: LOOK, dt: DT, calibrate}});
    w.on('error', e => { console.error(e); process.exit(2); });
    workers[k] = w;
    return new Promise(res => w.once('message', res));
  };
  const ask = (w, msg) => new Promise(res => { w.once('message', res); w.postMessage(msg); });
  const init = await Promise.all(Array.from({length: THREADS}, (_, k) => spawn(k, k === 0)));
  // Every worker must hold the same page list, in this order, with equal pristine pages outside it.
  const pageList = [...init[0].pages];
  const syncPages = async () => {
    const r = await Promise.all(workers.map(w => ask(w, {type: 'add', pages: pageList})));
    if (new Set(r.map(x => x.digest)).size !== 1) throw Error('workers differ on pristine pages outside the set');
  };
  await syncPages();
  console.log(`${THREADS} workers ready, ${pageList.length} pages, ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  // One batch over n items split in contiguous chunks; after it every worker has audited. If any
  // found pages outside the set, add them everywhere (sorted, same order) and redo the batch.
  // That also covers commit hash mismatches and crashes caused by those pages. A crashed worker
  // is replaced: a WASM trap can leave non-memory state (the stack pointer global) unrestored.
  async function batch(n, msgFor) {
    for (;;) {
      const per = Math.ceil(n / THREADS), res = [];
      await Promise.all(workers.map((w, k) => {
        const lo = k * per, hi = Math.min(n, lo + per);
        return lo < hi ? ask(w, msgFor(lo, hi)).then(r => { res[k] = r; }) : null;
      }));
      const found = [...new Set(res.flatMap(r => r?.pages ?? []))].sort((a, b) => a - b);
      const crashed = res.flatMap((r, k) => r?.redo ? [k] : []);
      if (!found.length) {
        if (crashed.length) throw Error(`worker ${crashed[0]} crashed with no page outside the set:\n${res[crashed[0]].error}`);
        return res.filter(Boolean);
      }
      pageList.push(...found);
      await Promise.all(crashed.map(k => { workers[k].terminate(); return spawn(k, false); }));
      await syncPages();
      console.log(`  page set grew by ${found.length} to ${pageList.length}${crashed.length ? `, replaced crashed workers ${crashed}` : ''}; redoing batch`);
    }
  }

  // Tree: parents[g][j] / acts[g][j] for node j of generation g (g >= 1); generation 0 is the start.
  const parents = [null], acts = [null];
  const masksTo = (g, j) => {
    const seq = [];
    for (; g > 0; g--) { seq.push(acts[g][j]); j = parents[g][j]; }
    return seq.reverse().flatMap(a => Array(DT).fill(ACTIONS[a]));
  };

  await batch(1, () => ({type: 'root', slot: 0}));
  let n = 1, gen = 0, last = null, finish = null;
  for (;;) {
    const cur = (gen % 2) * BEAM, next = ((gen + 1) % 2) * BEAM;
    const res = await batch(n, (lo, hi) => ({type: 'expand', slots: Int32Array.from({length: hi - lo}, (_, i) => cur + lo + i)}));
    const ch = new Float64Array(n * 9 * K);
    let at = 0;
    for (const r of res) { ch.set(r.out, at); at += r.out.length; }
    const N = n * 9;
    for (let c = 0; c < N; c++) {
      const f = ch[c * K + FIN];
      if (f >= 0 && (!finish || f < finish.frame)) finish = {frame: f, c, tick: ch[c * K + TICK]};
    }
    if (finish) break;
    const seen = new Set(), cand = [];
    for (let c = 0; c < N; c++) {
      if (ch[c * K + Y] < -5 || seen.has(ch[c * K + HASH])) continue;
      seen.add(ch[c * K + HASH]);
      cand.push(c);
    }
    cand.sort((a, b) => ch[b * K + SCORE] - ch[a * K + SCORE]);
    const binCount = new Map(), sel = [];
    for (const c of cand) {
      const b = c * K, yaw = Math.floor((ch[b + YAW] * 180 / Math.PI + 180) / YAWBIN), v = Math.floor(ch[b + SPEED] / VBIN);
      const key = `${ch[b + CP]},${Math.floor(ch[b + X] / BIN)},${Math.floor(ch[b + Y] / BIN)},${Math.floor(ch[b + Z] / BIN)},${yaw},${v}`;
      const k = binCount.get(key) ?? 0;
      if (k >= PERBIN) continue;
      binCount.set(key, k + 1);
      sel.push(c);
      if (sel.length === BEAM) break;
    }
    if (!sel.length) throw Error(`generation ${gen + 1}: every child fell off`);
    const commits = await batch(sel.length, (lo, hi) => {
      const jobs = new Int32Array(3 * (hi - lo)), want = new Float64Array(hi - lo);
      for (let j = lo; j < hi; j++) {
        const c = sel[j], i = 3 * (j - lo);
        jobs[i] = cur + Math.floor(c / 9); jobs[i + 1] = c % 9; jobs[i + 2] = next + j;
        want[j - lo] = ch[c * K + HASH];
      }
      return {type: 'commit', jobs, want};
    });
    const bad = commits.reduce((s, r) => s + r.bad, 0);
    if (bad) throw Error(`generation ${gen + 1}: ${bad} re-simulated survivors differ from their expansion`);
    gen++;
    parents.push(Int32Array.from(sel, c => Math.floor(c / 9)));
    acts.push(Uint8Array.from(sel, c => c % 9));
    n = sel.length;
    last = ch.slice(sel[0] * K, sel[0] * K + K);
    const wall = (performance.now() - t0) / 1000;
    const row = [gen, gen * DT, wall.toFixed(1), last[CP], last[SCORE].toFixed(1), last[SPEED].toFixed(1), seen.size, n, pageList.length];
    fs.appendFileSync(logPath, row.join(',') + '\n');
    console.log(`gen ${gen} sim ${gen * DT} ms wall ${wall.toFixed(1)} s cp ${last[CP]} score ${last[SCORE].toFixed(1)} speed ${last[SPEED].toFixed(0)} pos ${last[X].toFixed(0)},${last[Y].toFixed(0)},${last[Z].toFixed(0)} unique ${seen.size} beam ${n} pages ${pageList.length}`);
    if (gen >= GENS) break;
    if (gen * DT >= MAXMS) break;
  }
  for (const w of workers) w.terminate();
  const {run} = await import('../sim/sim.js');
  const wall_s = +((performance.now() - t0) / 1000).toFixed(1);

  if (selftest) {
    const controls = masksTo(gen, 0);
    const p = run('summer1', Uint8Array.from(controls)).states.at(-1).position;
    console.log(`replay ${controls.length} ms: node ${last[X]},${last[Y]},${last[Z]} fresh ${p.x},${p.y},${p.z}`);
    if (controls.length !== gen * DT || p.x !== last[X] || p.y !== last[Y] || p.z !== last[Z]) { console.log('selftest FAILED'); process.exit(1); }
    console.log('selftest ok');
  } else if (finish) {
    const c = finish.c, node = Math.floor(c / 9);
    const controls = masksTo(gen, node).concat(Array(finish.tick).fill(ACTIONS[c % 9]));
    fs.writeFileSync(out, JSON.stringify({controls, finishFrame: finish.frame, beam: BEAM, dt: DT, wall_s}));
    const pad = controls.concat(Array(3000).fill(controls.at(-1)));
    const fresh = run('summer1', Uint8Array.from(pad)).finishFrame;
    console.log(`wrote ${out}: ${controls.length} controls, beam finishFrame ${finish.frame}, fresh replay finishFrame ${fresh}, ${wall_s} s wall`);
    if (fresh !== finish.frame) { console.log('fresh replay finishFrame differs'); process.exit(1); }
  } else {
    console.log(`no finish within ${MAXMS} ms of sim time (${wall_s} s wall)`);
    process.exit(1);
  }
}
