// Whole-lap local search on Reflex-1's own line: random edits to the runs of constant mask
// (shift a boundary, change a run's action, insert a pulse, delete a short run), scored only by
// the sim's finishFrame. Workers keep sparse snapshots of the current line every --every ms, so
// a candidate replays only from its first changed tick and stops once it cannot beat the best.
// Input is an agent line (learn/), never human data (LOOP.md isolation rule).
//
//   node learn/refine.mjs --in learn/best.json --out learn/refined.json [--threads 6] [--budget 600]
//        [--seed 1] [--every 50] [--sideways 0.0]
//   node learn/refine.mjs --selftest     # prefix-snapshot evals == fresh sim runs
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';

const ACTIONS = [1, 9, 3, 0, 8, 2, 4, 12, 6]; // up, up+L, up+R, coast, L, R, brake, brake+L, brake+R
const PAD = 3000, BATCH = 8;

function arg(name, def) { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; }

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------- worker: snapshots + rollouts
if (!isMainThread) {
  const {Car} = await import('../ctrl/sim.mjs');
  const {every} = workerData;
  const car = new Car('summer1');
  car.calibratePages();
  let base = workerData.base, fin = null;
  const snaps = []; // snaps[k]: base state after k*every steps, up to the base's finish
  const take = k => {
    if (!snaps[k] || snaps[k].length < 4 + car.pages.length * 4096) snaps[k] = new Uint8Array(4 + (car.pages.length + 64) * 4096);
    car.snapshotPages(snaps[k]);
  };
  const build = k => {
    if (k === 0) { car.restore(car.pristine); take(0); } else car.restorePages(snaps[k]);
    snaps.length = k + 1;
    let t = k * every;
    fin = null;
    while (t < base.length) {
      const s = car.step(base[t++]);
      if (s[11] & 2) { fin = car.finishFrame; break; }
      if (t % every === 0) take(t / every);
    }
    return {found: car.auditPages(), fin};
  };
  parentPort.on('message', m => {
    if (m.type === 'pages') { car.addPages(m.list); parentPort.postMessage(m.rebuild ? build(0) : {found: []}); return; }
    if (m.type === 'base') { base = m.base; parentPort.postMessage(build(Math.min(Math.floor(m.f / every), snaps.length - 1))); return; }
    // eval: one candidate, replayed from the last snapshot at or before its first change.
    // Audit after every rollout: then each rollout starts with every page outside S pristine,
    // so its result is exact, and the pages it dirtied join S (restored as pristine) before the next.
    const k = Math.min(Math.floor(m.f / every), snaps.length - 1);
    car.restorePages(snaps[k]);
    let t = k * every, ff = null;
    const end = Math.min(m.c.length, m.limit); // finishFrame equals steps taken, so past limit nothing can win
    while (t < end) { const s = car.step(m.c[t++]); if (s[11] & 2) { ff = car.finishFrame; break; } }
    parentPort.postMessage({ff, ticks: t - k * every, found: car.auditPages()});
  });
  parentPort.postMessage(build(0));
} else if (process.argv[1] === new URL(import.meta.url).pathname) {
  const selftest = process.argv.includes('--selftest');
  const inPath = arg('--in', 'learn/best.json');
  const out = arg('--out', 'learn/refined.json');
  const threads = selftest ? 2 : Number(arg('--threads', 6));
  const budget = Number(arg('--budget', 600)) * 1000;
  const every = Number(arg('--every', 50));
  const sideways = Number(arg('--sideways', 0));
  const rand = rng(Number(arg('--seed', 1)));
  const t0 = performance.now();

  const input = JSON.parse(fs.readFileSync(inPath, 'utf8')).controls;
  let base = new Uint8Array(input.length + PAD).fill(input.at(-1));
  base.set(input);
  const L = base.length;

  const ask = (w, msg) => new Promise(res => { w.once('message', res); w.postMessage(msg); });
  const workers = Array.from({length: threads}, () => new Worker(new URL(import.meta.url), {workerData: {every, base}}));
  const ready = workers.map(w => new Promise(res => w.once('message', res)));
  // Exactness contract (ctrl/sim.mjs): pages any worker found dirty outside S go to every worker
  // in one sorted order. After a base build all snapshots are retaken; after a rollout none need
  // to be, since the new pages were pristine at every base snapshot (the base build audited clean).
  let evals = 0, ticks = 0, grows = 0;
  async function grow(found, rebuild) {
    for (;;) {
      const list = [...new Set(found)].sort((a, b) => a - b);
      if (!list.length) return;
      grows++;
      found = (await Promise.all(workers.map(w => ask(w, {type: 'pages', list, rebuild})))).flatMap(r => r.found);
    }
  }
  // Workers pull candidates one at a time (rollout lengths vary a lot); when a rollout finds new
  // pages, lanes drain, S grows in every worker, then the rest continue.
  async function evaluate(cands, limit) {
    const ff = new Array(cands.length);
    let next = 0, found = [];
    const lane = async w => {
      while (next < cands.length && !found.length) {
        const i = next++, r = await ask(w, {type: 'eval', c: cands[i].c, f: cands[i].f, limit});
        ff[i] = r.ff, ticks += r.ticks, found.push(...r.found);
      }
    };
    for (;;) {
      await Promise.all(workers.map(lane));
      if (!found.length) break;
      await grow(found, false);
      found = [];
    }
    evals += cands.length;
    return ff;
  }
  async function setBase(cand) {
    base = cand.c;
    const replies = await Promise.all(workers.map(w => ask(w, {type: 'base', base, f: cand.f})));
    await grow(replies.flatMap(r => r.found), true);
    return replies[0].fin;
  }

  // Runs of constant mask in c, as [start, end, mask].
  const runsOf = c => { const r = []; for (let i = 0; i < c.length; i++) if (i === 0 || c[i] !== c[i - 1]) { if (r.length) r.at(-1)[1] = i; r.push([i, c.length, c[i]]); } return r; };
  const geom = (mean, cap) => Math.min(cap, 1 + Math.floor(Math.log(1 - rand()) / Math.log(1 - 1 / mean)));
  const pick = a => a[Math.floor(rand() * a.length)];
  const other = m => { let a; do a = pick(ACTIONS); while (a === m); return a; };
  // One random edit of c in place, only on runs that start before the base finish tick.
  function move(c, finish) {
    const runs = runsOf(c).filter(r => r[0] < finish);
    const u = rand();
    if (u < 0.4 && runs.length > 1) { // shift a boundary
      const j = 1 + Math.floor(rand() * (runs.length - 1)), [s, e, m] = runs[j], [ps, , pm] = runs[j - 1];
      const d = geom(4, 500);
      if (rand() < 0.5) c.fill(pm, s, Math.min(s + d, e - 1));
      else c.fill(m, Math.max(s - d, ps + 1), s);
    } else if (u < 0.6) { // change a run's action
      const [s, e, m] = pick(runs);
      c.fill(other(m), s, e);
    } else if (u < 0.9) { // insert a pulse
      const [s, e, m] = pick(runs);
      const p = s + Math.floor(rand() * (Math.min(e, finish) - s));
      c.fill(other(m), p, Math.min(p + geom(25, 200), L));
    } else if (runs.length > 1) { // delete a short run into a neighbour
      const short = runs.map((r, j) => j).filter(j => runs[j][1] - runs[j][0] <= 200);
      const all = runsOf(c);
      const j = short.length ? pick(short) : Math.floor(rand() * runs.length);
      const [s, e] = all[j];
      c.fill(j > 0 && (j === all.length - 1 || rand() < 0.5) ? all[j - 1][2] : all[j + 1][2], s, e);
    }
  }
  function candidate(finish) {
    for (;;) {
      const c = base.slice();
      move(c, finish);
      if (rand() < 0.2) move(c, finish);
      let f = 0;
      while (f < L && c[f] === base[f]) f++;
      if (f < L) return {c, f};
    }
  }

  const started = await Promise.all(ready);
  await grow(started.flatMap(r => r.found), true);
  const fin0 = started[0].fin;
  const {run} = await import('../sim/sim.js');

  if (selftest) {
    let fin = fin0, n = 0;
    for (let round = 0; round < 2; round++) {
      // Round 0: no early stop. Round 1: around a new base (snapshots rebuilt from its first
      // change), early stop at the base finish.
      const limit = round === 0 ? Infinity : (fin ?? L);
      const cands = Array.from({length: 25}, () => candidate(fin ?? L));
      const ff = await evaluate(cands, limit);
      for (let i = 0; i < cands.length; i++, n++) {
        const ref = run('summer1', cands[i].c).finishFrame;
        assert.equal(ff[i], ref !== null && ref <= limit ? ref : null, `candidate ${n} (first change ${cands[i].f})`);
      }
      if (round === 0) {
        const next = cands[Math.max(0, ff.findIndex(x => x !== null))];
        fin = await setBase(next);
        assert.equal(fin, run('summer1', base).finishFrame, 'rebuilt base finish');
      }
    }
    console.log(`selftest ok: ${n} candidates match fresh runs (base finish ${fin0}, page set grew ${grows}x)`);
    for (const w of workers) w.terminate();
  } else {
    let best = fin0 ?? Infinity, accepts = 0;
    console.log(`t=${((performance.now() - t0) / 1000).toFixed(1)} start finish=${best} threads=${threads} every=${every}`);
    const save = () => {
      fs.writeFileSync(`${out}.tmp`, JSON.stringify({controls: Array.from(base.subarray(0, best)), finishFrame: best, from: inPath}));
      fs.renameSync(`${out}.tmp`, out); // a kill mid-write never leaves a broken --out
    };
    while (performance.now() - t0 < budget) {
      const cands = Array.from({length: BATCH * threads}, () => candidate(Number.isFinite(best) ? best : L));
      const ff = await evaluate(cands, Number.isFinite(best) ? (sideways > 0 ? best : best - 1) : L);
      let win = -1;
      for (let i = 0; i < ff.length; i++) if (ff[i] !== null && ff[i] < best && (win < 0 || ff[i] < ff[win])) win = i;
      if (win < 0 && sideways > 0) win = ff.findIndex(x => x === best && rand() < sideways);
      if (win < 0) continue;
      const improved = ff[win] < best;
      const fin = await setBase(cands[win]);
      assert.equal(fin, ff[win], 'rebuilt base finish');
      accepts++;
      if (!improved) continue;
      best = fin;
      save();
      console.log(`t=${((performance.now() - t0) / 1000).toFixed(1)} finish=${best} evals=${evals}`);
    }
    for (const w of workers) w.terminate();
    const sec = (performance.now() - t0) / 1000;
    console.log(`done: finish ${fin0} -> ${best}, evals=${evals} (${(evals / sec).toFixed(1)}/s), ${(ticks / sec / threads).toFixed(0)} ticks/s/thread, accepts=${accepts}, page set grew ${grows}x, ${sec.toFixed(0)} s`);
    if (best < (fin0 ?? Infinity)) {
      assert.equal(run('summer1', base).finishFrame, best, 'fresh sim disagrees with refined finish');
      console.log(`fresh run ok: ${out} finish ${best}`);
    }
  }
}
