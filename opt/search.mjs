// Toggle-time local search for Summer 1. Representation: sorted toggle list [[tick, bit], ...].
// Phase 1: coordinate descent (every toggle +/-1, +/-2 ms). Phase 2: randomized local search
// with shifts, pulse moves, inserts, deletes and tail shifts. Keeps only improvements.
//
//   node opt/search.mjs --budget 600 --seed 1 [--start opt/best.json] [--out opt/best.json] [--workers 12]
import fs from 'node:fs';
import {decodeRecording, run} from '../sim/sim.js';
import {Pool} from './pool.mjs';
import {togglesFromControls, controlsFromToggles, encodeRecording} from './recording.mjs';

const args = Object.fromEntries(process.argv.slice(2).map((a, i, v) => a.startsWith('--') ? [a.slice(2), v[i + 1]?.startsWith('--') || v[i + 1] === undefined ? true : v[i + 1]] : null).filter(Boolean));
const budgetS = Number(args.budget ?? 300);
const seed0 = Number(args.seed ?? 1);
const out = args.out ?? 'opt/best.json';
const workers = Number(args.workers ?? 12);
const WR = 22123;
const FINISH_X = -297.322; // finish plane x estimated from the WR crossing; sub-ms tie-break only
const vendor = new URL('../vendor/polytrack/0.6.3/', import.meta.url);

let seed = seed0 >>> 0;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const randInt = n => Math.floor(rand() * n);
const geometric = (p = 0.35) => { let k = 1; while (rand() > p && k < 60) k++; return k; };

function loadStart() {
  if (args.start && fs.existsSync(args.start)) {
    const c = JSON.parse(fs.readFileSync(args.start, 'utf8'));
    if (c.recording) return decodeRecording(c.recording, c.frames);
    if (c.controls) return Uint8Array.from(c.controls);
  }
  const rec = JSON.parse(fs.readFileSync(new URL('recording-59372564.json', vendor), 'utf8'))[0];
  return decodeRecording(rec.recording, rec.frames);
}

// Score: finishFrame - 1 + fraction of the last ms needed to reach the finish plane. Lower is better.
function score(r) {
  if (r.finishFrame === null) return Infinity;
  if (!r.posBefore) return r.finishFrame - 0.5;
  const x0 = r.posBefore[0], x1 = r.pos[0];
  let frac = (FINISH_X - x0) / (x1 - x0);
  if (!Number.isFinite(frac)) frac = 0.5;
  return r.finishFrame - 1 + Math.min(1, Math.max(0, frac));
}

const clone = t => t.map(x => [x[0], x[1]]);
function normalize(toggles) {
  // sort, drop duplicates (two toggles of one bit on one tick cancel), clamp tick >= 0
  const t = toggles.filter(x => x[0] >= 0).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const o = [];
  for (const x of t) {
    const last = o[o.length - 1];
    if (last && last[0] === x[0] && last[1] === x[1]) o.pop(); else o.push(x);
  }
  return o;
}
// Pulses: consecutive toggles of the same bit (press, release).
function pulses(t) {
  const byBit = new Map();
  t.forEach((x, i) => { if (!byBit.has(x[1])) byBit.set(x[1], []); byBit.get(x[1]).push(i); });
  const p = [];
  for (const idx of byBit.values()) for (let k = 0; k + 1 < idx.length; k += 2) p.push([idx[k], idx[k + 1]]);
  return p;
}

const MOVES = ['shift1', 'shift', 'pulse', 'width', 'insert', 'delete', 'tail', 'double'];
function mutate(base, kind) {
  const t = clone(base);
  const n = t.length;
  const sgn = () => rand() < 0.5 ? -1 : 1;
  switch (kind) {
    case 'shift1': { const i = 1 + randInt(n - 1); t[i][0] += sgn(); break; }
    case 'shift': { const i = 1 + randInt(n - 1); t[i][0] += sgn() * geometric(0.25); break; }
    case 'pulse': { const p = pulses(t); if (!p.length) return null; const [a, b] = p[randInt(p.length)]; const d = sgn() * geometric(0.3); t[a][0] += d; t[b][0] += d; break; }
    case 'width': { const p = pulses(t); if (!p.length) return null; const [a, b] = p[randInt(p.length)]; const d = sgn() * geometric(0.4); if (rand() < 0.5) t[a][0] += d; else t[b][0] += d; break; }
    case 'insert': { const bit = [1, 3, 2, 0][randInt(4)]; const tick = 150 + randInt(WR - 150); const w = 1 + geometric(0.15); t.push([tick, bit], [tick + w, bit]); break; }
    case 'delete': { const p = pulses(t); if (p.length < 2) return null; const [a, b] = p[randInt(p.length)]; t.splice(b, 1); t.splice(a, 1); break; }
    case 'tail': { const i = 1 + randInt(n - 1); const d = sgn() * geometric(0.5); for (let k = i; k < n; k++) t[k][0] += d; break; }
    case 'double': { for (let k = 0; k < 2; k++) { const i = 1 + randInt(n - 1); t[i][0] += sgn() * geometric(0.3); } break; }
  }
  return normalize(t);
}

const pool = new Pool(workers);
await pool.ready;
const startControls = loadStart();
let best = {toggles: normalize(togglesFromControls(startControls))};
const horizonFor = f => f + 60;
async function rebase() {
  best.controls = controlsFromToggles(best.toggles, WR + 3000);
  const b = await pool.setBase(best.controls, 250);
  const r = await pool.evaluate(best.controls, b.finishFrame === null ? WR + 3000 : b.finishFrame + 5);
  best.f = r.finishFrame; best.score = score(r); best.cpTicks = r.cpTicks;
}
await rebase();
const t0 = performance.now();
const elapsed = () => (performance.now() - t0) / 1000;
const log = [];
console.log(`start f=${best.f} score=${best.score.toFixed(3)} toggles=${best.toggles.length} cps=${JSON.stringify(best.cpTicks)}`);

function save() {
  const controls = controlsFromToggles(best.toggles, best.f);
  const recording = encodeRecording(controls);
  const check = run('summer1', controls).finishFrame; // independent confirmation with sim.run
  const json = {
    finishFrame: best.f, frames: best.f, delta_vs_wr_ms: best.f - WR, score: best.score, simRunFinish: check,
    recording, toggles: togglesFromControls(controls).map(([tick, bit]) => [tick, bit]),
    toggleCount: best.toggles.length, seed: seed0, elapsed_s: Math.round(elapsed()), log,
  };
  fs.writeFileSync(out, JSON.stringify(json));
  return check;
}

let evals = 0;
async function evalBatch(cands) {
  const H = horizonFor(best.f);
  const res = await pool.evaluateAll(cands.map(c => controlsFromToggles(c, H)), H);
  evals += cands.length;
  return res.map(score);
}
function accept(cand, s, f, method) {
  best.toggles = cand; best.f = f;
  log.push({t: Math.round(elapsed()), method, f, score: Number(s.toFixed(3)), evals});
  console.log(`[${elapsed().toFixed(0)}s] ${method}: f=${f} score=${s.toFixed(3)} (${f - WR} vs WR) toggles=${cand.length} evals=${evals}`);
}

// Phase 1: coordinate descent over single-toggle shifts of +/-1, +/-2.
let improved = true;
while (improved && elapsed() < budgetS) {
  improved = false;
  const cands = [];
  for (let i = 1; i < best.toggles.length; i++) for (const d of [-2, -1, 1, 2]) { const t = clone(best.toggles); t[i][0] += d; cands.push(normalize(t)); }
  const scores = await evalBatch(cands);
  let bi = -1;
  for (let k = 0; k < cands.length; k++) if (scores[k] < best.score - 1e-9 && (bi < 0 || scores[k] < scores[bi])) bi = k;
  if (bi >= 0) { accept(cands[bi], scores[bi], Math.ceil(scores[bi] - 1e-9), 'cd'); await rebase(); save(); improved = true; }
}
console.log(`coordinate descent done: f=${best.f} score=${best.score.toFixed(3)} after ${evals} evals, ${elapsed().toFixed(0)} s`);

// Phase 2: randomized local search with adaptive move selection.
const stats = Object.fromEntries(MOVES.map(m => [m, {tries: 1, wins: 0.2}]));
const pick = () => { const w = MOVES.map(m => stats[m].wins / stats[m].tries + 0.02); let r = rand() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < MOVES.length; i++) { r -= w[i]; if (r <= 0) return MOVES[i]; } return MOVES[0]; };
let gen = 0, stale = 0;
while (elapsed() < budgetS) {
  const K = workers * 8;
  const cands = [], kinds = [];
  while (cands.length < K) {
    const kind = pick();
    let c = mutate(best.toggles, kind);
    if (!c) continue;
    if (stale > 20 && rand() < 0.5) { const c2 = mutate(c, MOVES[randInt(MOVES.length)]); if (c2) c = c2; }
    cands.push(c); kinds.push(kind);
  }
  const scores = await evalBatch(cands);
  let bi = -1;
  for (let k = 0; k < cands.length; k++) { stats[kinds[k]].tries++; if (scores[k] < best.score - 1e-9) { stats[kinds[k]].wins++; if (bi < 0 || scores[k] < scores[bi]) bi = k; } }
  gen++;
  if (bi >= 0) { accept(cands[bi], scores[bi], Math.ceil(scores[bi] - 1e-9), kinds[bi]); await rebase(); save(); stale = 0; }
  else stale++;
  if (gen % 20 === 0) console.log(`[${elapsed().toFixed(0)}s] gen ${gen} evals=${evals} (${(evals / elapsed()).toFixed(0)}/s) best f=${best.f} score=${best.score.toFixed(3)} stale=${stale} moves=${MOVES.map(m => `${m}:${stats[m].wins.toFixed(0)}/${stats[m].tries}`).join(' ')}`);
}
const check = save();
console.log(`done: f=${best.f} (${best.f - WR} vs WR) score=${best.score.toFixed(3)} sim.run=${check} evals=${evals} in ${elapsed().toFixed(0)} s -> ${out}`);
await pool.close();
