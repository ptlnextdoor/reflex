// Gain search: random/local search over controller params, scored on a seed set, kept only if
// the held-out seed set also improves. Score = finish rate first, then p50 finish time.
//
//   node ctrl/tune.mjs --ref ctrl/segment.json --iters 40 --cells headline
//   node ctrl/tune.mjs --cells stress        # L in {60,90}, jitter in {2,5}, noise in {0.05,0.2}
//   node ctrl/tune.mjs --init ctrl/best_params.json --out ctrl/best_params.json
import fs from 'node:fs';
import {runJobs, pickReference, summarize, HEADLINE} from './eval.mjs';
import {DEFAULT_PARAMS} from './controller.mjs';
import {rng} from './episode.mjs';

function arg(name, def) { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; }

const SPACE = {
  Ky: {lo: 0.02, hi: 3, log: true},
  Kpsi: {lo: 0.5, hi: 40, log: true},
  Kvy: {lo: 0, hi: 0.3, log: false},
  speedExp: {lo: -1.5, hi: 1.5, log: false},
  deadband: {lo: 0, hi: 0.1, log: false},
  window: {lo: 10, hi: 120, log: false, int: true},
  progressAlpha: {lo: 0, hi: 1, log: false},
};

export function cellsFor(name) {
  if (name === 'headline') return [HEADLINE];
  if (name === 'stress') return [60, 90].flatMap(latency => [2, 5].flatMap(jitter => [0.05, 0.2].map(noise => ({latency, jitter, noise}))));
  if (name === 'all') return [8, 30, 60, 90].flatMap(latency => [0, 2, 5].flatMap(jitter => [0, 0.05, 0.2].map(noise => ({latency, jitter, noise}))));
  return [HEADLINE];
}

// Score: finish rate (weight 1000 ms per 1%), minus mean p50 over cells. Higher is better.
export function score(table) {
  const rate = table.reduce((a, r) => a + r.rate, 0) / table.length;
  const p50s = table.filter(r => r.p50 !== null).map(r => r.p50);
  const p50 = p50s.length ? p50s.reduce((a, b) => a + b, 0) / p50s.length : 1e6;
  return 100000 * rate - p50;
}

export async function evaluate(params, cells, seeds, refPath) {
  const jobs = cells.flatMap(c => seeds.map(seed => ({...c, seed, params})));
  const results = await runJobs(jobs, refPath);
  return cells.map((c, k) => ({...c, ...summarize(results.slice(k * seeds.length, (k + 1) * seeds.length))}));
}

function mutate(p, r, scale) {
  const q = {...p};
  const keys = Object.keys(SPACE);
  const n = 1 + Math.floor(r.next() * 2);
  for (let k = 0; k < n; k++) {
    const key = keys[Math.floor(r.next() * keys.length)];
    const s = SPACE[key];
    let v = q[key] ?? DEFAULT_PARAMS[key];
    if (s.log) v = Math.exp(Math.log(Math.max(v, s.lo)) + scale * 1.2 * r.gauss());
    else v = v + scale * (s.hi - s.lo) * 0.25 * r.gauss();
    v = Math.min(s.hi, Math.max(s.lo, v));
    if (s.int) v = Math.round(v);
    q[key] = Number(v.toFixed(4));
  }
  return q;
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const refPath = pickReference(arg('--ref'));
  const iters = Number(arg('--iters', 30));
  const cells = cellsFor(arg('--cells', 'headline'));
  const nSeeds = Number(arg('--seeds', 12));
  const train = Array.from({length: nSeeds}, (_, i) => 1 + i);
  const held = Array.from({length: nSeeds}, (_, i) => 501 + i);
  const out = arg('--out', 'ctrl/best_params.json');
  let best = arg('--init') ? JSON.parse(fs.readFileSync(arg('--init'), 'utf8')) : {...DEFAULT_PARAMS};
  const r = rng(Number(arg('--seed', 7)));
  const log = [];
  let bestTrain = await evaluate(best, cells, train, refPath), bestHeld = await evaluate(best, cells, held, refPath);
  let bestScore = Math.min(score(bestTrain), score(bestHeld));
  const fmt = t => `rate ${(t.reduce((a, x) => a + x.rate, 0) / t.length * 100).toFixed(0)}% p50 ${Math.round(t.filter(x => x.p50 !== null).reduce((a, x) => a + x.p50, 0) / Math.max(1, t.filter(x => x.p50 !== null).length))}`;
  console.log(`ref ${refPath}, ${cells.length} cells, ${nSeeds} train + ${nSeeds} held-out seeds, ${iters} iters`);
  console.log(`init ${JSON.stringify(best)}: train ${fmt(bestTrain)} | held ${fmt(bestHeld)} | score ${bestScore.toFixed(0)}`);
  let stale = 0;
  for (let it = 0; it < iters; it++) {
    const scale = stale > 6 ? 1.5 : 0.7;
    const cand = mutate(best, r, scale);
    const tr = await evaluate(cand, cells, train, refPath);
    const sTr = score(tr);
    if (sTr <= score(bestTrain)) { stale++; log.push({it, cand, train: fmt(tr), kept: false}); console.log(`it ${it}: ${JSON.stringify(cand)} train ${fmt(tr)} rejected`); continue; }
    const he = await evaluate(cand, cells, held, refPath);
    const s = Math.min(sTr, score(he));
    const kept = s > bestScore;
    console.log(`it ${it}: ${JSON.stringify(cand)} train ${fmt(tr)} | held ${fmt(he)} | score ${s.toFixed(0)} ${kept ? 'KEPT' : 'rejected (held-out)'}`);
    log.push({it, cand, train: fmt(tr), held: fmt(he), kept});
    if (kept) { best = cand; bestTrain = tr; bestHeld = he; bestScore = s; stale = 0; fs.writeFileSync(out, JSON.stringify(best, null, 1)); } else stale++;
  }
  fs.writeFileSync(out, JSON.stringify(best, null, 1));
  console.log(`best ${JSON.stringify(best)} -> ${out}`);
}
