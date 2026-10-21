// Eval grid for the closed-loop controller: latency x key jitter x position noise x seeds.
// Parallel over worker_threads (one sim instance per worker). Prints a table and writes JSON.
//
//   node ctrl/eval.mjs                       # headline cell + main grid, 20 seeds, reference auto
//   node ctrl/eval.mjs --ref ctrl/lines/segment.json --seeds 20 --params ctrl/best_params.json
//   node ctrl/eval.mjs --cell 60,2,0.05      # one cell
//   node ctrl/eval.mjs --motion               # heading derived from observed motion (box-only perception)
//   node ctrl/eval.mjs --out ctrl/results/grid.json
//
// Reference auto-selection: learn/best.json (Reflex-1's own planner) when present, else
// ctrl/lines/segment.json (own start-to-checkpoint-1 segment). Never a human run (LOOP.md isolation).
import fs from 'node:fs';
import os from 'node:os';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';

export const LATENCIES = [8, 30, 60, 90];
export const JITTERS = [0, 2, 5];
export const NOISES = [0, 0.05, 0.2];
export const HEADLINE = {latency: 60, jitter: 2, noise: 0.05};

export function pickReference(explicit) {
  if (explicit) return explicit;
  for (const p of ['learn/best.json', 'ctrl/lines/segment.json']) if (fs.existsSync(new URL(`../${p}`, import.meta.url))) return p;
  throw Error('no reference line: run `node ctrl/segment.mjs` or wait for learn/best.json');
}

function arg(name, def) { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; }

export function summarize(rows) {
  const fins = rows.filter(r => r.finish !== null).map(r => r.finish).sort((a, b) => a - b);
  const q = p => fins.length ? fins[Math.min(fins.length - 1, Math.floor(p * fins.length))] : null;
  return {n: rows.length, finished: fins.length, rate: fins.length / rows.length, p50: q(0.5), p90: q(0.9), best: fins[0] ?? null, worst: fins.at(-1) ?? null, maxEy: Math.max(...rows.map(r => r.maxEy))};
}

// Runs a list of jobs {latency, jitter, noise, seed, params} across workers, returns results in order.
export async function runJobs(jobs, refPath, {threads = Math.max(1, Math.min(os.availableParallelism() - 1, 12)), onProgress} = {}) {
  const results = new Array(jobs.length);
  let next = 0, done = 0;
  await Promise.all(Array.from({length: Math.min(threads, jobs.length)}, () => new Promise((resolve, reject) => {
    const w = new Worker(new URL(import.meta.url), {workerData: {refPath}});
    const feed = () => { if (next >= jobs.length) { w.terminate(); resolve(); return; } const i = next++; w.postMessage({i, job: jobs[i]}); };
    w.on('message', m => { results[m.i] = m.result; done++; onProgress?.(done, jobs.length); feed(); });
    w.on('error', reject);
    feed();
  })));
  return results;
}

if (!isMainThread) {
  const {loadLine, buildReference} = await import('./reference.mjs');
  const {episode} = await import('./episode.mjs');
  const ref = buildReference(loadLine(new URL(`../${workerData.refPath}`, import.meta.url)));
  parentPort.on('message', ({i, job}) => {
    const r = episode(ref, job);
    parentPort.postMessage({i, result: {finish: r.finish, aborted: r.aborted, ticks: r.ticks, maxEy: r.diag.maxEy, steerMs: r.diag.steerMs}});
  });
} else if (process.argv[1] === new URL(import.meta.url).pathname) {
  const refPath = pickReference(arg('--ref'));
  const seeds = Number(arg('--seeds', 20));
  const params = arg('--params') ? JSON.parse(fs.readFileSync(arg('--params'), 'utf8')) : {};
  const cell = arg('--cell');
  const cells = cell ? [Object.fromEntries(['latency', 'jitter', 'noise'].map((k, i) => [k, Number(cell.split(',')[i])]))]
    : [HEADLINE, ...LATENCIES.flatMap(latency => JITTERS.flatMap(jitter => NOISES.map(noise => ({latency, jitter, noise}))))];
  const jobs = cells.flatMap(c => Array.from({length: seeds}, (_, s) => ({...c, seed: 1000 + s, params, headingSource: process.argv.includes('--motion') ? 'motion' : 'exact'})));
  const {loadLine, buildReference, WR_FRAMES} = await import('./reference.mjs');
  const ref = buildReference(loadLine(new URL(`../${refPath}`, import.meta.url)));
  console.log(`reference ${refPath}: ${ref.n} ms, goal ${ref.goal.finish ? 'finish' : 'checkpoint ' + ref.goal.checkpoint}; params ${JSON.stringify(params)}; ${cells.length} cells x ${seeds} seeds`);
  const t0 = performance.now();
  const results = await runJobs(jobs, refPath, {onProgress: (d, n) => { if (d % 50 === 0 || d === n) process.stderr.write(`\r${d}/${n}`); }});
  process.stderr.write('\n');
  const table = cells.map((c, k) => ({...c, ...summarize(results.slice(k * seeds, (k + 1) * seeds))}));
  const refTime = ref.n;
  console.log(`latency jitter noise | finish  p50      p90      best   | d(p50) vs ref ${refTime}${ref.goal.finish ? ` (WR ${WR_FRAMES})` : ''} | maxEy`);
  for (const r of table) console.log(`${String(r.latency).padStart(7)} ${String(r.jitter).padStart(6)} ${String(r.noise).padStart(5)} | ${String(r.finished + '/' + r.n).padStart(6)}  ${String(r.p50 ?? '-').padStart(7)}  ${String(r.p90 ?? '-').padStart(7)}  ${String(r.best ?? '-').padStart(6)} | ${r.p50 === null ? '-' : (r.p50 - refTime >= 0 ? '+' : '') + (r.p50 - refTime)}${ref.goal.finish && r.p50 !== null ? ` (${r.p50 - WR_FRAMES >= 0 ? '+' : ''}${r.p50 - WR_FRAMES} vs WR)` : ''} | ${r.maxEy.toFixed(2)}`);
  console.log(`wall ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  const out = arg('--out');
  if (out) { fs.mkdirSync(new URL('./results/', import.meta.url), {recursive: true}); fs.writeFileSync(out, JSON.stringify({refPath, refTime, goal: ref.goal, params, seeds, table, results}, null, 1)); console.log('wrote', out); }
}
