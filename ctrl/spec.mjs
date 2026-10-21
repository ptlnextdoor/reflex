// Perception spec sweep: how much latency, white position noise, heading noise and constant
// position bias the closed loop tolerates (jitter fixed at +/-2 ms). Output: the largest value
// of each axis that keeps the finish rate >= 90% with the other axes at the headline values.
//   node ctrl/spec.mjs --ref ctrl/lines/segment.json --seeds 20 --params ctrl/best_params.json
import fs from 'node:fs';
import {runJobs, pickReference, summarize, HEADLINE} from './eval.mjs';

function arg(name, def) { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; }

export const AXES = {
  latency: [8, 30, 60, 90, 120, 150, 200, 250, 300],
  noise: [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2],
  bias: [0, 0.25, 0.5, 1, 1.5, 2, 3],
  headingNoise: [0, 0.02, 0.05, 0.1, 0.2, 0.3],
  latencyError: [-40, -20, -10, 0, 10, 20, 40, 80],
  latencyJitter: [0, 4, 8, 16, 32],
};

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const refPath = pickReference(arg('--ref'));
  const seeds = Number(arg('--seeds', 20));
  const params = arg('--params') ? JSON.parse(fs.readFileSync(arg('--params'), 'utf8')) : {};
  const cells = [];
  for (const [axis, values] of Object.entries(AXES)) for (const v of values) cells.push({...HEADLINE, bias: 0, headingNoise: 0, latencyError: 0, latencyJitter: 0, [axis]: v, axis});
  const jobs = cells.flatMap(c => Array.from({length: seeds}, (_, s) => ({...c, seed: 2000 + s, params})));
  const {loadLine, buildReference} = await import('./reference.mjs');
  const ref = buildReference(loadLine(new URL(`../${refPath}`, import.meta.url)));
  console.log(`reference ${refPath}: ${ref.n} ms; headline ${JSON.stringify(HEADLINE)}; params ${JSON.stringify(params)}; ${cells.length} cells x ${seeds} seeds`);
  const results = await runJobs(jobs, refPath);
  const table = cells.map((c, k) => ({...c, ...summarize(results.slice(k * seeds, (k + 1) * seeds))}));
  const spec = {};
  for (const axis of Object.keys(AXES)) {
    console.log(`\n${axis} sweep (others at headline):`);
    let limit = null;
    for (const r of table.filter(r => r.axis === axis)) {
      console.log(`  ${axis}=${String(r[axis]).padStart(5)} | ${r.finished}/${r.n} | p50 ${r.p50 ?? '-'} (${r.p50 === null ? '-' : (r.p50 - ref.n >= 0 ? '+' : '') + (r.p50 - ref.n)}) | p90 ${r.p90 ?? '-'} | maxEy ${r.maxEy.toFixed(2)}`);
      if (r.rate >= 0.9) limit = r[axis];
    }
    spec[axis] = axis === 'latencyError' ? table.filter(r => r.axis === axis && r.rate >= 0.9).map(r => r[axis]) : limit;
    console.log(`  -> ${axis} with >= 90% finish: ${JSON.stringify(spec[axis])}`);
  }
  const out = arg('--out');
  if (out) fs.writeFileSync(out, JSON.stringify({refPath, refTime: ref.n, params, seeds, spec, table}, null, 1));
  console.log('\nspec', JSON.stringify(spec));
}
