// Shortcut probe: can the car get from the cp1 gate area onto the raised cp2 road directly,
// skipping the hairpin loop? Starts from a top lap's state just after cp1 and runs a
// CEM over 3 s of masks, scored only by distance to the cp2 gate and passing it.
//   node learn/shortcut_probe.mjs [--from-tick 9700] [--seed 1] [--iters 40]
import fs from 'node:fs';
const {decodeRecording} = await import('../sim/sim.js');
const {Car, decodeState} = await import('../ctrl/sim.mjs');

const arg = (n, d) => { const i = process.argv.indexOf(n); return i < 0 ? d : process.argv[i + 1]; };
const FROM = Number(arg('--from-tick', 9700)), HOR = Number(arg('--horizon', 3500)), SEG = 100;
const ITERS = Number(arg('--iters', 40)), N = Number(arg('--n', 400));
let seed = Number(arg('--seed', 1)) >>> 0;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

const rec = JSON.parse(fs.readFileSync('vendor/polytrack/0.6.3/top/recording-59372564.json', 'utf8'))[0];
const wr = decodeRecording(rec.recording, 22123);
const track = JSON.parse(fs.readFileSync('learn/track.json', 'utf8'));
const G = track.checkpoints[2].center; // cp2 gate on the raised road

const car = new Car('summer1');
for (let t = 0; t < FROM; t++) car.step(wr[t]);
const s0 = decodeState(car.state);
const snap = car.snapshot();
console.log('start state', FROM, 'cp', s0.nextCheckpointIndex, 'pos', s0.position, 'v', s0.speedKmh.toFixed(0));

const A = [1, 3, 9, 0, 5, 13, 7, 4]; // W, WD, WA, coast, WS, WAS, WDS, S
const SEGS = HOR / SEG;
let probs = Array.from({length: SEGS}, () => A.map(() => 1 / A.length));
let best = {score: -Infinity};
function rollout(seq) {
  car.restore(snap);
  let minD = Infinity, passed = null;
  for (let j = 0; j < SEGS; j++) {
    for (let k = 0; k < SEG; k++) {
      const s = decodeState(car.step(A[seq[j]]));
      if (s.nextCheckpointIndex >= 3 && passed === null) passed = FROM + j * SEG + k;
      const d = Math.hypot(s.position.x - G[0], s.position.y - G[1], s.position.z - G[2]);
      if (d < minD) minD = d;
    }
    if (passed !== null) break;
  }
  return {score: passed !== null ? 1e6 - passed : -minD, minD, passed};
}
for (let it = 0; it < ITERS; it++) {
  const res = [];
  for (let n = 0; n < N; n++) {
    const seq = probs.map(p => { let r = rnd(), a = 0; while (a < p.length - 1 && (r -= p[a]) > 0) a++; return a; });
    res.push({seq, ...rollout(seq)});
  }
  res.sort((a, b) => b.score - a.score);
  if (res[0].score > best.score) best = res[0];
  const el = res.slice(0, Math.round(N * 0.1));
  probs = probs.map((p, j) => p.map((_, a) => 0.7 * (0.01 + el.filter(e => e.seq[j] === a).length) / (el.length + 0.01 * A.length) + 0.3 * p[a]));
  console.log(`iter ${it} best minD ${best.minD.toFixed(1)} m passed ${best.passed}`);
  if (best.passed !== null && it > 5) break;
}
const WR_CP2 = 14931;
console.log(best.passed !== null ? `SHORTCUT: cp2 at tick ${best.passed} (WR ${WR_CP2}, saves ${WR_CP2 - best.passed} ms)` : `no shortcut: closest ${best.minD.toFixed(1)} m from cp2 gate`);
fs.writeFileSync(`runs/top/shortcut_${FROM}_${arg('--seed', 1)}.json`, JSON.stringify({from: FROM, seq: best.seq.map(i => A[i]), seg: SEG, ...best, seq_idx: undefined}));
