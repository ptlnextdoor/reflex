// Reflex-1 learns a lap from scratch: receding-horizon cross-entropy planning on the exact
// sim. Inputs are only what a new player gets: the track gates (learn/track.json) and the
// physics (ctrl/sim.mjs). No human run or external data of any kind (LOOP.md isolation rule).
//
// Each planning step: sample N piecewise-constant mask sequences (SEG ms segments over a
// HORIZON), roll each out from a heap snapshot at the committed state on 12 worker threads,
// score = checkpoints passed * BIG - road distance to the next gate (learn/route.mjs) + small
// speed bonus (finish:
// BIG per gate plus BIG minus finish tick / 10, so any finish beats any non-finish), refit the per-segment categorical distribution to the
// elites, repeat ITERS times, commit the best first COMMIT ms, shift the distribution.
// If the best score stalls, back up BACK ms and re-plan with a flat distribution.
//
//   node learn/plan.mjs                       # first lap -> learn/best.json, learn/curve.csv
//   node learn/plan.mjs --samples 192 --iters 6 --horizon 2500 --out learn/best2.json
import fs from 'node:fs';
import os from 'node:os';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';

const ACTIONS = [1, 9, 3, 0, 8, 2, 4, 12, 6]; // up, up+L, up+R, coast, L, R, brake, brake+L, brake+R
const BIG = 1e4, LOOK = 1.5;
const track = JSON.parse(fs.readFileSync(new URL('./track.json', import.meta.url), 'utf8'));
const NGATES = track.checkpoints.length + 1;

function arg(name, def) { const i = process.argv.indexOf(name); return i < 0 ? def : process.argv[i + 1]; }

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------- worker: rollouts
if (!isMainThread) {
  const {Car} = await import('../ctrl/sim.mjs');
  const {buildFields} = await import('./route.mjs');
  const {progress} = buildFields();
  const car = new Car(workerData.track);
  const view = () => new DataView(car.state.buffer, car.state.byteOffset, car.state.length);
  const snap0 = car.snapshot();
  let snapCur = car.snapshot();
  // Progress of the current state: checkpoints passed * BIG - road distance to the next gate
  // (learn/route.mjs). Sampled along the rollout; the score takes the best progress seen, so
  // a flight over a lower road or a late crash does not hide the ground gained before it.
  const peek = () => {
    const s = car.state, v = view();
    const finished = (s[11] & 2) !== 0;
    const off = finished ? 15 : 12;
    const cp = v.getUint16(off, true);
    const x = v.getFloat32(off + 2, true), y = v.getFloat32(off + 6, true), z = v.getFloat32(off + 10, true);
    return {cp, x, y, z, speed: v.getFloat32(7, true), d: progress(cp, x, y, z)};
  };
  parentPort.on('message', m => {
    if (m.type === 'advance') { car.restore(snapCur); for (const k of m.masks) car.step(k); snapCur = car.snapshot(snapCur); parentPort.postMessage({type: 'ok'}); return; }
    if (m.type === 'replay') { car.restore(snap0); for (const k of m.controls) car.step(k); snapCur = car.snapshot(snapCur); parentPort.postMessage({type: 'ok'}); return; }
    // eval: m.seqs is a Uint8Array of n*segs action indices; expand each into masks of m.seg ms.
    const {seqs, n, segs, seg} = m;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      car.restore(snapCur);
      let fin = null, best = -Infinity;
      outer: for (let j = 0; j < segs; j++) {
        const mask = ACTIONS[seqs[i * segs + j]];
        for (let t = 0; t < seg; t++) { const s = car.step(mask); if (s[11] & 2) { fin = car.finishFrame; break outer; } }
        // Only grounded states count as progress: a flight is judged where it lands.
        if ((car.state[11] >> 3) & 15) { const p = peek(); best = Math.max(best, BIG * p.cp - p.d); }
      }
      // A rollout that ends in the air is judged where it lands, not where it flies: coast on
      // (keys do nothing airborne) for up to 3 s until the wheels have touched for 100 ms.
      if (fin === null && ((car.state[11] >> 3) & 15) === 0) {
        for (let t = 0, down = 0; t < 3000 && down < 100; t++) {
          const s = car.step(1);
          if (s[11] & 2) { fin = car.finishFrame; break; }
          down = ((s[11] >> 3) & 15) ? down + 1 : 0;
        }
      }
      const e = peek();
      if (fin !== null) out[i] = {score: BIG * (NGATES + 1) - fin / 10, cp: e.cp, finish: fin, x: e.x, y: e.y, z: e.z};
      else {
        // Speed term: the distance the car would add in LOOK more seconds, so a rollout that
        // arrives fast beats one that arrives slightly closer by crashing. Ending worse than
        // the best grounded point along the way (fell off, spun, landed on the floor) costs
        // half the regression: the end state is what the next step inherits.
        const end = BIG * e.cp - e.d;
        out[i] = {score: Math.max(best, end) - 0.5 * Math.max(0, best - end) + LOOK * e.speed / 3.6, cp: e.cp, finish: null, x: e.x, y: e.y, z: e.z};
      }
    }
    parentPort.postMessage({type: 'scores', out});
  });
  car.restore(snapCur);
} else if (process.argv[1] === new URL(import.meta.url).pathname) {
  const trackName = arg('--track', 'summer1');
  const N = Number(arg('--samples', 192)), ITERS = Number(arg('--iters', 6)), HORIZON = Number(arg('--horizon', 2500));
  const SEG = Number(arg('--seg', 50)), COMMIT = Number(arg('--commit', 200)), ELITE = Number(arg('--elite', 0.15));
  const BACK = Number(arg('--back', 1000)), STALL = Number(arg('--stall', 6)), MAXMS = Number(arg('--max', 60000));
  const out = arg('--out', new URL('./best.json', import.meta.url).pathname);
  const curvePath = arg('--curve', new URL('./curve.csv', import.meta.url).pathname);
  const seed = Number(arg('--seed', 1));
  const threads = Number(arg('--threads', Math.max(1, Math.min(os.availableParallelism() - 2, 12))));
  const rand = rng(seed);
  const SEGS = Math.ceil(HORIZON / SEG), CSEGS = COMMIT / SEG, A = ACTIONS.length;

  const workers = Array.from({length: threads}, () => new Worker(new URL(import.meta.url), {workerData: {track: trackName}}));
  const ask = (w, msg, transfer) => new Promise(res => { w.once('message', res); w.postMessage(msg, transfer); });
  const broadcast = msg => Promise.all(workers.map(w => ask(w, msg)));

  // Per-segment categorical distribution over actions; starts biased to full throttle.
  // Sampling is persistent: with probability RHO a segment repeats the previous one, so long
  // holds (brake for 500 ms, steer for 400 ms) are common instead of 9^40-rare.
  const RHO = Number(arg('--rho', 0.6));
  const flat = () => { const p = new Float64Array(SEGS * A); for (let j = 0; j < SEGS; j++) for (let a = 0; a < A; a++) p[j * A + a] = a < 3 ? 0.25 : 0.25 / 6; return p; };
  let probs = flat();
  const masksOf = seq => { const m = []; for (let j = 0; j < SEGS; j++) for (let t = 0; t < SEG; t++) m.push(ACTIONS[seq[j]]); return m; };
  const sample = (p, n) => {
    const seqs = new Uint8Array(n * SEGS);
    for (let i = 0; i < n; i++) for (let j = 0; j < SEGS; j++) {
      if (j > 0 && rand() < RHO) { seqs[i * SEGS + j] = seqs[i * SEGS + j - 1]; continue; }
      let u = rand(), a = 0;
      while (a < A - 1 && (u -= p[j * A + a]) > 0) a++;
      seqs[i * SEGS + j] = a;
    }
    return seqs;
  };
  async function evaluate(seqs, n) {
    const per = Math.ceil(n / threads);
    const parts = await Promise.all(workers.map((w, k) => {
      const lo = k * per, hi = Math.min(n, lo + per);
      if (lo >= hi) return [];
      return ask(w, {type: 'eval', seqs: seqs.subarray(lo * SEGS, hi * SEGS), n: hi - lo, segs: SEGS, seg: SEG}).then(r => r.out);
    }));
    return parts.flat();
  }

  const controls = [];
  const t0 = performance.now();
  // History for the learning montage: every attempt that changes the picture (new checkpoint,
  // backup after a stall, finish) plus every 10th step, as mask change points {masks: [[tick, mask], ...]}.
  const histDir = new URL('./history/', import.meta.url);
  fs.mkdirSync(histDir, {recursive: true});
  let histN = fs.readdirSync(histDir).length;
  const toggles = c => { const t = []; for (let i = 0; i < c.length; i++) if (i === 0 || c[i] !== c[i - 1]) t.push([i, c[i]]); return t; };
  const saveHistory = (ctrl, best, note) => {
    const rec = {masks: toggles(ctrl), frames: ctrl.length, finishFrame: best.finish ?? null, checkpoints: best.cp, progress: best.score, wall_s: +((performance.now() - t0) / 1000).toFixed(1), note};
    fs.writeFileSync(new URL(`./${String(histN++).padStart(3, '0')}.json`, histDir), JSON.stringify(rec));
  };
  let lastCp = -1;
  fs.writeFileSync(curvePath, 'step,seconds,tick,cp,best_score,finish_ms\n');
  let lastBest = -Infinity, everBest = -Infinity, stalls = 0, stall = 0, step = 0, finish = null, bestSeq = null;
  while (controls.length < MAXMS) {
    let best = null, bestSeqIter = null, elites = null;
    for (let it = 0; it < ITERS; it++) {
      let seqs = sample(probs, N);
      if (bestSeq) seqs.set(bestSeq, 0); // keep last plan (shifted) as one candidate
      const res = await evaluate(seqs, N);
      const order = res.map((r, i) => i).sort((a, b) => res[b].score - res[a].score);
      const ne = Math.max(4, Math.round(N * ELITE));
      if (!best || res[order[0]].score > best.score) { best = res[order[0]]; bestSeqIter = seqs.slice(order[0] * SEGS, (order[0] + 1) * SEGS); }
      // Refit: elite frequencies, smoothed toward the previous distribution.
      const p = new Float64Array(SEGS * A).fill(0.02);
      for (let e = 0; e < ne; e++) for (let j = 0; j < SEGS; j++) p[j * A + seqs[order[e] * SEGS + j]] += 1;
      for (let j = 0; j < SEGS; j++) { let s = 0; for (let a = 0; a < A; a++) s += p[j * A + a]; for (let a = 0; a < A; a++) p[j * A + a] = 0.7 * p[j * A + a] / s + 0.3 * probs[j * A + a]; }
      probs = p;
      elites = ne;
    }
    step++;
    const secs = (performance.now() - t0) / 1000;
    if (best.finish !== null) {
      // Commit the whole finishing plan up to the finish tick.
      const masks = masksOf(bestSeqIter);
      const upto = best.finish - controls.length;
      controls.push(...masks.slice(0, upto));
      finish = best.finish;
      saveHistory(controls, best, 'finish');
      fs.appendFileSync(curvePath, `${step},${secs.toFixed(1)},${controls.length},${best.cp},${best.score.toFixed(1)},${finish}\n`);
      console.log(`step ${step} ${secs.toFixed(1)}s FINISH at ${finish} ms`);
      break;
    }
    // Commit the first COMMIT ms of the best sequence.
    const masks = masksOf(bestSeqIter).slice(0, COMMIT);
    controls.push(...masks);
    await broadcast({type: 'advance', masks});
    if (best.cp > lastCp) { lastCp = best.cp; saveHistory(controls.concat(masksOf(bestSeqIter).slice(COMMIT)), best, `checkpoint ${best.cp} in reach`); }
    else if (step % 10 === 0) saveHistory(controls.concat(masksOf(bestSeqIter).slice(COMMIT)), best, 'periodic');
    // Shift distribution and the best plan by the committed segments.
    const shifted = flat();
    shifted.set(probs.subarray(CSEGS * A), 0);
    probs = shifted;
    bestSeq = new Uint8Array(SEGS); bestSeq.set(bestSeqIter.subarray(CSEGS)); bestSeq.fill(0, SEGS - CSEGS);
    fs.appendFileSync(curvePath, `${step},${secs.toFixed(1)},${controls.length},${best.cp},${best.score.toFixed(1)},\n`);
    console.log(`step ${step} ${secs.toFixed(1)}s tick ${controls.length} cp ${best.cp} score ${best.score.toFixed(1)} pos ${best.x?.toFixed(0)},${best.y?.toFixed(0)},${best.z?.toFixed(0)} elites ${elites}`);
    // Stall: the horizon-end score should rise every step; if not, back up and widen the search.
    if (best.score > lastBest + 1) { lastBest = best.score; stall = 0; if (best.score > everBest) { everBest = best.score; stalls = 0; } }
    else if (++stall >= STALL) {
      // Progressive backoff: each consecutive stall at the same place backs up further.
      const back = Math.min(BACK * 2 ** stalls, 8000);
      const keep = Math.max(0, controls.length - back);
      stalls++;
      controls.length = keep;
      await broadcast({type: 'replay', controls: Uint8Array.from(controls)});
      saveHistory(controls.concat(masksOf(bestSeqIter)), best, `stall, backed up ${back} ms`);
      probs = flat(); bestSeq = null; stall = 0; lastBest = -Infinity;
      console.log(`  stall: backed up to tick ${keep}`);
    }
    if (step % 5 === 0) fs.writeFileSync(out + '.partial', JSON.stringify({controls, finishFrame: null}));
  }
  for (const w of workers) w.terminate();
  if (finish !== null) {
    fs.writeFileSync(out, JSON.stringify({controls, finishFrame: finish}));
    console.log(`wrote ${out}: finish ${finish} ms, ${controls.length} controls, ${((performance.now() - t0) / 1000).toFixed(0)} s wall`);
  } else {
    console.log('no finish within --max ms');
    process.exit(1);
  }
}
