// Independent scorer for track A claims. Re-simulates a candidate from scratch and
// reports the finish frame, plus finish rate under per-toggle timing jitter.
// Never trusts the worker's numbers.
//
//   node loop/score.mjs opt/best.json            # candidate file with {controls: [...]} or {toggles: [[t,mask],...]}
//   node loop/score.mjs --wr                     # the world record, for calibration
import fs from 'node:fs';
import {run, decodeRecording} from '../sim/sim.js';

const WR_FRAMES = 22123;
const vendor = new URL('../vendor/polytrack/0.6.3/', import.meta.url);

function loadControls(arg) {
  if (arg === '--wr') {
    const rec = JSON.parse(fs.readFileSync(new URL('recording-59372564.json', vendor)))[0].recording;
    return decodeRecording(rec, WR_FRAMES);
  }
  const c = JSON.parse(fs.readFileSync(arg, 'utf8'));
  if (c.recording) return decodeRecording(c.recording, c.frames ?? c.finishFrame ?? undefined);
  if (Array.isArray(c.controls)) return Uint8Array.from(c.controls);
  if (Array.isArray(c.toggles)) {          // [[tick, mask], ...] = mask held from tick on
    const n = (c.frames ?? c.finishFrame ?? 30000) + 2000;
    const out = new Uint8Array(n);
    const t = [...c.toggles].sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < t.length; i++) out.fill(t[i][1], t[i][0], i + 1 < t.length ? t[i + 1][0] : n);
    return out;
  }
  throw Error('candidate needs recording, controls or toggles');
}

function toggles(controls) {
  const out = [[0, controls[0]]];
  for (let i = 1; i < controls.length; i++) if (controls[i] !== controls[i - 1]) out.push([i, controls[i]]);
  return out;
}

function fromToggles(t, n) {
  const out = new Uint8Array(n);
  for (let i = 0; i < t.length; i++) out.fill(t[i][1], t[i][0], i + 1 < t.length ? t[i + 1][0] : n);
  return out;
}

function jitter(controls, ms, samples, seed = 1) {
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const base = toggles(controls);
  const finishes = [];
  for (let k = 0; k < samples; k++) {
    let prev = 0;
    const t = base.map(([tick, mask], i) => {
      let nt = i === 0 ? 0 : tick + Math.round((rand() * 2 - 1) * ms);
      nt = Math.max(prev + 1, nt);
      prev = nt;
      return [i === 0 ? 0 : nt, mask];
    });
    finishes.push(run('summer1', fromToggles(t, controls.length + 3000)).finishFrame);
  }
  const ok = finishes.filter(f => f !== null).sort((a, b) => a - b);
  return {ms, samples, finish_rate: ok.length / samples,
          p50: ok[Math.floor(ok.length / 2)] ?? null, best: ok[0] ?? null, worst: ok.at(-1) ?? null};
}

const arg = process.argv[2] ?? '--wr';
const samples = Number(process.argv[3] ?? 40);
const controls = loadControls(arg);
const padded = new Uint8Array(controls.length + 3000);
padded.set(controls);
padded.fill(controls.at(-1), controls.length);
const {finishFrame} = run('summer1', padded);
const out = {candidate: arg, finishFrame, delta_vs_wr_ms: finishFrame === null ? null : finishFrame - WR_FRAMES,
             toggles: toggles(controls).length,
             jitter_1ms: jitter(controls, 1, samples), jitter_2ms: jitter(controls, 2, samples)};
console.log(JSON.stringify(out, null, 1));
