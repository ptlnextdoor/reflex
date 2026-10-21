// Print a candidate's toggle list as JSON: [[tick, mask], ...].
//   node loop/toggles.mjs --wr | <candidate.json>
import fs from 'node:fs';
import {decodeRecording} from '../sim/sim.js';

const a = process.argv[2] ?? '--wr';
let c;
if (a === '--wr') {
  const r = JSON.parse(fs.readFileSync(new URL('../vendor/polytrack/0.6.3/recording-59372564.json', import.meta.url)))[0].recording;
  c = decodeRecording(r, 22123);
} else {
  const j = JSON.parse(fs.readFileSync(a, 'utf8'));
  if (j.toggles) { console.log(JSON.stringify(j.toggles)); process.exit(0); }
  c = j.recording ? decodeRecording(j.recording, j.frames ?? j.finishFrame) : Uint8Array.from(j.controls);
}
const t = [[0, c[0]]];
for (let i = 1; i < c.length; i++) if (c[i] !== c[i - 1]) t.push([i, c[i]]);
console.log(JSON.stringify(t));
