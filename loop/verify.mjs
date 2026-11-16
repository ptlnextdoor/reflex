// Game's own Verify for an agent line: encode controls as a Polytrack recording and run
// verifyRecording at the claimed finish frame (true) and one frame earlier (false).
//
//   node loop/verify.mjs learn/best.json
import fs from 'node:fs';
import {run, verifyRecording} from '../sim/sim.js';
import {encodeRecording} from './recording.mjs';

const path = process.argv[2] ?? 'learn/best.json';
const c = JSON.parse(fs.readFileSync(path, 'utf8'));
const controls = Uint8Array.from(c.controls);
// Recordings hold the last keys to the finish, so the run does too.
const padded = new Uint8Array(controls.length + 3000).fill(controls.at(-1));
padded.set(controls);
const {finishFrame} = run('summer1', padded);
const recording = encodeRecording(controls);
const verify = finishFrame !== null && verifyRecording('summer1', recording, finishFrame);
const earlier = finishFrame !== null && verifyRecording('summer1', recording, finishFrame - 1);
console.log(JSON.stringify({candidate: path, finishFrame, verify, verify_minus_1: earlier, beats_22123: verify && finishFrame < 22123, recording}));
