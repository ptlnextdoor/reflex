import assert from 'node:assert/strict';
import fs from 'node:fs';
import {run, decodeRecording, verifyRecording, vendor} from '../sim/sim.js';
import {encodeRecording, encodeRecordingRaw, inflateRecording, togglesFromControls, controlsFromToggles} from './recording.mjs';

const rec = JSON.parse(fs.readFileSync(`${vendor}recording-59372564.json`, 'utf8'))[0];
const controls = decodeRecording(rec.recording, rec.frames);
const toggles = togglesFromControls(controls);
assert.equal(toggles.length, 134);
assert.deepEqual(controlsFromToggles(toggles, rec.frames), controls);
// Raw payload must match the original byte for byte; compressed bytes may differ by deflate settings.
const originalRaw = inflateRecording(rec.recording);
const ourRaw = encodeRecordingRaw(controls);
assert.deepEqual(ourRaw, originalRaw);
const encoded = encodeRecording(controls);
assert.deepEqual(decodeRecording(encoded, rec.frames), controls);
console.log(`raw payload match: ${ourRaw.length} bytes; our string ${encoded.length} chars vs original ${rec.recording.length}; identical string: ${encoded === rec.recording}`);
for (const level of [1, 6, 9]) console.log(`level ${level} identical: ${encodeRecording(controls, level) === rec.recording}`);
assert.equal(verifyRecording('summer1', encoded, rec.frames), true);
assert.equal(verifyRecording('summer1', encoded, rec.frames - 1), false);
// A trailing hold: decode with frames past the last toggle must still verify.
assert.equal(run('summer1', controls).finishFrame, rec.frames);
console.log('encodeRecording round trip PASS, Verify(22123)=true, Verify(22122)=false');
