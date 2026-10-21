# Reflex-1

A local visual System One model that learns to drive Polytrack from pixels, on its own.
No human runs in training: it gets the track, an exact physics copy, and its own attempts (N+1).

Status and every measured number: `LOOP.md`. Plan: `DESIGN.md`, `research/`.
Never submits to the public leaderboard.

## Records

| date | track | time | category | what it shows | files |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | Summer 1 | **22.111 s** | tool-assisted replay | Reflex reproduces the human world record (Monstrosity, 22.123 s) frame-exactly in the real game, and a local search shaves 12 ms off by shifting 15 key presses by 1 to 2 ms | [video](records/summer1_wr_replay_side_by_side.mp4), [recording](records/summer1_22111.json) |

This is a replay of a human run, not a self-taught lap, so it isn't counted as Reflex-1's own
result (see `LOOP.md`, "Excluded"). Self-taught records go here when they exist.

Verify it yourself (needs the game files, `sim/fetch.sh`):

```sh
node --input-type=module -e "import fs from 'node:fs'; import {verifyRecording} from './sim/sim.js';
const r = JSON.parse(fs.readFileSync('records/summer1_22111.json'));
console.log(verifyRecording('summer1', r.recording, 22111))"   # true: the game's own Verify
```
