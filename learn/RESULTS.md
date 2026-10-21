# learn: Reflex-1 learns Summer 1 from scratch (track R)

No human data touches this directory: inputs are the track file (gates, part footprints)
and the byte-exact sim. `python3 loop/isolation.py learn` passes.

## Files

| file | what |
| --- | --- |
| `track.mjs`, `track.json` | start pose, checkpoint and finish gates, part list, from the game's own track parser |
| `route.mjs` | drivable tiles from the game's part table (narrow roads keep their middle two columns), Dijkstra cost-to-go field per gate with ramp-aware jump edges, `progress(gate, x, y, z)` in road metres |
| `plan.mjs` | receding-horizon CEM: 192 piecewise-constant mask sequences (50 ms segments, 2.5 s horizon, persistent sampling), 6 refits, commit 200 ms, 12 worker threads on heap snapshots; stall = back up with doubling backoff |
| `best.json` | best own lap `{controls, finishFrame}` |
| `curve.csv` | step, wall seconds, committed tick, checkpoint, best score, finish |
| `history/NNN.json` | every attempt that changed the picture (new checkpoint in reach, stall backup, finish, every 10th step) as `{masks: [[tick, mask], ...], frames, finishFrame, checkpoints, progress, wall_s, note}` for the learning montage |

## Score (per rollout)

`checkpoints * 1e4 - road distance to next gate` sampled at every grounded segment end
(flights are judged where they land; a rollout that ends airborne coasts to landing), minus
half of any regression at the end (fell off, spun, landed on the floor), plus 1.5 s of the
end speed in metres. A finish scores `5e4 - finishFrame / 10`, above any non-finish.

## Iteration log

| # | change | result | command |
| --- | --- | --- | --- |
| 1 | Euclidean distance to the next gate | cp2 reached at 11 s, then stuck under the cp2 road (fell off the climb) | `node learn/plan.mjs` |
| 2 | road-distance field over drivable tiles | cp3 in 16.8 s, then stuck: airborne over lower road scored as regress, car dropped to the floor plane | |
| 3 | ramp-aware jumps, floor trap, progress = best grounded point along rollout, narrow roads, persistent sampling | reached the finish straight at 26.2 s but refused to finish: finish score was below a cp3 non-finish | |
| 4 | finish score above any non-finish | **first finish 26642 ms**, 665 s wall, 125 steps, 18 history files | `node learn/plan.mjs --max 45000`; verify `node loop/score.mjs learn/best.json 20` -> finishFrame 26642 |

Jitter: `loop/score.mjs` reports 0/20 finishes under +/-1 and +/-2 ms per-toggle jitter.
Shifting single toggles by 1 ms shows why: a 1 ms change at tick 2400 (the first corner) ends
the lap at checkpoint 1. The line bounces off kerbs and flies; open-loop it is knife-edge, the
same finding as LOOP.md row 1 for the human WR. Next: optimize the own line for p05 finish
under +/-2 ms jitter (toggle local search with jittered evaluation), and the closed-loop
controller (ctrl/) tracks it.
