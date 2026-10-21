# Trailer plan (compound.mov)

Two acts. Act 1 exists now; act 2 needs Reflex-1's first self-taught lap.

| # | beat | seconds | asset | status |
| --- | --- | --- | --- | --- |
| 1 | Hook: "Can an AI beat the best Polytrack driver?" over the starting grid | 3 | frame from `records/summer1_wr_replay_side_by_side.mp4` | ready |
| 2 | Act 1, "It can copy perfectly": side-by-side, Reflex 22.111 vs human WR 22.123, freeze on both timers | 8 | `records/summer1_wr_replay_side_by_side.mp4` (cut 18 to 27 s) | ready |
| 3 | Fragility: "but replaying blind crashes": open-loop crash at checkpoint 1 | 4 | `runs/live/wr_*.mp4` (crash near 17 s) | ready |
| 4 | Act 2, "Now it learns on its own": learning curve (laps per iteration, getting faster) | 6 | `learn/curve.csv` -> chart | blocked on first self-taught lap |
| 5 | Its own lap in the real game, HUD timer | 8 | `loop/record_line.py learn/best.json` | blocked |
| 6 | How: "exact physics copy, searches ahead, sees through a vision model" (3 quick cards) | 6 | numbers from `LOOP.md` | ready |
| 7 | End card: Reflex-1, github link | 3 | text | ready |

Rules for the copy: say "tool-assisted replay of the human WR" for act 1, never "beat the
world record" unless act 2 actually does it from scratch.
