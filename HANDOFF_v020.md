# HANDOFF: PolyTrack Track 1 under 20 s, deadline 2:45 pm PT (21:45 UTC)

You are omp on the crew server. Do everything here. The laptop must stay idle.
Repo: ~/code/reflex020 (branch bakeoff/lateral-heading). Python venv: .venv (playwright + chromium headless shell installed).
HARD RULE: never submit anything to the PolyTrack leaderboard. Any method is allowed (old versions, glitches, wall clips).

## Goal
A finished Track 1 lap under 20.0 s in PolyTrack **0.2.0**, plus an mp4 video of it. Version 0.6.3 cannot do it (WR 22.12 s, our best 22.093 s).

## Why 0.2.0 works
0.2.0 Track 1 has NO checkpoints, only start (200,55,40) and finish (-300,22.2,280). The speedrun.com 16.027 s run (video frames show it):
start ramp down to the low road heading -x (y=0, z~0). At ~x=-60..-100, t~7 s, it slams into the tall grey block on the
+z side (the y=20 elevated deck spanning x -140..-60, z 40..280), tumbles UP onto the deck at t~8-10 s, lands upright, then
drives +z along the deck and -x to the finish at (-300,22,280). Track parts: runs/top/v020_track1_parts.json (x,z *20 m, y *5 m).

## Harness (all in loop/)
- v020.py boot(): patched bundle vendor/polytrack/0.2.0/dist/bundle.patched.js. window.__plan={masks,len} drives controls
  (bit 1=W, 2=D, 4=S, 8=A). __manual=true + window.__world.__stepN(n) steps physics 1 ms each, deterministic.
  Car: window.__car.getPosition(), hasFinished(), getTime().
- v020_search.py Game: boot + select Track 1; restart via KeyR then KeyW.
- v020_plan.py: receding-horizon waypoint planner (Runner recycles browser every 120 rollouts, since Ammo leaks -> abort(OOM)).
- v020_climb.py: CEM that keeps the first --keep ms of a prefix and searches the next --horizon ms to get on top of the deck.
  Prefix: runs/top/v020_plan_1.json (11.5 s of controls driving the normal road; car at x~-60 around 5.5-6 s).
- Linux uses SwiftShader + 160x90 viewport (rendering eats CPU; load hit 38 with 6 browsers at 1280x720). Run at most ~5 browsers.

## Running now (tmux cl1..cl5 in ~/code/reflex020, logs runs/top/cl_*.log, best -> runs/top/v020_climb_<seed>.json)
Score > 10000 means it reached the deck. ex11 (0.6.3 explore) can be killed.

## Next steps
1. Watch cl_* logs. If none reaches score>10000 in ~10 min, fix the approach: try the drop-off point / speed (keep 4500-6500),
   bigger horizon, or a hand-tuned control sequence (W + hard A/D into the wall at max speed, like the video).
2. Once on the deck: run v020_plan.py --route with waypoints [-100,22,70]->[-110,21,130]->[-130,21,200]->[-170,21,250]->[-240,21,280]->[-300,22,280]
   seeded from the climb controls (add an --init flag that sets `committed` to those controls), until FINISH.
3. Optimise finish time (shorten prefix, CEM on segments) and stop once < 20 s.
4. Record video: v020.py has a run mode with video (record_video_dir); render the winning controls to runs/top/v020_best.mp4
   and convert with ffmpeg. Save controls to runs/top/v020_best.json. git commit + push.
5. Write a 3-line status to runs/top/STATUS.txt every ~10 min (best time so far, what is running).
