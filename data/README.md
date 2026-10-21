# data: real-game frames with exact sim labels

Headless Chromium (Metal, no window) renders the real Polytrack at exact sim frame indices.
`hook.js` takes over `requestAnimationFrame` and feeds the player car byte-exact 227-byte
states from the Node sim through the game's own UpdateResult callback, so the renderer
draws exactly sim frame k. No human recordings anywhere (`python3 loop/isolation.py data`).

- `drive.mjs <outDir> <track> <seeds a-b> <ms>`: seeded random drives, closed loop on `ctrl/sim.mjs`.
- `labels.mjs <controls.json> <outDir> [step]`: `states.bin` + `labels.jsonl` from any `{controls|toggles}` file.
- `render.py <controls.json ...> [--step 50] [--small 256x144]`: `runs/frames/<episode>/{k:06d}.png` + `small/`, labels, hitbox.
- `check.py [controls.json]`: assert-based proof, page car state == sim state (error 0.0), run before trusting a change.
- `python3 data/render.py runs/drives/summer1_s1.json`: ~20 frames/s at 512x288 on the M4 Pro (boot 3 s excluded).
