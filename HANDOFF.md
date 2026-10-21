# reflex: handoff state (2026-10-20 19:30 PT)

Read this first when resuming. Battery hit 6% mid-session, two Claude workers died on
an Anthropic API error, so work stopped mid-flight.

## Verified (re-run by the coordinator, not just reported)

- `sim/`: `node sim/check.js` passes. Node vs Chromium byte-exact over 5000 ticks,
  WR recording 59372564 finishes at frame 22123, the game's own Verify returns true.
  28,331 ticks/s single thread (50x target missed; use worker_threads).
- `.claude/hooks/leaderboard_guard.py`: blocks POSTs to vps.kodub.com (exit 2), allows GETs.
- `bench/infer/valen_ref.py`: Valen-0.8B (rev c2c481f) runs on CPU, 302 tokens/request,
  reference answers in `ref_cpu.jsonl`. Zero-shot steering is near useless (fine-tune needed).

## Built, not measured

- `eye/`: builds, `eye selftest` passes (commit 6369cc4). `bench-capture` saw only idle
  frames. Two causes the worker found: (1) the built-in display was asleep on battery
  (`CGActiveDisplayCount=0` until `caffeinate -u`), (2) Chrome sends few complete
  frames when not frontmost. Status codes seen: 0 = complete, 1 = idle. Resume:
  on wall power, `caffeinate -d &`, bring the `eye/probe/anim.html` Chrome window to front,
  `eye bench-capture --rect <content rect> --seconds 10`, then `bench-m2p`.

## Not started

- **Done since:** Valen MLX fp16 port (`bench/infer/valen_mlx.py`): **86.7 ms p50** end to end,
  forward 82.1 ms (ViT 16.8, text tower 63.2), **32/32 vs true fp32 reference**.
  PyTorch MPS was 421 ms. See `bench/infer/RESULTS.md`.
- eye benches on Desktop 2: run `eye/open-test-browser.sh`, user drags the window to
  Desktop 2, then `eye list-windows --all` for its id and `--window <id> --mode topid`.
  Never open windows on the user's active Desktop.

- Valen MPS baseline (`cd bench/infer && .venv/bin/python valen_ref.py --device mps`).
- mlx-vlm path: `mlx-vlm 0.7.9` installed in `bench/infer/.venv`, has `qwen3_5` with
  GatedDeltaNet. Valen keys are `backbone.language_model.*` -> `language_model.model.*`,
  `backbone.visual.*` -> `vision_tower.*`, plus `head.*` (Mixer, port by hand from
  `heads.py`). Take hidden states before `lm_head`.
- `research/ENGINE.md`: the Fable reviewer died before writing it. Re-spawn with the same prompt.
- Rust `engine/` (sys1rust fork), Box head, data engine, closed loop, L1 to L3.

## Order to resume

1. Wall power. Re-run `node sim/check.js` and `eye selftest` (smoke).
2. eye benches, then the Valen MPS baseline (one GPU job at a time).
3. Re-spawn the Fable engine review, then the mlx-vlm bake-off builder.
