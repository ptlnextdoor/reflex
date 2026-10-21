# reflex: handoff state (2026-10-20 21:00 PT)

Read this first when resuming. The user works on this Mac while agents run: **never open
windows or send keys on the active Desktop.** All GUI work goes to Desktop 2 (below).

## Verified (re-run by the coordinator, not just reported)

| what | number | where |
| --- | --- | --- |
| sim parity | Node vs Chromium byte-exact over 5000 ticks; WR 59372564 ends at frame 22123, game Verify=true | `node sim/check.js` |
| sim speed | 114,480 ticks/s single thread (114x realtime) | same |
| Valen PyTorch MPS | 421 ms p50, 542 p99 | `bench/infer/RESULTS.md` |
| **Valen MLX fp16** | **86.7 ms p50** e2e (ViT 16.8, text 63.2), **32/32 vs true fp32** | `valen_mlx.py` |
| text-tower scaling | 9.1 ms fixed + 0.19 ms/token, ~18 ms floor below 64 tokens | `scaling.py` |
| streaming (question-first) | 54.7 ms e2e, cache exact (32/32 vs uncached), **image-blind** (15/32) | `streaming.py` |
| why image-blind | Mixer head reads the frame 99.2% through candidate-role features; question-first freezes them (0.000 change) | `stream_audit.py` |
| leaderboard guard | blocks POSTs to vps.kodub.com (exit 2), allows GETs | `.claude/hooks/` |

Gate reference is `bench/infer/ref_cpu_fp32.jsonl` (true fp32). The older `ref_cpu.jsonl`
was secretly bf16 autocast, so don't gate against it.

## Built, not measured

- `eye/`: SCK capture, CGEvent keys, m2p bench, plus `--window <id>` capture (works on
  another Desktop) and `--mode topid` keys (no focus). Selftest passes.
- **Desktop 2 procedure:** user runs `eye/open-test-browser.sh` (Chrome for Testing,
  separate from their Chrome), drags it to Desktop 2, says "moved". Then
  `eye list-windows --all --app com.google.chrome.for.testing` for the id,
  `eye postid-test --app com.google.chrome.for.testing --window <id>` first (proves keys
  arrive without focus), then `bench-capture --window <id>` and `bench-m2p --window <id> --mode topid`.
  Tried and failed: `com.apple.spaces app-bindings` pin and off-screen window position
  (both opened on the user's Desktop; reverted).

## Next, in order

1. `research/ENGINE.md` from Fable (agent `bug`, running).
2. **Head retrain for streaming** (the architecture change): backbone frozen, a new head
   that reads the frame through the context/decision roles, distilled from the
   state-first Valen as teacher on sim frames. Target: ~55 ms and >= 99% agreement.
3. eye benches once the user says "moved".
4. Data engine: real-game frames + sim labels, then Box head, then the Rust `engine/`.

## Agents

- `bug` (claude-fable-5-1): ENGINE.md, read-only.
- `cat` (gpt-6.1-sol): hit its OpenAI usage limit after `fa62c27`. Don't re-spawn on gpt
  until the limit resets.
