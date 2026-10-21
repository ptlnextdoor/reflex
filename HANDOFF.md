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
| head-only fix | 41% -> 64% held-out; frozen features hold R^2 ~0.7 of the signal. Insufficient | `distill_head.py` |
| prompt trim | -39 tokens = -1.5 ms, answers change (17/32). Not a lever | `prompt_trim.py` |
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

## Next, in order (from research/ENGINE.md, fb9bb39)

1. **Data engine** (gates everything below): replay the WR and perturbed recordings in
   the headless sim, render frames, label each with the sim state (car pose, projected
   hitbox, the plan's next action). Real-game frames via eye once the user says "moved".
2. **Paired LoRA**: fine-tune Valen on sim-labeled frames twice, state-first and
   question-first, same data. Keep question-first if held-out accuracy is within 2 points.
   P ~0.7. Unlocks ~55 ms streaming.
3. **Token diet** on that checkpoint (terse questions, 1 readout token, 128 patches): ~30 ms.
4. **Student + Box**: ViT-S student (~3 to 5 ms) with a DFL box head on the pre-merge
   ViT grid (12x22), trained on sim hitboxes, Valen as teacher and fallback.
5. Rust `engine/` once the shapes are fixed (sys1rust measured 1.12 to 1.25x over Python MLX).
6. eye benches once the user says "moved".

## Agents

- `bug` (claude-fable-5-1): finished, wrote `research/ENGINE.md`.
- `cat` (gpt-6.1-sol): hit its OpenAI usage limit after `fa62c27`. Don't re-spawn on gpt
  until the limit resets.
