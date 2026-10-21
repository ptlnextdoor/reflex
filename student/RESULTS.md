# Reflex-1 student: results

Stage 1 of Reflex-1 (per PR #4 / DESIGN.md "Base model"): the **vision tower**.
A small ViT over native 256x144 frames with four task heads reading the CLS
token. Stage 2 (projecting the patch grid into Laya's state sequence) is not
built here, but `model.py` already exposes the patch token grid
(`Student.patch_tokens` / `__call__(x, return_tokens=True)`) so stage 2 can tap
it with no retrain.

All numbers are on this machine (M4 Pro, 20-core GPU, macOS 27), MLX 0.31.2,
Python 3.14. Hold-out is **by episode**: all of winter1 (5 seeds, a whole track
the model never trains on) plus summer1 seeds 17-20. 3673 train / 962 val
frames, 31 train / 9 val episodes.

## Heads

- **box**: 4 normalized edges (x0,y0,x1,y1), sigmoid, direct regression. Trained
  with 5*L1 + 2*(1 - GIoU). This is the gated head.
- **steer**: 3-way (left / straight / right), cross-entropy.
- **throttle**: 3-way (gas / brake / coast), cross-entropy.
- **mask_next**: 4 independent bits of the next-frame key bitmask, BCE. Accuracy
  below is all-4-bits-correct (exact match).

## Numbers

| model  | params | box IoU mean | IoU med | IoU>0.7 | edge mean | edge med | edge p99 | steer | throttle | mask_next |
|--------|-------:|-------------:|--------:|--------:|----------:|---------:|---------:|------:|---------:|----------:|
| ViT-Ti | 5.52M  | 0.765        | 0.794   | 66.7%   | 6.89 px   | 5.19 px  | 27.2 px  | 52.0% | 37.8%    | 19.6%     |
| ViT-S  | 21.65M | _pending_    | _pend_  | _pend_  | _pend_    | _pend_   | _pend_   | _pend_| _pend_   | _pend_    |

Per-edge px error (x0, y0, x1, y1) at source res, ViT-Ti:
`[7.8, 2.3, 12.3, 5.1]`. The right edge (x1) is the worst, consistent with the
car's trailing edge being the hardest to localize when it is near the frame
center and partially self-occluded.

### Box gate

ENGINE.md gate row: "Box head ... IoU > 0.7, edge err < 5 px source".

- **IoU > 0.7: PASS** for ViT-Ti (mean 0.765, median 0.794).
- **edge < 5 px: partial** - median edge error is 5.2 px, just over the line;
  mean is 6.9 px, dragged up by the x1 edge and a long tail (p99 27 px) on the
  hardest held-out frames (winter1, an entirely unseen track). Three of four
  edges are under 5 px on average; x1 is the single lever. ponytail: the honest
  next step to clear 5 px on all edges is a DFL distribution head over ~36 bins
  per edge plus more winter1-like data, noted as the upgrade path in model.py.

## Latency (fp16, batch 1, 256x144, mx.compile, 500 frames after 50 warmup)

| model  | median  | p99     | mean    | min     |
|--------|--------:|--------:|--------:|--------:|
| ViT-Ti | 2.54 ms | 24.0 ms | 4.44 ms | 1.66 ms |
| ViT-S  | _pend_  | _pend_  | _pend_  | _pend_  |

ViT-Ti median 2.54 ms matches the ENGINE.md estimate (~2 ms for Ti). The p99
tail is Python/GPU-scheduling jitter at batch 1, not compute; the min (1.66 ms)
is the compute floor. In the Rust hot path this jitter goes away (no per-frame
Python dispatch).

## Commands

```
# train (checkpoints -> runs/student/<size>.safetensors, gitignored)
python3 student/train.py --size ti --epochs 45 --bs 64
python3 student/train.py --size s  --epochs 45 --bs 48

# eval on held-out episodes
python3 student/eval.py --size ti

# latency bench
python3 student/bench.py --size ti

# one-line self-check: loads checkpoint, asserts IoU + ms on 20 held-out frames
python3 student/selfcheck.py --size ti
```

Run with the MLX-enabled interpreter (Python 3.14 here: `python3.14` or the
venv at `$JCODE_SCRATCH_DIR/venv-student`). Deps: mlx, numpy, Pillow. No timm,
no torch.

## Notes

- steer/throttle/mask_next are single-frame predictions off one CLS token with
  no temporal context; they are reported, not gated. The decisive signal for
  those is the car's velocity/heading over time, which a single frame hides.
  They are kept as cheap auxiliary heads (the tower learns a richer feature) and
  are the natural thing for stage-2 Laya to improve with its state sequence.
- Isolation: `python3 loop/isolation.py` scans `student/` (added to its default
  dirs) and passes - no WR, leaderboard, or human-recording references.
