# bakeoff: frozen encoders and a fine-tuned student on lateral + heading (milestone 1a)

Gate: lateral RMS <= 0.2 m on winter1 (a track never seen in training). **Not met.**
Run 2026-10-22 on the M4 Pro, fp16, batch 1.

## Labels (`data/roadframe.mjs`)

The dataset had pose but no road-frame target. `roadframe.mjs` writes `road.jsonl` per episode:
signed lateral offset (m, + = left of centre) and heading error vs route direction (rad).

- **Exact subset** (`exact: true`): car on a narrow straight part (Start, Checkpoint, Finish,
  Straight, Slope). Centreline = the part's own axis, so lateral and heading are exact.
  Frame-to-frame jump p50 3 cm, p99 0.59 m. 1,554 frames in dataset v1.
- Everything else: tile-quantized estimate (5 m cells). Widths come out 20+ m on 10.5 m
  roads, so these rows are 1 to 2 m noisy. **All numbers below use the exact subset only.**

## Frozen features + head (ridge / 1-layer MLP), exact labels

| encoder | dim | p50 ms | p99 ms | winter1 lat RMS (ridge / MLP) | summer1 held (MLP) | winter1 head (MLP) |
| --- | --- | --- | --- | --- | --- | --- |
| predict-the-mean | | | | **2.09** | | |
| student ViT-Ti | 384 | 2.72 | 3.88 | 1.78 / 3.41 | 1.18 | 55 deg |
| student ViT-S | 768 | 4.58 | 6.39 | 2.82 / 3.84 | 1.09 | 113 deg |
| MobileCLIP2-S0 | 512 | 21.69 | 26.80 | 2.15 / 2.55 | 1.05 | 59 deg |
| ModernVBERT | | | | not run: HF download stalled | | |

No frozen encoder beats predicting the mean on winter1. Frozen features don't carry metric
lateral position.

## Student ViT-Ti fine-tuned end to end (`finetune.py`)

| data | train frames | winter1 lat RMS | summer1 held lat RMS | p50 / p99 ms |
| --- | --- | --- | --- | --- |
| v1 exact (1,149) | 1,149 | **1.66 m** (corr 0.58) | **0.33 m** (s17-19: 0.09 to 0.13) | 1.83 / 3.02 |
| v1 + 48 random drives + 2 own laps | ~2,900 | 1.89 m (best epoch 0, then diverges to 3.1) | 0.38 m | 1.90 / 3.22 |

Reading:
- **Same track, unseen episode: 0.09 to 0.13 m on 3 of 4 summer1 seeds.** The gate is
  reachable on a trained track.
- **New track (winter1): 1.7 m, no bias, low correlation.** Winter1's look (blue/white walls,
  glare) is absent from training. More random summer drives made it worse. The fix is
  visual diversity (winter/desert tracks in training, colour augmentation), not more summer.
- For the 21 s Polytrack goal the controller drives summer1 only, so the same-track number is
  the one that matters for the demo. The winter1 gate measures generality, not the demo.

## Commands

```
node data/roadframe.mjs                                      # labels
python bench/bakeoff/bakeoff.py extract student_ti           # per encoder (student_s, mobileclip2_s0)
python bench/bakeoff/bakeoff.py fit                          # table above
python bench/bakeoff/finetune.py --size ti --epochs 40       # fine-tune
```
Deps for the torch encoders: a venv with torch, open_clip_torch, transformers, scikit-learn.

## Next

1. Train with winter1 seeds 1-2 in, hold out 3-5: does 1 seen episode of a track's look close
   the gap? (10 min)
2. Colour/glare augmentation + render winter1/desert1 own-line drives for diversity.
3. Retry ModernVBERT download (`huggingface_hub.snapshot_download('ModernVBERT/modernvbert')`).
