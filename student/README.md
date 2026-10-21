# student: Reflex-1 vision tower (stage 1)

A small ViT over native 256x144 frames with four heads reading the CLS token:
box (4 edges, direct regression), steer (3-way), throttle (3-way), mask_next
(4 bits). This is the vision tower of Reflex-1 (DESIGN.md "Base model"); stage 2
projects the patch grid into Laya's state sequence and is not built here.

Files:

- `model.py`  - ViT-Ti (5.5M) / ViT-S (21.7M). Exposes the patch token grid
  (`patch_tokens`, `__call__(x, return_tokens=True)`) for stage 2.
- `data.py`   - loads `runs/frames/<episode>/small/*.png` + `labels.jsonl`,
  holds out **by episode** (winter1 + summer1 seeds 17-20).
- `metrics.py`- IoU and edge-error-px (at source 512x288) math.
- `train.py`  - MLX training (AdamW, cosine LR, box = L1 + GIoU, CE + BCE aux).
- `eval.py`   - held-out IoU / edge px / steer / throttle / mask accuracy.
- `bench.py`  - fp16 batch-1 mx.compile latency, median + p99 over 500 frames.
- `selfcheck.py` - one-line proof: loads checkpoint, asserts IoU + ms.
- `RESULTS.md`- the numbers.

Run with an MLX interpreter (Python 3.14 here). Deps: mlx, numpy, Pillow.
Checkpoints land in `runs/student/` (gitignored), not git.

```
python3 student/train.py --size ti --epochs 45 --bs 64
python3 student/eval.py --size ti
python3 student/bench.py --size ti
python3 student/selfcheck.py --size ti
```
