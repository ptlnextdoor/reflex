"""Latency bench for the student backbone + heads.

fp16, batch 1, native 256x144, mx.compile, median and p99 over 500 frames
after a warmup, measured on this M4 Pro.

    python student/bench.py --size ti
    python student/bench.py --size s --ckpt runs/student/s.safetensors

ponytail: we time the full forward (patch conv -> blocks -> 4 heads) with the
array materialized via mx.eval each iteration, which is the honest per-frame
latency the eye would see. Weights are cast to fp16; inputs are fp16.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map, tree_unflatten

import model as M

ROOT = Path(__file__).resolve().parents[1]


def build(size, ckpt):
    model = M.Student(size)
    if ckpt and Path(ckpt).exists():
        model.update(tree_unflatten(list(mx.load(str(ckpt)).items())))
    # Cast all params to fp16.
    model.update(tree_map(lambda a: a.astype(mx.float16), model.parameters()))
    mx.eval(model.parameters())
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="ti", choices=["ti", "s"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    ckpt = args.ckpt or (ROOT / "runs" / "student" / f"{args.size}.safetensors")

    model = build(args.size, ckpt)
    nparams = M.param_count(model)

    @mx.compile
    def fwd(x):
        out = model(x)
        return out["box"], out["steer"], out["throttle"], out["mask"]

    rng = np.random.default_rng(0)
    # A fixed bank of random frames so compile sees one static shape.
    bank = mx.array(rng.standard_normal((args.frames, M.IMG_H, M.IMG_W, 3)).astype(np.float16))

    # Warmup (triggers compile + kernel cache).
    for i in range(args.warmup):
        o = fwd(bank[i % args.frames: i % args.frames + 1])
        mx.eval(o)

    times = []
    for i in range(args.frames):
        x = bank[i:i + 1]
        mx.eval(x)
        t0 = time.perf_counter()
        o = fwd(x)
        mx.eval(o)
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"ViT-{args.size.upper()} fp16 batch1 256x144 mx.compile, {args.frames} frames "
          f"(warmup {args.warmup}), {nparams/1e6:.2f}M params")
    print(f"  median {np.median(t):.3f} ms  p99 {np.percentile(t, 99):.3f} ms  "
          f"mean {t.mean():.3f} ms  min {t.min():.3f} ms")


if __name__ == "__main__":
    main()
