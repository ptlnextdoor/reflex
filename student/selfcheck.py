"""One-line self-check: load the checkpoint, assert IoU and latency on 20
held-out frames.

    python3 student/selfcheck.py            # ViT-Ti, default thresholds
    python3 student/selfcheck.py --size s

Exits non-zero (assert) if the checkpoint regresses below the gate. This is the
runnable proof the repo asks every change to leave behind.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map, tree_unflatten

import data as D
import metrics as Me
import model as M

ROOT = Path(__file__).resolve().parents[1]

# Gate: box IoU > 0.7 (ENGINE.md), and the student must be fast (student path is
# 3 to 5 ms for S, ~2 ms for Ti; allow generous headroom so the check is about
# "did the model load and run fast", not a flaky micro-bench).
GATES = {
    "ti": dict(iou=0.70, ms=6.0),
    "s":  dict(iou=0.70, ms=9.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="ti", choices=["ti", "s"])
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    gate = GATES[args.size]
    ckpt = ROOT / "runs" / "student" / f"{args.size}.safetensors"
    assert ckpt.exists(), f"missing checkpoint {ckpt}; run student/train.py --size {args.size}"

    model = M.Student(args.size)
    model.update(tree_unflatten(list(mx.load(str(ckpt)).items())))
    model.update(tree_map(lambda a: a.astype(mx.float16), model.parameters()))
    mx.eval(model.parameters())
    model.eval()

    # 20 held-out frames, evenly spaced across the val episodes.
    va = D.Dataset("val")
    idx = np.linspace(0, va.n - 1, args.n).astype(int)
    imgs, _, box_src, _, _, _ = va.batch(idx)
    x = mx.array(imgs.astype(np.float16))

    out = model(x)
    mx.eval(out)
    pred_norm = np.array(out["box"]).astype(np.float32)
    bm = Me.box_metrics(pred_norm, box_src)

    # Latency: median over 100 single-frame forwards after warmup.
    @mx.compile
    def fwd(f):
        o = model(f)
        return o["box"], o["steer"], o["throttle"], o["mask"]

    one = x[:1]
    for _ in range(20):
        mx.eval(fwd(one))
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        mx.eval(fwd(one))
        ts.append((time.perf_counter() - t0) * 1000.0)
    ms = float(np.median(ts))

    print(f"selfcheck ViT-{args.size.upper()}: IoU {bm['iou_mean']:.3f} "
          f"(>0.7 {bm['iou_at70']*100:.0f}%), edge {bm['edge_mean']:.2f}px, "
          f"{ms:.3f} ms/frame on {args.n} held-out frames")
    assert bm["iou_mean"] > gate["iou"], f"IoU {bm['iou_mean']:.3f} <= {gate['iou']}"
    assert ms < gate["ms"], f"latency {ms:.3f} ms >= {gate['ms']} ms"
    print("OK")


if __name__ == "__main__":
    main()
