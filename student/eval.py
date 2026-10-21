"""Evaluate a trained student checkpoint on the held-out episodes.

    python student/eval.py --size ti
    python student/eval.py --size s --ckpt runs/student/s.safetensors

Reports held-out box IoU (mean/median, % > 0.7), edge error px at source
resolution, and steer / throttle / mask_next accuracy.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

import data as D
import metrics as Me
import model as M

ROOT = Path(__file__).resolve().parents[1]


def load_model(size, ckpt):
    model = M.Student(size)
    weights = list(mx.load(str(ckpt)).items())
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    model.eval()
    return model


def run_eval(model, ds, bs=128):
    preds, steer_hit, thr_hit, mask_hit = [], 0, 0, 0
    per_edge = []
    gt_src = ds.t["box_src"]
    for s in range(0, ds.n, bs):
        idx = np.arange(s, min(s + bs, ds.n))
        imgs, _, _, st, th, mk = ds.batch(idx)
        out = model(mx.array(imgs))
        mx.eval(out)
        preds.append(np.array(out["box"]))
        steer_hit += int((np.array(out["steer"]).argmax(1) == st).sum())
        thr_hit += int((np.array(out["throttle"]).argmax(1) == th).sum())
        mbits = (np.array(out["mask"]) > 0.0).astype(np.float32)
        mask_hit += int((mbits == mk).all(axis=1).sum())
    pred_norm = np.concatenate(preds, axis=0)
    pred_src = Me.denorm_box(pred_norm)
    bm = Me.box_metrics(pred_norm, gt_src)
    bm["steer_acc"] = steer_hit / ds.n
    bm["throttle_acc"] = thr_hit / ds.n
    bm["mask_next_acc"] = mask_hit / ds.n
    # Per-edge px error (x0,y0,x1,y1).
    pe = np.abs(pred_src - gt_src).mean(axis=0)
    bm["edge_px_x0y0x1y1"] = [round(float(x), 2) for x in pe]
    bm["n"] = ds.n
    return bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="ti", choices=["ti", "s"])
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / "runs" / "student" / f"{args.size}.safetensors"
    model = load_model(args.size, ckpt)
    va = D.Dataset("val")
    bm = run_eval(model, va)
    print(f"ViT-{args.size.upper()}  params {M.param_count(model)/1e6:.2f}M  "
          f"val frames {bm['n']} ({len(va.episodes)} held-out episodes)")
    print(f"  box   IoU mean {bm['iou_mean']:.3f}  median {bm['iou_med']:.3f}  "
          f">0.7 {bm['iou_at70']*100:.1f}%")
    print(f"  edge  mean {bm['edge_mean']:.2f}px  median {bm['edge_med']:.2f}px  "
          f"p99 {bm['edge_p99']:.2f}px  per-edge {bm['edge_px_x0y0x1y1']}")
    print(f"  steer {bm['steer_acc']*100:.1f}%  throttle {bm['throttle_acc']*100:.1f}%  "
          f"mask_next {bm['mask_next_acc']*100:.1f}%")
    print(json.dumps(bm))


if __name__ == "__main__":
    main()
