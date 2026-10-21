"""Train the Reflex-1 student with MLX.

    python student/train.py --size ti --epochs 40 --bs 64
    python student/train.py --size s  --epochs 40 --bs 48

Loss: box = L1(normalized) + (1 - GIoU); steer/throttle = cross-entropy;
mask_next = BCE over 4 bits. Box dominates (that is the gated head).

Checkpoints go to runs/student/<size>.safetensors (gitignored). We keep the
best val-IoU checkpoint, not the last.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import data as D
import metrics as Me
import model as M

CKPT_DIR = Path(__file__).resolve().parents[1] / "runs" / "student"


def giou_loss(pred, gt):
    """1 - GIoU on normalized x0y0x1y1 boxes. pred,gt: (B,4). Differentiable."""
    px0, py0, px1, py1 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx0, gy0, gx1, gy1 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    # Keep predicted box well-ordered for area math.
    pxl, pxr = mx.minimum(px0, px1), mx.maximum(px0, px1)
    pyl, pyr = mx.minimum(py0, py1), mx.maximum(py0, py1)
    ix0 = mx.maximum(pxl, gx0)
    iy0 = mx.maximum(pyl, gy0)
    ix1 = mx.minimum(pxr, gx1)
    iy1 = mx.minimum(pyr, gy1)
    iw = mx.maximum(ix1 - ix0, 0.0)
    ih = mx.maximum(iy1 - iy0, 0.0)
    inter = iw * ih
    pa = (pxr - pxl) * (pyr - pyl)
    ga = (gx1 - gx0) * (gy1 - gy0)
    union = pa + ga - inter + 1e-7
    iou = inter / union
    # Smallest enclosing box.
    cx0 = mx.minimum(pxl, gx0)
    cy0 = mx.minimum(pyl, gy0)
    cx1 = mx.maximum(pxr, gx1)
    cy1 = mx.maximum(pyr, gy1)
    carea = (cx1 - cx0) * (cy1 - cy0) + 1e-7
    giou = iou - (carea - union) / carea
    return 1.0 - giou


def loss_fn(model, imgs, box_n, steer, throttle, mask):
    out = model(imgs)
    box = out["box"]
    l1 = mx.abs(box - box_n).mean()
    gl = giou_loss(box, box_n).mean()
    sce = nn.losses.cross_entropy(out["steer"], steer, reduction="mean")
    tce = nn.losses.cross_entropy(out["throttle"], throttle, reduction="mean")
    mbce = nn.losses.binary_cross_entropy(out["mask"], mask, reduction="mean")
    total = 5.0 * l1 + 2.0 * gl + 1.0 * sce + 1.0 * tce + 1.0 * mbce
    return total, (l1, gl, sce, tce, mbce)


def evaluate(model, ds, bs=128):
    model.eval()
    preds, steer_hit, thr_hit, mask_hit = [], 0, 0, 0
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
    bm = Me.box_metrics(pred_norm, gt_src)
    bm["steer_acc"] = steer_hit / ds.n
    bm["throttle_acc"] = thr_hit / ds.n
    bm["mask_next_acc"] = mask_hit / ds.n
    return bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="ti", choices=["ti", "s"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tr = D.Dataset("train")
    va = D.Dataset("val")
    print(f"train {tr.n} / val {va.n} frames; size ViT-{args.size.upper()}")

    model = M.Student(args.size)
    M.init_pos_sincos(model)
    nparams = M.param_count(model)
    print(f"params: {nparams/1e6:.2f}M")

    steps_per_epoch = (tr.n + args.bs - 1) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + np.cos(np.pi * p))

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    ckpt = CKPT_DIR / f"{args.size}.safetensors"
    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(tr.n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, tr.n, args.bs):
            idx = order[s:s + args.bs]
            imgs, box_n, _, st, th, mk = tr.batch(idx)
            opt.learning_rate = lr_at(step)
            (total, parts), grads = loss_and_grad(
                model, mx.array(imgs), mx.array(box_n), mx.array(st),
                mx.array(th), mx.array(mk))
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, total)
            ep_loss += float(total)
            nb += 1
            step += 1
        bm = evaluate(model, va)
        history.append({"epoch": epoch, "loss": ep_loss / nb, **bm})
        dt = time.time() - t0
        print(f"[{epoch:02d}] loss {ep_loss/nb:.3f} | val IoU {bm['iou_mean']:.3f} "
              f"(med {bm['iou_med']:.3f}, >.7 {bm['iou_at70']*100:.0f}%) | "
              f"edge {bm['edge_mean']:.2f}px (med {bm['edge_med']:.2f}) | "
              f"steer {bm['steer_acc']*100:.1f} thr {bm['throttle_acc']*100:.1f} "
              f"mask {bm['mask_next_acc']*100:.1f} | {dt:.0f}s", flush=True)
        prog = {"current": epoch + 1, "total": args.epochs, "unit": "epochs",
                "message": f"IoU {bm['iou_mean']:.3f}"}
        print("JCODE_PROGRESS " + json.dumps(prog), flush=True)
        if bm["iou_mean"] > best_iou:
            best_iou = bm["iou_mean"]
            flat = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(ckpt), flat,
                                metadata={"size": args.size,
                                          "params": str(nparams),
                                          "val_iou": f"{bm['iou_mean']:.4f}"})
    (CKPT_DIR / f"{args.size}_history.json").write_text(json.dumps(history, indent=2))
    print(f"best val IoU {best_iou:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
