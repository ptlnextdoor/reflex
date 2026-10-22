"""Fine-tune the student ViT end to end on lateral (m) + heading for milestone 1a.

Frozen features miss the 0.2 m gate by 10x (RESULTS.md table), so this trains the whole
tower with a 3-output head [lateral, sin heading, cos heading] on the EXACT labels only
(narrow straight parts, data/roadframe.mjs). Held out by episode: winter1 + summer1 s17..20.

    python bench/bakeoff/finetune.py --size ti --epochs 60
Writes runs/bakeoff/ft_<size>.json (history + final held-out RMS) and ft_<size>.safetensors.
Augmentation: horizontal flip negates lateral and heading (road frame is mirror-symmetric);
small brightness jitter. No crop: lateral is read from absolute image position.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / ".jcode/scratch/wt-student/student"))
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FR = ROOT / "runs" / "frames"
OUT = ROOT / "runs" / "bakeoff"
HELD = {f"winter1_s{i}" for i in range(1, 6)} | {f"summer1_s{i}" for i in (17, 18, 19, 20)}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load():
    X, Y, E = [], [], []
    for ep in sorted(p.name for p in FR.iterdir() if (p / "road.jsonl").exists()):
        for l in (FR / ep / "road.jsonl").read_text().splitlines():
            r = json.loads(l) if l.strip() else None
            if not r or not r.get("exact"):
                continue
            p = FR / ep / "small" / r["image"]
            if not p.exists():
                continue
            a = np.asarray(Image.open(p).convert("RGB").resize((M.IMG_W, M.IMG_H)), np.float32) / 255.0
            X.append((a - MEAN) / STD)
            Y.append([r["lateral"], np.sin(r["heading"]), np.cos(r["heading"])])
            E.append(ep)
    return np.stack(X), np.array(Y, np.float32), np.array(E)


class Reg(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.s = M.Student(size)
        self.head = nn.Sequential(nn.Linear(self.s.width * 2, 256), nn.GELU(), nn.Linear(256, 3))

    def __call__(self, x):
        seq = self.s.backbone(x)
        return self.head(mx.concatenate([seq[:, 0], seq[:, 1:].mean(axis=1)], axis=-1))


def metrics(pred, y, e):
    out = {}
    for name, msk in [("winter1", np.char.startswith(e.astype(str), "winter1")),
                      ("summer1_held", np.isin(e, [f"summer1_s{i}" for i in (17, 18, 19, 20)]))]:
        if msk.sum() == 0:
            continue
        lat = float(np.sqrt(np.mean((pred[msk, 0] - y[msk, 0]) ** 2)))
        hd = np.degrees(np.arctan2(pred[msk, 1], pred[msk, 2]) - np.arctan2(y[msk, 1], y[msk, 2]))
        hd = (hd + 180) % 360 - 180
        out[f"{name}_lat_rms_m"] = lat
        out[f"{name}_lat_mae_m"] = float(np.mean(np.abs(pred[msk, 0] - y[msk, 0])))
        out[f"{name}_head_rms_deg"] = float(np.sqrt(np.mean(hd ** 2)))
        out[f"{name}_n"] = int(msk.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="ti")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"])
    a = ap.parse_args()
    t0 = time.time()
    X, Y, E = load()
    held = np.isin(E, list(HELD))
    Xtr, Ytr, Xte, Yte, Ete = X[~held], Y[~held], X[held], Y[held], E[held]
    print(f"train {len(Xtr)} held {len(Xte)} load {time.time() - t0:.0f}s", flush=True)
    m = Reg(a.size)
    if a.init == "pretrained":
        m.s.load_weights(str(ROOT / "runs" / "student" / f"{a.size}.safetensors"), strict=False)
    mx.eval(m.parameters())
    steps = a.epochs * max(1, len(Xtr) // a.bs)
    sched = optim.join_schedules([optim.linear_schedule(1e-6, a.lr, 100), optim.cosine_decay(a.lr, steps)], [100])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.05)
    w = mx.array([1.0, 2.0, 2.0])  # lateral in m, heading in unit-circle components

    def loss_fn(model, x, y):
        return (nn.losses.huber_loss(model(x), y, delta=1.0, reduction="none") * w).mean()

    lg = nn.value_and_grad(m, loss_fn)

    @mx.compile
    def predict(x):
        return m(x)

    def evaluate():
        m.eval()
        P = np.concatenate([np.array(m(mx.array(Xte[i:i + 128]))) for i in range(0, len(Xte), 128)])
        m.train()
        return metrics(P, Yte, Ete)

    hist, best = [], None
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        idx = rng.permutation(len(Xtr)); tl = 0.0; nb = 0
        for i in range(0, len(idx) - a.bs + 1, a.bs):
            b = idx[i:i + a.bs]
            x = Xtr[b].copy(); y = Ytr[b].copy()
            flip = rng.random(len(b)) < 0.5
            x[flip] = x[flip, :, ::-1]; y[flip, 0] *= -1; y[flip, 1] *= -1
            x *= rng.uniform(0.85, 1.15, (len(b), 1, 1, 1)).astype(np.float32)
            loss, g = lg(m, mx.array(x), mx.array(y))
            opt.update(m, g); mx.eval(m.parameters(), opt.state)
            tl += float(loss); nb += 1
        r = {"epoch": ep, "loss": tl / max(nb, 1), **evaluate(), "wall_s": time.time() - t0}
        hist.append(r)
        if best is None or r["winter1_lat_rms_m"] < best["winter1_lat_rms_m"]:
            best = r
            m.save_weights(str(OUT / f"ft_{a.size}_{a.init}.safetensors"))
        print(f"JCODE_PROGRESS {json.dumps({'current': ep + 1, 'total': a.epochs, 'unit': 'epochs', 'message': f'w1 lat {r['winter1_lat_rms_m']:.2f} m'})}", flush=True)
        print(f"[{ep:02d}] loss {r['loss']:.3f} w1 lat RMS {r['winter1_lat_rms_m']:.3f} m head {r['winter1_head_rms_deg']:.1f} deg | s1 lat {r.get('summer1_held_lat_rms_m', 0):.3f}", flush=True)
    # latency of the full regressor, fp16, batch 1
    from mlx.utils import tree_map
    m.load_weights(str(OUT / f"ft_{a.size}_{a.init}.safetensors"))
    m.update(tree_map(lambda p: p.astype(mx.float16), m.parameters())); m.eval()
    x1 = mx.array(Xte[:1]).astype(mx.float16)
    for _ in range(20): mx.eval(predict(x1))
    ts = []
    for _ in range(300):
        s = time.perf_counter(); mx.eval(predict(x1)); ts.append((time.perf_counter() - s) * 1e3)
    out = {"size": a.size, "init": a.init, "best": best, "p50_ms": float(np.percentile(ts, 50)),
           "p99_ms": float(np.percentile(ts, 99)), "history": hist, "n_train": int(len(Xtr)), "n_held": int(len(Xte))}
    (OUT / f"ft_{a.size}_{a.init}.json").write_text(json.dumps(out, indent=1))
    print(f"DONE {a.size}/{a.init}: best epoch {best['epoch']} winter1 lat RMS {best['winter1_lat_rms_m']:.3f} m "
          f"head {best['winter1_head_rms_deg']:.1f} deg; p50 {out['p50_ms']:.2f} p99 {out['p99_ms']:.2f} ms; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
