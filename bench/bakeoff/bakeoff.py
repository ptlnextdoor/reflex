"""Encoder bake-off, frozen features -> lateral (m) + heading (rad) head.

Same protocol for every encoder: extract one feature vector per frame once, then fit
the same two heads (ridge, 2-layer MLP) on train episodes, score on held-out episodes.
Held out by EPISODE exactly as student/data.py: all of winter1 + summer1 s17..s20.
Gate (DESIGN.md milestone 1a): lateral RMS <= 0.2 m on winter1.

    python bench/bakeoff/bakeoff.py extract student_ti|student_s|mobileclip2_s0|modernvbert
    python bench/bakeoff/bakeoff.py fit        # all extracted encoders -> RESULTS table
Features land in runs/bakeoff/<enc>.npz (gitignored). Labels: runs/frames/*/road.jsonl
(data/roadframe.mjs). Frames: runs/frames/<ep>/<k>.png (512x288) and small/ (256x144).
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FR = ROOT / "runs" / "frames"
OUT = ROOT / "runs" / "bakeoff"
OUT.mkdir(parents=True, exist_ok=True)
HELD = {f"winter1_s{i}" for i in range(1, 6)} | {f"summer1_s{i}" for i in (17, 18, 19, 20)}


def rows():
    out = []
    for ep in sorted(p.name for p in FR.iterdir() if (p / "road.jsonl").exists()):
        road = [json.loads(l) for l in (FR / ep / "road.jsonl").read_text().splitlines() if l.strip()]
        for r in road:
            big, small = FR / ep / r["image"], FR / ep / "small" / r["image"]
            if big.exists() and small.exists():
                out.append(dict(ep=ep, big=str(big), small=str(small), **r))
    return out


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def lat_stats(fn, imgs, n=200):
    """batch-1 latency, ms: (p50, p99), after warmup."""
    for im in imgs[:10]:
        fn([im])
    t = []
    for im in imgs[:n]:
        s = time.perf_counter(); fn([im]); t.append((time.perf_counter() - s) * 1e3)
    return float(np.percentile(t, 50)), float(np.percentile(t, 99))


# ---------------------------------------------------------------- encoders
def enc_student(size):
    sys.path.insert(0, str(Path.home() / ".jcode/scratch/wt-student/student"))
    import mlx.core as mx
    from mlx.utils import tree_map
    import model as M
    m = M.Student(size)
    m.load_weights(str(ROOT / "runs" / "student" / f"{size}.safetensors"))
    m.update(tree_map(lambda a: a.astype(mx.float16), m.parameters()))
    m.eval()
    MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

    def prep(p):
        a = np.asarray(Image.open(p).convert("RGB").resize((M.IMG_W, M.IMG_H)), np.float32) / 255.0
        return (a - MEAN) / STD

    @mx.compile
    def feat(x):
        seq = m.backbone(x)
        return mx.concatenate([seq[:, 0], seq[:, 1:].mean(axis=1)], axis=-1)

    def fn(paths):
        x = mx.array(np.stack([prep(p) for p in paths])).astype(mx.float16)
        y = feat(x); mx.eval(y)
        return np.array(y.astype(mx.float32))
    return fn, "small"


def enc_mobileclip2():
    import torch, open_clip
    m, _, pre = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    dev = "mps"; m = m.visual.to(dev).eval().half()

    @torch.no_grad()
    def fn(paths):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths]).to(dev).half()
        y = m(x).float()
        torch.mps.synchronize()
        return y.cpu().numpy()
    return fn, "big"


def enc_modernvbert():
    import torch
    from transformers import AutoModel, AutoProcessor
    rid = "ModernVBERT/modernvbert"
    proc = AutoProcessor.from_pretrained(rid, trust_remote_code=True)
    full = AutoModel.from_pretrained(rid, trust_remote_code=True, torch_dtype=torch.float16)
    vis = getattr(full, "vision_model", None) or full.model.vision_model
    dev = "mps"; vis = vis.to(dev).eval()
    ip = proc.image_processor

    @torch.no_grad()
    def fn(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        px = ip(ims, return_tensors="pt", do_image_splitting=False)["pixel_values"]
        px = px.reshape(-1, *px.shape[-3:]).to(dev, torch.float16)
        h = vis(pixel_values=px).last_hidden_state.float()
        torch.mps.synchronize()
        return h.mean(dim=1).cpu().numpy()
    return fn, "big"


ENC = {
    "student_ti": lambda: enc_student("ti"),
    "student_s": lambda: enc_student("s"),
    "mobileclip2_s0": enc_mobileclip2,
    "modernvbert": enc_modernvbert,
}


def extract(name):
    R = rows()
    fn, res = ENC[name]()
    paths = [r[res] for r in R]
    t0 = time.time(); F = []
    for i, b in enumerate(batched(paths, 32)):
        F.append(fn(b))
        if i % 20 == 0:
            print(f"JCODE_PROGRESS {json.dumps({'current': i * 32, 'total': len(paths), 'message': name})}", flush=True)
    F = np.concatenate(F)
    p50, p99 = lat_stats(fn, paths)
    np.savez(OUT / f"{name}.npz", F=F, eps=np.array([r["ep"] for r in R]),
             valid=np.array([r["valid"] for r in R]), lat=np.array([r["lateral"] for r in R], np.float32),
             head=np.array([r["heading"] for r in R], np.float32), width=np.array([r.get("width", 0) for r in R], np.float32),
             p50=p50, p99=p99, secs=time.time() - t0)
    print(f"{name}: feats {F.shape} extract {time.time() - t0:.0f}s batch-1 p50 {p50:.2f} ms p99 {p99:.2f} ms")


# ---------------------------------------------------------------- heads
def _labels():
    R = rows()
    return (np.array([bool(r.get("exact")) for r in R]), np.array([r["lateral"] for r in R], np.float32),
            np.array([r["heading"] for r in R], np.float32))


def fit():
    global EXACT, LAB
    ex, lat, head = _labels()
    EXACT, LAB = {}, {}
    for f in OUT.glob("*.npz"):
        EXACT[f.stem] = ex; LAB[f.stem] = (lat, head)
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    res = []
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        ex = np.array(EXACT[f.stem]) if f.stem in EXACT else None
        v = ex if ex is not None else d["valid"]
        lat_all, head_all = LAB[f.stem]
        tr = v & ~np.isin(d["eps"], list(HELD))
        w1 = v & np.char.startswith(d["eps"].astype(str), "winter1")
        hs1 = v & np.isin(d["eps"], [f"summer1_s{i}" for i in (17, 18, 19, 20)])
        # Precise-label subset: narrow road (10 m between kerbs) and car aligned with it, where the
        # tile-quantized label is exact to ~0.25 m. Turns and wide parts carry 1-2 m label noise.
        ns = v & np.isin(d["eps"], list(HELD)) & (np.abs(head_all) < 0.35)
        sc = StandardScaler().fit(d["F"][tr]); X = sc.transform(d["F"])
        Y = np.stack([lat_all, np.sin(head_all), np.cos(head_all)], 1)
        row = dict(enc=f.stem, dim=d["F"].shape[1], n_train=int(tr.sum()), p50=float(d["p50"]), p99=float(d["p99"]))
        for hname, mk in [("ridge", lambda: RidgeCV(alphas=np.logspace(-1, 4, 12))),
                          ("mlp", lambda: MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0))]:
            m = mk().fit(X[tr], Y[tr]); P = m.predict(X)
            for split, msk in [("winter1", w1), ("summer1_held", hs1), ("held_narrow_straight", ns)]:
                lat_rms = float(np.sqrt(np.mean((P[msk, 0] - lat_all[msk]) ** 2)))
                hd = np.degrees(np.arctan2(P[msk, 1], P[msk, 2]) - head_all[msk])
                hd = (hd + 180) % 360 - 180
                row[f"{hname}_{split}_lat_rms_m"] = lat_rms
                row[f"{hname}_{split}_head_rms_deg"] = float(np.sqrt(np.mean(hd ** 2)))
            row[f"{hname}_n_w1"] = int(w1.sum()); row["n_narrow"] = int(ns.sum())
        base = float(np.sqrt(np.mean((lat_all[w1] - lat_all[tr].mean()) ** 2)))
        row["winter1_predict_mean_rms_m"] = base
        res.append(row)
        print(json.dumps(row))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"extract": lambda: extract(sys.argv[2]), "fit": fit}[sys.argv[1]]()
