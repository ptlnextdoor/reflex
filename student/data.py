"""Dataset for the Reflex-1 student.

Loads native 256x144 frames from runs/frames/<episode>/small/*.png and the
per-frame labels from labels.jsonl. Boxes are stored at source resolution
(512x288); we keep the source box for the gate and a normalized [0,1] copy for
the loss.

Hold out is by EPISODE, never by frame: a whole track (winter1, all 5 seeds)
plus four summer1 seeds. This is the only honest split for a driving model, a
frame-level split leaks the same corner into train and val.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

import model as M

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "runs" / "frames"

STEER = {"left": 0, "straight": 1, "right": 2}
THROTTLE = {"gas": 0, "brake": 1, "coast": 2}

# Held-out episodes: all of winter1 (a whole track) + 4 summer1 seeds.
HELD_OUT = (
    [f"winter1_s{i}" for i in range(1, 6)]
    + [f"summer1_s{i}" for i in (17, 18, 19, 20)]
)

# ImageNet-ish standardization keeps the patch conv in a sane range.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_episode(ep):
    rows = []
    lbl = FRAMES / ep / "labels.jsonl"
    for line in lbl.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    items = []
    for r in rows:
        img = FRAMES / ep / "small" / r["image"]
        if not img.exists():
            continue
        items.append((str(img), r))
    return items


def _decode_img(path):
    im = Image.open(path).convert("RGB")
    if im.size != (M.IMG_W, M.IMG_H):
        im = im.resize((M.IMG_W, M.IMG_H), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return a  # H, W, 3


def _targets(rows):
    n = len(rows)
    box_src = np.zeros((n, 4), np.float32)   # source-res px (for the gate)
    box_norm = np.zeros((n, 4), np.float32)  # normalized [0,1] (for the loss)
    steer = np.zeros((n,), np.int32)
    throttle = np.zeros((n,), np.int32)
    mask = np.zeros((n, 4), np.float32)      # 4 bits of mask_next
    for i, r in enumerate(rows):
        b = r["box"]
        sw, sh = r["size"]
        box_src[i] = b
        box_norm[i] = [b[0] / sw, b[1] / sh, b[2] / sw, b[3] / sh]
        steer[i] = STEER[r["steer"]]
        throttle[i] = THROTTLE[r["throttle"]]
        mn = int(r["mask_next"])
        mask[i] = [(mn >> k) & 1 for k in range(4)]
    return dict(box_src=box_src, box_norm=box_norm, steer=steer,
               throttle=throttle, mask=mask)


class Dataset:
    def __init__(self, split="train"):
        eps = sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
        held = set(HELD_OUT)
        if split == "train":
            eps = [e for e in eps if e not in held]
        elif split == "val":
            eps = [e for e in eps if e in held]
        else:
            raise ValueError(split)
        self.split = split
        self.episodes = eps
        self.paths = []
        rows = []
        for ep in eps:
            for path, r in _load_episode(ep):
                self.paths.append(path)
                rows.append(r)
        self.rows = rows
        self.t = _targets(rows)
        self.n = len(rows)
        # Decode lazily but cache: the set is small (<5k, 256x144x3 ~ 42 MB fp32).
        self._cache = {}

    def image(self, i):
        a = self._cache.get(i)
        if a is None:
            a = _decode_img(self.paths[i])
            self._cache[i] = a
        return a

    def batch(self, idx):
        imgs = np.stack([self.image(i) for i in idx]).astype(np.float32)
        return (
            imgs,
            self.t["box_norm"][idx],
            self.t["box_src"][idx],
            self.t["steer"][idx],
            self.t["throttle"][idx],
            self.t["mask"][idx],
        )

    def iter_batches(self, bs, shuffle=True, rng=None):
        order = np.arange(self.n)
        if shuffle:
            (rng or np.random).shuffle(order)
        for s in range(0, self.n, bs):
            yield self.batch(order[s:s + bs])


if __name__ == "__main__":
    tr = Dataset("train")
    va = Dataset("val")
    print(f"train: {tr.n} frames, {len(tr.episodes)} episodes")
    print(f"val:   {va.n} frames, {len(va.episodes)} episodes -> {va.episodes}")
    img, bn, bs, st, th, mk = tr.batch(np.arange(4))
    print("img", img.shape, img.dtype, "range", round(float(img.min()), 2),
          round(float(img.max()), 2))
    print("box_norm[0]", bn[0], "box_src[0]", bs[0])
    print("steer", st, "throttle", th)
    print("mask[0]", mk[0])
