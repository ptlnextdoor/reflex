"""Streaming head distillation: can a head reading only frame-aware roles recover Valen?

Root cause (stream_audit.py): Valen's Mixer reads the frame ~99% through the candidate
role, which question-first streaming freezes. Fix under test: keep the 0.8B backbone
frozen, run it question-first ONCE per frame to cache role features, then train a
new head on CPU to match the state-first teacher's probabilities. Only the head learns.

    HF_HUB_OFFLINE=1 .venv/bin/python distill_head.py --n 400        # features + train + eval

Frames: procedurally varied synthetic road scenes (road offset, curve, car position,
colors). ponytail: synthetic until eye/ captures real Polytrack frames; the question
here is whether the frozen question-first features carry the information at all.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image, ImageDraw

from streaming import Streaming, reorder
from valen_mlx import HERE, SNAPSHOT, Runner, compile_frame, native

QIDS = ["steer", "throttle", "on_road", "risk"]


def make_frames(out: Path, n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        im = Image.new("RGB", (256, 144), tuple(rng.randint(90, 200) for _ in range(3)))
        d = ImageDraw.Draw(im)
        horizon = rng.randint(55, 85)
        d.rectangle([0, horizon, 256, 144], fill=tuple(rng.randint(40, 170) for _ in range(3)))
        vx = 128 + rng.randint(-90, 90)               # vanishing point offset = curve / heading error
        half = rng.randint(50, 110)                   # road width at bottom
        bx = 128 + rng.randint(-60, 60)               # road center at bottom = lateral offset
        road = tuple(rng.randint(50, 95) for _ in range(3))
        d.polygon([(vx - 6, horizon), (vx + 6, horizon), (bx + half, 144), (bx - half, 144)], fill=road)
        cx = 128 + rng.randint(-25, 25)
        d.rectangle([cx - 11, 110, cx + 11, 130], fill=(rng.randint(150, 255), rng.randint(0, 80), rng.randint(0, 80)))
        p = out / f"d{seed}_{i:04d}.png"
        im.save(p)
        paths.append(p)
    return paths


def teacher_and_features(n: int, seed: int, cache: Path):
    if cache.exists():
        return torch.load(cache, weights_only=False)
    frames = make_frames(HERE / "distill_frames", n, seed)
    module, config, compiler = native(SNAPSHOT)
    runner = Runner(SNAPSHOT, "float16", module, config, compiler)
    data = []
    for k, f in enumerate(frames):
        s = compile_frame(compiler, f)
        h = runner.forward(runner.inputs(s)); mx.eval(h)
        teacher_hidden = torch.from_numpy(np.array(h.astype(mx.float32)))
        q = reorder(s, compiler.tokenizer)[0]
        h2 = runner.forward(runner.inputs(q)); mx.eval(h2)
        student_hidden = torch.from_numpy(np.array(h2.astype(mx.float32)))
        rec = {}
        with torch.no_grad():
            for tq, sq in zip(s.questions, q.questions):
                tfeat = [runner.head.extract_features(teacher_hidden, b).float() for b in tq.branches]
                logits = torch.cat([runner.head.score_features(x) for x in tfeat])
                sfeat = torch.cat([runner.head.extract_features(student_hidden, b).float() for b in sq.branches])
                rec[tq.qid] = {"teacher": logits.softmax(-1), "feat": sfeat, "kind": tq.kind, "keys": list(tq.keys)}
        data.append(rec)
        if (k + 1) % 50 == 0:
            print(json.dumps({"features": k + 1, "of": n}), flush=True)
    torch.save(data, cache)
    return data


class StreamHead(torch.nn.Module):
    """Valen's Mixer, retrained, plus a frame path. probe_roles.py showed the frame
    is linearly readable from the context and decision rows (R^2 0.62 to 0.81) and absent
    from instruction and candidate rows (R^2 < 0). So: keep the Mixer for question
    semantics, and add a per-candidate bilinear term between the frame-aware rows
    (context, decision) and the candidate row. The candidate row says WHICH option,
    the frame rows say WHAT the frame shows; the product scores the match."""
    def __init__(self, base, width=256):
        super().__init__()
        import copy
        self.m = copy.deepcopy(base)
        hidden = base.projection.in_features
        self.norm = torch.nn.LayerNorm(hidden)
        self.frame = torch.nn.Linear(2 * hidden, width)
        self.cand = torch.nn.Linear(hidden, width)
        self.scale = width ** -0.5

    def forward(self, feat):
        x = self.m.blocks(self.m.project(feat))
        mixer = self.m.output(x[:, 2]).squeeze(-1)
        f = self.norm(feat.float())
        frame = self.frame(torch.cat([f[:, 0], f[:, 3]], -1))   # context, decision
        return mixer + (frame * self.cand(f[:, 2])).sum(-1) * self.scale


def evaluate(head, data, read):
    agree = total = 0
    per = {q: [0, 0] for q in QIDS}
    with torch.inference_mode():
        for rec in data:
            for qid, r in rec.items():
                p = read(head, r["feat"]).softmax(-1)
                t = r["teacher"]
                if r["kind"] == "choice":
                    ok = int(p.argmax()) == int(t.argmax())
                elif r["kind"] == "noul":
                    i = r["keys"].index("true")
                    ok = abs(float(p[i] - t[i])) <= 0.02
                else:
                    idx = torch.arange(len(p), dtype=torch.float32)
                    ok = abs(float((p * idx).sum() - (t * idx).sum())) <= 0.05
                agree += ok; total += 1
                per[qid][0] += ok; per[qid][1] += 1
    return agree / total, {q: f"{a}/{b}" for q, (a, b) in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    data = teacher_and_features(a.n, a.seed, HERE / f".venv/distill_{a.n}_{a.seed}.pt")
    rng = random.Random(0)
    idx = list(range(len(data))); rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    train, held = [data[i] for i in idx[:cut]], [data[i] for i in idx[cut:]]

    module, config, _ = native(SNAPSHOT)
    from safetensors import safe_open
    base = module.build_head(config.text_config.hidden_size, config.valen_head).float()
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt", device="cpu") as f:
        base.load_state_dict({k.removeprefix("head."): f.get_tensor(k) for k in f.keys() if k.startswith("head.")})
    frozen_read = lambda h, x: h.score_features(x)
    before = evaluate(base, held, frozen_read)
    print(json.dumps({"held_out_agreement_original_head_question_first": before}), flush=True)

    head = StreamHead(base)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(a.epochs):
        rng.shuffle(train)
        tot = 0.0
        for rec in train:
            loss = 0.0
            for r in rec.values():
                logp = head(r["feat"]).log_softmax(-1)
                loss = loss + torch.nn.functional.kl_div(logp, r["teacher"], reduction="sum")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0:
            print(json.dumps({"epoch": ep + 1, "train_kl": round(tot / len(train), 5),
                              "held_out": evaluate(head, held, lambda h, x: h(x))}), flush=True)
    after = evaluate(head, held, lambda h, x: h(x))
    result = {"n_frames": len(data), "train": len(train), "held_out": len(held),
              "before_original_head": before, "after_streaming_head": after}
    (HERE / "distill_head.json").write_text(json.dumps(result, indent=2))
    torch.save(head.state_dict(), HERE / ".venv/stream_head.pt")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
