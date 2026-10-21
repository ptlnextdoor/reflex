"""Valen-0.8B reference: answers + latency on this Mac.

The PyTorch answers are the correctness reference every faster runtime must match
(>= 99% agreement, sys1rust's gate). Timing here is the baseline to beat.

    .venv/bin/python valen_ref.py --frames ../../runs/frames --device mps --out ref.jsonl
    .venv/bin/python valen_ref.py --selftest   # no model, checks the request builder

Frames: any directory of PNG/JPG game captures. With no frames dir it renders
synthetic frames so the harness runs before eye/ exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

QUESTIONS = {
    "steer": {"type": "choice", "instructions": "Which way should the car in the image steer to stay on the road?",
              "criteria": {"left": "Steer left", "straight": "Keep straight", "right": "Steer right"}},
    "throttle": {"type": "choice", "instructions": "What should the driver do with the throttle?",
                 "criteria": {"gas": "Accelerate", "coast": "Release both pedals", "brake": "Brake"}},
    "on_road": {"type": "noul", "instructions": "Is the car in the image fully on the road surface?"},
    "risk": {"type": "score", "instructions": "How likely is the car to crash or fall off within one second?",
             "criteria": ["Very unlikely", "Possible", "Likely", "Almost certain"]},
}


def request(frame: Path) -> dict:
    state = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "A frame from the racing game Polytrack, third-person view behind the player's car."},
        {"type": "image_url", "image_url": {"url": str(frame.resolve())}},
    ]}]}
    return {"state": state, "questions": QUESTIONS}


def synthetic_frames(out: Path, n: int = 8) -> list[Path]:
    # ponytail: synthetic road frames until eye/ captures real ones; timing is shape-bound so it's fine for latency.
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        im = Image.new("RGB", (256, 144), (120, 180, 230))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 72, 256, 144], fill=(90, 160, 80))
        shift = (i - n // 2) * 12
        d.polygon([(128 + shift - 10, 72), (128 + shift + 10, 72), (220, 144), (36, 144)], fill=(70, 70, 75))
        d.rectangle([118, 110, 138, 128], fill=(200, 30, 30))
        p = out / f"synth_{i:02d}.png"
        im.save(p)
        paths.append(p)
    return paths


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--model", default="Valen-Team/Valen-0.8B")
    ap.add_argument("--revision", default="c2c481fcb319bd9ae834abc7c6373271710cf24e")  # pinned: remote code runs locally
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("ref.jsonl"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="disable Valen's bf16 backbone autocast (true fp32 reference)")
    a = ap.parse_args()

    if a.selftest:
        p = synthetic_frames(Path("selftest_frames"), 2)[0]
        r = request(p)
        assert set(r["questions"]) == {"steer", "throttle", "on_road", "risk"}
        assert r["state"]["messages"][0]["content"][1]["image_url"]["url"].endswith(".png")
        assert all(q["type"] in {"choice", "noul", "score"} for q in QUESTIONS.values())
        print("selftest ok")
        return

    import torch
    from transformers import AutoModel

    frames = sorted([*a.frames.glob("*.png"), *a.frames.glob("*.jpg")]) if a.frames else synthetic_frames(Path("synth_frames"))
    assert frames, f"no frames in {a.frames}"

    t = time.perf_counter()
    model = AutoModel.from_pretrained(a.model, revision=a.revision, trust_remote_code=True, dtype="auto", attn_implementation="sdpa").to(a.device).eval()
    if a.fp32:
        model.backbone_autocast_dtype = None  # snapshot defaults to bf16 autocast even on CPU
    load_s = time.perf_counter() - t
    sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)

    for f in frames[: a.warmup]:
        model.predict(request(f), execution="shared_state")
    sync()

    times, rows = [], []
    for f in frames:
        for k in range(a.repeats):
            t = time.perf_counter()
            res = model.predict(request(f), execution="shared_state")
            sync()
            times.append((time.perf_counter() - t) * 1000)
        rows.append({"frame": f.name, "answers": res["answers"], "usage": res.get("usage"), "internal_usage": res.get("internal_usage")})

    with a.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"model": a.model, "device": a.device, "load_s": round(load_s, 1), "n": len(times),
               "p50_ms": round(statistics.median(times), 1), "p90_ms": round(pct(times, 0.9), 1),
               "p99_ms": round(pct(times, 0.99), 1), "min_ms": round(min(times), 1),
               "tokens": rows[0]["usage"], "compute_tokens": rows[0]["internal_usage"]}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
