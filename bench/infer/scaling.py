"""How does Valen's MLX text tower scale with token count?

If time is ~linear in tokens, cutting per-frame tokens (question-first streaming,
fewer visual tokens) pays off proportionally. If there's a big fixed floor (kernel
launches, DeltaNet scan setup), token cuts buy little and the fix is kernels or a
smaller model. Timing only, not answers: random embeddings through the real weights.

    HF_HUB_OFFLINE=1 .venv/bin/python scaling.py
"""
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.qwen3_5.config import ModelConfig
from mlx_vlm.models.qwen3_5.qwen3_5 import Model

SNAP = next((Path.home() / ".cache/huggingface/hub/models--Valen-Team--Valen-0.8B/snapshots").glob("c2c481f*"))


def load():
    m = Model(ModelConfig.from_dict(json.loads((SNAP / "config.json").read_text())))
    w = mx.load(str(SNAP / "model.safetensors"))
    bb = m.sanitize({k.replace("backbone.", "model.", 1): v for k, v in w.items() if k.startswith("backbone.")})
    m.load_weights([(k, v if k.endswith("A_log") else v.astype(mx.float16)) for k, v in bb.items()], strict=True)
    m.eval()
    mx.eval(m.parameters())
    return m.language_model.model


def time_tokens(tower, n, reps=15, warm=3):
    emb = (mx.random.normal((1, n, 1024)) * 0.02).astype(mx.float16)
    ids = mx.zeros((1, n), dtype=mx.int32)
    pos = mx.broadcast_to(mx.arange(n)[None, None, :], (3, 1, n))
    ts = []
    for i in range(warm + reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(tower(ids, inputs_embeds=emb, position_ids=pos))
        mx.synchronize()
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), max(ts)


if __name__ == "__main__":
    tower = load()
    rows = []
    for n in (8, 16, 32, 64, 96, 128, 192, 302, 512):
        p50, mx_ = time_tokens(tower, n)
        rows.append((n, p50))
        print(f"tokens={n:4d}  p50={p50:7.2f} ms  max={mx_:7.2f} ms  ms/token={p50 / n:6.3f}", flush=True)
    # least-squares fit t = a + b*n
    xs, ys = [r[0] for r in rows], [r[1] for r in rows]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - xm) * (y - ym) for x, y in rows) / sum((x - xm) ** 2 for x in xs)
    a = ym - b * xm
    print(f"fit: t = {a:.2f} ms fixed + {b:.4f} ms/token")
