"""Prompt trim: shorter question text, same model, same order. Does Valen still agree?

Every 50 tokens cut is ~9.5 ms (scaling.py). Only exact-accuracy speed lever that needs
no training. Gate: compare TERSE answers to the full-prompt fp16 answers (ref_mlx.jsonl)
on the same frames; also time the forward.

    HF_HUB_OFFLINE=1 .venv/bin/python prompt_trim.py
"""
import json
import statistics

import mlx.core as mx

import valen_ref
from valen_mlx import HERE, SNAPSHOT, Runner, agreement, compile_frame, measure, native

TERSE = {
    "steer": {"type": "choice", "instructions": "Steer which way?",
              "criteria": {"left": "left", "straight": "straight", "right": "right"}},
    "throttle": {"type": "choice", "instructions": "Throttle?",
                 "criteria": {"gas": "gas", "coast": "coast", "brake": "brake"}},
    "on_road": {"type": "noul", "instructions": "Car fully on road?"},
    "risk": {"type": "score", "instructions": "Crash risk within 1 s?",
             "criteria": ["very low", "possible", "likely", "certain"]},
}


def run(questions, frames, runner, compiler):
    valen_ref.QUESTIONS.clear(); valen_ref.QUESTIONS.update(questions)
    rows, ms = [], []
    tokens = None
    for f in frames:
        s = compile_frame(compiler, f)
        tokens = s.inputs["input_ids"].shape[1]
        inp = runner.inputs(s)
        t, h = measure(runner, lambda: runner.forward(inp))
        ms.append(t)
        rows.append({"frame": f.name, "answers": runner.answers(h, s)})
    return rows, tokens, ms


def main():
    module, config, compiler = native(SNAPSHOT)
    runner = Runner(SNAPSHOT, "float16", module, config, compiler)
    frames = sorted((HERE / "synth_frames").glob("*.png"))
    full_q = dict(valen_ref.QUESTIONS)
    run(full_q, frames[:3], runner, compiler)  # warm-up
    full, ftok, fms = run(full_q, frames * 3, runner, compiler)
    terse, ttok, tms = run(TERSE, frames * 3, runner, compiler)
    valen_ref.QUESTIONS.clear(); valen_ref.QUESTIONS.update(full_q)
    ref = [json.loads(l) for l in (HERE / "ref_cpu_fp32.jsonl").read_text().splitlines()]
    full8 = full[:len(frames)]; terse8 = terse[:len(frames)]
    out = {"full_tokens": ftok, "terse_tokens": ttok,
           "full_forward_p50_ms": round(statistics.median(fms), 1),
           "terse_forward_p50_ms": round(statistics.median(tms), 1),
           "full_vs_fp32_ref": agreement(full8, ref)["fraction"],
           "terse_vs_full_same_frames": agreement(terse8, full8)}
    out["terse_vs_full_same_frames"].pop("failures")
    print(json.dumps(out, indent=1))
    for a, b in zip(full8, terse8):
        print(a["frame"], {q: (round(a["answers"][q].get("noul", a["answers"][q].get("score", 0)), 3) if q in ("on_road", "risk") else a["answers"][q]["choice"],
                               round(b["answers"][q].get("noul", b["answers"][q].get("score", 0)), 3) if q in ("on_road", "risk") else b["answers"][q]["choice"]) for q in TERSE})
    (HERE / "prompt_trim.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
