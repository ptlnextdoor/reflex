"""Why is question-first streaming image-blind? Root-cause audit.

Valen's Mixer head scores 4 role vectors per candidate: context, instruction,
candidate, decision, and reads the CANDIDATE row after mixing (heads.py). In the
question-first order the instruction and candidate tokens precede the image, so
under causal attention + recurrent DeltaNet they cannot depend on the frame.
This measures, per role, how much the pooled features change across two frames,
in both orders, and how much the answer moves when only image-aware roles change.

    HF_HUB_OFFLINE=1 .venv/bin/python stream_audit.py
"""
import json

import mlx.core as mx
import numpy as np
import torch

from streaming import reorder
from valen_mlx import HERE, SNAPSHOT, Runner, compile_frame, native

ROLES = ["context", "instruction", "candidate", "decision"]


def role_feats(runner, state):
    h = runner.forward(runner.inputs(state))
    mx.eval(h)
    hidden = torch.from_numpy(np.array(h.astype(mx.float32)))
    return {q.qid: [runner.head.extract_features(hidden, b).float() for b in q.branches] for q in state.questions}, hidden


def rel(a, b):
    return float((a - b).norm() / (a.norm() + 1e-9))


def main():
    module, config, compiler = native(SNAPSHOT)
    runner = Runner(SNAPSHOT, "float16", module, config, compiler)
    frames = sorted((HERE / "synth_frames").glob("*.png"))
    fa, fb = frames[0], frames[-1]
    out = {}
    for order in ("state_first", "question_first"):
        sa, sb = compile_frame(compiler, fa), compile_frame(compiler, fb)
        if order == "question_first":
            sa, sb = reorder(sa, compiler.tokenizer)[0], reorder(sb, compiler.tokenizer)[0]
        A, _ = role_feats(runner, sa)
        B, _ = role_feats(runner, sb)
        per_role = {r: [] for r in ROLES}
        for qid in A:
            for xa, xb in zip(A[qid], B[qid]):
                for i, r in enumerate(ROLES):
                    per_role[r].append(rel(xa[:, i], xb[:, i]))
        out[order] = {r: round(float(np.mean(v)), 4) for r, v in per_role.items()}
        print(order, "relative feature change across frames by role:", out[order], flush=True)

    # Head sensitivity in the TRAINED (state-first) order: score frame A, then swap in
    # frame B's role rows one role at a time and see how far the answer probabilities move.
    sa, sb = compile_frame(compiler, fa), compile_frame(compiler, fb)
    A, _ = role_feats(runner, sa)
    B, _ = role_feats(runner, sb)
    sens = {r: [] for r in ROLES}
    with torch.inference_mode():
        for qid in A:
            for xa, xb in zip(A[qid], B[qid]):
                base = torch.softmax(runner.head.score_features(xa), -1)
                full = torch.softmax(runner.head.score_features(xb), -1)
                span = float((full - base).abs().max()) + 1e-9
                for i, r in enumerate(ROLES):
                    mix = xa.clone()
                    mix[:, i] = xb[:, i]
                    moved = float((torch.softmax(runner.head.score_features(mix), -1) - base).abs().max())
                    sens[r].append(moved / span)
    out["head_sensitivity_state_first"] = {r: round(float(np.mean(v)), 3) for r, v in sens.items()}
    print("state-first: fraction of the A->B answer change recovered by swapping one role:",
          out["head_sensitivity_state_first"], flush=True)
    (HERE / "stream_audit.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
