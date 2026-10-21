"""Is the frame information even present in question-first role features?

Linear-probe ceiling: ridge-regress the teacher's answer (steer probs, on_road, risk
score) from question-first features of each role, on the 400 cached frames, 80/20
split. If R^2 is low for every role, no head can fix streaming: the information never
reaches tokens the head can read, so candidates must follow the image.

    HF_HUB_OFFLINE=1 .venv/bin/python probe_roles.py
"""
import json
import random

import numpy as np
import torch

from valen_mlx import HERE

ROLES = ["context", "instruction", "candidate", "decision"]


def target(r):
    t = r["teacher"]
    if r["kind"] == "noul":
        return float(t[r["keys"].index("true")])
    if r["kind"] == "score":
        return float((t * torch.arange(len(t))).sum())
    return float(t.max())  # choice: confidence of the teacher's pick, varies with the frame


def r2(Xtr, ytr, Xte, yte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ w + ytr.mean()
    return 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean()) ** 2).sum() + 1e-12)


def main():
    data = torch.load(HERE / ".venv/distill_400_7.pt", weights_only=False)
    idx = list(range(len(data))); random.Random(0).shuffle(idx)
    cut = int(0.8 * len(idx))
    out = {}
    for qid in ["steer", "on_road", "risk"]:
        y = np.array([target(data[i][qid]) for i in idx])
        out[qid] = {"teacher_std": round(float(y.std()), 4)}
        for ri, role in enumerate(ROLES):
            X = np.stack([data[i][qid]["feat"][0, ri].numpy() for i in idx])
            # PCA to 64 dims on train only, so ridge is well-posed with 320 samples.
            Xc = X - X[:cut].mean(0)
            _, _, vt = np.linalg.svd(Xc[:cut], full_matrices=False)
            Z = Xc @ vt[:64].T
            out[qid][role] = round(float(r2(Z[:cut], y[:cut], Z[cut:], y[cut:])), 3)
        print(qid, out[qid], flush=True)
    (HERE / "probe_roles.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
