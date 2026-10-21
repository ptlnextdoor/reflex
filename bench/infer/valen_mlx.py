"""Correctness-first MLX bake-off of the pinned Valen shared-state workload.

Run from bench/infer, using .venv/bin/python. Only the backbone uses the GPU.
The unchanged snapshot Compiler, fp32 torch CPU MixerHead and response decoder
preserve Valen's exact tokenization, pooling and answer semantics. No LM head.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from valen_ref import pct, request

REVISION = "c2c481fcb319bd9ae834abc7c6373271710cf24e"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Valen-Team--Valen-0.8B/snapshots" / REVISION
HERE = Path(__file__).resolve().parent


def native(snapshot):
    cls = get_class_from_dynamic_module(
        "modeling_valen.ValenQwenForDecisionMaking", str(snapshot), local_files_only=True
    )
    module = importlib.import_module(cls.__module__)
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    compiler = module.Compiler(processor, max_length=config.valen_max_length,
                               media_kwargs=config.valen_media_kwargs, execution="shared_state")
    return module, config, compiler


def compile_frame(compiler, frame):
    state = compiler.compile({"request": request(frame)})
    assert state.inputs is not None
    assert all(b.inputs is state.inputs for q in state.questions for b in q.branches)
    return state


def decode(module, head, hidden, state):
    # ponytail: the 1.32M-parameter head stays on torch CPU, not a second GPU job.
    with torch.inference_mode():
        hidden = torch.from_numpy(np.array(hidden, dtype=np.float32, copy=True))
        return {q.qid: module.answer(q, torch.cat([
            head.score_features(head.extract_features(hidden, b)) for b in q.branches
        ])) for q in state.questions}


def agreement(rows, reference):
    ref = {r["frame"]: r for r in reference}
    assert len(rows) == len(ref) and {r["frame"] for r in rows} == set(ref)
    passed, total, failures, probability_delta = 0, 0, [], 0.0
    for row in rows:
        expected = ref[row["frame"]]["answers"]
        assert set(row["answers"]) == set(expected)
        for key, actual in row["answers"].items():
            target = expected[key]
            assert actual["type"] == target["type"]
            if actual["type"] == "choice":
                ok = actual["choice"] == target["choice"]
            else:
                field = actual["type"]
                delta = abs(actual[field] - target[field])
                ok = delta <= (0.02 if field == "noul" else 0.05)
            if "probabilities" in actual:
                probability_delta = max(probability_delta, max(
                    abs(p - target["probabilities"][k]) for k, p in actual["probabilities"].items()))
            else:
                probability_delta = max(probability_delta, abs(actual["noul"] - target["noul"]))
            total += 1
            passed += int(ok)
            if not ok:
                failures.append({"frame": row["frame"], "question": key,
                                 "actual": actual, "reference": target})
    return {"passed": passed, "total": total, "fraction": passed / total,
            "max_probability_delta": probability_delta, "failures": failures}


def cpu_reference(snapshot, module, compiler, frame, out):
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True,
                                      dtype="auto", attn_implementation="sdpa").cpu().eval()
    state = compile_frame(compiler, frame)
    with torch.inference_mode():
        # This is the exact valen_ref.py path, including snapshot CPU bf16 autocast.
        hidden = model._hidden(state.inputs).float()
        answers = decode(module, model.head, hidden.numpy(), state)
        arrays = {"reference_hidden": hidden.numpy(), "frame": np.array(frame.name),
                  "reference_answers": np.array(json.dumps(answers))}
        model.backbone_autocast_dtype = None
        arrays["fp32_hidden"] = model._hidden(state.inputs).float().numpy()
        arrays["fp32_answers"] = np.array(json.dumps(decode(
            module, model.head, arrays["fp32_hidden"], state)))
        # Compare position IDs independently of either backend's transformer layers.
        position_ids, _ = model.backbone.get_rope_index(
            state.inputs["input_ids"], image_grid_thw=state.inputs["image_grid_thw"],
            attention_mask=state.inputs["attention_mask"],
            mm_token_type_ids=state.inputs.get("mm_token_type_ids"))
        arrays["position_ids"] = position_ids.numpy()
    np.savez(out, **arrays)
    print(json.dumps({"cpu_hidden_saved": str(out), "shape": list(hidden.shape),
                      "snapshot_cpu_autocast": "bfloat16", "reference_answers": answers,
                      "fp32_answers": json.loads(str(arrays["fp32_answers"]))}), flush=True)
    del model
    gc.collect()


class Runner:
    def __init__(self, snapshot, dtype, module, config, compiler):
        import mlx.core as mx
        from mlx_vlm.models.qwen3_5.config import ModelConfig
        from mlx_vlm.models.qwen3_5.qwen3_5 import Model
        from safetensors import safe_open

        self.mx, self.module, self.compiler = mx, module, compiler
        self.model = Model(ModelConfig.from_dict(json.loads((snapshot / "config.json").read_text())))
        weights = mx.load(str(snapshot / "model.safetensors"))
        # Present full HF names to sanitize so it applies +1 to zero-centered
        # Qwen RMSNorm weights, and transposes depthwise Conv1d and patch Conv3d.
        backbone = {k.replace("backbone.", "model.", 1): v
                    for k, v in weights.items() if k.startswith("backbone.")}
        backbone = self.model.sanitize(backbone)
        dtype = getattr(mx, dtype)
        # A_log remains fp32 in the native Qwen convention (see cast_predicate).
        self.model.load_weights([(k, v if k.endswith("A_log") else v.astype(dtype))
                                 for k, v in backbone.items()], strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.head = module.build_head(config.text_config.hidden_size, config.valen_head).eval()
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as f:
            self.head.load_state_dict({k.removeprefix("head."): f.get_tensor(k)
                                       for k in f.keys() if k.startswith("head.")}, strict=True)
        self.head_params = sum(p.numel() for p in self.head.parameters())
        del weights, backbone
        gc.collect()

    def inputs(self, state):
        return {k: self.mx.array(v.numpy()) for k, v in state.inputs.items()
                if k in {"input_ids", "pixel_values", "image_grid_thw", "attention_mask"}}

    def vit(self, inputs):
        pixels = inputs["pixel_values"].astype(self.model.vision_tower.patch_embed.proj.weight.dtype)
        return self.model.vision_tower(pixels, inputs["image_grid_thw"])[0]

    def embeddings(self, inputs, visual):
        return self.model.get_input_embeddings(
            inputs["input_ids"], pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"], mask=inputs["attention_mask"],
            cached_image_features=visual)

    def text_tower(self, inputs, emb):
        return self.model.language_model.model(
            inputs["input_ids"], inputs_embeds=emb.inputs_embeds, position_ids=emb.position_ids)

    def text(self, inputs, visual):
        return self.text_tower(inputs, self.embeddings(inputs, visual))

    def forward(self, inputs):
        return self.text(inputs, self.vit(inputs))

    def answers(self, hidden, state):
        self.mx.eval(hidden)
        self.mx.synchronize()
        return decode(self.module, self.head, np.array(hidden.astype(self.mx.float32)), state)

    def end_to_end(self, frame):
        state = compile_frame(self.compiler, frame)
        return self.answers(self.forward(self.inputs(state)), state)


def measure(runner, fn):
    mx = runner.mx
    mx.synchronize()
    start = time.perf_counter_ns()
    value = fn()
    if isinstance(value, mx.array):
        mx.eval(value)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, value


def summary(samples):
    return {"n": len(samples), "p50_ms": statistics.median(samples),
            "p90_ms": pct(samples, 0.90), "p99_ms": pct(samples, 0.99), "max_ms": max(samples)}


def selftest():
    row = {"frame": "test.png", "answers": {
        "choice": {"type": "choice", "choice": "left", "probabilities": {"left": 0.7, "right": 0.3}},
        "score": {"type": "score", "score": 0.0, "probabilities": {"0": 1.0}},
        "noul": {"type": "noul", "noul": 0.0}}}
    import copy
    changed = copy.deepcopy(row)
    changed["answers"]["score"]["score"] = 0.05
    changed["answers"]["noul"]["noul"] = 0.02
    assert agreement([changed], [row])["passed"] == 3
    changed["answers"]["score"]["score"] = 0.05001
    changed["answers"]["noul"]["noul"] = 0.02001
    changed["answers"]["choice"]["choice"] = "right"
    assert agreement([changed], [row])["passed"] == 0
    assert summary([1.0, 2.0, 3.0])["p50_ms"] == 2.0
    print("selftest ok: choice and numeric gate boundaries, latency summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--selftest", action="store_true", help="Test gate logic without loading a model")
    ap.add_argument("--frames", type=Path, default=HERE / "synth_frames")
    ap.add_argument("--reference", type=Path, default=HERE / "ref_cpu.jsonl")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--cpu-reference", type=Path, help="Export one CPU hidden reference, then exit")
    ap.add_argument("--hidden-reference", type=Path, default=HERE / ".venv/valen_cpu_hidden.npz")
    ap.add_argument("--out", type=Path, default=HERE / "ref_mlx.jsonl")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    torch.set_num_threads(4)
    frames = sorted([*args.frames.glob("*.png"), *args.frames.glob("*.jpg")])
    assert frames, "No frames"
    module, config, compiler = native(args.snapshot)
    if args.cpu_reference:
        cpu_reference(args.snapshot, module, compiler, frames[0], args.cpu_reference)
        return
    assert args.hidden_reference.is_file(), "Export --cpu-reference first"
    runner = Runner(args.snapshot, args.dtype, module, config, compiler)
    hidden_ref = np.load(args.hidden_reference)
    assert str(hidden_ref["frame"]) == frames[0].name
    rows, hidden_check = [], {}
    for i, frame in enumerate(frames):
        state = compile_frame(compiler, frame)
        inputs = runner.inputs(state)
        hidden = runner.forward(inputs)
        answers = runner.answers(hidden, state)
        if i == 0:
            actual = np.array(hidden.astype(runner.mx.float32))
            for name in ("reference_hidden", "fp32_hidden"):
                delta = actual - hidden_ref[name]
                hidden_check[name] = {"max_abs": float(np.max(np.abs(delta))),
                                      "mean_abs": float(np.mean(np.abs(delta))),
                                      "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(hidden_ref[name])),
                                      "reference_max_abs": float(np.max(np.abs(hidden_ref[name])))}
            emb = runner.embeddings(inputs, runner.vit(inputs))
            positions = np.array(emb.position_ids)
            assert np.array_equal(positions, hidden_ref["position_ids"]), "mRoPE position ID mismatch"
            hidden_check["position_ids_exact"] = True
        rows.append({"frame": frame.name, "answers": answers,
                     "usage": {"input_tokens": state.logical_tokens, "output_tokens": 0},
                     "internal_usage": {"compute_tokens": state.compute_tokens}})
    gate = agreement(rows, [json.loads(line) for line in args.reference.read_text().splitlines()])
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    from importlib.metadata import version
    result = {"dtype": args.dtype, "revision": REVISION, "agreement": gate,
              "versions": {name: version(name) for name in ("torch", "transformers", "mlx", "mlx-vlm")},
              "hidden_check": hidden_check, "head": "torch CPU fp32",
              "head_parameters": runner.head_params}
    print(json.dumps(result), flush=True)
    if gate["fraction"] < 0.90:
        if args.summary:
            args.summary.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit("Correctness gate failed. No timing was performed.")
    if args.bench:
        assert args.warmup >= 3 and len(frames) * args.repeats >= 40
        for i in range(args.warmup):
            runner.end_to_end(frames[i % len(frames)])
        samples = {stage: [] for stage in ("end_to_end", "forward", "vit", "text")}
        for repeat in range(args.repeats):
            for frame in frames:
                state = compile_frame(compiler, frame)
                inputs = runner.inputs(state)
                runner.mx.eval(inputs)
                ms, _ = measure(runner, lambda: runner.forward(inputs))
                samples["forward"].append(ms)
                ms, visual = measure(runner, lambda: runner.vit(inputs))
                samples["vit"].append(ms)
                emb = runner.embeddings(inputs, visual)
                runner.mx.eval(emb.inputs_embeds, emb.position_ids)
                ms, _ = measure(runner, lambda: runner.text_tower(inputs, emb))
                samples["text"].append(ms)
                ms, _ = measure(runner, lambda: runner.end_to_end(frame))
                samples["end_to_end"].append(ms)
            print(json.dumps({"completed_repeat": repeat + 1}), flush=True)
        result["timings"] = {stage: summary(values) for stage, values in samples.items()}
        result["warmup"] = args.warmup
        result["stage_definitions"] = {
            "end_to_end": "Compiler, input conversion, full backbone, CPU head and answer decoding",
            "forward": "ViT, embedding lookup, image merge, mRoPE indices, text tower, no head",
            "vit": "ViT patch embed through 2x2 merger, no input preprocessing",
            "text": "24 decoder layers and final norm, materialized embeddings and mRoPE indices"}
        result["samples_ms"] = samples
        result["peak_gpu_memory_bytes"] = runner.mx.get_peak_memory()
        print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}), flush=True)
    if args.summary:
        args.summary.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
