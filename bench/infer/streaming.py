"""Question-first Valen prototype, with isolated KV and DeltaNet prefix caches.

Reorders existing Compiler token blocks without adding or retokenizing content:
[question user message] [original state user message] [original assistant readouts].
Run correctness first, then --bench. No training or state-first parity is implied.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map

from valen_mlx import HERE, REVISION, SNAPSHOT, Runner, agreement, compile_frame, measure, native, summary


def reorder(original, tokenizer):
    """Move the exact question message, including its closing delimiter, first."""
    state = copy.copy(original)
    n = original.inputs["input_ids"].shape[1]
    context_end = original.questions[0].branches[0].context_span[1]
    last = original.questions[-1].branches[-1]
    schema_end = last.candidate_positions[-1] + 1 + len(tokenizer.encode("\n", add_special_tokens=False))
    close = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    assert original.inputs["input_ids"][0, schema_end:schema_end + len(close)].tolist() == close
    split = schema_end + len(close)
    order = list(range(context_end, split)) + list(range(context_end)) + list(range(split, n))
    assert sorted(order) == list(range(n))
    inverse = [0] * n
    for new, old in enumerate(order):
        inverse[old] = new

    def span(old):
        start, end = old
        result = (inverse[start], inverse[end - 1] + 1)
        assert inverse[start:end] == list(range(*result)), "Role span must stay contiguous"
        return result

    state.inputs = {k: v[:, order] if isinstance(v, torch.Tensor) and v.shape == (1, n) else v
                    for k, v in original.inputs.items()}
    state.questions = []
    for q in original.questions:
        question = copy.copy(q)
        question.branches = []
        for b in q.branches:
            branch = copy.copy(b)
            branch.inputs = state.inputs
            branch.context_span = span(b.context_span)
            branch.instruction_span = span(b.instruction_span)
            branch.candidate_spans = [span(s) for s in b.candidate_spans]
            branch.candidate_positions = [inverse[p] for p in b.candidate_positions]
            branch.decision_position = inverse[b.decision_position]
            question.branches.append(branch)
        state.questions.append(question)
    prefix = split - context_end
    counts = {"total": n, "cached_prefix": prefix, "per_frame": n - prefix,
              "context": context_end, "readout": n - split,
              "visual": sum(int(g.prod()) // 4 for g in original.inputs["image_grid_thw"])}
    return state, counts


def detach(tree):
    # Fresh containers and MLX handles prevent list and in-place array aliasing.
    # MLX shares immutable backing buffers until a handle is updated.
    return tree_map(lambda value: mx.array(value) if isinstance(value, mx.array) else value, tree)


def arrays(tree):
    return [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]


def digest(tree):
    h = hashlib.sha256()
    for key, value in tree_flatten(tree):
        h.update(key.encode())
        if isinstance(value, mx.array):
            h.update(str((value.shape, value.dtype)).encode())
            h.update(np.array(value).tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


class Streaming(Runner):
    def __init__(self, snapshot, dtype, module, config, compiler, first_frame):
        super().__init__(snapshot, dtype, module, config, compiler)
        state, self.counts = reorder(compile_frame(compiler, first_frame), compiler.tokenizer)
        self.prefix = self.counts["cached_prefix"]
        self.prefix_ids = state.inputs["input_ids"][:, :self.prefix].clone()
        inputs = self.inputs(state)
        prefix_ids = inputs["input_ids"][:, :self.prefix]
        positions = mx.broadcast_to(mx.arange(self.prefix)[None, None, :], (3, 1, self.prefix))
        caches = self.model.language_model.make_cache()
        self.prefix_hidden = self.model.language_model.model(prefix_ids, cache=caches, position_ids=positions)
        self.snapshot = detach([c.prefix_cache_snapshot() for c in caches])
        mx.eval(self.prefix_hidden, arrays(self.snapshot))
        mx.synchronize()
        self.snapshot_hash = digest(self.snapshot)
        self.cache_bytes = sum(v.nbytes for v in arrays(self.snapshot))
        self.cache_classes = [type(c).__name__ for c in caches]
        assert self.cache_classes.count("KVCache") == 6
        assert self.cache_classes.count("ArraysCache") == 18
        for cache, saved in zip(caches, self.snapshot):
            if type(cache).__name__ == "ArraysCache":
                assert len(saved["state"]) == 2 and all(v is not None for v in saved["state"])
        self.assert_snapshot()

    def prepare(self, frame):
        # ponytail: reuse the full native Compiler each frame. It repeats cheap
        # static tokenization, but no question-prefix transformer work is repeated.
        state, counts = reorder(compile_frame(self.compiler, frame), self.compiler.tokenizer)
        assert counts == self.counts, "This fixed-workload prototype requires a fixed frame grid"
        assert torch.equal(state.inputs["input_ids"][:, :self.prefix], self.prefix_ids)
        return state, self.inputs(state)

    def fresh_cache(self):
        caches = self.model.language_model.make_cache()
        for cache, saved in zip(caches, self.snapshot):
            cache.prefix_cache_restore(detach(saved))
            if type(cache).__name__ == "KVCache":
                assert cache.offset == self.prefix
        return caches

    def cached_hidden(self, inputs, emb):
        tail = self.model.language_model.model(
            inputs["input_ids"][:, self.prefix:], inputs_embeds=emb.inputs_embeds[:, self.prefix:],
            position_ids=emb.position_ids[..., self.prefix:], cache=self.fresh_cache())
        # Reconstruct full hidden rows so the unchanged native CPU head can pool
        # question roles from the prefix and context/decisions from this frame.
        return mx.concatenate([self.prefix_hidden, tail], axis=1)

    def end_to_end(self, frame):
        state, inputs = self.prepare(frame)
        emb = self.embeddings(inputs, self.vit(inputs))
        return self.answers(self.cached_hidden(inputs, emb), state)

    def assert_snapshot(self):
        assert digest(self.snapshot) == self.snapshot_hash, "Prefix cache was mutated"


def breakdown(rows, reference):
    return {qid: agreement(
        [{"frame": r["frame"], "answers": {qid: r["answers"][qid]}} for r in rows],
        [{"frame": r["frame"], "answers": {qid: r["answers"][qid]}} for r in reference])
        for qid in rows[0]["answers"]}


def row(frame, answers, counts, cached):
    return {"frame": frame.name, "answers": answers,
            "usage": {"input_tokens": counts["total"], "output_tokens": 0},
            "internal_usage": {"compute_tokens": counts["per_frame"] if cached else counts["total"]}}


def selftest(compiler, frames):
    prefix_ids = None
    for frame in frames:
        original = compile_frame(compiler, frame)
        state, counts = reorder(original, compiler.tokenizer)
        ids, old = state.inputs["input_ids"][0], original.inputs["input_ids"][0]
        for q, oq in zip(state.questions, original.questions):
            for b, ob in zip(q.branches, oq.branches):
                for span, old_span in [(b.context_span, ob.context_span),
                                       (b.instruction_span, ob.instruction_span),
                                       *zip(b.candidate_spans, ob.candidate_spans)]:
                    assert torch.equal(ids[slice(*span)], old[slice(*old_span)])
                assert ids[b.decision_position] == old[ob.decision_position]
                assert b.context_span == (counts["cached_prefix"], counts["cached_prefix"] + counts["context"])
                assert b.instruction_span[1] <= counts["cached_prefix"]
                assert all(s[1] <= counts["cached_prefix"] for s in b.candidate_spans)
                assert b.decision_position >= counts["cached_prefix"] + counts["context"]
        current = ids[:counts["cached_prefix"]]
        if prefix_ids is not None:
            assert torch.equal(current, prefix_ids)
        prefix_ids = current
    # Test both mutable-container and mutable-array isolation used by snapshots.
    source = {"state": [mx.array([1., 2.]), mx.array([3.])]}
    copied = detach(source)
    copied["state"][0][0] = 99.
    copied["state"][1] = mx.array([7.])
    assert source["state"][0].tolist() == [1., 2.] and source["state"][1].item() == 3.
    print(json.dumps({"selftest": "exact role token spans, static prefix, isolated snapshot handles", "tokens": counts}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--frames", type=Path, default=HERE / "synth_frames")
    ap.add_argument("--reference", type=Path, default=HERE / "ref_cpu_fp32.jsonl")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--out", type=Path, default=HERE / "ref_streaming.jsonl")
    ap.add_argument("--uncached-out", type=Path, default=HERE / "ref_streaming_uncached.jsonl")
    ap.add_argument("--summary", type=Path, default=HERE / "streaming_results.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(4)
    frames = sorted([*args.frames.glob("*.png"), *args.frames.glob("*.jpg")])
    assert frames
    module, config, compiler = native(args.snapshot)
    selftest(compiler, frames)
    if args.selftest:
        return
    runner = Streaming(args.snapshot, args.dtype, module, config, compiler, frames[0])
    cached, uncached, errors = [], [], []
    for frame in frames:
        state, inputs = runner.prepare(frame)
        emb = runner.embeddings(inputs, runner.vit(inputs))
        if not cached:
            positions = np.array(emb.position_ids)[..., :runner.prefix]
            assert np.array_equal(positions, np.broadcast_to(np.arange(runner.prefix), positions.shape))
        full = runner.text_tower(inputs, emb)
        cached_hidden = runner.cached_hidden(inputs, emb)
        mx.eval(full, cached_hidden)
        mx.synchronize()
        uncached.append(row(frame, runner.answers(full, state), runner.counts, False))
        cached.append(row(frame, runner.answers(cached_hidden, state), runner.counts, True))
        difference = np.array(full.astype(mx.float32)) - np.array(cached_hidden.astype(mx.float32))
        errors.append({"frame": frame.name, "max_abs": float(np.max(np.abs(difference))),
                       "mean_abs": float(np.mean(np.abs(difference)))})
        runner.assert_snapshot()
    # A -> other frames -> A checks both cache and saved prefix-hidden isolation.
    repeated = runner.end_to_end(frames[0])
    assert repeated == cached[0]["answers"], "Frame A changed after other frames"
    runner.assert_snapshot()
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    mechanics = agreement(cached, uncached)
    result = {"revision": REVISION, "dtype": args.dtype, "tokens": runner.counts,
              "versions": {name: version(name) for name in ("mlx", "mlx-vlm", "torch", "transformers")},
              "cache_vs_uncached": mechanics, "cache_per_question": breakdown(cached, uncached),
              "zero_shot_vs_state_first_fp32": agreement(cached, reference),
              "zero_shot_per_question": breakdown(cached, reference),
              "cache_hidden_error": errors, "prefix_snapshot_sha256": runner.snapshot_hash,
              "prefix_snapshot_unchanged": True, "frame_a_replay_exact": True,
              "prefix_cache_bytes": runner.cache_bytes,
              "cache_layers": {name: runner.cache_classes.count(name) for name in set(runner.cache_classes)},
              "head": "native torch CPU fp32", "reference": args.reference.name}
    args.out.write_text("".join(json.dumps(r) + "\n" for r in cached))
    args.uncached_out.write_text("".join(json.dumps(r) + "\n" for r in uncached))
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"zero_shot_per_question", "cache_per_question"}}), flush=True)
    if mechanics["fraction"] < .99:
        raise SystemExit("Cache correctness gate failed. No timing was performed.")
    if args.bench:
        assert args.warmup >= 3 and len(frames) * args.repeats >= 40
        for i in range(args.warmup):
            runner.end_to_end(frames[i % len(frames)])
        samples = {stage: [] for stage in ("vit", "text_from_cache", "end_to_end")}
        for repeat in range(args.repeats):
            for frame in frames:
                state, inputs = runner.prepare(frame)
                mx.eval(inputs)
                ms, visual = measure(runner, lambda: runner.vit(inputs))
                samples["vit"].append(ms)
                emb = runner.embeddings(inputs, visual)
                mx.eval(emb.inputs_embeds, emb.position_ids)
                ms, _ = measure(runner, lambda: runner.cached_hidden(inputs, emb))
                samples["text_from_cache"].append(ms)
                ms, _ = measure(runner, lambda: runner.end_to_end(frame))
                samples["end_to_end"].append(ms)
            print(json.dumps({"completed_repeat": repeat + 1}), flush=True)
        runner.assert_snapshot()
        result["timings"] = {stage: summary(values) for stage, values in samples.items()}
        result["samples_ms"] = samples
        result["warmup"] = args.warmup
        result["stage_definitions"] = {
            "vit": "ViT through image merger, materialized processor pixels",
            "text_from_cache": "Fresh detached prefix-cache restore, tail text tower, hidden concat, materialized embeddings and mRoPE indices",
            "end_to_end": "Native Compiler, reorder, ViT, embedding/merge/mRoPE, cache restore and tail text, CPU head and answers"}
        result["peak_gpu_memory_bytes"] = mx.get_peak_memory()
        args.summary.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"timings": result["timings"], "tokens": runner.counts}), flush=True)


if __name__ == "__main__":
    main()
