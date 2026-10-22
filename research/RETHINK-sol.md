# Reflex-1: cache language, not changing evidence

2026-10-22. **[M]** measured previously, not rerun. **[S]** sourced at link/path. **[E]** proposal, target, or subjective probability. All training/search/live trials below are future AC-powered work. None ran today.

## 1. Jev: features and weaknesses

**[S]** Multiple PNG/JPEG assets work throughout nested state, instructions, and criteria. Documented compositions include reference-image identity matching, pictured Choice options and Score/Noul criteria, frame-history change detection, coarse-to-fine crops, and tile search. Images work in 1.13/1.14. Current limits require both dimensions >=280 pixels, rejecting native 256x144. [Images](https://docs.typesafe.ai/images).

**[S]** Jev 1.14 returns one normalized, axis-aligned Box per question, located by description or reference image. Box always returns coordinates, even for absent targets: pair it with Noul presence. Vision remains experimental and header-gated. These docs specify no persistent video tracker/local stream. [Box](https://docs.typesafe.ai/primitives/box). Choice/Score confidence measures distribution concentration, not independently validated correctness. [Confidence](https://docs.typesafe.ai/confidence).

**[S]** The independent **Visual Jev** paper is not commercial Jev's implementation disclosure. Its 5.7 ms/question means roughly 182 ms for 32 answers on RTX 5090, excluding network/queueing. Its candidate probabilities are explicitly not calibrated correctness probabilities. [Table 3 and limitations](https://arxiv.org/html/2609.25845v1).

## 2. Concrete architecture and training

**[S]** sys1rust builds separate state-plus-question rows (`runtime/crates/laya-core/src/sequence.rs`). **[M]** Its 11.6–15.5 ms short/single-question results are M5 text timings, not M4 four-question vision (`sys1rust/results/SPEED.md`). **[E]** Neither repeated language passes nor caching bidirectional states after inserting changing images is the desired architecture.

**[E]** Use pretrained **MobileCLIP2-S0** dense features, independently encoded Laya question/candidate and text-state features, then two small cross-attention blocks before typed heads. Project images to Laya width 1024 with 2D, asset, and timestamp embeddings. Images enter Laya's *extended decision head*, not every encoder layer. Cache unchanged text/reference assets only, invalidating on content/preprocessing changes. Project changing numeric/key-history fields separately, not as fresh prose. Each question sees shared state plus its own assets, never another question. Preserve `/v1/systemone` schemas. This requires training, not an answers-preserving runtime patch.

**[S]** S0 has an 11.4M image tower; Apple's 1.5 ms image result is its mobile benchmark, not this Mac. [Official models](https://github.com/apple-aiml-research/ml-mobileclip). [FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models) supports hybrid encoding/fewer tokens. [UltraViT, July 2026](https://arxiv.org/html/2607.23373v1) supports dense distillation, but speedups use INT8 Qualcomm hardware. [LiteFrame, 2026](https://jjihwan.github.io/projects/LiteFrame/) directly distills compressed representations. Borrow methods, not their hardware numbers or pretraining budgets.

**[E]** Retain stride-8/16 intermediate features for query-conditioned Box edge distributions plus sub-bin offsets, ordered edges, presence, and calibrated localization quality. Compress semantic evidence to about 36 tokens. Preserve Choice/Score/Noul confidence formulas. Add only versioned Point, a heatmap with uncertainty. A small recurrent state carries target identity and road-relative belief. “More” is local timestamped streaming, reusable assets, tracking, and validated abstention, not documented as native Jev features. Entropy alone is not calibration.

**[E]** Train projector/fusion/heads with frozen towers on roughly 100k–300k license-audited non-driving image-QA, grounding, OCR/UI, and image-exemplar examples. Then adapt late layers with LoRA, dense teacher distillation, exact sim boxes, and planner corrections, including recoveries/absent targets. Hold out images, episodes, tracks, and question templates. Calibrate separately using Brier/ECE and Box risk-versus-coverage. The brief's 4,635 frames cannot establish general Jev parity.

**[E]** Laya offers language/typed-head/runtime reuse, not proven visual superiority. Compare this split model against already vision-aligned FastVLM-0.5B with typed readout under matched data. Keep Laya only if quality per millisecond wins. P(success)=0.55; likely failure is cross-modal reasoning after late fusion.

## 3. Latency plan

**[E]** Four-question cached-text target: **6 ms p50, <=10 ms p99** per 256x144 frame. Budget: preprocessing 0.5, vision 3.0, projection/fusion 1.5, heads/transport 1.0 ms. Fresh text adds estimated 15–25 ms per 128-token encoder sequence, not per arbitrary multiquestion request. Capture age and key delivery separately budget <=8.3 and <=2 ms. High-resolution OCR/crops are outside this target. None is measured yet.

**[M/E]** Reuse sys1rust fp16, fixed shapes, warm graphs, selective readout rows, bounded allocations, and fused elementwise work in Swift capture plus Rust/MLX. Share evidence explicitly, not by assuming stock Laya already does. M5-specific NAX kernels do not transfer blindly to M4. Real INT8 failed multilingual agreement (`SPEED.md`), so defer it. Gate runtime changes at >=99% agreement with the frozen trained PyTorch checkpoint, separately from ground-truth model quality. Valen remains offline baseline: measured 86.7/93.5 ms p50/p99 (`bench/infer/RESULTS.md`).

**[E]** Jev hosted round-trip is unknown. Compare identical questions/content, padding to Jev's image minimum, including upload/queue/network at matched accuracy. No claimed speedup until measured. Also report mJ/frame and dropped frames.

## 4. Self-taught sub-21, then live pixels

**[M/E]** Brief reports 26.463 s, versus older `learn/RESULTS.md` at 26.642 s. `learn/profile.mjs` is absent here. Provisionally use brief: saving 5.463 s is required, so its reported roughly 2 s first-jump loss is insufficient.

**[E]** Replace greedy 2.5 s horizons/200 ms commits with archived approaches and multistart 6–8 s searches spanning takeoff, landing, and recovery. Retain alternate routes/landing states, score completed laps, then refine control boundaries 50 to 10 to 1 ms. Harden finalists with controller-in-loop jitter/latency. Count nonfinishes as failures. Gate: Verify <21000 ms and >=90/100 independent perturbed finishes. Seeds use only geometry, physics, own runs. P(sim success)=0.35; likely failure is missing a faster feasible route.

**[E]** Per frame, estimate road-relative lateral error, heading, speed, airborne state, landmarks, and uncertainty from recent pixels plus issued keys. Chase-camera car Box alone cannot identify world pose. Planner supplies own-run reference/recovery tube; CPU feedback corrects timestamped 1 ms schedules. No full CEM/generative VLM per frame. Propagate observation age, drop stale frames/corrections, reset when tracking fails. Desktop 2, real OS keys, no game-state hooks. HUD/Verify scores finishes. P(live <21)=0.20; likely failure is observability/timing margin.

## 5. Scope cuts

**[E]** Keep exact sim/Verify, isolation, own-run renderer, `eye/`, sys1rust, existing student baseline, and controller. Freeze contaminated `opt/` assets. Cut Valen streaming retraining, broad runtime bake-offs, learned world models, backbone RL, online weight updates, trailers, and agent infrastructure. Later correct stale WR-seeding text in DESIGN.md, never import those seeds. P(cuts shorten delivery)=0.9; failure is needing the deferred larger visual model.

## 6. Top three risks and cheapest tests

**[E]** **Alignment, P=0.45:** late fusion learns templates. Next AC session, bounded training pilot plus held-out paraphrases, swapped images, absent objects, and permuted options against pretrained VLM. Reject if visual interventions do not change relevant answers.

**[E]** **Live state ambiguity/noise, P=0.55:** IoU hides poor pose/speed estimates. Evaluate 200 own-run Desktop-2 frames with independently synchronized labels, then inject observed errors into controller. Replayed sent keys alone are not exact live labels without synchronization checks.

**[E]** **21 s infeasible within budget, P=0.65:** longer search finds no faster robust basin. Later run fixed-budget paired jump-search ablation and independent jitter evaluation. Faster inference cannot prove lap feasibility.

## 7. First milestones

**[E]** **M1, 0.5–1 day:** bounded serial benchmarks of existing student and S0 feature path. End with M4 p50/p99, memory, and student held-out IoU. Brief baseline: IoU 0.72, edge error 8.1 px, latency unknown.

**[E]** **M2, 5–10 days plus serial AC GPU work:** train fused model, port settled graph. End with four-question p50/p99, >=99% runtime agreement, held-out accuracy/IoU, ECE, selective error. Targets: <=8 ms p50, <=10 ms p99 cached stream, IoU >=0.80, ECE <=0.05. Measure Jev if credentials exist.

**[E]** **M3, 3–10 additional days, feasibility-dependent:** search plus Desktop-2 closed loop. End with Verify best time, independent finish count, live HUD best/median over 20 trials. Targets: sim/live <21000 ms, >=90/100 simulated finishes. Publish actual gap if missed. Never submit bot times.
