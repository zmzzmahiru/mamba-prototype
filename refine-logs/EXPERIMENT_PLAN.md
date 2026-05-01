# Mamba/fullB Prototype Experiment Plan

## Scope

Benchmark and analyze four prototype variants:

- plain Mamba
- halting_only
- always_attention
- fullB / routing + halting

Long experiments are out of scope until explicitly approved. This pass stops after a tiny sanity benchmark.

## Phase 1: Environment Check

- Run `nvidia-smi`.
- Record Python version.
- Record PyTorch version, CUDA availability, and GPU name.
- List repository files up to depth 3.
- Note existing dirty worktree state before edits.

## Phase 2: Architecture Inspection

Identify and inspect:

- Mamba3 prototype implementation.
- Halting logic.
- Router / attention trigger logic.
- `benchmarks/benchmark_generation_mamba_simple.py`.
- `train_wikitext103.py`.
- Result summarization scripts.

Answer before behavior changes:

- Does fullB actually skip attention computation, or compute attention and mask/gate afterward?
- Is attention called once per refinement step, once per token, or once for the whole sequence?
- Do halted tokens still go through compute?
- How do `halting_threshold` and `router_threshold` affect `active_fraction`, `halted_fraction`, and `attention_trigger_rate`?

## Phase 3: Tiny Sanity Benchmark

Run only tiny benchmark-only checks:

- variants: `plain`, `halting_only`, `always_attention`, `fullB`
- batch size: 1
- prompt length: 16
- `d_model`: 128
- `n_layer`: 2
- `d_state`: 32
- `headdim`: 32
- repeats: 3 to 5
- seed: 0
- output directory: `results/fullb_experiments/`

Record command, device, dtype, model shape, thresholds, speed, runtime, memory if available, and prototype stats if available.

## Phase 4: Benchmark-Only Speed Sweep

Prepare exact commands for a later benchmark-only speed sweep after sanity passes. Candidate axes:

- variants: `plain`, `halting_only`, `always_attention`, `fullB`
- seeds: at least 0, then optionally 1 and 2
- batch sizes: small set such as 1, 4, 8
- prompt lengths: small set such as 16, 64, 128
- thresholds: baseline current defaults plus selected `halting_threshold` / `router_threshold` values

Do not launch this phase without approval.

## Phase 5: Short WikiText-103 Quality/Speed Run

After benchmark-only results are understood, run a short WikiText-103 quality/speed check:

- all four variants
- limited `max_train_steps`
- structured JSON/CSV/Markdown outputs in `results/fullb_experiments/`
- capture validation loss, train tokens/sec, runtime, memory, and prototype stats

Do not launch this phase without approval.

## Phase 6: Optional Longer Confirmation

Only after human approval:

- repeat promising settings across multiple seeds
- run longer train schedules
- compare mean and variance of speed, loss, active fraction, halted fraction, attention trigger rate, halt step, and executed steps

