# Mamba/fullB Prototype Experiment Tracker

## Status

- Current phase: tiny sanity benchmark complete.
- Long experiments: not started.
- Tiny sanity benchmark: complete.
- Human checkpoint required before benchmark-only speed sweep or WikiText-103 runs.

## Environment Check

- `nvidia-smi`: NVIDIA GeForce RTX 5090, 32607 MiB, driver 570.153.02, CUDA 12.8, no running GPU processes.
- System `python --version`: Python 3.12.3.
- System PyTorch check: torch 2.8.0+cu128, CUDA available, NVIDIA GeForce RTX 5090.
- Repo `.venv` check for benchmark execution: Python 3.12.3, torch 2.11.0+cu128, CUDA available, transformers 5.5.4.
- The system Python does not have `transformers`; benchmark runs used `.venv/bin/python`.

## Architecture Inspection

- Mamba3 implementation: `mamba_ssm/modules/mamba3.py`.
- Halting logic: `Mamba3._forward_conditional`, lines around halt-head probability, `newly_halted`, `active_mask`, and `halt_steps`.
- Router / attention trigger logic: `Mamba3._compute_step_router_prob` and `_forward_conditional`.
- Benchmark script: `benchmarks/benchmark_generation_mamba_simple.py`.
- WikiText-103 training script: `train_wikitext103.py`.
- Result summarization script: `scripts/summarize_wikitext103_results.py`.

Architecture answers:

- fullB skips attention calls at the step level when `router_prob < router_threshold` or no active tokens remain. It does not sparsify the attention operation across active tokens when attention is triggered; `nn.MultiheadAttention` receives the whole `current_states` sequence, and the result is masked/gated afterward with `torch.where(next_active_mask.unsqueeze(-1), attention_out, current_states)`.
- Attention is called at most once per refinement step per layer, over the whole sequence tensor, not once per token.
- Halted tokens do not update their states after halting because Mamba and attention outputs are gated with `next_active_mask`; however, Mamba compute still runs over the whole sequence each executed refinement step, including halted tokens. If attention triggers, attention compute also runs over the whole sequence, including halted tokens.
- Higher `halt_threshold` makes halting harder, so it generally raises `active_fraction`, lowers `halted_fraction`, and can increase work. Lower `halt_threshold` makes tokens halt earlier. Higher `router_threshold` makes attention triggers rarer and lowers `attention_trigger_rate`; lower `router_threshold` triggers attention more often. Router probability is computed from the mean pooled active-token states after the Mamba update.

## Tiny Sanity Benchmark

Completed with batch 1, promptlen 16, d_model 128, n_layer 2, d_state 32, headdim 32, repeats 5, seed 0, dtype float16.

| Variant | Completed | Tok/s | Avg ms | Trigger Rate | Active Fraction | Halted Fraction | Halt Step | Executed Steps | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | yes | 3539.87 | 4.52 | NA | NA | NA | NA | NA | 9.59 |
| halting_only | yes | 1973.21 | 8.11 | 0.00 | 0.3906 | 0.6094 | 1.5938 | 2.00 | 10.03 |
| always_attention | yes | 1195.31 | 13.39 | 1.00 | 1.0000 | 0.0000 | 2.0000 | 2.00 | 11.45 |
| fullB / full | yes | 1618.82 | 9.88 | 0.75 | 0.2969 | 0.7031 | 1.4688 | 2.00 | 11.41 |

## Commands Run

- `nvidia-smi`
- `python --version`
- `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"`
- `find . -maxdepth 3 -type f | sed 's#^\./##' | sort | head -200`
- `python benchmarks/benchmark_generation_mamba_simple.py ...` failed under system Python because `transformers` is missing.
- `.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant plain --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_plain_seed0.json`
- `.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant halting_only --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_halting_only_seed0.json`
- `.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant always_attention --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_always_attention_seed0.json`
- `.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant full --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_full_seed0.json`

## Files Modified

- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `benchmarks/benchmark_generation_mamba_simple.py`
- `train_wikitext103.py`
- `results/fullb_experiments/sanity_plain_seed0.json`
- `results/fullb_experiments/sanity_halting_only_seed0.json`
- `results/fullb_experiments/sanity_always_attention_seed0.json`
- `results/fullb_experiments/sanity_full_seed0.json`
- `results/fullb_experiments/EXPERIMENT_RESULTS.md`

## Next Commands

Benchmark-only speed sweep candidates, not launched:

```bash
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant plain --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 20 --seed 0 --dtype float16 --output-json results/fullb_experiments/sweep_plain_b1_p16_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant halting_only --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 20 --seed 0 --dtype float16 --halt-threshold 0.5 --output-json results/fullb_experiments/sweep_halting_only_b1_p16_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant always_attention --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 20 --seed 0 --dtype float16 --output-json results/fullb_experiments/sweep_always_attention_b1_p16_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant full --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 20 --seed 0 --dtype float16 --halt-threshold 0.5 --router-threshold 0.5 --output-json results/fullb_experiments/sweep_full_b1_p16_seed0.json
```

Suggested sweep expansion after approval: repeat those four commands for `--promptlen 64`, `--promptlen 128`, and `--batch 4`; keep the same model shape first so results isolate sequence/batch effects.
