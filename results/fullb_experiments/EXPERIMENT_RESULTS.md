# Mamba/fullB Tiny Sanity and Speed Sweep Results

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB.
- Driver/CUDA from `nvidia-smi`: 570.153.02 / CUDA 12.8.
- System Python: 3.12.3.
- System PyTorch check: torch 2.8.0+cu128, CUDA available.
- Benchmark Python: `.venv/bin/python`, torch 2.11.0+cu128, transformers 5.5.4.
- Note: system Python failed to run the benchmark because `transformers` is not installed.

## Architecture Findings

- fullB saves attention compute only by skipping entire attention calls for refinement steps where `router_prob < router_threshold` or no active tokens remain.
- When attention does run, it is full-sequence `nn.MultiheadAttention(current_states, current_states, current_states)`, followed by `torch.where` masking for active tokens. It does not avoid attention computation for halted tokens within a triggered step.
- Attention is called at most once per refinement step per layer, over the whole sequence.
- Halted tokens keep their prior states, but Mamba compute still runs over the whole sequence each executed step. Attention compute also runs over the whole sequence on triggered attention steps.
- `halt_threshold` controls how easily tokens halt: lower threshold means lower active fraction and higher halted fraction; higher threshold keeps more tokens active. `router_threshold` controls how often attention triggers: lower threshold increases `attention_trigger_rate`; higher threshold reduces it.

## Tiny Sanity Commands

```bash
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant plain --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_plain_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant halting_only --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_halting_only_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant always_attention --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_always_attention_seed0.json
.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --variant full --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 5 --seed 0 --dtype float16 --output-json results/fullb_experiments/sanity_full_seed0.json
```

## Tiny Sanity Metrics

| Variant | Completed | Tok/s | Avg ms | Trigger Rate | Active Fraction | Halted Fraction | Halt Step | Executed Steps | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | yes | 3539.87 | 4.52 | NA | NA | NA | NA | NA | 9.59 |
| halting_only | yes | 1973.21 | 8.11 | 0.00 | 0.3906 | 0.6094 | 1.5938 | 2.00 | 10.03 |
| always_attention | yes | 1195.31 | 13.39 | 1.00 | 1.0000 | 0.0000 | 2.0000 | 2.00 | 11.45 |
| fullB / full | yes | 1618.82 | 9.88 | 0.75 | 0.2969 | 0.7031 | 1.4688 | 2.00 | 11.41 |

## Benchmark-Only Speed Sweep

Completed on 2026-05-01. No WikiText-103 training was launched.

- Command shape: `.venv/bin/python benchmarks/benchmark_generation_mamba_simple.py --prototype --batch 1 --promptlen 16 --d-model 128 --n-layer 2 --d-state 32 --headdim 32 --repeats 20 --dtype float16 ...`
- Variants: `plain`, `halting_only`, `always_attention`, `fullB` (`--variant full`).
- Seeds: 0, 1, 2.
- `halting_only` and `fullB` halting thresholds: 0.5, 0.7, 0.9.
- `fullB` router thresholds: 0.3, 0.5, 0.7.
- Raw logs: `results/fullb_experiments/logs/`.
- Structured results: `results/fullb_experiments/raw_results.csv`.
- Aggregate results: `results/fullb_experiments/summary_results.csv`.

| Variant | Halt | Router | Runs | Mean tok/s | Speed vs plain | Avg ms | Trigger Rate | Active Fraction | Halted Fraction | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 0.5 | 0.5 | 3 | 4764.09 | 1.000 | 3.40 | NA | NA | NA | 9.59 |
| always_attention | 0.5 | 0.0 | 3 | 1541.52 | 0.324 | 10.40 | 1.000 | 1.000 | 0.000 | 11.45 |
| halting_only | 0.5 | 0.5 | 3 | 1722.57 | 0.362 | 9.30 | 0.000 | 0.333 | 0.667 | 10.03 |
| halting_only | 0.7 | 0.5 | 3 | 1694.49 | 0.356 | 9.48 | 0.000 | 0.958 | 0.042 | 10.03 |
| halting_only | 0.9 | 0.5 | 3 | 1688.51 | 0.354 | 9.51 | 0.000 | 1.000 | 0.000 | 10.03 |
| fullB | 0.5 | 0.3 | 3 | 1476.87 | 0.310 | 10.85 | 0.833 | 0.281 | 0.719 | 11.42 |
| fullB | 0.5 | 0.5 | 3 | 1516.55 | 0.318 | 10.77 | 0.500 | 0.281 | 0.719 | 11.03 |
| fullB | 0.5 | 0.7 | 3 | 1587.58 | 0.333 | 10.10 | 0.000 | 0.281 | 0.719 | 10.28 |
| fullB | 0.7 | 0.3 | 3 | 1465.84 | 0.308 | 10.94 | 1.000 | 0.948 | 0.052 | 11.45 |
| fullB | 0.7 | 0.5 | 3 | 1618.86 | 0.340 | 9.89 | 0.417 | 0.927 | 0.073 | 11.35 |
| fullB | 0.7 | 0.7 | 3 | 1736.63 | 0.365 | 9.22 | 0.000 | 0.927 | 0.073 | 10.28 |
| fullB | 0.9 | 0.3 | 3 | 1515.45 | 0.318 | 10.57 | 1.000 | 1.000 | 0.000 | 11.45 |
| fullB | 0.9 | 0.5 | 3 | 1613.34 | 0.339 | 9.92 | 0.333 | 1.000 | 0.000 | 11.34 |
| fullB | 0.9 | 0.7 | 3 | 1706.01 | 0.358 | 9.40 | 0.000 | 1.000 | 0.000 | 10.28 |

## Speed Sweep Conclusions

- Fastest variant: `plain`, 4764.09 mean tok/s.
- Best non-plain setting: `fullB` with `halt_threshold=0.7`, `router_threshold=0.7`, 1736.63 mean tok/s.
- Speed ratio relative to plain: fastest overall is `plain` at 1.000x. Best fullB is 0.365x plain, so it is about 2.74x slower than plain.
- Halting threshold effect: lower halting threshold reduces active fraction, but it does not produce a real speedup in this implementation. In `halting_only`, active fraction rises from 0.333 at h=0.5 to 1.000 at h=0.9, while speed stays around 1689-1723 tok/s, only about 0.35-0.36x plain.
- Router threshold effect: higher router threshold reduces attention trigger rate and generally improves speed for fullB. For h=0.7, trigger rate falls from 1.000 at r=0.3 to 0.000 at r=0.7, and speed rises from 1465.84 to 1736.63 tok/s.
- Competitive with plain: no fullB setting is competitive with plain in this sweep. The best fullB result reaches only 36.5% of plain throughput.
- Implementation implication: yes, the current prototype needs token compaction and/or real sparse execution to get actual speedup. Today it still runs Mamba over the full sequence, and triggered attention is full-sequence `nn.MultiheadAttention` with masking applied afterward.

## Route B Checkpoint-Curve Analysis

Completed on 2026-05-01. No new WikiText-103 training was launched.

- Inputs: existing Route B-style JSON results at 200, 500, and 1000 train steps for `plain` and `full_h07`, seeds 0, 1, and 2.
- 200-step source: `results/200step_largerdata_confirm/`.
- 500-step source: `results/routeB_plain_vs_full_500step/`.
- 1000-step source: `results/routeB_plain_vs_full_1000step/`.
- Outputs: `results/fullb_experiments/routeB_checkpoint_curve_raw.csv`, `results/fullb_experiments/routeB_checkpoint_curve_summary.csv`, and `results/fullb_experiments/routeB_val_loss_curve.png`.
- Limitation: these are endpoint evaluations from separate 200/500/1000-step runs, not dense checkpoint histories within one run. The crossover is therefore resolved only to the observed checkpoints.

| Config | Step | Runs | Mean val loss | Std | Mean val ppl | Mean train tok/s |
|---|---:|---:|---:|---:|---:|---:|
| plain | 200 | 3 | 7.8125 | 0.0196 | 2471.61 | 1669.27 |
| plain | 500 | 3 | 7.8748 | 0.0043 | 2630.21 | 3852.18 |
| plain | 1000 | 3 | 8.4736 | 0.0641 | 4793.53 | 6721.79 |
| full_h07 | 200 | 3 | 8.0486 | 0.0472 | 3131.86 | 1566.29 |
| full_h07 | 500 | 3 | 7.9318 | 0.0733 | 2789.50 | 3251.47 |
| full_h07 | 1000 | 3 | 8.3211 | 0.1446 | 4138.47 | 5123.50 |

Answers:

1. `full_h07` overtakes `plain` only at the 1000-step observation. At 200 and 500 steps, `plain` has lower mean validation loss. Per seed, seed 0 already favors `full_h07` at 500 steps, but all three seeds favor `full_h07` at 1000 steps.
2. The best observed `plain` checkpoint is 200 steps, with mean validation loss 7.8125.
3. The best observed `full_h07` checkpoint is 500 steps, with mean validation loss 7.9318.
4. `plain` has the better best-checkpoint validation loss: 7.8125 for `plain` at 200 steps vs. 7.9318 for `full_h07` at 500 steps.
5. Yes. At this resolution, 1000 steps is past the best observed stopping point for both configs. `plain` degrades from 7.8125 at 200 steps to 8.4736 at 1000 steps; `full_h07` improves from 200 to 500, then degrades to 8.3211 at 1000.
6. No. `full_h07` does not show a best-checkpoint quality gain over `plain`, and the speed sweep shows it is not throughput competitive with `plain` in the current implementation.
7. The benchmark-only sweep changes the final conclusion from "full_h07 may be a quality/speed tradeoff worth extending" to "the current fullB/halting implementation is a quality experiment, not a speed win." The 1000-step endpoint still shows `full_h07` can beat an overtrained `plain` endpoint, but checkpoint-curve analysis shows `plain` has the better early-stopped validation loss.

## Final Conclusion

Plain remains the best practical configuration under best-checkpoint selection. The best observed `plain` result is 200 steps with mean validation loss 7.8125, while the best observed `full_h07` result is 500 steps with mean validation loss 7.9318.

The apparent `full_h07` advantage at 1000 steps is endpoint-dependent. At the 1000-step endpoint, `full_h07` has lower mean validation loss than `plain` (8.3211 vs. 8.4736), but both configurations are already past their best observed checkpoints by that point. `plain` degrades from 7.8125 at 200 steps to 8.4736 at 1000 steps, and `full_h07` improves from 8.0486 at 200 steps to 7.9318 at 500 steps before degrading to 8.3211 at 1000 steps.

`full_h07` changes the later-horizon trajectory, but it does not beat `plain` on the best observed validation loss. The mechanism may affect optimization dynamics or regularization at longer horizons, but the current evidence does not support selecting it as the practical winner.

The benchmark-only sweep shows the current fullB/halting implementation is not speed competitive. The best fullB benchmark setting reaches only 36.5% of `plain` throughput, and the trained Route B `full_h07` setting is also slower than `plain` in training throughput at each observed checkpoint. This is consistent with the architecture finding: halted tokens are masked after full-sequence computation, so the prototype does not perform real token-level compute skipping.

The current dynamic mechanism should therefore be presented as a scientific/prototype finding, not a practical speedup. The next technical step is real token compaction or sparse execution, not more threshold sweeping. Further threshold sweeps are unlikely to change the practical conclusion until the implementation can actually avoid Mamba and attention work for halted tokens.

## Mentor Update Summary

- 200-step larger-data result: `plain` is strongest at the short horizon, with mean validation loss 7.8125. `full_h07` reaches 8.0486, `halting_only_h07` reaches 8.0531, and `always_attention` reaches 8.1601.
- Route B 1000-step endpoint: `full_h07` beats the 1000-step `plain` endpoint, with mean validation loss 8.3211 vs. 8.4736 across seeds 0, 1, and 2.
- Checkpoint-curve analysis: the 1000-step result is not the best-checkpoint comparison. `plain` is best at 200 steps, `full_h07` is best at 500 steps, and both are worse by 1000 steps. Best observed validation loss remains `plain` at 7.8125.
- Benchmark-only speed sweep: current fullB/halting is not speed competitive. Best fullB reaches 0.365x `plain` throughput in the benchmark-only sweep because the prototype does not do real token-level compute skipping.
- Final recommendation: keep `plain` as the practical baseline/winner for this experiment. Treat `full_h07` as evidence that dynamic computation changes later-horizon behavior, but do not claim a speedup or practical win. Move next to real token compaction or sparse execution before spending more time on threshold sweeps.

## Files Modified

- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `benchmarks/benchmark_generation_mamba_simple.py`
- `train_wikitext103.py`
- `scripts/analyze_routeB_checkpoint_curve.py`
- `results/fullb_experiments/sanity_plain_seed0.json`
- `results/fullb_experiments/sanity_halting_only_seed0.json`
- `results/fullb_experiments/sanity_always_attention_seed0.json`
- `results/fullb_experiments/sanity_full_seed0.json`
- `results/fullb_experiments/logs/*.log`
- `results/fullb_experiments/sweep_*.json`
- `results/fullb_experiments/raw_results.csv`
- `results/fullb_experiments/summary_results.csv`
- `results/fullb_experiments/routeB_checkpoint_curve_raw.csv`
- `results/fullb_experiments/routeB_checkpoint_curve_summary.csv`
- `results/fullb_experiments/routeB_val_loss_curve.png`
- `results/fullb_experiments/EXPERIMENT_RESULTS.md`
