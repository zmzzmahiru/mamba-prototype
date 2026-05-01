# Halting Speed Optimization

No WikiText-103 training was launched.

## Forward-Path Inspection

- Mamba is called in `Mamba3._forward_conditional()` via `self._forward_mamba(current_states, seq_idx=seq_idx)`.
- Before this change, halt probabilities and `next_active_mask` were computed before Mamba, but Mamba was still called after the mask update even if every token had just halted.
- Halting skipped state writeback with `torch.where(next_active_mask.unsqueeze(-1), mamba_out, current_states)`. It did not skip the expensive full-sequence Mamba computation for the current refinement step.
- The loop runs up to `prototype_num_refinement_steps` refinement steps. In the tiny benchmark configuration this is 2. The old path could break after all tokens halted, but only after paying for the Mamba call in that iteration.

## Implemented Change

- Added opt-in `enable_halting_early_exit` / `--enable-halting-early-exit`.
- When enabled, the conditional path checks `next_active_mask.any()` before the Mamba call. If no tokens remain active, it breaks the refinement loop and records the early exit.
- The default remains disabled, so `halting_only_original` is still available as a baseline.

## New Logging Fields

- `mean_actual_executed_steps`
- `max_refinement_steps`
- `early_exit_trigger_rate`
- `mean_active_fraction`
- `mean_halted_fraction`
- `benchmark_tok_s`
- `wall_clock_ms`
- `peak_gpu_memory`

## Tiny Benchmark Status

The requested tiny benchmark matrix was attempted with:

- `batch=1`, `promptlen=16`, `d_model=128`, `n_layer=2`, `d_state=32`, `headdim=32`, `repeats=5`
- seeds 0, 1, 2
- halting thresholds 0.5, 0.7, 0.9
- configs: `plain`, `halting_only_original`, `halting_only_early_exit`

The run is currently blocked because PyTorch reports no CUDA devices in this session:

- `.venv` PyTorch: `torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`
- system PyTorch: `torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`

The Mamba3 prototype path is not CPU-safe for this benchmark because its Triton RMSNorm wrapper enters a CUDA device context. The first attempted run fell back to CPU and failed in `mamba_ssm/ops/triton/layernorm_gated.py`.

## Control-Flow Microcheck

A CPU-safe microcheck was run with `halt_threshold=0.0` and `enable_halting_early_exit=True`, which forces all tokens to halt before Mamba is called. It returned the input unchanged and recorded:

- `executed_steps`: 1
- `actual_executed_steps`: 0
- `max_refinement_steps`: 2
- `early_exit_triggered`: true
- `mean_active_fraction`: 0.0
- `mean_halted_fraction`: 1.0

This verifies the new pre-Mamba early-exit control flow, but it is not a throughput benchmark.

## Files Prepared

- Benchmark driver: `scripts/run_halting_speed_optimization.py`
- Intended outputs: `results/halting_speed_optimization/raw_results.csv`, `results/halting_speed_optimization/summary_results.csv`, per-run JSON files, and logs in `results/halting_speed_optimization/logs/`

## Expected Interpretation Once CUDA Is Available

Sequence-level early exit can only skip an entire refinement step after all tokens in the sequence halt. It cannot skip compute for partially halted sequences. If early exit does not improve throughput materially, the next required optimization is token-level sparse execution or block-level compaction. Sample-level skipping matters for batch sizes above 1, but the requested tiny benchmark uses batch 1, so the relevant missing capability is within-sequence compaction/sparsity.
