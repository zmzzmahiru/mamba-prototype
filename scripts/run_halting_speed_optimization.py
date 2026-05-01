#!/usr/bin/env python3
"""Tiny benchmark for halting_only sequence-level early exit."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "halting_speed_optimization"
LOG_DIR = OUT_DIR / "logs"

SEEDS = (0, 1, 2)
HALT_THRESHOLDS = (0.5, 0.7, 0.9)

RAW_FIELDS = [
    "config",
    "variant",
    "seed",
    "halt_threshold",
    "enable_halting_early_exit",
    "benchmark_tok_s",
    "wall_clock_ms",
    "peak_gpu_memory",
    "mean_executed_steps",
    "mean_actual_executed_steps",
    "max_refinement_steps",
    "early_exit_trigger_rate",
    "mean_active_fraction",
    "mean_halted_fraction",
    "mean_halt_step",
    "logit_l2_vs_plain_mamba3",
    "logit_mae_vs_plain_mamba3",
    "output_json",
]

SUMMARY_FIELDS = [
    "config",
    "halt_threshold",
    "num_runs",
    "benchmark_tok_s_mean",
    "benchmark_tok_s_std",
    "wall_clock_ms_mean",
    "wall_clock_ms_std",
    "speed_vs_plain",
    "peak_gpu_memory_mean",
    "mean_executed_steps_mean",
    "mean_actual_executed_steps_mean",
    "max_refinement_steps_mean",
    "early_exit_trigger_rate_mean",
    "mean_active_fraction_mean",
    "mean_halted_fraction_mean",
    "mean_halt_step_mean",
]


def run_command(cmd: list[str], log_path: Path) -> None:
    print(" ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See {log_path}")


def benchmark_command(output_json: Path, seed: int, variant: str, halt_threshold: float | None, early_exit: bool) -> list[str]:
    cmd = [
        sys.executable,
        "benchmarks/benchmark_generation_mamba_simple.py",
        "--prototype",
        "--variant",
        variant,
        "--batch",
        "1",
        "--promptlen",
        "16",
        "--d-model",
        "128",
        "--n-layer",
        "2",
        "--d-state",
        "32",
        "--headdim",
        "32",
        "--repeats",
        "5",
        "--seed",
        str(seed),
        "--dtype",
        "float16",
        "--output-json",
        str(output_json),
    ]
    if halt_threshold is not None:
        cmd.extend(["--halt-threshold", str(halt_threshold)])
    if early_exit:
        cmd.append("--enable-halting-early-exit")
    return cmd


def read_result(path: Path, config: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "config": config,
        "variant": data.get("variant"),
        "seed": data.get("seed"),
        "halt_threshold": data.get("halting_threshold") if config != "plain" else "",
        "enable_halting_early_exit": data.get("enable_halting_early_exit", False),
        "benchmark_tok_s": data.get("benchmark_tok_s", data.get("benchmark_tokens_per_second")),
        "wall_clock_ms": data.get("wall_clock_ms", data.get("average_forward_time_ms")),
        "peak_gpu_memory": data.get("peak_gpu_memory", data.get("peak_gpu_memory_mb")),
        "mean_executed_steps": data.get("mean_executed_steps"),
        "mean_actual_executed_steps": data.get("mean_actual_executed_steps"),
        "max_refinement_steps": data.get("max_refinement_steps"),
        "early_exit_trigger_rate": data.get("early_exit_trigger_rate"),
        "mean_active_fraction": data.get("mean_active_fraction"),
        "mean_halted_fraction": data.get("mean_halted_fraction"),
        "mean_halt_step": data.get("mean_halt_step"),
        "logit_l2_vs_plain_mamba3": data.get("logit_l2_vs_plain_mamba3"),
        "logit_mae_vs_plain_mamba3": data.get("logit_mae_vs_plain_mamba3"),
        "output_json": str(path.relative_to(ROOT)),
    }


def finite(values: list[object]) -> list[float]:
    out = []
    for value in values:
        if value in ("", None):
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return out


def mean(values: list[object]) -> float | None:
    vals = finite(values)
    return statistics.fmean(vals) if vals else None


def std(values: list[object]) -> float | None:
    vals = finite(values)
    if not vals:
        return None
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    plain_speed = mean([row["benchmark_tok_s"] for row in rows if row["config"] == "plain"])
    groups: list[tuple[str, object]] = [("plain", "")]
    for config in ("halting_only_original", "halting_only_early_exit"):
        for threshold in HALT_THRESHOLDS:
            groups.append((config, threshold))

    summaries = []
    for config, threshold in groups:
        group = [
            row
            for row in rows
            if row["config"] == config and (config == "plain" or float(row["halt_threshold"]) == float(threshold))
        ]
        if not group:
            continue
        tok_s = mean([row["benchmark_tok_s"] for row in group])
        summaries.append(
            {
                "config": config,
                "halt_threshold": threshold,
                "num_runs": len(group),
                "benchmark_tok_s_mean": tok_s,
                "benchmark_tok_s_std": std([row["benchmark_tok_s"] for row in group]),
                "wall_clock_ms_mean": mean([row["wall_clock_ms"] for row in group]),
                "wall_clock_ms_std": std([row["wall_clock_ms"] for row in group]),
                "speed_vs_plain": tok_s / plain_speed if tok_s is not None and plain_speed else None,
                "peak_gpu_memory_mean": mean([row["peak_gpu_memory"] for row in group]),
                "mean_executed_steps_mean": mean([row["mean_executed_steps"] for row in group]),
                "mean_actual_executed_steps_mean": mean([row["mean_actual_executed_steps"] for row in group]),
                "max_refinement_steps_mean": mean([row["max_refinement_steps"] for row in group]),
                "early_exit_trigger_rate_mean": mean([row["early_exit_trigger_rate"] for row in group]),
                "mean_active_fraction_mean": mean([row["mean_active_fraction"] for row in group]),
                "mean_halted_fraction_mean": mean([row["mean_halted_fraction"] for row in group]),
                "mean_halt_step_mean": mean([row["mean_halt_step"] for row in group]),
            }
        )
    return summaries


def row_for(summary: list[dict[str, object]], config: str, threshold: float | str) -> dict[str, object] | None:
    for row in summary:
        if row["config"] == config and str(row["halt_threshold"]) == str(threshold):
            return row
    return None


def write_report(summary: list[dict[str, object]]) -> None:
    plain = row_for(summary, "plain", "")
    lines = [
        "# Halting Speed Optimization",
        "",
        "No WikiText-103 training was launched. This run only benchmarks tiny local prototype forwards.",
        "",
        "## Forward-Path Inspection",
        "",
        "- Mamba is called in `Mamba3._forward_conditional()` via `self._forward_mamba(current_states, seq_idx=seq_idx)`.",
        "- Before this change, the halt probabilities and `next_active_mask` were computed before Mamba, but Mamba was still called after the mask update even if every token had just halted.",
        "- Halting skipped state writeback with `torch.where(next_active_mask.unsqueeze(-1), mamba_out, current_states)`. It did not skip the expensive full-sequence Mamba computation for the current refinement step.",
        "- The loop runs up to `prototype_num_refinement_steps` refinement steps. In the tiny benchmark this is 2. The old path could break after all tokens halted, but only after paying for the Mamba call in that iteration.",
        "",
        "## Change",
        "",
        "- Added opt-in `enable_halting_early_exit` / `--enable-halting-early-exit`.",
        "- When enabled, the conditional path checks `next_active_mask.any()` before the Mamba call. If no tokens remain active, it breaks the refinement loop and records the early exit.",
        "- The default remains disabled, so `halting_only_original` is still available as a baseline.",
        "",
        "## Tiny Benchmark Setup",
        "",
        "- `batch=1`, `promptlen=16`, `d_model=128`, `n_layer=2`, `d_state=32`, `headdim=32`, `repeats=5`.",
        "- Seeds: 0, 1, 2.",
        "- Halting thresholds: 0.5, 0.7, 0.9.",
        "- Outputs: `raw_results.csv`, `summary_results.csv`, per-run JSON files, and logs in `logs/`.",
        "",
        "## Summary",
        "",
        "| Config | Halt | Runs | Tok/s | Speed vs plain | Avg ms | Actual steps | Early exit rate | Active frac | Halted frac | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {config} | {halt} | {runs} | {tok_s:.2f} | {speed:.3f} | {ms:.2f} | {actual} | {exit_rate} | {active} | {halted} | {mem} |".format(
                config=row["config"],
                halt=row["halt_threshold"] if row["halt_threshold"] != "" else "NA",
                runs=row["num_runs"],
                tok_s=float(row["benchmark_tok_s_mean"]),
                speed=float(row["speed_vs_plain"]) if row["speed_vs_plain"] is not None else 1.0,
                ms=float(row["wall_clock_ms_mean"]),
                actual=f"{float(row['mean_actual_executed_steps_mean']):.3f}" if row["mean_actual_executed_steps_mean"] is not None else "NA",
                exit_rate=f"{float(row['early_exit_trigger_rate_mean']):.3f}" if row["early_exit_trigger_rate_mean"] is not None else "NA",
                active=f"{float(row['mean_active_fraction_mean']):.3f}" if row["mean_active_fraction_mean"] is not None else "NA",
                halted=f"{float(row['mean_halted_fraction_mean']):.3f}" if row["mean_halted_fraction_mean"] is not None else "NA",
                mem=f"{float(row['peak_gpu_memory_mean']):.2f}" if row["peak_gpu_memory_mean"] is not None else "NA",
            )
        )
    lines.extend(["", "## Interpretation", ""])
    if plain is not None:
        lines.append(f"- Plain mean throughput: {float(plain['benchmark_tok_s_mean']):.2f} tok/s.")
    for threshold in HALT_THRESHOLDS:
        original = row_for(summary, "halting_only_original", threshold)
        early = row_for(summary, "halting_only_early_exit", threshold)
        if original is None or early is None:
            continue
        delta = float(early["benchmark_tok_s_mean"]) / float(original["benchmark_tok_s_mean"])
        lines.append(
            f"- h={threshold}: early exit rate {float(early['early_exit_trigger_rate_mean']):.3f}; "
            f"actual steps {float(original['mean_actual_executed_steps_mean']):.3f} -> {float(early['mean_actual_executed_steps_mean']):.3f}; "
            f"tok/s ratio vs original {delta:.3f}."
        )
    lines.extend(
        [
            "",
            "Sequence-level early exit can only skip an entire refinement step after all tokens in the sequence halt. It cannot skip compute for partially halted sequences, so the next required optimization after this minimal path is token-level sparse execution or block-level compaction. Sample-level skipping matters for batch sizes above 1, but this tiny benchmark uses batch 1, so the relevant missing capability is within-sequence compaction/sparsity.",
        ]
    )
    (OUT_DIR / "HALTING_SPEED_OPTIMIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, list[str]]] = []
    for seed in SEEDS:
        out = OUT_DIR / f"plain_seed{seed}.json"
        jobs.append(("plain", out, benchmark_command(out, seed, "plain", None, False)))
        for threshold in HALT_THRESHOLDS:
            safe_threshold = str(threshold).replace(".", "p")
            out = OUT_DIR / f"halting_only_original_h{safe_threshold}_seed{seed}.json"
            jobs.append(("halting_only_original", out, benchmark_command(out, seed, "halting_only", threshold, False)))
            out = OUT_DIR / f"halting_only_early_exit_h{safe_threshold}_seed{seed}.json"
            jobs.append(("halting_only_early_exit", out, benchmark_command(out, seed, "halting_only", threshold, True)))

    rows = []
    for config, output_json, cmd in jobs:
        log_path = LOG_DIR / f"{output_json.stem}.log"
        run_command(cmd, log_path)
        rows.append(read_result(output_json, config))

    summary = summarize(rows)
    write_csv(OUT_DIR / "raw_results.csv", rows, RAW_FIELDS)
    write_csv(OUT_DIR / "summary_results.csv", summary, SUMMARY_FIELDS)
    write_report(summary)
    print(f"Wrote {OUT_DIR / 'raw_results.csv'}")
    print(f"Wrote {OUT_DIR / 'summary_results.csv'}")
    print(f"Wrote {OUT_DIR / 'HALTING_SPEED_OPTIMIZATION.md'}")


if __name__ == "__main__":
    main()
