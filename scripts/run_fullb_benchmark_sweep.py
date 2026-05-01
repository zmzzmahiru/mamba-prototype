#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


RESULT_DIR = Path("results/fullb_experiments")
LOG_DIR = RESULT_DIR / "logs"
RAW_CSV = RESULT_DIR / "raw_results.csv"
SUMMARY_CSV = RESULT_DIR / "summary_results.csv"

BASE_ARGS = [
    ".venv/bin/python",
    "benchmarks/benchmark_generation_mamba_simple.py",
    "--prototype",
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
    "20",
    "--dtype",
    "float16",
]

RAW_FIELDS = [
    "variant_label",
    "variant",
    "seed",
    "halting_threshold",
    "router_threshold",
    "benchmark_tokens_per_second",
    "average_forward_time_ms",
    "peak_gpu_memory_mb",
    "mean_attention_trigger_rate",
    "mean_active_fraction",
    "mean_halted_fraction",
    "mean_halt_step",
    "mean_executed_steps",
    "logit_l2_vs_plain_mamba3",
    "logit_mae_vs_plain_mamba3",
    "json_path",
    "log_path",
    "command",
]

SUMMARY_FIELDS = [
    "variant_label",
    "variant",
    "halting_threshold",
    "router_threshold",
    "num_runs",
    "mean_tokens_per_second",
    "std_tokens_per_second",
    "mean_speed_ratio_vs_plain",
    "mean_forward_time_ms",
    "mean_attention_trigger_rate",
    "mean_active_fraction",
    "mean_halted_fraction",
    "mean_halt_step",
    "mean_peak_gpu_memory_mb",
    "mean_logit_l2_vs_plain_mamba3",
    "mean_logit_mae_vs_plain_mamba3",
]


def sweep_configs():
    for seed in (0, 1, 2):
        yield {
            "variant_label": "plain",
            "variant": "plain",
            "seed": seed,
            "halt": None,
            "router": None,
        }
        yield {
            "variant_label": "always_attention",
            "variant": "always_attention",
            "seed": seed,
            "halt": None,
            "router": None,
        }
        for halt in (0.5, 0.7, 0.9):
            yield {
                "variant_label": "halting_only",
                "variant": "halting_only",
                "seed": seed,
                "halt": halt,
                "router": None,
            }
        for halt in (0.5, 0.7, 0.9):
            for router in (0.3, 0.5, 0.7):
                yield {
                    "variant_label": "fullB",
                    "variant": "full",
                    "seed": seed,
                    "halt": halt,
                    "router": router,
                }


def fmt_threshold(value):
    return "default" if value is None else str(value).replace(".", "p")


def file_stem(config):
    parts = [
        "sweep",
        config["variant_label"],
        "b1",
        "p16",
        f"seed{config['seed']}",
    ]
    if config["halt"] is not None:
        parts.append(f"h{fmt_threshold(config['halt'])}")
    if config["router"] is not None:
        parts.append(f"r{fmt_threshold(config['router'])}")
    return "_".join(parts)


def command_for(config, json_path):
    cmd = BASE_ARGS.copy()
    cmd.extend(["--variant", config["variant"]])
    cmd.extend(["--seed", str(config["seed"])])
    if config["halt"] is not None:
        cmd.extend(["--halt-threshold", str(config["halt"])])
    if config["router"] is not None:
        cmd.extend(["--router-threshold", str(config["router"])])
    cmd.extend(["--output-json", str(json_path)])
    return cmd


def run_sweep():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for config in sweep_configs():
        stem = file_stem(config)
        json_path = RESULT_DIR / f"{stem}.json"
        log_path = LOG_DIR / f"{stem}.log"
        cmd = command_for(config, json_path)
        print("RUN", " ".join(cmd), flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(cmd) + "\n")
            log_file.flush()
            completed = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def number(value):
    return "" if value is None else value


def read_rows():
    rows = []
    for config in sweep_configs():
        stem = file_stem(config)
        json_path = RESULT_DIR / f"{stem}.json"
        log_path = LOG_DIR / f"{stem}.log"
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        row = {field: "" for field in RAW_FIELDS}
        row.update(
            {
                "variant_label": config["variant_label"],
                "variant": data.get("variant"),
                "seed": data.get("seed"),
                "halting_threshold": data.get("halting_threshold"),
                "router_threshold": data.get("router_threshold"),
                "benchmark_tokens_per_second": data.get("benchmark_tokens_per_second"),
                "average_forward_time_ms": data.get("average_forward_time_ms"),
                "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb"),
                "mean_attention_trigger_rate": data.get("mean_attention_trigger_rate"),
                "mean_active_fraction": data.get("mean_active_fraction"),
                "mean_halted_fraction": data.get("mean_halted_fraction"),
                "mean_halt_step": data.get("mean_halt_step"),
                "mean_executed_steps": data.get("mean_executed_steps"),
                "logit_l2_vs_plain_mamba3": data.get("logit_l2_vs_plain_mamba3"),
                "logit_mae_vs_plain_mamba3": data.get("logit_mae_vs_plain_mamba3"),
                "json_path": str(json_path),
                "log_path": str(log_path),
                "command": data.get("command"),
            }
        )
        rows.append(row)
    return rows


def write_raw(rows):
    with RAW_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(v) for v in values if v != "" and v is not None]
    if not values:
        return ""
    return sum(values) / len(values)


def std(values):
    values = [float(v) for v in values if v != "" and v is not None]
    if len(values) < 2:
        return 0.0 if values else ""
    avg = sum(values) / len(values)
    return (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def write_summary(rows):
    plain_mean = mean(
        row["benchmark_tokens_per_second"]
        for row in rows
        if row["variant_label"] == "plain"
    )
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["variant_label"],
            row["variant"],
            row["halting_threshold"],
            row["router_threshold"],
        )
        groups[key].append(row)

    summary_rows = []
    for key in sorted(groups.keys(), key=lambda item: (item[0], str(item[2]), str(item[3]))):
        group_rows = groups[key]
        tokens = [row["benchmark_tokens_per_second"] for row in group_rows]
        mean_tps = mean(tokens)
        summary_rows.append(
            {
                "variant_label": key[0],
                "variant": key[1],
                "halting_threshold": key[2],
                "router_threshold": key[3],
                "num_runs": len(group_rows),
                "mean_tokens_per_second": mean_tps,
                "std_tokens_per_second": std(tokens),
                "mean_speed_ratio_vs_plain": mean_tps / plain_mean if plain_mean else "",
                "mean_forward_time_ms": mean(row["average_forward_time_ms"] for row in group_rows),
                "mean_attention_trigger_rate": mean(row["mean_attention_trigger_rate"] for row in group_rows),
                "mean_active_fraction": mean(row["mean_active_fraction"] for row in group_rows),
                "mean_halted_fraction": mean(row["mean_halted_fraction"] for row in group_rows),
                "mean_halt_step": mean(row["mean_halt_step"] for row in group_rows),
                "mean_peak_gpu_memory_mb": mean(row["peak_gpu_memory_mb"] for row in group_rows),
                "mean_logit_l2_vs_plain_mamba3": mean(row["logit_l2_vs_plain_mamba3"] for row in group_rows),
                "mean_logit_mae_vs_plain_mamba3": mean(row["logit_mae_vs_plain_mamba3"] for row in group_rows),
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def aggregate_only():
    rows = read_rows()
    write_raw(rows)
    write_summary(rows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run_sweep()
        aggregate_only()
    elif mode == "aggregate":
        aggregate_only()
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
