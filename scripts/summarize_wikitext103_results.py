import csv
import json
import math
import argparse
from pathlib import Path
from statistics import mean, stdev

METRICS = [
    "val_loss",
    "val_ppl",
    "train_tokens_per_second",
    "mean_attention_trigger_rate",
    "mean_active_fraction",
    "mean_halted_fraction",
    "mean_halt_step",
    "mean_executed_steps",
]


def parse_config_name(path: Path):
    stem = path.stem
    if stem.startswith("wikitext103_plain_seed"):
        return "plain"
    if stem.startswith("wikitext103_always_attention_seed"):
        return "always_attention"
    if stem.startswith("wikitext103_halting_only_h07_seed"):
        return "halting_only_h07"
    if stem.startswith("wikitext103_full_h07_seed"):
        return "full_h07"
    return stem


def load_runs():
    runs = []
    for path in sorted(results_dir.glob("wikitext103_*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["config_name"] = parse_config_name(path)
        data["result_file"] = path.name
        runs.append(data)
    return runs


def to_scalar(value):
    return "" if isinstance(value, list) else value


def safe_mean(values):
    return mean(values) if values else float("nan")


def safe_std(values):
    if len(values) <= 1:
        return 0.0 if values else float("nan")
    return stdev(values)


def write_run_csv(runs):
    fieldnames = [
        "config_name",
        "variant",
        "seed",
        "result_file",
        "parameter_count",
        "train_loss",
        "train_ppl",
        "train_tokens_per_second",
        "train_step_time_ms",
        "val_loss",
        "val_ppl",
        "eval_tokens_per_second",
        "eval_step_time_ms",
        "mean_attention_trigger_rate",
        "mean_active_fraction",
        "mean_halted_fraction",
        "mean_halt_step",
        "mean_executed_steps",
    ]
    with RUN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {key: to_scalar(run.get(key, "")) for key in fieldnames}
            writer.writerow(row)


def aggregate_runs(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["config_name"], []).append(run)

    aggregated_rows = []
    for config_name, config_runs in sorted(grouped.items()):
        row = {"config_name": config_name, "num_runs": len(config_runs)}
        for metric in METRICS:
            values = [
                float(run[metric])
                for run in config_runs
                if metric in run and not isinstance(run[metric], list)
            ]
            row[f"{metric}_mean"] = safe_mean(values)
            row[f"{metric}_std"] = safe_std(values)
        aggregated_rows.append(row)
    return aggregated_rows


def write_agg_csv(aggregated_rows):
    fieldnames = ["config_name", "num_runs"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with AGG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.4f}"


def write_summary_md(aggregated_rows):
    rows = sorted(aggregated_rows, key=lambda row: row["val_loss_mean"])
    lines = [
        "# WikiText-103 Confirmation Summary",
        "",
        "| Config | Val Loss (mean+/-std) | Val PPL (mean+/-std) | Train Tok/s (mean+/-std) | Trigger Rate | Active Fraction | Halted Fraction | Halt Step | Executed Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {vlm}+/-{vls} | {ppm}+/-{pps} | {ttm}+/-{tts} | {trm}+/-{trs} | {afm}+/-{afs} | {hfm}+/-{hfs} | {hsm}+/-{hss} | {esm}+/-{ess} |".format(
                config=row["config_name"],
                vlm=fmt(row["val_loss_mean"]),
                vls=fmt(row["val_loss_std"]),
                ppm=fmt(row["val_ppl_mean"]),
                pps=fmt(row["val_ppl_std"]),
                ttm=fmt(row["train_tokens_per_second_mean"]),
                tts=fmt(row["train_tokens_per_second_std"]),
                trm=fmt(row["mean_attention_trigger_rate_mean"]),
                trs=fmt(row["mean_attention_trigger_rate_std"]),
                afm=fmt(row["mean_active_fraction_mean"]),
                afs=fmt(row["mean_active_fraction_std"]),
                hfm=fmt(row["mean_halted_fraction_mean"]),
                hfs=fmt(row["mean_halted_fraction_std"]),
                hsm=fmt(row["mean_halt_step_mean"]),
                hss=fmt(row["mean_halt_step_std"]),
                esm=fmt(row["mean_executed_steps_mean"]),
                ess=fmt(row["mean_executed_steps_std"]),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize WikiText-103 JSON results from one experiment directory")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    global results_dir, RUN_CSV, AGG_CSV, SUMMARY_MD
    results_dir = Path(args.results_dir)
    RUN_CSV = results_dir / "wikitext103_runs.csv"
    AGG_CSV = results_dir / "wikitext103_aggregated.csv"
    SUMMARY_MD = results_dir / "wikitext103_summary.md"

    runs = load_runs()
    if not runs:
        raise FileNotFoundError(f"No matching result JSON files found in {results_dir}.")
    write_run_csv(runs)
    aggregated_rows = aggregate_runs(runs)
    write_agg_csv(aggregated_rows)
    write_summary_md(aggregated_rows)
    print(f"Wrote {RUN_CSV}")
    print(f"Wrote {AGG_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
