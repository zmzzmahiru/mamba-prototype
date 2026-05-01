#!/usr/bin/env python3
"""Build a small Route B checkpoint curve from existing WikiText result JSONs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "fullb_experiments"

RUN_DIRS = {
    200: ROOT / "results" / "200step_largerdata_confirm",
    500: ROOT / "results" / "routeB_plain_vs_full_500step",
    1000: ROOT / "results" / "routeB_plain_vs_full_1000step",
}

CONFIG_FILES = {
    "plain": "wikitext103_plain_seed{seed}.json",
    "full_h07": "wikitext103_full_h07_seed{seed}.json",
}

RAW_FIELDS = [
    "config",
    "variant",
    "seed",
    "step",
    "source_file",
    "val_loss",
    "val_ppl",
    "train_tok_s",
    "attention_trigger_rate",
    "active_fraction",
    "halted_fraction",
    "mean_halt_step",
]

SUMMARY_FIELDS = [
    "config",
    "step",
    "num_runs",
    "val_loss_mean",
    "val_loss_std",
    "val_ppl_mean",
    "val_ppl_std",
    "train_tok_s_mean",
    "train_tok_s_std",
    "attention_trigger_rate_mean",
    "active_fraction_mean",
    "halted_fraction_mean",
    "mean_halt_step_mean",
]


def finite(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and math.isfinite(v)]


def mean(values: list[float | None]) -> float | None:
    vals = finite(values)
    return statistics.fmean(vals) if vals else None


def std(values: list[float | None]) -> float | None:
    vals = finite(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None


def fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step, run_dir in RUN_DIRS.items():
        for config, pattern in CONFIG_FILES.items():
            for seed in (0, 1, 2):
                path = run_dir / pattern.format(seed=seed)
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                rows.append(
                    {
                        "config": config,
                        "variant": data.get("variant", ""),
                        "seed": seed,
                        "step": step,
                        "source_file": str(path.relative_to(ROOT)),
                        "val_loss": data.get("val_loss"),
                        "val_ppl": data.get("val_ppl"),
                        "train_tok_s": data.get("train_tokens_per_second"),
                        "attention_trigger_rate": data.get("mean_attention_trigger_rate"),
                        "active_fraction": data.get("mean_active_fraction"),
                        "halted_fraction": data.get("mean_halted_fraction"),
                        "mean_halt_step": data.get("mean_halt_step"),
                    }
                )
    return sorted(rows, key=lambda r: (r["config"], r["step"], r["seed"]))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for config in sorted({str(r["config"]) for r in rows}):
        for step in sorted({int(r["step"]) for r in rows if r["config"] == config}):
            group = [r for r in rows if r["config"] == config and r["step"] == step]
            summaries.append(
                {
                    "config": config,
                    "step": step,
                    "num_runs": len(group),
                    "val_loss_mean": mean([r["val_loss"] for r in group]),  # type: ignore[list-item]
                    "val_loss_std": std([r["val_loss"] for r in group]),  # type: ignore[list-item]
                    "val_ppl_mean": mean([r["val_ppl"] for r in group]),  # type: ignore[list-item]
                    "val_ppl_std": std([r["val_ppl"] for r in group]),  # type: ignore[list-item]
                    "train_tok_s_mean": mean([r["train_tok_s"] for r in group]),  # type: ignore[list-item]
                    "train_tok_s_std": std([r["train_tok_s"] for r in group]),  # type: ignore[list-item]
                    "attention_trigger_rate_mean": mean([r["attention_trigger_rate"] for r in group]),  # type: ignore[list-item]
                    "active_fraction_mean": mean([r["active_fraction"] for r in group]),  # type: ignore[list-item]
                    "halted_fraction_mean": mean([r["halted_fraction"] for r in group]),  # type: ignore[list-item]
                    "mean_halt_step_mean": mean([r["mean_halt_step"] for r in group]),  # type: ignore[list-item]
                }
            )
    return summaries


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        start = y * width
        for r, g, b in pixels[start : start + width]:
            raw.extend((r, g, b))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def draw_line(pixels: list[tuple[int, int, int]], width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for yy in range(max(0, y0 - 1), min(height, y0 + 2)):
            for xx in range(max(0, x0 - 1), min(width, x0 + 2)):
                pixels[yy * width + xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_curve(path: Path, summaries: list[dict[str, object]]) -> None:
    width, height = 900, 560
    left, right, top, bottom = 85, 35, 45, 75
    pixels = [(255, 255, 255)] * (width * height)

    vals = [float(s["val_loss_mean"]) for s in summaries if s["val_loss_mean"] is not None]
    ymin = math.floor((min(vals) - 0.05) * 10) / 10
    ymax = math.ceil((max(vals) + 0.05) * 10) / 10

    def px(step: int) -> int:
        return round(left + (step - 200) / 800 * (width - left - right))

    def py(loss: float) -> int:
        return round(top + (ymax - loss) / (ymax - ymin) * (height - top - bottom))

    axis = (35, 35, 35)
    grid = (225, 225, 225)
    for step in (200, 500, 1000):
        x = px(step)
        draw_line(pixels, width, height, x, top, x, height - bottom, grid)
    for i in range(6):
        y = top + round(i * (height - top - bottom) / 5)
        draw_line(pixels, width, height, left, y, width - right, y, grid)
    draw_line(pixels, width, height, left, top, left, height - bottom, axis)
    draw_line(pixels, width, height, left, height - bottom, width - right, height - bottom, axis)

    colors = {"plain": (32, 96, 160), "full_h07": (190, 65, 45)}
    for config in ("plain", "full_h07"):
        points = [
            (int(s["step"]), float(s["val_loss_mean"]))
            for s in summaries
            if s["config"] == config and s["val_loss_mean"] is not None
        ]
        points.sort()
        for (s0, l0), (s1, l1) in zip(points, points[1:]):
            draw_line(pixels, width, height, px(s0), py(l0), px(s1), py(l1), colors[config])
        for step, loss in points:
            x, y = px(step), py(loss)
            for yy in range(max(0, y - 5), min(height, y + 6)):
                for xx in range(max(0, x - 5), min(width, x + 6)):
                    if (xx - x) ** 2 + (yy - y) ** 2 <= 25:
                        pixels[yy * width + xx] = colors[config]

    write_png(path, width, height, pixels)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    summaries = summarize(rows)
    write_csv(OUT_DIR / "routeB_checkpoint_curve_raw.csv", rows, RAW_FIELDS)
    write_csv(OUT_DIR / "routeB_checkpoint_curve_summary.csv", summaries, SUMMARY_FIELDS)
    draw_curve(OUT_DIR / "routeB_val_loss_curve.png", summaries)
    print(f"Wrote {len(rows)} raw rows and {len(summaries)} summary rows.")


if __name__ == "__main__":
    main()
