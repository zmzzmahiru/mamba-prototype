#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="results/200step_confirm"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
  --dataset-name wikitext
  --dataset-config wikitext-103-raw-v1
  --tokenizer EleutherAI/gpt-neox-20b
  --device cuda
  --dtype float32
  --d-model 128
  --n-layer 2
  --d-state 32
  --headdim 32
  --expand 2
  --block-size 128
  --batch-size 2
  --eval-batch-size 2
  --max-train-steps 200
  --max-eval-batches 10
  --train-text-samples 100
  --val-text-samples 50
  --max-train-tokens 8192
  --max-val-tokens 4096
  --learning-rate 3e-4
)

run() {
  echo
  echo "==> $*"
  "$@"
}

for seed in 0 1 2; do
  run python train_wikitext103.py \
    --variant plain \
    --seed "$seed" \
    --output-json "${RESULTS_DIR}/wikitext103_plain_seed${seed}.json" \
    "${COMMON_ARGS[@]}"

  run python train_wikitext103.py \
    --variant always_attention \
    --seed "$seed" \
    --output-json "${RESULTS_DIR}/wikitext103_always_attention_seed${seed}.json" \
    "${COMMON_ARGS[@]}"

  run python train_wikitext103.py \
    --variant halting_only \
    --halt-threshold 0.7 \
    --seed "$seed" \
    --output-json "${RESULTS_DIR}/wikitext103_halting_only_h07_seed${seed}.json" \
    "${COMMON_ARGS[@]}"

  run python train_wikitext103.py \
    --variant full \
    --halt-threshold 0.7 \
    --router-threshold 0.5 \
    --router-temperature 1.0 \
    --seed "$seed" \
    --output-json "${RESULTS_DIR}/wikitext103_full_h07_seed${seed}.json" \
    "${COMMON_ARGS[@]}"
done
