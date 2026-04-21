import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def parse_dtype(dtype_name):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def get_variant_settings(args):
    if args.variant == "plain":
        return {
            "enable_conditional_execution": False,
            "enable_attention_router": False,
            "enable_token_halting": False,
            "router_threshold": args.router_threshold,
        }
    if args.variant == "halting_only":
        return {
            "enable_conditional_execution": True,
            "enable_attention_router": False,
            "enable_token_halting": True,
            "router_threshold": args.router_threshold,
        }
    if args.variant == "always_attention":
        return {
            "enable_conditional_execution": True,
            "enable_attention_router": True,
            "enable_token_halting": False,
            "router_threshold": 0.0,
        }
    if args.variant == "full":
        return {
            "enable_conditional_execution": True,
            "enable_attention_router": True,
            "enable_token_halting": True,
            "router_threshold": args.router_threshold,
        }
    raise ValueError(f"Unsupported variant: {args.variant}")


def build_model(args, device, dtype):
    settings = get_variant_settings(args)
    config = MambaConfig(
        d_model=args.d_model,
        d_intermediate=0,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        ssm_cfg={
            "layer": "Mamba3",
            "d_state": args.d_state,
            "headdim": args.headdim,
            "expand": args.expand,
            "chunk_size": args.chunk_size,
            "enable_conditional_execution": settings["enable_conditional_execution"],
            "enable_attention_router": settings["enable_attention_router"],
            "enable_token_halting": settings["enable_token_halting"],
            "prototype_num_refinement_steps": args.refinement_steps,
            "router_threshold": settings["router_threshold"],
            "router_temperature": args.router_temperature,
            "halt_threshold": args.halt_threshold,
            "attention_num_heads": args.attention_num_heads,
            "collect_prototype_stats": True,
        },
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        tie_embeddings=False,
    )
    return MambaLMHeadModel(config, device=device, dtype=dtype)


def collect_prototype_stats(model):
    layer_stats = []
    for layer in model.backbone.layers:
        mixer = getattr(layer, "mixer", None)
        stats = getattr(mixer, "_last_prototype_stats", None)
        if stats is not None:
            layer_stats.append(stats)
    if not layer_stats:
        return {}

    def mean_scalar(key):
        return sum(stats[key] for stats in layer_stats) / len(layer_stats)

    def mean_position_list(key):
        tensors = [torch.tensor(stats[key], dtype=torch.float32) for stats in layer_stats]
        return torch.stack(tensors, dim=0).mean(dim=0).tolist()

    return {
        "num_layers_with_stats": len(layer_stats),
        "mean_attention_trigger_rate": mean_scalar("attention_trigger_rate"),
        "mean_active_fraction": mean_scalar("mean_active_fraction"),
        "mean_halted_fraction": mean_scalar("mean_halted_fraction"),
        "mean_halt_step": mean_scalar("mean_halt_step"),
        "mean_executed_steps": mean_scalar("executed_steps"),
        "mean_halt_step_by_position": mean_position_list("mean_halt_step_by_position"),
        "halted_fraction_by_position": mean_position_list("halted_fraction_by_position"),
    }


def build_block_dataset(tokenizer, texts, block_size, max_tokens=None):
    joined = "\n\n".join(text for text in texts if text and text.strip())
    token_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    n_blocks = len(token_ids) // (block_size + 1)
    if n_blocks == 0:
        raise ValueError("Not enough tokens to form one training block. Increase text samples or decrease block size.")
    trimmed = token_ids[: n_blocks * (block_size + 1)]
    tokens = torch.tensor(trimmed, dtype=torch.long).view(n_blocks, block_size + 1)
    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    return TensorDataset(input_ids, labels)


def load_wikitext_dataloaders(args, tokenizer):
    if load_dataset is None:
        raise ImportError("The 'datasets' package is required. Install it with `pip install datasets`.")

    train_split = f"train[:{args.train_text_samples}]"
    valid_split = f"validation[:{args.val_text_samples}]"
    train_dataset = load_dataset(args.dataset_name, args.dataset_config, split=train_split)
    valid_dataset = load_dataset(args.dataset_name, args.dataset_config, split=valid_split)

    train_blocks = build_block_dataset(
        tokenizer,
        train_dataset["text"],
        block_size=args.block_size,
        max_tokens=args.max_train_tokens,
    )
    valid_blocks = build_block_dataset(
        tokenizer,
        valid_dataset["text"],
        block_size=args.block_size,
        max_tokens=args.max_val_tokens,
    )
    train_loader = DataLoader(train_blocks, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_blocks, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader


def compute_lm_loss(logits, labels):
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))


def evaluate(model, dataloader, device, max_eval_batches=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    stats_accum = {}
    start = time.time()

    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids).logits
            loss = compute_lm_loss(logits, labels)
            total_loss += float(loss.item())
            total_batches += 1
            total_tokens += labels.numel()

            batch_stats = collect_prototype_stats(model)
            for key, value in batch_stats.items():
                stats_accum.setdefault(key, []).append(value)

    elapsed = max(time.time() - start, 1e-6)
    mean_loss = total_loss / max(total_batches, 1)
    results = {
        "val_loss": mean_loss,
        "val_ppl": math.exp(mean_loss),
        "eval_tokens_per_second": total_tokens / elapsed,
        "eval_step_time_ms": elapsed / max(total_batches, 1) * 1000,
    }

    for key, values in stats_accum.items():
        if isinstance(values[0], list):
            results[key] = torch.stack([torch.tensor(v, dtype=torch.float32) for v in values]).mean(dim=0).tolist()
        else:
            results[key] = sum(values) / len(values)
    return results


def train(args):
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = parse_dtype(args.dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    args.vocab_size = vocab_size

    train_loader, valid_loader = load_wikitext_dataloaders(args, tokenizer)
    model = build_model(args, device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_tokens = 0
    train_loss_sum = 0.0
    train_steps = 0
    start = time.time()

    train_iter = iter(train_loader)
    for step in range(1, args.max_train_steps + 1):
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids).logits
        loss = compute_lm_loss(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += float(loss.item())
        train_steps += 1
        total_tokens += labels.numel()

        if step % args.log_interval == 0 or step == args.max_train_steps:
            print(f"[train] step={step} loss={loss.item():.4f}")

    elapsed = max(time.time() - start, 1e-6)
    train_metrics = {
        "train_loss": train_loss_sum / max(train_steps, 1),
        "train_ppl": math.exp(train_loss_sum / max(train_steps, 1)),
        "train_tokens_per_second": total_tokens / elapsed,
        "train_step_time_ms": elapsed / max(train_steps, 1) * 1000,
    }

    eval_metrics = evaluate(model, valid_loader, device=device, max_eval_batches=args.max_eval_batches)
    results = {
        "variant": args.variant,
        "seed": args.seed,
        "device": device.type,
        "dtype": args.dtype,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "max_train_steps": args.max_train_steps,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **train_metrics,
        **eval_metrics,
    }

    print(json.dumps(results, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Minimal WikiText-103 training for local Mamba3 variants")
    parser.add_argument("--variant", type=str, required=True, choices=["plain", "halting_only", "always_attention", "full"])
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--attention-num-heads", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=2)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train-text-samples", type=int, default=200)
    parser.add_argument("--val-text-samples", type=int, default=100)
    parser.add_argument("--max-train-tokens", type=int, default=20000)
    parser.add_argument("--max-val-tokens", type=int, default=10000)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
