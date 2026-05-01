# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def parse_dtype(dtype_name):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def get_local_variant_settings(args):
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


def build_local_model(args, device, dtype, variant=None):
    variant = args.variant if variant is None else variant
    settings = get_local_variant_settings(args)
    if variant != args.variant:
        original_variant = args.variant
        args.variant = variant
        settings = get_local_variant_settings(args)
        args.variant = original_variant
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
            "enable_halting_early_exit": args.enable_halting_early_exit,
            "attention_num_heads": args.attention_num_heads,
            "collect_prototype_stats": True,
        },
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(config, device=device, dtype=dtype)
    return model


def collect_prototype_stats(model):
    layer_stats = []
    for layer in model.backbone.layers:
        mixer = getattr(layer, "mixer", None)
        stats = getattr(mixer, "_last_prototype_stats", None)
        if stats is not None:
            layer_stats.append(stats)
    if not layer_stats:
        return None

    mean = lambda key: sum(stats[key] for stats in layer_stats) / len(layer_stats)
    return {
        "num_layers_with_stats": len(layer_stats),
        "mean_executed_steps": mean("executed_steps"),
        "mean_actual_executed_steps": mean("actual_executed_steps"),
        "max_refinement_steps": mean("max_refinement_steps"),
        "early_exit_trigger_rate": mean("early_exit_triggered"),
        "mean_attention_trigger_rate": mean("attention_trigger_rate"),
        "mean_active_fraction": mean("mean_active_fraction"),
        "mean_halted_fraction": mean("mean_halted_fraction"),
        "mean_halt_step": mean("mean_halt_step"),
        "router_probs_first_layer": layer_stats[0]["router_probs"],
        "active_fractions_first_layer": layer_stats[0]["active_fractions"],
        "halted_fractions_first_layer": layer_stats[0]["halted_fractions"],
    }


def run_local_benchmark(args, device, dtype):
    print(f"Running local Mamba3 full-sequence benchmark ({args.variant})")
    run_start = time.time()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_local_model(args, device=device, dtype=dtype, variant=args.variant)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    input_ids = torch.randint(1, args.vocab_size, (args.batch, args.promptlen), dtype=torch.long, device=device)

    def fn():
        return model(input_ids).logits

    logits = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(args.repeats):
        logits = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start
    avg_ms = elapsed / args.repeats * 1000
    benchmark_tokens_per_second = args.batch * args.promptlen * args.repeats / max(elapsed, 1e-6)
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )

    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Average forward time: {avg_ms:.0f}ms")
    print(f"Benchmark tokens/sec: {benchmark_tokens_per_second:.2f}")
    if peak_gpu_memory_mb is not None:
        print(f"Peak GPU memory allocated: {peak_gpu_memory_mb:.2f} MiB")

    stats = collect_prototype_stats(model)
    if stats is not None:
        print("Prototype stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    if args.variant != "plain":
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        reference_model = build_local_model(args, device=device, dtype=dtype, variant="plain")
        reference_model.eval()
        with torch.no_grad():
            reference_logits = reference_model(input_ids).logits
        drift_l2 = torch.norm((logits - reference_logits).float()).item()
        drift_mae = F.l1_loss(logits.float(), reference_logits.float()).item()
        print("Output drift vs plain Mamba3:")
        print(f"  logit_l2_vs_plain_mamba3: {drift_l2}")
        print(f"  logit_mae_vs_plain_mamba3: {drift_mae}")
    else:
        drift_l2 = None
        drift_mae = None

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "d_state": args.d_state,
        "headdim": args.headdim,
        "batch_size": args.batch,
        "promptlen": args.promptlen,
        "repeats": args.repeats,
        "halting_threshold": args.halt_threshold,
        "enable_halting_early_exit": args.enable_halting_early_exit,
        "router_threshold": 0.0 if args.variant == "always_attention" else args.router_threshold,
        "benchmark_tokens_per_second": benchmark_tokens_per_second,
        "benchmark_tok_s": benchmark_tokens_per_second,
        "average_forward_time_ms": avg_ms,
        "wall_clock_ms": avg_ms,
        "wall_clock_runtime_seconds": time.time() - run_start,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "peak_gpu_memory": peak_gpu_memory_mb,
        "logit_l2_vs_plain_mamba3": drift_l2,
        "logit_mae_vs_plain_mamba3": drift_mae,
    }
    if stats is not None:
        result.update(stats)

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")


def run_generation_benchmark(args, device, dtype):
    repeats = args.repeats
    print(f"Loading model {args.model_name}")
    is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
    if is_mamba:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(args.model_name, device=device.type, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device.type}, torch_dtype=dtype)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + args.genlen

    if is_mamba:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
    out = fn()
    if args.prompt is not None:
        print(tokenizer.batch_decode(out.sequences.tolist()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
    print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="float16")
parser.add_argument("--prototype", action="store_true")
parser.add_argument("--variant", type=str, default="full", choices=["plain", "halting_only", "always_attention", "full"])
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--n-layer", type=int, default=4)
parser.add_argument("--vocab-size", type=int, default=1024)
parser.add_argument("--d-state", type=int, default=64)
parser.add_argument("--expand", type=int, default=2)
parser.add_argument("--headdim", type=int, default=64)
parser.add_argument("--chunk-size", type=int, default=64)
parser.add_argument("--refinement-steps", type=int, default=2)
parser.add_argument("--router-threshold", type=float, default=0.5)
parser.add_argument("--router-temperature", type=float, default=1.0)
parser.add_argument("--halt-threshold", type=float, default=0.5)
parser.add_argument("--enable-halting-early-exit", action="store_true")
parser.add_argument("--attention-num-heads", type=int, default=4)
parser.add_argument("--output-json", type=str, default=None)
args = parser.parse_args()

device_name = args.device
if device_name == "cuda" and not torch.cuda.is_available():
    device_name = "cpu"
elif device_name == "cuda":
    device_name = "cuda:0"
device = torch.device(device_name)
dtype = parse_dtype(args.dtype)

torch.manual_seed(args.seed)

if args.prototype:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    run_local_benchmark(args, device=device, dtype=dtype)
else:
    run_generation_benchmark(args, device=device, dtype=dtype)
