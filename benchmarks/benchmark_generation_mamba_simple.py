# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time

import torch

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


def build_prototype_model(args, device, dtype):
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
            "enable_conditional_execution": True,
            "enable_attention_router": not args.disable_attention,
            "enable_token_halting": not args.disable_halting,
            "prototype_num_refinement_steps": args.refinement_steps,
            "router_threshold": args.router_threshold,
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
        "mean_attention_trigger_rate": mean("attention_trigger_rate"),
        "mean_active_fraction": mean("mean_active_fraction"),
        "mean_halted_fraction": mean("mean_halted_fraction"),
        "mean_halt_step": mean("mean_halt_step"),
        "router_probs_first_layer": layer_stats[0]["router_probs"],
        "active_fractions_first_layer": layer_stats[0]["active_fractions"],
        "halted_fractions_first_layer": layer_stats[0]["halted_fractions"],
    }


def run_prototype_benchmark(args, device, dtype):
    print("Running local Mamba3 full-sequence prototype benchmark")
    model = build_prototype_model(args, device=device, dtype=dtype)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    input_ids = torch.randint(1, args.vocab_size, (args.batch, args.promptlen), dtype=torch.long, device=device)

    def fn():
        return model(input_ids).logits

    logits = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(args.repeats):
        logits = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    avg_ms = (time.time() - start) / args.repeats * 1000

    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Average forward time: {avg_ms:.0f}ms")

    stats = collect_prototype_stats(model)
    if stats is not None:
        print("Prototype stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


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
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="float16")
parser.add_argument("--prototype", action="store_true")
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--n-layer", type=int, default=4)
parser.add_argument("--vocab-size", type=int, default=1024)
parser.add_argument("--d-state", type=int, default=64)
parser.add_argument("--expand", type=int, default=2)
parser.add_argument("--headdim", type=int, default=64)
parser.add_argument("--chunk-size", type=int, default=64)
parser.add_argument("--refinement-steps", type=int, default=2)
parser.add_argument("--router-threshold", type=float, default=0.5)
parser.add_argument("--halt-threshold", type=float, default=0.5)
parser.add_argument("--attention-num-heads", type=int, default=4)
parser.add_argument("--disable-attention", action="store_true")
parser.add_argument("--disable-halting", action="store_true")
args = parser.parse_args()

device_name = args.device
if device_name == "cuda" and not torch.cuda.is_available():
    device_name = "cpu"
device = torch.device(device_name)
dtype = parse_dtype(args.dtype)

if args.prototype:
    run_prototype_benchmark(args, device=device, dtype=dtype)
else:
    run_generation_benchmark(args, device=device, dtype=dtype)
