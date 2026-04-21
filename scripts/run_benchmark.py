#!/usr/bin/env python3
"""Main benchmark script: load model, profile prefill + decode, export CSV."""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.hook_profiler import HookProfiler
def load_model(model_id: str, quantization: str):
    """Load model in FP16 or INT4."""
    print(f"Loading {model_id} in {quantization}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda",
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    model.eval()
    print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    return model, tokenizer


def make_prompt_ids(tokenizer, target_len: int, device: str = "cuda"):
    """Create input_ids of approximately target_len tokens."""
    # Repeat a simple sentence to reach desired length
    text = "The quick brown fox jumps over the lazy dog. " * (target_len // 8 + 1)
    ids = tokenizer.encode(text, return_tensors="pt")[:, :target_len].to(device)
    return ids


def warmup(model, tokenizer, device="cuda"):
    """Run a short forward pass to warm up CUDA kernels."""
    print("Warming up...")
    ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=5)
    torch.cuda.synchronize()


def run_prefill(model, profiler: HookProfiler, input_ids):
    """Measure prefill: one forward pass with the full prompt."""
    profiler.current_phase = "prefill"
    profiler.current_decode_step = None
    profiler.recording = True

    with torch.no_grad():
        model(input_ids)

    torch.cuda.synchronize()


def run_decode(model, profiler: HookProfiler, tokenizer, prompt_ids,
               max_new_tokens: int):
    """Measure decode: generate tokens one by one.

    Uses manual decode loop so each step triggers hooks individually.
    """
    profiler.current_phase = "decode"
    profiler.recording = True

    with torch.no_grad():
        # First get the KV cache from prefill (unrecorded)
        profiler.recording = False
        outputs = model(prompt_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        profiler.recording = True

        # Start decoding from the last predicted token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens):
            profiler.current_decode_step = step
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    profiler.current_decode_step = None


def build_phase_summary(df, wall_time_s: float) -> dict:
    """Summarize one phase from the collected hook records."""
    if df.empty:
        return {
            "records": 0,
            "wall_time_s": wall_time_s,
            "hook_total_time_ms": 0.0,
            "peak_vram_mb": 0.0,
            "layer_types": {},
        }

    return {
        "records": int(len(df)),
        "wall_time_s": wall_time_s,
        "hook_total_time_ms": float(df["time_ms"].sum()),
        "peak_vram_mb": float(df["mem_peak_mb"].max()),
        "layer_types": {
            str(k): int(v) for k, v in df["layer_type"].value_counts().to_dict().items()
        },
    }


def write_metadata(path: str, payload: dict):
    """Write JSON metadata next to benchmark CSVs."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Quantization Tax Profiler")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--quantization", choices=["fp16", "int4"], default="fp16",
                        help="Quantization mode")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Prompt length in tokens for prefill")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Number of tokens to generate in decode")
    parser.add_argument("--output-dir", default="data",
                        help="Directory for CSV output")
    parser.add_argument("--run-id", default="single-run",
                        help="Identifier written to every CSV record")
    parser.add_argument("--metadata-path",
                        help="Optional JSON path for benchmark metadata")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_started_at = datetime.now(timezone.utc)

    # Load model
    model, tokenizer = load_model(args.model, args.quantization)

    # Setup profiler
    profiler = HookProfiler(model)
    profiler.run_id = args.run_id
    profiler.attach()

    # Warmup
    warmup(model, tokenizer)

    # --- Prefill ---
    print(f"\n=== Prefill ({args.prompt_len} tokens) ===")
    profiler.clear_records()
    prompt_ids = make_prompt_ids(tokenizer, args.prompt_len)
    t0 = time.perf_counter()
    run_prefill(model, profiler, prompt_ids)
    prefill_time = time.perf_counter() - t0
    print(f"Prefill wall time: {prefill_time:.2f}s")

    prefill_df = profiler.to_dataframe()
    prefill_path = os.path.join(args.output_dir, f"{args.quantization}_prefill.csv")
    prefill_df.to_csv(prefill_path, index=False)
    print(f"Exported {len(prefill_df)} records to {prefill_path}")

    # --- Decode ---
    print(f"\n=== Decode ({args.max_new_tokens} tokens) ===")
    profiler.clear_records()
    short_prompt = tokenizer.encode("Once upon a time", return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    run_decode(model, profiler, tokenizer, short_prompt, args.max_new_tokens)
    decode_time = time.perf_counter() - t0
    print(f"Decode wall time: {decode_time:.2f}s")

    decode_df = profiler.to_dataframe()
    decode_path = os.path.join(args.output_dir, f"{args.quantization}_decode.csv")
    decode_df.to_csv(decode_path, index=False)
    print(f"Exported {len(decode_df)} records to {decode_path}")

    # Summary
    profiler.detach()
    print(f"\n=== Summary ===")
    print(f"Model:        {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Run ID:       {args.run_id}")
    print(f"Prefill:      {prefill_time:.2f}s ({args.prompt_len} tokens)")
    print(f"Decode:       {decode_time:.2f}s ({args.max_new_tokens} tokens)")
    if decode_time > 0:
        print(f"Decode speed: {args.max_new_tokens / decode_time:.1f} tokens/s")
    print(f"Peak VRAM:    {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    if args.metadata_path:
        metadata = {
            "run_id": args.run_id,
            "model": args.model,
            "quantization": args.quantization,
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "output_dir": os.path.abspath(args.output_dir),
            "generated_at_utc": run_started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "host_platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_memory_mb": (
                torch.cuda.get_device_properties(0).total_memory / 1e6
                if torch.cuda.is_available() else None
            ),
            "prefill": build_phase_summary(prefill_df, prefill_time),
            "decode": build_phase_summary(decode_df, decode_time),
        }
        write_metadata(args.metadata_path, metadata)


if __name__ == "__main__":
    main()
