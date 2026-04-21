#!/usr/bin/env python3
"""Inspect one real bitsandbytes Linear4bit layer from the model."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3_utils import (  # noqa: E402
    DEFAULT_LAYER_PATH,
    DEFAULT_MODEL_ID,
    extract_linear4bit_artifacts,
    load_int4_model,
    make_decode_input,
    resolve_module,
    run_reference_layer,
    summarize_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect one Linear4bit layer and sample decode input")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer", default=DEFAULT_LAYER_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, _ = load_int4_model(args.model)
    layer = resolve_module(model, args.layer)
    artifacts = extract_linear4bit_artifacts(layer, args.layer)
    decode_input = make_decode_input(artifacts.in_features, seed=args.seed)
    reference_output = run_reference_layer(layer, decode_input)

    summary = summarize_artifacts(artifacts)
    summary["sample_input_shape"] = tuple(int(dim) for dim in decode_input.shape)
    summary["sample_input_dtype"] = str(decode_input.dtype)
    summary["reference_output_shape"] = tuple(int(dim) for dim in reference_output.shape)
    summary["reference_output_dtype"] = str(reference_output.dtype)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("Next step:")
    print("1. Use this layer shape to implement kernels/fp16_gemv.py first.")
    print("2. Then implement kernels/fused_fp4_gemv.py against the same layer.")


if __name__ == "__main__":
    main()
