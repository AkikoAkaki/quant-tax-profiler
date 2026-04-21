#!/usr/bin/env python3
"""Reference-vs-kernel harness for a standalone fused fp4 prototype."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_fp4_gemv import fused_fp4_gemv_triton  # noqa: E402
from phase3_utils import (  # noqa: E402
    DEFAULT_LAYER_PATH,
    DEFAULT_MODEL_ID,
    extract_linear4bit_artifacts,
    load_int4_model,
    make_decode_input,
    resolve_module,
    run_reference_layer,
)


def main():
    parser = argparse.ArgumentParser(description="Verify a fused fp4 prototype on one real Linear4bit layer")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer", default=DEFAULT_LAYER_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, _ = load_int4_model(args.model)
    layer = resolve_module(model, args.layer)
    artifacts = extract_linear4bit_artifacts(layer, args.layer)
    decode_input = make_decode_input(artifacts.in_features, seed=args.seed)
    reference = run_reference_layer(layer, decode_input).reshape(-1)
    x = decode_input.reshape(-1)

    print(f"Layer: {artifacts.layer_path}")
    print(f"logical weight shape: {artifacts.logical_weight_shape}")
    print(f"packed weight shape:  {artifacts.packed_weight_shape}")
    print(f"quant_type: {artifacts.quant_type}, blocksize: {artifacts.blocksize}")

    try:
        candidate = fused_fp4_gemv_triton(artifacts, x)
    except NotImplementedError as exc:
        print(exc)
        print("Reference path is ready. Implement the fused prototype, then rerun this script.")
        return

    if candidate.ndim != 1:
        raise ValueError(f"candidate output must be 1D, got {tuple(candidate.shape)}")

    max_abs_err = (candidate - reference).abs().max().item()
    mean_abs_err = (candidate - reference).abs().mean().item()
    allclose = torch.allclose(candidate, reference, atol=1e-2, rtol=1e-2)

    print(f"candidate shape: {tuple(candidate.shape)}")
    print(f"reference shape: {tuple(reference.shape)}")
    print(f"max_abs_err: {max_abs_err:.6f}")
    print(f"mean_abs_err: {mean_abs_err:.6f}")
    print(f"allclose(atol=1e-2, rtol=1e-2): {allclose}")

    if not allclose:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
