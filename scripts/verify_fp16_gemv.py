#!/usr/bin/env python3
"""Reference-vs-kernel harness for the first Triton FP16 GEMV exercise."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fp16_gemv import fp16_gemv_reference, fp16_gemv_triton  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Verify a Triton FP16 GEMV against torch.matmul")
    parser.add_argument("--in-features", type=int, default=1536)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weight = torch.randn((args.out_features, args.in_features), device="cuda", dtype=torch.float16)
    x = torch.randn((args.in_features,), device="cuda", dtype=torch.float16)

    reference = fp16_gemv_reference(weight, x)
    print(f"Reference output shape: {tuple(reference.shape)}")

    try:
        candidate = fp16_gemv_triton(weight, x)
    except NotImplementedError as exc:
        print(exc)
        print("Reference path is ready. Implement the Triton kernel, then rerun this script.")
        return

    max_abs_err = (candidate - reference).abs().max().item()
    mean_abs_err = (candidate - reference).abs().mean().item()
    allclose = torch.allclose(candidate, reference, atol=1e-2, rtol=1e-2)

    print(f"candidate shape: {tuple(candidate.shape)}")
    print(f"max_abs_err: {max_abs_err:.6f}")
    print(f"mean_abs_err: {mean_abs_err:.6f}")
    print(f"allclose(atol=1e-2, rtol=1e-2): {allclose}")

    if not allclose:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
