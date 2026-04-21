"""Learning scaffold for a Triton FP16 GEMV kernel."""

from __future__ import annotations

import torch


def fp16_gemv_reference(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference GEMV: [out, in] @ [in] -> [out]."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if weight.shape[1] != x.shape[0]:
        raise ValueError(f"shape mismatch: weight={tuple(weight.shape)} x={tuple(x.shape)}")
    return torch.matmul(weight, x)


def fp16_gemv_triton(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """TODO(user): replace this stub with a Triton decode-style GEMV kernel."""
    raise NotImplementedError(
        "Implement fp16_gemv_triton() in kernels/fp16_gemv.py. "
        "Start with one output block, one input vector, and compare against fp16_gemv_reference()."
    )
