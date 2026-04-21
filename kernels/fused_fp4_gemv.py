"""Learning scaffold for a fused bitsandbytes fp4 dequant-GEMV kernel."""

from __future__ import annotations

import torch

from phase3_utils import Linear4bitArtifacts


def fused_fp4_gemv_triton(artifacts: Linear4bitArtifacts, x: torch.Tensor) -> torch.Tensor:
    """TODO(user): replace this stub with a Triton fused fp4 decode kernel.

    Expected contract:
    - artifacts.packed_weight is the raw uint8 packed tensor from bitsandbytes
    - artifacts.absmax and artifacts.code come from QuantState
    - x is one decode-time vector with shape [in_features]
    - return shape is [out_features]
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if x.shape[0] != artifacts.in_features:
        raise ValueError(
            f"input width mismatch: expected {artifacts.in_features}, got {x.shape[0]}"
        )
    raise NotImplementedError(
        "Implement fused_fp4_gemv_triton() in kernels/fused_fp4_gemv.py. "
        "Read one real Linear4bit layer via scripts/extract_linear4bit_reference.py first, "
        "then add unpack -> code lookup -> scale -> dot accumulation."
    )
