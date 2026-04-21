"""Arithmetic-intensity estimation and roofline chart generation."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualize import aggregate_layer_timings, is_linear_layer_type

MEMORY_BANDWIDTH_GB_S = 272.0
COMPUTE_TFLOPS_FP16 = 22.0
RIDGE_POINT = COMPUTE_TFLOPS_FP16 * 1e12 / (MEMORY_BANDWIDTH_GB_S * 1e9)


def parse_shape(shape_str: str) -> tuple | None:
    """Parse shape strings such as '(1, 128, 1536)' into tuples."""
    if not shape_str or shape_str == "None":
        return None
    try:
        return ast.literal_eval(str(shape_str))
    except (ValueError, SyntaxError):
        return None


def estimate_linear_flops_and_bytes(
    input_shape: tuple | None,
    output_shape: tuple | None,
    quant: str,
) -> tuple[float, float]:
    """Estimate FLOPs and bytes touched by one linear layer invocation."""
    if input_shape is None or output_shape is None:
        return 0.0, 0.0

    tokens = int(np.prod(input_shape[:-1])) if len(input_shape) >= 2 else 1
    in_features = input_shape[-1]
    out_features = output_shape[-1]
    flops = 2.0 * tokens * in_features * out_features

    if quant == "fp16":
        weight_bytes = in_features * out_features * 2.0
    else:
        weight_bytes = (
            in_features * out_features * 0.5
            + in_features * out_features * 2.0
            + in_features * out_features * 2.0
        )

    activation_bytes = tokens * in_features * 2.0
    return flops, weight_bytes + activation_bytes


def compute_roofline_data(dfs: dict[str, pd.DataFrame], phase: str = "decode") -> pd.DataFrame:
    """Compute arithmetic intensity and achieved throughput for linear layers."""
    rows = []
    for quant in ("fp16", "int4"):
        key = f"{quant}_{phase}"
        if key not in dfs:
            continue

        df = dfs[key]
        linear_df = df[df["layer_type"].map(is_linear_layer_type)]
        layer_agg = aggregate_layer_timings(linear_df)

        for _, row in layer_agg.iterrows():
            input_shape = parse_shape(row["input_shape"])
            output_shape = parse_shape(row["output_shape"])
            flops, bytes_accessed = estimate_linear_flops_and_bytes(
                input_shape,
                output_shape,
                quant,
            )
            if flops == 0.0 or bytes_accessed == 0.0 or row["time_ms_mean"] == 0.0:
                continue

            time_s = row["time_ms_mean"] / 1000.0
            rows.append({
                "layer_name": row["layer_name"],
                "quantization": quant,
                "arithmetic_intensity": flops / bytes_accessed,
                "gflops": (flops / time_s) / 1e9,
                "time_ms_mean": row["time_ms_mean"],
                "time_ms_std": row["time_ms_std"],
            })

    return pd.DataFrame(rows)


def plot_roofline(dfs: dict[str, pd.DataFrame], output_dir: str, phase: str = "decode"):
    """Figure 4: roofline chart for the selected phase."""
    output_dir = Path(output_dir)
    data = compute_roofline_data(dfs, phase)
    if data.empty:
        print("No data for Roofline chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        f"Roofline Model - {phase.capitalize()} Phase\n"
        f"RTX 4060 Laptop: {MEMORY_BANDWIDTH_GB_S:.0f} GB/s | "
        f"{COMPUTE_TFLOPS_FP16:.0f} TFLOPS FP16 peak",
        fontsize=12,
        fontweight="bold",
    )

    x_range = np.logspace(-2, 3, 500)
    mem_ceiling = x_range * MEMORY_BANDWIDTH_GB_S
    compute_ceiling = np.full_like(x_range, COMPUTE_TFLOPS_FP16 * 1000.0)
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    ax.loglog(x_range, roofline, "k-", linewidth=2, label="Hardware roofline")
    ax.axvline(RIDGE_POINT, color="gray", linestyle="--", linewidth=1,
               label=f"Ridge point ({RIDGE_POINT:.0f} FLOP/Byte)")

    colors = {"fp16": "#4C72B0", "int4": "#DD8452"}
    markers = {"fp16": "o", "int4": "^"}
    for quant, group in data.groupby("quantization"):
        ax.scatter(
            group["arithmetic_intensity"],
            group["gflops"],
            c=colors[quant],
            marker=markers[quant],
            alpha=0.65,
            s=28,
            label=quant.upper(),
            zorder=5,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=11)
    ax.set_ylabel("Achieved Throughput (GFLOPS)", fontsize=11)
    ax.set_xlim(1e-2, 1e3)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(
        0.05,
        0.09,
        "Decode remains deep in the memory-bound region",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
        style="italic",
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = output_dir / "fig4_roofline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
