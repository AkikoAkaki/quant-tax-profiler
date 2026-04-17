"""Arithmetic intensity estimation and Roofline chart generation."""

import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# RTX 4060 Laptop hardware limits
MEMORY_BANDWIDTH_GB_S = 272.0   # GB/s
COMPUTE_TFLOPS_FP16 = 22.0      # TFLOPS (FP16 Tensor Core)
RIDGE_POINT = COMPUTE_TFLOPS_FP16 * 1e12 / (MEMORY_BANDWIDTH_GB_S * 1e9)  # ~80 FLOP/Byte


def parse_shape(shape_str: str) -> tuple | None:
    """Parse shape string like '(1, 128, 1536)' into a tuple."""
    if not shape_str or shape_str == "None":
        return None
    try:
        return ast.literal_eval(str(shape_str))
    except Exception:
        return None


def estimate_linear_flops_and_bytes(input_shape: tuple, output_shape: tuple,
                                     quant: str) -> tuple[float, float]:
    """Estimate FLOPs and memory bytes accessed for a Linear layer.

    For a Linear(in_features → out_features) with batch*seq tokens:
      FLOPs  = 2 * tokens * in_features * out_features   (multiply-add = 2 ops)
      Bytes  = weight_bytes + activation_bytes

    FP16 weights:  in * out * 2 bytes
    INT4 weights:  in * out * 0.5 bytes (packed 4-bit)
                   + in * out * 2 bytes (dequant to FP16, write to VRAM)
                   + in * out * 2 bytes (re-read FP16 for matmul)
                   = 4.5x the weight size  ← the quantization tax!

    Input activations are always FP16: tokens * in * 2 bytes
    """
    if input_shape is None or output_shape is None:
        return 0.0, 0.0

    # tokens = batch * seq_len
    tokens = int(np.prod(input_shape[:-1])) if len(input_shape) >= 2 else 1
    in_features = input_shape[-1]
    out_features = output_shape[-1]

    flops = 2.0 * tokens * in_features * out_features

    if quant == "fp16":
        weight_bytes = in_features * out_features * 2      # FP16 weight
    else:
        # INT4: read packed + write dequant FP16 + re-read FP16 for matmul
        weight_bytes = (in_features * out_features * 0.5   # packed INT4
                        + in_features * out_features * 2   # write dequant
                        + in_features * out_features * 2)  # re-read for matmul

    activation_bytes = tokens * in_features * 2           # FP16 activations

    total_bytes = weight_bytes + activation_bytes
    return flops, total_bytes


def compute_roofline_data(dfs: dict, phase: str = "decode") -> pd.DataFrame:
    """Compute arithmetic intensity and observed GFLOPS per Linear layer."""
    rows = []
    for quant in ("fp16", "int4"):
        key = f"{quant}_{phase}"
        if key not in dfs:
            continue
        df = dfs[key]
        linear_df = df[df["layer_type"] == "Linear"]

        avg = linear_df.groupby("layer_name").agg(
            time_ms=("time_ms", "mean"),
            input_shape=("input_shape", "first"),
            output_shape=("output_shape", "first"),
        ).reset_index()

        for _, row in avg.iterrows():
            in_shape = parse_shape(row["input_shape"])
            out_shape = parse_shape(row["output_shape"])
            flops, bytes_accessed = estimate_linear_flops_and_bytes(
                in_shape, out_shape, quant
            )
            if bytes_accessed == 0 or row["time_ms"] == 0:
                continue

            time_s = row["time_ms"] / 1000.0
            arithmetic_intensity = flops / bytes_accessed   # FLOP/Byte
            gflops = (flops / time_s) / 1e9               # GFLOPS achieved

            rows.append({
                "layer_name": row["layer_name"],
                "quantization": quant,
                "arithmetic_intensity": arithmetic_intensity,
                "gflops": gflops,
                "time_ms": row["time_ms"],
            })

    return pd.DataFrame(rows)


def plot_roofline(dfs: dict, output_dir: str, phase: str = "decode"):
    """Figure 4: Roofline model chart."""
    output_dir = Path(output_dir)
    data = compute_roofline_data(dfs, phase)

    if data.empty:
        print("No data for Roofline chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Roofline Model — {phase.capitalize()} Phase\n"
                 f"RTX 4060 Laptop: {MEMORY_BANDWIDTH_GB_S} GB/s | "
                 f"{COMPUTE_TFLOPS_FP16} TFLOPS FP16",
                 fontsize=12, fontweight="bold")

    # Draw hardware ceiling lines
    x_range = np.logspace(-2, 3, 500)  # 0.01 to 1000 FLOP/Byte

    # Memory bandwidth ceiling: perf = intensity × bandwidth
    mem_ceiling = x_range * MEMORY_BANDWIDTH_GB_S * 1e9 / 1e9  # in GFLOPS
    # Compute ceiling: flat line at peak TFLOPS → in GFLOPS
    compute_ceiling = np.full_like(x_range, COMPUTE_TFLOPS_FP16 * 1000)  # GFLOPS
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    ax.loglog(x_range, roofline, "k-", linewidth=2, label="Roofline (hardware limit)")
    ax.axvline(x=RIDGE_POINT, color="gray", linestyle="--", linewidth=1,
               label=f"Ridge point ({RIDGE_POINT:.0f} FLOP/Byte)")

    # Plot layer data points
    colors = {"fp16": "#4C72B0", "int4": "#DD8452"}
    markers = {"fp16": "o", "int4": "^"}

    for quant, grp in data.groupby("quantization"):
        ax.scatter(grp["arithmetic_intensity"], grp["gflops"],
                   c=colors[quant], marker=markers[quant],
                   alpha=0.6, s=25, label=quant.upper(), zorder=5)

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=11)
    ax.set_ylabel("Achieved Performance (GFLOPS)", fontsize=11)
    ax.set_xlim(1e-2, 1e3)

    # Annotate memory-bound region
    ax.text(0.05, 0.1, "Memory-bound\n(below roofline)",
            transform=ax.transAxes, fontsize=9, color="gray", style="italic")

    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "fig4_roofline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
