"""Chart generation: per-layer latency comparison, memory growth, Top-10 slowest layers."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all four CSVs. Returns dict keyed by 'fp16_prefill' etc."""
    data_dir = Path(data_dir)
    dfs = {}
    for quant in ("fp16", "int4"):
        for phase in ("prefill", "decode"):
            key = f"{quant}_{phase}"
            path = data_dir / f"{key}.csv"
            if path.exists():
                dfs[key] = pd.read_csv(path)
            else:
                print(f"WARNING: {path} not found, skipping.")
    return dfs


def plot_layerwise_latency(dfs: dict, output_dir: str):
    """Figure 1: Per-layer latency FP16 vs INT4, split by phase.

    For decode, averages across all token steps.
    Marks layers where INT4 > FP16 (quantization tax hotspots).
    """
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle("Per-Layer Latency: FP16 vs INT4", fontsize=14, fontweight="bold")

    for ax, phase in zip(axes, ("prefill", "decode")):
        fp16_key = f"fp16_{phase}"
        int4_key = f"int4_{phase}"

        if fp16_key not in dfs or int4_key not in dfs:
            ax.set_title(f"{phase.capitalize()} (data missing)")
            continue

        fp16 = dfs[fp16_key].groupby("layer_name")["time_ms"].mean().reset_index()
        int4 = dfs[int4_key].groupby("layer_name")["time_ms"].mean().reset_index()

        merged = fp16.merge(int4, on="layer_name", suffixes=("_fp16", "_int4"))
        merged["layer_idx"] = range(len(merged))
        merged["int4_slower"] = merged["time_ms_int4"] > merged["time_ms_fp16"]

        x = merged["layer_idx"].values

        ax.plot(x, merged["time_ms_fp16"], color="#4C72B0", linewidth=0.8,
                label="FP16", alpha=0.85)
        ax.plot(x, merged["time_ms_int4"], color="#DD8452", linewidth=0.8,
                label="INT4", alpha=0.85)

        # Highlight quantization tax hotspots (INT4 > FP16 by >10%)
        tax_mask = (merged["time_ms_int4"] > merged["time_ms_fp16"] * 1.1)
        if tax_mask.any():
            ax.fill_between(x, merged["time_ms_fp16"], merged["time_ms_int4"],
                            where=tax_mask, alpha=0.25, color="red",
                            label="Quant tax zone (INT4 >10% slower)")

        ax.set_title(f"{phase.capitalize()} phase", fontsize=11)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Avg latency (ms)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig1_layerwise_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_memory_growth(dfs: dict, output_dir: str):
    """Figure 2: KV cache VRAM growth across decode steps (FP16 vs INT4)."""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("KV Cache Memory Growth During Decode", fontsize=14, fontweight="bold")

    colors = {"fp16": "#4C72B0", "int4": "#DD8452"}

    for quant in ("fp16", "int4"):
        key = f"{quant}_decode"
        if key not in dfs:
            continue
        df = dfs[key]

        # Assign a step number to each token generation
        # Each step has ~197 layers; we identify steps by tracking layer order repeats
        df = df.copy()
        df["step"] = (df.groupby("layer_name").cumcount())

        step_mem = df.groupby("step")["mem_peak_mb"].max().reset_index()
        ax.plot(step_mem["step"], step_mem["mem_peak_mb"],
                color=colors[quant], label=quant.upper(), linewidth=1.5)

    ax.set_xlabel("Decode step (token #)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} MB"))
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig2_memory_growth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_top10_slowest(dfs: dict, output_dir: str):
    """Figure 3: Top-10 slowest layers, FP16 vs INT4 side-by-side (decode phase)."""
    output_dir = Path(output_dir)

    fp16_key, int4_key = "fp16_decode", "int4_decode"
    if fp16_key not in dfs or int4_key not in dfs:
        print("Skipping Top-10 chart: decode data missing.")
        return

    fp16_avg = dfs[fp16_key].groupby("layer_name")["time_ms"].mean()
    int4_avg = dfs[int4_key].groupby("layer_name")["time_ms"].mean()

    combined = pd.DataFrame({"FP16": fp16_avg, "INT4": int4_avg}).dropna()

    # Rank by max of the two (shows the worst-case bottleneck layers)
    combined["max_time"] = combined[["FP16", "INT4"]].max(axis=1)
    top10 = combined.nlargest(10, "max_time").sort_values("max_time")

    # Shorten layer names for display
    top10.index = [n.replace("model.layers.", "L").replace(".weight", "")
                   for n in top10.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Top-10 Slowest Layers: FP16 vs INT4 (Decode Phase, avg per step)",
                 fontsize=12, fontweight="bold")

    y = np.arange(len(top10))
    h = 0.35
    bars_fp16 = ax.barh(y - h / 2, top10["FP16"], h, label="FP16",
                        color="#4C72B0", alpha=0.85)
    bars_int4 = ax.barh(y + h / 2, top10["INT4"], h, label="INT4",
                        color="#DD8452", alpha=0.85)

    # Label bars where INT4 is slower
    for i, (_, row) in enumerate(top10.iterrows()):
        if row["INT4"] > row["FP16"]:
            pct = (row["INT4"] / row["FP16"] - 1) * 100
            ax.text(row["INT4"] + 0.001, i + h / 2,
                    f"+{pct:.0f}%", va="center", fontsize=7.5, color="red")

    ax.set_yticks(y)
    ax.set_yticklabels(top10.index, fontsize=8)
    ax.set_xlabel("Avg latency per decode step (ms)")
    ax.legend()
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig3_top10_slowest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def compute_summary_stats(dfs: dict) -> pd.DataFrame:
    """Compute summary statistics for the analysis report."""
    rows = []
    for quant in ("fp16", "int4"):
        for phase in ("prefill", "decode"):
            key = f"{quant}_{phase}"
            if key not in dfs:
                continue
            df = dfs[key]
            avg_per_layer = df.groupby("layer_name")["time_ms"].mean()
            rows.append({
                "quantization": quant,
                "phase": phase,
                "total_time_ms": df["time_ms"].sum(),
                "avg_layer_time_ms": avg_per_layer.mean(),
                "max_layer_time_ms": avg_per_layer.max(),
                "slowest_layer": avg_per_layer.idxmax(),
                "peak_vram_mb": df["mem_peak_mb"].max(),
            })
    return pd.DataFrame(rows)


def compute_quant_tax_layers(dfs: dict) -> pd.DataFrame:
    """Find layers where INT4 decode is slower than FP16."""
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        return pd.DataFrame()

    fp16 = dfs["fp16_decode"].groupby("layer_name")["time_ms"].mean()
    int4 = dfs["int4_decode"].groupby("layer_name")["time_ms"].mean()

    tax = pd.DataFrame({"fp16_ms": fp16, "int4_ms": int4}).dropna()
    tax["slowdown_pct"] = (tax["int4_ms"] / tax["fp16_ms"] - 1) * 100
    tax = tax[tax["slowdown_pct"] > 0].sort_values("slowdown_pct", ascending=False)
    return tax
