"""Phase 2 analysis helpers and chart generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

CSV_NAME_RE = re.compile(r"^(fp16|int4)_(prefill|decode)\.csv$")
LAYER_ORDER = {
    "input_layernorm": 0,
    "self_attn.q_proj": 1,
    "self_attn.k_proj": 2,
    "self_attn.v_proj": 3,
    "self_attn.o_proj": 4,
    "post_attention_layernorm": 5,
    "mlp.gate_proj": 6,
    "mlp.up_proj": 7,
    "mlp.down_proj": 8,
    "norm": 9,
    "lm_head": 10,
}


def is_linear_layer_type(layer_type: str) -> bool:
    """Return True for both FP16 and bitsandbytes linear modules."""
    return str(layer_type).startswith("Linear")


def _sort_key(layer_name: str) -> tuple:
    match = re.search(r"model\.layers\.(\d+)\.(.+)", layer_name)
    if match:
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        return (0, layer_idx, LAYER_ORDER.get(suffix, 999), suffix)
    return (1, 0, LAYER_ORDER.get(layer_name, 999), layer_name)


def shorten_layer_name(layer_name: str) -> str:
    """Compact layer labels for plots and tables."""
    return layer_name.replace("model.layers.", "L")


def _infer_decode_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill decode_step for older CSVs that predate the explicit column."""
    df = df.copy()
    df["decode_step"] = df.groupby("layer_name").cumcount()
    return df


def _normalize_df(df: pd.DataFrame, phase: str, fallback_run_id: str) -> pd.DataFrame:
    df = df.copy()

    if "run_id" not in df.columns:
        df["run_id"] = fallback_run_id
    df["run_id"] = df["run_id"].fillna(fallback_run_id).astype(str)

    if "decode_step" not in df.columns:
        if phase == "decode":
            df = _infer_decode_steps(df)
        else:
            df["decode_step"] = pd.NA

    df["decode_step"] = pd.to_numeric(df["decode_step"], errors="coerce").astype("Int64")
    return df


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Recursively load all matching benchmark CSVs under a directory."""
    root = Path(data_dir)
    dfs: dict[str, list[pd.DataFrame]] = {}

    for path in sorted(root.rglob("*.csv")):
        match = CSV_NAME_RE.match(path.name)
        if not match:
            continue

        quant, phase = match.groups()
        key = f"{quant}_{phase}"
        fallback_run_id = path.parent.name if path.parent != root else path.stem
        df = pd.read_csv(path)
        df = _normalize_df(df, phase=phase, fallback_run_id=fallback_run_id)
        df["source_csv"] = str(path)
        dfs.setdefault(key, []).append(df)

    return {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in dfs.items()
        if parts
    }


def load_benchmark_metadata(data_dir: str) -> list[dict]:
    """Load per-run metadata JSON files when available."""
    root = Path(data_dir)
    results = []
    for path in sorted(root.rglob("benchmark_metadata.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def aggregate_layer_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-layer timings across decode steps and repeated runs."""
    if df.empty:
        return pd.DataFrame(columns=[
            "layer_name",
            "layer_type",
            "time_ms_mean",
            "time_ms_std",
            "input_shape",
            "output_shape",
        ])

    per_run = (
        df.groupby(["run_id", "layer_name"], as_index=False)
        .agg(
            time_ms=("time_ms", "mean"),
            layer_type=("layer_type", "first"),
            input_shape=("input_shape", "first"),
            output_shape=("output_shape", "first"),
        )
    )

    agg = (
        per_run.groupby("layer_name", as_index=False)
        .agg(
            layer_type=("layer_type", "first"),
            time_ms_mean=("time_ms", "mean"),
            time_ms_std=("time_ms", "std"),
            input_shape=("input_shape", "first"),
            output_shape=("output_shape", "first"),
        )
    )
    agg["time_ms_std"] = agg["time_ms_std"].fillna(0.0)
    agg["sort_key"] = agg["layer_name"].map(_sort_key)
    agg = agg.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return agg


def compute_summary_stats(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-phase summary stats across repeated runs."""
    rows = []
    for quant in ("fp16", "int4"):
        for phase in ("prefill", "decode"):
            key = f"{quant}_{phase}"
            if key not in dfs:
                continue

            df = dfs[key]
            total_per_run = df.groupby("run_id")["time_ms"].sum()
            peak_per_run = df.groupby("run_id")["mem_peak_mb"].max()
            per_run_layer = (
                df.groupby(["run_id", "layer_name"], as_index=False)["time_ms"]
                .mean()
            )
            avg_layer_per_run = per_run_layer.groupby("run_id")["time_ms"].mean()
            layer_agg = aggregate_layer_timings(df)
            slowest_layer = layer_agg.iloc[layer_agg["time_ms_mean"].idxmax()]

            rows.append({
                "quantization": quant,
                "phase": phase,
                "num_runs": int(df["run_id"].nunique()),
                "total_time_ms_mean": float(total_per_run.mean()),
                "total_time_ms_std": float(total_per_run.std(ddof=1) if len(total_per_run) > 1 else 0.0),
                "peak_vram_mb_mean": float(peak_per_run.mean()),
                "peak_vram_mb_std": float(peak_per_run.std(ddof=1) if len(peak_per_run) > 1 else 0.0),
                "avg_layer_time_ms_mean": float(avg_layer_per_run.mean()),
                "avg_layer_time_ms_std": float(avg_layer_per_run.std(ddof=1) if len(avg_layer_per_run) > 1 else 0.0),
                "slowest_layer": str(slowest_layer["layer_name"]),
                "slowest_layer_time_ms_mean": float(slowest_layer["time_ms_mean"]),
                "slowest_layer_time_ms_std": float(slowest_layer["time_ms_std"]),
            })

    return pd.DataFrame(rows)


def compute_quant_tax_layers(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find decode layers where INT4 is slower than FP16."""
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        return pd.DataFrame()

    fp16 = aggregate_layer_timings(dfs["fp16_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    int4 = aggregate_layer_timings(dfs["int4_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    merged = fp16.merge(int4, on="layer_name", suffixes=("_fp16", "_int4"))
    merged["slowdown_pct"] = (merged["time_ms_mean_int4"] / merged["time_ms_mean_fp16"] - 1.0) * 100.0
    merged = merged[merged["slowdown_pct"] > 0].sort_values("slowdown_pct", ascending=False)
    return merged.reset_index(drop=True)


def plot_layerwise_latency(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 1: per-layer latency, aggregated across repeated runs."""
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle("Per-Layer Latency (mean ± std): FP16 vs INT4", fontsize=14, fontweight="bold")

    for ax, phase in zip(axes, ("prefill", "decode")):
        fp16_key = f"fp16_{phase}"
        int4_key = f"int4_{phase}"

        if fp16_key not in dfs or int4_key not in dfs:
            ax.set_title(f"{phase.capitalize()} phase (data missing)")
            continue

        fp16 = aggregate_layer_timings(dfs[fp16_key])
        int4 = aggregate_layer_timings(dfs[int4_key])
        merged = fp16.merge(
            int4,
            on="layer_name",
            suffixes=("_fp16", "_int4"),
        )
        merged["layer_idx"] = range(len(merged))
        x = merged["layer_idx"].to_numpy()

        fp16_mean = merged["time_ms_mean_fp16"].to_numpy()
        fp16_std = merged["time_ms_std_fp16"].to_numpy()
        int4_mean = merged["time_ms_mean_int4"].to_numpy()
        int4_std = merged["time_ms_std_int4"].to_numpy()

        ax.plot(x, fp16_mean, color="#4C72B0", linewidth=1.1, label="FP16")
        ax.fill_between(x, fp16_mean - fp16_std, fp16_mean + fp16_std,
                        color="#4C72B0", alpha=0.12)

        ax.plot(x, int4_mean, color="#DD8452", linewidth=1.1, label="INT4")
        ax.fill_between(x, int4_mean - int4_std, int4_mean + int4_std,
                        color="#DD8452", alpha=0.12)

        tax_mask = int4_mean > fp16_mean * 1.1
        if tax_mask.any():
            ax.fill_between(
                x,
                fp16_mean,
                int4_mean,
                where=tax_mask,
                color="red",
                alpha=0.18,
                label="Quantization tax zone (>10% slower)",
            )

        ax.set_title(f"{phase.capitalize()} phase", fontsize=11)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Latency per layer (ms)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig1_layerwise_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_memory_growth(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 2: KV cache VRAM growth during decode, aggregated across runs."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("KV Cache Memory Growth During Decode (mean ± std)",
                 fontsize=14, fontweight="bold")

    colors = {"fp16": "#4C72B0", "int4": "#DD8452"}

    for quant in ("fp16", "int4"):
        key = f"{quant}_decode"
        if key not in dfs:
            continue

        df = dfs[key].dropna(subset=["decode_step"]).copy()
        if df.empty:
            continue

        per_run = (
            df.groupby(["run_id", "decode_step"], as_index=False)["mem_peak_mb"]
            .max()
        )
        agg = per_run.groupby("decode_step")["mem_peak_mb"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)

        steps = agg["decode_step"].to_numpy()
        mean_mb = agg["mean"].to_numpy()
        std_mb = agg["std"].to_numpy()

        ax.plot(steps, mean_mb, color=colors[quant], linewidth=1.8, label=quant.upper())
        ax.fill_between(steps, mean_mb - std_mb, mean_mb + std_mb,
                        color=colors[quant], alpha=0.15)

    ax.set_xlabel("Decode step (token index)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} MB"))
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig2_memory_growth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_top10_slowest(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 3: top-10 slowest decode layers with error bars."""
    output_dir = Path(output_dir)
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        print("Skipping Top-10 chart: decode data missing.")
        return

    fp16 = aggregate_layer_timings(dfs["fp16_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    int4 = aggregate_layer_timings(dfs["int4_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    combined = fp16.merge(int4, on="layer_name", suffixes=("_fp16", "_int4"))
    combined["max_time_mean"] = combined[["time_ms_mean_fp16", "time_ms_mean_int4"]].max(axis=1)
    top10 = combined.nlargest(10, "max_time_mean").sort_values("max_time_mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Top-10 Slowest Decode Layers (mean ± std)", fontsize=12, fontweight="bold")

    y = np.arange(len(top10))
    height = 0.35
    ax.barh(
        y - height / 2,
        top10["time_ms_mean_fp16"],
        height,
        xerr=top10["time_ms_std_fp16"],
        label="FP16",
        color="#4C72B0",
        alpha=0.85,
    )
    ax.barh(
        y + height / 2,
        top10["time_ms_mean_int4"],
        height,
        xerr=top10["time_ms_std_int4"],
        label="INT4",
        color="#DD8452",
        alpha=0.85,
    )

    for idx, (_, row) in enumerate(top10.iterrows()):
        if row["time_ms_mean_int4"] > row["time_ms_mean_fp16"]:
            pct = (row["time_ms_mean_int4"] / row["time_ms_mean_fp16"] - 1.0) * 100.0
            ax.text(row["time_ms_mean_int4"] + 0.01, idx + height / 2,
                    f"+{pct:.0f}%", va="center", fontsize=8, color="red")

    ax.set_yticks(y)
    ax.set_yticklabels([shorten_layer_name(name) for name in top10["layer_name"]], fontsize=8)
    ax.set_xlabel("Latency per decode step (ms)")
    ax.legend()
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig3_top10_slowest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
