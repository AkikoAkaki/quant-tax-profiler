#!/usr/bin/env python3
"""Main analysis entry point: generate all charts and analysis_report.md."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualize import (
    load_data,
    plot_layerwise_latency,
    plot_memory_growth,
    plot_top10_slowest,
    compute_summary_stats,
    compute_quant_tax_layers,
)
from analysis.roofline import plot_roofline


def write_report(dfs, output_dir: Path, summary: "pd.DataFrame",
                 tax_layers: "pd.DataFrame"):
    """Write analysis_report.md."""
    import pandas as pd

    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    path = report_dir / "analysis_report.md"

    def row(quant, phase):
        mask = (summary["quantization"] == quant) & (summary["phase"] == phase)
        r = summary[mask]
        return r.iloc[0] if len(r) > 0 else None

    fp16_p = row("fp16", "prefill")
    int4_p = row("int4", "prefill")
    fp16_d = row("fp16", "decode")
    int4_d = row("int4", "decode")

    prefill_slowdown = (int4_p["total_time_ms"] / fp16_p["total_time_ms"] - 1) * 100 if fp16_p is not None and int4_p is not None else 0
    decode_slowdown  = (int4_d["total_time_ms"] / fp16_d["total_time_ms"] - 1) * 100 if fp16_d is not None and int4_d is not None else 0
    vram_savings     = (1 - int4_d["peak_vram_mb"] / fp16_d["peak_vram_mb"]) * 100 if fp16_d is not None and int4_d is not None else 0

    top_tax = tax_layers.head(5) if not tax_layers.empty else None

    lines = [
        "# LLM Quantization Tax Analysis Report",
        "",
        "**Model**: Qwen/Qwen2.5-1.5B-Instruct  ",
        "**Hardware**: RTX 4060 Laptop 8GB (272 GB/s, 22 TFLOPS FP16)  ",
        "**Quantization**: FP16 baseline vs INT4 (bitsandbytes load_in_4bit)",
        "",
        "---",
        "",
        "## 1. Key Findings",
        "",
        "| Metric | FP16 | INT4 | Change |",
        "|--------|------|------|--------|",
    ]

    if fp16_p is not None and int4_p is not None:
        lines.append(
            f"| Prefill total time (ms) | {fp16_p['total_time_ms']:.1f} | "
            f"{int4_p['total_time_ms']:.1f} | **+{prefill_slowdown:.1f}%** |"
        )
    if fp16_d is not None and int4_d is not None:
        lines.append(
            f"| Decode total time (ms) | {fp16_d['total_time_ms']:.1f} | "
            f"{int4_d['total_time_ms']:.1f} | **+{decode_slowdown:.1f}%** |"
        )
        lines.append(
            f"| Peak VRAM (MB) | {fp16_d['peak_vram_mb']:.0f} | "
            f"{int4_d['peak_vram_mb']:.0f} | **-{vram_savings:.1f}%** ✓ |"
        )

    lines += [
        "",
        f"> **Quantization tax confirmed**: despite a ~4× weight compression and "
        f"{vram_savings:.0f}% VRAM savings, INT4 decode is **{decode_slowdown:.1f}% slower** than FP16.",
        "",
        "---",
        "",
        "## 2. Worst Offenders (Top 5 layers by decode slowdown)",
        "",
    ]

    if top_tax is not None:
        lines.append("| Layer | FP16 (ms) | INT4 (ms) | Slowdown |")
        lines.append("|-------|-----------|-----------|----------|")
        for name, r in top_tax.iterrows():
            lines.append(
                f"| `{name}` | {r['fp16_ms']:.4f} | {r['int4_ms']:.4f} | "
                f"+{r['slowdown_pct']:.1f}% |"
            )
    else:
        lines.append("*No slowdown data available.*")

    lines += [
        "",
        "---",
        "",
        "## 3. Theoretical Explanation: Why Does INT4 Slow Down Decode?",
        "",
        "### 3.1 Decode is Memory-Bound",
        "",
        "During decode, the model generates **one token at a time** (batch=1, seq=1).",
        "Each Linear layer degenerates to a **matrix-vector multiply (GEMV)**:",
        "",
        "```",
        "Arithmetic Intensity = FLOPs / Bytes",
        "  FP16 Linear: 2 × in × out / (in × out × 2) = 1 FLOP/Byte",
        "  Ridge point:  22 TFLOPS / 272 GB/s ≈ 80 FLOP/Byte",
        "```",
        "",
        "At 1 FLOP/Byte vs a ridge point of 80 FLOP/Byte, decode is deep in the",
        "**memory-bound** regime. Performance scales directly with bytes transferred.",
        "",
        "### 3.2 The Dequantization Round-Trip",
        "",
        "Out-of-place INT4 inference (bitsandbytes default) follows this path:",
        "",
        "```",
        "VRAM[INT4 weights]  ──read──►  Dequant to FP16  ──write──►  VRAM[FP16]",
        "                                                  ──read───►  GEMV  ──►  output",
        "```",
        "",
        "Bytes accessed per weight element:",
        "```",
        "FP16:  2 bytes read                                    = 2.0 bytes",
        "INT4:  0.5 read (INT4) + 2.0 write (FP16) + 2.0 read  = 4.5 bytes",
        "```",
        "",
        "INT4 accesses **2.25× more bytes** than FP16 despite weights being 4× smaller.",
        "This negates and reverses the bandwidth savings in the decode phase.",
        "",
        "### 3.3 Fused Kernel Fix (Phase 3)",
        "",
        "A Triton fused kernel eliminates the intermediate VRAM round-trip:",
        "",
        "```",
        "VRAM[INT4]  ──read──►  GPU SRAM (shared memory)  ──decompress + GEMV──►  output",
        "```",
        "",
        "Bytes accessed:",
        "```",
        "Fused INT4:  0.5 bytes read only  (4× less than FP16, as intended)",
        "```",
        "",
        "---",
        "",
        "## 4. Charts",
        "",
        "| Figure | Description |",
        "|--------|-------------|",
        "| ![fig1](../outputs/fig1_layerwise_latency.png) | Per-layer latency FP16 vs INT4 |",
        "| ![fig2](../outputs/fig2_memory_growth.png) | KV cache VRAM growth during decode |",
        "| ![fig3](../outputs/fig3_top10_slowest.png) | Top-10 slowest layers |",
        "| ![fig4](../outputs/fig4_roofline.png) | Roofline model analysis |",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis charts and report")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    print("Loading CSV data...")
    dfs = load_data(args.data_dir)
    if not dfs:
        print("ERROR: No CSV files found in", args.data_dir)
        sys.exit(1)
    print(f"Loaded: {list(dfs.keys())}")

    print("\nGenerating charts...")
    plot_layerwise_latency(dfs, args.output_dir)
    plot_memory_growth(dfs, args.output_dir)
    plot_top10_slowest(dfs, args.output_dir)
    plot_roofline(dfs, args.output_dir)

    print("\nComputing statistics...")
    summary = compute_summary_stats(dfs)
    tax_layers = compute_quant_tax_layers(dfs)

    print("\nTop 10 layers with quantization tax (decode):")
    if not tax_layers.empty:
        print(tax_layers.head(10).to_string())
    else:
        print("  None found.")

    print("\nWriting analysis report...")
    write_report(dfs, Path(args.output_dir), summary, tax_layers)

    print("\n=== Done ===")
    print(f"Charts in:  {args.output_dir}/")
    print(f"Report in:  report/analysis_report.md")


if __name__ == "__main__":
    main()
