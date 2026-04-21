#!/usr/bin/env python3
"""Generate Phase 2 charts and a bilingual Markdown report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.roofline import plot_roofline
from analysis.visualize import (
    compute_quant_tax_layers,
    compute_summary_stats,
    load_benchmark_metadata,
    load_data,
    plot_layerwise_latency,
    plot_memory_growth,
    plot_top10_slowest,
)


def _summary_row(summary, quant: str, phase: str):
    match = summary[(summary["quantization"] == quant) & (summary["phase"] == phase)]
    return match.iloc[0] if not match.empty else None


def _fmt_pm(mean: float, std: float, precision: int = 1, suffix: str = "") -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}{suffix}"


def _relative_output_path(report_path: Path, output_path: Path) -> str:
    return os.path.relpath(output_path, start=report_path.parent).replace("\\", "/")


def write_report(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    summary,
    tax_layers,
    metadata: list[dict],
):
    """Write the bilingual Markdown report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fp16_prefill = _summary_row(summary, "fp16", "prefill")
    int4_prefill = _summary_row(summary, "int4", "prefill")
    fp16_decode = _summary_row(summary, "fp16", "decode")
    int4_decode = _summary_row(summary, "int4", "decode")

    decode_slowdown = 0.0
    prefill_slowdown = 0.0
    vram_savings = 0.0
    if fp16_prefill is not None and int4_prefill is not None:
        prefill_slowdown = (int4_prefill["total_time_ms_mean"] / fp16_prefill["total_time_ms_mean"] - 1.0) * 100.0
    if fp16_decode is not None and int4_decode is not None:
        decode_slowdown = (int4_decode["total_time_ms_mean"] / fp16_decode["total_time_ms_mean"] - 1.0) * 100.0
        vram_savings = (1.0 - int4_decode["peak_vram_mb_mean"] / fp16_decode["peak_vram_mb_mean"]) * 100.0

    top_tax = tax_layers.head(10).copy() if not tax_layers.empty else None
    gpu_name = None
    if metadata:
        gpu_name = next((entry.get("gpu_name") for entry in metadata if entry.get("gpu_name")), None)
    if gpu_name is None:
        gpu_name = "RTX 4060 Laptop GPU (expected target)"

    num_fp16_runs = int(fp16_decode["num_runs"]) if fp16_decode is not None else 0
    num_int4_runs = int(int4_decode["num_runs"]) if int4_decode is not None else 0

    fig1_rel = _relative_output_path(report_path, output_dir / "fig1_layerwise_latency.png")
    fig2_rel = _relative_output_path(report_path, output_dir / "fig2_memory_growth.png")
    fig3_rel = _relative_output_path(report_path, output_dir / "fig3_top10_slowest.png")
    fig4_rel = _relative_output_path(report_path, output_dir / "fig4_roofline.png")

    lines = [
        "# Phase 2 Quantization Tax Report / Phase 2 量化税分析报告",
        "",
        "## Executive Summary / 执行摘要",
        "",
        f"- INT4 decode remains slower than FP16 by **{decode_slowdown:.1f}%** on average, while cutting peak VRAM by **{vram_savings:.1f}%**.",
        f"- INT4 在平均 decode 延迟上比 FP16 **慢 {decode_slowdown:.1f}%**，但峰值显存平均减少 **{vram_savings:.1f}%**。",
        "- The worst slowdown concentrates in attention projection layers, especially `k_proj` and `v_proj`, which are the first Phase 3 optimization targets.",
        "- 最明显的 slowdown 集中在 attention 投影层，尤其是 `k_proj` 和 `v_proj`，它们应作为 Phase 3 的首批优化目标。",
        "- Roofline analysis still places decode firmly in the memory-bound region; the current INT4 path is penalized by the out-of-place dequantization round-trip.",
        "- Roofline 分析仍然表明 decode 深处于 memory-bound 区域，当前 INT4 路径的主要问题是 out-of-place dequantization 带来的额外 VRAM round-trip。",
        "",
        "## Experiment Setup / 实验设置",
        "",
        f"- Model / 模型: `Qwen/Qwen2.5-1.5B-Instruct`",
        f"- Hardware / 硬件: `{gpu_name}`",
        f"- Data root / 数据目录: `{data_dir}`",
        f"- Repeats / 重复次数: FP16 `{num_fp16_runs}` runs, INT4 `{num_int4_runs}` runs",
        "- Prompt length / Prompt 长度: `512`",
        "- Decode length / Decode 长度: `128` tokens",
        "- Aggregation / 聚合方式: per-run metrics aggregated as mean ± std",
        "",
        "## Key Metrics / 关键指标",
        "",
        "| Metric | FP16 | INT4 | Delta |",
        "|--------|------|------|-------|",
    ]

    if fp16_prefill is not None and int4_prefill is not None:
        lines.append(
            "| Prefill total time (ms) / Prefill 总时延 | "
            f"{_fmt_pm(fp16_prefill['total_time_ms_mean'], fp16_prefill['total_time_ms_std'])} | "
            f"{_fmt_pm(int4_prefill['total_time_ms_mean'], int4_prefill['total_time_ms_std'])} | "
            f"+{prefill_slowdown:.1f}% |"
        )
    if fp16_decode is not None and int4_decode is not None:
        lines.append(
            "| Decode total time (ms) / Decode 总时延 | "
            f"{_fmt_pm(fp16_decode['total_time_ms_mean'], fp16_decode['total_time_ms_std'])} | "
            f"{_fmt_pm(int4_decode['total_time_ms_mean'], int4_decode['total_time_ms_std'])} | "
            f"+{decode_slowdown:.1f}% |"
        )
        lines.append(
            "| Peak VRAM (MB) / 峰值显存 | "
            f"{_fmt_pm(fp16_decode['peak_vram_mb_mean'], fp16_decode['peak_vram_mb_std'], suffix=' MB')} | "
            f"{_fmt_pm(int4_decode['peak_vram_mb_mean'], int4_decode['peak_vram_mb_std'], suffix=' MB')} | "
            f"-{vram_savings:.1f}% |"
        )

    lines += [
        "",
        "## Figures / 图表",
        "",
        "### Figure 1 / 图 1: Per-layer latency",
        "",
        f"![Figure 1]({fig1_rel})",
        "",
        "_FP16 and INT4 layer latencies are aggregated across repeated runs. The red band marks layers where INT4 is more than 10% slower._  ",
        "_图中展示了跨重复实验聚合后的逐层延迟。红色区域表示 INT4 比 FP16 慢超过 10% 的层。_",
        "",
        "### Figure 2 / 图 2: KV cache memory growth",
        "",
        f"![Figure 2]({fig2_rel})",
        "",
        "_Memory growth remains nearly flat across decode steps; the main story is the lower VRAM baseline of INT4 rather than a growing cache bottleneck._  ",
        "_Decode 过程中显存曲线整体较平，说明这次实验的主要现象不是 KV cache 增长，而是 INT4 的更低显存基线。_",
        "",
        "### Figure 3 / 图 3: Top-10 slowest decode layers",
        "",
        f"![Figure 3]({fig3_rel})",
        "",
        "_The slowest decode layers are dominated by attention projections, which is consistent with the small-matrix / dequant-overhead hypothesis._  ",
        "_最慢的 decode 层主要被 attention 投影层占据，这与“小矩阵 + dequant 固定开销”这一假设一致。_",
        "",
        "### Figure 4 / 图 4: Roofline",
        "",
        f"![Figure 4]({fig4_rel})",
        "",
        "_Both FP16 and INT4 decode points sit far to the left of the ridge point, confirming that decode remains bandwidth-limited rather than compute-limited._  ",
        "_FP16 和 INT4 的 decode 点都远离 ridge point 左侧，说明当前 decode 受限于带宽而不是算力。_",
        "",
        "## Worst Offenders / 关键慢层",
        "",
    ]

    if top_tax is None or top_tax.empty:
        lines.append("No positive INT4 slowdown was detected. / 未检测到 INT4 正 slowdown。")
    else:
        lines.append("| Layer | FP16 (ms) | INT4 (ms) | Slowdown |")
        lines.append("|-------|-----------|-----------|----------|")
        for _, row in top_tax.iterrows():
            lines.append(
                f"| `{row['layer_name']}` | "
                f"{_fmt_pm(row['time_ms_mean_fp16'], row['time_ms_std_fp16'], precision=3)} | "
                f"{_fmt_pm(row['time_ms_mean_int4'], row['time_ms_std_int4'], precision=3)} | "
                f"+{row['slowdown_pct']:.1f}% |"
            )

    lines += [
        "",
        "## Interpretation / 结果解释",
        "",
        "1. Decode is still memory-bound. For seq=1 decode, each linear layer behaves like GEMV, so arithmetic intensity stays far below the hardware ridge point.",
        "2. Decode 仍是典型的 memory-bound 场景。对 seq=1 的 decode 来说，各个线性层都接近 GEMV，算术强度远低于硬件 ridge point。",
        "3. The INT4 path pays an extra dequantization round-trip: read packed weights, materialize FP16 weights, then read them again for matmul.",
        "4. 当前 INT4 路径多付出了 dequantization round-trip：先读 packed 权重，再写回 FP16 权重，最后再次读取做 matmul。",
        "5. This means INT4 reduces storage footprint but increases bandwidth pressure during decode, which is exactly the quantization tax that Phase 3 must remove.",
        "6. 这意味着 INT4 虽然降低了存储占用，却在 decode 时增加了带宽压力，这正是 Phase 3 需要消除的 quantization tax。",
        "",
        "## Phase 3 Handoff / Phase 3 交接",
        "",
        "- Prioritize fused dequant-GEMV for attention `k_proj` and `v_proj`, then extend to `q_proj` / `o_proj` and the MLP projections.",
        "- Phase 3 优先目标应是 attention `k_proj` / `v_proj` 的 fused dequant-GEMV，再扩展到 `q_proj` / `o_proj` 与 MLP 投影层。",
        "- Validation target: preserve INT4 VRAM savings while pulling decode latency materially below the current INT4 baseline and closer to FP16.",
        "- 验证目标：保住 INT4 的显存优势，同时把 decode 延迟显著拉低到当前 INT4 基线之下，并尽量逼近 FP16。",
        "- Phase 3 benchmarking should keep the same model and prompt/decode settings so results remain directly comparable to this report.",
        "- Phase 3 benchmark 应继续沿用同一模型与同一 prompt/decode 配置，保证结果可以直接与本报告对比。",
        "",
        "## Artifact Checklist / 交付清单",
        "",
        f"- Canonical figures / 标准图表: `{output_dir}`",
        f"- Final report / 最终报告: `{report_path}`",
        f"- Raw repeated runs / 原始重复实验数据: `{data_dir}`",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 charts and report")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--report-path", default="report/analysis_report.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV data...")
    dfs = load_data(str(data_dir))
    if not dfs:
        print(f"ERROR: no benchmark CSV files found under {data_dir}")
        sys.exit(1)
    print(f"Loaded datasets: {', '.join(sorted(dfs.keys()))}")

    print("\nGenerating charts...")
    plot_layerwise_latency(dfs, str(output_dir))
    plot_memory_growth(dfs, str(output_dir))
    plot_top10_slowest(dfs, str(output_dir))
    plot_roofline(dfs, str(output_dir))

    print("\nComputing statistics...")
    summary = compute_summary_stats(dfs)
    tax_layers = compute_quant_tax_layers(dfs)
    metadata = load_benchmark_metadata(str(data_dir))

    if not tax_layers.empty:
        print("\nTop 10 layers with quantization tax:")
        print(tax_layers.head(10).to_string(index=False))
    else:
        print("\nNo positive INT4 slowdown detected.")

    print("\nWriting report...")
    write_report(
        data_dir=data_dir,
        output_dir=output_dir,
        report_path=report_path,
        summary=summary,
        tax_layers=tax_layers,
        metadata=metadata,
    )

    print("\n=== Done ===")
    print(f"Charts: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
