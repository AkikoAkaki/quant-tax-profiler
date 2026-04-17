# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Purpose

**quant-tax-profiler** measures the "quantization tax" in LLM inference: INT4 quantization compresses weights 4×, but out-of-place dequantization can introduce an extra VRAM round-trip that makes certain layers *slower* than FP16. The goal is to measure this phenomenon, explain it via Roofline analysis, and optionally fix it with a Triton fused kernel.

**Hardware target**: RTX 4060 Laptop 8GB VRAM (272 GB/s bandwidth, 22 TFLOPS FP16, ridge point ~80 FLOP/Byte). Decode phase is memory-bound (~1 FLOP/Byte).

**Run environment**: WSL2 Ubuntu. bitsandbytes and triton do not work on native Windows.

## Commands

### Environment setup (WSL2, first time)
```bash
bash setup.sh
source venv/bin/activate
```

### Run benchmarks
```bash
# FP16 baseline
python scripts/run_benchmark.py --quantization fp16 --prompt-len 512 --max-new-tokens 128

# INT4 quantized
python scripts/run_benchmark.py --quantization int4 --prompt-len 512 --max-new-tokens 128

# Quick test (fast, for code validation)
python scripts/run_benchmark.py --quantization fp16 --prompt-len 128 --max-new-tokens 32
```

Outputs CSV files to `data/` (gitignored): `{fp16,int4}_{prefill,decode}.csv`

### Run analysis (Phase 2, not yet implemented)
```bash
python scripts/run_analysis.py
```

## Architecture

### Data flow
```
run_benchmark.py
  → load_model()         # HuggingFace AutoModelForCausalLM (fp16 or bitsandbytes int4)
  → HookProfiler.attach()  # registers pre+post hooks on all nn.Linear + nn.LayerNorm
  → warmup()             # one short generation, not recorded
  → run_prefill()        # 1 forward pass, 512 tokens → 197 records
  → run_decode()         # 128 generate steps, 1 token each → 6304 records
  → export_csv()         # pandas DataFrame → data/*.csv
```

### Key classes
- **`HookProfiler`** (`profiler/hook_profiler.py`): Attaches `register_forward_pre_hook` + `register_forward_hook` pairs. Pre-hook starts a `CUDATimer` and snapshots `memory_allocated`. Post-hook stops the timer (calls `torch.cuda.synchronize()`), records peak memory, and appends a dict to `self.records`.
- **`CUDATimer`** (`profiler/metrics.py`): Wraps `torch.cuda.Event(enable_timing=True)`. Must use CUDA Events (not `time.time()`) because GPU execution is async.
- **`PhaseDetector`** (`profiler/phase_detector.py`): `input_ids.shape[1] > 1` → prefill; `== 1` → decode.

### CSV schema
`layer_name, layer_type, phase, time_ms, mem_before_mb, mem_after_mb, mem_peak_mb, input_shape, output_shape`

### Decode loop detail
`run_decode()` manually drives the KV-cache loop (instead of `model.generate()`) so each token step triggers hooks individually. The initial prefill to build the KV cache is done with `profiler.recording = False` to avoid double-counting.

## Development phases

| Phase | Status | Files |
|-------|--------|-------|
| 1: Benchmark + data capture | ✅ Done | `profiler/`, `scripts/run_benchmark.py` |
| 2: Analysis + visualization | 🔲 Next | `analysis/visualize.py`, `analysis/roofline.py`, `scripts/run_analysis.py` |
| 3: Triton fused kernel | 🔲 Optional | `kernels/fused_dequant_matmul.py` |

## Phase 2 spec (next to implement)

Four charts needed in `outputs/`:
1. **Per-layer latency**: FP16 vs INT4, split by prefill/decode. Annotate layers where INT4 > FP16.
2. **KV cache memory growth**: memory_allocated vs decode step.
3. **Top-10 slowest layers**: horizontal bar chart, FP16 vs INT4 side-by-side.
4. **Roofline**: scatter plot of arithmetic intensity (FLOP/Byte) vs throughput (GFLOPS), with 272 GB/s and 22 TFLOPS ceiling lines.

Arithmetic intensity for a Linear layer (batch=1, seq=1):
- FP16: `2 * in * out` FLOPs / `in * out * 2` bytes = **1 FLOP/Byte**
- INT4: same FLOPs / `(in * out * 0.5 + in * out * 2 + in * out * 2)` bytes = **~0.44 FLOP/Byte** (worse due to dequant round-trip)

## Model

Default: `Qwen/Qwen2.5-1.5B-Instruct` (FP16 ~3GB, INT4 ~1GB). Downloaded and cached in `~/.cache/huggingface/` on first run.
