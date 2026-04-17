# Quant Tax Profiler — Roadmap

> 状态截至 2026-04-06

---

## 当前数据解读（先读懂再动手）

### 数据修正：之前分析脚本有 bug

`run_analysis.py` 的 layer type 过滤只统计了 `Linear`，
但 bitsandbytes INT4 把量化层替换成了 `Linear4bit`，导致之前的 "INT4 Linear 只用了 219ms" 是错的。

**实际数字：**

| 指标 | FP16 | INT4 |
|------|------|------|
| Decode 总时间 | 895.74 ms | 2604.47 ms |
| Decode 耗时比 | 1.00x | **2.91x 更慢** |
| 量化层类型 | `Linear` × 6304 | `Linear4bit` × 6272 + `Linear` × 32 |
| VRAM 峰值 | 3098 MB | 1252 MB（-59.6%） |
| Prefill 耗时比 | 1.00x | **3.60x 更慢** |

**实际 INT4 Linear4bit 耗时 = 2604.47 - 219.81 = 2384.66 ms**，
比 FP16 全部 Linear（895.74 ms）慢 **2.66 倍**。

---

### 图 1：Per-layer Latency（FP16 vs INT4）

**说明了什么：**
- Decode 阶段，INT4 **每一层都比 FP16 慢**（没有任何层是 INT4 更快）
- 最严重的层全部是 attention 的 `k_proj` / `v_proj`，慢 8–12x
- MLP 层（gate_proj, up_proj, down_proj）慢约 2x，相对温和
- 原因是矩阵形状不同（见下）

**关键洞察 — 为什么 k/v proj 最惨：**

Qwen2.5-1.5B 使用 GQA（Grouped Query Attention），k_proj 和 v_proj 的输出维度只有 **256**（vs q_proj/o_proj 的 1536）。矩阵越小，FLOPs 越少，但 dequant 开销是固定的 → 相对开销更大。

```
q_proj: (1, 1, 1536) → (1, 1, 1536)  ratio ≈ 2.5x
k_proj: (1, 1, 1536) → (1, 1, 256)   ratio ≈ 8–12x  ← 最惨
v_proj: (1, 1, 1536) → (1, 1, 256)   ratio ≈ 8–12x  ← 最惨
o_proj: (1, 1, 1536) → (1, 1, 1536)  ratio ≈ 2.8x
```

---

### 图 2：KV Cache Memory Growth

**说明了什么：**
- FP16 decode 全程 VRAM ≈ 3098 MB，INT4 ≈ 1252 MB，差值稳定
- 两条线都**几乎是平的**（32 步内 KV cache 增长微乎其微）
- 这是符合预期的：32 个 decode step × 每步 1 token，cache 增量极小

**结论：** KV cache 增长不是这次实验的瓶颈，图 2 主要证明了"量化权重确实省了 VRAM"。

---

### 图 3：Top-10 Slowest Layers

**说明了什么：**
- 前 10 全是 `k_proj` / `v_proj`，这验证了"小矩阵 + dequant overhead = 最大 tax"
- 水平 bar 图直观展示了差距的量级（橙 vs 蓝）
- `lm_head` 没有进 top 10，因为它是 `Linear`（没有被量化）

---

### 图 4：Roofline

**说明了什么：**
- 所有 decode 层都在 roofline 的**左下角**（极度 memory-bound）
- FP16 点（蓝）的算术强度 ≈ 1 FLOP/Byte
- INT4 点（橙）的算术强度 ≈ 0.44 FLOP/Byte（更差，因为 dequant round-trip 增加了 bytes）
- 两者离脊点（80 FLOP/Byte）都极远，都没有发挥出计算资源
- **结论：当前 INT4 实现反而加重了内存瓶颈**

---

## Phase 3 前置学习（顺序执行）

### Step L1：GPU 内存层级（1–2 天）

理解 fused kernel 在 fuse 什么，这是一切的基础。

**需要掌握：**
```
VRAM（HBM/GDDR6）
  ↓ ~272 GB/s（你的卡）
L2 Cache（共享，~32MB）
  ↓ ~3 TB/s
L1 Cache / Shared Memory（SRAM，per SM，~64KB）
  ↓ ~20 TB/s
Registers（per thread）
```

**核心概念：**
- Shared Memory（SRAM）是片上的，读写不消耗 VRAM 带宽
- 当前 bitsandbytes 的问题：INT4 → FP16 dequant **写回 VRAM**，再从 VRAM 读出来做 matmul，浪费了 2 次 VRAM 带宽
- Fused kernel 的目标：把 INT4 读进 Shared Memory，在片上 dequant，直接做 GEMV，**不写回 VRAM**

**学习资源：**
- CUDA Programming Guide Chapter 5（Memory Hierarchy）
- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)

---

### Step L2：bitsandbytes NF4 存储格式（半天）

你需要知道你要解码的是什么格式。

**关键点：**
- NF4（Normal Float 4）：专为正态分布权重设计的非线性 4-bit 编码，有 16 个不均匀分布的码点
- **Block-wise quantization**：每 64 个权重共享 1 个 FP32 scale（absmax）
- 实际存储：packed INT8（每字节装 2 个 4-bit 值）+ scale tensor

**解码逻辑（伪代码）：**
```python
# 每个 4-bit 值映射到 NF4 码本中的 FP16 值
NF4_CODEBOOK = [-1.0, -0.6962, -0.5251, ..., 1.0]  # 16 个值

def dequant_nf4_block(packed_weights, scale):
    # packed_weights: int8 tensor，每字节2个4-bit值
    # scale: float32，该 block 的 absmax
    hi = (packed_weights >> 4) & 0xF
    lo = packed_weights & 0xF
    return [NF4_CODEBOOK[i] * scale for i in interleave(hi, lo)]
```

**学习资源：**
- bitsandbytes 论文：[QLoRA (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314) Section 2.1
- bitsandbytes 源码：`bitsandbytes/functional.py` 中的 `dequantize_blockwise`

---

### Step L3：Triton 入门（2–3 天）

**核心概念（需要掌握）：**
- Triton 的编程模型：程序以 **block** 为单位，每个 block 处理一块数据
- `tl.program_id(axis)` — 等价于 CUDA 的 `blockIdx`
- `tl.load(ptr + offsets, mask)` — 从 VRAM 加载一个 block 的数据
- `tl.store(ptr + offsets, val, mask)` — 写回结果
- `tl.dot(a, b)` — block 级矩阵乘法（使用 Tensor Core）
- `@triton.autotune` — 自动搜索最优 tile size

**先写这个练手（按顺序）：**
1. Vector add kernel（官方 Getting Started）
2. FP16 GEMV kernel（matrix × vector，batch=1）
3. 在 FP16 GEMV 上加 INT4 dequant → fused dequant-GEMV

**学习资源：**
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)（前 3 个 tutorial 必看）
- [Flash Attention Triton 实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)（理解 fused kernel 结构）

---

## Phase 3 实现计划

### Step 1：修复 run_analysis.py 的 bug（30 分钟）

`analysis/visualize.py` 中所有过滤 `layer_type == "Linear"` 的地方，
需要改为同时包含 `"Linear4bit"`。

涉及函数：`compute_quant_tax_layers`、`plot_top10_slowest` 中的聚合。

---

### Step 2：搭建 Triton kernel 框架（半天）

新建 `kernels/fused_dequant_gemv.py`：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def dequant_gemv_kernel(
    # 输入指针
    x_ptr,          # 输入激活 (FP16): [in_features]
    w_ptr,          # 量化权重 (INT8 packed): [out_features, in_features // 2]
    scale_ptr,      # scale (FP32): [out_features, in_features // block_size]
    # 输出
    out_ptr,        # 输出 (FP16): [out_features]
    # 维度
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    block_size: tl.constexpr,   # 64, bitsandbytes 默认
    BLOCK_OUT: tl.constexpr,    # tile 大小（autotuned）
):
    # 每个 program 负责 BLOCK_OUT 个输出神经元
    ...
```

---

### Step 3：实现 NF4 dequant + GEMV（2–3 天）

核心逻辑：
1. 在 kernel 内 decode NF4：packed INT8 → 2 个 4-bit index → NF4 码本 → FP16
2. 乘以 scale（per-block）
3. 与输入激活做点积累加
4. 写出结果

关键挑战：
- NF4 码本要放在 `tl.constexpr` 或 L1 cache 中
- pack 格式：bitsandbytes 的实际 layout 需要对照源码确认
- Scale 的 layout：`[n_blocks_out, n_blocks_in]`，需要正确索引

---

### Step 4：集成进 benchmark（半天）

在 `scripts/run_benchmark.py` 中增加 `--quantization int4-fused` 模式，
用 `torch.nn.Module` 替换 `Linear4bit`，forward 调用 Triton kernel。

验证正确性：
```python
# 对比 bitsandbytes 输出 vs Triton kernel 输出
assert torch.allclose(bnb_out, triton_out, atol=1e-2)
```

---

### Step 5：跑对比 benchmark + 更新报告（半天）

```bash
python scripts/run_benchmark.py --quantization fp16 ...
python scripts/run_benchmark.py --quantization int4 ...
python scripts/run_benchmark.py --quantization int4-fused ...
python scripts/run_analysis.py  # 生成新图，加第5张：FP16 vs INT4 vs INT4-fused
```

**预期结果：**
- INT4-fused decode 时间应接近甚至优于 FP16（理论上 0.5 bytes/weight vs 2.0 bytes/weight）
- k_proj / v_proj 应该从 8-12x slower 变成 ~0.5x（更快）

---

## 里程碑总结

| 里程碑 | 状态 | 预计行动 |
|--------|------|----------|
| Phase 1: Benchmark + profiler | ✅ Done | — |
| Phase 2: Analysis + charts | ✅ Done（有小 bug） | 修复 Linear4bit 过滤 |
| L1: GPU 内存层级 | 🔲 学习 | 读资料 |
| L2: NF4 存储格式 | 🔲 学习 | 读 QLoRA 论文 + bnb 源码 |
| L3: Triton 入门 | 🔲 学习 | 跑官方 tutorial |
| Phase 3 Step 1: fix bug | 🔲 | 改 visualize.py |
| Phase 3 Step 2: kernel 框架 | 🔲 | 建文件，写签名 |
| Phase 3 Step 3: NF4+GEMV | 🔲 | 核心实现 |
| Phase 3 Step 4: 集成 | 🔲 | 替换 bnb 层 |
| Phase 3 Step 5: 对比 | 🔲 | 跑图，写结论 |

---

## 可选延伸（Phase 3 完成后）

- **Nsight Compute 分析**：用 `ncu` 对比 bitsandbytes kernel vs 自己的 Triton kernel，
  看 memory throughput、L2 hit rate、warp occupancy
- **Batch size 实验**：在 batch=1/4/8/16 下测 roofline，看什么时候从 memory-bound 变 compute-bound
- **AWQ/GPTQ 对比**：它们为什么比 bitsandbytes 快？（提示：它们已经用了 fused kernel）
