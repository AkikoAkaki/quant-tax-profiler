# PROJECT BRIEF: LLM Quantization Tax Profiler
> 提供给 Claude Code 的完整上下文文档，用于从零开始构建此项目。

---

## 一、项目背景与目标

### 项目名称
**llm-quant-profiler** — 低显存消费级 GPU 上的 LLM 量化税剖析与算子级优化

### 核心问题
在 8GB VRAM 的 RTX 4060 笔记本 GPU 上，INT4 量化模型理论上权重大小只有 FP16 的 1/4，
但实测中某些层（尤其是 MLP Linear 层）的 Decode 阶段延迟**不降反升**。
本项目的目标是：测量这一现象 → 用系统理论解释它 → 通过 Triton fused kernel 工程性地解决它。

### 项目定位
- 面向 CSC 290/420 Machine Learning Systems for Efficient AI（University of Rochester，Sreepathi Pai 教授）课程的自学项目
- 工程产出型，不是综述型：有可运行代码、可复现数据、可量化结论
- 目标受众：课程教授（用于建立学术联系）+ 简历（ML Systems / LLM Inference Engineer 方向）

### 核心约束
- 硬件：RTX 4060 Laptop 8GB VRAM，ROG Zephyrus G16 2023，Win11 + WSL2 Ubuntu
- 成本：零，全部使用开源模型和工具
- 网络：WSL2 内网络正常（可访问 HuggingFace、PyPI）
- 时间：越快越好，但阶段性可交付，不需要一次做完

---

## 二、核心理论知识（Claude Code 需要理解的背景）

### 2.1 为什么 LLM 推理是 Memory-Bound

GPU 有两个性能上限：
- **Compute bound**：受限于 FLOPS（算力）
- **Memory bound**：受限于内存带宽

RTX 4060 Laptop 关键参数：
- Memory Bandwidth: ~272 GB/s
- FP16 Tensor Core 算力: ~22 TFLOPS
- Ridge Point: 22T / 272G ≈ 80 FLOP/Byte（超过这个值是 compute-bound，低于是 memory-bound）

LLM 的 **Decode 阶段**（每次生成一个 token）是典型的 Memory-bound：
- 每次只做一个 token 的前向传播（batch=1, seq=1）
- 需要把所有模型权重从 VRAM 读一遍
- 矩阵运算退化为 矩阵×向量（GEMV），arithmetic intensity 极低（远低于 ridge point）

因此：减少权重的字节数（量化）= 直接减少内存读取量 = 直接提速，这是量化在 decode 阶段有效的根本原因。

### 2.2 量化税（Quantization Tax）是什么

INT4 量化的原始推理路径（两次全局显存访问）：
```
VRAM[INT4 权重] → 读出 → 反量化(Dequant)成 FP16 → 写回 VRAM → 再读出 → 矩阵乘法 → 输出
```

问题：反量化操作引入了一次额外的 VRAM 读写（INT4 → FP16），这个开销在某些层会抵消甚至超过权重压缩带来的带宽节省，导致 INT4 比 FP16 还慢。这就是"量化税"。

Fused Kernel 的解法（一次全局显存访问）：
```
VRAM[INT4 权重] → 读入 GPU SRAM（共享内存）→ 片上解压 + 直接做矩阵乘 → 写出结果
```
彻底消除中间的 VRAM round-trip。

### 2.3 Prefill vs Decode 阶段的区别

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入 | 完整 prompt（N tokens，N >> 1） | 单个 token（seq_len=1） |
| 运算类型 | GEMM（矩阵×矩阵） | GEMV（矩阵×向量） |
| 瓶颈类型 | Compute-bound | Memory-bound |
| 量化收益 | 有限 | 显著（理论上） |
| KV Cache | 初始化 | 逐步增长 |

在代码中通过 input_ids 的 shape 判断阶段：
- `input_ids.shape[1] > 1` → Prefill
- `input_ids.shape[1] == 1` → Decode

### 2.4 Arithmetic Intensity 估算（用于 Roofline 分析）

对一个 Linear 层 (in_features=H, out_features=4H, batch=1, seq=1)：

FP16 情况：
- FLOPs = 2 × H × 4H（矩阵乘法）
- Bytes = H × 4H × 2（读 FP16 权重）
- Arithmetic Intensity = (2 × H × 4H) / (H × 4H × 2) = 1 FLOP/Byte （极低，memory-bound）

INT4 情况（含反量化）：
- FLOPs ≈ 2 × H × 4H（矩阵乘）+ 反量化 FLOPs（小量）
- Bytes = H × 4H × 0.5（读 INT4）+ H × 4H × 2（写 FP16）+ H × 4H × 2（再读 FP16）
- 实际 Bytes 比 FP16 还多（如果反量化是 out-of-place 的）

---

## 三、项目结构（最终目标）

```
llm-quant-profiler/
├── README.md                      # 项目说明，包含结论摘要和复现步骤
├── requirements.txt               # 所有依赖
├── setup.sh                       # 一键环境配置脚本
├── profiler/
│   ├── __init__.py
│   ├── hook_profiler.py           # 核心：PyTorch Hook + CUDA Event 测量
│   ├── metrics.py                 # 工具函数：CUDA Event 计时、显存读取
│   └── phase_detector.py          # Prefill/Decode 阶段自动识别
├── analysis/
│   ├── __init__.py
│   ├── visualize.py               # 图表生成：逐层对比、显存水位线
│   └── roofline.py                # Arithmetic Intensity 估算 + Roofline 图
├── kernels/
│   ├── __init__.py
│   └── fused_dequant_matmul.py    # Triton fused kernel（阶段3）
├── scripts/
│   ├── run_benchmark.py           # 主入口：运行完整 benchmark
│   └── run_analysis.py            # 主入口：生成分析报告和图表
├── data/                          # 自动生成，不提交到 git
│   ├── fp16_prefill.csv
│   ├── fp16_decode.csv
│   ├── int4_prefill.csv
│   └── int4_decode.csv
├── outputs/                       # 图表和报告输出目录
└── report/
    └── analysis_report.md         # 最终分析报告模板
```

---

## 四、分阶段开发计划

### 阶段 0：环境搭建（不需要 Claude Code 写代码，仅记录）

目标：WSL2 Ubuntu 能用 CUDA 跑 PyTorch

验证命令：
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出：True NVIDIA GeForce RTX 4060 Laptop GPU
```

依赖安装：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install pandas matplotlib seaborn
pip install triton
```

---

### 阶段 1：Benchmark 脚本与数据捕获

#### 1.1 hook_profiler.py 核心逻辑

需要实现：
1. `HookProfiler` 类，接收一个 `torch.nn.Module`（HuggingFace 模型）
2. 用 `register_forward_pre_hook` 和 `register_forward_hook` 包裹每个目标层
3. 目标层类型：`torch.nn.Linear`（主要）、`torch.nn.LayerNorm`（次要）
4. 每层记录：
   - `layer_name`：模块路径，如 `model.layers.0.mlp.gate_proj`
   - `layer_type`：`Linear` / `LayerNorm` 等
   - `phase`：`prefill` 或 `decode`（由外部传入）
   - `time_ms`：用 CUDA Event 测量的真实 GPU 耗时
   - `mem_before_mb`：执行前 `torch.cuda.memory_allocated() / 1e6`
   - `mem_after_mb`：执行后同上
   - `mem_peak_mb`：`torch.cuda.max_memory_allocated() / 1e6`
   - `input_shape`：输入张量的 shape（用于 FLOPs 估算）
   - `output_shape`：输出张量的 shape
5. 提供 `export_csv(path)` 方法导出数据

**关键技术细节：**
- GPU 计算是异步的，必须用 `torch.cuda.Event` 而不是 `time.time()`
- 正确用法：
  ```python
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  # 在 pre_hook 里：start_event.record()
  # 在 post_hook 里：end_event.record(); torch.cuda.synchronize(); elapsed = start_event.elapsed_time(end_event)
  ```
- `torch.cuda.reset_peak_memory_stats()` 在每层开始前调用，确保峰值显存是该层的
- Hook 需要在测量结束后用 `handle.remove()` 清除，避免影响正常推理

#### 1.2 run_benchmark.py 主脚本逻辑

```
1. 解析参数：--model_id, --quantization (fp16/int4), --prompt, --max_new_tokens, --output_dir
2. 加载模型：
   - FP16: AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, device_map="cuda")
   - INT4: AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, device_map="cuda")
3. 实例化 HookProfiler，注入 hooks
4. Warmup：跑一次短推理（不记录），让 GPU 预热
5. Prefill 测量：
   - 输入 512 token 长 prompt
   - 只做一次前向（不生成），或用 max_new_tokens=1
   - 标记 phase="prefill"
6. Decode 测量：
   - 输入短 prompt（10 tokens）
   - 生成 128 tokens
   - 逐 token 记录（每次 forward 都是一次 decode step）
   - 标记 phase="decode"
7. 导出 CSV 到 output_dir
8. 打印汇总统计：总耗时、峰值显存、每阶段平均 tokens/s
```

推荐模型：`Qwen/Qwen2.5-1.5B-Instruct`（FP16 约 3GB，INT4 约 1GB，均可在 8GB 内运行）
INT4 量化替代：`Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` 或用 bitsandbytes `load_in_4bit=True`

---

### 阶段 2：分析与可视化

#### 2.1 visualize.py 需要生成的图表

**图 1：逐层耗时对比（FP16 vs INT4）**
- X 轴：层名称（按执行顺序）
- Y 轴：耗时（ms）
- 两条线：FP16（蓝）和 INT4（橙）
- 分 Prefill 和 Decode 两张子图
- 重点：标注出 INT4 比 FP16 慢的层（量化税最重的地方）

**图 2：KV Cache 显存增长曲线**
- X 轴：Decode step（生成第几个 token）
- Y 轴：累计显存占用（MB）
- 展示 KV Cache 随序列长度线性增长的行为

**图 3：Top-10 最慢层对比**
- 水平条形图
- FP16 和 INT4 并排对比
- 直观展示哪些层是瓶颈

**图 4：Roofline 图**
- X 轴：Arithmetic Intensity（FLOP/Byte，log scale）
- Y 轴：实测性能（GFLOPS，log scale）
- 画出理论内存带宽上限线（272 GB/s）和计算上限线（22 TFLOPS）
- 每层用散点表示，FP16/INT4 用不同颜色
- 展示大多数层落在内存带宽限制下方（memory-bound）

#### 2.2 roofline.py 核心函数

```python
def estimate_arithmetic_intensity(layer_name, input_shape, output_shape, dtype):
    """
    根据层名和 shape 估算 FLOPs 和 Bytes，返回 Arithmetic Intensity
    主要针对 Linear 层：
    - FLOPs = 2 * in_features * out_features * batch_size
    - Bytes(FP16) = in_features * out_features * 2
    - Bytes(INT4) = in_features * out_features * 0.5 + dequant_overhead
    """
```

---

### 阶段 3：Triton Fused Kernel（可选，有余力时实现）

#### 目标
实现 `fused_dequant_matmul.py`，包含：
1. 一个 Triton kernel：接受 INT4 packed weights，在 GPU SRAM 内解压并做矩阵乘法
2. 一个 Python wrapper：`fused_dequant_matmul(x, w_int4, scales, zeros)`
3. 数值正确性验证：`torch.allclose(result, reference, atol=1e-2)`
4. 性能对比：与 bitsandbytes 原生实现的耗时对比

**Triton 入门资源：**
- 官方教程：`triton-lang.org/main/getting-started/tutorials/`
- 先读 Tutorial 01（vector add）和 Tutorial 03（matrix multiplication）
- INT4 packing：两个 INT4 值打包在一个 INT8 里，解包用位运算

**简化实现策略：**
- 不需要实现完整的 GPTQ/AWQ 量化格式，只需实现最简单的 per-tensor 或 per-row INT4 量化
- 重点在于展示 fusion 的概念和测量 fusion 带来的收益
- 能解释为什么快（或为什么没有预期快）比跑赢工业级库更重要

---

## 五、技术选型与工具链

| 工具 | 用途 | 备注 |
|------|------|------|
| `transformers` | 加载 HuggingFace 模型 | 主要依赖 |
| `bitsandbytes` | INT4/INT8 量化加载 | `load_in_4bit=True` |
| `accelerate` | 设备管理 `device_map="cuda"` | transformers 依赖 |
| `torch.cuda.Event` | 精确 GPU 计时 | 核心，不用 time.time() |
| `torch.cuda.memory_allocated()` | 显存追踪 | |
| `register_forward_hook` | 非侵入式层级测量 | 核心 |
| `pandas` | 数据处理和 CSV 导出 | |
| `matplotlib` + `seaborn` | 可视化 | |
| `triton` | 阶段3 fused kernel | 需 WSL2 |
| `nvidia-smi` | 全局显存监控 | 终端工具 |

---

## 六、已知坑和注意事项

### 环境相关
- **bitsandbytes** 在 Windows 原生不支持，必须在 WSL2 里运行
- **Triton** 同样只支持 Linux（WSL2 OK）
- WSL2 里 `nvidia-smi` 必须能看到 GPU（需要 Win11 + 较新 WSL2 版本）
- PyTorch CUDA 版本要和驱动匹配，用 `cu121` 版本（对应 CUDA 12.1）

### 测量相关
- GPU 计算异步，hook 里直接用 `time.time()` 测出来的是 CPU 时间，不是 GPU 时间
- 必须调用 `torch.cuda.synchronize()` 后才能读取 CUDA Event 时间
- Warmup 很重要：第一次推理有 CUDA kernel 编译开销，不能算入正式测量
- 每次测量前调用 `torch.cuda.reset_peak_memory_stats()` 清零峰值显存统计

### 模型相关
- `Qwen2.5-1.5B-Instruct` FP16 约 3GB，INT4 约 1GB，是最安全的选择
- 如果要测 `Phi-3-mini`（3.8B），FP16 约 7.6GB，需要确认 8GB 够用
- HuggingFace 模型首次下载需要网络，之后会缓存在 `~/.cache/huggingface/`
- `device_map="cuda"` 会把整个模型放在 GPU，`device_map="auto"` 会自动分配（可能 CPU offload）

### Hook 相关
- `register_forward_hook` 在 forward 之后触发，看不到 forward 内部
- 要测单层耗时需要 `register_forward_pre_hook`（before）和 `register_forward_hook`（after）配对使用
- Hook 的 `output` 参数可能是 tuple，需要处理 `output[0]` 的情况
- 不要在 hook 里做重型计算，会影响推理速度

---

## 七、里程碑与交付标准

### 阶段 1 完成标准
- [ ] `run_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --quantization fp16` 可正常运行
- [ ] `run_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --quantization int4` 可正常运行
- [ ] 生成 4 个 CSV 文件：fp16_prefill, fp16_decode, int4_prefill, int4_decode
- [ ] 每个 CSV 包含至少：layer_name, time_ms, mem_before_mb, mem_after_mb, mem_peak_mb, phase

### 阶段 2 完成标准
- [ ] `run_analysis.py` 生成 4 张图表（逐层对比、显存增长、Top-10 对比、Roofline）
- [ ] 能明确指出：哪几个层是量化税最重的（INT4 比 FP16 慢 X%）
- [ ] analysis_report.md 包含：现象描述、理论解释（用 Memory Hierarchy 和 Arithmetic Intensity）、图表引用

### 阶段 3 完成标准（可选）
- [ ] `fused_dequant_matmul` 数值正确（`torch.allclose` 通过）
- [ ] 有与原始实现的性能对比数据
- [ ] 能解释结果（快/慢的原因）

---

## 八、简历 Bullet 模板（待填入真实数据）

```
LLM Quantization Tax Analysis & Operator Optimization | Self-directed Project (CSC 290/420 curriculum)
• Built custom layer-wise profiler using PyTorch forward hooks + CUDA Events (~150 LOC, 
  no external profiling tools) to measure FP16 vs INT4 inference on RTX 4060 8GB across 
  Prefill/Decode phases; identified dequantization overhead as primary bottleneck in MLP layers
• Quantified "quantization tax": INT4 decode latency on [target layer] exceeded FP16 by X% 
  despite 4× weight compression; explained mechanism via Roofline model and memory hierarchy analysis
• [阶段3完成后] Implemented Triton fused dequant+matmul kernel eliminating intermediate 
  VRAM round-trip; achieved Y% throughput improvement on bottleneck layers vs. baseline
• Open-sourced reproducible benchmark suite: github.com/[handle]/llm-quant-profiler
```

---

## 九、与教授联系的时机和策略

**时机：** 阶段 2 数据出来后，不要等项目完成。

**核心原则：** 带具体数据和具体问题，不要泛泛请教。

**邮件模板：**

> Subject: Self-study project on CSC 290/420 material — question on quantization overhead and roofline modeling
>
> Dear Prof. Pai,
>
> I've been self-studying your CSC 290/420 course materials and running experiments on my RTX 4060 Laptop (8GB VRAM). I built a layer-wise profiler using PyTorch hooks and CUDA Events to measure FP16 vs INT4 inference behavior on Qwen2.5-1.5B.
>
> I found something that surprised me: in the decode phase, certain MLP linear layers show *higher* latency under INT4 than FP16, despite weights being 4× smaller. My hypothesis is that out-of-place dequantization introduces an extra global memory round-trip that negates the bandwidth savings — but when I try to model this on the roofline, I'm unsure how to properly account for the dequant byte cost.
>
> Would you have time for a brief conversation? I'd be happy to share my profiling data and scripts.
>
> Best,
> Aki

---

## 十、课程知识映射（项目内容对应课程哪里）

| 项目内容 | 对应课程讲义 |
|---------|-------------|
| GPU 并行计算原理 | Lecture 03: Compute SIMD and GPUs |
| Memory Hierarchy 分析 | Lecture 06: Memory |
| 显存水位线追踪 | Lecture 11: Memory and Storage in ML Programs |
| ML 程序计算图结构 | Lecture 08: ML programs as Computation Graphs |
| Prefill/Decode 执行模式 | Lecture 10: Executing ML Programs |
| Arithmetic Intensity / Roofline | Lecture 06 + 09 (Loop-Intensive Code) |
| 量化与 Fine-tuning | Lecture 15: Fine Tuning |

---

*本文档由 Claude (claude.ai) 生成，基于与用户 Aki 的多轮对话整理，涵盖课程调研、硬件约束分析、多 AI 系统讨论综合后的最终方案。*
