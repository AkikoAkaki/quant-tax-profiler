# Phase 3 Learning Runbook

This phase is intentionally split into scaffolding first, kernel second.

## Goal

Build enough local understanding to implement the kernel yourself without also fighting the repo structure.

The repo baseline is:

- 4-bit format: `bitsandbytes fp4`
- blocksize: `64`
- target hardware: WSL2 + RTX 4060 Laptop
- target workload: decode-only GEMV

## Files To Start With

- [phase3_utils.py](/I:/Projects/quant-tax-profiler/phase3_utils.py)
- [kernels/fp16_gemv.py](/I:/Projects/quant-tax-profiler/kernels/fp16_gemv.py)
- [kernels/fused_fp4_gemv.py](/I:/Projects/quant-tax-profiler/kernels/fused_fp4_gemv.py)
- [scripts/extract_linear4bit_reference.py](/I:/Projects/quant-tax-profiler/scripts/extract_linear4bit_reference.py)
- [scripts/verify_fp16_gemv.py](/I:/Projects/quant-tax-profiler/scripts/verify_fp16_gemv.py)
- [scripts/verify_fused_fp4.py](/I:/Projects/quant-tax-profiler/scripts/verify_fused_fp4.py)

## Step 1: Inspect One Real Linear4bit Layer

Run in WSL:

```bash
source venv/bin/activate
python scripts/extract_linear4bit_reference.py
```

Success looks like:

- you see `quant_type: fp4`
- you see `blocksize: 64`
- you see packed weight shape `(196608, 1)` for `k_proj`
- you can explain why `256 * 1536 / 2 = 196608`

## Step 2: Implement FP16 GEMV First

Edit `kernels/fp16_gemv.py`.

Do not jump to fp4 yet. First make the simple case work:

- one input vector
- one weight matrix
- output shape `[out_features]`

Run:

```bash
source venv/bin/activate
python scripts/verify_fp16_gemv.py
python scripts/verify_fp16_gemv.py --out-features 1536
```

Success looks like:

- no crash
- output shape matches reference
- `allclose(atol=1e-2, rtol=1e-2): True`

## Step 3: Read The fp4 Artifacts Again

Before writing the fused prototype, inspect the real layer data again and answer:

- each uint8 stores how many 4-bit values?
- how many blocks are there?
- what does `absmax.shape` tell you about block count?
- what does the 16-entry `code` tensor mean?

If you cannot answer those four questions clearly, stop and inspect before coding.

## Step 4: Implement The Fused fp4 Prototype

Edit `kernels/fused_fp4_gemv.py`.

Implement in this order:

1. unpack 4-bit values from each uint8
2. map indices through the fp4 codebook
3. apply block scale from `absmax`
4. accumulate dot products into one output vector
5. add bias if present

Run:

```bash
source venv/bin/activate
python scripts/verify_fused_fp4.py
```

Success looks like:

- candidate output shape `[256]`
- no crash
- error decreases over iterations
- eventually `allclose(atol=1e-2, rtol=1e-2): True`

## Step 5: What You Should Be Able To Explain

Before moving to `int4-fused-kv`, you should be able to explain:

- why decode behaves like GEMV
- why GEMV is memory-bound on this GPU
- what the current fp4 round-trip wastes
- how a fused kernel avoids materializing FP16 weights in VRAM
- why `k_proj` and `v_proj` are the best first integration targets

## What Comes Next

After the standalone prototype is correct, the next phase is:

- wrap only `k_proj` and `v_proj`
- add `int4-fused-kv` to `scripts/run_benchmark.py`
- compare `fp16`, `int4`, and `int4-fused-kv`

Do not start that integration until the standalone verifier is stable.
