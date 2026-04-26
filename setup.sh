#!/bin/bash
set -e

echo "=== llm-quant-profiler environment setup (WSL2) ==="

# Check we're running in WSL2
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "WARNING: This script is designed for WSL2. Proceed at your own risk."
fi

# Check NVIDIA GPU is visible
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure you have NVIDIA drivers installed on Windows."
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Using Python: $(python --version) at $(which python)"

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing project dependencies..."
pip install transformers accelerate bitsandbytes
pip install pandas matplotlib seaborn
pip install triton

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers, accelerate, bitsandbytes
print(f'transformers:  {transformers.__version__}')
print(f'accelerate:    {accelerate.__version__}')
print(f'bitsandbytes:  {bitsandbytes.__version__}')
print()
print('All good! Run: python scripts/run_benchmark.py --help')
"
