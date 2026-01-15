#!/bin/bash
# Installation script - Ensure PyTorch GPU version is installed correctly

echo "=== Install PyTorch GPU version (CUDA 11.8) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "=== Install other dependencies ==="
pip install transformers>=4.45.0 accelerate>=0.25.0 peft>=0.7.0 pillow>=10.0.0 qwen-vl-utils>=0.0.8 -i https://mirrors.aliyun.com/pypi/simple/

echo ""
echo "=== Verify installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"