#!/bin/bash
# 安装脚本 - 确保 PyTorch GPU 版本正确安装

echo "=== 安装 PyTorch GPU 版本 (CUDA 11.8) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "=== 安装其他依赖 ==="
pip install transformers>=4.45.0 accelerate>=0.25.0 peft>=0.7.0 pillow>=10.0.0 qwen-vl-utils>=0.0.8 -i https://mirrors.aliyun.com/pypi/simple/

echo ""
echo "=== 验证安装 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

