#!/usr/bin/env python3
"""
图表总结评估系统 - 主入口

功能：
- 功能一：批量图片总结生成
- 功能二：五维度评分 + improved_summary
- 功能三：仅五维度评分

链路：
- 链路一：构建数据集
- 链路二：用户优化
- 链路三：直接评分

微调：
- 方案一 (l-1)：含 improved_summary
- 方案二 (l-2)：不含 improved_summary
"""
import sys
import os

# ============================================================
# 重要：在任何其他 import 之前设置 GPU
# 某些库（如 transformers）在 import 时会初始化 CUDA 上下文
# 必须在此之前设置 CUDA_VISIBLE_DEVICES
# ============================================================
def _setup_gpu_early():
    """在任何 CUDA 相关库导入之前设置 GPU"""
    gpu_id = None
    
    # 检查命令行中是否有 --gpu 参数
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            break
        elif arg.startswith('--gpu='):
            gpu_id = arg.split('=')[1]
            break
    
    if gpu_id is not None:
        # 重要：设置设备顺序为 PCI 总线顺序，确保与 nvidia-smi 一致
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # 清除可能存在的旧设置
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"[GPU] 清除旧设置: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # 设置新的 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"[GPU] 设置 CUDA_DEVICE_ORDER=PCI_BUS_ID")
        print(f"[GPU] 设置 CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # 验证设置
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] PyTorch 可见 GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"[GPU] cuda:{i} = {props.name}, 显存: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("[GPU] 警告: CUDA 不可用!")

_setup_gpu_early()
# ============================================================

import argparse
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature1_summary import batch_generate_summaries, generate_summary
from src.feature2_evaluate import evaluate_with_improvement
from src.feature3_score import score_only
from src.pipeline import (
    pipeline1_build_dataset,
    pipeline2_user_optimize,
    pipeline3_direct_score,
)
from src.utils import save_to_jsonl, get_total_score


def cmd_feature1(args):
    """功能一：批量图片总结生成"""
    print("执行功能一：批量图片总结生成")
    
    results = batch_generate_summaries(
        image_folder=args.image_folder,
        low_ratio=args.low_ratio,
        medium_ratio=args.medium_ratio,
        high_ratio=args.high_ratio,
        model_path=args.model_path,
        lora_path=args.lora_path,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.output,
        resume=args.resume,
    )
    
    print(f"完成，共 {len(results)} 条结果")
    if args.output:
        print(f"结果已保存到: {args.output}")


def cmd_feature2(args):
    """功能二：五维度评分 + improved_summary"""
    print("执行功能二：五维度评分 + improved_summary")
    
    result = evaluate_with_improvement(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"结果已保存到: {args.output}")
    else:
        print("评估失败")


def cmd_feature3(args):
    """功能三：仅五维度评分"""
    print("执行功能三：仅五维度评分")
    
    result = score_only(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        total = get_total_score(result)
        print(f"总分: {total}/10")
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"结果已保存到: {args.output}")
    else:
        print("评分失败")


def cmd_pipeline1(args):
    """链路一：构建数据集"""
    print("执行链路一：构建数据集")
    
    success, total = pipeline1_build_dataset(
        image_folder=args.image_folder,
        output_path=args.output,
        low_ratio=args.low_ratio,
        medium_ratio=args.medium_ratio,
        high_ratio=args.high_ratio,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_path_feature2=args.lora_path_f2,
        lora_path_feature3=args.lora_path_f3,
        max_retries=args.max_retries,
        resume=args.resume,
    )
    
    print(f"完成: 成功 {success}/{total}")


def cmd_pipeline2(args):
    """链路二：用户优化"""
    print("执行链路二：用户优化")
    
    result = pipeline2_user_optimize(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_retries=args.max_retries,
    )
    
    if result:
        print("\n最终结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"结果已保存到: {args.output}")
    else:
        print("优化失败")


def cmd_pipeline3(args):
    """链路三：直接评分"""
    print("执行链路三：直接评分")
    
    result = pipeline3_direct_score(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
    )
    
    if result:
        print("\n评分结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"结果已保存到: {args.output}")
    else:
        print("评分失败")


def cmd_train(args):
    """执行 LoRA 微调"""
    from training.train_lora import train_lora
    from training.data_format import convert_to_training_format
    
    # 如果需要先转换数据格式
    if args.raw_data:
        print("转换数据格式...")
        training_data_path = args.data_path.replace(".jsonl", "_training.json")
        convert_to_training_format(
            input_path=args.raw_data,
            output_path=training_data_path,
            include_improved_summary=(args.scheme == "l-1"),
        )
        args.data_path = training_data_path
    
    # 设置输出目录
    if not args.output_dir:
        args.output_dir = f"./lora_weights/{args.scheme}"
    
    train_lora(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        resume_lora_path=args.resume_lora_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="图表总结评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 通用参数
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="指定使用的 GPU ID（如 0, 1, 2...），会设置 CUDA_VISIBLE_DEVICES",
    )
    common_parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="模型路径",
    )
    common_parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 权重路径",
    )
    common_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度",
    )
    common_parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p 采样",
    )
    common_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径",
    )
    
    # 功能一
    f1_parser = subparsers.add_parser(
        "feature1",
        parents=[common_parser],
        help="功能一：批量图片总结生成",
    )
    f1_parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="图片文件夹路径",
    )
    f1_parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.3,
        help="低质量比例",
    )
    f1_parser.add_argument(
        "--medium_ratio",
        type=float,
        default=0.3,
        help="中质量比例",
    )
    f1_parser.add_argument(
        "--high_ratio",
        type=float,
        default=0.4,
        help="高质量比例",
    )
    f1_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="断点续传（默认开启）",
    )
    f1_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="禁用断点续传，从头开始",
    )
    f1_parser.set_defaults(func=cmd_feature1)
    
    # 功能二
    f2_parser = subparsers.add_parser(
        "feature2",
        parents=[common_parser],
        help="功能二：五维度评分 + improved_summary",
    )
    f2_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图片路径",
    )
    f2_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="原始总结",
    )
    f2_parser.set_defaults(func=cmd_feature2)
    
    # 功能三
    f3_parser = subparsers.add_parser(
        "feature3",
        parents=[common_parser],
        help="功能三：仅五维度评分",
    )
    f3_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图片路径",
    )
    f3_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="待评估的总结",
    )
    f3_parser.set_defaults(func=cmd_feature3)
    
    # 链路一
    p1_parser = subparsers.add_parser(
        "pipeline1",
        parents=[common_parser],
        help="链路一：构建数据集",
    )
    p1_parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="图片文件夹路径",
    )
    p1_parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.3,
        help="低质量比例",
    )
    p1_parser.add_argument(
        "--medium_ratio",
        type=float,
        default=0.3,
        help="中质量比例",
    )
    p1_parser.add_argument(
        "--high_ratio",
        type=float,
        default=0.4,
        help="高质量比例",
    )
    p1_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="最大重试次数",
    )
    p1_parser.add_argument(
        "--lora_path_f2",
        type=str,
        default=None,
        help="功能二（评分+改进总结）使用的 LoRA 权重路径，如 ./lora_weights/l-1",
    )
    p1_parser.add_argument(
        "--lora_path_f3",
        type=str,
        default=None,
        help="功能三（仅评分）使用的 LoRA 权重路径，如 ./lora_weights/l-2",
    )
    p1_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="断点续传（默认开启）",
    )
    p1_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="禁用断点续传，从头开始",
    )
    p1_parser.set_defaults(func=cmd_pipeline1)
    
    # 链路二
    p2_parser = subparsers.add_parser(
        "pipeline2",
        parents=[common_parser],
        help="链路二：用户优化",
    )
    p2_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图片路径",
    )
    p2_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="原始总结",
    )
    p2_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="最大重试次数",
    )
    p2_parser.set_defaults(func=cmd_pipeline2)
    
    # 链路三
    p3_parser = subparsers.add_parser(
        "pipeline3",
        parents=[common_parser],
        help="链路三：直接评分",
    )
    p3_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="图片路径",
    )
    p3_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="待评估的总结",
    )
    p3_parser.set_defaults(func=cmd_pipeline3)
    
    # 微调
    train_parser = subparsers.add_parser(
        "train",
        help="LoRA 微调",
    )
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="基础模型路径",
    )
    train_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练数据路径（JSON 格式）",
    )
    train_parser.add_argument(
        "--raw_data",
        type=str,
        default=None,
        help="原始 JSONL 数据路径（如提供则先转换格式）",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录",
    )
    train_parser.add_argument(
        "--scheme",
        type=str,
        choices=["l-1", "l-2"],
        default="l-1",
        help="微调方案",
    )
    train_parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    train_parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    train_parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    train_parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小",
    )
    train_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    train_parser.add_argument(
        "--resume_lora_path",
        type=str,
        default=None,
        help="现有 LoRA 权重路径（用于增量微调，在已有权重基础上继续训练）",
    )
    train_parser.set_defaults(func=cmd_train)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == "__main__":
    main()

