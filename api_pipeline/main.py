#!/usr/bin/env python3
"""
API Pipeline - 使用 API 调用实现链路一（跳过步骤一）

用法:
    # 链路一：从 step1 结果构建数据集
    python api_pipeline/main.py pipeline1 --input ./data/output/dataset_step1.jsonl --output ./data/output/dataset_api.jsonl

    # 链路二：用户优化（单张图片）
    python api_pipeline/main.py pipeline2 --image ./data/sample.png --summary "原始总结内容"

    # 链路三：直接评分（单张图片）
    python api_pipeline/main.py pipeline3 --image ./data/sample.png --summary "待评分的总结内容"
"""
import sys
import os
import argparse
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_pipeline.pipeline import (
    pipeline1_build_dataset_from_step1,
    pipeline2_user_optimize,
    pipeline3_direct_score,
)
from src.utils import save_to_jsonl, get_total_score


def cmd_pipeline1(args):
    """链路一：从 step1 结果构建数据集"""
    print("执行 API 版链路一：从 step1 结果构建数据集")
    
    success, total = pipeline1_build_dataset_from_step1(
        input_path=args.input,
        output_path=args.output,
        max_retries=args.max_retries,
        resume=args.resume,
    )
    
    print(f"完成: 成功 {success}/{total}")


def cmd_pipeline2(args):
    """链路二：用户优化"""
    print("执行 API 版链路二：用户优化")
    
    result = pipeline2_user_optimize(
        image_path=args.image,
        summary=args.summary,
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
    print("执行 API 版链路三：直接评分")
    
    result = pipeline3_direct_score(
        image_path=args.image,
        summary=args.summary,
    )
    
    if result:
        print("\n评分结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        total = get_total_score(result)
        print(f"总分: {total}/10")
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"结果已保存到: {args.output}")
    else:
        print("评分失败")


def main():
    parser = argparse.ArgumentParser(
        description="API Pipeline - 使用 API 调用实现链路功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 链路一
    p1_parser = subparsers.add_parser(
        "pipeline1",
        help="链路一：从 step1 结果构建数据集",
    )
    p1_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="step1 结果文件路径（dataset_step1.jsonl）",
    )
    p1_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径",
    )
    p1_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="最大重试次数（默认：3）",
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
        help="链路二：用户优化（单张图片）",
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
        help="最大重试次数（默认：3）",
    )
    p2_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（可选）",
    )
    p2_parser.set_defaults(func=cmd_pipeline2)
    
    # 链路三
    p3_parser = subparsers.add_parser(
        "pipeline3",
        help="链路三：直接评分（单张图片）",
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
    p3_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（可选）",
    )
    p3_parser.set_defaults(func=cmd_pipeline3)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == "__main__":
    main()

