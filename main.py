#!/usr/bin/env python3
"""
Figure Summary Evaluation System - Main Entry Point

Features:
- Feature 1: Batch Image Summary Generation
- Feature 2: Five-Dimensional Scoring + improved_summary
- Feature 3: Five-Dimensional Scoring Only

Pipelines:
- Pipeline 1: Build Dataset
- Pipeline 2: User Optimization
- Pipeline 3: Direct Scoring

Fine-tuning:
- Scheme 1 (l-1): Includes improved_summary
- Scheme 2 (l-2): Excludes improved_summary
"""
import sys
import os

# ============================================================
# Important: Set GPU before any other imports
# Some libraries (e.g., transformers) initialize CUDA context on import
# Must set CUDA_VISIBLE_DEVICES before this
# ============================================================
def _setup_gpu_early():
    """Set GPU before importing any CUDA-related libraries"""
    gpu_id = None
    
    # Check if --gpu argument exists in command line
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            break
        elif arg.startswith('--gpu='):
            gpu_id = arg.split('=')[1]
            break
    
    if gpu_id is not None:
        # Important: Set device order to PCI bus order to match nvidia-smi
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Clear possible old settings
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(f"[GPU] Clearing old setting: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Set new GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"[GPU] Set CUDA_DEVICE_ORDER=PCI_BUS_ID")
        print(f"[GPU] Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # Verify settings
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] Number of PyTorch visible GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"[GPU] cuda:{i} = {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("[GPU] Warning: CUDA is not available!")

_setup_gpu_early()
# ============================================================

import argparse
import json

# Add project path
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
    """Feature 1: Batch Image Summary Generation"""
    print("Executing Feature 1: Batch Image Summary Generation")
    
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
    
    print(f"Completed, total {len(results)} results")
    if args.output:
        print(f"Results saved to: {args.output}")


def cmd_feature2(args):
    """Feature 2: Five-Dimensional Scoring + improved_summary"""
    print("Executing Feature 2: Five-Dimensional Scoring + improved_summary")
    
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
            print(f"Results saved to: {args.output}")
    else:
        print("Evaluation failed")


def cmd_feature3(args):
    """Feature 3: Five-Dimensional Scoring Only"""
    print("Executing Feature 3: Five-Dimensional Scoring Only")
    
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
        print(f"Total Score: {total}/10")
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"Results saved to: {args.output}")
    else:
        print("Scoring failed")


def cmd_pipeline1(args):
    """Pipeline 1: Build Dataset"""
    print("Executing Pipeline 1: Build Dataset")
    
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
    
    print(f"Completed: Success {success}/{total}")


def cmd_pipeline2(args):
    """Pipeline 2: User Optimization"""
    print("Executing Pipeline 2: User Optimization")
    
    result = pipeline2_user_optimize(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_retries=args.max_retries,
    )
    
    if result:
        print("\nFinal Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"Results saved to: {args.output}")
    else:
        print("Optimization failed")


def cmd_pipeline3(args):
    """Pipeline 3: Direct Scoring"""
    print("Executing Pipeline 3: Direct Scoring")
    
    result = pipeline3_direct_score(
        image_path=args.image,
        summary=args.summary,
        model_path=args.model_path,
        lora_path=args.lora_path,
    )
    
    if result:
        print("\nScoring Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"Results saved to: {args.output}")
    else:
        print("Scoring failed")


def cmd_train(args):
    """Execute LoRA Fine-tuning"""
    from training.train_lora import train_lora
    from training.data_format import convert_to_training_format
    
    # Convert data format if needed
    if args.raw_data:
        print("Converting data format...")
        training_data_path = args.data_path.replace(".jsonl", "_training.json")
        convert_to_training_format(
            input_path=args.raw_data,
            output_path=training_data_path,
            include_improved_summary=(args.scheme == "l-1"),
        )
        args.data_path = training_data_path
    
    # Set output directory
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
        description="Figure Summary Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Specify GPU ID to use (e.g., 0, 1, 2...), sets CUDA_VISIBLE_DEVICES",
    )
    common_parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="Model path",
    )
    common_parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA weights path",
    )
    common_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    common_parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    common_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    
    # Feature 1
    f1_parser = subparsers.add_parser(
        "feature1",
        parents=[common_parser],
        help="Feature 1: Batch Image Summary Generation",
    )
    f1_parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Image folder path",
    )
    f1_parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.3,
        help="Low-quality ratio",
    )
    f1_parser.add_argument(
        "--medium_ratio",
        type=float,
        default=0.3,
        help="Medium-quality ratio",
    )
    f1_parser.add_argument(
        "--high_ratio",
        type=float,
        default=0.4,
        help="High-quality ratio",
    )
    f1_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from breakpoint (enabled by default)",
    )
    f1_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable breakpoint resume, start from scratch",
    )
    f1_parser.set_defaults(func=cmd_feature1)
    
    # Feature 2
    f2_parser = subparsers.add_parser(
        "feature2",
        parents=[common_parser],
        help="Feature 2: Five-Dimensional Scoring + improved_summary",
    )
    f2_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path",
    )
    f2_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Original summary",
    )
    f2_parser.set_defaults(func=cmd_feature2)
    
    # Feature 3
    f3_parser = subparsers.add_parser(
        "feature3",
        parents=[common_parser],
        help="Feature 3: Five-Dimensional Scoring Only",
    )
    f3_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path",
    )
    f3_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Summary to be evaluated",
    )
    f3_parser.set_defaults(func=cmd_feature3)
    
    # Pipeline 1
    p1_parser = subparsers.add_parser(
        "pipeline1",
        parents=[common_parser],
        help="Pipeline 1: Build Dataset",
    )
    p1_parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Image folder path",
    )
    p1_parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.3,
        help="Low-quality ratio",
    )
    p1_parser.add_argument(
        "--medium_ratio",
        type=float,
        default=0.3,
        help="Medium-quality ratio",
    )
    p1_parser.add_argument(
        "--high_ratio",
        type=float,
        default=0.4,
        help="High-quality ratio",
    )
    p1_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries",
    )
    p1_parser.add_argument(
        "--lora_path_f2",
        type=str,
        default=None,
        help="LoRA weights path for Feature 2 (Scoring + Improved Summary), e.g., ./lora_weights/l-1",
    )
    p1_parser.add_argument(
        "--lora_path_f3",
        type=str,
        default=None,
        help="LoRA weights path for Feature 3 (Scoring Only), e.g., ./lora_weights/l-2",
    )
    p1_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from breakpoint (enabled by default)",
    )
    p1_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable breakpoint resume, start from scratch",
    )
    p1_parser.set_defaults(func=cmd_pipeline1)
    
    # Pipeline 2
    p2_parser = subparsers.add_parser(
        "pipeline2",
        parents=[common_parser],
        help="Pipeline 2: User Optimization",
    )
    p2_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path",
    )
    p2_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Original summary",
    )
    p2_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries",
    )
    p2_parser.set_defaults(func=cmd_pipeline2)
    
    # Pipeline 3
    p3_parser = subparsers.add_parser(
        "pipeline3",
        parents=[common_parser],
        help="Pipeline 3: Direct Scoring",
    )
    p3_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path",
    )
    p3_parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Summary to be evaluated",
    )
    p3_parser.set_defaults(func=cmd_pipeline3)
    
    # Fine-tuning
    train_parser = subparsers.add_parser(
        "train",
        help="LoRA Fine-tuning",
    )
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="Base model path",
    )
    train_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Training data path (JSON format)",
    )
    train_parser.add_argument(
        "--raw_data",
        type=str,
        default=None,
        help="Raw JSONL data path (convert format first if provided)",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    train_parser.add_argument(
        "--scheme",
        type=str,
        choices=["l-1", "l-2"],
        default="l-1",
        help="Fine-tuning scheme",
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
        help="Learning rate",
    )
    train_parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    train_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    train_parser.add_argument(
        "--resume_lora_path",
        type=str,
        default=None,
        help="Existing LoRA weights path (for incremental fine-tuning, continue training on existing weights)",
    )
    train_parser.set_defaults(func=cmd_train)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == "__main__":
    main()