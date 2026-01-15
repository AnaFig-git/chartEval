#!/usr/bin/env python3
"""
API Pipeline - Implement Pipeline 1 via API Calls (Skip Step 1)

Usage:
    # Pipeline 1: Build dataset from step1 results
    python api_pipeline/main.py pipeline1 --input ./data/output/dataset_step1.jsonl --output ./data/output/dataset_api.jsonl

    # Pipeline 2: User Optimization (Single Image)
    python api_pipeline/main.py pipeline2 --image ./data/sample.png --summary "Original summary content"

    # Pipeline 3: Direct Scoring (Single Image)
    python api_pipeline/main.py pipeline3 --image ./data/sample.png --summary "Summary content to be scored"
"""
import sys
import os
import argparse
import json

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_pipeline.pipeline import (
    pipeline1_build_dataset_from_step1,
    pipeline2_user_optimize,
    pipeline3_direct_score,
)
from src.utils import save_to_jsonl, get_total_score


def cmd_pipeline1(args):
    """Pipeline 1: Build dataset from step1 results"""
    print("Executing API Version Pipeline 1: Build dataset from step1 results")
    
    success, total = pipeline1_build_dataset_from_step1(
        input_path=args.input,
        output_path=args.output,
        max_retries=args.max_retries,
        resume=args.resume,
    )
    
    print(f"Completed: Success {success}/{total}")


def cmd_pipeline2(args):
    """Pipeline 2: User Optimization"""
    print("Executing API Version Pipeline 2: User Optimization")
    
    result = pipeline2_user_optimize(
        image_path=args.image,
        summary=args.summary,
        max_retries=args.max_retries,
    )
    
    if result:
        print("\nFinal Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"Result saved to: {args.output}")
    else:
        print("Optimization failed")


def cmd_pipeline3(args):
    """Pipeline 3: Direct Scoring"""
    print("Executing API Version Pipeline 3: Direct Scoring")
    
    result = pipeline3_direct_score(
        image_path=args.image,
        summary=args.summary,
    )
    
    if result:
        print("\nScoring Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        total = get_total_score(result)
        print(f"Total Score: {total}/10")
        if args.output:
            save_to_jsonl(result, args.output)
            print(f"Result saved to: {args.output}")
    else:
        print("Scoring failed")


def main():
    parser = argparse.ArgumentParser(
        description="API Pipeline - Implement pipeline functions via API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pipeline 1
    p1_parser = subparsers.add_parser(
        "pipeline1",
        help="Pipeline 1: Build dataset from step1 results",
    )
    p1_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to step1 result file (dataset_step1.jsonl)",
    )
    p1_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    p1_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries (default: 3)",
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
        help="Pipeline 2: User Optimization (Single Image)",
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
        help="Maximum number of retries (default: 3)",
    )
    p2_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)",
    )
    p2_parser.set_defaults(func=cmd_pipeline2)
    
    # Pipeline 3
    p3_parser = subparsers.add_parser(
        "pipeline3",
        help="Pipeline 3: Direct Scoring (Single Image)",
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
    p3_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional)",
    )
    p3_parser.set_defaults(func=cmd_pipeline3)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == "__main__":
    main()