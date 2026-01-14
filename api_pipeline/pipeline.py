"""
API 版链路实现 - 从 step1 结果开始构建数据集
跳过功能一（批量图片总结），直接执行功能二和功能三
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List, Tuple

from tqdm import tqdm

from .feature2_evaluate import evaluate_with_improvement
from .feature3_score import score_only
from src.utils import (
    validate_scores,
    get_total_score,
    save_to_jsonl,
    format_result_for_training,
    get_processed_images,
    load_jsonl,
)


def pipeline1_build_dataset_from_step1(
    input_path: str,
    output_path: str,
    max_retries: int = 3,
    resume: bool = True,
) -> Tuple[int, int]:
    """
    从 step1 结果构建数据集（API 版本）
    
    跳过功能一（批量图片总结），直接从 dataset_step1.jsonl 读取数据，
    执行功能二和功能三进行评估和验证。
    
    流程：
    1. 读取 step1 结果（image_path, quality_level, summary）
    2. 功能二：评估并生成改进版本
    3. 功能三：验证改进后的总结
    4. 验证不通过则重试（最多3次）
    5. 通过则保存，失败则丢弃
    
    Args:
        input_path: step1 结果文件路径（dataset_step1.jsonl）
        output_path: 输出 JSONL 文件路径
        max_retries: 最大重试次数
        resume: 是否从断点续传
        
    Returns:
        (成功数量, 总数量)
    """
    print("=" * 60)
    print("API 版链路一：从 step1 结果构建数据集")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取 step1 结果
    print(f"\n[读取数据] 从 {input_path} 加载 step1 结果...")
    
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在: {input_path}")
        return 0, 0
    
    summaries = load_jsonl(input_path)
    
    if not summaries:
        print("错误：没有读取到任何数据")
        return 0, 0
    
    print(f"读取到 {len(summaries)} 条记录")
    
    # 检查已完成的图片（断点续传）
    processed_images = set()
    if resume:
        processed_images = get_processed_images(output_path)
        if processed_images:
            print(f"\n发现 {len(processed_images)} 条已完成的最终结果，将跳过这些图片")
    
    # 过滤待处理的项目
    pending_summaries = [
        item for item in summaries 
        if item.get("image_path") not in processed_images
    ]
    
    total_count = len(summaries)
    already_done = len(processed_images)
    
    if not pending_summaries:
        print("所有图片已处理完成")
        return already_done, total_count
        
    print(f"\n[步骤2-3] 评估并验证...")
    print(f"待处理: {len(pending_summaries)} 张 | 已完成: {already_done} 张 | 总计: {total_count} 张")
    
    success_count = already_done
    fail_count = 0
    
    # 带进度条处理
    pbar = tqdm(
        pending_summaries,
        desc="评估验证中",
        unit="张",
        ncols=100,
    )
    
    for item in pbar:
        image_path = item.get("image_path", "")
        original_summary = item.get("summary", "")
        quality_level = item.get("quality_level", "medium")
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            tqdm.write(f"警告: 图片不存在，跳过: {image_path}")
            fail_count += 1
            continue
        
        # 重试逻辑
        temperatures = [0.7, 0.8, 0.9]
        passed = False
        
        for retry in range(max_retries):
            temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
            
            # 步骤2：功能二 - 评估并改进
            result2 = evaluate_with_improvement(
                image_path=image_path,
                summary=original_summary,
                temperature=temp,
                top_p=0.9,
                do_sample=True,
            )
            
            if result2 is None:
                continue
                
            improved_summary = result2.get("improved_summary", "")
            if not improved_summary:
                continue
                
            # 步骤3：功能三 - 验证改进后的总结
            result3 = score_only(
                image_path=image_path,
                summary=improved_summary,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            
            if result3 is None:
                continue
                
            # 验证分数
            is_valid, total_score, reason = validate_scores(result3)
            
            if is_valid:
                # 保存结果
                training_data = format_result_for_training(
                    image_path=image_path,
                    original_summary=original_summary,
                    result=result2,
                    include_improved_summary=True,
                )
                training_data["quality_level"] = quality_level
                training_data["validation_scores"] = result3.get("scores", {})
                
                save_to_jsonl(training_data, output_path)
                success_count += 1
                passed = True
                break
                
        if not passed:
            fail_count += 1
            
        # 更新进度条
        pbar.set_postfix({
            "成功": success_count,
            "失败": fail_count,
            "通过率": f"{success_count/(success_count+fail_count)*100:.1f}%" if (success_count+fail_count) > 0 else "N/A",
        })
    
    pbar.close()
    
    print("\n" + "=" * 60)
    print(f"API 版链路一完成:")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {fail_count}")
    print(f"  - 总计: {total_count}")
    print(f"  - 通过率: {success_count/total_count*100:.1f}%")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    return success_count, total_count


def pipeline2_user_optimize(
    image_path: str,
    summary: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    链路二：用户优化（API 版本）
    
    流程：
    1. 功能二：评估并生成改进版本
    2. 功能三：验证改进后的总结
    3. 验证不通过则重试（最多3次）
    4. 3次均不通过则选择总分最高的结果
    
    Args:
        image_path: 图片路径
        summary: 用户提供的原始总结
        max_retries: 最大重试次数
        
    Returns:
        最终结果（包含 scores, reasons, weights, improved_summary）
    """
    print("=" * 60)
    print("API 版链路二：用户优化")
    print("=" * 60)
    
    temperatures = [0.7, 0.8, 0.9]
    all_results = []  # 保存所有尝试的结果和分数
    
    pbar = tqdm(range(max_retries), desc="优化尝试", unit="次", ncols=100)
    
    for retry in pbar:
        temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
        pbar.set_postfix({"temperature": f"{temp:.2f}"})
        
        # 步骤1：功能二 - 评估并改进
        result2 = evaluate_with_improvement(
            image_path=image_path,
            summary=summary,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
        )
        
        if result2 is None:
            tqdm.write(f"  尝试 {retry+1}: 功能二失败")
            continue
            
        improved_summary = result2.get("improved_summary", "")
        if not improved_summary:
            tqdm.write(f"  尝试 {retry+1}: 未生成 improved_summary")
            continue
            
        # 步骤2：功能三 - 验证改进后的总结
        result3 = score_only(
            image_path=image_path,
            summary=improved_summary,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        if result3 is None:
            tqdm.write(f"  尝试 {retry+1}: 功能三失败")
            all_results.append((result2, 0))
            continue
            
        # 验证分数
        is_valid, total_score, reason = validate_scores(result3)
        all_results.append((result2, total_score))
        
        if is_valid:
            pbar.close()
            print(f"\n验证通过！总分: {total_score}")
            print("\n" + "=" * 60)
            print("API 版链路二完成")
            print("=" * 60)
            return result2
        else:
            tqdm.write(f"  尝试 {retry+1}: 验证失败 - {reason} (总分: {total_score})")
    
    pbar.close()
            
    # 所有尝试均未通过，选择总分最高的结果
    if all_results:
        best_result, best_score = max(all_results, key=lambda x: x[1])
        print(f"\n所有尝试均未通过验证，选择总分最高的结果 (分数: {best_score})")
        print("\n" + "=" * 60)
        print("API 版链路二完成（选择最佳结果）")
        print("=" * 60)
        return best_result
    else:
        print("\n所有尝试均失败")
        print("\n" + "=" * 60)
        print("API 版链路二失败")
        print("=" * 60)
        return None


def pipeline3_direct_score(
    image_path: str,
    summary: str,
) -> Optional[Dict[str, Any]]:
    """
    链路三：直接评分（API 版本）
    
    流程：
    1. 功能三：评分并直接输出
    
    Args:
        image_path: 图片路径
        summary: 用户提供的总结
        
    Returns:
        评分结果（包含 scores, reasons, weights）
    """
    print("=" * 60)
    print("API 版链路三：直接评分")
    print("=" * 60)
    
    result = score_only(
        image_path=image_path,
        summary=summary,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    if result is not None:
        total_score = get_total_score(result)
        print(f"\n评分完成，总分: {total_score}")
        print(f"各维度分数: {result.get('scores', {})}")
    else:
        print("\n评分失败")
        
    print("\n" + "=" * 60)
    print("API 版链路三完成")
    print("=" * 60)
    
    return result

