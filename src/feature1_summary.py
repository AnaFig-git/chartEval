"""
功能一：批量图片总结生成
输入一批图片，为每个图片输出一段总结，可调整低/中/高质量比例
支持断点续传和实时保存
"""
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from tqdm import tqdm

from .model_loader import get_model
from .utils import get_image_files, save_to_jsonl, get_processed_images_from_step1


# 五维度评分标准（用于提示词中）
SCORING_DIMENSIONS = """[Scoring Dimension Definitions]
1. Faithfulness:  
   Ensure the summary strictly adheres to the chart's facts, including values, trends, proportions, and labels; no fabrication, distortion, or misinterpretation.  
   - 0: The summary severely deviates from the chart's facts, with errors or fabricated content.  
   - 1: Largely consistent with the chart, but with partial deviations or inaccurate details.  
   - 2: Fully faithful to the chart's facts, with all information correct.

2. Completeness:  
   Check whether the summary covers core elements, main trends, and key comparisons in the chart.  
   - 0: Misses major content or key information.  
   - 1: Covers the main information but with secondary omissions.  
   - 2: Fully covers all key information.

3. Conciseness:  
   Check whether the summary is compact, information-dense, and free of redundancy.  
   - 0: Redundant, or overly brief to the point of missing key information.  
   - 1: Generally concise but can be further optimized.  
   - 2: Tersely written with efficient expression.

4. Logicality:  
   Check whether the logic is coherent, sequence is reasonable, and causal relations are clear.  
   - 0: Logical confusion or self-contradiction.  
   - 1: Basically coherent but with slight jumps or ambiguity.  
   - 2: Clear logic and strong consistency.

5. Analysis:  
   Check whether the summary meets domain-standard expression (e.g., finance, statistics, economics).  
   - 0: Lacks professionalism, or uses incorrect terminology/units.  
   - 1: Professional expression is insufficient or slightly colloquial.  
   - 2: Accurate terminology, formal tone, and analytical depth."""

# 不同质量等级的提示词
QUALITY_PROMPTS = {
    "low": f"""Analyze this figure and generate a summary that would score LOW quality (total score ≤ 5 out of 10) based on the following scoring dimensions:

{SCORING_DIMENSIONS}

---

Generate a very brief, incomplete summary that would achieve a total score of no more than 5.
The summary should be overly short and miss important details, specific values, or key trends.
Keep it to 1-2 short sentences only. Do NOT be comprehensive.""",

    "medium": f"""Analyze this figure and generate a summary that would score MEDIUM quality (total score between 6 and 7 out of 10) based on the following scoring dimensions:

{SCORING_DIMENSIONS}

---

Generate a basic summary that would achieve a total score between 6 and 7.
The summary should cover the main idea but lack specific details and professional terminology.
Mention the general trend but omit precise values and deeper insights.
Keep it moderate in length (2-3 sentences).""",

    "high": f"""Analyze this figure and generate a summary that would score HIGH quality (total score 8 or above out of 10) based on the following scoring dimensions:

{SCORING_DIMENSIONS}

---

Generate a decent summary that would achieve a total score of 8 or above.
Include key data points, trends, and use professional terminology.
Make it relatively complete with good coverage of the main information (3-4 sentences).""",
}


def distribute_quality_levels(
    num_images: int,
    low_ratio: float = 0.3,
    medium_ratio: float = 0.3,
    high_ratio: float = 0.4,
    seed: int = 42,
) -> List[str]:
    """
    根据比例分配每张图片的质量等级
    
    Args:
        num_images: 图片数量
        low_ratio: 低质量比例
        medium_ratio: 中质量比例
        high_ratio: 高质量比例
        seed: 随机种子（保证断点续传时分配一致）
        
    Returns:
        质量等级列表
    """
    # 归一化比例
    total = low_ratio + medium_ratio + high_ratio
    low_ratio /= total
    medium_ratio /= total
    high_ratio /= total
    
    # 计算每个等级的数量
    num_low = int(num_images * low_ratio)
    num_medium = int(num_images * medium_ratio)
    num_high = num_images - num_low - num_medium
    
    # 生成等级列表并随机打乱（使用固定种子保证一致性）
    levels = ["low"] * num_low + ["medium"] * num_medium + ["high"] * num_high
    random.seed(seed)
    random.shuffle(levels)
    
    return levels


def generate_summary(
    image_path: str,
    quality_level: str,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    为单张图片生成指定质量等级的总结
    
    Args:
        image_path: 图片路径
        quality_level: 质量等级 (low/medium/high)
        model_path: 模型路径
        lora_path: LoRA 权重路径
        temperature: 温度参数
        top_p: top-p 采样参数
        
    Returns:
        生成的总结
    """
    model = get_model(model_path=model_path, lora_path=lora_path)
    prompt = QUALITY_PROMPTS.get(quality_level, QUALITY_PROMPTS["medium"])
    
    summary = model.generate(
        image_path=image_path,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    
    return summary


def batch_generate_summaries(
    image_folder: str,
    low_ratio: float = 0.3,
    medium_ratio: float = 0.3,
    high_ratio: float = 0.4,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    output_path: Optional[str] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量生成图片总结（支持断点续传和实时保存）
    
    Args:
        image_folder: 图片文件夹路径
        low_ratio: 低质量比例
        medium_ratio: 中质量比例
        high_ratio: 高质量比例
        model_path: 模型路径
        lora_path: LoRA 权重路径
        temperature: 温度参数
        top_p: top-p 采样参数
        output_path: 中间结果保存路径（用于断点续传）
        resume: 是否从断点续传
        
    Returns:
        生成结果列表，每项包含 {image_path, quality_level, summary}
    """
    # 获取所有图片
    image_paths = get_image_files(image_folder)
    
    if not image_paths:
        print(f"警告: 文件夹 {image_folder} 中没有找到图片")
        return []
        
    print(f"找到 {len(image_paths)} 张图片")
    
    # 分配质量等级（使用固定种子保证断点续传时一致）
    quality_levels = distribute_quality_levels(
        len(image_paths), low_ratio, medium_ratio, high_ratio, seed=42
    )
    
    # 统计各等级数量
    level_counts = {"low": 0, "medium": 0, "high": 0}
    for level in quality_levels:
        level_counts[level] += 1
    print(f"质量分布: 低={level_counts['low']}, 中={level_counts['medium']}, 高={level_counts['high']}")
    
    # 检查已处理的图片（断点续传）
    processed_data = {}
    if resume and output_path:
        processed_data = get_processed_images_from_step1(output_path)
        if processed_data:
            print(f"发现 {len(processed_data)} 张已处理的图片，将跳过这些图片")
    
    # 准备待处理的图片
    results = []
    pending_items = []
    
    for image_path, quality_level in zip(image_paths, quality_levels):
        if image_path in processed_data:
            # 已处理的直接加入结果
            results.append({
                "image_path": image_path,
                "quality_level": processed_data[image_path]["quality_level"],
                "summary": processed_data[image_path]["summary"],
            })
        else:
            pending_items.append((image_path, quality_level))
    
    if not pending_items:
        print("所有图片已处理完成")
        return results
        
    print(f"待处理: {len(pending_items)} 张图片")
    
    # 生成总结（带进度条）
    model = get_model(model_path=model_path, lora_path=lora_path)
    
    success_count = 0
    error_count = 0
    
    pbar = tqdm(
        pending_items,
        desc="功能一：生成总结",
        unit="张",
        ncols=100,
    )
    
    for image_path, quality_level in pbar:
        prompt = QUALITY_PROMPTS[quality_level]
        
        try:
            summary = model.generate(
                image_path=image_path,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
            
            result = {
                "image_path": image_path,
                "quality_level": quality_level,
                "summary": summary,
            }
            
            results.append(result)
            success_count += 1
            
            # 实时保存到文件
            if output_path:
                save_to_jsonl(result, output_path)
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"错误处理 {Path(image_path).name}: {e}")
            continue
            
        # 更新进度条描述
        pbar.set_postfix({
            "成功": success_count,
            "失败": error_count,
            "质量": quality_level,
        })
    
    pbar.close()
    print(f"\n功能一完成: 成功 {success_count}, 失败 {error_count}, 总计 {len(results)}")
    
    return results
