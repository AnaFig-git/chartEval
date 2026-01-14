"""
功能三：仅五维度评分（不生成 improved_summary）
输入图片和总结，输出评分、解释和权重
"""
from typing import Dict, Any, Optional

from .model_loader import get_model
from .utils import parse_final_json


# 功能三的评估提示词（不包含 improved_summary）
EVALUATE_PROMPT_SCORE_ONLY = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

Based on the input "figure image" and "original summary,"
you must evaluate the summary along the following five dimensions with a strict 0/1/2 scoring scheme, and provide reasons. And set weights (ranging from 0 to 1) for each dimension, and ensure that the sum of the weights for the five dimensions is 1.

---

[Scoring Dimension Definitions]
1. Faithfulness:  
   Ensure the summary strictly adheres to the figure's facts, including values, trends, proportions, and labels; no fabrication, distortion, or misinterpretation.  
   - 0: The summary severely deviates from the figure's facts, with errors or fabricated content.  
   - 1: Largely consistent with the figure, but with partial deviations or inaccurate details.  
   - 2: Fully faithful to the figure's facts, with all information correct.

2. Completeness:  
   Check whether the summary covers core elements, main trends, and key comparisons in the figure.  
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
   - 2: Accurate terminology, formal tone, and analytical depth.

---

[Original Summary]
{summary}

---

[Output Requirements]
1. You must output standardized JSON inside `<evaluation>` tags, with the following structure:

<evaluation>
{{"scores": {{"faithfulness": 0-2, "completeness": 0-2, "conciseness": 0-2, "logicality": 0-2, "analysis": 0-2}}, "reasons": {{"faithfulness": "brief explanation", "completeness": "brief explanation", "conciseness": "brief explanation", "logicality": "brief explanation", "analysis": "brief explanation"}}, "weights": {{"faithfulness": 0.35, "completeness": 0.25, "conciseness": 0.20, "logicality": 0.15, "analysis": 0.05}}}}
</evaluation>

2. The JSON must be valid, single-line JSON.  
3. The output format must include `<evaluation>` tags containing scores, reasons, and weights.'''


def score_only(
    image_path: str,
    summary: str,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    功能三：仅评分（不生成改进版本）
    
    Args:
        image_path: 图片路径
        summary: 待评估的总结
        model_path: 模型路径
        lora_path: LoRA 权重路径（可选）
        temperature: 温度参数
        top_p: top-p 采样参数
        do_sample: 是否采样
        
    Returns:
        包含 scores, reasons, weights 的字典
    """
    model = get_model(model_path=model_path, lora_path=lora_path)
    
    # 构建提示词
    prompt = EVALUATE_PROMPT_SCORE_ONLY.format(summary=summary)
    
    # 生成评估结果
    output = model.generate(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    
    # 解析结果
    result = parse_final_json(output)
    
    if result is None:
        print(f"警告: 无法解析评分结果")
        print(f"原始输出: {output[:500]}...")
        
    return result


def score_only_retry(
    image_path: str,
    summary: str,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    max_retries: int = 3,
    base_temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    带重试机制的功能三
    
    Args:
        image_path: 图片路径
        summary: 待评估的总结
        model_path: 模型路径
        lora_path: LoRA 权重路径
        max_retries: 最大重试次数
        base_temperature: 基础温度
        
    Returns:
        评分结果
    """
    temperatures = [base_temperature, base_temperature + 0.1, base_temperature + 0.2]
    
    for i in range(max_retries):
        temp = temperatures[i] if i < len(temperatures) else base_temperature + 0.1 * i
        print(f"功能三尝试 {i+1}/{max_retries} (temperature={temp:.2f})")
        
        result = score_only(
            image_path=image_path,
            summary=summary,
            model_path=model_path,
            lora_path=lora_path,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
        )
        
        if result is not None:
            return result
            
    print("功能三所有重试均失败")
    return None

