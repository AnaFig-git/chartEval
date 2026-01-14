"""
数据格式转换 - 将 JSONL 数据转换为训练格式
"""
import json
from typing import List, Dict, Any
from pathlib import Path


# 功能二的提示词模板（包含 improved_summary）
PROMPT_TEMPLATE_WITH_IMPROVEMENT = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

Based on the input "figure image" and "original summary,"
you must evaluate the summary along the following five dimensions with a strict 0/1/2 scoring scheme, and provide reasons. And set weights (ranging from 0 to 1) for each dimension, and ensure that the sum of the weights for the five dimensions is 1. Last Please generate an improved summary that addresses the weaknesses identified in the feedback, especially focusing on dimensions that scored below 2. Make sure to:
- Fix any factual inaccuracies (Faithfulness)
- Add missing key information (Comprehensiveness)
- Ensure concise expression (Conciseness)
- Maintain logical flow (Logical Consistency)
- Use professional terminology (Professional Relevance)

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
<modification>
{{"improved_summary": "new summary"}}
</modification>

2. The JSON must be valid, single-line JSON.  
3. The output format must include `<evaluation>` tags containing scores, reasons, and weights, followed by `<modification>` tags containing the improved_summary.'''


# 功能三的提示词模板（不包含 improved_summary）
PROMPT_TEMPLATE_SCORE_ONLY = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

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


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_output_json(output: Dict[str, Any], include_improved_summary: bool = True) -> str:
    """
    格式化输出为 <evaluation>...</evaluation> 格式
    如果 include_improved_summary 为 True，还会添加 <modification>...</modification> 部分
    """
    evaluation_result = {
        "scores": output.get("scores", {}),
        "reasons": output.get("reasons", {}),
        "weights": output.get("weights", {}),
    }
    
    if include_improved_summary:
        modification_result = {
            "improved_summary": output.get("improved_summary", ""),
        }
        return f"<evaluation>\n{json.dumps(evaluation_result, ensure_ascii=False)}\n</evaluation>\n<modification>\n{json.dumps(modification_result, ensure_ascii=False)}\n</modification>"
    else:
        return f"<evaluation>\n{json.dumps(evaluation_result, ensure_ascii=False)}\n</evaluation>"


def convert_to_training_format(
    input_path: str,
    output_path: str,
    include_improved_summary: bool = True,
):
    """
    将 JSONL 数据转换为训练格式
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出训练数据文件路径
        include_improved_summary: 是否包含 improved_summary
    """
    data = load_jsonl(input_path)
    
    training_data = []
    
    prompt_template = (
        PROMPT_TEMPLATE_WITH_IMPROVEMENT 
        if include_improved_summary 
        else PROMPT_TEMPLATE_SCORE_ONLY
    )
    
    for item in data:
        image_path = item.get("image_path", "")
        original_summary = item.get("original_summary", "")
        output = item.get("output", {})
        
        # 构建 prompt
        prompt = prompt_template.format(summary=original_summary)
        
        # 构建 response
        response = format_output_json(output, include_improved_summary)
        
        training_item = {
            "image": image_path,
            "conversations": [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": response,
                },
            ],
        }
        
        training_data.append(training_item)
        
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
        
    print(f"转换完成: {len(training_data)} 条数据")
    print(f"输出文件: {output_path}")


def generate_training_files(
    input_path: str,
    output_dir: str = "./data/output",
):
    """
    一次性生成两个训练数据文件
    
    Args:
        input_path: 输入 JSONL 文件路径
        output_dir: 输出目录路径
        
    生成文件:
        - training_data_l1.json: 包含 improved_summary (用于 l-1 方案)
        - training_data_l2.json: 不包含 improved_summary (用于 l-2 方案)
    """
    # 确保输出目录存在
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 加载数据（只加载一次）
    data = load_jsonl(input_path)
    print(f"加载数据: {len(data)} 条记录")
    
    # 生成 l-1 训练数据 (包含 improved_summary)
    l1_output_path = output_dir_path / "training_data_l1.json"
    l1_training_data = []
    
    for item in data:
        image_path = item.get("image_path", "")
        original_summary = item.get("original_summary", "")
        output = item.get("output", {})
        
        prompt = PROMPT_TEMPLATE_WITH_IMPROVEMENT.format(summary=original_summary)
        response = format_output_json(output, include_improved_summary=True)
        
        l1_training_data.append({
            "image": image_path,
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        })
    
    with open(l1_output_path, "w", encoding="utf-8") as f:
        json.dump(l1_training_data, f, ensure_ascii=False, indent=2)
    print(f"生成 l-1 训练数据: {l1_output_path} ({len(l1_training_data)} 条)")
    
    # 生成 l-2 训练数据 (不包含 improved_summary)
    l2_output_path = output_dir_path / "training_data_l2.json"
    l2_training_data = []
    
    for item in data:
        image_path = item.get("image_path", "")
        original_summary = item.get("original_summary", "")
        output = item.get("output", {})
        
        prompt = PROMPT_TEMPLATE_SCORE_ONLY.format(summary=original_summary)
        response = format_output_json(output, include_improved_summary=False)
        
        l2_training_data.append({
            "image": image_path,
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        })
    
    with open(l2_output_path, "w", encoding="utf-8") as f:
        json.dump(l2_training_data, f, ensure_ascii=False, indent=2)
    print(f"生成 l-2 训练数据: {l2_output_path} ({len(l2_training_data)} 条)")
    
    print("\n训练数据生成完成!")
    print(f"  - l-1 方案 (含 improved_summary): {l1_output_path}")
    print(f"  - l-2 方案 (不含 improved_summary): {l2_output_path}")
    
    return str(l1_output_path), str(l2_output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据格式转换")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", default=None, help="输出训练数据文件路径（单文件模式）")
    parser.add_argument("--output-dir", default="./data/output", help="输出目录（批量模式）")
    parser.add_argument(
        "--no-improved-summary",
        action="store_true",
        help="不包含 improved_summary（用于微调方案二）",
    )
    parser.add_argument(
        "--generate-both",
        action="store_true",
        help="一次性生成两个训练数据文件（l-1 和 l-2）",
    )
    
    args = parser.parse_args()
    
    if args.generate_both:
        # 批量生成两个文件
        generate_training_files(
            input_path=args.input,
            output_dir=args.output_dir,
        )
    else:
        # 单文件模式
        if not args.output:
            parser.error("单文件模式需要指定 --output 参数")
        convert_to_training_format(
            input_path=args.input,
            output_path=args.output,
            include_improved_summary=not args.no_improved_summary,
        )

