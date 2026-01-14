"""
工具函数 - JSON解析、分数验证、结果保存
"""
import json
import re
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


def parse_final_json(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中解析 <evaluation>...</evaluation> 和 <modification>...</modification> 标签内的 JSON
    
    Args:
        text: 模型输出文本
        
    Returns:
        解析后的字典，解析失败返回 None
    """
    result = {}
    
    # 预处理：移除 markdown 代码块标记
    # 处理 ```json<evaluation>... 这种格式
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # 匹配 <evaluation>...</evaluation> 标签
    eval_pattern = r'<evaluation>\s*([\s\S]*?)\s*</evaluation>'
    eval_match = re.search(eval_pattern, text, re.IGNORECASE)
    
    # 如果没有找到完整的标签，尝试只匹配 <evaluation> 开始标签后的 JSON
    if not eval_match:
        # 匹配 <evaluation> 后面直到遇到完整的 JSON 对象
        partial_pattern = r'<evaluation>\s*(\{[\s\S]*\})\s*(?:</evaluation>|`|$)'
        partial_match = re.search(partial_pattern, text, re.IGNORECASE)
        if partial_match:
            eval_match = partial_match
    
    if not eval_match:
        # 向后兼容：尝试匹配旧的 <final> 标签
        final_pattern = r'<final>\s*([\s\S]*?)\s*</final>'
        final_match = re.search(final_pattern, text, re.IGNORECASE)
        if final_match:
            json_str = final_match.group(1).strip()
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                try:
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败: {e}")
                    print(f"原始内容: {json_str[:500]}...")
                    return None
        
        # 尝试直接解析 JSON（某些模型直接返回 JSON 格式）
        # 查找包含 "scores" 的 JSON 对象
        json_pattern = r'\{[^{}]*"scores"\s*:\s*\{[^{}]*\}[^{}]*\}'
        json_match = re.search(json_pattern, text)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                if "scores" in result:
                    return result
            except json.JSONDecodeError:
                pass
        
        print("警告: 未找到 <evaluation> 或 <final> 标签")
        return None
    
    eval_json_str = eval_match.group(1).strip()
    
    try:
        # 尝试直接解析 evaluation
        result = json.loads(eval_json_str)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        try:
            eval_json_str = re.sub(r'\s+', ' ', eval_json_str)
            result = json.loads(eval_json_str)
        except json.JSONDecodeError as e:
            print(f"evaluation JSON 解析失败: {e}")
            print(f"原始内容: {eval_json_str[:500]}...")
            return None
    
    # 匹配可选的 <modification>...</modification> 标签
    mod_pattern = r'<modification>\s*([\s\S]*?)\s*</modification>'
    mod_match = re.search(mod_pattern, text, re.IGNORECASE)
    
    if mod_match:
        mod_json_str = mod_match.group(1).strip()
        try:
            mod_result = json.loads(mod_json_str)
            # 合并 modification 结果到主结果中
            if "improved_summary" in mod_result:
                result["improved_summary"] = mod_result["improved_summary"]
        except json.JSONDecodeError:
            try:
                mod_json_str = re.sub(r'\s+', ' ', mod_json_str)
                mod_result = json.loads(mod_json_str)
                if "improved_summary" in mod_result:
                    result["improved_summary"] = mod_result["improved_summary"]
            except json.JSONDecodeError as e:
                print(f"modification JSON 解析失败: {e}")
                print(f"原始内容: {mod_json_str[:500]}...")
                # 继续返回 evaluation 结果，即使 modification 解析失败
    
    return result


def validate_scores(result: Dict[str, Any]) -> Tuple[bool, int, str]:
    """
    验证评分结果是否符合要求
    
    规则:
    - 五个维度得分总和 >= 5
    - 任何一个维度得分不能为 0
    
    Args:
        result: 包含 scores 的字典
        
    Returns:
        (是否通过, 总分, 原因描述)
    """
    if result is None:
        return False, 0, "结果为空"
        
    scores = result.get("scores", {})
    
    required_dims = ["faithfulness", "completeness", "conciseness", "logicality", "analysis"]
    
    total_score = 0
    for dim in required_dims:
        score = scores.get(dim, 0)
        if not isinstance(score, (int, float)):
            return False, 0, f"维度 {dim} 的分数无效: {score}"
        if score == 0:
            return False, total_score, f"维度 {dim} 得分为 0"
        total_score += score
        
    if total_score < 5:
        return False, total_score, f"总分 {total_score} < 5"
        
    return True, total_score, "验证通过"


def get_total_score(result: Dict[str, Any]) -> int:
    """
    计算五个维度的总分
    
    Args:
        result: 包含 scores 的字典
        
    Returns:
        总分
    """
    if result is None:
        return 0
        
    scores = result.get("scores", {})
    total = 0
    for dim in ["faithfulness", "completeness", "conciseness", "logicality", "analysis"]:
        score = scores.get(dim, 0)
        if isinstance(score, (int, float)):
            total += score
    return total


def save_to_jsonl(data: Dict[str, Any], output_path: str, mode: str = "a"):
    """
    保存单条记录到 JSONL 文件
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
        mode: 写入模式 ('a' 追加, 'w' 覆盖)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载数据
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_image_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    获取文件夹中的所有图片文件
    
    Args:
        folder_path: 文件夹路径
        extensions: 支持的扩展名列表
        
    Returns:
        图片文件路径列表
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
        
    folder = Path(folder_path)
    image_files = []
    
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
        
    return sorted([str(f) for f in image_files])


def format_result_for_training(
    image_path: str,
    original_summary: str,
    result: Dict[str, Any],
    include_improved_summary: bool = True,
) -> Dict[str, Any]:
    """
    格式化结果为训练数据格式
    
    Args:
        image_path: 图片路径
        original_summary: 原始总结
        result: 评估结果
        include_improved_summary: 是否包含 improved_summary
        
    Returns:
        训练数据格式的字典
    """
    output = {
        "scores": result.get("scores", {}),
        "reasons": result.get("reasons", {}),
        "weights": result.get("weights", {}),
    }
    
    if include_improved_summary and "improved_summary" in result:
        output["improved_summary"] = result["improved_summary"]
        
    return {
        "image_path": image_path,
        "original_summary": original_summary,
        "output": output,
    }


def get_processed_images(jsonl_path: str) -> set:
    """
    从 JSONL 文件中获取已处理的图片路径集合
    
    Args:
        jsonl_path: JSONL 文件路径
        
    Returns:
        已处理的图片路径集合
    """
    processed = set()
    
    if not os.path.exists(jsonl_path):
        return processed
        
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # 支持多种格式：直接的 image_path 或嵌套的
                        if "image_path" in data:
                            processed.add(data["image_path"])
                        elif "image" in data:
                            processed.add(data["image"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取已处理记录时出错: {e}")
        
    return processed


def get_processed_images_from_step1(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从功能一的中间结果文件中获取已处理的数据
    
    Args:
        jsonl_path: 功能一中间结果 JSONL 文件路径
        
    Returns:
        {image_path: {quality_level, summary}} 的字典
    """
    processed = {}
    
    if not os.path.exists(jsonl_path):
        return processed
        
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "image_path" in data:
                            processed[data["image_path"]] = {
                                "quality_level": data.get("quality_level", "medium"),
                                "summary": data.get("summary", ""),
                            }
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取功能一中间结果时出错: {e}")
        
    return processed

