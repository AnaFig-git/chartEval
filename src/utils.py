"""
Utility Functions - JSON Parsing, Score Validation, Result Saving
"""
import json
import re
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


def parse_final_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON inside <evaluation>...</evaluation> and <modification>...</modification> tags from model output
    
    Args:
        text: Model output text
        
    Returns:
        Parsed dictionary, returns None if parsing fails
    """
    result = {}
    
    # Preprocessing: Remove markdown code block markers
    # Handle formats like ```json<evaluation>...
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Match <evaluation>...</evaluation> tags
    eval_pattern = r'<evaluation>\s*([\s\S]*?)\s*</evaluation>'
    eval_match = re.search(eval_pattern, text, re.IGNORECASE)
    
    # If complete tags not found, try matching JSON after <evaluation> start tag only
    if not eval_match:
        # Match after <evaluation> until a complete JSON object is found
        partial_pattern = r'<evaluation>\s*(\{[\s\S]*\})\s*(?:</evaluation>|`|$)'
        partial_match = re.search(partial_pattern, text, re.IGNORECASE)
        if partial_match:
            eval_match = partial_match
    
    if not eval_match:
        # Backward compatibility: Try matching old <final> tags
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
                    print(f"JSON parsing failed: {e}")
                    print(f"Raw content: {json_str[:500]}...")
                    return None
        
        # Try parsing JSON directly (some models return JSON format directly)
        # Find JSON object containing "scores"
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
        
        print("Warning: <evaluation> or <final> tags not found")
        return None
    
    eval_json_str = eval_match.group(1).strip()
    
    try:
        # Try parsing evaluation directly
        result = json.loads(eval_json_str)
    except json.JSONDecodeError:
        # Try fixing common issues
        try:
            eval_json_str = re.sub(r'\s+', ' ', eval_json_str)
            result = json.loads(eval_json_str)
        except json.JSONDecodeError as e:
            print(f"evaluation JSON parsing failed: {e}")
            print(f"Raw content: {eval_json_str[:500]}...")
            return None
    
    # Match optional <modification>...</modification> tags
    mod_pattern = r'<modification>\s*([\s\S]*?)\s*</modification>'
    mod_match = re.search(mod_pattern, text, re.IGNORECASE)
    
    if mod_match:
        mod_json_str = mod_match.group(1).strip()
        try:
            mod_result = json.loads(mod_json_str)
            # Merge modification results into main result
            if "improved_summary" in mod_result:
                result["improved_summary"] = mod_result["improved_summary"]
        except json.JSONDecodeError:
            try:
                mod_json_str = re.sub(r'\s+', ' ', mod_json_str)
                mod_result = json.loads(mod_json_str)
                if "improved_summary" in mod_result:
                    result["improved_summary"] = mod_result["improved_summary"]
            except json.JSONDecodeError as e:
                print(f"modification JSON parsing failed: {e}")
                print(f"Raw content: {mod_json_str[:500]}...")
                # Continue returning evaluation result even if modification parsing fails
    
    return result


def validate_scores(result: Dict[str, Any]) -> Tuple[bool, int, str]:
    """
    Validate if scoring results meet requirements
    
    Rules:
    - Total score of five dimensions >= 5
    - No dimension score can be 0
    
    Args:
        result: Dictionary containing scores
        
    Returns:
        (whether passed, total score, reason description)
    """
    if result is None:
        return False, 0, "Result is empty"
        
    scores = result.get("scores", {})
    
    required_dims = ["faithfulness", "completeness", "conciseness", "logicality", "analysis"]
    
    total_score = 0
    for dim in required_dims:
        score = scores.get(dim, 0)
        if not isinstance(score, (int, float)):
            return False, 0, f"Invalid score for dimension {dim}: {score}"
        if score == 0:
            return False, total_score, f"Score for dimension {dim} is 0"
        total_score += score
        
    if total_score < 5:
        return False, total_score, f"Total score {total_score} < 5"
        
    return True, total_score, "Validation passed"


def get_total_score(result: Dict[str, Any]) -> int:
    """
    Calculate total score of five dimensions
    
    Args:
        result: Dictionary containing scores
        
    Returns:
        Total score
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
    Save a single record to JSONL file
    
    Args:
        data: Data to save
        output_path: Output file path
        mode: Write mode ('a' for append, 'w' for overwrite)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file
    
    Args:
        input_path: Input file path
        
    Returns:
        List of data
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
    Get all image files in a folder
    
    Args:
        folder_path: Folder path
        extensions: List of supported extensions
        
    Returns:
        List of image file paths
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
    Format results into training data format
    
    Args:
        image_path: Image path
        original_summary: Original summary
        result: Evaluation result
        include_improved_summary: Whether to include improved_summary
        
    Returns:
        Dictionary in training data format
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
    Get set of processed image paths from JSONL file
    
    Args:
        jsonl_path: JSONL file path
        
    Returns:
        Set of processed image paths
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
                        # Support multiple formats: direct image_path or nested
                        if "image_path" in data:
                            processed.add(data["image_path"])
                        elif "image" in data:
                            processed.add(data["image"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading processed records: {e}")
        
    return processed


def get_processed_images_from_step1(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Get processed data from intermediate result file of Feature 1
    
    Args:
        jsonl_path: JSONL file path of Feature 1 intermediate results
        
    Returns:
        Dictionary of {image_path: {quality_level, summary}}
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
        print(f"Error reading Feature 1 intermediate results: {e}")
        
    return processed