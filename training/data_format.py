"""
Data Format Conversion - Convert JSONL data to training format
"""
import json
from typing import List, Dict, Any
from pathlib import Path



PROMPT_TEMPLATE_WITH_IMPROVEMENT = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

Based on the input "figure image" and "original summary,"
you must evaluate the summary along the following five dimensions with a strict 0/1/2 scoring scheme, and provide reasons. Last Please generate an improved summary that addresses the weaknesses identified in the feedback, especially focusing on dimensions that scored below 2. Make sure to:
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
{{"scores": {{"faithfulness": 0-2, "completeness": 0-2, "conciseness": 0-2, "logicality": 0-2, "analysis": 0-2}}, "reasons": {{"faithfulness": "brief explanation", "completeness": "brief explanation", "conciseness": "brief explanation", "logicality": "brief explanation", "analysis": "brief explanation"}}}}
</evaluation>
<modification>
{{"improved_summary": "new summary"}}
</modification>

2. The JSON must be valid, single-line JSON.  
3. The output format must include `<evaluation>` tags containing scores, reasons, and weights, followed by `<modification>` tags containing the improved_summary.'''



PROMPT_TEMPLATE_SCORE_ONLY = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

Based on the input "figure image" and "original summary,"
you must evaluate the summary along the following five dimensions with a strict 0/1/2 scoring scheme, and provide reasons. 

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
{{"scores": {{"faithfulness": 0-2, "completeness": 0-2, "conciseness": 0-2, "logicality": 0-2, "analysis": 0-2}}, "reasons": {{"faithfulness": "brief explanation", "completeness": "brief explanation", "conciseness": "brief explanation", "logicality": "brief explanation", "analysis": "brief explanation"}}}}
</evaluation>

2. The JSON must be valid, single-line JSON.  
3. The output format must include `<evaluation>` tags containing scores, reasons, and weights.'''


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_output_json(output: Dict[str, Any], include_improved_summary: bool = True) -> str:
    """
    Format output into <evaluation>...</evaluation> format
    If include_improved_summary is True, also add the <modification>...</modification> section
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
    Convert JSONL data to training format
    
    Args:
        input_path: Input JSONL file path
        output_path: Output training data file path
        include_improved_summary: Whether to include improved_summary
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
        
        
        prompt = prompt_template.format(summary=original_summary)
        
        
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
        
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
        
    print(f"Conversion completed: {len(training_data)} data entries")
    print(f"Output file: {output_path}")


def generate_training_files(
    input_path: str,
    output_dir: str = "./data/output",
):
    """
    Generate two training data files at once
    
    Args:
        input_path: Input JSONL file path
        output_dir: Output directory path
        
    Generated files:
        - training_data_l1.json: Includes improved_summary (for l-1 scheme)
        - training_data_l2.json: Excludes improved_summary (for l-2 scheme)
    """
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    
    data = load_jsonl(input_path)
    print(f"Loaded data: {len(data)} records")
    
    
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
    print(f"Generated l-1 training data: {l1_output_path} ({len(l1_training_data)} entries)")
    
    
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
    print(f"Generated l-2 training data: {l2_output_path} ({len(l2_training_data)} entries)")
    
    print("\nTraining data generation completed!")
    print(f"  - l-1 scheme (includes improved_summary): {l1_output_path}")
    print(f"  - l-2 scheme (excludes improved_summary): {l2_output_path}")
    
    return str(l1_output_path), str(l2_output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Format Conversion")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", default=None, help="Output training data file path (single file mode)")
    parser.add_argument("--output-dir", default="./data/output", help="Output directory (batch mode)")
    parser.add_argument(
        "--no-improved-summary",
        action="store_true",
        help="Exclude improved_summary (for fine-tuning scheme two)",
    )
    parser.add_argument(
        "--generate-both",
        action="store_true",
        help="Generate two training data files at once (l-1 and l-2)",
    )
    
    args = parser.parse_args()
    
    if args.generate_both:
       
        generate_training_files(
            input_path=args.input,
            output_dir=args.output_dir,
        )
    else:
        
        if not args.output:
            parser.error("Single file mode requires specifying the --output parameter")
        convert_to_training_format(
            input_path=args.input,
            output_path=args.output,
            include_improved_summary=not args.no_improved_summary,
        )