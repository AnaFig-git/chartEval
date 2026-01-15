"""
Feature 2: Five-dimensional Scoring + improved_summary Generation (API Version)
Input an image and original summary, output scores, explanations, weights, and the improved summary
"""
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional

from .api_model import get_model
from src.utils import parse_final_json



EVALUATE_PROMPT_WITH_IMPROVEMENT = '''You are a professional expert in figure-summary evaluation, skilled at conducting strict five-dimension evaluations based on multimodal input (image + text).

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


def evaluate_with_improvement(
    image_path: str,
    summary: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Feature 2: Evaluate Summary and Generate Improved Version (API Version)
    
    Args:
        image_path: Image path
        summary: Original summary
        temperature: Temperature parameter
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Dictionary containing scores, reasons, weights, and improved_summary
    """
    model = get_model()
    
    
    prompt = EVALUATE_PROMPT_WITH_IMPROVEMENT.format(summary=summary)
    
    
    output = model.generate(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    
    if not output:
        print(f"Warning: API returned empty result")
        return None
    
    
    result = parse_final_json(output)
    
    if result is None:
        print(f"Warning: Failed to parse evaluation result")
        print(f"Original output: {output[:500]}...")
        
    return result


def evaluate_with_improvement_retry(
    image_path: str,
    summary: str,
    max_retries: int = 3,
    base_temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Feature 2 with Retry Mechanism (API Version)
    
    Increase temperature for each retry to get different results
    
    Args:
        image_path: Image path
        summary: Original summary
        max_retries: Maximum number of retries
        base_temperature: Base temperature
        
    Returns:
        Evaluation result
    """
    temperatures = [base_temperature, base_temperature + 0.1, base_temperature + 0.2]
    
    for i in range(max_retries):
        temp = temperatures[i] if i < len(temperatures) else base_temperature + 0.1 * i
        print(f"Feature 2 attempt {i+1}/{max_retries} (temperature={temp:.2f})")
        
        result = evaluate_with_improvement(
            image_path=image_path,
            summary=summary,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
        )
        
        if result is not None:
            return result
            
    print("All retries for Feature 2 failed")
    return None