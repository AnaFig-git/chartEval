"""
Function 1: Batch Image Summary Generation
Input a batch of images and generate a summary for each image, with adjustable ratios for low/medium/high quality levels
Supports resuming from breakpoints and real-time saving
"""
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from tqdm import tqdm

from .model_loader import get_model
from .utils import get_image_files, save_to_jsonl, get_processed_images_from_step1


# Five-dimensional scoring criteria (used in prompts)
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

# Prompts for different quality levels
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
    Distribute quality levels for each image according to specified ratios
    
    Args:
        num_images: Number of images
        low_ratio: Ratio of low quality
        medium_ratio: Ratio of medium quality
        high_ratio: Ratio of high quality
        seed: Random seed (ensures consistent distribution when resuming from breakpoints)
        
    Returns:
        List of quality levels
    """
    # Normalize ratios
    total = low_ratio + medium_ratio + high_ratio
    low_ratio /= total
    medium_ratio /= total
    high_ratio /= total
    
    # Calculate number of images for each level
    num_low = int(num_images * low_ratio)
    num_medium = int(num_images * medium_ratio)
    num_high = num_images - num_low - num_medium
    
    # Generate level list and shuffle randomly (fixed seed for consistency)
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
    Generate a summary of specified quality level for a single image
    
    Args:
        image_path: Path to image
        quality_level: Quality level (low/medium/high)
        model_path: Path to model
        lora_path: Path to LoRA weights
        temperature: Temperature parameter
        top_p: Top-p sampling parameter
        
    Returns:
        Generated summary
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
    Batch generate image summaries (supports resuming from breakpoints and real-time saving)
    
    Args:
        image_folder: Path to image folder
        low_ratio: Ratio of low quality
        medium_ratio: Ratio of medium quality
        high_ratio: Ratio of high quality
        model_path: Path to model
        lora_path: Path to LoRA weights
        temperature: Temperature parameter
        top_p: Top-p sampling parameter
        output_path: Path to save intermediate results (for resuming from breakpoints)
        resume: Whether to resume from breakpoints
        
    Returns:
        List of generated results, each containing {image_path, quality_level, summary}
    """
    # Get all images
    image_paths = get_image_files(image_folder)
    
    if not image_paths:
        print(f"Warning: No images found in folder {image_folder}")
        return []
        
    print(f"Found {len(image_paths)} images")
    
    # Distribute quality levels (fixed seed for consistency when resuming)
    quality_levels = distribute_quality_levels(
        len(image_paths), low_ratio, medium_ratio, high_ratio, seed=42
    )
    
    # Count levels
    level_counts = {"low": 0, "medium": 0, "high": 0}
    for level in quality_levels:
        level_counts[level] += 1
    print(f"Quality distribution: Low={level_counts['low']}, Medium={level_counts['medium']}, High={level_counts['high']}")
    
    # Check processed images (resume from breakpoints)
    processed_data = {}
    if resume and output_path:
        processed_data = get_processed_images_from_step1(output_path)
        if processed_data:
            print(f"Found {len(processed_data)} processed images, these will be skipped")
    
    # Prepare pending items
    results = []
    pending_items = []
    
    for image_path, quality_level in zip(image_paths, quality_levels):
        if image_path in processed_data:
            # Add processed items directly to results
            results.append({
                "image_path": image_path,
                "quality_level": processed_data[image_path]["quality_level"],
                "summary": processed_data[image_path]["summary"],
            })
        else:
            pending_items.append((image_path, quality_level))
    
    if not pending_items:
        print("All images have been processed")
        return results
        
    print(f"Pending processing: {len(pending_items)} images")
    
    # Generate summaries (with progress bar)
    model = get_model(model_path=model_path, lora_path=lora_path)
    
    success_count = 0
    error_count = 0
    
    pbar = tqdm(
        pending_items,
        desc="Function 1: Generating summaries",
        unit="image",
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
            
            # Save to file in real-time
            if output_path:
                save_to_jsonl(result, output_path)
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {Path(image_path).name}: {e}")
            continue
            
        # Update progress bar
        pbar.set_postfix({
            "Success": success_count,
            "Failed": error_count,
            "Quality": quality_level,
        })
    
    pbar.close()
    print(f"\nFunction 1 completed: Success {success_count}, Failed {error_count}, Total {len(results)}")
    
    return results