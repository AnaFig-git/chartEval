"""
Implementation of Three Complete Pipelines
- Pipeline 1: Build Dataset (supports resuming from breakpoints, progress bar, real-time saving)
- Pipeline 2: User Optimization
- Pipeline 3: Direct Scoring
"""
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from tqdm import tqdm

from .feature1_summary import batch_generate_summaries
from .feature2_evaluate import evaluate_with_improvement
from .feature3_score import score_only
from .utils import (
    validate_scores,
    get_total_score,
    save_to_jsonl,
    format_result_for_training,
    get_processed_images,
)


def pipeline1_build_dataset(
    image_folder: str,
    output_path: str,
    low_ratio: float = 0.3,
    medium_ratio: float = 0.3,
    high_ratio: float = 0.4,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    lora_path_feature2: Optional[str] = None,
    lora_path_feature3: Optional[str] = None,
    max_retries: int = 3,
    resume: bool = True,
) -> Tuple[int, int]:
    """
    Pipeline 1: Build Dataset (supports resuming from breakpoints)
    
    Workflow:
    1. Feature 1: Batch generate summaries
    2. Feature 2: Evaluate and generate improved version
    3. Feature 3: Validate the improved summary
    4. Retry if validation fails (max 3 times)
    5. Save if passed, discard if failed
    
    Args:
        image_folder: Path to image folder
        output_path: Path to output JSONL file
        low_ratio: Ratio of low quality
        medium_ratio: Ratio of medium quality
        high_ratio: Ratio of high quality
        model_path: Path to model
        lora_path: Path to LoRA weights (deprecated, use lora_path_feature2/feature3 instead)
        lora_path_feature2: Path to LoRA weights for Feature 2 (scoring + improved summary)
        lora_path_feature3: Path to LoRA weights for Feature 3 (scoring only)
        max_retries: Maximum number of retries
        resume: Whether to resume from breakpoints
        
    Returns:
        (number of successes, total number)
    """
    print("=" * 60)
    print("Pipeline 1: Building Dataset")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Intermediate result path for Feature 1
    step1_output = output_path.replace(".jsonl", "_step1.jsonl")
    
    # Step 1: Feature 1 - Batch generate summaries
    print("\n[Step 1] Feature 1: Batch generating summaries...")
    summaries = batch_generate_summaries(
        image_folder=image_folder,
        low_ratio=low_ratio,
        medium_ratio=medium_ratio,
        high_ratio=high_ratio,
        model_path=model_path,
        lora_path=None,  # Feature 1 does not use LoRA
        output_path=step1_output,  # Save intermediate results
        resume=resume,
    )
    
    if not summaries:
        print("Error: No summaries were generated")
        return 0, 0
    
    # Check completed images (resume from breakpoints for steps 2-3)
    processed_images = set()
    if resume:
        processed_images = get_processed_images(output_path)
        if processed_images:
            print(f"\nFound {len(processed_images)} completed final results, these images will be skipped")
    
    # Filter pending items
    pending_summaries = [
        item for item in summaries 
        if item["image_path"] not in processed_images
    ]
    
    total_count = len(summaries)
    already_done = len(processed_images)
    
    if not pending_summaries:
        print("All images have been processed")
        return already_done, total_count
        
    print(f"\n[Steps 2-3] Evaluating and validating...")
    print(f"Pending: {len(pending_summaries)} images | Completed: {already_done} images | Total: {total_count} images")
    
    success_count = already_done
    fail_count = 0
    
    # Process with progress bar
    pbar = tqdm(
        pending_summaries,
        desc="Evaluating and validating",
        unit="image",
        ncols=100,
    )
    
    for item in pbar:
        image_path = item["image_path"]
        original_summary = item["summary"]
        quality_level = item["quality_level"]
        
        # Retry logic
        temperatures = [0.7, 0.8, 0.9]
        passed = False
        
        for retry in range(max_retries):
            temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
            
            # Step 2: Feature 2 - Evaluate and improve
            result2 = evaluate_with_improvement(
                image_path=image_path,
                summary=original_summary,
                model_path=model_path,
                lora_path=lora_path_feature2 or lora_path,
                temperature=temp,
                top_p=0.9,
                do_sample=True,
            )
            
            if result2 is None:
                continue
                
            improved_summary = result2.get("improved_summary", "")
            if not improved_summary:
                continue
                
            # Step 3: Feature 3 - Validate improved summary
            result3 = score_only(
                image_path=image_path,
                summary=improved_summary,
                model_path=model_path,
                lora_path=lora_path_feature3 or lora_path,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            
            if result3 is None:
                continue
                
            # Validate scores
            is_valid, total_score, reason = validate_scores(result3)
            
            if is_valid:
                # Save result
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
            
        # Update progress bar
        pbar.set_postfix({
            "Success": success_count,
            "Failed": fail_count,
            "Pass Rate": f"{success_count/(success_count+fail_count)*100:.1f}%" if (success_count+fail_count) > 0 else "N/A",
        })
    
    pbar.close()
    
    print("\n" + "=" * 60)
    print(f"Pipeline 1 completed:")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {fail_count}")
    print(f"  - Total: {total_count}")
    print(f"  - Pass Rate: {success_count/total_count*100:.1f}%")
    print(f"Output file: {output_path}")
    print("=" * 60)
    
    return success_count, total_count


def pipeline2_user_optimize(
    image_path: str,
    summary: str,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline 2: User Optimization
    
    Workflow:
    1. Feature 2: Evaluate and generate improved version
    2. Feature 3: Validate the improved summary
    3. Retry if validation fails (max 3 times)
    4. Select the result with the highest total score if all 3 attempts fail
    
    Args:
        image_path: Path to image
        summary: Original summary provided by user
        model_path: Path to model
        lora_path: Path to LoRA weights
        max_retries: Maximum number of retries
        
    Returns:
        Final result (contains scores, reasons, weights, improved_summary)
    """
    print("=" * 60)
    print("Pipeline 2: User Optimization")
    print("=" * 60)
    
    temperatures = [0.7, 0.8, 0.9]
    all_results = []  # Save all attempted results and scores
    
    pbar = tqdm(range(max_retries), desc="Optimization attempts", unit="attempt", ncols=100)
    
    for retry in pbar:
        temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
        pbar.set_postfix({"temperature": f"{temp:.2f}"})
        
        # Step 1: Feature 2 - Evaluate and improve
        result2 = evaluate_with_improvement(
            image_path=image_path,
            summary=summary,
            model_path=model_path,
            lora_path=lora_path,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
        )
        
        if result2 is None:
            tqdm.write(f"  Attempt {retry+1}: Feature 2 failed")
            continue
            
        improved_summary = result2.get("improved_summary", "")
        if not improved_summary:
            tqdm.write(f"  Attempt {retry+1}: No improved_summary generated")
            continue
            
        # Step 2: Feature 3 - Validate improved summary
        result3 = score_only(
            image_path=image_path,
            summary=improved_summary,
            model_path=model_path,
            lora_path=lora_path,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        if result3 is None:
            tqdm.write(f"  Attempt {retry+1}: Feature 3 failed")
            all_results.append((result2, 0))
            continue
            
        # Validate scores
        is_valid, total_score, reason = validate_scores(result3)
        all_results.append((result2, total_score))
        
        if is_valid:
            pbar.close()
            print(f"\nValidation passed! Total score: {total_score}")
            print("\n" + "=" * 60)
            print("Pipeline 2 completed")
            print("=" * 60)
            return result2
        else:
            tqdm.write(f"  Attempt {retry+1}: Validation failed - {reason} (Total score: {total_score})")
    
    pbar.close()
            
    # All attempts failed validation, select the result with the highest total score
    if all_results:
        best_result, best_score = max(all_results, key=lambda x: x[1])
        print(f"\nAll attempts failed validation, selecting result with highest total score (Score: {best_score})")
        print("\n" + "=" * 60)
        print("Pipeline 2 completed (selected best result)")
        print("=" * 60)
        return best_result
    else:
        print("\nAll attempts failed")
        print("\n" + "=" * 60)
        print("Pipeline 2 failed")
        print("=" * 60)
        return None


def pipeline3_direct_score(
    image_path: str,
    summary: str,
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline 3: Direct Scoring
    
    Workflow:
    1. Feature 3: Score and output directly
    
    Args:
        image_path: Path to image
        summary: Summary provided by user
        model_path: Path to model
        lora_path: Path to LoRA weights
        
    Returns:
        Scoring result (contains scores, reasons, weights)
    """
    print("=" * 60)
    print("Pipeline 3: Direct Scoring")
    print("=" * 60)
    
    result = score_only(
        image_path=image_path,
        summary=summary,
        model_path=model_path,
        lora_path=lora_path,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    if result is not None:
        total_score = get_total_score(result)
        print(f"\nScoring completed, total score: {total_score}")
        print(f"Scores by dimension: {result.get('scores', {})}")
    else:
        print("\nScoring failed")
        
    print("\n" + "=" * 60)
    print("Pipeline 3 completed")
    print("=" * 60)
    
    return result