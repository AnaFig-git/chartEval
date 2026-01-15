"""
API Version Pipeline Implementation - Build Dataset from Step1 Results
Skip Feature 1 (batch image summarization) and directly execute Feature 2 and Feature 3
"""
import sys
import os

# Add project root directory to path
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
    Build dataset from step1 results (API Version)
    
    Skip Feature 1 (batch image summarization), read data directly from dataset_step1.jsonl,
    and execute Feature 2 and Feature 3 for evaluation and validation.
    
    Workflow:
    1. Read step1 results (image_path, quality_level, summary)
    2. Feature 2: Evaluate and generate improved version
    3. Feature 3: Validate the improved summary
    4. Retry if validation fails (max 3 times)
    5. Save if passed, discard if failed
    
    Args:
        input_path: Path to step1 result file (dataset_step1.jsonl)
        output_path: Output JSONL file path
        max_retries: Maximum number of retries
        resume: Whether to resume from breakpoint
        
    Returns:
        (Number of successes, Total number)
    """
    print("=" * 60)
    print("API Version Pipeline 1: Build Dataset from Step1 Results")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read step1 results
    print(f"\n[Reading Data] Loading step1 results from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return 0, 0
    
    summaries = load_jsonl(input_path)
    
    if not summaries:
        print("Error: No data was read")
        return 0, 0
    
    print(f"Read {len(summaries)} records")
    
    # Check completed images (breakpoint resume)
    processed_images = set()
    if resume:
        processed_images = get_processed_images(output_path)
        if processed_images:
            print(f"\nFound {len(processed_images)} completed final results, will skip these images")
    
    # Filter pending items
    pending_summaries = [
        item for item in summaries 
        if item.get("image_path") not in processed_images
    ]
    
    total_count = len(summaries)
    already_done = len(processed_images)
    
    if not pending_summaries:
        print("All images have been processed")
        return already_done, total_count
        
    print(f"\n[Steps 2-3] Evaluating and Validating...")
    print(f"Pending: {len(pending_summaries)} images | Completed: {already_done} images | Total: {total_count} images")
    
    success_count = already_done
    fail_count = 0
    
    # Process with progress bar
    pbar = tqdm(
        pending_summaries,
        desc="Evaluating and Validating",
        unit="image",
        ncols=100,
    )
    
    for item in pbar:
        image_path = item.get("image_path", "")
        original_summary = item.get("summary", "")
        quality_level = item.get("quality_level", "medium")
        
        # Check if image exists
        if not os.path.exists(image_path):
            tqdm.write(f"Warning: Image does not exist, skipping: {image_path}")
            fail_count += 1
            continue
        
        # Retry logic
        temperatures = [0.7, 0.8, 0.9]
        passed = False
        
        for retry in range(max_retries):
            temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
            
            # Step 2: Feature 2 - Evaluate and improve
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
                
            # Step 3: Feature 3 - Validate improved summary
            result3 = score_only(
                image_path=image_path,
                summary=improved_summary,
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
            "Success Rate": f"{success_count/(success_count+fail_count)*100:.1f}%" if (success_count+fail_count) > 0 else "N/A",
        })
    
    pbar.close()
    
    print("\n" + "=" * 60)
    print(f"API Version Pipeline 1 Completed:")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {fail_count}")
    print(f"  - Total: {total_count}")
    print(f"  - Success Rate: {success_count/total_count*100:.1f}%")
    print(f"Output File: {output_path}")
    print("=" * 60)
    
    return success_count, total_count


def pipeline2_user_optimize(
    image_path: str,
    summary: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline 2: User Optimization (API Version)
    
    Workflow:
    1. Feature 2: Evaluate and generate improved version
    2. Feature 3: Validate the improved summary
    3. Retry if validation fails (max 3 times)
    4. Select the result with the highest total score if all 3 attempts fail
    
    Args:
        image_path: Image path
        summary: Original summary provided by user
        max_retries: Maximum number of retries
        
    Returns:
        Final result (contains scores, reasons, weights, improved_summary)
    """
    print("=" * 60)
    print("API Version Pipeline 2: User Optimization")
    print("=" * 60)
    
    temperatures = [0.7, 0.8, 0.9]
    all_results = []  # Save all attempted results and scores
    
    pbar = tqdm(range(max_retries), desc="Optimization Attempts", unit="attempt", ncols=100)
    
    for retry in pbar:
        temp = temperatures[retry] if retry < len(temperatures) else 0.7 + 0.1 * retry
        pbar.set_postfix({"temperature": f"{temp:.2f}"})
        
        # Step 1: Feature 2 - Evaluate and improve
        result2 = evaluate_with_improvement(
            image_path=image_path,
            summary=summary,
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
            print(f"\nValidation Passed! Total Score: {total_score}")
            print("\n" + "=" * 60)
            print("API Version Pipeline 2 Completed")
            print("=" * 60)
            return result2
        else:
            tqdm.write(f"  Attempt {retry+1}: Validation Failed - {reason} (Total Score: {total_score})")
    
    pbar.close()
            
    # Select the result with the highest total score if all attempts failed
    if all_results:
        best_result, best_score = max(all_results, key=lambda x: x[1])
        print(f"\nAll attempts failed validation, selecting result with highest total score (Score: {best_score})")
        print("\n" + "=" * 60)
        print("API Version Pipeline 2 Completed (Best Result Selected)")
        print("=" * 60)
        return best_result
    else:
        print("\nAll attempts failed")
        print("\n" + "=" * 60)
        print("API Version Pipeline 2 Failed")
        print("=" * 60)
        return None


def pipeline3_direct_score(
    image_path: str,
    summary: str,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline 3: Direct Scoring (API Version)
    
    Workflow:
    1. Feature 3: Score and output directly
    
    Args:
        image_path: Image path
        summary: Summary provided by user
        
    Returns:
        Scoring result (contains scores, reasons, weights)
    """
    print("=" * 60)
    print("API Version Pipeline 3: Direct Scoring")
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
        print(f"\nScoring Completed, Total Score: {total_score}")
        print(f"Scores by Dimension: {result.get('scores', {})}")
    else:
        print("\nScoring Failed")
        
    print("\n" + "=" * 60)
    print("API Version Pipeline 3 Completed")
    print("=" * 60)
    
    return result