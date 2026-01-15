"""
Mixed Dataset - For Knowledge Distillation Training
Supports mixed sampling of scoring data and refinement data
"""
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MixedFigureDataset(Dataset):
    """
    Mixed Dataset: Combines scoring data and refinement data
    
    Used for knowledge distillation training, each sample is labeled with its type (score/refine),
    to calculate different losses during training.
    """
    
    def __init__(
        self,
        score_data_path: str,
        refine_data_path: str,
        processor,
        score_ratio: float = 0.3,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            score_data_path: Path to scoring data (training_data_l2.json)
            refine_data_path: Path to refinement data (training_data_l1.json)
            processor: Model processor
            score_ratio: Ratio of scoring data in mixed data (0-1)
            max_pixels: Maximum number of pixels for images
            min_pixels: Minimum number of pixels for images
            shuffle: Whether to shuffle data
            seed: Random seed
        """
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.score_ratio = score_ratio
        
        # Load data
        with open(score_data_path, "r", encoding="utf-8") as f:
            self.score_data = json.load(f)
        with open(refine_data_path, "r", encoding="utf-8") as f:
            self.refine_data = json.load(f)
            
        print(f"Loaded scoring data: {len(self.score_data)} entries")
        print(f"Loaded refinement data: {len(self.refine_data)} entries")
        
        # Build mixed dataset
        self.data = self._build_mixed_dataset(shuffle, seed)
        print(f"Total mixed dataset: {len(self.data)} entries")
        print(f"  - Scoring data: {sum(1 for d in self.data if d['task_type'] == 'score')} entries")
        print(f"  - Refinement data: {sum(1 for d in self.data if d['task_type'] == 'refine')} entries")
        
    def _build_mixed_dataset(self, shuffle: bool, seed: int) -> List[Dict]:
        """Build mixed dataset"""
        mixed_data = []
        
        # Add task type label to each data entry
        for item in self.score_data:
            mixed_data.append({
                **item,
                "task_type": "score"
            })
            
        for item in self.refine_data:
            mixed_data.append({
                **item,
                "task_type": "refine"
            })
        
        # Sample according to ratio
        # Calculate target quantities
        total_score = len(self.score_data)
        total_refine = len(self.refine_data)
        
        if self.score_ratio > 0 and self.score_ratio < 1:
            # Adjust according to ratio
            # Assume the desired mixed ratio is score_ratio : (1 - score_ratio)
            # Select an appropriate baseline
            target_score = int(total_refine * self.score_ratio / (1 - self.score_ratio))
            target_score = min(target_score, total_score)  # Cannot exceed actual quantity
            
            random.seed(seed)
            score_samples = random.sample(
                [d for d in mixed_data if d["task_type"] == "score"],
                target_score
            )
            refine_samples = [d for d in mixed_data if d["task_type"] == "refine"]
            
            mixed_data = score_samples + refine_samples
        
        if shuffle:
            random.seed(seed)
            random.shuffle(mixed_data)
            
        return mixed_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        
        image_path = item["image"]
        conversations = item["conversations"]
        task_type = item["task_type"]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")
            
        # Build messages
        user_content = conversations[0]["content"]
        assistant_content = conversations[1]["content"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_content},
                ],
            },
        ]
        
        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        
        # Remove batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_type": task_type,  # Key: label task type
        }
        
        # Process visual features
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(0)
            result["pixel_values"] = pixel_values
            
        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.squeeze(0)
            result["image_grid_thw"] = image_grid_thw
            
        return result


def mixed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Data collation function - Dynamic padding, preserves task type information
    """
    # Find maximum length
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    task_types = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            input_ids = torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=item["input_ids"].dtype)
            ])
            attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=item["attention_mask"].dtype)
            ])
            labels = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=item["labels"].dtype)
            ])
        else:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        task_types.append(item["task_type"])
    
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "task_types": task_types,  # Preserve task type list
    }
    
    # Process visual features
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        result["pixel_values"] = pixel_values
        
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        result["image_grid_thw"] = image_grid_thw
        
    return result


class ScoreOnlyDataset(Dataset):
    """
    Scoring-only Dataset - For teacher model distillation
    Contains only scoring task data, used to calculate distillation loss
    """
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
    ):
        """
        Args:
            data_path: Path to scoring data
            processor: Model processor
            max_pixels: Maximum number of pixels for images
            min_pixels: Minimum number of pixels for images
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        
        image_path = item["image"]
        conversations = item["conversations"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")
            
        user_content = conversations[0]["content"]
        assistant_content = conversations[1]["content"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_content},
                ],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(0)
            result["pixel_values"] = pixel_values
            
        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.squeeze(0)
            result["image_grid_thw"] = image_grid_thw
            
        return result