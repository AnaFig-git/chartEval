"""
混合数据集 - 用于知识蒸馏训练
支持打分数据和改进数据的混合采样
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
    混合数据集：结合打分数据和改进数据
    
    用于知识蒸馏训练，每个样本会标记其类型（score/refine），
    以便在训练时计算不同的损失。
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
            score_data_path: 打分数据路径 (training_data_l2.json)
            refine_data_path: 改进数据路径 (training_data_l1.json)
            processor: 模型处理器
            score_ratio: 打分数据在混合数据中的比例 (0-1)
            max_pixels: 图片最大像素数
            min_pixels: 图片最小像素数
            shuffle: 是否打乱数据
            seed: 随机种子
        """
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.score_ratio = score_ratio
        
        # 加载数据
        with open(score_data_path, "r", encoding="utf-8") as f:
            self.score_data = json.load(f)
        with open(refine_data_path, "r", encoding="utf-8") as f:
            self.refine_data = json.load(f)
            
        print(f"加载打分数据: {len(self.score_data)} 条")
        print(f"加载改进数据: {len(self.refine_data)} 条")
        
        # 构建混合数据集
        self.data = self._build_mixed_dataset(shuffle, seed)
        print(f"混合数据集总计: {len(self.data)} 条")
        print(f"  - 打分数据: {sum(1 for d in self.data if d['task_type'] == 'score')} 条")
        print(f"  - 改进数据: {sum(1 for d in self.data if d['task_type'] == 'refine')} 条")
        
    def _build_mixed_dataset(self, shuffle: bool, seed: int) -> List[Dict]:
        """构建混合数据集"""
        mixed_data = []
        
        # 为每条数据添加任务类型标记
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
        
        # 根据比例采样
        # 计算目标数量
        total_score = len(self.score_data)
        total_refine = len(self.refine_data)
        
        if self.score_ratio > 0 and self.score_ratio < 1:
            # 根据比例调整
            # 假设我们想要的混合比例是 score_ratio : (1 - score_ratio)
            # 选择一个合适的基准
            target_score = int(total_refine * self.score_ratio / (1 - self.score_ratio))
            target_score = min(target_score, total_score)  # 不能超过实际数量
            
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
        
        # 加载图片
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            image = Image.new("RGB", (224, 224), color="white")
            
        # 构建消息
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
        
        # 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        
        # 移除 batch 维度
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_type": task_type,  # 关键：标记任务类型
        }
        
        # 处理视觉特征
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
    数据批处理函数 - 动态 padding，保留任务类型信息
    """
    # 找到最大长度
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
        "task_types": task_types,  # 保留任务类型列表
    }
    
    # 处理视觉特征
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        result["pixel_values"] = pixel_values
        
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        result["image_grid_thw"] = image_grid_thw
        
    return result


class ScoreOnlyDataset(Dataset):
    """
    仅打分数据集 - 用于教师模型的蒸馏
    只包含打分任务的数据，用于计算蒸馏损失
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
            data_path: 打分数据路径
            processor: 模型处理器
            max_pixels: 图片最大像素数
            min_pixels: 图片最小像素数
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
            print(f"加载图片失败 {image_path}: {e}")
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

