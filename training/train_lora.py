"""
LoRA 微调脚本 - 使用 PEFT 库
支持两种微调方案：
- l-1: 包含 improved_summary
- l-2: 不包含 improved_summary

参考：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
import os
import json
import argparse
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)


class FigureEvaluationDataset(Dataset):
    """图表评估数据集 - 按照 Qwen3-VL 官方推荐方式处理"""
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_pixels: int = 1280 * 28 * 28,  # 控制图片大小，减少显存
        min_pixels: int = 256 * 28 * 28,
    ):
        """
        Args:
            data_path: 训练数据 JSON 文件路径
            processor: 模型处理器
            max_pixels: 图片最大像素数（用于控制显存）
            min_pixels: 图片最小像素数
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item["image"]
        conversations = item["conversations"]
        
        # 加载图片
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            # 返回一个空白图片
            image = Image.new("RGB", (224, 224), color="white")
            
        # 构建消息 - 使用 URL/路径 格式而非直接传入 PIL Image
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
        
        # 按照官方文档推荐的方式处理
        # 使用 apply_chat_template 一步完成 tokenize
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
        
        # 创建 labels（复制 input_ids）
        labels = input_ids.clone()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # 处理 pixel_values
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 5:  # (batch, num_images, channels, height, width)
                pixel_values = pixel_values.squeeze(0)
            result["pixel_values"] = pixel_values
        
        # 处理 image_grid_thw
        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 3:  # (batch, num_images, 3)
                image_grid_thw = image_grid_thw.squeeze(0)
            result["image_grid_thw"] = image_grid_thw
            
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    数据批处理函数 - 动态 padding
    """
    # 找到最大长度
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Padding input_ids, attention_mask, labels
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            # Pad input_ids with pad_token_id (0)
            input_ids = torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=item["input_ids"].dtype)
            ])
            # Pad attention_mask with 0
            attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=item["attention_mask"].dtype)
            ])
            # Pad labels with -100 (ignore index)
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
    
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }
    
    # 处理 pixel_values - 使用 cat 合并
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        result["pixel_values"] = pixel_values
    
    # 处理 image_grid_thw - 使用 cat 合并
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        result["image_grid_thw"] = image_grid_thw
        
    return result


def train_lora(
    model_path: str,
    data_path: str,
    output_dir: str,
    resume_lora_path: Optional[str] = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_pixels: int = 1280 * 28 * 28,
    save_steps: int = 100,
    use_flash_attn: bool = False,
):
    """
    执行 LoRA 微调
    
    Args:
        model_path: 基础模型路径
        data_path: 训练数据路径
        output_dir: 输出目录
        resume_lora_path: 现有 LoRA 权重路径（用于增量微调）
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 学习率
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        max_pixels: 图片最大像素数（控制显存）
        save_steps: 保存步数
        use_flash_attn: 是否使用 flash attention 2
    """
    print("=" * 60)
    print("LoRA 微调 - Qwen3-VL")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # 使用 flash attention 2 可以节省显存并加速
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("使用 Flash Attention 2")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs,
    )
    
    # 启用 gradient checkpointing 来节省显存
    model.gradient_checkpointing_enable()
    print("已启用 Gradient Checkpointing")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=256 * 28 * 28,
    )
    
    # 配置 LoRA
    if resume_lora_path:
        # 增量微调：加载现有 LoRA 权重继续训练
        print(f"\n加载现有 LoRA 权重进行增量微调: {resume_lora_path}")
        model = PeftModel.from_pretrained(
            model,
            resume_lora_path,
            is_trainable=True,  # 关键：允许继续训练
        )
        print("已加载现有 LoRA 权重")
    else:
        # 从头开始：创建新的 LoRA 配置
        print("\n创建新的 LoRA 配置...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # 加载数据集
    print(f"\n加载数据集: {data_path}")
    dataset = FigureEvaluationDataset(
        data_path=data_path,
        processor=processor,
        max_pixels=max_pixels,
    )
    print(f"数据集大小: {len(dataset)}")
    print(f"图片最大像素: {max_pixels} ({int((max_pixels / 28 / 28) ** 0.5 * 28)}x{int((max_pixels / 28 / 28) ** 0.5 * 28)} 左右)")
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        # 优化显存
        optim="adamw_torch_fused",
        dataloader_pin_memory=False,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存 LoRA 权重
    print(f"\n保存 LoRA 权重到: {output_dir}")
    model.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调脚本")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="基础模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练数据路径（JSON 格式）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["l-1", "l-2"],
        default="l-1",
        help="微调方案: l-1（含improved_summary）或 l-2（不含）",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="图片最大像素数（控制显存，默认约 1280x1280）",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="使用 Flash Attention 2（需要安装 flash-attn）",
    )
    parser.add_argument(
        "--resume_lora_path",
        type=str,
        default=None,
        help="现有 LoRA 权重路径（用于增量微调，在已有权重基础上继续训练）",
    )
    
    args = parser.parse_args()
    
    # 根据方案设置默认输出目录
    if args.output_dir is None:
        args.output_dir = f"./lora_weights/{args.scheme}"
        
    train_lora(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        resume_lora_path=args.resume_lora_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_pixels=args.max_pixels,
        use_flash_attn=args.flash_attn,
    )


if __name__ == "__main__":
    main()
