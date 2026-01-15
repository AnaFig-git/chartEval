"""
LoRA Fine-tuning Script - Using PEFT Library
Supports two fine-tuning schemes:
- l-1: Includes improved_summary
- l-2: Excludes improved_summary

Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
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
    """Figure Evaluation Dataset - Processed according to Qwen3-VL official recommendations"""
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_pixels: int = 1280 * 28 * 28,  # Control image size to reduce VRAM usage
        min_pixels: int = 256 * 28 * 28,
    ):
        """
        Args:
            data_path: Path to training data JSON file
            processor: Model processor
            max_pixels: Maximum number of pixels for images (to control VRAM usage)
            min_pixels: Minimum number of pixels for images
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
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            # Return a blank image
            image = Image.new("RGB", (224, 224), color="white")
            
        # Build messages - Use URL/path format instead of passing PIL Image directly
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
        
        # Process according to official documentation recommendations
        # Use apply_chat_template to complete tokenization in one step
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
        
        # Create labels (copy input_ids)
        labels = input_ids.clone()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Process pixel_values
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if pixel_values.dim() == 5:  # (batch, num_images, channels, height, width)
                pixel_values = pixel_values.squeeze(0)
            result["pixel_values"] = pixel_values
        
        # Process image_grid_thw
        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"]
            if image_grid_thw.dim() == 3:  # (batch, num_images, 3)
                image_grid_thw = image_grid_thw.squeeze(0)
            result["image_grid_thw"] = image_grid_thw
            
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Data Collation Function - Dynamic padding
    """
    # Find maximum length
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
    
    # Process pixel_values - Merge with cat
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        result["pixel_values"] = pixel_values
    
    # Process image_grid_thw - Merge with cat
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
    Perform LoRA Fine-tuning
    
    Args:
        model_path: Path to base model
        data_path: Path to training data
        output_dir: Output directory
        resume_lora_path: Path to existing LoRA weights (for incremental fine-tuning)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_pixels: Maximum number of pixels for images (controls VRAM usage)
        save_steps: Number of steps between checkpoints
        use_flash_attn: Whether to use Flash Attention 2
    """
    print("=" * 60)
    print("LoRA Fine-tuning - Qwen3-VL")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Using Flash Attention 2 can save VRAM and accelerate training
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    print("Gradient Checkpointing enabled")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=256 * 28 * 28,
    )
    
    # Configure LoRA
    if resume_lora_path:
        # Incremental fine-tuning: Load existing LoRA weights to continue training
        print(f"\nLoading existing LoRA weights for incremental fine-tuning: {resume_lora_path}")
        model = PeftModel.from_pretrained(
            model,
            resume_lora_path,
            is_trainable=True,  # Key: allow continued training
        )
        print("Existing LoRA weights loaded")
    else:
        # Start from scratch: Create new LoRA configuration
        print("\nCreating new LoRA configuration...")
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
    
    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    dataset = FigureEvaluationDataset(
        data_path=data_path,
        processor=processor,
        max_pixels=max_pixels,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Maximum image pixels: {max_pixels} (approximately {int((max_pixels / 28 / 28) ** 0.5 * 28)}x{int((max_pixels / 28 / 28) ** 0.5 * 28)})")
    
    # Configure training parameters
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
        # Optimize VRAM usage
        optim="adamw_torch_fused",
        dataloader_pin_memory=False,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    # Save LoRA weights
    print(f"\nSaving LoRA weights to: {output_dir}")
    model.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="Path to base model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["l-1", "l-2"],
        default="l-1",
        help="Fine-tuning scheme: l-1 (includes improved_summary) or l-2 (excludes)",
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
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum number of pixels for images (controls VRAM, default ~1280x1280)",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use Flash Attention 2 (requires flash-attn installation)",
    )
    parser.add_argument(
        "--resume_lora_path",
        type=str,
        default=None,
        help="Path to existing LoRA weights (for incremental fine-tuning, continue training on existing weights)",
    )
    
    args = parser.parse_args()
    
    # Set default output directory based on scheme
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