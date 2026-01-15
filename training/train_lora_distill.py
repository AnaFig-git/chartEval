"""
Knowledge Distillation + Experience Replay LoRA Fine-tuning Script

Core Ideas:
- Use the second-stage scoring model as the teacher model
- The student model learns both scoring and refinement capabilities
- Maintain scoring capability through distillation loss and experience replay

Loss Function:
L_total = L_refine + β × L_distill + γ × L_replay

Where:
- L_refine: Cross-entropy loss on refinement data
- L_distill: KL divergence between student and teacher on scoring data
- L_replay: Cross-entropy loss on scoring data
"""
import os
import json
import argparse
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
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

from mixed_dataset import MixedFigureDataset, mixed_collate_fn


class DistillationTrainer(Trainer):
    """
    Knowledge Distillation Trainer

    Inherits from HuggingFace Trainer, overrides the compute_loss method
    Implements calculation of three loss components:
    1. L_refine: Cross-entropy loss for refinement tasks
    2. L_distill: Distillation loss (KL divergence) for scoring tasks
    3. L_replay: Experience replay loss (cross-entropy) for scoring tasks
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        distill_beta: float = 0.5,
        replay_gamma: float = 0.3,
        temperature: float = 2.0,
        *args,
        **kwargs
    ):
        """
        Args:
            teacher_model: Teacher model (frozen scoring model)
            distill_beta: Weight for distillation loss
            replay_gamma: Weight for experience replay loss
            temperature: Distillation temperature
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_beta = distill_beta
        self.replay_gamma = replay_gamma
        self.temperature = temperature
        
        # Ensure teacher model is not trainable
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        print(f"Distillation Configuration:")
        print(f"  - distill_beta (distillation weight): {self.distill_beta}")
        print(f"  - replay_gamma (replay weight): {self.replay_gamma}")
        print(f"  - temperature (distillation temperature): {self.temperature}")
        
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Calculate mixed loss
        
        L_total = L_refine + β × L_distill + γ × L_replay
        """
        # Extract task types
        task_types = inputs.pop("task_types", None)
        
        # Student model forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits
        labels = inputs["labels"]
        
        # Separate samples by task type
        batch_size = student_logits.size(0)
        
        if task_types is not None:
            score_mask = torch.tensor(
                [t == "score" for t in task_types],
                device=student_logits.device
            )
            refine_mask = torch.tensor(
                [t == "refine" for t in task_types],
                device=student_logits.device
            )
        else:
            # If no task type info, treat all as refinement tasks
            score_mask = torch.zeros(batch_size, dtype=torch.bool, device=student_logits.device)
            refine_mask = torch.ones(batch_size, dtype=torch.bool, device=student_logits.device)
        
        total_loss = torch.tensor(0.0, device=student_logits.device)
        loss_components = {}
        
        # 1. Refinement task loss (L_refine)
        if refine_mask.any():
            refine_logits = student_logits[refine_mask]
            refine_labels = labels[refine_mask]
            shift_logits = refine_logits[...,:-1,:].contiguous()
            shift_labels = refine_labels[..., 1:].contiguous()
            # Cross-entropy loss
            loss_refine = F.cross_entropy(
                refine_logits.view(-1, refine_logits.size(-1)),
                refine_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            total_loss = total_loss + loss_refine
            loss_components["loss_refine"] = loss_refine.item()
        
        # 2 & 3. Scoring task loss (L_distill + L_replay)
        if score_mask.any():
            score_logits = student_logits[score_mask]
            score_labels = labels[score_mask]
            
            # Get teacher model outputs
            with torch.no_grad():
                # Build teacher model inputs
                teacher_inputs = {
                    k: v[score_mask] if isinstance(v, torch.Tensor) and v.size(0) == batch_size else v
                    for k, v in inputs.items()
                    if k != "task_types"
                }
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            
            # L_distill: KL divergence distillation loss
            # Soften probability distributions with temperature
            student_log_probs = F.log_softmax(score_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # Calculate KL divergence only for non-padding positions
            valid_mask = (score_labels != -100).unsqueeze(-1).expand_as(student_log_probs)
            
            # KL(teacher || student) = sum(teacher * log(teacher/student))
            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none"
            )
            kl_div = (kl_div * valid_mask.float()).sum() / valid_mask.float().sum()
            loss_distill = kl_div * (self.temperature ** 2)  # Temperature scaling
            
            total_loss = total_loss + self.distill_beta * loss_distill
            loss_components["loss_distill"] = loss_distill.item()
            
            # L_replay: Experience replay loss (cross-entropy)
            loss_replay = F.cross_entropy(
                score_logits.view(-1, score_logits.size(-1)),
                score_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            total_loss = total_loss + self.replay_gamma * loss_replay
            loss_components["loss_replay"] = loss_replay.item()
        
        # Log loss components (for logging)
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_loss_components(loss_components, total_loss.item())
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _log_loss_components(self, components: Dict[str, float], total: float):
        """Log loss components"""
        log_str = f"Total Loss: {total:.4f}"
        for name, value in components.items():
            log_str += f" | {name}: {value:.4f}"
        print(log_str)


def train_with_distillation(
    base_model_path: str,
    teacher_lora_path: str,
    score_data_path: str,
    refine_data_path: str,
    output_dir: str,
    student_lora_path: Optional[str] = None,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_pixels: int = 1280 * 28 * 28,
    save_steps: int = 100,
    distill_beta: float = 0.5,
    replay_gamma: float = 0.3,
    temperature: float = 2.0,
    score_ratio: float = 0.3,
    use_flash_attn: bool = False,
):
    """
    Perform Knowledge Distillation LoRA Fine-tuning
    
    Args:
        base_model_path: Path to base model
        teacher_lora_path: Path to teacher model LoRA weights (scoring model l-2)
        score_data_path: Path to scoring data
        refine_data_path: Path to refinement data
        output_dir: Output directory
        student_lora_path: Initial LoRA path for student model (optional, for continuing from l-2)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_pixels: Maximum number of pixels for images
        save_steps: Number of steps between checkpoints
        distill_beta: Weight for distillation loss
        replay_gamma: Weight for experience replay loss
        temperature: Distillation temperature
        score_ratio: Ratio of scoring data in mixed dataset
        use_flash_attn: Whether to use Flash Attention
    """
    print("=" * 60)
    print("Knowledge Distillation LoRA Fine-tuning - Qwen3-VL")
    print("=" * 60)
    
    # Model loading configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    # ========== 1. Load Teacher Model ==========
    print(f"\n[1/4] Loading teacher model...")
    print(f"  Base Model: {base_model_path}")
    print(f"  Teacher LoRA: {teacher_lora_path}")
    
    teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **model_kwargs,
    )
    teacher_model = PeftModel.from_pretrained(
        teacher_model,
        teacher_lora_path,
        is_trainable=False,
    )
    teacher_model.eval()
    print("Teacher model loaded and frozen")
    
    # ========== 2. Load Student Model ==========
    print(f"\n[2/4] Loading student model...")
    
    student_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **model_kwargs,
    )
    student_model.gradient_checkpointing_enable()
    
    if student_lora_path:
        # Continue training from existing LoRA weights
        print(f"  Continuing from existing LoRA: {student_lora_path}")
        student_model = PeftModel.from_pretrained(
            student_model,
            student_lora_path,
            is_trainable=True,
        )
    else:
        # Create new LoRA configuration
        print("  Creating new LoRA configuration...")
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
        student_model = get_peft_model(student_model, lora_config)
    
    student_model.print_trainable_parameters()
    
    # ========== 3. Load Processor and Dataset ==========
    print(f"\n[3/4] Loading dataset...")
    
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=256 * 28 * 28,
    )
    
    dataset = MixedFigureDataset(
        score_data_path=score_data_path,
        refine_data_path=refine_data_path,
        processor=processor,
        score_ratio=score_ratio,
        max_pixels=max_pixels,
    )
    
    # ========== 4. Configure Training ==========
    print(f"\n[4/4] Configuring training parameters...")
    
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
        optim="adamw_torch_fused",
        dataloader_pin_memory=False,
    )
    
    # Create distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        distill_beta=distill_beta,
        replay_gamma=replay_gamma,
        temperature=temperature,
        model=student_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=mixed_collate_fn,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting knowledge distillation training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save model
    print(f"\nSaving LoRA weights to: {output_dir}")
    student_model.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation LoRA Fine-tuning")
    
    # Model paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="Path to base model",
    )
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default="./lora_weights/l-2",
        help="Path to teacher model LoRA weights (scoring model)",
    )
    parser.add_argument(
        "--student_lora_path",
        type=str,
        default=None,
        help="Initial LoRA path for student model (optional, for continued training)",
    )
    
    # Data paths
    parser.add_argument(
        "--score_data_path",
        type=str,
        default="./data/output/training_data_l2.json",
        help="Path to scoring data",
    )
    parser.add_argument(
        "--refine_data_path",
        type=str,
        default="./data/output/training_data_l1.json",
        help="Path to refinement data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_weights/l-3-distill",
        help="Output directory",
    )
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28, help="Maximum number of pixels for images")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of steps between checkpoints")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    
    # Distillation configuration
    parser.add_argument("--distill_beta", type=float, default=0.5, help="Weight for distillation loss")
    parser.add_argument("--replay_gamma", type=float, default=0.3, help="Weight for experience replay loss")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--score_ratio", type=float, default=0.3, help="Ratio of scoring data in mixed dataset")
    
    args = parser.parse_args()
    
    train_with_distillation(
        base_model_path=args.base_model_path,
        teacher_lora_path=args.teacher_lora_path,
        student_lora_path=args.student_lora_path,
        score_data_path=args.score_data_path,
        refine_data_path=args.refine_data_path,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_pixels=args.max_pixels,
        save_steps=args.save_steps,
        distill_beta=args.distill_beta,
        replay_gamma=args.replay_gamma,
        temperature=args.temperature,
        score_ratio=args.score_ratio,
        use_flash_attn=args.flash_attn,
    )


if __name__ == "__main__":
    main()
