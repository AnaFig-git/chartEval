"""
知识蒸馏 + 经验回放 LoRA 微调脚本

核心思路：
- 使用第二阶段打分模型作为教师模型
- 学生模型学习打分和改进两种能力
- 通过蒸馏损失和经验回放保持打分能力

损失函数：
L_total = L_refine + β × L_distill + γ × L_replay

其中：
- L_refine: 改进数据的交叉熵损失
- L_distill: 打分数据上学生与教师的 KL 散度
- L_replay: 打分数据的交叉熵损失
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
    知识蒸馏训练器
    
    继承自 HuggingFace Trainer，重写 compute_loss 方法
    实现三部分损失的计算：
    1. L_refine: 改进任务的交叉熵损失
    2. L_distill: 打分任务的蒸馏损失（KL散度）
    3. L_replay: 打分任务的经验回放损失（交叉熵）
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
            teacher_model: 教师模型（冻结的打分模型）
            distill_beta: 蒸馏损失权重
            replay_gamma: 经验回放损失权重
            temperature: 蒸馏温度
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_beta = distill_beta
        self.replay_gamma = replay_gamma
        self.temperature = temperature
        
        # 确保教师模型不训练
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        print(f"蒸馏配置:")
        print(f"  - distill_beta (蒸馏权重): {self.distill_beta}")
        print(f"  - replay_gamma (回放权重): {self.replay_gamma}")
        print(f"  - temperature (蒸馏温度): {self.temperature}")
        
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        计算混合损失
        
        L_total = L_refine + β × L_distill + γ × L_replay
        """
        # 提取任务类型
        task_types = inputs.pop("task_types", None)
        
        # 学生模型前向传播
        outputs = model(**inputs)
        student_logits = outputs.logits
        labels = inputs["labels"]
        
        # 分离不同任务类型的样本
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
            # 如果没有任务类型信息，全部当作改进任务
            score_mask = torch.zeros(batch_size, dtype=torch.bool, device=student_logits.device)
            refine_mask = torch.ones(batch_size, dtype=torch.bool, device=student_logits.device)
        
        total_loss = torch.tensor(0.0, device=student_logits.device)
        loss_components = {}
        
        # 1. 改进任务损失 (L_refine)
        if refine_mask.any():
            refine_logits = student_logits[refine_mask]
            refine_labels = labels[refine_mask]
            
            # 交叉熵损失
            loss_refine = F.cross_entropy(
                refine_logits.view(-1, refine_logits.size(-1)),
                refine_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            total_loss = total_loss + loss_refine
            loss_components["loss_refine"] = loss_refine.item()
        
        # 2 & 3. 打分任务损失 (L_distill + L_replay)
        if score_mask.any():
            score_logits = student_logits[score_mask]
            score_labels = labels[score_mask]
            
            # 获取教师模型输出
            with torch.no_grad():
                # 构建教师模型的输入
                teacher_inputs = {
                    k: v[score_mask] if isinstance(v, torch.Tensor) and v.size(0) == batch_size else v
                    for k, v in inputs.items()
                    if k != "task_types"
                }
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            
            # L_distill: KL 散度蒸馏损失
            # 使用温度软化概率分布
            student_log_probs = F.log_softmax(score_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # 只计算非 padding 位置的 KL 散度
            valid_mask = (score_labels != -100).unsqueeze(-1).expand_as(student_log_probs)
            
            # KL(teacher || student) = sum(teacher * log(teacher/student))
            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none"
            )
            kl_div = (kl_div * valid_mask.float()).sum() / valid_mask.float().sum()
            loss_distill = kl_div * (self.temperature ** 2)  # 温度缩放
            
            total_loss = total_loss + self.distill_beta * loss_distill
            loss_components["loss_distill"] = loss_distill.item()
            
            # L_replay: 经验回放损失（交叉熵）
            loss_replay = F.cross_entropy(
                score_logits.view(-1, score_logits.size(-1)),
                score_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            total_loss = total_loss + self.replay_gamma * loss_replay
            loss_components["loss_replay"] = loss_replay.item()
        
        # 记录损失组件（用于日志）
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_loss_components(loss_components, total_loss.item())
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _log_loss_components(self, components: Dict[str, float], total: float):
        """记录损失组件"""
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
    执行知识蒸馏 LoRA 微调
    
    Args:
        base_model_path: 基础模型路径
        teacher_lora_path: 教师模型 LoRA 路径（打分模型 l-2）
        score_data_path: 打分数据路径
        refine_data_path: 改进数据路径
        output_dir: 输出目录
        student_lora_path: 学生模型初始 LoRA 路径（可选，用于从 l-2 继续）
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 学习率
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        max_pixels: 图片最大像素数
        save_steps: 保存步数
        distill_beta: 蒸馏损失权重
        replay_gamma: 经验回放损失权重
        temperature: 蒸馏温度
        score_ratio: 打分数据比例
        use_flash_attn: 是否使用 Flash Attention
    """
    print("=" * 60)
    print("知识蒸馏 LoRA 微调 - Qwen3-VL")
    print("=" * 60)
    
    # 模型加载配置
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("使用 Flash Attention 2")
    
    # ========== 1. 加载教师模型 ==========
    print(f"\n[1/4] 加载教师模型...")
    print(f"  基础模型: {base_model_path}")
    print(f"  教师 LoRA: {teacher_lora_path}")
    
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
    print("教师模型加载完成并冻结")
    
    # ========== 2. 加载学生模型 ==========
    print(f"\n[2/4] 加载学生模型...")
    
    student_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        **model_kwargs,
    )
    student_model.gradient_checkpointing_enable()
    
    if student_lora_path:
        # 从现有 LoRA 继续训练
        print(f"  从现有 LoRA 继续: {student_lora_path}")
        student_model = PeftModel.from_pretrained(
            student_model,
            student_lora_path,
            is_trainable=True,
        )
    else:
        # 创建新的 LoRA
        print("  创建新的 LoRA 配置...")
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
    
    # ========== 3. 加载处理器和数据集 ==========
    print(f"\n[3/4] 加载数据集...")
    
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
    
    # ========== 4. 配置训练 ==========
    print(f"\n[4/4] 配置训练参数...")
    
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
    
    # 创建蒸馏训练器
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
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始知识蒸馏训练...")
    print("=" * 60)
    
    trainer.train()
    
    # 保存模型
    print(f"\n保存 LoRA 权重到: {output_dir}")
    student_model.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="知识蒸馏 LoRA 微调")
    
    # 模型路径
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./Qwen3-VL-8B-Instruct",
        help="基础模型路径",
    )
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default="./lora_weights/l-2",
        help="教师模型 LoRA 路径（打分模型）",
    )
    parser.add_argument(
        "--student_lora_path",
        type=str,
        default=None,
        help="学生模型初始 LoRA 路径（可选，用于继续训练）",
    )
    
    # 数据路径
    parser.add_argument(
        "--score_data_path",
        type=str,
        default="./data/output/training_data_l2.json",
        help="打分数据路径",
    )
    parser.add_argument(
        "--refine_data_path",
        type=str,
        default="./data/output/training_data_l1.json",
        help="改进数据路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_weights/l-3-distill",
        help="输出目录",
    )
    
    # LoRA 配置
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 训练配置
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28, help="图片最大像素数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--flash_attn", action="store_true", help="使用 Flash Attention 2")
    
    # 蒸馏配置
    parser.add_argument("--distill_beta", type=float, default=0.5, help="蒸馏损失权重")
    parser.add_argument("--replay_gamma", type=float, default=0.3, help="经验回放损失权重")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--score_ratio", type=float, default=0.3, help="打分数据比例")
    
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

