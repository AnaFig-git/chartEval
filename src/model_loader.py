"""
模型加载器 - 支持 Qwen3-VL-8B-Instruct 和 LoRA 动态加载
参考: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
import os
from typing import Optional, Union, List
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


class Qwen3VLModel:
    """Qwen3-VL 模型封装，支持 LoRA 动态加载"""
    
    def __init__(
        self,
        model_path: str = "./Qwen3-VL-8B-Instruct",
        device_id: Optional[int] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        初始化模型
        
        Args:
            model_path: 模型路径
            device_id: GPU 设备 ID（如 0, 1, 2...），None 表示自动选择
            torch_dtype: 模型精度
        """
        self.model_path = model_path
        self.device_id = device_id
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self.current_lora_path = None
        
    def load_model(self):
        """加载基础模型和处理器"""
        if self.model is not None:
            return
            
        print(f"正在加载模型: {self.model_path}")
        
        # 检查可用 GPU 数量
        num_gpus = torch.cuda.device_count()
        print(f"可见 GPU 数量: {num_gpus}")
        
        # 确定设备映射
        if self.device_id is not None:
            # 明确指定 GPU 设备，确保模型完全加载到该 GPU
            device = f"cuda:{self.device_id}"
            device_map = {"": device}
            print(f"使用指定 GPU: {device}")
        elif num_gpus == 1:
            # 只有一个可见 GPU 时，直接加载到 cuda:0（避免 auto 的 offload 问题）
            device_map = {"": "cuda:0"}
            print("检测到单 GPU，直接加载到 cuda:0")
        elif num_gpus > 1:
            # 多个 GPU 时使用自动分配
            device_map = "auto"
            print("检测到多 GPU，使用自动设备映射")
        else:
            # 没有 GPU 时使用 CPU
            device_map = "cpu"
            print("未检测到 GPU，使用 CPU")
        
        # 使用官方推荐的 Qwen3VLForConditionalGeneration
        # 参考: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # 设置模型为推理模式
        self.model.eval()
        
        # 打印实际加载的设备信息
        print(f"模型加载完成，实际设备: {self.model.device}")
        
    def load_lora(self, lora_path: str):
        """
        加载 LoRA 适配器
        
        Args:
            lora_path: LoRA 权重路径
        """
        if self.model is None:
            self.load_model()
            
        if self.current_lora_path == lora_path:
            print(f"LoRA 已加载: {lora_path}")
            return
            
        # 如果已有 LoRA，先卸载
        if self.current_lora_path is not None:
            self.unload_lora()
            
        if lora_path and os.path.exists(lora_path):
            print(f"正在加载 LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                is_trainable=False,
            )
            self.current_lora_path = lora_path
            print("LoRA 加载完成")
        else:
            print(f"LoRA 路径不存在: {lora_path}")
            
    def unload_lora(self):
        """卸载当前 LoRA 适配器"""
        if self.current_lora_path is not None and hasattr(self.model, 'unload'):
            print("正在卸载 LoRA...")
            self.model = self.model.unload()
            self.current_lora_path = None
            print("LoRA 已卸载")
            
    def generate(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
    ) -> str:
        """
        生成回复（按官方推荐方式）
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: top-p 采样参数
            top_k: top-k 采样参数
            do_sample: 是否采样
            
        Returns:
            生成的文本
        """
        if self.model is None:
            self.load_model()
            
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 构建消息（官方格式）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 官方推荐的处理方式：apply_chat_template 直接返回 tensor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 生成参数（参考官方推荐）
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        
        if do_sample:
            generate_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
            
        # 解码输出（只取新生成的部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text
    
    def generate_batch(
        self,
        image_paths: List[Union[str, Path]],
        prompts: List[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
    ) -> List[str]:
        """
        批量生成回复（逐个处理）
        
        Args:
            image_paths: 图片路径列表
            prompts: 提示词列表
            其他参数同 generate
            
        Returns:
            生成的文本列表
        """
        results = []
        for image_path, prompt in zip(image_paths, prompts):
            result = self.generate(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
            results.append(result)
        return results


# 全局模型实例（单例模式）
_model_instance: Optional[Qwen3VLModel] = None


def get_model(
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    device_id: Optional[int] = None,
) -> Qwen3VLModel:
    """
    获取模型实例（单例模式）
    
    Args:
        model_path: 模型路径
        lora_path: LoRA 权重路径（可选）
        device_id: GPU 设备 ID（如 0, 1, 2...），None 表示自动选择
        
    Returns:
        模型实例
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = Qwen3VLModel(model_path=model_path, device_id=device_id)
        
    _model_instance.load_model()
    
    if lora_path:
        _model_instance.load_lora(lora_path)
    elif _model_instance.current_lora_path is not None:
        _model_instance.unload_lora()
        
    return _model_instance
