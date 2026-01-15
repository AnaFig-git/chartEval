"""
Model Loader - Supports Qwen3-VL-8B-Instruct and dynamic LoRA loading
Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
import os
from typing import Optional, Union, List
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


class Qwen3VLModel:
    """Qwen3-VL Model Wrapper with dynamic LoRA loading support"""
    
    def __init__(
        self,
        model_path: str = "./Qwen3-VL-8B-Instruct",
        device_id: Optional[int] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize model
        
        Args:
            model_path: Model path
            device_id: GPU device ID (e.g., 0, 1, 2...), None for automatic selection
            torch_dtype: Model precision
        """
        self.model_path = model_path
        self.device_id = device_id
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self.current_lora_path = None
        
    def load_model(self):
        """Load base model and processor"""
        if self.model is not None:
            return
            
        print(f"Loading model: {self.model_path}")
        
        # Check number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of visible GPUs: {num_gpus}")
        
        # Determine device mapping
        if self.device_id is not None:
            # Specify GPU device explicitly to ensure model loads completely to this GPU
            device = f"cuda:{self.device_id}"
            device_map = {"": device}
            print(f"Using specified GPU: {device}")
        elif num_gpus == 1:
            # Load directly to cuda:0 when only one GPU is visible (avoid auto offload issues)
            device_map = {"": "cuda:0"}
            print("Single GPU detected, loading directly to cuda:0")
        elif num_gpus > 1:
            # Use automatic allocation for multiple GPUs
            device_map = "auto"
            print("Multiple GPUs detected, using automatic device mapping")
        else:
            # Use CPU when no GPU is available
            device_map = "cpu"
            print("No GPUs detected, using CPU")
        
        # Use officially recommended Qwen3VLForConditionalGeneration
        # Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
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
        
        # Set model to inference mode
        self.model.eval()
        
        # Print actual loaded device information
        print(f"Model loaded successfully, actual device: {self.model.device}")
        
    def load_lora(self, lora_path: str):
        """
        Load LoRA adapter
        
        Args:
            lora_path: Path to LoRA weights
        """
        if self.model is None:
            self.load_model()
            
        if self.current_lora_path == lora_path:
            print(f"LoRA already loaded: {lora_path}")
            return
            
        # Unload existing LoRA if present
        if self.current_lora_path is not None:
            self.unload_lora()
            
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                is_trainable=False,
            )
            self.current_lora_path = lora_path
            print("LoRA loaded successfully")
        else:
            print(f"LoRA path does not exist: {lora_path}")
            
    def unload_lora(self):
        """Unload current LoRA adapter"""
        if self.current_lora_path is not None and hasattr(self.model, 'unload'):
            print("Unloading LoRA...")
            self.model = self.model.unload()
            self.current_lora_path = None
            print("LoRA unloaded successfully")
            
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
        Generate response (following official recommended method)
        
        Args:
            image_path: Path to image
            prompt: Prompt
            max_new_tokens: Maximum new tokens
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
            
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Build messages (official format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Officially recommended processing: apply_chat_template returns tensor directly
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generation parameters (refer to official recommendations)
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
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
            
        # Decode output (only take newly generated part)
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
        Batch generate responses (process one by one)
        
        Args:
            image_paths: List of image paths
            prompts: List of prompts
            Other parameters same as generate
            
        Returns:
            List of generated texts
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


# Global model instance (singleton pattern)
_model_instance: Optional[Qwen3VLModel] = None


def get_model(
    model_path: str = "./Qwen3-VL-8B-Instruct",
    lora_path: Optional[str] = None,
    device_id: Optional[int] = None,
) -> Qwen3VLModel:
    """
    Get model instance (singleton pattern)
    
    Args:
        model_path: Model path
        lora_path: Path to LoRA weights (optional)
        device_id: GPU device ID (e.g., 0, 1, 2...), None for automatic selection
        
    Returns:
        Model instance
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