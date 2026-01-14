"""
API 模型调用模块
使用智增增 API (Google Gemini 格式) 进行多模态推理
文档: https://doc.zhizengzeng.com/doc-6882601
"""
import base64
import time
from typing import Optional, Union
from pathlib import Path

import requests

from .config import API_KEY, API_BASE_URL, MODEL_NAME, REQUEST_TIMEOUT, RETRY_DELAY, HTTP_PROXY, HTTPS_PROXY


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    将图片编码为 base64 字符串
    
    Args:
        image_path: 图片路径
        
    Returns:
        base64 编码的图片字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: Union[str, Path]) -> str:
    """
    获取图片的 MIME 类型
    
    Args:
        image_path: 图片路径
        
    Returns:
        MIME 类型字符串
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    
    return mime_types.get(suffix, "image/png")


class APIModel:
    """API 模型封装，使用 Google Gemini 格式"""
    
    def __init__(
        self,
        api_key: str = API_KEY,
        api_base_url: str = API_BASE_URL,
        model_name: str = MODEL_NAME,
    ):
        """
        初始化 API 模型
        
        Args:
            api_key: API 密钥
            api_base_url: API 基础 URL
            model_name: 模型名称
        """
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.model_name = model_name
        # Google Gemini API 格式的端点
        self.endpoint = f"{self.api_base_url}/v1beta/models/{model_name}:generateContent"
        
    def generate(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        生成回复（多模态输入）
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: top-p 采样参数
            do_sample: 是否采样（API 调用时通过 temperature 控制）
            
        Returns:
            生成的文本
        """
        # 编码图片
        image_base64 = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        # 构建请求体（Google Gemini 格式）
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_base64
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_new_tokens,
                "temperature": temperature if do_sample else 0,
                "topP": top_p,
            }
        }
        
        # 请求头（Google Gemini 格式使用 X-goog-api-key）
        headers = {
            "X-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # 代理配置
        proxies = {}
        if HTTP_PROXY:
            proxies["http"] = HTTP_PROXY
        if HTTPS_PROXY:
            proxies["https"] = HTTPS_PROXY
        
        # 发送请求
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                proxies=proxies if proxies else None,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取生成的文本（Google Gemini 格式）
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            
            print(f"警告: API 响应格式异常: {result}")
            return ""
                
        except requests.exceptions.Timeout:
            print(f"错误: API 请求超时 (>{REQUEST_TIMEOUT}s)")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"错误: API HTTP 错误: {e}")
            print(f"响应内容: {e.response.text if e.response else 'N/A'}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"错误: API 请求失败: {e}")
            return ""
        except Exception as e:
            print(f"错误: 未知错误: {e}")
            return ""
    
    def generate_with_retry(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_retries: int = 3,
    ) -> str:
        """
        带重试机制的生成
        
        Args:
            image_path: 图片路径
            prompt: 提示词
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: top-p 采样参数
            do_sample: 是否采样
            max_retries: 最大重试次数
            
        Returns:
            生成的文本
        """
        for attempt in range(max_retries):
            result = self.generate(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            
            if result:
                return result
                
            if attempt < max_retries - 1:
                print(f"重试 {attempt + 1}/{max_retries}...")
                time.sleep(RETRY_DELAY)
                
        return ""


# 全局模型实例（单例模式）
_model_instance: Optional[APIModel] = None


def get_model(
    api_key: str = API_KEY,
    api_base_url: str = API_BASE_URL,
    model_name: str = MODEL_NAME,
) -> APIModel:
    """
    获取模型实例（单例模式）
    
    Args:
        api_key: API 密钥
        api_base_url: API 基础 URL
        model_name: 模型名称
        
    Returns:
        模型实例
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = APIModel(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
        )
        
    return _model_instance
