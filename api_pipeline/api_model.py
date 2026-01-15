import base64
import time
from typing import Optional, Union
from pathlib import Path

import requests

from .config import API_KEY, API_BASE_URL, MODEL_NAME, REQUEST_TIMEOUT, RETRY_DELAY, HTTP_PROXY, HTTPS_PROXY


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode image to base64 string
    
    Args:
        image_path: Image path
        
    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: Union[str, Path]) -> str:
    """
    Get MIME type of the image
    
    Args:
        image_path: Image path
        
    Returns:
        MIME type string
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
    """API model wrapper, using Google Gemini format"""
    
    def __init__(
        self,
        api_key: str = API_KEY,
        api_base_url: str = API_BASE_URL,
        model_name: str = MODEL_NAME,
    ):
        """
        Initialize API model
        
        Args:
            api_key: API key
            api_base_url: API base URL
            model_name: Model name
        """
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.model_name = model_name
        # Google Gemini API format endpoint
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
        Generate response (multimodal input)
        
        Args:
            image_path: Image path
            prompt: Prompt text
            max_new_tokens: Maximum generation length
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling (controlled by temperature in API call)
            
        Returns:
            Generated text
        """
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        # Build request body (Google Gemini format)
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
        
        # Request headers (Google Gemini format uses X-goog-api-key)
        headers = {
            "X-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Proxy configuration
        proxies = {}
        if HTTP_PROXY:
            proxies["http"] = HTTP_PROXY
        if HTTPS_PROXY:
            proxies["https"] = HTTPS_PROXY
        
        # Send request
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
            
            # Extract generated text (Google Gemini format)
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            
            print(f"Warning: Abnormal API response format: {result}")
            return ""
                
        except requests.exceptions.Timeout:
            print(f"Error: API request timed out (>{REQUEST_TIMEOUT}s)")
            return ""
        except requests.exceptions.HTTPError as e:
            print(f"Error: API HTTP error: {e}")
            print(f"Response content: {e.response.text if e.response else 'N/A'}")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error: API request failed: {e}")
            return ""
        except Exception as e:
            print(f"Error: Unknown error: {e}")
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
        Generate with retry mechanism
        
        Args:
            image_path: Image path
            prompt: Prompt text
            max_new_tokens: Maximum generation length
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            max_retries: Maximum number of retries
            
        Returns:
            Generated text
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
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(RETRY_DELAY)
                
        return ""


# Global model instance (singleton pattern)
_model_instance: Optional[APIModel] = None


def get_model(
    api_key: str = API_KEY,
    api_base_url: str = API_BASE_URL,
    model_name: str = MODEL_NAME,
) -> APIModel:
    """
    Get model instance (singleton pattern)
    
    Args:
        api_key: API key
        api_base_url: API base URL
        model_name: Model name
        
    Returns:
        Model instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = APIModel(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
        )
        
    return _model_instance