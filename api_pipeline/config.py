"""
API 配置文件
智增增 API (Google Gemini 格式)
文档: https://doc.zhizengzeng.com/doc-6882601
"""
import os

# API 配置
API_KEY = os.environ.get("ZHIZENGZENG_API_KEY", "sk-zk28745521e04a5062a1f90db5e7f3aa2da6697cbbd2a5d8")
# 智增增的 Google Gemini API 基础 URL
API_BASE_URL = os.environ.get("ZHIZENGZENG_API_URL", "https://api.zhizengzeng.com/google")
MODEL_NAME = os.environ.get("ZHIZENGZENG_MODEL", "gemini-2.5-flash-lite")

# 请求超时设置（秒）
REQUEST_TIMEOUT = 120

# 重试设置
MAX_RETRIES = 3
RETRY_DELAY = 1  # 重试间隔（秒）

# 代理配置（从环境变量读取，或手动设置）
# 例如：export HTTPS_PROXY="http://127.0.0.1:7890"
HTTP_PROXY = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
HTTPS_PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

