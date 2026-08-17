import os

from dotenv import load_dotenv

load_dotenv()


class LMConfig:
    """大语言模型 / 视觉语言模型配置"""

    def __init__(self):
        # 默认 LLM（DeepSeek）
        self.api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.api_base: str = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
        self.default_model: str = os.getenv("LLM_DEFAULT_MODEL", "deepseek-chat")
        self.temperature: float = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))

        # 视觉语言模型（百炼 DashScope）
        self.vl_api_key: str = os.getenv("VL_API_KEY", "")
        self.vl_api_base: str = os.getenv("VL_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vl_model: str = os.getenv("VL_MODEL", "qwen-vl-plus")


lm_config = LMConfig()
