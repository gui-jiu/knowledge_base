import os

from dotenv import load_dotenv

load_dotenv()


class MinerUConfig:
    """MinerU 在线解析 API 配置"""

    def __init__(self):
        self.api_token: str = os.getenv("MINERU_API_TOKEN", "")
        self.base_url: str = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4")


mineru_config = MinerUConfig()
