import os

from dotenv import load_dotenv

load_dotenv()


class RerankerConfig:
    """重排序模型配置（阿里云百炼 qwen3-rerank）"""

    def __init__(self):
        # 复用百炼 DashScope 的 Key
        self.api_key: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("VL_API_KEY", ""))
        self.model: str = os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank")
        self.instruct: str = os.getenv("TEXT_RERANK_INSTRUCT", "针对给定的查询，检索能够解答该查询的相关段落")


reranker_config = RerankerConfig()
