import os

from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig:
    """BGE-M3 向量模型配置"""

    def __init__(self):
        self.bge_m3_path: str = os.getenv("BGE_M3_PATH", "")
        self.bge_m3: str = os.getenv("BGE_M3", "BAAI/bge-m3")
        self.bge_device: str = os.getenv("BGE_DEVICE", "cpu")
        self.bge_fp16: bool = os.getenv("BGE_FP16", "False").lower() in ("1", "true")


embedding_config = EmbeddingConfig()
