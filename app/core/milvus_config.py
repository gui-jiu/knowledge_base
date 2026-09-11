import os

from dotenv import load_dotenv

load_dotenv()


class MilvusConfig:
    """Milvus 向量数据库配置"""

    def __init__(self):
        self.milvus_url: str = os.getenv("MILVUS_URL", "http://localhost:19530")
        self.chunks_collection: str = os.getenv("CHUNKS_COLLECTION", "kb_chunks")
        self.item_name_collection: str = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names")
        self.metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
        self.min_cosine_score: float = float(os.getenv("MILVUS_MIN_COSINE_SCORE", "0.75"))


milvus_config = MilvusConfig()
