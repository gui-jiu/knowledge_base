from functools import lru_cache

from pymilvus import MilvusClient

from app.core.logger import logger
from app.core.milvus_config import milvus_config


@lru_cache(maxsize=1)
def get_milvus_client():
    """
    获取 Milvus 客户端（单例，复用连接）
    使用 MilvusClient（高层 API），支持 has_collection/create_collection/insert/delete 等操作
    """
    try:
        client = MilvusClient(uri=milvus_config.milvus_url)
        # 验证连接：尝试列出集合
        client.list_collections()
        logger.info(f"Milvus 连接成功：{milvus_config.milvus_url}")
        return client
    except Exception as e:
        logger.error(f"Milvus 连接失败：{e}")
        return None


def escape_milvus_string(text: str) -> str:
    """
    转义 Milvus 过滤表达式中的特殊字符
    防止商品名称含引号/反斜杠等导致 filter 表达式解析失败
    """
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", " ")
    )
