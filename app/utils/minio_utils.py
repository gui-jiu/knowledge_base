from functools import lru_cache

from minio import Minio

from app.core.minio_config import minio_config


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    """
    获取 MinIO 客户端（单例，复用连接）
    """
    return Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
    )
