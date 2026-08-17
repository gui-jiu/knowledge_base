import os

from dotenv import load_dotenv

load_dotenv()


class MinIOConfig:
    """MinIO 对象存储配置"""

    def __init__(self):
        self.endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key: str = os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key: str = os.getenv("MINIO_SECRET_KEY", "")
        self.bucket_name: str = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
        self.minio_img_dir: str = os.getenv("MINIO_IMG_DIR", "/upload-images")
        self.minio_secure: bool = os.getenv("MINIO_SECURE", "False").lower() == "true"


minio_config = MinIOConfig()
