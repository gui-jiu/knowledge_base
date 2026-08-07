import os
from pathlib import Path

# 项目根目录（app/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 临时文件根目录
TEMP_DIR = PROJECT_ROOT / "temp_data"
# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 中间件默认配置（可通过 .env 覆盖）
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "kb001")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
