import os
from dotenv import load_dotenv

load_dotenv()

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))
print("LLM_DEFAULT_MODEL:", os.getenv("LLM_DEFAULT_MODEL"))
print("MILVUS_URL:", os.getenv("MILVUS_URL"))
print("MONGO_URL:", os.getenv("MONGO_URL"))
print("MINIO_ENDPOINT:", os.getenv("MINIO_ENDPOINT"))
