import time
from functools import lru_cache
from typing import List, Dict, Optional

from bson import ObjectId
from pymongo import MongoClient

from app.core.logger import logger
from app.core.milvus_config import milvus_config  # noqa: F401  确保配置加载
import os


@lru_cache(maxsize=1)
def _get_db():
    """获取 MongoDB 数据库连接（单例）"""
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "kb001")
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=8000)
    db = client[db_name]
    logger.info(f"MongoDB 连接成功：{db_name}")
    return db


def _history_collection():
    return _get_db()["chat_history"]


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    item_names: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    保存一条对话消息到 MongoDB
    :param message_id: 若指定则更新该消息，否则新增
    :return: 消息 ID 字符串
    """
    collection = _history_collection()
    doc = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names or [],
        "image_urls": image_urls or [],
        "ts": int(time.time()),
    }

    if message_id:
        try:
            collection.update_one({"_id": ObjectId(message_id)}, {"$set": doc})
            return message_id
        except Exception as e:
            logger.warning(f"更新历史消息失败，改为新增：{e}")

    result = collection.insert_one(doc)
    return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict]:
    """
    获取会话最近 N 条历史消息（按时间正序返回）
    """
    collection = _history_collection()
    records = list(
        collection.find({"session_id": session_id}).sort("ts", -1).limit(limit)
    )
    # 反转为时间正序
    records.reverse()

    messages = []
    for r in records:
        messages.append({
            "_id": str(r.get("_id")),
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "image_urls": r.get("image_urls", []),
            "ts": r.get("ts"),
        })
    return messages


def update_message_item_names(message_ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史消息的 item_names 字段
    :return: 更新条数
    """
    if not message_ids:
        return 0
    collection = _history_collection()
    object_ids = []
    for mid in message_ids:
        try:
            object_ids.append(ObjectId(mid))
        except Exception:
            continue
    if not object_ids:
        return 0
    result = collection.update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {"item_names": item_names}},
    )
    return result.modified_count


def clear_history(session_id: str) -> int:
    """删除某会话的所有历史记录，返回删除条数"""
    collection = _history_collection()
    result = collection.delete_many({"session_id": session_id})
    return result.deleted_count
