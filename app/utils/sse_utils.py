import asyncio
import json
import queue
from typing import Any, Dict


class SSEEvent:
    """SSE 事件类型常量"""
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"


# 每个会话一个线程安全队列：{session_id: queue.Queue}
_SSE_QUEUES: Dict[str, "queue.Queue"] = {}


def create_sse_queue(session_id: str) -> None:
    """为会话创建 SSE 队列"""
    _SSE_QUEUES[session_id] = queue.Queue()


def get_sse_queue(session_id: str):
    """获取会话的 SSE 队列（不存在则创建）"""
    if session_id not in _SSE_QUEUES:
        _SSE_QUEUES[session_id] = queue.Queue()
    return _SSE_QUEUES[session_id]


def remove_sse_queue(session_id: str) -> None:
    """移除会话的 SSE 队列"""
    _SSE_QUEUES.pop(session_id, None)


def push_to_session(session_id: str, event: str, data: Any) -> None:
    """
    向会话推送一条 SSE 事件（线程安全，可在后台线程调用）
    """
    q = get_sse_queue(session_id)
    q.put({"event": event, "data": data})


def _format_sse(event: str, data: Any) -> str:
    """格式化为 SSE 协议字符串"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def sse_generator(session_id: str, request=None):
    """
    SSE 异步生成器：持续从队列取事件推送给前端
    - 收到 final 事件后结束
    - 客户端断开时结束
    - 空闲时发送心跳保活
    """
    q = get_sse_queue(session_id)

    # 先发 ready 握手
    yield _format_sse(SSEEvent.READY, {"session_id": session_id, "status": "connected"})

    while True:
        # 客户端断开检测
        if request is not None:
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                pass

        try:
            item = q.get_nowait()
        except queue.Empty:
            # 无数据，发心跳并等待
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)
            continue

        event = item["event"]
        yield _format_sse(event, item["data"])

        if event == SSEEvent.FINAL:
            break

    remove_sse_queue(session_id)
