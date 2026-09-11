from pathlib import Path
import uuid

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.query_process.agent.kb_query_workflow import KBQueryWorkflow
from app.query_process.agent.state import create_default_state
from app.utils.task_utils import (
    update_task_status,
    get_task_result,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator, push_to_session
from app.utils.mongo_history_utils import get_recent_messages, clear_history
from app.core.logger import logger


app = FastAPI(title="query service", description="掌柜智库查询服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


@app.get("/chat.html")
async def chat():
    """返回聊天页面"""
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    return FileResponse(chat_html_path)


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    接收查询请求：
    - 流式：创建 SSE 队列，后台执行图，立即返回 session_id
    - 非流式：同步执行图，返回答案
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"开始处理流程... 流式: {is_stream}, 查询: {user_query}, session_id: {session_id}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        return {"message": "结果正在处理中...", "session_id": session_id}
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        return {"message": "处理完成！", "session_id": session_id, "answer": answer, "done_list": []}


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    """执行查询流程图"""
    logger.info(f"开始流程图处理...{session_id} {user_query} {is_stream}")
    default_state = create_default_state(
        original_query=user_query,
        session_id=session_id,
        is_stream=is_stream,
    )
    try:
        KBQueryWorkflow.create_and_run(default_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.exception(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 实时返回结果"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """查询当前会话历史记录"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清空会话历史"""
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


@app.get("/health")
async def health():
    """健康检查"""
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
