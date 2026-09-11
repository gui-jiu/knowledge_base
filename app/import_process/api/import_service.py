import os
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.import_process.agent.kb_import_workflow import KBImportWorkflow
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import (
    update_task_status,
    get_task_status,
    get_task_status_str,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from app.core.paths import PROJECT_ROOT
from app.core.logger import logger


app = FastAPI(title="import service", description="掌柜智库文档导入服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = PROJECT_ROOT / "temp_data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/import.html")
async def import_page():
    """返回导入页面"""
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    page_path = current_dir_parent_path / "page" / "import.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{page_path}！")
    return FileResponse(page_path)


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    接收上传文件并启动后台导入流程
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".md"):
        raise HTTPException(status_code=400, detail="仅支持 .pdf / .md 文件")

    task_id = str(uuid.uuid4())
    # 保存上传文件
    save_path = UPLOAD_DIR / f"{task_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"文件上传成功：{save_path}，task_id: {task_id}")
    update_task_status(task_id, TASK_STATUS_PROCESSING, False)

    # 后台执行导入
    background_tasks.add_task(run_import_graph, task_id, str(save_path))

    return {"message": "文件已接收，正在导入...", "task_id": task_id}


def run_import_graph(task_id: str, file_path: str):
    """执行导入流程图"""
    logger.info(f"开始导入流程... task_id={task_id}, file={file_path}")
    local_dir = str(PROJECT_ROOT / "output")
    init_state = create_default_state(
        task_id=task_id,
        local_file_path=file_path,
        local_dir=local_dir,
    )
    try:
        KBImportWorkflow.create_and_run(init_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED, False)
        logger.info(f"导入流程完成：task_id={task_id}")
    except Exception as e:
        logger.exception(f"导入流程异常: {e}")
        update_task_status(task_id, TASK_STATUS_FAILED, False)


@app.get("/status/{task_id}")
async def status(task_id: str):
    """查询导入任务状态"""
    return {
        "task_id": task_id,
        "status": get_task_status_str(task_id),
        "nodes": get_task_status(task_id),
    }


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
