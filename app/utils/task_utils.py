from typing import Any, Dict

from app.core.logger import logger

# ============ 任务状态常量 ============
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 全局任务状态追踪表
# 结构: { task_id/session_id: { "nodes": {node_name: status}, "status": overall, "results": {} } }
TASK_TRACKER: Dict[str, Dict[str, Any]] = {}


def _ensure_task(task_id: str) -> None:
    """确保任务存在，不存在则创建"""
    if task_id not in TASK_TRACKER:
        TASK_TRACKER[task_id] = {
            "nodes": {},
            "status": TASK_STATUS_PENDING,
            "results": {},
        }
        logger.debug(f"创建任务追踪记录: {task_id}")


def _push_progress(task_id: str, is_stream: bool) -> None:
    """流式模式下推送进度事件"""
    if not is_stream:
        return
    try:
        from app.utils.sse_utils import push_to_session, SSEEvent
        nodes = TASK_TRACKER.get(task_id, {}).get("nodes", {})
        done_list = [n for n, s in nodes.items() if s == "done"]
        running_list = [n for n, s in nodes.items() if s == "running"]
        push_to_session(task_id, SSEEvent.PROGRESS, {
            "done_list": done_list,
            "running_list": running_list,
        })
    except Exception as e:
        logger.debug(f"推送进度失败: {e}")


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """标记节点为运行中"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id]["nodes"][node_name] = "running"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> running")
    _push_progress(task_id, is_stream)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """标记节点为已完成"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id]["nodes"][node_name] = "done"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> done")
    _push_progress(task_id, is_stream)


def add_failed_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """标记节点为失败"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id]["nodes"][node_name] = "failed"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> failed")
    _push_progress(task_id, is_stream)


def update_task_status(task_id: str, status: str, is_stream: bool = False) -> None:
    """更新任务整体状态"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id]["status"] = status
    logger.debug(f"任务[{task_id}] 整体状态 -> {status}")


def set_task_result(task_id: str, key: str, value: Any) -> None:
    """存储任务结果"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id]["results"][key] = value


def get_task_result(task_id: str, key: str, default: Any = None) -> Any:
    """获取任务结果"""
    return TASK_TRACKER.get(task_id, {}).get("results", {}).get(key, default)


def get_task_status(task_id: str) -> Dict[str, str]:
    """获取任务所有节点状态"""
    return TASK_TRACKER.get(task_id, {}).get("nodes", {})


def get_task_status_str(task_id: str) -> str:
    """获取任务状态汇总字符串"""
    task = TASK_TRACKER.get(task_id, {})
    nodes = task.get("nodes", {})
    if not nodes:
        return task.get("status", "pending")

    total = len(nodes)
    done = sum(1 for v in nodes.values() if v == "done")
    failed = sum(1 for v in nodes.values() if v == "failed")

    if failed > 0:
        return "failed"
    elif done == total:
        return "done"
    else:
        return "running"


def remove_task(task_id: str) -> None:
    """清除任务追踪记录"""
    if task_id in TASK_TRACKER:
        del TASK_TRACKER[task_id]
        logger.debug(f"清除任务追踪记录: {task_id}")
