from typing import Dict

from app.core.logger import logger

# 全局任务状态追踪表
# 结构: { task_id: { node_name: "pending"|"running"|"done"|"failed" } }
TASK_TRACKER: Dict[str, Dict[str, str]] = {}


def _ensure_task(task_id: str) -> None:
    """确保任务存在，不存在则创建空字典"""
    if task_id not in TASK_TRACKER:
        TASK_TRACKER[task_id] = {}
        logger.debug(f"创建任务追踪记录: {task_id}")


def add_running_task(task_id: str, node_name: str) -> None:
    """标记节点为运行中"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id][node_name] = "running"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> running")


def add_done_task(task_id: str, node_name: str) -> None:
    """标记节点为已完成"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id][node_name] = "done"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> done")


def add_failed_task(task_id: str, node_name: str) -> None:
    """标记节点为失败（异常时用）"""
    _ensure_task(task_id)
    TASK_TRACKER[task_id][node_name] = "failed"
    logger.debug(f"任务[{task_id}] 节点[{node_name}] 状态 -> failed")


def get_task_status(task_id: str) -> Dict[str, str]:
    """获取任务所有节点的状态"""
    return TASK_TRACKER.get(task_id, {})


def get_task_status_str(task_id: str) -> str:
    """获取任务状态汇总字符串，用于前端展示进度"""
    nodes = TASK_TRACKER.get(task_id, {})
    if not nodes:
        return "pending"

    total = len(nodes)
    done = sum(1 for v in nodes.values() if v == "done")
    running = sum(1 for v in nodes.values() if v == "running")
    failed = sum(1 for v in nodes.values() if v == "failed")

    if failed > 0:
        return "failed"
    elif done == total:
        return "done"
    elif running > 0:
        return "running"
    else:
        return "pending"


def remove_task(task_id: str) -> None:
    """清除任务追踪记录"""
    if task_id in TASK_TRACKER:
        del TASK_TRACKER[task_id]
        logger.debug(f"清除任务追踪记录: {task_id}")
