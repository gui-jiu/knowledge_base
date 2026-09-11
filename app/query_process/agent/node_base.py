from abc import ABC, abstractmethod

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


class NodeBase(ABC):
    """
    查询流程节点基类
    所有节点类都应继承此基类，实现 process 方法
    """

    # 节点名称，子类应覆盖
    name: str = "base_node"

    def __init__(self):
        """强制子类设置 name"""
        if self.name == "base_node":
            raise ValueError(f"子类 {self.__class__.__name__} 必须覆盖 name 类属性")

    def __call__(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点执行入口
        LangGraph 调用节点时会调用此方法，提供统一的日志输出、任务追踪和异常处理
        """
        # 节点启动日志
        logger.info(f"{'*' * 20}【{self.name}】节点启动{'*' * 20}")
        logger.debug(f"【{self.name}】节点当前工作流状态：{format_state(state)}")

        session_id = state.get("session_id", "")
        is_stream = state.get("is_stream", False)

        # 记录节点运行状态
        if session_id:
            add_running_task(session_id, self.name, is_stream)

        try:
            # 节点核心处理逻辑
            result = self.process(state)

            # 处理节点返回新字典的情况（部分更新）：
            # - 若 process 返回的是新字典（增量），合并进 state 后返回"增量"，
            #   避免并行节点同时返回完整 state 导致 LangGraph 并发更新冲突
            # - 若返回的是同一个 state 对象（原地修改），直接返回完整 state
            if isinstance(result, dict) and result is not state:
                state.update(result)
                output = result
            else:
                output = state

            # 记录节点完成状态
            if session_id:
                add_done_task(session_id, self.name, is_stream)

            logger.debug(f"【{self.name}】节点更新后工作流状态：{format_state(state)}")
            logger.info(f"{'*' * 20}【{self.name}】节点执行完成{'*' * 20}\n")

            return output

        except Exception as e:
            error_msg = f"【{self.name}】流程执行失败：{str(e)}"
            logger.exception(error_msg)
            raise

    @abstractmethod
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """节点核心处理逻辑，子类必须实现"""
        pass
