from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodeEntry(NodeBase):
    """
    节点: 入口节点
    作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        # 模拟简单的路由逻辑
        if "local_file_path" in state:
            path = state["local_file_path"]
            if path.endswith(".pdf"):
                state["is_pdf_read_enabled"] = True
            elif path.endswith(".md"):
                state["is_md_read_enabled"] = True

        return state