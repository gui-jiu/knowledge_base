from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

class NodeMdImg(NodeBase):
    """
    节点: 图片处理
    处理 Markdown 中的图片资源。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_md_img"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # TODO
        logger.info(f"【{self.name}】节点逻辑")

        return state