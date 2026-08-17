import os.path
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState


class NodeEntry(NodeBase):
    """
    节点: 入口节点 (EntryNode)
    接收外部输入并决定流程走向。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        必要参数：
        - 必须包含 task_id(任务ID)
        - local_file_path(原始文件路径)
        - local_dir(输出文件的放置路径)
        更新参数：
        - is_pdf_read_enabled/is_md_read_enabled
        - pdf_path/md_path
        - file_title

        :param state: 工作流状态对象(包含入参的设置)
        :return: 更新后的状态对象(出参发生了改变)
        """

        # 1、核心参数提取与非空校验
        # state.get("key", "默认值")：安全提取状态值，无 key 时返回默认值
        #  .strip() 防止用户传入 " test.pdf " 带空格的路径
        local_file_path = state.get("local_file_path", "").strip()
        local_dir = state.get("local_dir", "").strip()

        if not local_file_path:
            raise ValueError("缺失参数：local_file_path")
        if not local_dir:
            raise ValueError("缺失参数：local_dir")

        # 2、根据文件后缀判断类型，设置对应解析开关
        # endswith 是字符串的方法（函数），用来判断一个字符串是不是以某个内容结尾。
        if local_file_path.endswith(".pdf"):
            logger.info(f"文件类型校验通过：{local_file_path} → PDF格式，开启PDF解析流程")
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = local_file_path
        elif local_file_path.endswith(".md"):
            logger.info(f"文件类型校验通过：{local_file_path} → MD格式，开启MD解析流程")
            state["is_md_read_enabled"] = True
            state["md_path"] = local_file_path
        else:
            logger.warning(f"文件类型校验失败：{local_file_path} → 不支持的格式，仅支持.pdf/.md")

        # 3、提取不包含后缀的文件名，作为全局业务标识
        # os.path.basename()：从完整路径提取文件名
        file_name = os.path.basename(local_file_path)
        # splitext()：拆分文件名和后缀
        state["file_title"] = splitext(file_name)[0]
        logger.info(f"文件业务标识提取完成：file_title = {state['file_title']}")

        return state


if __name__ == "__main__":

    from app.import_process.agent.state import create_default_state

    node_entry = NodeEntry()

    # 测试1: PDF 文件
    node_state = create_default_state(
        task_id="task_001",
        local_file_path="d:/abc.pdf",
        local_dir="d:/output"
    )
    node_state_final = node_entry(node_state)
    print("PDF 测试 → is_pdf_read_enabled:", node_state_final["is_pdf_read_enabled"],
          "| file_title:", node_state_final["file_title"])

    # 测试2: MD 文件
    node_state = create_default_state(
        task_id="task_002",
        local_file_path="d:/abc.md",
        local_dir="d:/output"
    )
    node_state_final = node_entry(node_state)
    print("MD 测试 → is_md_read_enabled:", node_state_final["is_md_read_enabled"],
          "| file_title:", node_state_final["file_title"])

    # 测试3: 不支持的文件
    node_state = create_default_state(
        task_id="task_003",
        local_file_path="d:/abc.txt",
        local_dir="d:/output"
    )
    node_state_final = node_entry(node_state)
    print("TXT 测试 → pdf/md 开关:", node_state_final["is_pdf_read_enabled"],
          node_state_final["is_md_read_enabled"], "| file_title:", node_state_final["file_title"])
